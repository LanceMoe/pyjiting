# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

import ast as py_ast
import ctypes

from llvmlite import ir

from . import ast as core
from .errors import CodegenError
from .ll_types import mangler
from .registry import get as get_registered, keep_callback
from .types import (bool_t, double64_t, float32_t, int32_t, int64_t, is_array,
                    is_float, is_integer, shape_t, void_t)


ir_i1 = ir.IntType(1)
ir_i32 = ir.IntType(32)
ir_i64 = ir.IntType(64)
ir_f32 = ir.FloatType()
ir_f64 = ir.DoubleType()
ir_void = ir.VoidType()

ERROR_NONE = 0
ERROR_DIVISION_BY_ZERO = 1
ERROR_RANGE_STEP_ZERO = 2
ERROR_ARRAY_DIMENSION_MISMATCH = 3


def array_type(element):
    name = f'pyjiting.ndarray.{element}'
    struct = ir.global_context.get_identified_type(name)
    if not struct.elements:
        struct.set_body(ir.PointerType(element), ir_i64, ir.PointerType(ir_i64), ir.PointerType(ir_i64))
    return ir.PointerType(struct)


TYPE_MAP = {
    int32_t: ir_i32, int64_t: ir_i64, bool_t: ir_i64, float32_t: ir_f32,
    double64_t: ir_f64, void_t: ir_void,
}


def to_lltype(ty):
    if is_array(ty): return array_type(to_lltype(ty.b))
    try: return TYPE_MAP[ty]
    except KeyError as error: raise CodegenError(f'no LLVM type for {ty}') from error


def determined(ty): return ty is not None


class LLVMCodeGen:
    def __init__(self, module, return_type, args):
        self.module, self.return_type, self.args = module, return_type, args
        self.function = self.builder = None
        self.locals, self.arrays, self.shapes = {}, {}, {}
        self.break_blocks, self.continue_blocks = [], []
        self.exit_block = self.return_slot = None
        self.error_ptr = None
        self.org_func_name = None
        self.counter = 0

    def new_block(self, prefix):
        self.counter += 1
        return self.function.append_basic_block(f'{prefix}_{self.counter}')

    def set_block(self, block): self.builder.position_at_end(block)
    def terminated(self): return self.builder.block.terminator is not None

    def alloca(self, ty, name):
        raise CodegenError(f'local {name} was not allocated during function setup')

    def start_function(self, name):
        arg_types = [to_lltype(ty) for ty in self.args] + [ir.PointerType(ir_i32)]
        self.function = ir.Function(self.module, ir.FunctionType(to_lltype(self.return_type), arg_types), name)
        entry = self.function.append_basic_block('entry')
        self.exit_block = self.function.append_basic_block('exit')
        self.builder = ir.IRBuilder(entry)

    def finish_function(self):
        if not self.terminated(): self.builder.branch(self.exit_block)
        self.set_block(self.exit_block)
        if self.return_type == void_t: self.builder.ret_void()
        else: self.builder.ret(self.builder.load(self.return_slot))

    def visit(self, node):
        if isinstance(node, list):
            result = None
            for item in node:
                if self.terminated(): break
                result = self.visit(item)
            return result
        method = getattr(self, f'visit_{type(node).__name__}', None)
        if method is None: raise CodegenError(f'no code generator for {type(node).__name__}', node)
        return method(node)

    def const(self, value, ty=None):
        ty = ty or (ir_i64 if isinstance(value, int) else ir_f64)
        return ir.Constant(ty, value)

    def cast(self, value, source, target):
        if source == target: return value
        target_ll = to_lltype(target)
        if source == bool_t:
            source = int64_t
            if source == target: return value
        if target == bool_t: return self.truthy(value, source, normalize=True)
        if is_integer(source) and is_integer(target):
            return self.builder.sext(value, target_ll) if value.type.width < target_ll.width else self.builder.trunc(value, target_ll)
        if is_integer(source) and is_float(target): return self.builder.sitofp(value, target_ll)
        if is_float(source) and is_float(target):
            return self.builder.fpext(value, target_ll) if source == float32_t else self.builder.fptrunc(value, target_ll)
        raise CodegenError(f'cannot cast {source} to {target}')

    def truthy(self, value, ty, normalize=False):
        if ty == bool_t or is_integer(ty): result = self.builder.icmp_signed('!=', value, ir.Constant(value.type, 0))
        elif is_float(ty): result = self.builder.fcmp_ordered('!=', value, ir.Constant(value.type, 0.0))
        else: raise CodegenError(f'cannot use {ty} as a condition')
        return self.builder.zext(result, ir_i64) if normalize else result

    def guard_nonzero(self, value, ty, error_code):
        if is_float(ty):
            is_nonzero = self.builder.fcmp_ordered('!=', value, ir.Constant(value.type, 0.0))
        else:
            is_nonzero = self.builder.icmp_signed('!=', value, ir.Constant(value.type, 0))
        continue_block = self.new_block('nonzero')
        error_block = self.new_block('runtime_error')
        self.builder.cbranch(is_nonzero, continue_block, error_block)
        self.set_block(error_block)
        self.builder.store(ir.Constant(ir_i32, error_code), self.error_ptr)
        self.builder.branch(self.exit_block)
        self.set_block(continue_block)

    def guard(self, condition, error_code):
        continue_block = self.new_block('guard_pass')
        error_block = self.new_block('runtime_error')
        self.builder.cbranch(condition, continue_block, error_block)
        self.set_block(error_block)
        self.builder.store(ir.Constant(ir_i32, error_code), self.error_ptr)
        self.builder.branch(self.exit_block)
        self.set_block(continue_block)

    def propagate_error(self):
        clean_block = self.new_block('error_clear')
        error_block = self.new_block('error_propagate')
        is_clean = self.builder.icmp_signed('==', self.builder.load(self.error_ptr), ir.Constant(ir_i32, ERROR_NONE))
        self.builder.cbranch(is_clean, clean_block, error_block)
        self.set_block(error_block)
        self.builder.branch(self.exit_block)
        self.set_block(clean_block)

    def visit_Fun(self, node):
        self.org_func_name = node.fname
        self.start_function(mangler(node.fname, self.args))
        local_types = {}
        for item in self.walk_nodes(node):
            if isinstance(item, core.Assign): local_types[item.ref] = item.type
            elif isinstance(item, core.Loop): local_types[item.var.id] = int64_t
        if self.return_type != void_t:
            self.return_slot = self.builder.alloca(to_lltype(self.return_type), name='retval')
        for name, ty in local_types.items():
            self.locals[name] = self.builder.alloca(to_lltype(ty), name=name)
        self.error_ptr = self.function.args[-1]
        self.error_ptr.name = 'error'
        for core_arg, ll_arg, ty in zip(node.args, self.function.args, self.args):
            ll_arg.name = core_arg.id
            if is_array(ty):
                self.locals[core_arg.id] = ll_arg
                zero = ir.Constant(ir_i32, 0)
                data = self.builder.gep(ll_arg, [zero, zero]); ndim = self.builder.gep(ll_arg, [zero, ir.Constant(ir_i32, 1)])
                shape = self.builder.gep(ll_arg, [zero, ir.Constant(ir_i32, 2)]); strides = self.builder.gep(ll_arg, [zero, ir.Constant(ir_i32, 3)])
                self.arrays[core_arg.id] = {'data': self.builder.load(data), 'ndim': self.builder.load(ndim), 'shape': self.builder.load(shape), 'strides': self.builder.load(strides), 'element': ty.b}
            else:
                ptr = self.builder.alloca(to_lltype(ty), name=core_arg.id); self.builder.store(ll_arg, ptr); self.locals[core_arg.id] = ptr
        if self.return_type != void_t:
            self.builder.store(ir.Constant(to_lltype(self.return_type), None), self.return_slot)
        self.visit(node.body)
        self.finish_function()
        return self.function

    def walk_nodes(self, node):
        if isinstance(node, list):
            for item in node:
                yield from self.walk_nodes(item)
            return
        if not isinstance(node, py_ast.AST):
            return
        yield node
        for _, value in py_ast.iter_fields(node):
            if isinstance(value, (py_ast.AST, list)):
                yield from self.walk_nodes(value)

    def visit_LitInt(self, node): return ir.Constant(to_lltype(node.type), node.n)
    def visit_LitFloat(self, node): return ir.Constant(to_lltype(node.type), node.n)
    def visit_LitBool(self, node): return ir.Constant(ir_i64, int(node.n))
    def visit_Var(self, node):
        if node.id in self.arrays: return self.locals[node.id]
        return self.builder.load(self.locals[node.id])

    def _index_address(self, value, indices):
        if not isinstance(value, core.Var) or value.id not in self.arrays: raise CodegenError('only parameter arrays can be indexed', value)
        metadata = self.arrays[value.id]
        dimension_matches = self.builder.icmp_signed('==', metadata['ndim'], ir.Constant(ir_i64, len(indices)))
        self.guard(dimension_matches, ERROR_ARRAY_DIMENSION_MISMATCH)
        offset = ir.Constant(ir_i64, 0)
        for dim, index in enumerate(indices):
            stride_ptr = self.builder.gep(metadata['strides'], [ir.Constant(ir_i64, dim)])
            stride = self.builder.load(stride_ptr)
            offset = self.builder.add(offset, self.builder.mul(self.cast(self.visit(index), index.type, int64_t), stride))
        return self.builder.gep(metadata['data'], [offset]), metadata['element']

    def visit_Index(self, node):
        if node.value.type == shape_t:
            if not isinstance(node.value, core.Prim) or not isinstance(node.value.args[0], core.Var): raise CodegenError('shape value is not indexable', node)
            shape = self.arrays[node.value.args[0].id]['shape']
            index = self.cast(self.visit(node.indices[0]), node.indices[0].type, int64_t)
            return self.builder.load(self.builder.gep(shape, [index]))
        address, _ = self._index_address(node.value, node.indices)
        return self.builder.load(address)

    def visit_Assign(self, node):
        value = self.visit(node.value); value = self.cast(value, node.value.type, node.type)
        ptr = self.locals.get(node.ref)
        if ptr is None: raise CodegenError(f'unknown local {node.ref}', node)
        self.builder.store(value, ptr)

    def visit_StoreIndex(self, node):
        address, element = self._index_address(node.value, node.indices)
        value = self.cast(self.visit(node.rhs), node.rhs.type, element)
        self.builder.store(value, address)

    def visit_AugStoreIndex(self, node):
        address, element = self._index_address(node.value, node.indices)
        left = self.cast(self.builder.load(address), element, node.operand_type)
        right = self.cast(self.visit(node.rhs), node.rhs.type, node.operand_type)
        if node.fn == 'add#': result = self.builder.fadd(left, right) if is_float(node.type) else self.builder.add(left, right)
        elif node.fn == 'sub#': result = self.builder.fsub(left, right) if is_float(node.type) else self.builder.sub(left, right)
        elif node.fn == 'mult#': result = self.builder.fmul(left, right) if is_float(node.type) else self.builder.mul(left, right)
        elif node.fn == 'div#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            result = self.builder.fdiv(self.cast(left, node.operand_type, node.type), self.cast(right, node.operand_type, node.type))
        elif node.fn == 'floordiv#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            result = self.builder.call(self._llvm_floor(node.type), [self.builder.fdiv(left, right)]) if is_float(node.type) else self._integer_floor_div(left, right)
        elif node.fn == 'mod#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            result = self._float_mod(left, right) if is_float(node.type) else self._integer_mod(left, right)
        elif node.fn == 'pow#':
            result = self._pow_values(node, left, right)
        else:
            raise CodegenError(f'unsupported augmented assignment primitive {node.fn}', node)
        self.builder.store(self.cast(result, node.type, element), address)

    def _compare(self, op, left, right, ty):
        if is_float(ty):
            predicates = {'eq#': '==', 'ne#': '!=', 'lt#': '<', 'le#': '<=', 'gt#': '>', 'ge#': '>='}
            if op == 'ne#': return self.builder.fcmp_unordered(predicates[op], left, right)
            return self.builder.fcmp_ordered(predicates[op], left, right)
        predicates = {'eq#': '==', 'ne#': '!=', 'lt#': '<', 'le#': '<=', 'gt#': '>', 'ge#': '>='}
        return self.builder.icmp_signed(predicates[op], left, right)

    def visit_Compare(self, node):
        left, left_ty = self.visit(node.left), node.left.type
        if len(node.ops) == 1:
            comparator = node.comparators[0]
            right = self.visit(comparator); common = self._common(left_ty, comparator.type)
            return self.builder.zext(self._compare(node.ops[0], self.cast(left, left_ty, common), self.cast(right, comparator.type, common), common), ir_i64)

        end = self.new_block('compare_end')
        false_blocks = []
        for position, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            right = self.visit(comparator)
            common = self._common(left_ty, comparator.type)
            passed = self._compare(op, self.cast(left, left_ty, common), self.cast(right, comparator.type, common), common)
            current = self.builder.block
            if position == len(node.ops) - 1:
                failed = self.new_block('compare_false')
                self.builder.cbranch(passed, end, failed)
                true_block = current
                self.set_block(failed); self.builder.branch(end); false_blocks.append(failed)
                break
            next_block = self.new_block('compare_next')
            failed = self.new_block('compare_false')
            self.builder.cbranch(passed, next_block, failed)
            self.set_block(failed); self.builder.branch(end); false_blocks.append(failed)
            self.set_block(next_block)
            left, left_ty = right, comparator.type
        self.set_block(end)
        result = self.builder.phi(ir_i64)
        result.add_incoming(ir.Constant(ir_i64, 1), true_block)
        for block in false_blocks: result.add_incoming(ir.Constant(ir_i64, 0), block)
        return result

    def _common(self, left, right):
        from .types import promote_numeric
        result = promote_numeric(left, right)
        if result is None: raise CodegenError(f'no common type for {left}, {right}')
        return result

    def visit_Prim(self, node):
        if node.fn == 'shape#': return None
        if node.fn in ('and#', 'or#'): return self._short_circuit(node)
        if node.fn == 'not#':
            truth = self.truthy(self.visit(node.args[0]), node.args[0].type)
            return self.builder.zext(self.builder.not_(truth), ir_i64)
        if node.fn == 'neg#':
            value = self.visit(node.args[0]); return self.builder.fneg(value) if is_float(node.type) else self.builder.neg(value)
        left = self.cast(self.visit(node.args[0]), node.args[0].type, node.operand_type)
        right = self.cast(self.visit(node.args[1]), node.args[1].type, node.operand_type)
        if node.fn == 'add#': return self.builder.fadd(left, right) if is_float(node.type) else self.builder.add(left, right)
        if node.fn == 'sub#': return self.builder.fsub(left, right) if is_float(node.type) else self.builder.sub(left, right)
        if node.fn == 'mult#': return self.builder.fmul(left, right) if is_float(node.type) else self.builder.mul(left, right)
        if node.fn == 'div#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            return self.builder.fdiv(self.cast(left, node.operand_type, node.type), self.cast(right, node.operand_type, node.type))
        if node.fn == 'floordiv#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            if is_float(node.type):
                division = self.builder.fdiv(left, right); floor = self._llvm_floor(node.type); return self.builder.call(floor, [division])
            return self._integer_floor_div(left, right)
        if node.fn == 'mod#':
            self.guard_nonzero(right, node.operand_type, ERROR_DIVISION_BY_ZERO)
            return self._float_mod(left, right) if is_float(node.type) else self._integer_mod(left, right)
        if node.fn == 'pow#': return self._pow(node, left, right)
        raise CodegenError(f'unknown primitive {node.fn}', node)

    def _llvm_floor(self, ty):
        name = 'llvm.floor.f32' if ty == float32_t else 'llvm.floor.f64'
        return self.module.globals.get(name) or ir.Function(self.module, ir.FunctionType(to_lltype(ty), [to_lltype(ty)]), name)

    def _integer_floor_div(self, left, right):
        quotient = self.builder.sdiv(left, right)
        remainder = self.builder.srem(left, right)
        has_remainder = self.builder.icmp_signed('!=', remainder, ir.Constant(left.type, 0))
        different_sign = self.builder.icmp_signed('<', self.builder.xor(left, right), ir.Constant(left.type, 0))
        correction = self.builder.and_(has_remainder, different_sign)
        return self.builder.sub(quotient, self.builder.zext(correction, left.type))

    def _integer_mod(self, left, right):
        remainder = self.builder.srem(left, right)
        has_remainder = self.builder.icmp_signed('!=', remainder, ir.Constant(left.type, 0))
        different_sign = self.builder.icmp_signed('<', self.builder.xor(left, right), ir.Constant(left.type, 0))
        correction = self.builder.and_(has_remainder, different_sign)
        return self.builder.add(remainder, self.builder.select(correction, right, ir.Constant(right.type, 0)))

    def _float_mod(self, left, right):
        remainder = self.builder.frem(left, right)
        has_remainder = self.builder.fcmp_ordered('!=', remainder, ir.Constant(left.type, 0.0))
        different_sign = self.builder.fcmp_ordered('<', self.builder.fmul(remainder, right), ir.Constant(left.type, 0.0))
        correction = self.builder.and_(has_remainder, different_sign)
        return self.builder.fadd(remainder, self.builder.select(correction, right, ir.Constant(right.type, 0.0)))

    def _pow(self, node, left, right):
        return self._pow_values(node, left, right)

    def _pow_values(self, node, left, right):
        if is_float(node.type):
            ty = to_lltype(node.type); name = 'llvm.pow.f32' if node.type == float32_t else 'llvm.pow.f64'
            fn = self.module.globals.get(name) or ir.Function(self.module, ir.FunctionType(ty, [ty, ty]), name)
            return self.builder.call(fn, [self.cast(left, node.operand_type, node.type), self.cast(right, node.operand_type, node.type)])
        exponent = node.rhs.n if isinstance(node, core.AugStoreIndex) else node.args[1].n
        result = ir.Constant(to_lltype(node.type), 1); base = left
        while exponent:
            if exponent & 1: result = self.builder.mul(result, base)
            exponent >>= 1
            if exponent: base = self.builder.mul(base, base)
        return result

    def _short_circuit(self, node):
        left = self.cast(self.visit(node.args[0]), node.args[0].type, node.type)
        lhs_block = self.builder.block; rhs_block = self.new_block('bool_rhs'); end_block = self.new_block('bool_end')
        condition = self.truthy(left, node.type)
        if node.fn == 'and#': self.builder.cbranch(condition, rhs_block, end_block)
        else: self.builder.cbranch(condition, end_block, rhs_block)
        self.set_block(rhs_block); right = self.cast(self.visit(node.args[1]), node.args[1].type, node.type); self.builder.branch(end_block); rhs_block = self.builder.block
        self.set_block(end_block); phi = self.builder.phi(to_lltype(node.type)); phi.add_incoming(left, lhs_block); phi.add_incoming(right, rhs_block); return phi

    def visit_If(self, node):
        test_block, then_block, else_block, end_block = self.new_block('if_test'), self.new_block('if_then'), self.new_block('if_else'), self.new_block('if_end')
        self.builder.branch(test_block); self.set_block(test_block); self.builder.cbranch(self.truthy(self.visit(node.test), node.test.type), then_block, else_block)
        self.set_block(then_block); self.visit(node.body);
        if not self.terminated(): self.builder.branch(end_block)
        self.set_block(else_block); self.visit(node.orelse)
        if not self.terminated(): self.builder.branch(end_block)
        self.set_block(end_block)

    def visit_While(self, node):
        test, body, after = self.new_block('while_test'), self.new_block('while_body'), self.new_block('while_after')
        otherwise = self.new_block('while_else') if node.orelse else after
        self.builder.branch(test); self.break_blocks.append(after); self.continue_blocks.append(test)
        self.set_block(test); self.builder.cbranch(self.truthy(self.visit(node.test), node.test.type), body, otherwise if node.orelse else after)
        self.set_block(body); self.visit(node.body)
        if not self.terminated(): self.builder.branch(test)
        self.continue_blocks.pop(); self.break_blocks.pop()
        if node.orelse:
            self.set_block(otherwise); self.visit(node.orelse)
            if not self.terminated(): self.builder.branch(after)
        self.set_block(after)

    def visit_Loop(self, node):
        init, test, body, latch, after = (self.new_block('for_init'), self.new_block('for_test'), self.new_block('for_body'), self.new_block('for_latch'), self.new_block('for_after'))
        otherwise = self.new_block('for_else') if node.orelse else after
        self.builder.branch(init); self.set_block(init)
        ptr = self.locals[node.var.id]; self.builder.store(self.cast(self.visit(node.begin), node.begin.type, int64_t), ptr); self.builder.branch(test)
        self.set_block(test); step = self.cast(self.visit(node.step), node.step.type, int64_t); self.guard_nonzero(step, int64_t, ERROR_RANGE_STEP_ZERO); current = self.builder.load(ptr); stop = self.cast(self.visit(node.end), node.end.type, int64_t)
        positive = self.builder.icmp_signed('>', step, ir.Constant(ir_i64, 0)); negative = self.builder.icmp_signed('<', step, ir.Constant(ir_i64, 0)); less = self.builder.icmp_signed('<', current, stop); greater = self.builder.icmp_signed('>', current, stop)
        condition = self.builder.select(positive, less, self.builder.select(negative, greater, ir.Constant(ir_i1, 0))); self.builder.cbranch(condition, body, otherwise if node.orelse else after)
        self.break_blocks.append(after); self.continue_blocks.append(latch)
        self.set_block(body); self.visit(node.body)
        if not self.terminated(): self.builder.branch(latch)
        self.set_block(latch); self.builder.store(self.builder.add(self.builder.load(ptr), step), ptr); self.builder.branch(test)
        self.continue_blocks.pop(); self.break_blocks.pop()
        if node.orelse:
            self.set_block(otherwise); self.visit(node.orelse)
            if not self.terminated(): self.builder.branch(after)
        self.set_block(after)

    def visit_Break(self, node): self.builder.branch(self.break_blocks[-1])
    def visit_Continue(self, node): self.builder.branch(self.continue_blocks[-1])

    def visit_Return(self, node):
        if self.return_type != void_t: self.builder.store(self.cast(self.visit(node.value), node.value.type, self.return_type), self.return_slot)
        self.builder.branch(self.exit_block)

    def visit_CallFunc(self, node):
        args = [self.visit(arg) for arg in node.args]
        if node.fn.id == self.org_func_name:
            result = self.builder.call(self.function, args + [self.error_ptr])
            self.propagate_error()
            return result
        registered = get_registered(node.fn.id)
        if registered is None: raise CodegenError(f'function {node.fn.id!r} is not registered', node)
        fn, signature = registered; ll_args = [to_lltype(ty) for ty in signature.args]; ll_return = to_lltype(signature.return_type)
        c_args = [ctypes.c_int64 if ty in (int64_t, bool_t) else ctypes.c_int32 if ty == int32_t else ctypes.c_float if ty == float32_t else ctypes.c_double for ty in signature.args]
        c_return = ctypes.c_int64 if signature.return_type in (int64_t, bool_t) else ctypes.c_int32 if signature.return_type == int32_t else ctypes.c_float if signature.return_type == float32_t else ctypes.c_double
        callback = ctypes.CFUNCTYPE(c_return, *c_args)(fn); address = keep_callback(node.fn.id, callback)
        pointer_ty = ir.PointerType(ir.FunctionType(ll_return, ll_args)); pointer = self.builder.inttoptr(ir.Constant(ir_i64, address), pointer_ty)
        return self.builder.call(pointer, args)

    def visit_Expr(self, node): return self.visit(node.value)
    def visit_Noop(self, node): return None
