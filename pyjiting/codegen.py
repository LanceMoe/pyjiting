# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

import ast as py_ast
import ctypes
from typing import Any

from llvmlite import ir

from . import ast as core
from .errors import CodegenError
from .intrinsics import (MATH_INTRINSICS, STRING_INTRINSICS, STRING_PREDICATES,
                         STRING_TRANSFORMS)
from .ll_types import mangler
from .registry import get as get_registered, keep_callback
from .types import (TupleType, bool_t, double64_t, float32_t, int32_t, int64_t,
                    is_array, is_float, is_integer, is_string, is_tuple, shape_t,
                    str_t, void_t)
from .string_runtime import (StringPointer, allocation_address, callback_address,
                             literal_address, make_string, set_pending_exception,
                             to_python)


ir_i1 = ir.IntType(1)
ir_i8 = ir.IntType(8)
ir_i32 = ir.IntType(32)
ir_i64 = ir.IntType(64)
ir_f32 = ir.FloatType()
ir_f64 = ir.DoubleType()
ir_void = ir.VoidType()

ERROR_NONE = 0
ERROR_DIVISION_BY_ZERO = 1
ERROR_RANGE_STEP_ZERO = 2
ERROR_ARRAY_DIMENSION_MISMATCH = 3
ERROR_INDEX_OUT_OF_BOUNDS = 4
ERROR_MATH_DOMAIN = 8
ERROR_MATH_RANGE = 9
ERROR_ARRAY_READONLY = 10

ARRAY_WRITEABLE = 1 << 0


def array_type(element):
    name = f'pyjiting.ndarray.{element}'
    struct = ir.global_context.get_identified_type(name)
    if not struct.elements:
        struct.set_body(ir.PointerType(ir_i8), ir_i64, ir.PointerType(ir_i64),
                        ir.PointerType(ir_i64), ir_i64, ir_i64)
    return ir.PointerType(struct)


def string_type():
    struct = ir.global_context.get_identified_type('pyjiting.string')
    if not struct.elements:
        struct.set_body(ir.PointerType(ir_i32), ir_i64)
    return ir.PointerType(struct)


TYPE_MAP = {
    int32_t: ir_i32, int64_t: ir_i64, bool_t: ir_i64, float32_t: ir_f32,
    double64_t: ir_f64, void_t: ir_void,
    str_t: string_type(),
}


def to_lltype(ty):
    if is_array(ty): return array_type(to_lltype(ty.b))
    if is_tuple(ty): return ir.PointerType(ir.LiteralStructType([to_lltype(element) for element in ty.elements]))
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
        elif is_float(ty): result = self.builder.fcmp_unordered('!=', value, ir.Constant(value.type, 0.0))
        elif is_string(ty):
            length = self.builder.load(self.builder.gep(value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
            result = self.builder.icmp_signed('!=', length, ir.Constant(ir_i64, 0))
        elif is_tuple(ty): result = ir.Constant(ir_i1, int(bool(ty.elements)))
        else: raise CodegenError(f'cannot use {ty} as a condition')
        return self.builder.zext(result, ir_i64) if normalize else result

    def guard_nonzero(self, value, ty, error_code):
        if is_float(ty):
            is_nonzero = self.builder.fcmp_unordered('!=', value, ir.Constant(value.type, 0.0))
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
        self.start_function(mangler(node.symbol, self.args))
        local_types = {}
        for item in self.walk_nodes(node):
            if isinstance(item, core.Assign): local_types[item.ref] = item.type
            elif isinstance(item, core.UnpackAssign):
                for name, ty in zip(item.refs, item.ref_types): local_types[name] = ty
            elif isinstance(item, core.Loop): local_types[item.var.id] = int64_t
            elif isinstance(item, core.ForEach): local_types[item.var.id] = item.var.type
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
                itemsize = self.builder.gep(ll_arg, [zero, ir.Constant(ir_i32, 4)]); flags = self.builder.gep(ll_arg, [zero, ir.Constant(ir_i32, 5)])
                self.arrays[core_arg.id] = {
                    'data': self.builder.load(data), 'ndim': self.builder.load(ndim),
                    'shape': self.builder.load(shape), 'strides': self.builder.load(strides),
                    'itemsize': self.builder.load(itemsize), 'flags': self.builder.load(flags),
                    'element': ty.b,
                }
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
    def visit_LitStr(self, node):
        return self.builder.inttoptr(ir.Constant(ir_i64, literal_address(node.value)), string_type())
    def visit_LitTuple(self, node):
        pointer_type = to_lltype(node.type)
        pointer = self._allocate_structure(pointer_type)
        for index, element in enumerate(node.elements):
            address = self.builder.gep(pointer, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, index)])
            self.builder.store(self.visit(element), address)
        return pointer

    def _allocate_structure(self, pointer_type):
        null = ir.Constant(pointer_type, None)
        size = self.builder.ptrtoint(
            self.builder.gep(null, [ir.Constant(ir_i64, 1)]), ir_i64)
        allocator_type = ir.PointerType(ir.FunctionType(
            ir.PointerType(ir_i8), [ir_i64, ir.PointerType(ir_i32)]))
        allocator = self.builder.inttoptr(
            ir.Constant(ir_i64, allocation_address()), allocator_type)
        allocated = self.builder.call(allocator, [size, self.error_ptr])
        self.propagate_error()
        return self.builder.bitcast(allocated, pointer_type)

    def _string_character(self, value, raw_index):
        length = self.builder.load(self.builder.gep(
            value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
        negative = self.builder.icmp_signed('<', raw_index, ir.Constant(ir_i64, 0))
        index = self.builder.select(negative, self.builder.add(raw_index, length), raw_index)
        lower_ok = self.builder.icmp_signed('>=', index, ir.Constant(ir_i64, 0))
        upper_ok = self.builder.icmp_signed('<', index, length)
        self.guard(self.builder.and_(lower_ok, upper_ok), ERROR_INDEX_OUT_OF_BOUNDS)
        descriptor = self._allocate_structure(string_type())
        data = self.builder.load(self.builder.gep(
            value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 0)]))
        character = self.builder.gep(data, [index])
        self.builder.store(character, self.builder.gep(
            descriptor, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 0)]))
        self.builder.store(ir.Constant(ir_i64, 1), self.builder.gep(
            descriptor, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
        return descriptor
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
            shape_ptr = self.builder.gep(metadata['shape'], [ir.Constant(ir_i64, dim)])
            size = self.builder.load(shape_ptr)
            raw_index = self.cast(self.visit(index), index.type, int64_t)
            negative = self.builder.icmp_signed('<', raw_index, ir.Constant(ir_i64, 0))
            normalized = self.builder.select(negative, self.builder.add(raw_index, size), raw_index)
            lower_ok = self.builder.icmp_signed('>=', normalized, ir.Constant(ir_i64, 0))
            upper_ok = self.builder.icmp_signed('<', normalized, size)
            self.guard(self.builder.and_(lower_ok, upper_ok), ERROR_INDEX_OUT_OF_BOUNDS)
            stride_ptr = self.builder.gep(metadata['strides'], [ir.Constant(ir_i64, dim)])
            stride = self.builder.load(stride_ptr)
            offset = self.builder.add(offset, self.builder.mul(normalized, stride))
        return self._array_element_address(metadata, offset), metadata['element']

    def _array_element_address(self, metadata, byte_offset):
        byte_address = self.builder.gep(metadata['data'], [byte_offset])
        return self.builder.bitcast(byte_address, ir.PointerType(to_lltype(metadata['element'])))

    def _guard_array_writeable(self, value):
        if not isinstance(value, core.Var) or value.id not in self.arrays:
            raise CodegenError('only parameter arrays can be written', value)
        flags = self.arrays[value.id]['flags']
        writeable = self.builder.icmp_unsigned(
            '!=', self.builder.and_(flags, ir.Constant(ir_i64, ARRAY_WRITEABLE)),
            ir.Constant(ir_i64, 0))
        self.guard(writeable, ERROR_ARRAY_READONLY)

    def visit_Index(self, node):
        if is_tuple(node.value.type):
            value = self.visit(node.value)
            address = self.builder.gep(value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, node.tuple_index)])
            return self.builder.load(address)
        if is_string(node.value.type):
            value = self.visit(node.value); index = node.indices[0]
            if isinstance(index, core.Slice):
                lower = self.cast(self.visit(index.lower), index.lower.type, int64_t) if index.lower else ir.Constant(ir_i64, 0)
                upper = self.cast(self.visit(index.upper), index.upper.type, int64_t) if index.upper else ir.Constant(ir_i64, 0)
                step = self.cast(self.visit(index.step), index.step.type, int64_t) if index.step else ir.Constant(ir_i64, 0)
                result = self._runtime_call(
                    'slice', string_type(),
                    [string_type(), ir_i64, ir_i64, ir_i64, ir_i64, ir_i64, ir_i64, ir.PointerType(ir_i32)],
                    [value, ir.Constant(ir_i64, int(index.lower is not None)), lower,
                     ir.Constant(ir_i64, int(index.upper is not None)), upper,
                     ir.Constant(ir_i64, int(index.step is not None)), step, self.error_ptr])
                self.propagate_error(); return result
            raw_index = self.cast(self.visit(index), index.type, int64_t)
            return self._string_character(value, raw_index)
        if node.value.type == shape_t:
            if not isinstance(node.value, core.Prim) or not isinstance(node.value.args[0], core.Var): raise CodegenError('shape value is not indexable', node)
            metadata = self.arrays[node.value.args[0].id]
            shape = metadata['shape']
            raw_index = self.cast(self.visit(node.indices[0]), node.indices[0].type, int64_t)
            negative = self.builder.icmp_signed('<', raw_index, ir.Constant(ir_i64, 0))
            index = self.builder.select(negative, self.builder.add(raw_index, metadata['ndim']), raw_index)
            lower_ok = self.builder.icmp_signed('>=', index, ir.Constant(ir_i64, 0))
            upper_ok = self.builder.icmp_signed('<', index, metadata['ndim'])
            self.guard(self.builder.and_(lower_ok, upper_ok), ERROR_INDEX_OUT_OF_BOUNDS)
            return self.builder.load(self.builder.gep(shape, [index]))
        address, _ = self._index_address(node.value, node.indices)
        return self.builder.load(address, align=1)

    def visit_Assign(self, node):
        value = self.visit(node.value); value = self.cast(value, node.value.type, node.type)
        ptr = self.locals.get(node.ref)
        if ptr is None: raise CodegenError(f'unknown local {node.ref}', node)
        self.builder.store(value, ptr)

    def visit_UnpackAssign(self, node):
        value = self.visit(node.value)
        for index, name in enumerate(node.refs):
            address = self.builder.gep(value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, index)])
            item = self.cast(self.builder.load(address), node.source_types[index], node.ref_types[index])
            self.builder.store(item, self.locals[name])

    def visit_StoreIndex(self, node):
        address, element = self._index_address(node.value, node.indices)
        self._guard_array_writeable(node.value)
        value = self.cast(self.visit(node.rhs), node.rhs.type, element)
        self.builder.store(value, address, align=1)

    def visit_AugStoreIndex(self, node):
        address, element = self._index_address(node.value, node.indices)
        self._guard_array_writeable(node.value)
        left = self.cast(self.builder.load(address, align=1), element, node.operand_type)
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
        self.builder.store(self.cast(result, node.type, element), address, align=1)

    def _compare(self, op, left, right, ty):
        if is_string(ty):
            if op in core.MEMBERSHIP_OPS:
                contains = self._checked_runtime_call(
                    'contains', ir_i64, [string_type(), string_type()], [left, right])
                present = self.builder.icmp_signed('!=', contains, ir.Constant(ir_i64, 0))
                return self.builder.not_(present) if op == 'notin#' else present
            compared = self._checked_runtime_call(
                'compare', ir_i64, [string_type(), string_type()], [left, right])
            zero = ir.Constant(ir_i64, 0)
            predicates = {'eq#': '==', 'ne#': '!=', 'lt#': '<', 'le#': '<=', 'gt#': '>', 'ge#': '>='}
            return self.builder.icmp_signed(predicates[op], compared, zero)
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
        if is_string(left) and is_string(right): return str_t
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
        if node.fn == 'add#' and is_string(node.type):
            return self._checked_runtime_call(
                'concat', string_type(), [string_type(), string_type()],
                [self.visit(node.args[0]), self.visit(node.args[1])])
        if node.fn == 'mult#' and is_string(node.type):
            if is_string(node.args[0].type): value, count_node = self.visit(node.args[0]), node.args[1]
            else: value, count_node = self.visit(node.args[1]), node.args[0]
            count = self.cast(self.visit(count_node), count_node.type, int64_t)
            return self._checked_runtime_call(
                'repeat', string_type(), [string_type(), ir_i64], [value, count])
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
        quotient, remainder = self._truncating_divmod(left, right)
        has_remainder = self.builder.icmp_signed('!=', remainder, ir.Constant(left.type, 0))
        different_sign = self.builder.icmp_signed('<', self.builder.xor(left, right), ir.Constant(left.type, 0))
        correction = self.builder.and_(has_remainder, different_sign)
        return self.builder.sub(quotient, self.builder.zext(correction, left.type))

    def _integer_mod(self, left, right):
        _, remainder = self._truncating_divmod(left, right)
        has_remainder = self.builder.icmp_signed('!=', remainder, ir.Constant(left.type, 0))
        different_sign = self.builder.icmp_signed('<', self.builder.xor(left, right), ir.Constant(left.type, 0))
        correction = self.builder.and_(has_remainder, different_sign)
        return self.builder.add(remainder, self.builder.select(correction, right, ir.Constant(right.type, 0)))

    def _truncating_divmod(self, left, right):
        """Lower signed div/rem without LLVM poison for MIN_INT / -1."""
        width = left.type.width
        minimum = ir.Constant(left.type, -(1 << (width - 1)))
        minus_one = ir.Constant(right.type, -1)
        is_minimum = self.builder.icmp_signed('==', left, minimum)
        is_minus_one = self.builder.icmp_signed('==', right, minus_one)
        overflow = self.builder.and_(is_minimum, is_minus_one)
        normal_block = self.new_block('divmod_normal')
        overflow_block = self.new_block('divmod_overflow')
        end_block = self.new_block('divmod_end')
        self.builder.cbranch(overflow, overflow_block, normal_block)

        self.set_block(normal_block)
        quotient = self.builder.sdiv(left, right)
        remainder = self.builder.srem(left, right)
        self.builder.branch(end_block)

        self.set_block(overflow_block)
        self.builder.branch(end_block)

        self.set_block(end_block)
        quotient_result = self.builder.phi(left.type)
        quotient_result.add_incoming(quotient, normal_block)
        quotient_result.add_incoming(minimum, overflow_block)
        remainder_result = self.builder.phi(left.type)
        remainder_result.add_incoming(remainder, normal_block)
        remainder_result.add_incoming(ir.Constant(left.type, 0), overflow_block)
        return quotient_result, remainder_result

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
        exponent_node = node.rhs if isinstance(node, core.AugStoreIndex) else node.args[1]
        exponent = core.integer_constant_value(exponent_node)
        if exponent is None: raise CodegenError('integer power requires a constant exponent', node)
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

    def visit_ForEach(self, node):
        iterable = self.visit(node.iterable)
        init, test, body, latch, after = (self.new_block('foreach_init'), self.new_block('foreach_test'),
                                          self.new_block('foreach_body'), self.new_block('foreach_latch'),
                                          self.new_block('foreach_after'))
        otherwise = self.new_block('foreach_else') if node.orelse else after
        index_ptr = self.builder.alloca(ir_i64, name=f'foreach_index_{self.counter}')
        character_descriptor = None
        if is_string(node.iterable.type):
            character_descriptor = self._allocate_structure(string_type())
        self.builder.branch(init); self.set_block(init)
        self.builder.store(ir.Constant(ir_i64, 0), index_ptr)
        if is_array(node.iterable.type):
            if not isinstance(node.iterable, core.Var) or node.iterable.id not in self.arrays:
                raise CodegenError('only parameter arrays can be iterated', node.iterable)
            metadata = self.arrays[node.iterable.id]
            self.guard(self.builder.icmp_signed('==', metadata['ndim'], ir.Constant(ir_i64, 1)),
                       ERROR_ARRAY_DIMENSION_MISMATCH)
            length = self.builder.load(metadata['shape'])
        else:
            length = self.builder.load(self.builder.gep(iterable, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
        self.builder.branch(test); self.set_block(test)
        index = self.builder.load(index_ptr)
        self.builder.cbranch(self.builder.icmp_signed('<', index, length), body, otherwise)
        self.break_blocks.append(after); self.continue_blocks.append(latch)
        self.set_block(body)
        if is_array(node.iterable.type):
            stride = self.builder.load(metadata['strides'])
            address = self._array_element_address(metadata, self.builder.mul(index, stride))
            item = self.builder.load(address, align=1)
        else:
            data = self.builder.load(self.builder.gep(
                iterable, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 0)]))
            character = self.builder.gep(data, [index])
            self.builder.store(character, self.builder.gep(
                character_descriptor, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 0)]))
            self.builder.store(ir.Constant(ir_i64, 1), self.builder.gep(
                character_descriptor, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
            item = character_descriptor
        self.builder.store(item, self.locals[node.var.id]); self.visit(node.body)
        if not self.terminated(): self.builder.branch(latch)
        self.set_block(latch); self.builder.store(self.builder.add(self.builder.load(index_ptr), ir.Constant(ir_i64, 1)), index_ptr); self.builder.branch(test)
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
        if node.fn.id == 'len':
            value = args[0]
            if is_tuple(node.args[0].type): return ir.Constant(ir_i64, len(node.args[0].type.elements))
            if is_string(node.args[0].type):
                return self.builder.load(self.builder.gep(value, [ir.Constant(ir_i32, 0), ir.Constant(ir_i32, 1)]))
            if not isinstance(node.args[0], core.Var) or node.args[0].id not in self.arrays:
                raise CodegenError('len only supports parameter arrays', node.args[0])
            metadata = self.arrays[node.args[0].id]
            self.guard(self.builder.icmp_signed('>', metadata['ndim'], ir.Constant(ir_i64, 0)), ERROR_ARRAY_DIMENSION_MISMATCH)
            return self.builder.load(metadata['shape'])
        if node.fn.id == 'abs':
            value = args[0]
            if is_float(node.type):
                name = 'llvm.fabs.f32' if node.type == float32_t else 'llvm.fabs.f64'
                fn = self.module.globals.get(name) or ir.Function(self.module, ir.FunctionType(value.type, [value.type]), name)
                return self.builder.call(fn, [value])
            negative = self.builder.icmp_signed('<', value, ir.Constant(value.type, 0))
            return self.builder.select(negative, self.builder.neg(value), value)
        if node.fn.id in ('min', 'max'):
            left = self.cast(args[0], node.args[0].type, node.type); right = self.cast(args[1], node.args[1].type, node.type)
            if is_float(node.type): predicate = self.builder.fcmp_ordered('<=' if node.fn.id == 'min' else '>=', left, right)
            else: predicate = self.builder.icmp_signed('<=' if node.fn.id == 'min' else '>=', left, right)
            return self.builder.select(predicate, left, right)
        if node.fn.id == 'ord':
            result = self._runtime_call('ord', ir_i64, [string_type(), ir.PointerType(ir_i32)],
                                        [args[0], self.error_ptr])
            self.propagate_error(); return result
        if node.fn.id == 'chr':
            value = self.cast(args[0], node.args[0].type, int64_t)
            result = self._runtime_call('chr', string_type(), [ir_i64, ir.PointerType(ir_i32)],
                                        [value, self.error_ptr])
            self.propagate_error(); return result
        if node.fn.id in MATH_INTRINSICS:
            value = self.cast(args[0], node.args[0].type, double64_t)
            return self._math_intrinsic(node.fn.id[5:], value)
        if node.fn.id in ('sum', 'any', 'all'):
            return self._array_reduction(node, node.fn.id)
        if node.fn.id in STRING_INTRINSICS:
            name = node.fn.id[4:]
            if node.fn.id in STRING_TRANSFORMS:
                return self._checked_runtime_call(name, string_type(), [string_type()], args)
            if node.fn.id == 'str.replace':
                return self._checked_runtime_call(name, string_type(), [string_type()] * 3, args)
            if node.fn.id in STRING_PREDICATES:
                return self._checked_runtime_call(name, ir_i64, [string_type()], args)
            return self._checked_runtime_call(
                name, ir_i64, [string_type(), string_type()], args)
        if node.fn.id == self.org_func_name:
            result = self.builder.call(self.function, args + [self.error_ptr])
            self.propagate_error()
            return result
        if hasattr(node, 'jit_signature'):
            signature = node.jit_signature
            ll_args = [to_lltype(ty) for ty in signature.args]
            function_type = ir.FunctionType(to_lltype(signature.return_type), ll_args + [ir.PointerType(ir_i32)])
            callee = self.module.globals.get(node.jit_symbol)
            if callee is None: callee = ir.Function(self.module, function_type, node.jit_symbol)
            result = self.builder.call(callee, args + [self.error_ptr])
            self.propagate_error()
            return result
        registered_identifier = getattr(node, 'registered_id', node.fn.id)
        registered = get_registered(registered_identifier)
        if registered is None: raise CodegenError(f'function {node.fn.id!r} is not registered', node)
        fn, signature = registered; ll_args = [to_lltype(ty) for ty in signature.args]; ll_return = to_lltype(signature.return_type)
        def callback_type(ty) -> Any:
            if ty in (int64_t, bool_t): return ctypes.c_int64
            if ty == int32_t: return ctypes.c_int32
            if ty == float32_t: return ctypes.c_float
            if ty == double64_t: return ctypes.c_double
            if ty == str_t: return StringPointer
            if ty == void_t: return None
            raise CodegenError(f'unsupported callback type {ty}', node)
        c_args = [callback_type(ty) for ty in signature.args]
        c_return = ctypes.c_void_p if signature.return_type == str_t else callback_type(signature.return_type)
        def error_result():
            if signature.return_type in (str_t, void_t): return None
            if signature.return_type in (float32_t, double64_t): return 0.0
            return 0
        def bridge(*bridge_values):
            values, error = bridge_values[:-1], bridge_values[-1]
            try:
                python_values = [to_python(value) if ty == str_t else value for value, ty in zip(values, signature.args)]
                result = fn(*python_values)
                if signature.return_type == str_t:
                    if not isinstance(result, str): raise TypeError('@reg callback must return str')
                    return ctypes.cast(make_string(result), ctypes.c_void_p).value
                if signature.return_type == void_t: return None
                if signature.return_type in (float32_t, double64_t): return float(result)
                return int(result)
            except BaseException as exception:
                set_pending_exception(exception)
                error[0] = 11
                return error_result()
        callback = ctypes.CFUNCTYPE(c_return, *c_args, ctypes.POINTER(ctypes.c_int32))(bridge)
        address = keep_callback(registered_identifier, callback)
        pointer_ty = ir.PointerType(ir.FunctionType(ll_return, ll_args + [ir.PointerType(ir_i32)]))
        pointer = self.builder.inttoptr(ir.Constant(ir_i64, address), pointer_ty)
        result = self.builder.call(pointer, args + [self.error_ptr])
        self.propagate_error()
        return result

    def _math_intrinsic(self, name, value):
        infinity = ir.Constant(ir_f64, float('inf'))
        negative_infinity = ir.Constant(ir_f64, float('-inf'))
        is_positive_infinite = self.builder.fcmp_ordered('==', value, infinity)
        is_negative_infinite = self.builder.fcmp_ordered('==', value, negative_infinity)
        is_infinite = self.builder.or_(is_positive_infinite, is_negative_infinite)
        is_nan = self.builder.fcmp_unordered('!=', value, value)
        if name == 'isnan': return self.builder.zext(is_nan, ir_i64)
        if name == 'isinf': return self.builder.zext(is_infinite, ir_i64)
        if name == 'isfinite': return self.builder.zext(self.builder.not_(self.builder.or_(is_nan, is_infinite)), ir_i64)

        if name in ('sin', 'cos'):
            self.guard(self.builder.not_(is_infinite), ERROR_MATH_DOMAIN)
        elif name == 'sqrt':
            nonnegative = self.builder.fcmp_ordered('>=', value, ir.Constant(ir_f64, 0.0))
            self.guard(self.builder.or_(is_nan, nonnegative), ERROR_MATH_DOMAIN)
        elif name in ('log', 'log2', 'log10'):
            positive = self.builder.fcmp_ordered('>', value, ir.Constant(ir_f64, 0.0))
            self.guard(self.builder.or_(is_nan, positive), ERROR_MATH_DOMAIN)

        llvm_name = f'llvm.{name}.f64'
        intrinsic = self.module.globals.get(llvm_name) or ir.Function(
            self.module, ir.FunctionType(ir_f64, [ir_f64]), llvm_name)
        result = self.builder.call(intrinsic, [value])
        if name == 'exp':
            result_is_infinite = self.builder.fcmp_ordered('==', result, infinity)
            finite_input = self.builder.not_(self.builder.or_(is_nan, is_infinite))
            self.guard(self.builder.not_(self.builder.and_(finite_input, result_is_infinite)), ERROR_MATH_RANGE)
        return result

    def _array_reduction(self, node, operation):
        argument = node.args[0]
        if not isinstance(argument, core.Var) or argument.id not in self.arrays:
            raise CodegenError(f'{operation} only supports parameter arrays', argument)
        metadata = self.arrays[argument.id]
        result_type = to_lltype(node.type)
        initial = 1 if operation == 'all' else 0
        initial_value = ir.Constant(result_type, float(initial) if is_float(node.type) else initial)
        result_ptr = self.builder.alloca(result_type, name=f'{operation}_result_{self.counter}')
        index_ptr = self.builder.alloca(ir_i64, name=f'{operation}_index_{self.counter}')
        length_ptr = self.builder.alloca(ir_i64, name=f'{operation}_length_{self.counter}')
        dimension_ptr = self.builder.alloca(ir_i64, name=f'{operation}_dimension_{self.counter}')
        remaining_ptr = self.builder.alloca(ir_i64, name=f'{operation}_remaining_{self.counter}')
        offset_ptr = self.builder.alloca(ir_i64, name=f'{operation}_offset_{self.counter}')
        self.counter += 1
        self.builder.store(initial_value, result_ptr)
        self.builder.store(ir.Constant(ir_i64, 0), index_ptr)
        self.builder.store(ir.Constant(ir_i64, 1), length_ptr)
        dimension_test = self.new_block(f'{operation}_dimension_test')
        dimension_body = self.new_block(f'{operation}_dimension_body')
        dimension_done = self.new_block(f'{operation}_dimension_done')
        self.builder.store(ir.Constant(ir_i64, 0), dimension_ptr)
        self.builder.branch(dimension_test)
        self.set_block(dimension_test)
        dimension = self.builder.load(dimension_ptr)
        self.builder.cbranch(self.builder.icmp_signed('<', dimension, metadata['ndim']),
                             dimension_body, dimension_done)
        self.set_block(dimension_body)
        size = self.builder.load(self.builder.gep(metadata['shape'], [dimension]))
        length = self.builder.load(length_ptr)
        self.builder.store(self.builder.mul(length, size), length_ptr)
        self.builder.store(self.builder.add(dimension, ir.Constant(ir_i64, 1)), dimension_ptr)
        self.builder.branch(dimension_test)
        self.set_block(dimension_done)
        test = self.new_block(f'{operation}_test')
        body = self.new_block(f'{operation}_body')
        offset_test = self.new_block(f'{operation}_offset_test')
        offset_body = self.new_block(f'{operation}_offset_body')
        offset_done = self.new_block(f'{operation}_offset_done')
        after = self.new_block(f'{operation}_after')
        self.builder.branch(test)
        self.set_block(test)
        index = self.builder.load(index_ptr)
        length = self.builder.load(length_ptr)
        self.builder.cbranch(self.builder.icmp_signed('<', index, length), body, after)
        self.set_block(body)
        self.builder.store(index, remaining_ptr)
        self.builder.store(ir.Constant(ir_i64, 0), offset_ptr)
        self.builder.store(self.builder.sub(metadata['ndim'], ir.Constant(ir_i64, 1)), dimension_ptr)
        self.builder.branch(offset_test)
        self.set_block(offset_test)
        dimension = self.builder.load(dimension_ptr)
        self.builder.cbranch(self.builder.icmp_signed('>=', dimension, ir.Constant(ir_i64, 0)),
                             offset_body, offset_done)
        self.set_block(offset_body)
        size = self.builder.load(self.builder.gep(metadata['shape'], [dimension]))
        remaining = self.builder.load(remaining_ptr)
        coordinate = self.builder.srem(remaining, size)
        self.builder.store(self.builder.sdiv(remaining, size), remaining_ptr)
        stride = self.builder.load(self.builder.gep(metadata['strides'], [dimension]))
        offset = self.builder.load(offset_ptr)
        self.builder.store(self.builder.add(offset, self.builder.mul(coordinate, stride)), offset_ptr)
        self.builder.store(self.builder.sub(dimension, ir.Constant(ir_i64, 1)), dimension_ptr)
        self.builder.branch(offset_test)
        self.set_block(offset_done)
        address = self._array_element_address(metadata, self.builder.load(offset_ptr))
        element = self.builder.load(address, align=1)
        if operation == 'sum':
            widened = self.cast(element, metadata['element'], node.type)
            current = self.builder.load(result_ptr)
            updated = self.builder.fadd(current, widened) if is_float(node.type) else self.builder.add(current, widened)
        else:
            truth = self.truthy(element, metadata['element'], normalize=True)
            current = self.builder.load(result_ptr)
            updated = self.builder.or_(current, truth) if operation == 'any' else self.builder.and_(current, truth)
        self.builder.store(updated, result_ptr)
        self.builder.store(self.builder.add(index, ir.Constant(ir_i64, 1)), index_ptr)
        self.builder.branch(test)
        self.set_block(after)
        return self.builder.load(result_ptr)

    def _runtime_call(self, name, return_type, arg_types, args):
        pointer_type = ir.PointerType(ir.FunctionType(return_type, arg_types))
        pointer = self.builder.inttoptr(ir.Constant(ir_i64, callback_address(name)), pointer_type)
        return self.builder.call(pointer, args)

    def _checked_runtime_call(self, name, return_type, arg_types, args):
        result = self._runtime_call(
            name, return_type, arg_types + [ir.PointerType(ir_i32)],
            args + [self.error_ptr])
        self.propagate_error()
        return result

    def visit_Expr(self, node): return self.visit(node.value)
    def visit_Noop(self, node): return None
