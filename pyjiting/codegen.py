import ast
from collections import defaultdict
from ctypes import c_int64, c_void_p
import ctypes

import llvmlite.llvmpy.core as lc
from llvmlite import ir

from pyjiting.ll_types import mangler

from .ast import (LLVM_PRIM_OPS, Assign, Break, CallFunc, Compare, Const, Expr, Fun, If, Index,
                  LitFloat, LitInt, Loop, Noop, Prim, Return, Var)
from .types import *

'''
Codegen is a module that takes an AST and generates LLVM IR.
'''

reg_known_func = {}

ir_ptr_t = ir.PointerType
ir_int32_t = ir.IntType(32)
ir_int64_t = ir.IntType(64)
ir_float_t = ir.FloatType()
ir_double_t = ir.DoubleType()
ir_bool_t = ir.IntType(64)
ir_void_t = ir.VoidType()
ir_void_ptr_t = ir_ptr_t(ir.IntType(8))


def array_type(elt_type):
    struct_type = ir.global_context.get_identified_type(
        'ndarray_' + str(elt_type))

    # The type can already exist.
    if struct_type.elements:
        return struct_type

    # If not, initialize it.
    struct_type.set_body(
        ir_ptr_t(elt_type),        # data
        ir_int32_t,           # dimensions
        ir_ptr_t(ir_int32_t),  # shape
    )
    return struct_type


ir_int32_array_t = ir_ptr_t(array_type(ir_int32_t))
ir_int64_array_t = ir_ptr_t(array_type(ir_int64_t))
ir_double_array_t = ir_ptr_t(array_type(ir_double_t))

lltypes_map = {
    int32_t: ir_int32_t,
    int64_t: ir_int64_t,
    bool_t: ir_bool_t,
    float32_t: ir_float_t,
    double64_t: ir_double_t,
    int32_array_t: ir_int32_array_t,
    int64_array_t: ir_int64_array_t,
    double64_array_t: ir_double_array_t,
    void_t: ir_void_t,
}


def to_lltype(ptype):
    return lltypes_map[ptype]


def determined(ty):
    return len(ftv(ty)) == 0


def reg_func(func_name, func):
    reg_known_func[func_name] = func


def get_reg_func(func_name):
    return reg_known_func.get(func_name, None)


def arg_ctype(arg):
    if arg == ir_int64_t:
        return c_int64
    raise RuntimeError('Unsupported type:', arg)


def arg_classtype(arg):
    if arg is int:
        return int64_t
    elif arg is float:
        return double64_t
    else:
        raise RuntimeError('Unsupported type:', arg)


class LLVMCodeGen(object):
    def __init__(self, module, spec_types, return_type, args):
        self.module = module             # LLVM Module
        self.function = None             # LLVM Function
        self.builder = None              # LLVM Builder
        self.locals = {}                 # Local variables
        self.arrays = defaultdict(dict)  # Array metadata
        self.exit_block = None           # Exit block
        self.spec_types = spec_types     # Type specialization
        self.return_type = return_type   # Return type
        self.args = args                 # Argument types
        self.org_func_name = None        # Original function name
        self.break_block_stack = []      # Break block stack

    def start_function(self, name, module, ir_ret_type, argtypes):
        func_type = ir.FunctionType(ir_ret_type, argtypes, False)
        function = ir.Function(module, func_type, name)
        entry_block = function.append_basic_block('entry')
        builder = ir.IRBuilder(entry_block)
        self.exit_block = function.append_basic_block('exit')
        self.function = function
        self.builder = builder

    def end_function(self):
        self.builder.position_at_end(self.exit_block)

        if 'retval' in self.locals:
            retval = self.builder.load(self.locals['retval'])
            self.builder.ret(retval)
        else:
            self.builder.ret_void()

    def add_block(self, name):
        return self.function.append_basic_block(name)

    def set_block(self, block):
        self.block = block
        self.builder.position_at_end(block)

    def cbranch(self, cond, true_block, false_block):
        self.builder.cbranch(cond, true_block, false_block)

    def branch(self, next_block):
        self.builder.branch(next_block)

    def specialize(self, value):
        if isinstance(value.type, VarType):
            return to_lltype(self.spec_types[value.type.s])
        if isinstance(value.type, BaseType):
            return to_lltype(value.type)
        return to_lltype(value.type)

    def const(self, value):
        if value is None:
            return ir.Constant(ir_void_t, None)
        elif isinstance(value, ir.Constant):
            return value
        elif isinstance(value, bool):
            return ir.Constant(ir_bool_t, int(value))
        elif isinstance(value, int):
            return ir.Constant(ir_int64_t, value)
        elif isinstance(value, float):
            return ir.Constant(ir_double_t, value)
        elif isinstance(value, str):
            # raise NotImplementedError
            return lc.Constant.stringz(value)
        else:
            print(value, type(value))
            raise NotImplementedError

    def visit_Const(self, node: Const):
        return self.const(node.value)

    def visit_LitInt(self, node: LitInt):
        ty = self.specialize(node)
        if ty is ir_double_t:
            return ir.Constant(ir_double_t, node.n)
        elif ty == ir_int64_t:
            return ir.Constant(ir_int64_t, node.n)
        elif ty == ir_int32_t:
            return ir.Constant(ir_int32_t, node.n)

    def visit_LitFloat(self, node: LitFloat):
        ty = self.specialize(node)
        if ty is ir_double_t:
            return ir.Constant(ir_double_t, node.n)
        elif ty == ir_int64_t:
            return ir.Constant(ir_int64_t, node.n)
        elif ty == ir_int32_t:
            return ir.Constant(ir_int32_t, node.n)

    def visit_Noop(self, node: Noop):
        pass

    def visit_Fun(self, node: Fun):
        ir_ret_type = to_lltype(self.return_type)
        argtypes = list(map(to_lltype, self.args))
        # Create a unique specialized name
        func_name = mangler(node.fname, self.args)
        self.org_func_name = node.fname
        self.start_function(func_name, self.module, ir_ret_type, argtypes)

        for (ar, llarg, argty) in list(zip(node.args, self.function.args, self.args)):
            name = ar.id
            llarg.name = name

            if is_array(argty):
                zero = self.const(0)
                one = self.const(1)
                two = self.const(2)

                data = self.builder.gep(llarg, [
                                        zero, zero], name=(name + '_data'))
                dims = self.builder.gep(llarg, [
                                        zero, one], name=(name + '_dims'))
                shape = self.builder.gep(llarg, [
                                         zero, two], name=(name + '_strides'))

                self.arrays[name]['data'] = self.builder.load(data)
                self.arrays[name]['dims'] = self.builder.load(dims)
                self.arrays[name]['shape'] = self.builder.load(shape)
                self.locals[name] = llarg
            else:
                argref = self.builder.alloca(to_lltype(argty))
                self.builder.store(llarg, argref)
                self.locals[name] = argref

        # Setup the register for return type.
        if ir_ret_type is not ir_void_t:
            self.locals['retval'] = self.builder.alloca(
                ir_ret_type, name='retval')

        list(map(self.visit, node.body))
        self.end_function()

    def visit_Index(self, node: Index):
        if isinstance(node.value, Var) and node.value.id in self.arrays:
            value = self.visit(node.value)
            ix = self.visit(node.ix)
            dataptr = self.arrays[node.value.id]['data']
            ret = self.builder.gep(dataptr, [ix])
            return self.builder.load(ret)
        else:
            value = self.visit(node.value)
            ix = self.visit(node.ix)
            ret = self.builder.gep(value, [ix])
            return self.builder.load(ret)

    def visit_Var(self, node: Var):
        return self.builder.load(self.locals[node.id])

    def visit_Return(self, node: Return):
        value = self.visit(node.value)
        if value.type != ir_void_t:
            self.builder.store(value, self.locals['retval'])
        self.builder.branch(self.exit_block)

    def visit_Loop(self, node: Loop):
        if not hasattr(self, '_for_counter'):
            self._for_counter = 0
        self._for_counter += 1
        init_block = self.add_block(f'for_init_{self._for_counter}')
        test_block = self.add_block(f'for_cond_{self._for_counter}')
        body_block = self.add_block(f'for_body_{self._for_counter}')
        end_block = self.add_block(f'for_after_{self._for_counter}')
        self.break_block_stack.append(end_block)

        self.branch(init_block)
        self.set_block(init_block)

        start = self.visit(node.begin)
        stop = self.visit(node.end)
        step = self.visit(node.step)

        # Setup the increment variable
        varname = node.var.id
        inc = self.builder.alloca(ir_int64_t, name=varname)
        self.builder.store(start, inc)
        self.locals[varname] = inc

        # Setup the loop condition
        self.branch(test_block)
        self.set_block(test_block)
        cond = self.builder.icmp_signed('<', self.builder.load(inc), stop)
        self.builder.cbranch(cond, body_block, end_block)

        # Generate the loop body
        self.set_block(body_block)
        list(map(self.visit, node.body))

        # Increment the counter
        succ = self.builder.add(self.const(step), self.builder.load(inc))
        self.builder.store(succ, inc)

        # Exit the loop
        self.builder.branch(test_block)
        self.set_block(end_block)

        # Pop the break block
        self.break_block_stack.pop()

    def visit_Break(self, node: Break):
        if self.block.terminator is None:
            self.branch(self.break_block_stack[-1])

    def visit_Prim(self, node: Prim):
        if node.fn == 'shape#':
            ref = node.args[0]
            shape = self.arrays[ref.id]['shape']
            return shape
        elif node.fn not in LLVM_PRIM_OPS:
            raise NotImplementedError(ast.dump(node))
        if node.fn == 'mult#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fmul(a, b)
            else:
                return self.builder.mul(a, b)
        elif node.fn == 'add#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fadd(a, b)
            else:
                return self.builder.add(a, b)
        elif node.fn == 'sub#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fsub(a, b)
            else:
                return self.builder.sub(a, b)
        elif node.fn == 'div#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fdiv(a, b)
            else:
                return self.builder.sdiv(a, b)
        elif node.fn == 'mod#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.frem(a, b)
            else:
                return self.builder.srem(a, b)
        elif node.fn == 'lt#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('<', a, b)
            else:
                return self.builder.icmp_signed('<', a, b)
        elif node.fn == 'gt#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('>', a, b)
            else:
                return self.builder.icmp_signed('>', a, b)
        elif node.fn == 'le#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('<=', a, b)
            else:
                return self.builder.icmp_signed('<=', a, b)
        elif node.fn == 'ge#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('>=', a, b)
            else:
                return self.builder.icmp_signed('>=', a, b)
        elif node.fn == 'eq#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('==', a, b)
            else:
                return self.builder.icmp_signed('==', a, b)
        elif node.fn == 'ne#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == ir_double_t:
                return self.builder.fcmp_unordered('!=', a, b)
            else:
                return self.builder.icmp_signed('!=', a, b)
        elif node.fn == 'and#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            return self.builder.and_(a, b)
        elif node.fn == 'or#':
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            return self.builder.or_(a, b)
        elif node.fn == 'not#':
            a = self.visit(node.args[0])
            return self.builder.not_(a)
        elif node.fn == 'neg#':
            a = self.visit(node.args[0])
            if a.type == ir_double_t:
                return self.builder.fsub(self.const(0), a)
            else:
                return self.builder.sub(self.const(0), a)
        elif node.fn == 'pow#':
            # a = self.visit(node.args[0])
            # b = self.visit(node.args[1])
            # return self.builder.call(pow_func, [a, b])
            raise NotImplementedError('pow#', ast.dump(node))

    def visit_Assign(self, node: Assign):
        # Subsequent assignment
        if node.ref in self.locals:
            name = node.ref
            ptr = self.locals[name]
            value = self.visit(node.value)
            self.builder.store(value, ptr)
            self.locals[name] = ptr
            return ptr

        # First assignment
        else:
            name = node.ref
            value = self.visit(node.value)
            ty = self.specialize(node)
            ptr = self.builder.alloca(ty, name=name)
            self.builder.store(value, ptr)
            self.locals[name] = ptr
            return ptr

    def visit_NoneType(self, node: None):
        return None

    def visit_If(self, node: If):
        if not hasattr(self, '_if_counter'):
            self._if_counter = 0
        self._if_counter += 1
        test_block = self.add_block(f'if_cond_{self._if_counter}')
        then_block = self.add_block(f'if_then_{self._if_counter}')
        if has_else := len(node.orelse) > 0:
            else_block = self.add_block(f'if_orelse_{self._if_counter}')
        end_block = self.add_block(f'if_after_{self._if_counter}')

        self.branch(test_block)
        self.set_block(test_block)
        test = self.visit(node.test)
        self.builder.cbranch(
            test, then_block, else_block if has_else else end_block)

        self.set_block(then_block)
        list(map(self.visit, node.body))
        if self.block.terminator is None:
            self.branch(end_block)

        if has_else:
            self.set_block(else_block)
            list(map(self.visit, node.orelse))
            if self.block.terminator is None:
                self.branch(end_block)

        self.set_block(end_block)

    def visit_Compare(self, node: Compare):
        # Setup the increment variable
        lf = self.visit(node.left)
        rt = self.visit(node.comparators[0])
        op = {
            'eq#': '==',
            'ne#': '!=',
            'lt#': '<',
            'gt#': '>',
            'le#': '<=',
            'ge#': '>=',
        }.get(node.ops[0], None)
        if op is None:
            raise NotImplementedError(node.ops[0])
        cond = self.builder.icmp_signed(op, lf, rt)
        return cond

    def visit_CallFunc(self, node: CallFunc):
        func_name = node.fn.id
        args = [self.visit(arg) for arg in node.args]
        func = None
        if func_name == self.org_func_name:
            # Implement recursion
            func = self.function
            return self.builder.call(func, args)

        # Call registered functions
        fn = get_reg_func(func_name)
        if fn is None:
            raise NotImplementedError(
                f'CallFunc: {func_name} function is not registered!')

        fname = fn.__name__
        types = fn.__annotations__
        arg_names = fn.__code__.co_varnames
        ir_return_type = to_lltype(arg_classtype(types['return']))
        ir_args = [to_lltype(arg_classtype(types[arg_name]))
                   for arg_name in arg_names]
        wrap_caller_func_t = ir.FunctionType(ir_return_type, ir_args)
        wrap_caller_func_t_ptr = wrap_caller_func_t.as_pointer()

        # Get source function addr
        c_args = list(map(arg_ctype, ir_args))
        FUNC_T = ctypes.CFUNCTYPE(arg_ctype(ir_return_type), *c_args)
        pyfunc_ptr = ctypes.cast(FUNC_T(fn), c_void_p).value

        # Make llvm ir wrapper function
        func_ptr = self.builder.inttoptr(
            ir.Constant(ir_int64_t, pyfunc_ptr),
            wrap_caller_func_t_ptr, name=f'{mangler(fname, args)}_ptr'
        )
        return self.builder.call(func_ptr, args)

    def visit_Expr(self, node: Expr):
        return self.visit(node.value)

    def visit(self, node):
        name = f'visit_{type(node).__name__}'
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))
