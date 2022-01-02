
import ast
from collections import defaultdict

import llvmlite.llvmpy.core as lc
from llvmlite import ir

from pyjiting.type_mapping import mangler

from .core_lang import LLVM_PRIM_OPS, Var
from .type_system import *

# == LLVM Codegen ==

ir_ptr_t = ir.PointerType
ir_int32_t = ir.IntType(32)
ir_int64_t = ir.IntType(64)
ir_float_t = ir.FloatType()
ir_double_t = ir.DoubleType()
ir_bool_t = ir.IntType(32)
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
        ir.IntType(32),           # dimensions
        ir_ptr_t(ir.IntType(32)),  # shape
    )
    return struct_type


ir_int32_array_t = ir_ptr_t(array_type(ir_int32_t))
ir_int64_array_t = ir_ptr_t(array_type(ir_int64_t))
ir_double_array_t = ir_ptr_t(array_type(ir_double_t))

lltypes_map = {
    int32_t: ir_int32_t,
    int64_t: ir_int64_t,
    float32_t: ir_float_t,
    double64_t: ir_double_t,
    int32_array_t: ir_int32_array_t,
    int64_array_t: ir_int64_array_t,
    double64_array_t: ir_double_array_t
}


def to_lltype(ptype):
    return lltypes_map[ptype]


def determined(ty):
    return len(ftv(ty)) == 0


class LLVMCodeGen(object):
    def __init__(self, module, spec_types, retty, argtys):
        self.module = module             # LLVM Module
        self.function = None             # LLVM Function
        self.builder = None              # LLVM Builder
        self.locals = {}                 # Local variables
        self.arrays = defaultdict(dict)  # Array metadata
        self.exit_block = None           # Exit block
        self.spec_types = spec_types     # Type specialization
        self.retty = retty               # Return type
        self.argtys = argtys             # Argument types

    def start_function(self, name, module, rettype, argtypes):
        func_type = ir.FunctionType(rettype, argtypes, False)
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
            # print('specialize', value.type.s)
            return to_lltype(self.spec_types[value.type.s])
        else:
            return value.type

    def const(self, value):
        if isinstance(value, bool):
            return ir.Constant(ir_bool_t, int(value))
        elif isinstance(value, int):
            return ir.Constant(ir_int64_t, value)
        elif isinstance(value, float):
            return ir.Constant(ir_double_t, value)
        elif isinstance(value, str):
            # raise NotImplementedError
            return lc.Constant.stringz(value)
        else:
            raise NotImplementedError

    def visit_Const(self, node):
        return self.const(node.value)

    def visit_LitInt(self, node):
        ty = self.specialize(node)
        if ty is ir_double_t:
            return ir.Constant(ir_double_t, node.n)
        elif ty == ir_int64_t:
            return ir.Constant(ir_int64_t, node.n)
        elif ty == ir_int32_t:
            return ir.Constant(ir_int32_t, node.n)

    def visit_LitFloat(self, node):
        ty = self.specialize(node)
        if ty is ir_double_t:
            return ir.Constant(ir_double_t, node.n)
        elif ty == ir_int64_t:
            return ir.Constant(ir_int64_t, node.n)
        elif ty == ir_int32_t:
            return ir.Constant(ir_int32_t, node.n)

    def visit_Noop(self, node):
        pass

    def visit_Fun(self, node):
        rettype = to_lltype(self.retty)
        argtypes = list(map(to_lltype, self.argtys))
        # Create a unique specialized name
        func_name = mangler(node.fname, self.argtys)
        self.start_function(func_name, self.module, rettype, argtypes)

        for (ar, llarg, argty) in list(zip(node.args, self.function.args, self.argtys)):
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
        if rettype is not ir_void_t:
            self.locals['retval'] = self.builder.alloca(rettype, name='retval')

        list(map(self.visit, node.body))
        self.end_function()

    def visit_Index(self, node):
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

    def visit_Var(self, node):
        return self.builder.load(self.locals[node.id])

    def visit_Return(self, node):
        value = self.visit(node.value)
        if value.type != ir_void_t:
            self.builder.store(value, self.locals['retval'])
        self.builder.branch(self.exit_block)

    def visit_Loop(self, node):
        init_block = self.function.append_basic_block('for.init')
        test_block = self.function.append_basic_block('for.cond')
        body_block = self.function.append_basic_block('for.body')
        end_block = self.function.append_basic_block('for.end')

        self.branch(init_block)
        self.set_block(init_block)

        start = self.visit(node.begin)
        stop = self.visit(node.end)
        step = 1

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

    def visit_Prim(self, node):
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
        elif node.fn == 'neq#':
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

    def visit_Assign(self, node):
        # Subsequent assignment
        if node.ref in self.locals:
            name = node.ref
            var = self.locals[name]
            value = self.visit(node.value)
            self.builder.store(value, var)
            self.locals[name] = var
            return var

        # First assignment
        else:
            name = node.ref
            value = self.visit(node.value)
            ty = self.specialize(node)
            var = self.builder.alloca(ty, name=name)
            # print('visit_Assign', type(node.value), node.value, var, sep='???')
            self.builder.store(value, var)
            self.locals[name] = var
            return var

    def visit_NoneType(self, node):
        return None

    def visit_If(self, node):
        # TODO:
        pass

    def visit(self, node):
        name = f'visit_{type(node).__name__}'
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))
