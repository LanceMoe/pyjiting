import ctypes
import sys
from ctypes import c_int64, c_void_p

import llvmlite.binding as llvm
import llvmlite.ir as ir

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

target_machine = llvm.Target.from_default_triple().create_target_machine()

c_func_t = ir.FunctionType(
    ir.IntType(64),
    [ir.IntType(64), ir.IntType(64)]
)
c_func_t_ptr = c_func_t.as_pointer()
i64_t = ir.IntType(64)


def create_wrap_caller_module(addr):
    module = ir.Module()
    wrap_caller_func_t = ir.FunctionType(i64_t, [i64_t, i64_t])
    wrap_caller_func = ir.Function(
        module, wrap_caller_func_t, name='wrap_caller')
    a = wrap_caller_func.args[0]
    a.name = 'a'
    b = wrap_caller_func.args[1]
    b.name = 'b'
    ir_builder = ir.IRBuilder(wrap_caller_func.append_basic_block('entry'))
    func_ptr = ir_builder.inttoptr(
        ir.Constant(i64_t, addr),
        c_func_t_ptr, name='func_ptr'
    )
    call = ir_builder.call(func_ptr, [a, b])
    ir_builder.ret(call)
    return module


def foo(a, b):
    print('foo: a={0}, b={1}'.format(a, b))
    return a + b


def main():
    FUNC_T = ctypes.CFUNCTYPE(c_int64, c_int64, c_int64)

    foo_ptr = ctypes.cast(FUNC_T(foo), c_void_p).value
    print(f'foo_ptr = 0x{foo_ptr:x}')

    module = create_wrap_caller_module(foo_ptr)
    print(module)

    llvm_module = llvm.parse_assembly(str(module))

    with llvm.create_mcjit_compiler(llvm_module, target_machine) as ee:
        ee.finalize_object()
        print('Calling wrap_caller')
        wrap_caller = FUNC_T(ee.get_function_address('wrap_caller'))
        res = wrap_caller(114, 514)
        print('Output:', res)


if __name__ == '__main__':
    main()
