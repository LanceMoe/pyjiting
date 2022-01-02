import sys

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from .constraint_solver import apply, compose, solve, unify
from .core_translator import ASTVisitor
from .llvm_codegen import LLVMCodeGen, determined
from .type_inference import TypeInferencer, UnderDetermined
from .type_mapping import mangler, wrap_module
from .type_system import *
from ast import dump as ast_dump

### == Toplevel ==
DEBUG = True


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

module = ir.Module('numpile.module')
engine = None
function_cache = {}

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
backing_mod = llvm.parse_assembly('')
engine = llvm.create_mcjit_compiler(backing_mod, target_machine)


def autojit(fn):
    transformer = ASTVisitor()
    ast = transformer(fn)
    (ty, mgu) = typeinfer(ast)
    # print('ty:', ty, '\nmgu:', mgu)
    debug(ast_dump(ast, indent=4))
    return specialize(ast, ty, mgu)


def arg_pytype(arg):
    if isinstance(arg, np.ndarray):
        if arg.dtype == np.dtype('int64'):
            return make_array_type(int64_t)
        elif arg.dtype == np.dtype('double'):
            return make_array_type(double64_t)
        elif arg.dtype == np.dtype('float'):
            return make_array_type(float32_t)
    elif isinstance(arg, int) and arg <= sys.maxsize:
        return int64_t
    elif isinstance(arg, float):
        return double64_t
    else:
        raise Exception(f'Type not supported: {type(arg)}')


def specialize(ast, infer_ty, mgu):
    def _wrapper(*args):
        types = list(map(arg_pytype, list(args)))
        spec_ty = FuncType(argtys=types, retty=VarType('$retty'))
        unifier = unify(infer_ty, spec_ty)
        specializer = compose(unifier, mgu)
        debug('specializer:', specializer)

        retty = apply(specializer, VarType('$retty'))
        argtys = [apply(specializer, ty) for ty in types]
        debug('Specialized Function:', FuncType(argtys, retty))

        is_deteremined_retty = determined(retty)
        is_deteremined_argtys = all(map(determined, argtys))
        print('is_deteremined_retty', is_deteremined_retty)
        print('is_deteremined_argtys', is_deteremined_argtys)
        if is_deteremined_retty and is_deteremined_argtys:
            key = mangler(ast.fname, argtys)
            # Don't recompile after we've specialized.
            if key in function_cache:
                return function_cache[key](*args)
            else:
                llfunc = codegen(module, ast, specializer, retty, argtys)
                pyfunc = wrap_module(argtys, llfunc, engine)
                function_cache[key] = pyfunc
                return pyfunc(*args)
        else:
            raise UnderDetermined()
    return _wrapper


def typeinfer(ast):
    infer = TypeInferencer()
    ty = infer.visit(ast)
    mgu = solve(infer.constraints)
    infer_ty = apply(mgu, ty)
    debug('infer_ty', infer_ty)
    debug('mgu', mgu)
    debug('infer.constraints', infer.constraints)
    return (infer_ty, mgu)


def codegen(module, ast, specializer, retty, argtys):
    cgen = LLVMCodeGen(module, specializer, retty, argtys)
    cgen.visit(ast)

    print(str(module))
    mod = llvm.parse_assembly(str(module))
    mod.verify()

    pmb = llvm.PassManagerBuilder()
    pmb.opt_level = 3
    pmb.loop_vectorize = True

    pm = llvm.ModulePassManager()
    pmb.populate(pm)

    pm.run(mod)

    engine.add_module(mod)

    debug(cgen.function)
    debug(target_machine.emit_assembly(mod))
    return cgen.function


def debug(fmt, *args):
    if DEBUG:
        print('=' * 80)
        print(fmt, *args)
