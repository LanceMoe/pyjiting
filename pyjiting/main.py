import inspect
import sys
from ast import dump as ast_dump
from ast import parse as ast_parse
from textwrap import dedent

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from .codegen import LLVMCodeGen, determined
from .infer import TypeInferencer, UnderDetermined
from .ll_types import mangler, wrap_module
from .parser import ASTVisitor
from .types import *
from .utils import apply, compose, solve, unify

# Output debug info
DEBUG = False


def debug(fmt, *args):
    if not DEBUG:
        return
    print('=' * 80)
    print(fmt, *args)


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

module = ir.Module('pyjiting.module')
function_cache = {}
target_machine = llvm.Target.from_default_triple().create_target_machine()
backing_mod = llvm.parse_assembly('')
engine = llvm.create_mcjit_compiler(backing_mod, target_machine)


def reg(fn):
    return fn

def jit(fn):
    debug(ast_dump(ast_parse(dedent(inspect.getsource(fn))), indent=4))
    transformer = ASTVisitor()
    ast = transformer(fn)
    (ty, mgu) = typeinfer(ast)
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
        raise RuntimeError('Unsupported type:', type(arg))


def specialize(ast, infer_ty, mgu):
    def _wrapper(*func_args):
        types = list(map(arg_pytype, list(func_args)))
        spec_ty = FuncType(args=types, return_type=VarType('$return_type'))
        unifier = unify(infer_ty, spec_ty)
        specializer = compose(unifier, mgu)
        debug('specializer:', specializer)

        return_type = apply(specializer, VarType('$return_type'))
        args = [apply(specializer, ty) for ty in types]
        debug('Specialized Function:', FuncType(
            args=args, return_type=return_type))

        is_deteremined_return_type = determined(return_type)
        if is_deteremined_return_type and all(map(determined, args)):
            key = mangler(ast.fname, args)
            # Don't recompile after we've specialized.
            if key in function_cache:
                return function_cache[key](*func_args)
            else:
                llfunc = codegen(module, ast, specializer, return_type, args)
                pyfunc = wrap_module(args, llfunc, engine)
                function_cache[key] = pyfunc
                return pyfunc(*func_args)
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


def codegen(module, ast, specializer, return_type, args):
    cgen = LLVMCodeGen(module, specializer, return_type, args)
    cgen.visit(ast)

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
