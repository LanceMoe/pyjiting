# pyright: reportArgumentType=false

import hashlib
import sys

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from .codegen import LLVMCodeGen
from .errors import InferError
from .infer import TypeInferencer
from .ll_types import mangler, wrap_module
from .parser import ASTVisitor
from .registry import register, signatures
from .types import bool_t, double64_t, float32_t, int32_t, int64_t, make_array_type


DEBUG = False
function_cache = {}
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target_machine = llvm.Target.from_default_triple().create_target_machine()
engine = llvm.create_mcjit_compiler(llvm.parse_assembly(''), target_machine)


def debug(*values):
    if DEBUG: print(*values)


def reg(fn): return register(fn)


def arg_pytype(arg):
    if isinstance(arg, np.ndarray):
        dtype_map = {np.dtype(np.int32): int32_t, np.dtype(np.int64): int64_t, np.dtype(np.float32): float32_t, np.dtype(np.float64): double64_t}
        try: return make_array_type(dtype_map[np.dtype(arg.dtype)])
        except KeyError as error: raise TypeError(f'Unsupported ndarray dtype: {arg.dtype}') from error
    if isinstance(arg, (bool, np.bool_)): return bool_t
    if isinstance(arg, np.int32): return int32_t
    if isinstance(arg, np.int64): return int64_t
    if isinstance(arg, np.float32): return float32_t
    if isinstance(arg, np.floating) or isinstance(arg, float): return double64_t
    if isinstance(arg, int) and -sys.maxsize - 1 <= arg <= sys.maxsize: return int64_t
    raise TypeError(f'Unsupported type: {type(arg).__name__}')


def typeinfer(tree, arg_types, registry=None):
    return TypeInferencer(arg_types, registry or signatures()).visit(tree)


def compile_specialization(tree, arg_types):
    function_type = typeinfer(tree, arg_types)
    key = mangler(tree.symbol, arg_types)
    if key in function_cache: return function_cache[key]
    module = ir.Module(name=f'pyjiting.{key}')
    module.triple = llvm.get_default_triple()
    llfunc = LLVMCodeGen(module, function_type.return_type, arg_types).visit(tree)
    binding_module = llvm.parse_assembly(str(module)); binding_module.verify()
    pto = llvm.create_pipeline_tuning_options(speed_level=3); pto.loop_vectorization = True
    pass_builder = llvm.create_pass_builder(target_machine, pto)
    pass_builder.getModulePassManager().run(binding_module, pass_builder)
    engine.add_module(binding_module); engine.finalize_object()
    wrapper = wrap_module(arg_types, llfunc, engine); function_cache[key] = wrapper
    debug(module)
    return wrapper


def jit(fn):
    tree = ASTVisitor()(fn)
    identity = '\0'.join((fn.__module__, fn.__qualname__, fn.__code__.co_filename, str(fn.__code__.co_firstlineno))).encode()
    tree.symbol = 'jit_' + hashlib.sha256(identity).hexdigest()[:16]
    def wrapper(*args): return compile_specialization(tree, [arg_pytype(arg) for arg in args])(*args)
    wrapper.__name__, wrapper.__doc__, wrapper.__wrapped__ = fn.__name__, fn.__doc__, fn
    return wrapper


def codegen(module, ast, specializer=None, return_type=None, args=None):
    """Compatibility entry point used by external callers."""
    if args is None: raise InferError('codegen requires concrete argument types')
    function_type = typeinfer(ast, args)
    return LLVMCodeGen(module, return_type or function_type.return_type, args).visit(ast)
