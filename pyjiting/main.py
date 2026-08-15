# pyright: reportArgumentType=false

import ast as py_ast
import copy
import functools
import hashlib
import itertools
import sys
import threading
from typing import Any
import warnings

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from .codegen import LLVMCodeGen
from .errors import CompileError, InferError, SpecializationLimitError
from .infer import TypeInferencer
from .ll_types import mangler, wrap_module
from .parser import ASTVisitor
from .registry import (callback_count, get as get_registered, register,
                       registration_id, signatures)
from .string_runtime import literal_count
from .types import (TupleType, bool_t, contains_array, double64_t, float32_t,
                    int32_t, int64_t, make_array_type, str_t)


DEBUG = False
function_cache = {}
function_signatures = {}
cache_lock = threading.RLock()
compile_lock = cache_lock  # Backward-compatible internal name.
engine_lock = threading.RLock()
compilation_unit_ids = itertools.count(1)
compilation_states = {}
specialization_generations = {}
retained_modules = []
runtime_counters = {'compile_hits': 0, 'compile_misses': 0, 'compile_failures': 0}
DEFAULT_MAX_SPECIALIZATIONS = 64
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target_machine = llvm.Target.from_default_triple().create_target_machine()
engine = llvm.create_mcjit_compiler(llvm.parse_assembly(''), target_machine)


def debug(*values):
    if DEBUG: print(*values)


def validate_specialization_limit(value):
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError('max_specializations must be a positive integer or None')


def reg(fn): return register(fn)


def arg_pytype(arg):
    if isinstance(arg, tuple):
        elements = [arg_pytype(element) for element in arg]
        if any(contains_array(element) for element in elements):
            raise TypeError('ndarray elements inside tuples are not supported')
        return TupleType(elements)
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
    if isinstance(arg, str): return str_t
    raise TypeError(f'Unsupported type: {type(arg).__name__}')


def typeinfer(tree, arg_types, registry=None, jit_resolver=None, reg_resolver=None):
    return TypeInferencer(arg_types, registry or signatures(), jit_resolver, reg_resolver).visit(tree)


def ensure_compilation_unit(tree):
    """Assign a process-unique identity to one decorated/parsed function tree."""
    unit_id = getattr(tree, 'compilation_unit_id', None)
    if unit_id is not None:
        return unit_id
    with compile_lock:
        unit_id = getattr(tree, 'compilation_unit_id', None)
        if unit_id is None:
            unit_id = next(compilation_unit_ids)
            tree.compilation_unit_id = unit_id
            fingerprint_source = py_ast.dump(tree, include_attributes=False)
            tree.semantic_fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
            base_symbol = getattr(tree, 'symbol', tree.fname)
            tree.symbol = f'{base_symbol}_{tree.semantic_fingerprint[:12]}_u{unit_id:x}'
    return unit_id


def specialization_key(tree, arg_types):
    return ensure_compilation_unit(tree), tuple(arg_types)


def native_symbol(tree, arg_types):
    ensure_compilation_unit(tree)
    key = specialization_key(tree, arg_types)
    with cache_lock:
        generation = specialization_generations.get(key, 0)
    return mangler(f'{tree.symbol}_g{generation}', arg_types)


def visible_binding(tree, name):
    bindings = getattr(tree, 'bindings', {})
    if name in bindings:
        return bindings[name]
    return getattr(tree, 'namespace', {}).get(name)


def compile_specialization(tree, arg_types):
    key = specialization_key(tree, arg_types)
    owner = threading.get_ident()
    with cache_lock:
        while key in compilation_states:
            state = compilation_states[key]
            if state['owner'] == owner:
                raise InferError('mutual recursion is not supported by the current MCJIT backend', tree)
            state['condition'].wait()
            if key in function_cache:
                runtime_counters['compile_hits'] += 1
                return function_cache[key]
        if key in function_cache:
            runtime_counters['compile_hits'] += 1
            return function_cache[key]
        maximum = getattr(tree, 'max_specializations', DEFAULT_MAX_SPECIALIZATIONS)
        existing = sum(1 for cached_key in function_cache if cached_key[0] == key[0])
        if maximum is not None and existing >= maximum:
            raise SpecializationLimitError(
                f'{tree.fname} reached its specialization limit ({maximum})', tree)
        runtime_counters['compile_misses'] += 1
        condition = threading.Condition(cache_lock)
        compilation_states[key] = {'owner': owner, 'condition': condition}

    try:
        specialized = copy.deepcopy(tree)
        with cache_lock:
            generation = specialization_generations.get(key, 0)
        specialized.symbol = f'{tree.symbol}_g{generation}'
        symbol = mangler(specialized.symbol, arg_types)

        def resolve_jit(name, call_arg_types):
            candidate = visible_binding(tree, name)
            callee_tree = getattr(candidate, '__pyjiting_tree__', None)
            if callee_tree is None: return None, None
            compile_specialization(callee_tree, call_arg_types)
            callee_key = specialization_key(callee_tree, call_arg_types)
            with cache_lock:
                return function_signatures[callee_key], native_symbol(callee_tree, call_arg_types)

        def resolve_reg(name):
            candidate = visible_binding(tree, name)
            identifier = registration_id(candidate) if candidate is not None else None
            registered = get_registered(identifier if identifier is not None else name)
            if registered is None:
                return None, None
            return registered[1], identifier

        function_type = typeinfer(specialized, arg_types, jit_resolver=resolve_jit,
                                  reg_resolver=resolve_reg)
        module = ir.Module(name=f'pyjiting.{symbol}')
        module.triple = llvm.get_default_triple()
        llfunc = LLVMCodeGen(module, function_type.return_type, arg_types).visit(specialized)
        with engine_lock:
            binding_module = llvm.parse_assembly(str(module)); binding_module.verify()
            pto = llvm.create_pipeline_tuning_options(speed_level=3); pto.loop_vectorization = True
            pass_builder = llvm.create_pass_builder(target_machine, pto)
            pass_builder.getModulePassManager().run(binding_module, pass_builder)
            engine.add_module(binding_module); engine.finalize_object()
            wrapper = wrap_module(arg_types, llfunc, engine)
        with cache_lock:
            retained_modules.append(binding_module)
            function_signatures[key] = function_type
            function_cache[key] = wrapper
            state = compilation_states.pop(key)
            state['condition'].notify_all()
            debug(module)
            return wrapper
    except BaseException:
        with cache_lock:
            runtime_counters['compile_failures'] += 1
            specialization_generations[key] = specialization_generations.get(key, 0) + 1
            if key not in function_cache:
                function_signatures.pop(key, None)
            state = compilation_states.pop(key, None)
            if state is not None:
                state['condition'].notify_all()
        raise


def _wrapper_for_tree(tree, fn=None, fallback=False, max_specializations=DEFAULT_MAX_SPECIALIZATIONS):
    validate_specialization_limit(max_specializations)
    tree.max_specializations = max_specializations
    ensure_compilation_unit(tree)
    warned = False

    def warn_fallback(error):
        nonlocal warned
        if not warned:
            warnings.warn(f'pyjiting fallback for {tree.fname}: {error}',
                          RuntimeWarning, stacklevel=3)
            warned = True

    def wrapper(*args, **kwargs):
        if kwargs:
            if fallback and fn is not None:
                warn_fallback('keyword arguments are not supported by the JIT dispatcher')
                return fn(*args, **kwargs)
            raise TypeError('JIT calls do not support keyword arguments')
        if len(args) != len(tree.args):
            raise TypeError(f'{tree.fname}() takes {len(tree.args)} positional arguments but {len(args)} were given')
        try:
            compiled = compile_specialization(tree, [arg_pytype(arg) for arg in args])
        except CompileError as error:
            if not fallback or fn is None:
                raise
            warn_fallback(error)
            return fn(*args)
        return compiled(*args)

    if fn is not None:
        functools.update_wrapper(wrapper, fn)
    else:
        wrapper.__name__ = tree.fname
    wrapper.__pyjiting_tree__ = tree
    return wrapper


def jit(fn=None, *, fallback=False, max_specializations=DEFAULT_MAX_SPECIALIZATIONS) -> Any:
    validate_specialization_limit(max_specializations)
    if fn is None:
        return lambda decorated: jit(decorated, fallback=fallback,
                                     max_specializations=max_specializations)
    try:
        tree = ASTVisitor()(fn)
    except CompileError as error:
        if not fallback:
            raise
        fallback_error = error
        warned = False

        @functools.wraps(fn)
        def fallback_wrapper(*args, **kwargs):
            nonlocal warned
            if not warned:
                warnings.warn(f'pyjiting fallback for {fn.__qualname__}: {fallback_error}',
                              RuntimeWarning, stacklevel=2)
                warned = True
            return fn(*args, **kwargs)

        return fallback_wrapper
    identity = '\0'.join((fn.__module__, fn.__qualname__, fn.__code__.co_filename,
                          str(fn.__code__.co_firstlineno))).encode()
    tree.symbol = 'jit_' + hashlib.sha256(identity).hexdigest()[:16]
    tree.namespace = fn.__globals__
    return _wrapper_for_tree(tree, fn, fallback, max_specializations)


def jit_from_source(source, *, namespace=None, max_specializations=DEFAULT_MAX_SPECIALIZATIONS):
    validate_specialization_limit(max_specializations)
    tree = ASTVisitor()(source)
    tree.symbol = f'jit_source_{hashlib.sha256(source.encode()).hexdigest()[:16]}'
    tree.namespace = namespace if namespace is not None else {}
    return _wrapper_for_tree(tree, max_specializations=max_specializations)


setattr(jit, 'from_source', jit_from_source)


def runtime_stats():
    with compile_lock:
        units = {key[0] for key in function_cache}
        return {
            'specializations': len(function_cache),
            'compilation_units': len(units),
            'retained_modules': len(retained_modules),
            'registered_callbacks': callback_count(),
            'string_literals': literal_count(),
            **runtime_counters,
        }


def clear_cache(function=None):
    """Forget cached dispatch entries without claiming to free MCJIT code memory."""
    with compile_lock:
        if function is None:
            targets = list(function_cache)
        else:
            tree = getattr(function, '__pyjiting_tree__', None)
            if tree is None:
                raise TypeError('clear_cache expects a @jit function or None')
            unit_id = ensure_compilation_unit(tree)
            targets = [key for key in function_cache if key[0] == unit_id]
        for key in targets:
            function_cache.pop(key, None)
            function_signatures.pop(key, None)
            specialization_generations[key] = specialization_generations.get(key, 0) + 1
        return len(targets)


def codegen(module, ast, specializer=None, return_type=None, args=None):
    """Compatibility entry point used by external callers."""
    if args is None: raise InferError('codegen requires concrete argument types')
    function_type = typeinfer(ast, args)
    return LLVMCodeGen(module, return_type or function_type.return_type, args).visit(ast)
