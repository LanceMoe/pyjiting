# pyright: reportArgumentType=false

import ast as py_ast
import copy
import functools
import hashlib
import inspect
import itertools
import sys
import threading
from time import perf_counter_ns, time_ns
from types import MappingProxyType
from typing import Any, Callable, ParamSpec, TypeVar, overload
import warnings

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from .codegen import LLVMCodeGen
from .errors import (CodegenError, CompileError, FallbackWarning, InferError, RuntimeClosedError,
                     RuntimeResourceError, SpecializationLimitError)
from .infer import TypeInferencer
from .ll_types import mangler, wrap_module
from .parser import ASTVisitor
from .registry import (callback_count, callback_stats as registered_callback_stats,
                       get as get_registered, register, unregister,
                       registration_id, signatures)
from .string_runtime import callback_stats, literal_count
from .types import (TupleType, bool_t, contains_array, double64_t, float32_t,
                    int32_t, int64_t, make_array_type, str_t)


DEFAULT_MAX_SPECIALIZATIONS = 64
DEBUG = False
P = ParamSpec('P')
R = TypeVar('R')
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


class RuntimeState:
    def __init__(self, *, max_specializations=None, max_modules=None):
        if (max_specializations is not None and
                (not isinstance(max_specializations, int) or max_specializations < 1)):
            raise ValueError('max_specializations must be a positive integer or None')
        if max_modules is not None and (not isinstance(max_modules, int) or max_modules < 1):
            raise ValueError('max_modules must be a positive integer or None')
        self.cache_lock = threading.RLock()
        self.compile_lock = self.cache_lock
        self.engine_lock = threading.RLock()
        self.compilation_unit_ids = itertools.count(1)
        self.function_cache = {}
        self.function_signatures = {}
        self.compilation_states = {}
        self.specialization_generations = {}
        self.retained_modules = []
        self.runtime_counters = {
            'compile_hits': 0, 'compile_misses': 0, 'compile_failures': 0,
            'failure_cache_hits': 0, 'compile_waits': 0,
        }
        self.specialization_metrics = {}
        self.specialization_ir = {}
        self.failure_cache = {}
        self.failure_details = {}
        self.max_specializations = max_specializations
        self.max_modules = max_modules
        self.closed = False
        self.registered_functions = []
        self.target_machine = llvm.Target.from_default_triple().create_target_machine()
        self.engine = llvm.create_mcjit_compiler(llvm.parse_assembly(''), self.target_machine)

    def ensure_open(self):
        if self.closed:
            raise RuntimeClosedError('JIT runtime is closed')


default_runtime = RuntimeState()
function_cache = default_runtime.function_cache
function_signatures = default_runtime.function_signatures
cache_lock = default_runtime.cache_lock
compile_lock = cache_lock
engine_lock = default_runtime.engine_lock
compilation_unit_ids = default_runtime.compilation_unit_ids
compilation_states = default_runtime.compilation_states
specialization_generations = default_runtime.specialization_generations
retained_modules = default_runtime.retained_modules
runtime_counters = default_runtime.runtime_counters
specialization_metrics = default_runtime.specialization_metrics
specialization_ir = default_runtime.specialization_ir
target_machine = default_runtime.target_machine
engine = default_runtime.engine


def debug(*values):
    if DEBUG: print(*values)


def validate_specialization_limit(value):
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError('max_specializations must be a positive integer or None')


def reg(fn: Callable[P, R]) -> Callable[P, R]:
    return register(fn)


def arg_pytype(arg):
    if isinstance(arg, tuple):
        elements = [arg_pytype(element) for element in arg]
        if any(contains_array(element) for element in elements):
            raise TypeError('ndarray elements inside tuples are not supported')
        return TupleType(elements)
    if isinstance(arg, np.ndarray):
        dtype_map = {np.dtype(np.int32): int32_t, np.dtype(np.int64): int64_t,
                     np.dtype(np.float32): float32_t,
                     np.dtype(np.float64): double64_t}
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


def _fingerprint_value(value):
    """Serialize Core AST fields without depending on CPython's ``ast.dump``."""
    if isinstance(value, py_ast.AST):
        fields = getattr(value, '_fields', ())
        serialized_fields = ','.join(
            f'{name}={_fingerprint_value(getattr(value, name, None))}'
            for name in fields)
        return f'{type(value).__module__}.{type(value).__qualname__}({serialized_fields})'
    if isinstance(value, list):
        return '[' + ','.join(_fingerprint_value(item) for item in value) + ']'
    if isinstance(value, tuple):
        return '(' + ','.join(_fingerprint_value(item) for item in value) + ')'
    return f'{type(value).__module__}.{type(value).__qualname__}:{value!r}'


def ensure_compilation_unit(tree):
    """Assign a process-unique identity to one decorated/parsed function tree."""
    unit_id = getattr(tree, 'compilation_unit_id', None)
    if unit_id is not None:
        return unit_id
    state = getattr(tree, 'runtime_state', default_runtime)
    state.ensure_open()
    with state.compile_lock:
        unit_id = getattr(tree, 'compilation_unit_id', None)
        if unit_id is None:
            unit_id = next(state.compilation_unit_ids)
            tree.compilation_unit_id = unit_id
            fingerprint_source = _fingerprint_value(tree)
            tree.semantic_fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
            base_symbol = getattr(tree, 'symbol', tree.fname)
            tree.symbol = f'{base_symbol}_{tree.semantic_fingerprint[:12]}_u{unit_id:x}'
    return unit_id


def specialization_key(tree, arg_types):
    return ensure_compilation_unit(tree), tuple(arg_types)


def native_symbol(tree, arg_types):
    ensure_compilation_unit(tree)
    key = specialization_key(tree, arg_types)
    state = getattr(tree, 'runtime_state', default_runtime)
    with state.cache_lock:
        generation = state.specialization_generations.get(key, 0)
    return mangler(f'{tree.symbol}_g{generation}', arg_types)


def visible_binding(tree, name):
    bindings = getattr(tree, 'bindings', {})
    if name in bindings:
        return bindings[name]
    return getattr(tree, 'namespace', {}).get(name)


def compile_specialization(tree, arg_types):
    state = getattr(tree, 'runtime_state', default_runtime)
    state.ensure_open()
    key = specialization_key(tree, arg_types)
    owner = threading.get_ident()
    with state.cache_lock:
        while key in state.compilation_states:
            compilation = state.compilation_states[key]
            if compilation['owner'] == owner:
                raise InferError('mutual recursion is not supported by the current MCJIT backend', tree)
            state.runtime_counters['compile_waits'] += 1
            compilation['condition'].wait()
            if key in state.function_cache:
                state.runtime_counters['compile_hits'] += 1
                return state.function_cache[key]
        if key in state.function_cache:
            state.runtime_counters['compile_hits'] += 1
            return state.function_cache[key]
        if key in state.failure_cache:
            state.runtime_counters['failure_cache_hits'] += 1
            error_type, error_args = state.failure_cache[key]
            raise error_type(*error_args)
        maximum = getattr(tree, 'max_specializations', DEFAULT_MAX_SPECIALIZATIONS)
        existing = sum(1 for cached_key in state.function_cache if cached_key[0] == key[0])
        pending = sum(1 for pending_key in state.compilation_states if pending_key[0] == key[0])
        if maximum is not None and existing + pending >= maximum:
            raise SpecializationLimitError(
                f'{tree.fname} reached its specialization limit ({maximum})', tree)
        reserved = len(state.function_cache) + len(state.compilation_states)
        if state.max_specializations is not None and reserved >= state.max_specializations:
            raise RuntimeResourceError(
                f'JIT runtime reached its specialization limit ({state.max_specializations})')
        reserved_modules = len(state.retained_modules) + len(state.compilation_states)
        if state.max_modules is not None and reserved_modules >= state.max_modules:
            raise RuntimeResourceError(f'JIT runtime reached its module limit ({state.max_modules})')
        state.runtime_counters['compile_misses'] += 1
        condition = threading.Condition(state.cache_lock)
        state.compilation_states[key] = {'owner': owner, 'condition': condition}

    started_ns = perf_counter_ns()
    try:
        specialized = copy.deepcopy(tree)
        with state.cache_lock:
            generation = state.specialization_generations.get(key, 0)
        specialized.symbol = f'{tree.symbol}_g{generation}'
        symbol = mangler(specialized.symbol, arg_types)

        def resolve_jit(name, call_arg_types):
            candidate = visible_binding(tree, name)
            callee_tree = getattr(candidate, '__pyjiting_tree__', None)
            if callee_tree is None: return None, None
            if getattr(callee_tree, 'runtime_state', default_runtime) is not state:
                raise InferError('JIT functions from different runtime contexts cannot call each other', tree)
            compile_specialization(callee_tree, call_arg_types)
            callee_key = specialization_key(callee_tree, call_arg_types)
            with state.cache_lock:
                return state.function_signatures[callee_key], native_symbol(callee_tree, call_arg_types)

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
        with state.engine_lock:
            binding_module = llvm.parse_assembly(str(module)); binding_module.verify()
            pto = llvm.create_pipeline_tuning_options(speed_level=3); pto.loop_vectorization = True
            pass_builder = llvm.create_pass_builder(state.target_machine, pto)
            unoptimized_ir = str(module)
            pass_builder.getModulePassManager().run(binding_module, pass_builder)
            optimized_ir = str(binding_module)
            state.engine.add_module(binding_module); state.engine.finalize_object()
            wrapper = wrap_module(arg_types, llfunc, state.engine)
        with state.cache_lock:
            state.retained_modules.append(binding_module)
            state.function_signatures[key] = function_type
            state.function_cache[key] = wrapper
            state.specialization_ir[key] = (unoptimized_ir, optimized_ir)
            state.specialization_metrics[key] = {
                'argument_types': tuple(map(str, arg_types)),
                'return_type': str(function_type.return_type),
                'native_symbol': symbol,
                'compile_time_ns': perf_counter_ns() - started_ns,
                'compile_count': 1,
                'generation': generation,
                'calls': 0,
            }
            compilation = state.compilation_states.pop(key)
            compilation['condition'].notify_all()
            debug(module)
            return wrapper
    except BaseException as error:
        with state.cache_lock:
            state.runtime_counters['compile_failures'] += 1
            state.specialization_generations[key] = state.specialization_generations.get(key, 0) + 1
            if key not in state.function_cache:
                state.function_signatures.pop(key, None)
            if isinstance(error, CompileError) and not isinstance(error, SpecializationLimitError):
                state.failure_cache[key] = (type(error), error.args)
                state.failure_details[key] = {
                    'function': tree.fname,
                    'argument_types': tuple(map(str, arg_types)),
                    'stage': ('codegen' if isinstance(error, CodegenError) else
                              'inference' if isinstance(error, InferError) else 'frontend'),
                    'error_type': type(error).__name__,
                    'message': str(error),
                    'timestamp_ns': time_ns(),
                }
            compilation = state.compilation_states.pop(key, None)
            if compilation is not None:
                compilation['condition'].notify_all()
        raise


def fallback_reason(error):
    if isinstance(error, CodegenError): return 'codegen'
    if isinstance(error, InferError): return 'inference'
    return 'frontend'


def validate_fallback_warning(value):
    if value not in ('once', 'always', 'ignore'):
        raise ValueError("fallback_warning must be 'once', 'always', or 'ignore'")


def emit_fallback_warning(policy, warned, function, error, *, stacklevel):
    if policy != 'ignore' and (policy == 'always' or not warned):
        warnings.warn(
            FallbackWarning(function, fallback_reason(error), error), stacklevel=stacklevel)
    return warned or policy != 'ignore'


def _wrapper_for_tree(tree, fn=None, fallback=False, max_specializations=DEFAULT_MAX_SPECIALIZATIONS,
                      fallback_warning='once'):
    validate_specialization_limit(max_specializations)
    validate_fallback_warning(fallback_warning)
    tree.max_specializations = max_specializations
    ensure_compilation_unit(tree)
    warned = False

    def warn_fallback(error):
        nonlocal warned
        warned = emit_fallback_warning(
            fallback_warning, warned, tree.fname, error, stacklevel=3)

    signature = inspect.signature(fn) if fn is not None else None
    if signature is not None:
        for parameter in signature.parameters.values():
            if parameter.default is not inspect.Parameter.empty:
                try:
                    arg_pytype(parameter.default)
                except TypeError as error:
                    raise CompileError(
                        f'unsupported default value for parameter {parameter.name!r}: {error}') from error

    def normalize_args(args, kwargs):
        if signature is None:
            if kwargs:
                raise TypeError('jit.from_source calls do not support keyword arguments')
            return args
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            if not kwargs:
                raise TypeError(
                    f'{tree.fname}() takes {len(tree.args)} positional arguments but {len(args)} were given') from None
            raise
        bound.apply_defaults()
        return tuple(bound.arguments[name] for name in signature.parameters)

    def wrapper(*args, **kwargs):
        state = getattr(tree, 'runtime_state', default_runtime)
        state.ensure_open()
        args = normalize_args(args, kwargs)
        if len(args) != len(tree.args):
            raise TypeError(f'{tree.fname}() takes {len(tree.args)} positional arguments but {len(args)} were given')
        try:
            compiled = compile_specialization(tree, [arg_pytype(arg) for arg in args])
        except CompileError as error:
            if isinstance(error, SpecializationLimitError):
                raise
            if not fallback or fn is None:
                raise
            warn_fallback(error)
            return fn(*args)
        result = compiled(*args)
        key = specialization_key(tree, [arg_pytype(arg) for arg in args])
        with state.cache_lock:
            if key in state.specialization_metrics:
                state.specialization_metrics[key]['calls'] += 1
        return result

    if fn is not None:
        functools.update_wrapper(wrapper, fn)
    else:
        wrapper.__name__ = tree.fname
    wrapper.__pyjiting_tree__ = tree

    def specialize(*args, **kwargs):
        normalized = normalize_args(args, kwargs)
        if len(normalized) != len(tree.args):
            raise TypeError(f'{tree.fname}() takes {len(tree.args)} positional arguments but {len(normalized)} were given')
        arg_types = [arg_pytype(arg) for arg in normalized]
        compile_specialization(tree, arg_types)
        key = specialization_key(tree, arg_types)
        state = getattr(tree, 'runtime_state', default_runtime)
        with state.cache_lock:
            return MappingProxyType(dict(state.specialization_metrics[key]))

    wrapper.specialize = specialize
    wrapper.warmup = specialize
    return wrapper


def _jit_with_state(state, fn: Any = None, *, fallback: bool = False,
                    max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS,
                    fallback_warning: str = 'once') -> Any:
    state.ensure_open()
    validate_specialization_limit(max_specializations)
    validate_fallback_warning(fallback_warning)
    if fn is None:
        return lambda decorated: _jit_with_state(
            state, decorated, fallback=fallback, max_specializations=max_specializations,
            fallback_warning=fallback_warning)
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
            warned = emit_fallback_warning(
                fallback_warning, warned, fn.__qualname__, fallback_error, stacklevel=2)
            return fn(*args, **kwargs)

        return fallback_wrapper
    identity = '\0'.join((fn.__module__, fn.__qualname__, fn.__code__.co_filename,
                          str(fn.__code__.co_firstlineno))).encode()
    tree.symbol = 'jit_' + hashlib.sha256(identity).hexdigest()[:16]
    tree.namespace = fn.__globals__
    tree.runtime_state = state
    return _wrapper_for_tree(tree, fn, fallback, max_specializations, fallback_warning)


@overload
def jit(fn: Callable[P, R], *, fallback: bool = False,
        max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS,
        fallback_warning: str = 'once') -> Callable[P, R]: ...


@overload
def jit(fn: None = None, *, fallback: bool = False,
        max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS, fallback_warning: str = 'once'
        ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def jit(fn: str, *, fallback: bool = False,
        max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS,
        fallback_warning: str = 'once') -> Any: ...


def jit(fn: Any = None, *, fallback: bool = False,
        max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS,
        fallback_warning: str = 'once') -> Any:
    return _jit_with_state(default_runtime, fn, fallback=fallback,
                           max_specializations=max_specializations, fallback_warning=fallback_warning)


def _jit_from_source_with_state(state, source, *, namespace=None,
                                max_specializations=DEFAULT_MAX_SPECIALIZATIONS):
    state.ensure_open()
    validate_specialization_limit(max_specializations)
    tree = ASTVisitor()(source)
    tree.symbol = f'jit_source_{hashlib.sha256(source.encode()).hexdigest()[:16]}'
    tree.namespace = namespace if namespace is not None else {}
    tree.runtime_state = state
    return _wrapper_for_tree(tree, max_specializations=max_specializations)


def jit_from_source(source, *, namespace=None, max_specializations=DEFAULT_MAX_SPECIALIZATIONS):
    return _jit_from_source_with_state(
        default_runtime, source, namespace=namespace, max_specializations=max_specializations)


setattr(jit, 'from_source', jit_from_source)


def runtime_stats(function=None):
    tree = getattr(function, '__pyjiting_tree__', None) if function is not None else None
    if function is not None and tree is None:
        raise TypeError('runtime_stats expects a @jit function or None')
    state = getattr(tree, 'runtime_state', default_runtime)
    with state.compile_lock:
        if function is None:
            keys = list(state.function_cache)
        else:
            unit_id = ensure_compilation_unit(tree)
            keys = [key for key in state.function_cache if key[0] == unit_id]
        units = {key[0] for key in keys}
        if function is not None:
            return {
                'specializations': len(keys),
                'compilation_units': len(units),
                'compile_time_ns': sum(state.specialization_metrics[key]['compile_time_ns'] for key in keys),
                'calls': sum(state.specialization_metrics[key]['calls'] for key in keys),
                'signatures': tuple(MappingProxyType(dict(state.specialization_metrics[key])) for key in keys),
                'failures': tuple(
                    MappingProxyType(dict(details))
                    for key, details in state.failure_details.items()
                    if key[0] == unit_id
                ),
            }
        return {
            'specializations': len(state.function_cache),
            'compilation_units': len(units),
            'retained_modules': len(state.retained_modules),
            'closed': state.closed,
            'registered_callbacks': callback_count(),
            'registered_callback_calls': registered_callback_stats(),
            'string_literals': literal_count(),
            'string_callbacks': callback_stats(),
            'recent_failures': tuple(
                MappingProxyType(dict(details))
                for details in state.failure_details.values()
            ),
            **state.runtime_counters,
        }


def clear_cache(function=None):
    """Forget cached dispatch entries without claiming to free MCJIT code memory."""
    tree = getattr(function, '__pyjiting_tree__', None) if function is not None else None
    if function is not None and tree is None:
        raise TypeError('clear_cache expects a @jit function or None')
    state = getattr(tree, 'runtime_state', default_runtime)
    state.ensure_open()
    with state.compile_lock:
        if function is None:
            targets = list(state.function_cache.keys() | state.failure_cache.keys())
        else:
            unit_id = ensure_compilation_unit(tree)
            targets = [
                key for key in state.function_cache.keys() | state.failure_cache.keys()
                if key[0] == unit_id
            ]
        for key in targets:
            state.function_cache.pop(key, None)
            state.function_signatures.pop(key, None)
            state.specialization_metrics.pop(key, None)
            state.specialization_ir.pop(key, None)
            state.failure_cache.pop(key, None)
            state.failure_details.pop(key, None)
            state.specialization_generations[key] = state.specialization_generations.get(key, 0) + 1
        return len(targets)


def inspect_specializations(function):
    return runtime_stats(function)['signatures']


def get_llvm_ir(function, *sample_args, optimized=False):
    tree = getattr(function, '__pyjiting_tree__', None)
    if tree is None:
        raise TypeError('get_llvm_ir expects a @jit function')
    arg_types = [arg_pytype(arg) for arg in sample_args]
    compile_specialization(tree, arg_types)
    key = specialization_key(tree, arg_types)
    state = getattr(tree, 'runtime_state', default_runtime)
    with state.cache_lock:
        return state.specialization_ir[key][1 if optimized else 0]


class JITContext:
    """Own an isolated MCJIT engine and its specialization resources."""

    def __init__(self, *, max_specializations=None, max_modules=None):
        self._state = RuntimeState(
            max_specializations=max_specializations, max_modules=max_modules)

    def jit(self, fn=None, *, fallback: bool = False,
            max_specializations: int | None = DEFAULT_MAX_SPECIALIZATIONS,
            fallback_warning: str = 'once'):
        return _jit_with_state(self._state, fn, fallback=fallback,
                               max_specializations=max_specializations,
                               fallback_warning=fallback_warning)

    def from_source(self, source, *, namespace=None,
                    max_specializations=DEFAULT_MAX_SPECIALIZATIONS):
        return _jit_from_source_with_state(
            self._state, source, namespace=namespace,
            max_specializations=max_specializations)

    def reg(self, fn):
        self._state.ensure_open()
        registered = register(fn)
        self._state.registered_functions.append(registered)
        return registered

    def stats(self):
        with self._state.cache_lock:
            return {
                'specializations': len(self._state.function_cache),
                'retained_modules': len(self._state.retained_modules),
                'closed': self._state.closed,
                **self._state.runtime_counters,
            }

    def close(self):
        with self._state.cache_lock:
            if self._state.compilation_states:
                raise RuntimeError('cannot close a JIT context while compilation is active')
            self._state.closed = True
            self._state.function_cache.clear()
            self._state.function_signatures.clear()
            self._state.specialization_metrics.clear()
            self._state.specialization_ir.clear()
            self._state.failure_cache.clear()
            self._state.failure_details.clear()
            for function in self._state.registered_functions:
                unregister(function)
            self._state.registered_functions.clear()
        return None

    def __enter__(self):
        self._state.ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def codegen(module, ast, specializer=None, return_type=None, args=None):
    """Compatibility entry point used by external callers."""
    if args is None: raise InferError('codegen requires concrete argument types')
    function_type = typeinfer(ast, args)
    return LLVMCodeGen(module, return_type or function_type.return_type, args).visit(ast)
