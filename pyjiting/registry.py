import ctypes
import inspect
import itertools
import typing

from .types import FuncType, bool_t, double64_t, float32_t, int32_t, int64_t, str_t, void_t


_registered = {}
_registered_names = {}
_callbacks = {}
_callback_invocations = {}
_registration_ids = itertools.count(1)


def annotation_type(annotation):
    mapping = {int: int64_t, float: double64_t, bool: bool_t, str: str_t, type(None): void_t}
    try:
        import numpy as np
        mapping.update({np.int32: int32_t, np.int64: int64_t, np.float32: float32_t, np.float64: double64_t})
    except ImportError:
        pass
    return mapping.get(annotation)


def register(fn):
    hints, signature = typing.get_type_hints(fn), inspect.signature(fn)
    args = []
    for parameter in signature.parameters.values():
        ty = annotation_type(hints.get(parameter.name))
        if ty is None: raise TypeError(f'@reg function {fn.__name__} needs a supported annotation for {parameter.name}')
        if ty == void_t: raise TypeError(f'@reg function {fn.__name__} cannot use None for {parameter.name}')
        args.append(ty)
    return_ty = annotation_type(hints.get('return'))
    if return_ty is None: raise TypeError(f'@reg function {fn.__name__} needs a supported return annotation')
    registration_id = next(_registration_ids)
    registered = (fn, FuncType(args=args, return_type=return_ty))
    _registered[registration_id] = registered
    _registered_names.setdefault(fn.__name__, []).append(registration_id)
    fn.__pyjiting_registered_id__ = registration_id
    return fn


def unregister(fn):
    identifier = registration_id(fn)
    if identifier is None:
        raise TypeError('unregister expects an @reg function')
    registered = _registered.pop(identifier, None)
    if registered is None:
        return False
    name = registered[0].__name__
    ids = _registered_names.get(name, [])
    if identifier in ids:
        ids.remove(identifier)
    if not ids:
        _registered_names.pop(name, None)
    _callbacks.pop(identifier, None)
    _callback_invocations.pop(identifier, None)
    try:
        del fn.__pyjiting_registered_id__
    except AttributeError:
        pass
    return True


def signatures():
    """Return legacy unambiguous short-name signatures for source-only callers."""
    return {
        name: _registered[ids[0]][1]
        for name, ids in _registered_names.items()
        if len(ids) == 1
    }


def get(identifier):
    if isinstance(identifier, int):
        return _registered.get(identifier)
    ids = _registered_names.get(identifier, ())
    return _registered.get(ids[0]) if len(ids) == 1 else None


def registration_id(fn):
    return getattr(fn, '__pyjiting_registered_id__', None)


def keep_callback(identifier, callback):
    retained = _callbacks.setdefault(identifier, callback)
    return ctypes.cast(retained, ctypes.c_void_p).value


def callback_count():
    return len(_callbacks)


def record_callback(identifier):
    _callback_invocations[identifier] = _callback_invocations.get(identifier, 0) + 1


def callback_stats():
    return {
        'total': sum(_callback_invocations.values()),
        'by_registration': dict(_callback_invocations),
    }
