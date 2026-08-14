import ctypes
import inspect
import typing

from .types import FuncType, bool_t, double64_t, float32_t, int32_t, int64_t, void_t


_registered = {}
_callbacks = {}


def annotation_type(annotation):
    mapping = {int: int64_t, float: double64_t, bool: bool_t, type(None): void_t}
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
        args.append(ty)
    return_ty = annotation_type(hints.get('return'))
    if return_ty is None: raise TypeError(f'@reg function {fn.__name__} needs a supported return annotation')
    _registered[fn.__name__] = (fn, FuncType(args=args, return_type=return_ty))
    return fn


def signatures(): return {name: item[1] for name, item in _registered.items()}
def get(name): return _registered.get(name)


def keep_callback(name, callback):
    _callbacks[name] = callback
    return ctypes.cast(callback, ctypes.c_void_p).value
