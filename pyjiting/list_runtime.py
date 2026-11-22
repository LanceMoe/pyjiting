"""Python objects retained by the current JIT dispatch."""

import ctypes
from functools import lru_cache
import operator
from typing import Any

from .string_runtime import ErrorPointer, StringPointer, _callback_failed, keep_alive, make_string, to_python


class PythonObject(ctypes.Structure):
    pass


ObjectPointer = ctypes.POINTER(PythonObject)
_callbacks = {}


def retain(value):
    keep_alive(value)
    return id(value)


def from_pointer(pointer):
    return ctypes.cast(pointer, ctypes.py_object).value


def _next(iterator):
    # None is a valid item. A null handle alone marks exhaustion.
    try:
        return retain(next(iterator))
    except StopIteration:
        return None


def _as_int(value):
    number = operator.index(value)
    if not -(1 << 63) <= number < (1 << 63):
        raise OverflowError('integer is outside the supported Int64 range')
    return number


def _as_list(value):
    if not isinstance(value, list):
        raise TypeError(f'expected list, got {type(value).__name__}')
    return value


def _as_string(value):
    if not isinstance(value, str):
        raise TypeError(f'expected str, got {type(value).__name__}')
    return make_string(value)


@lru_cache(maxsize=128)
def _key_code(source):
    return compile(source, '<jit sort key>', 'eval')


def _sort_key(spec, *values):
    source, names, namespace, bindings = spec
    scope = dict(namespace)
    scope.update(bindings)
    scope.update(zip(names, values))
    return eval(_key_code(source), scope)


def _unpack(value, size, star):
    if star >= 0:
        items = list(value)
        minimum = size - 1
        if len(items) < minimum:
            raise ValueError(f'not enough values to unpack (expected at least {minimum}, got {len(items)})')
        tail = len(items) - (minimum - star)
        return [*items[:star], items[star:tail], *items[tail:]]
    items = iter(value)
    result = []
    for index in range(size):
        try:
            result.append(next(items))
        except StopIteration:
            raise ValueError(f'not enough values to unpack (expected {size}, got {index})') from None
    try:
        next(items)
    except StopIteration:
        return result
    raise ValueError(f'too many values to unpack (expected {size})')


_primitives = {
    'add#': operator.add, 'sub#': operator.sub, 'mult#': operator.mul,
    'div#': operator.truediv, 'floordiv#': operator.floordiv, 'mod#': operator.mod,
    'pow#': operator.pow, 'neg#': operator.neg,
    'eq#': operator.eq, 'ne#': operator.ne, 'lt#': operator.lt,
    'le#': operator.le, 'gt#': operator.gt, 'ge#': operator.ge,
    'in#': lambda item, value: operator.contains(value, item),
    'notin#': lambda item, value: not operator.contains(value, item),
    'is#': operator.is_, 'isnot#': operator.is_not,
}
_inplace_primitives = {
    'add#': operator.iadd, 'sub#': operator.isub, 'mult#': operator.imul,
    'div#': operator.itruediv, 'floordiv#': operator.ifloordiv,
    'mod#': operator.imod, 'pow#': operator.ipow,
}
_operations = {
    **_primitives,
    **{f'inplace:{name}': function for name, function in _inplace_primitives.items()},
    'none': lambda: None,
    'getitem': operator.getitem, 'setitem': operator.setitem, 'delitem': operator.delitem,
    'slice': slice, 'list': list, 'tuple': tuple, 'len': len, 'bool': bool,
    'append': list.append, 'extend': list.extend, 'unpack': _unpack,
    'int': int, 'float': float, 'str': str,
    'sum': sum, 'min': min, 'max': max, 'any': any, 'all': all, 'abs': abs,
    'chr': chr, 'ord': ord, 'iter': iter, 'next': _next, 'range': range,
    'enumerate': enumerate, 'zip': zip, 'reversed': reversed,
    'box_int': int, 'box_float': float, 'box_bool': bool, 'box_str': to_python,
    'unbox_int': _as_int, 'unbox_float': float, 'unbox_str': _as_string, 'as_list': _as_list,
    'sort_key': _sort_key,
}


def callback_address(name, output_type, arg_types, keywords=()):
    key = (name, output_type, tuple(arg_types), keywords)
    callback = _callbacks.get(key)
    if callback is None:
        if name.startswith('method:') or name == 'sorted':
            method = name.removeprefix('method:')

            def operation(*args):
                positional_count = len(args) - len(keywords)
                positional = args[:positional_count]
                kwargs = dict(zip(keywords, args[positional_count:]))
                if name == 'sorted':
                    return sorted(*positional, **kwargs)
                return getattr(positional[0], method)(*positional[1:], **kwargs)
        else:
            operation = _operations[name]

        result_ctype = ctypes.c_void_p if output_type in (ObjectPointer, StringPointer) else output_type
        callback_type = ctypes.CFUNCTYPE(result_ctype, *arg_types, ErrorPointer)

        @callback_type
        def bridge(*args):
            error = args[-1]
            try:
                values = [from_pointer(value) if arg_type == ObjectPointer else value
                          for arg_type, value in zip(arg_types, args[:-1])]
                result: Any = operation(*values)
                if name == 'next':
                    return result
                if output_type == ObjectPointer:
                    return retain(result)
                if output_type == StringPointer:
                    return ctypes.cast(result, ctypes.c_void_p).value
                if output_type == ctypes.c_int64 and name in _primitives:
                    return int(bool(result))
                return result if output_type is not None else None
            except BaseException as exception:
                _callback_failed(error, exception)
                return None if result_ctype in (ctypes.c_void_p, None) else 0

        # Keep exactly the callback whose address callers will embed in LLVM IR.
        callback = _callbacks.setdefault(key, bridge)
    return ctypes.cast(callback, ctypes.c_void_p).value
