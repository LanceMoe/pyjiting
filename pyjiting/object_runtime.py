"""Python objects retained by the current JIT dispatch."""

import builtins
import ctypes
import operator
from types import FunctionType
from typing import Any

from .errors import InferError
from .intrinsics import OBJECT_BUILTINS, OBJECT_CLASS_METHODS, OBJECT_CONSTRUCTORS
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


def _require_type(value, expected):
    if not isinstance(value, expected):
        raise TypeError(f'expected {expected.__name__}, got {type(value).__name__}')
    return value


def _as_string(value):
    if not isinstance(value, str):
        raise TypeError(f'expected str, got {type(value).__name__}')
    return make_string(value)


def _check_tuple_size(value, size):
    _require_type(value, tuple)
    if tuple.__len__(value) != size:
        raise TypeError(f'expected tuple of length {size}, got {tuple.__len__(value)}')


def _load_global(scope, name):
    namespace, closure_cells = scope
    if name in closure_cells:
        try:
            return closure_cells[name].cell_contents
        except ValueError:
            raise NameError(f'free variable {name!r} has no value', name=name) from None
    if name in namespace:
        return namespace[name]
    available = namespace.get('__builtins__', builtins)
    try:
        return available[name] if isinstance(available, dict) else getattr(available, name)
    except (KeyError, AttributeError):
        raise NameError(f"name {name!r} is not defined", name=name) from None


def _check_call_context(function, state):
    if isinstance(function, FunctionType):
        tree = getattr(function, '__pyjiting_tree__', None)
        if tree is not None and tree.runtime_state is not state:
            raise InferError('JIT functions from different runtime contexts cannot call each other')


def _format_convert(value, conversion):
    if conversion == ord('s'):
        return str(value)
    if conversion == ord('r'):
        return repr(value)
    if conversion == ord('a'):
        return ascii(value)
    return value


def _format_part(value, spec, conversion):
    return make_string(format(_format_convert(value, conversion), to_python(spec)))


def _join_strings(*parts):
    return make_string(''.join(to_python(part) for part in parts))


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


def _dict_unpack(target, mapping):
    if type(mapping) is dict:
        target.update(mapping)
    else:
        target.update({**mapping})


def _merge_keywords(function, target, mapping):
    try:
        keys = mapping.keys
    except AttributeError:
        name = getattr(function, '__qualname__', type(function).__name__)
        raise TypeError(
            f'{name}() argument after ** must be a mapping, not {type(mapping).__name__}'
        ) from None
    for key in list(keys()):
        _check_keyword(function, target, key)
        target[key] = mapping[key]


def _check_keyword(function, target, key):
    if key in target:
        name = getattr(function, '__qualname__', type(function).__name__)
        raise TypeError(f'{name}() got multiple values for keyword argument {key!r}')


_primitives = {
    'add#': operator.add, 'sub#': operator.sub, 'mult#': operator.mul,
    'div#': operator.truediv, 'floordiv#': operator.floordiv, 'mod#': operator.mod,
    'pow#': operator.pow, 'neg#': operator.neg,
    'bitor#': operator.or_, 'bitand#': operator.and_, 'bitxor#': operator.xor,
    'lshift#': operator.lshift, 'rshift#': operator.rshift,
    'pos#': operator.pos, 'invert#': operator.invert,
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
    'bitor#': operator.ior, 'bitand#': operator.iand, 'bitxor#': operator.ixor,
    'lshift#': operator.ilshift, 'rshift#': operator.irshift,
}
_operations = {
    **OBJECT_CONSTRUCTORS, **OBJECT_CLASS_METHODS, **OBJECT_BUILTINS,
    **_primitives,
    **{f'inplace:{name}': function for name, function in _inplace_primitives.items()},
    'none': lambda: None,
    'getitem': operator.getitem, 'setitem': operator.setitem, 'delitem': operator.delitem,
    'list': list, 'len': len, 'bool': bool,
    'append': list.append, 'extend': list.extend, 'unpack': _unpack,
    'set': set, 'frozenset': frozenset, 'set_add': set.add, 'set_update': set.update,
    'dict': dict, 'dict.fromkeys': dict.fromkeys, 'dict_unpack': _dict_unpack,
    'pair': lambda key, value: (key, value),
    'getattr': getattr, 'merge_keywords': _merge_keywords,
    'call': lambda function, args, kwargs: function(*args, **kwargs),
    'int': int, 'float': float, 'str': str,
    'sum': sum, 'min': min, 'max': max, 'any': any, 'all': all, 'abs': abs,
    'chr': chr, 'ord': ord, 'iter': iter, 'iter_next': _next, 'next': next, 'hash': hash,
    'enumerate': enumerate, 'zip': zip, 'reversed': reversed, 'sorted': sorted,
    'box_int': int, 'box_float': float, 'box_bool': bool, 'box_str': to_python,
    'unbox_int': _as_int, 'unbox_float': float, 'unbox_str': _as_string,
    'require_type': _require_type,
    'check_tuple_size': _check_tuple_size, 'tuple_item': tuple.__getitem__,
    'load_global': _load_global, 'check_call_context': _check_call_context,
    'format_convert': _format_convert, 'format_part': _format_part, 'join_strings': _join_strings,
}


def callback_address(name, output_type, arg_types, keywords=()):
    key = (name, output_type, tuple(arg_types), keywords)
    callback = _callbacks.get(key)
    if callback is None:
        if name.startswith('builtin:'):
            function = _operations[name.removeprefix('builtin:')]

            def builtin_operation(*args):
                return function

            operation = builtin_operation
        elif name == 'invoke' or name.startswith('method:') or keywords:
            method = name.removeprefix('method:')

            def operation(*args):
                positional_count = len(args) - len(keywords)
                positional = args[:positional_count]
                kwargs = dict(zip(keywords, args[positional_count:]))
                if name == 'invoke':
                    function, *positional = positional
                elif name.startswith('method:'):
                    function = getattr(positional[0], method)
                    positional = positional[1:]
                else:
                    function = _operations[name]
                return function(*positional, **kwargs)
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
                if name == 'iter_next':
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
