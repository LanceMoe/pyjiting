import builtins


OBJECT_CONSTRUCTORS = {
    'tuple': tuple, 'complex': complex, 'bytes': bytes, 'bytearray': bytearray,
    'memoryview': memoryview, 'range': range, 'slice': slice,
}
OBJECT_CLASS_METHODS = {
    'bytes.fromhex': bytes.fromhex, 'bytearray.fromhex': bytearray.fromhex,
    'int.from_bytes': int.from_bytes,
}
OBJECT_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        'ascii', 'bin', 'callable', 'compile', 'delattr', 'divmod',
        'filter', 'format', 'getattr', 'hasattr', 'hex', 'id', 'input', 'isinstance',
        'issubclass', 'map', 'oct', 'open', 'pow', 'print', 'repr', 'round',
        'setattr', 'type', 'aiter', 'anext', 'property', 'classmethod', 'staticmethod',
        '__import__',
    )
}
# These builtins can inspect their caller's Python frame.
FRAME_BUILTINS = frozenset({'dir', 'eval', 'exec', 'globals', 'locals', 'super', 'vars', 'breakpoint'})
FUNCTION_INTRINSICS = frozenset({
    'len', 'abs', 'min', 'max', 'ord', 'chr', 'sum', 'any', 'all', 'list', 'set', 'frozenset', 'dict', 'dict.fromkeys',
    'tuple', 'iter', 'next', 'hash',
    'sorted', 'range', 'slice', 'enumerate', 'zip', 'reversed', 'int', 'float', 'str', 'bool',
}) | OBJECT_CONSTRUCTORS.keys() | OBJECT_CLASS_METHODS.keys() | OBJECT_BUILTINS.keys()
MATH_FUNCTIONS = frozenset({
    'sin', 'cos', 'sqrt', 'exp', 'log', 'log2', 'log10',
    'isnan', 'isinf', 'isfinite',
})
MATH_INTRINSICS = frozenset(f'math.{name}' for name in MATH_FUNCTIONS)
MATH_CONSTANTS = frozenset({'pi', 'e', 'tau', 'inf', 'nan'})
STRING_METHODS = frozenset({
    'startswith', 'endswith', 'find', 'count', 'upper', 'lower', 'strip',
    'lstrip', 'rstrip', 'replace', 'isalpha', 'isalnum', 'isdigit', 'isspace',
})
STRING_INTRINSICS = frozenset(f'str.{name}' for name in STRING_METHODS)
OBJECT_METHODS = frozenset({
    name for python_type in (str, bytes, bytearray, memoryview, tuple, range, slice, complex, int, float, bool)
    for name in dir(python_type)
    if not name.startswith('_') and callable(getattr(python_type, name))
})
LIST_METHODS = frozenset({
    'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert',
    'pop', 'remove', 'reverse', 'sort',
})
LIST_RETURNING_METHODS = frozenset({'copy'})
LIST_INTEGER_METHODS = frozenset({'count', 'index'})
SET_RETURNING_METHODS = frozenset({'copy', 'union', 'intersection', 'difference', 'symmetric_difference'})
SET_METHODS = SET_RETURNING_METHODS | frozenset({
    'isdisjoint', 'issubset', 'issuperset', 'add', 'clear', 'discard', 'pop', 'remove',
    'update', 'intersection_update', 'difference_update', 'symmetric_difference_update',
})
DICT_METHODS = frozenset({
    'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values',
})
STRING_TRANSFORMS = frozenset(f'str.{name}' for name in ('upper', 'lower', 'strip', 'lstrip', 'rstrip'))
STRING_PREDICATES = frozenset(f'str.{name}' for name in ('isalpha', 'isalnum', 'isdigit', 'isspace'))


def is_intrinsic(name):
    return name in FUNCTION_INTRINSICS or name in STRING_INTRINSICS or name in MATH_INTRINSICS
