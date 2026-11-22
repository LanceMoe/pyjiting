FUNCTION_INTRINSICS = frozenset({
    'len', 'abs', 'min', 'max', 'ord', 'chr', 'sum', 'any', 'all', 'list', 'set', 'frozenset',
    'tuple', 'iter', 'next', 'hash',
    'sorted', 'range', 'slice', 'enumerate', 'zip', 'reversed', 'int', 'float', 'str', 'bool',
})
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
STRING_TRANSFORMS = frozenset(f'str.{name}' for name in ('upper', 'lower', 'strip', 'lstrip', 'rstrip'))
STRING_PREDICATES = frozenset(f'str.{name}' for name in ('isalpha', 'isalnum', 'isdigit', 'isspace'))


def is_intrinsic(name):
    return name in FUNCTION_INTRINSICS or name in STRING_INTRINSICS or name in MATH_INTRINSICS
