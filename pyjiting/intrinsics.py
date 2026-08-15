FUNCTION_INTRINSICS = frozenset({'len', 'abs', 'min', 'max', 'ord', 'chr'})
STRING_METHODS = frozenset({
    'startswith', 'endswith', 'find', 'count', 'upper', 'lower', 'strip',
    'lstrip', 'rstrip', 'replace', 'isalpha', 'isalnum', 'isdigit', 'isspace',
})
STRING_INTRINSICS = frozenset(f'str.{name}' for name in STRING_METHODS)
STRING_TRANSFORMS = frozenset(f'str.{name}' for name in ('upper', 'lower', 'strip', 'lstrip', 'rstrip'))
STRING_PREDICATES = frozenset(f'str.{name}' for name in ('isalpha', 'isalnum', 'isdigit', 'isspace'))


def is_intrinsic(name):
    return name in FUNCTION_INTRINSICS or name in STRING_INTRINSICS
