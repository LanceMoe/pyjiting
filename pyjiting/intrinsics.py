FUNCTION_INTRINSICS = frozenset({'len', 'abs', 'min', 'max'})
STRING_METHODS = frozenset({'startswith', 'endswith', 'find', 'count'})
STRING_INTRINSICS = frozenset(f'str.{name}' for name in STRING_METHODS)


def is_intrinsic(name):
    return name in FUNCTION_INTRINSICS or name in STRING_INTRINSICS
