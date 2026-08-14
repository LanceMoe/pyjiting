from pyjiting import jit


@jit
def add_two(value: int) -> int:
    return value + 2


print(add_two(40))
