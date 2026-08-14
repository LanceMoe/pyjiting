from pyjiting import jit


@jit
def compute(value):
    return -value + 7 / 2 + 2.0 ** value


print(compute(3.0))
