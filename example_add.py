from pyjiting import jit


@jit
def add(a: float, b: float) -> float:
    return a + b


a = 3.1415926
b = 2.7182818
result = add(a, b)
print(result, a + b)
