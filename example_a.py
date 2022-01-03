from pyjiting import jit


@jit
def add(x):
    res = 0
    for i in range(x):
        res = res + i
    return res

print(add(100))
