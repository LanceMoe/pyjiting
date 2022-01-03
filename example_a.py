from pyjiting import jit


@jit
def add(x):
    res = 0
    for i in range(x):
        res = res + i
    return res

print(add(100))


@jit
def test(a, b):
    return a + b

print(test(114, 514))
print(test(11.4, 51.4))