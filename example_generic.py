# This test is used to test pyjiting support for dynamic types.
from pyjiting import jit


@jit
def test(a, b):
    return a + b


print(test(114, 514))
print(test(11.4, 51.4))
