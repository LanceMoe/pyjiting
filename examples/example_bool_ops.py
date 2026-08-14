from pyjiting import jit


@jit
def select(left, right):
    return (left and right) + (not left)


print(select(0, 7))
print(select(3, 7))
