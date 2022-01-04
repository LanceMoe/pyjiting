from pyjiting import jit

@jit
def fib(x):
    if x < 3:
        return 1
    return fib(x-1) + fib(x-2)

print(fib(40))