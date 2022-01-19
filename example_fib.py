# Fibonacci is mainly used to show the performance of recursion.
from pyjiting import jit
import time


@jit
def fib_jit(x):
    if x < 3:
        return 1
    return fib_jit(x-1) + fib_jit(x-2)


def fib_nojit(x):
    if x < 3:
        return 1
    return fib_nojit(x-1) + fib_nojit(x-2)


fib_jit(0)
fib_nojit(0)


start_time = time.time()
result = fib_jit(40)
cost_time_ms = (time.time() - start_time) * 1000
print('fib_jit(40) =', result, f'(cost time: {cost_time_ms} ms)')

start_time = time.time()
result = fib_nojit(40)
cost_time_ms_nojit = (time.time() - start_time) * 1000
print('fib_nojit(40) =', result, f'(cost time: {cost_time_ms_nojit} ms)')
print('rate:', 'Infinite' if cost_time_ms == 0 else cost_time_ms_nojit / cost_time_ms)
