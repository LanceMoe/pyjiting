# This is a pure loop performance test
from pyjiting import jit
import time


@jit
def loop_jit(n):
    for _ in range(n):
        n += 1
    return n


def loop_nojit(n):
    for _ in range(n):
        n += 1
    return n


loop_jit(0)
loop_nojit(0)


start_time = time.time()
result = loop_jit(100000000)
cost_time_ms = (time.time() - start_time) * 1000
print('loop_jit(100000000) =', result, f'(cost time: {cost_time_ms} ms)')

start_time = time.time()
result = loop_nojit(100000000)
cost_time_ms = (time.time() - start_time) * 1000
print('loop_nojit(100000000) =', result, f'(cost time: {cost_time_ms} ms)')
