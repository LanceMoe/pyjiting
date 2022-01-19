# Test while
from pyjiting import jit

import time


def test_while(x: int) -> int:
    res = 0
    while res < x:
        res = res + 1
    return res


test_while_jit = jit(test_while)

test_while_jit(0)
test_while(0)

start_time = time.time()
result = test_while_jit(100000000)
cost_time_ms = (time.time() - start_time) * 1000
print('test_while_jit(100000000) =', result,
      f'(cost time: {cost_time_ms} ms)')

start_time = time.time()
result = test_while(100000000)
cost_time_ms_nojit = (time.time() - start_time) * 1000
print('test_while_nojit(100000000) =', result,
      f'(cost time: {cost_time_ms_nojit} ms)')
print('rate:', 'Infinite' if cost_time_ms == 0 else cost_time_ms_nojit / cost_time_ms)
