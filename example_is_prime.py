from pyjiting import jit
import time



@jit
def is_prime_jit(x: int) -> int:
    for i in range(2, x):
        if x % i == 0:
            return 0
    return 1


def is_prime_nojit(x: int) -> int:
    for i in range(2, x):
        if x % i == 0:
            return 0
    return 1

is_prime_jit(1)
is_prime_nojit(1)

start_time = time.time()
result = bool(is_prime_jit(169941229))
cost_time_ms = (time.time() - start_time) * 1000
print('is_prime_jit(169941229) =', result, f'(cost time: {cost_time_ms} ms)')

start_time = time.time()
result = bool(is_prime_nojit(169941229))
cost_time_ms = (time.time() - start_time) * 1000
print('is_prime_nojit(169941229) =', result, f'(cost time: {cost_time_ms} ms)')
