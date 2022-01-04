# Find primes tests the performance of calling functions that are not been jited (Registered python code) in jited functions
from pyjiting import jit, reg
import time

all_primes = []


# jit
@reg
def do_something_jit(x: int) -> int:
    print(f'{x} is prime!')
    all_primes.append(x)
    return 0


@jit
def find_primes_jit(n):
    for i in range(2, n):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime == True:
            do_something_jit(i)
    return 0

# nojit
def do_something_nojit(x: int) -> int:
    print(f'{x} is prime!')
    all_primes.append(x)
    return 0


def find_primes_nojit(n):
    for i in range(2, n):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime == True:
            do_something_nojit(i)
    return 0


find_primes_nojit(0)
find_primes_jit(0)

start_time = time.time()
result = find_primes_jit(100000)
cost_time_ms = (time.time() - start_time) * 1000
jit_cost = cost_time_ms


start_time = time.time()
result = find_primes_nojit(100000)
cost_time_ms = (time.time() - start_time) * 1000
nojit_cost = cost_time_ms

print('find_primes_jit(100000) =', result, f'(cost time: {jit_cost} ms)')
print('find_primes_nojit(100000) =', result, f'(cost time: {nojit_cost} ms)')
