from pyjiting import jit, reg


all_primes = []

@reg
def do_something(x: int) -> int:
    print(f'{x} is prime!')
    all_primes.append(x)
    return 0

@jit
def find_primes(n):
    for i in range(2, n):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime == True:
            do_something(i)
    return 0


find_primes(100000)
print(all_primes)
