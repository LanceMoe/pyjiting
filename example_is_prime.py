from pyjiting import autojit


@autojit
def is_prime(x: int) -> int:
    for i in range(2, x):
        if x % i == 0:
            return 0
    return 1


result = bool(is_prime(3571))
print(result)
