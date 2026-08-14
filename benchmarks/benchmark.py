"""Small benchmark separating compilation, cold-call, and warm-call costs."""

from time import perf_counter

from pyjiting import jit


@jit
def modular_sum(limit):
    total = 0
    for value in range(limit):
        total += value % 97
    return total


def modular_sum_python(limit):
    total = 0
    for value in range(limit):
        total += value % 97
    return total


def measure(label, function):
    started = perf_counter()
    result = function()
    elapsed_ms = (perf_counter() - started) * 1000
    print(f'{label}: {elapsed_ms:.3f} ms (result={result})')


if __name__ == '__main__':
    # The first call includes specialization, LLVM verification and JIT setup.
    measure('cold call / compile + execute', lambda: modular_sum(100_000))
    measure('warm call / cached specialization', lambda: modular_sum(100_000))
    measure('CPython reference', lambda: modular_sum_python(100_000))
