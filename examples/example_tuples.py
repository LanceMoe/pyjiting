import numpy as np

from pyjiting import jit


@jit
def statistics(total: float, count: int) -> tuple[float, int, str]:
    label = 'empty'
    if count:
        label = 'ready'
    return total / max(count, 1), count, label


@jit
def format_statistics(values):
    average, count, label = statistics(sum(values), len(values))
    return (label, average), count


print(format_statistics(np.array([1.0, 2.0, 6.0], dtype=np.float64)))
