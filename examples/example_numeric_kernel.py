import math

import numpy as np

from pyjiting import jit


BIAS = 0.25


@jit
def normalized_energy(values):
    total = sum(values) + BIAS
    if not all(values):
        return math.nan
    return math.sqrt(total) + math.log(total)


values = np.array([1.0, 2.0, 3.0], dtype=np.float32)[::-1]
print(normalized_energy(values))
