"""Caller-owned output arrays avoid ndarray allocation and ownership ambiguity."""

import numpy as np

from pyjiting import jit


@jit
def scale_into(values, output, factor):
    if len(values) != len(output):
        return -1
    for index in range(len(values)):
        output[index] = values[index] * factor
    return len(output)


values = np.arange(8, dtype=np.float64)
output = np.empty_like(values)
assert scale_into(values, output, 2.5) == len(values)
print(output)

# Exact aliasing is supported for element-wise kernels. Avoid partially
# overlapping views because sequential writes may change later reads.
scale_into(output, output, 0.5)
print(output)
