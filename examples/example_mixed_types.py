import numpy as np

from pyjiting import jit


@jit
def add_offset(values, offset):
    return values[0] + offset


print(add_offset(np.array([4], dtype=np.int32), 3))
