import numpy as np

from pyjiting import jit


@jit
def set_cell(values, row, column, value):
    values[row, column] = value
    return values[row, column] + values.shape[0]


matrix = np.zeros((2, 3), dtype=np.float32).T
print(set_cell(matrix, 1, 0, np.float32(4.5)))
print(matrix)
