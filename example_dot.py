import numpy as np
from pyjiting import autojit


@autojit
def dot(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
        c += a[i] * b[i]
    return c


a = np.arange(100, 200, dtype="int64")
b = np.arange(300, 400, dtype="int64")
result = dot(a, b)
print(result)
