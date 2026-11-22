from pyjiting import jit


@jit
def read_sample(buffer: bytearray, positions: range, scale: complex | None = None):
    view = memoryview(buffer)
    for index in positions:
        if view[index] == 44:
            view[index] = 32
    real, imag = view.tobytes().decode().split()
    value = complex(float(real), float(imag))
    return value if scale is None else value * scale


data = bytearray(b'3,4')
result = read_sample(data, range(len(data)), 2 + 0j)
assert data == b'3 4' and result == 6 + 8j
print(result)
