import numpy as np
import pytest

from pyjiting import jit


@jit
def read_2d(values, row, col):
    return values[row, col]


@jit
def write_2d(values, row, col, value):
    values[row, col] = value
    return values[row, col]


@jit
def shape_arithmetic(values):
    return values.shape[0] * 10 + values.shape[1]


@jit
def negative_power_store(values, index):
    values[index] **= -2
    return values[index]


@pytest.mark.parametrize(
    'values',
    [
        np.arange(12, dtype=np.int64).reshape(3, 4),
        np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4)),
        np.arange(12, dtype=np.int64).reshape(3, 4).T,
        np.arange(24, dtype=np.int64).reshape(4, 6)[::-1, 1::2],
    ],
)
def test_multidimensional_reads_follow_numpy_strides(values):
    row, col = values.shape[0] - 1, values.shape[1] - 1
    assert read_2d(values, row, col) == values[row, col]
    assert shape_arithmetic(values) == values.shape[0] * 10 + values.shape[1]


def test_multidimensional_write_updates_a_transposed_view():
    base = np.zeros((2, 3), dtype=np.float32)
    view = base.T

    assert write_2d(view, 1, 0, np.float32(6.25)) == pytest.approx(6.25)
    assert base[0, 1] == pytest.approx(6.25)


def test_float_array_augmented_negative_power_uses_float_result():
    values = np.array([2.0], dtype=np.float64)
    assert negative_power_store(values, 0) == pytest.approx(0.25)


def test_readonly_array_store_raises_without_mutation():
    values = np.zeros((2, 3), dtype=np.float32)
    values.flags.writeable = False

    with pytest.raises(ValueError, match='assignment destination is read-only'):
        write_2d(values, 0, 0, np.float32(7.0))

    np.testing.assert_array_equal(values, np.zeros((2, 3), dtype=np.float32))


def test_byte_strides_and_unaligned_views_match_numpy():
    base = np.arange(4, dtype=np.int64)
    byte_strided = np.lib.stride_tricks.as_strided(base, shape=(2, 2), strides=(8, 1))
    assert read_2d(byte_strided, 0, 1) == byte_strided[0, 1]

    buffer = bytearray(17)
    unaligned = np.ndarray((2,), dtype=np.int64, buffer=buffer, offset=1)
    unaligned[:] = [123, 456]
    assert not unaligned.flags.aligned
    assert read_2d(unaligned.reshape(1, 2), 0, 1) == 456
