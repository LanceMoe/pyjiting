import numpy as np
import pytest

from pyjiting import jit


@jit
def scale_into(values, output, factor):
    if len(values) != len(output):
        return -1
    for index in range(len(values)):
        output[index] = values[index] * factor
    return len(output)


def test_caller_owned_independent_output():
    values = np.arange(8, dtype=np.float64)
    output = np.empty_like(values)
    assert scale_into(values, output, 2.5) == 8
    np.testing.assert_allclose(output, values * 2.5)


def test_caller_owned_exact_in_place_output():
    values = np.arange(8, dtype=np.float64)
    expected = values * 3.0
    assert scale_into(values, values, 3.0) == 8
    np.testing.assert_allclose(values, expected)


def test_nonoverlapping_views_with_a_shared_base_are_supported():
    base = np.arange(16, dtype=np.float64)
    values, output = base[:8], base[8:]
    expected = values.copy() * 2.0
    assert scale_into(values, output, 2.0) == 8
    np.testing.assert_allclose(output, expected)


def test_output_shape_and_writeability_are_guarded():
    values = np.arange(8, dtype=np.float64)
    short = np.empty(7, dtype=np.float64)
    assert scale_into(values, short, 2.0) == -1

    readonly = np.empty_like(values)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        scale_into(values, readonly, 2.0)


def test_output_dtype_must_accept_the_kernel_result():
    values = np.arange(8, dtype=np.float64)
    output = np.empty(8, dtype=np.int64)
    with pytest.raises(Exception, match='cannot use'):
        scale_into(values, output, 2.5)
