import numpy as np
import pytest

from pyjiting import jit


@jit
def dynamic_step(step):
    total = 0
    for value in range(3, 0, step):
        total += value
    return total


@jit
def wrong_dimensions(values):
    return values[0]


def test_dynamic_range_zero_step_is_a_python_value_error():
    with pytest.raises(ValueError, match='arg 3 must not be zero'):
        dynamic_step(0)


def test_array_dimension_mismatch_is_a_python_value_error():
    with pytest.raises(ValueError, match='index count'):
        wrong_dimensions(np.zeros((2, 2), dtype=np.float64))
