import math

import pytest

from pyjiting import jit


@jit
def float_floor(left, right):
    return left // right


@jit
def float_mod(left, right):
    return left % right


@jit
def float_truth(value):
    return not value


@jit
def float_divide(left, right):
    return left / right


@jit
def integer_negative_power(value):
    return value ** -2


def test_float_floor_mod_and_nan_semantics():
    assert float_floor(-5.5, 2.0) == -3.0
    assert float_mod(-5.5, 2.0) == 0.5
    assert float_truth(float('nan')) == 0
    assert float_truth(0.0) == 1
    assert math.isnan(float_divide(1.0, float('nan')))


def test_float_division_by_zero_uses_runtime_error_channel():
    with pytest.raises(ZeroDivisionError, match='division by zero'):
        float_divide(1.0, 0.0)


def test_integer_negative_constant_power_returns_float():
    assert integer_negative_power(2) == 0.25
