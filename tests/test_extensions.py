from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pyjiting import jit, reg
from pyjiting.errors import InferError


@jit
def helper_add(value):
    return value + 2


@jit
def helper_string(value: str) -> str:
    return '[' + value + ']'


@jit
def helper_array(values, index):
    return values[index]


@jit
def composed_scalar(value):
    return helper_add(value) * 3


@jit
def composed_string(value: str) -> str:
    return helper_string(value) + helper_string('好')


@jit
def composed_array(values, index):
    return helper_array(values, index)


@reg
def registered_string(value: str) -> str:
    return value.upper() + '!'


@jit
def call_registered_string(value: str) -> str:
    return registered_string(value)


@jit
def array_iter_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


@jit
def matrix_item(values, row, column):
    return values[row, column]


@jit
def shape_item(values, dimension):
    return values.shape[dimension]


@jit
def builtin_values(left, right):
    return abs(left) + min(left, right) + max(left, right)


def test_jit_functions_call_other_jit_specializations():
    assert composed_scalar(4) == 18
    assert composed_scalar(1.5) == pytest.approx(10.5)
    assert composed_string('你') == '[你][好]'
    values = np.array([2, 5, 8], dtype=np.int64)
    assert composed_array(values, -1) == 8
    with pytest.raises(IndexError, match='index out of range'):
        composed_array(values, 3)


def test_registered_callbacks_accept_and_return_strings():
    assert call_registered_string('a你') == 'A你!'


def test_intrinsics_and_one_dimensional_array_iteration():
    assert builtin_values(-4, 7) == 7
    values = np.arange(8, dtype=np.float64)[::-2]
    assert array_iter_sum(values) == pytest.approx(sum(values))
    with pytest.raises(ValueError, match='index count'):
        array_iter_sum(np.zeros((2, 2), dtype=np.float64))


def test_array_indices_are_checked_and_negative_indices_are_normalized():
    values = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert composed_array(values.reshape(-1), -1) == 5
    with pytest.raises(IndexError, match='index out of range'):
        composed_array(values.reshape(-1), -7)
    assert matrix_item(values, -1, -2) == values[-1, -2]
    with pytest.raises(IndexError, match='index out of range'):
        matrix_item(values, 2, 0)
    assert shape_item(values, -1) == 3
    with pytest.raises(IndexError, match='index out of range'):
        shape_item(values, 2)


def test_concurrent_first_specializations_are_isolated():
    @jit
    def concurrent(value):
        return value + value

    values = [3, 2.5, 7, 1.25] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(concurrent, values))
    assert results == [value + value for value in values]


def test_loop_only_assignment_is_not_available_after_the_loop():
    @jit
    def invalid(limit):
        for value in range(limit):
            result = value
        return result

    with pytest.raises(InferError, match='unknown variable result'):
        invalid(1)
