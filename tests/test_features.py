import math

import numpy as np
import pytest

from pyjiting import jit, reg
from pyjiting.errors import CompileError, InferError
from pyjiting.ll_types import mangler
from pyjiting.main import arg_pytype
from pyjiting.parser import ASTVisitor
from pyjiting.types import GenericType, int64_t


@jit
def arithmetic(a, b):
    return (a + b) * 2


@jit
def true_division(a, b):
    return a / b


@jit
def dynamic_range_step(step):
    total = 0
    for value in range(3, 0, step):
        total += value
    return total


@jit
def floor_and_mod(a, b):
    return a // b + a % b


@jit
def count_down(start):
    total = 0
    for value in range(start, 0, -1):
        if value == 3:
            continue
        total += value
    else:
        total += 100
    return total


@jit
def bool_ops(a, b):
    return (a and b) + (not a)


@jit
def chained(a, b, c):
    return a < b < c


@jit
def branch_assignment(value):
    if value > 0:
        result = 4
    else:
        result = 7
    return result


@jit
def multiple_assignment(value):
    left = right = value + 1
    return left + right


@jit
def array_sum(values, offset):
    return values[0] + values[1] + offset


@jit
def array_store(values, index, value):
    values[index] = value
    return values[index]


@jit
def array_augmented_store(values, index, value):
    values[index] += value
    values[index] *= 2
    return values[index]


@jit
def array_power_store(values, index):
    values[index] **= 3
    return values[index]


@jit
def invalid_array_dimension(values):
    return values[0]


@jit
def array_2d(values, row, col, value):
    values[row, col] = value
    return values[row, col] + values.shape[0]


@reg
def callback_increment(value: int) -> int:
    return value + 1


@jit
def callback_user(value):
    return callback_increment(value)


@jit
def annotated(value: int) -> int:
    return value + 2


def test_scalar_arithmetic_and_specialization():
    assert arithmetic(3, 4) == 14
    assert arithmetic(1.5, 2.5) == 8.0
    assert true_division(7, 2) == 3.5
    assert math.isclose(true_division(7.0, 2.0), 3.5)
    assert floor_and_mod(-5, 2) == -2


def test_control_flow_and_bool_semantics():
    assert count_down(5) == 112
    assert bool_ops(0, 5) == 1
    assert bool_ops(2, 5) == 5
    assert chained(1, 2, 3) == 1
    assert chained(1, 3, 2) == 0
    assert branch_assignment(1) == 4
    assert branch_assignment(-1) == 7
    assert multiple_assignment(4) == 10


def test_range_zero_step_is_rejected():
    @jit
    def invalid():
        for value in range(3, 0, 0):
            return value
        return 0
    with pytest.raises(InferError, match='must not be zero'):
        invalid()


def test_runtime_errors_are_translated_to_python_exceptions():
    with pytest.raises(ZeroDivisionError, match='division by zero'):
        true_division(1, 0)
    with pytest.raises(ZeroDivisionError, match='division by zero'):
        floor_and_mod(1, 0)
    with pytest.raises(ValueError, match='arg 3 must not be zero'):
        dynamic_range_step(0)


def test_arrays_cover_dtype_promotion_store_shape_and_strides():
    values = np.array([2, 4], dtype=np.int32)
    assert array_sum(values, 3) == 9
    assert array_store(values, 1, np.int32(8)) == 8
    wide_values = np.array([2, 8], dtype=np.int64)
    assert array_augmented_store(wide_values[::-1], 0, 3) == 22
    assert array_power_store(np.array([5], dtype=np.int64), 0) == 125
    matrix = np.zeros((2, 3), dtype=np.float32).T
    assert array_2d(matrix, 1, 0, np.float32(4.5)) == pytest.approx(7.5)
    with pytest.raises(ValueError, match='index count'):
        invalid_array_dimension(matrix)


def test_annotations_callbacks_and_mangler():
    assert annotated(4) == 6
    assert callback_user(4) == 5
    assert mangler('sample', [int64_t]) == 'sample__i64'
    array_type = arg_pytype(np.array([1], dtype=np.int64))
    assert isinstance(array_type, GenericType)
    assert array_type.b == int64_t


def test_parser_reports_unsupported_constant_and_location():
    with pytest.raises(CompileError, match='line'):
        ASTVisitor()('def unsupported():\n    return "nope"')


def test_integer_power_rejects_dynamic_exponent():
    @jit
    def unsupported(base, exponent):
        return base ** exponent
    with pytest.raises(InferError, match='constant exponent'):
        unsupported(2, 3)


def test_annotations_and_missing_return_paths_are_rejected():
    @jit
    def string_annotated(value: 'int') -> 'int':
        return value + 1

    assert string_annotated(2) == 3

    @jit
    def missing_return(value):
        if value:
            return 1

    with pytest.raises(InferError, match='path without return'):
        missing_return(1)
