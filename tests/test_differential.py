import math
import random

import numpy as np

from pyjiting import jit


@jit
def numeric_differential(left, right, limit):
    total = 0
    for value in range(limit):
        if value % 3:
            total += (left * value + right) // 7
        else:
            total -= (right - value) % 5
    return total


def numeric_reference(left, right, limit):
    total = 0
    for value in range(limit):
        if value % 3:
            total += (left * value + right) // 7
        else:
            total -= (right - value) % 5
    return total


@jit
def string_differential(value, index):
    return value[index] + value[::-1]


@jit
def array_differential(values, row, column):
    return values[row, column] * 2.0 + values.shape[0]


def test_seeded_numeric_programs_match_cpython():
    randomizer = random.Random(20260815)
    for _ in range(200):
        left = randomizer.randint(-10_000, 10_000)
        right = randomizer.randint(-10_000, 10_000)
        limit = randomizer.randint(0, 50)
        assert numeric_differential(left, right, limit) == numeric_reference(left, right, limit)


def test_unicode_string_operations_match_cpython():
    for value in ('abc', '你A🙂', 'a\0b', '𐐷ß'):
        for index in range(-len(value), len(value)):
            assert string_differential(value, index) == value[index] + value[::-1]


def test_strided_array_reads_match_numpy():
    base = np.arange(48, dtype=np.float64).reshape(6, 8)
    for values in (base, base.T, base[::-2, 1::3], np.asfortranarray(base)):
        row, column = values.shape[0] - 1, values.shape[1] - 1
        expected = values[row, column] * 2.0 + values.shape[0]
        assert math.isclose(array_differential(values, row, column), expected)
