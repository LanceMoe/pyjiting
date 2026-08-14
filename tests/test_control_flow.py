import pytest

from pyjiting import jit


@jit
def nested_loop_else(limit):
    total = 0
    for outer in range(limit):
        for inner in range(5):
            if inner == 2:
                continue
            if outer == 3:
                break
            total += outer + inner
        else:
            total += 100
    else:
        total += 1_000
    return total


@jit
def while_else_with_break(limit):
    value = 0
    while value < limit:
        if value == 3:
            break
        value += 1
    else:
        value += 100
    return value


@jit
def recursive_division_error(value):
    if value == 0:
        return 1 / 0
    return recursive_division_error(value - 1)


@jit
def negative_dynamic_range(start, step):
    total = 0
    for value in range(start, -4, step):
        total += value
    return total


def python_nested_loop_else(limit):
    total = 0
    for outer in range(limit):
        for inner in range(5):
            if inner == 2:
                continue
            if outer == 3:
                break
            total += outer + inner
        else:
            total += 100
    else:
        total += 1_000
    return total


def test_nested_break_continue_and_loop_else_match_python():
    assert nested_loop_else(5) == python_nested_loop_else(5)
    assert while_else_with_break(2) == 102
    assert while_else_with_break(6) == 3


def test_dynamic_negative_range_matches_python():
    assert negative_dynamic_range(5, -2) == sum(range(5, -4, -2))


def test_recursive_runtime_errors_propagate_to_the_dispatcher():
    with pytest.raises(ZeroDivisionError, match='division by zero'):
        recursive_division_error(3)
