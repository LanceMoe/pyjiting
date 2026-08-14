from typing import Callable

import pytest

from pyjiting import jit


@jit
def one_argument(value):
    return value + 1


@jit
def fixed_width_floor_div(value):
    return value // -1


@jit
def fixed_width_mod(value):
    return value % -1


def test_jit_wrapper_rejects_extra_and_missing_positional_arguments():
    dynamic_call: Callable[..., int] = one_argument
    with pytest.raises(TypeError, match=r'one_argument\(\) takes 1 positional arguments but 0 were given'):
        dynamic_call()
    with pytest.raises(TypeError, match=r'one_argument\(\) takes 1 positional arguments but 2 were given'):
        dynamic_call(1, 2)


def test_minimum_int_division_by_negative_one_uses_fixed_width_results():
    minimum = -(1 << 63)
    assert fixed_width_floor_div(minimum) == minimum
    assert fixed_width_mod(minimum) == 0
