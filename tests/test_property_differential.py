import numpy as np
from hypothesis import given, settings, strategies as st

from pyjiting import jit


@jit
def property_numeric(left, right):
    return (left * 3 + right) // 7, (left - right) % 11


@jit
def property_array_sum(values):
    return sum(values)


@given(st.integers(-(1 << 30), (1 << 30) - 1),
       st.integers(-(1 << 30), (1 << 30) - 1))
@settings(max_examples=100, deadline=None)
def test_generated_numeric_semantics_match_python(left, right):
    assert property_numeric(left, right) == (
        (left * 3 + right) // 7, (left - right) % 11)


@given(st.lists(st.integers(-1000, 1000), max_size=50), st.booleans())
@settings(max_examples=60, deadline=None)
def test_generated_strided_arrays_match_numpy(items, reverse):
    values = np.asarray(items, dtype=np.int64)
    if reverse:
        values = values[::-1]
    assert property_array_sum(values) == np.sum(values)
