import math

import numpy as np
import pytest

from pyjiting import jit
from pyjiting.errors import CompileError, InferError
from tests.conftest import verified_module
from pyjiting.types import double64_t


GLOBAL_SCALE = 2.5
GLOBAL_LABEL = 'captured'


@jit
def captured_globals(value):
    return value * GLOBAL_SCALE


@jit
def captured_string() -> str:
    return GLOBAL_LABEL


@jit
def trig(value):
    return math.sin(value) + math.cos(value)


@jit
def logarithms(value):
    return math.log(value) + math.log2(value) + math.log10(value)


@jit
def root_and_exp(value):
    return math.sqrt(value) + math.exp(value)


@jit
def classify(value):
    return math.isnan(value) + math.isinf(value) * 2 + math.isfinite(value) * 4


@jit
def constants():
    return math.pi + math.e + math.tau


@jit
def array_sum_builtin(values):
    return sum(values)


@jit
def array_any_all(values):
    return any(values) + all(values) * 2


@jit
def invalid_sqrt(value):
    return math.sqrt(value)


@jit
def call_invalid_sqrt(value):
    return invalid_sqrt(value)


def test_global_and_closure_constants_are_captured_at_decoration_time():
    assert captured_globals(4) == 10.0
    assert captured_string() == 'captured'
    original_scale = globals()['GLOBAL_SCALE']
    globals()['GLOBAL_SCALE'] = 9.0
    try:
        assert captured_globals(4) == 10.0
    finally:
        globals()['GLOBAL_SCALE'] = original_scale

    factor = np.float32(1.25)

    @jit
    def closure(value):
        return value * factor

    factor = np.float32(9.0)
    assert closure(np.float32(4.0)) == pytest.approx(5.0)


def test_math_intrinsics_and_constants_match_python():
    for value in (-0.0, 0.25, 2.0):
        assert trig(value) == pytest.approx(math.sin(value) + math.cos(value))
    for value in (0.25, 1.0, 10.0):
        assert logarithms(value) == pytest.approx(math.log(value) + math.log2(value) + math.log10(value))
    for value in (0.0, 0.5, 2.0):
        assert root_and_exp(value) == pytest.approx(math.sqrt(value) + math.exp(value))
    assert constants() == pytest.approx(math.pi + math.e + math.tau)

    import math as aliased_math

    @jit
    def aliased(value):
        return aliased_math.sqrt(value) + aliased_math.pi

    assert aliased(9.0) == pytest.approx(3.0 + math.pi)


@pytest.mark.parametrize('value', [0.0, float('inf'), float('-inf'), float('nan')])
def test_math_classification_matches_python(value):
    expected = math.isnan(value) + math.isinf(value) * 2 + math.isfinite(value) * 4
    assert classify(value) == expected


def test_math_errors_match_python_and_propagate_between_jit_functions():
    for value in (-1.0, float('-inf')):
        with pytest.raises(ValueError, match='math domain error'):
            invalid_sqrt(value)
        with pytest.raises(ValueError, match='math domain error'):
            call_invalid_sqrt(value)
    with pytest.raises(ValueError, match='math domain error'):
        trig(float('inf'))
    with pytest.raises(ValueError, match='math domain error'):
        logarithms(0.0)
    with pytest.raises(OverflowError, match='math range error'):
        root_and_exp(1000.0)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_array_sum_supports_dtypes_empty_arrays_and_negative_strides(dtype):
    values = np.arange(8, dtype=dtype)[::-2]
    assert array_sum_builtin(values) == pytest.approx(sum(values))
    assert array_sum_builtin(np.array([], dtype=dtype)) == 0


def test_array_any_all_match_python_identities_and_nan_truthiness():
    for values in (
        np.array([], dtype=np.int64),
        np.array([0, 0, 3], dtype=np.int32),
        np.array([1.0, float('nan')], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float32)[::-1],
    ):
        assert array_any_all(values) == int(any(values)) + int(all(values)) * 2
    with pytest.raises(ValueError, match='index count'):
        array_sum_builtin(np.zeros((2, 2), dtype=np.float64))


def test_numeric_intrinsics_reject_unsupported_static_inputs():
    @jit
    def invalid_math(value: str):
        return math.sin(value)  # pyright: ignore[reportArgumentType]

    @jit
    def invalid_sum(value):
        return sum(value)

    with pytest.raises(InferError, match='expects one numeric argument'):
        invalid_math('x')
    with pytest.raises(InferError, match='expects one array argument'):
        invalid_sum(3)


def test_math_intrinsics_verify_from_source_and_large_constants_are_rejected():
    verified_module('''
        def kernel(value):
            return math.sin(value) + math.pi
    ''', [double64_t])

    with pytest.raises(CompileError, match='outside the supported Int64 range'):
        @jit
        def too_large():
            return 1267650600228229401496703205376
