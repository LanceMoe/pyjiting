import numpy as np
import pytest

from pyjiting import jit
from pyjiting.errors import CompileError, InferError
from pyjiting.ll_types import mangler
from pyjiting.main import arg_pytype
from pyjiting.types import TupleType, double64_t, int64_t, str_t
from tests.conftest import verified_module


CAPTURED_TUPLE = (7, '你', (2.5, True))


@jit
def make_pair(value) -> tuple[int, float]:
    return value, value / 2


@jit
def consume_pair(value: tuple[int, str]) -> str:
    return value[1] + str_marker(value[0])


@jit
def str_marker(value: int) -> str:
    if value:
        return '!'
    return '?'


@jit
def unpack_pair(value):
    number, text = value
    return text + chr(number)


@jit
def nested_roundtrip(value):
    return value, (value[-1], len(value))


@jit
def captured_tuple():
    return CAPTURED_TUPLE


@jit
def tuple_truth(value):
    if value:
        return 1
    return 0


def test_tuple_arguments_returns_annotations_and_unpacking():
    assert make_pair(8) == (8, 4.0)
    assert consume_pair((1, 'ok')) == 'ok!'
    assert unpack_pair((65, 'value=')) == 'value=A'


def test_nested_tuple_and_captured_tuple_round_trip():
    value = ('A', 3)
    assert nested_roundtrip(value) == (value, (3, 2))
    assert captured_tuple() == CAPTURED_TUPLE


def test_tuple_constant_index_len_negative_index_and_truthiness():
    @jit
    def inspect_tuple(value):
        return value[0], value[-1], len(value)

    assert inspect_tuple((2, 'end')) == (2, 'end', 2)
    assert tuple_truth(()) == 0
    assert tuple_truth((0,)) == 1


def test_tuple_specialization_type_and_mangler_are_structural():
    ty = arg_pytype((1, 'x', (2.5,)))
    assert ty == TupleType([int64_t, str_t, TupleType([double64_t])])
    assert mangler('sample', [ty]) == 'sample__tuple_i64_str_tuple_f64_end_end'


def test_tuple_invalid_index_unpack_and_array_elements_are_rejected():
    @jit
    def dynamic_index(value, index):
        return value[index]

    @jit
    def wrong_unpack(value):
        left, right, extra = value
        return left

    with pytest.raises(InferError, match='compile-time integer'):
        dynamic_index((1, 2), 0)
    with pytest.raises(InferError, match='cannot unpack tuple of length 2 into 3 names'):
        wrong_unpack((1, 2))
    with pytest.raises(TypeError, match='ndarray elements inside tuples'):
        nested_roundtrip((np.arange(2),))
    with pytest.raises(TypeError, match='ndarray elements inside tuples'):
        nested_roundtrip(((np.arange(2),),))


def test_tuple_annotation_rejects_variadic_shape():
    with pytest.raises(CompileError, match='unsupported annotation'):
        @jit
        def variadic(value: tuple[int, ...]):
            return value


def test_tuple_generated_module_verifies():
    verified_module('''
        def pair(value):
            result = (value, "ok")
            left, right = result
            return (left, right, len(result))
    ''', [int64_t])
