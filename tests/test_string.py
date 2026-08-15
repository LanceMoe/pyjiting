import pytest

from pyjiting import jit
from pyjiting.errors import InferError


@jit
def string_length(value: str) -> int:
    return len(value)


@jit
def string_build(left: str, right: str, count: int) -> str:
    return (left + right) * count


@jit
def string_index(value: str, index: int) -> str:
    return value[index]


@jit
def string_slice(value: str, begin: int, end: int) -> str:
    return value[begin:end]


@jit
def string_queries(value: str, needle: str) -> int:
    return (value.startswith(needle) * 1000 + value.endswith(needle) * 100 +
            value.find(needle) * 10 + value.count(needle))


@jit
def string_compare(left: str, right: str) -> int:
    return (left == right) + (left < right) * 2 + (left > right) * 4


@jit
def string_truth(value: str) -> int:
    if value:
        return 1
    return 0


@jit
def iterate_string(value: str) -> str:
    result = ''
    for char in value:
        result = char + result
    return result


@pytest.mark.parametrize('value', ['', 'abc', 'a\x00b', '你好', 'A😀B'])
def test_string_length_and_truth_match_python(value):
    assert string_length(value) == len(value)
    assert string_truth(value) == int(bool(value))


def test_string_results_support_unicode_nul_concat_and_repeat():
    assert string_build('你\x00', '好😀', 2) == '你\x00好😀你\x00好😀'
    assert string_build('abc', '', -2) == ''


def test_string_index_slice_and_iteration_match_python():
    value = 'A你😀Z'
    for index in range(-len(value), len(value)):
        assert string_index(value, index) == value[index]
    assert string_slice(value, -3, 99) == value[-3:99]
    assert string_slice(value, 3, 1) == ''
    assert iterate_string(value) == value[::-1]
    with pytest.raises(IndexError, match='index out of range'):
        string_index(value, len(value))


def test_string_comparison_and_queries_match_python():
    for left, right in [('abc', 'abc'), ('abc', 'abd'), ('你', '😀'), ('', 'a')]:
        assert string_compare(left, right) == ((left == right) + (left < right) * 2 + (left > right) * 4)
    value, needle = 'bananana', 'ana'
    expected = (value.startswith(needle) * 1000 + value.endswith(needle) * 100 +
                value.find(needle) * 10 + value.count(needle))
    assert string_queries(value, needle) == expected


@jit
def stepped_slice(value: str, begin: int, end: int, step: int) -> str:
    return value[begin:end:step]


@jit
def reverse_string(value: str) -> str:
    return value[::-1]


def test_string_slices_support_positive_negative_and_dynamic_steps():
    value = 'A你😀bcZ'
    for begin, end, step in [(0, 99, 2), (-1, -99, -1), (4, 0, -2), (-5, 5, 3)]:
        assert stepped_slice(value, begin, end, step) == value[begin:end:step]
    assert reverse_string(value) == value[::-1]
    with pytest.raises(ValueError, match='slice step cannot be zero'):
        stepped_slice(value, 0, 3, 0)
