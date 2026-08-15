import pytest

from pyjiting import jit
from pyjiting.errors import InferError


@jit
def membership(needle: str, value: str) -> int:
    return (needle in value) + (needle not in value) * 2


@jit
def transform(value: str, old: str, new: str) -> str:
    return value.strip().lower().replace(old, new).upper()


@jit
def trim_variants(value: str) -> str:
    return value.lstrip() + '|' + value.rstrip()


@jit
def predicates(value: str) -> int:
    return (value.isalpha() + value.isalnum() * 2 + value.isdigit() * 4 +
            value.isspace() * 8)


@jit
def codepoint_roundtrip(value: str) -> str:
    return chr(ord(value))


@jit
def invalid_ord(value: str) -> int:
    return ord(value)


@jit
def invalid_chr(value: int) -> str:
    return chr(value)


@jit
def call_invalid_chr(value: int) -> str:
    return invalid_chr(value)


@jit
def call_invalid_ord(value: str) -> int:
    return invalid_ord(value)


@jit
def recursive_slice(value: str, step: int, depth: int) -> str:
    if depth == 0:
        return value[::step]
    return recursive_slice(value, step, depth - 1)


@pytest.mark.parametrize(
    ('needle', 'value'),
    [('', ''), ('', 'abc'), ('a', 'abc'), ('😀', 'A😀B'), ('x', '你好')],
)
def test_string_membership_matches_python(needle, value):
    expected = (needle in value) + (needle not in value) * 2
    assert membership(needle, value) == expected


def test_string_transforms_and_trim_variants_match_python():
    value, old, new = '  Straße你\x00  ', 'ss', 'X'
    expected = value.strip().lower().replace(old, new).upper()
    assert transform(value, old, new) == expected
    assert trim_variants(value) == value.lstrip() + '|' + value.rstrip()


@pytest.mark.parametrize('value', ['', 'abc', '１２３', 'a1', ' \t', '你', '😀'])
def test_string_predicates_match_python(value):
    expected = (value.isalpha() + value.isalnum() * 2 + value.isdigit() * 4 +
                value.isspace() * 8)
    assert predicates(value) == expected


@pytest.mark.parametrize('value', ['A', '你', '😀', '\x00'])
def test_ord_chr_roundtrip_matches_python(value):
    assert codepoint_roundtrip(value) == value


def test_ord_chr_errors_use_python_exception_classes_and_propagate():
    for value in ('', 'ab'):
        with pytest.raises(TypeError, match=r'ord\(\) expected a character'):
            invalid_ord(value)
        with pytest.raises(TypeError, match=r'ord\(\) expected a character'):
            call_invalid_ord(value)
    for value in (-1, 0x110000):
        with pytest.raises(ValueError, match=r'chr\(\) arg not in range'):
            invalid_chr(value)
        with pytest.raises(ValueError, match=r'chr\(\) arg not in range'):
            call_invalid_chr(value)
    with pytest.raises(ValueError, match='slice step cannot be zero'):
        recursive_slice('abc', 0, 3)


def test_new_string_intrinsics_reject_invalid_static_types_and_arities():
    @jit
    def invalid_membership(value):
        return value in 3

    @jit
    def invalid_replace(value: str) -> str:
        return value.replace('x')  # pyright: ignore[reportCallIssue]

    with pytest.raises(InferError, match='requires string operands'):
        invalid_membership('x')
    with pytest.raises(InferError, match='replace expects two string arguments'):
        invalid_replace('x')
