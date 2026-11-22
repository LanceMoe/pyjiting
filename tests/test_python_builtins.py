# pyright: reportCallIssue=false, reportArgumentType=false

import math

import pytest

from pyjiting import jit, jit_from_source, runtime_stats


def test_round_pow_and_conversions_preserve_python_results_and_errors():
    def conversions(number, digits):
        return (round(number), round(number, digits), pow(2, -1, 5),
                pow(base=2, exp=65), bin(-12), oct(12), hex(12), ascii('café'))

    compiled = jit(conversions)
    for value in (2.5, -2.5, 2.675, -0.05):
        assert compiled(value, 2) == conversions(value, 2)
        assert type(compiled(value, 2)[0]) is int
    assert runtime_stats(compiled)['backend'] == 'llvm'
    with pytest.raises(ValueError):
        compiled(math.nan, 2)
    with pytest.raises(OverflowError):
        compiled(math.inf, 2)

    class Rounded:
        def __round__(self, ndigits=None):
            return ('custom', ndigits)

    @jit
    def custom(value: object):
        return round(value), round(value, 2)

    assert custom(Rounded()) == (('custom', None), ('custom', 2))


def test_iterator_builtins_keywords_and_iterable_reductions():
    @jit
    def iterators(values, transform, condition, zip_options):
        mapped = map(transform, values)
        filtered = filter(condition, mapped)
        return filtered, zip(values, values, **zip_options), all('abc'), any('')

    visited = []

    def triple(value):
        visited.append(value)
        return value * 3

    def even(value):
        return value % 2 == 0

    selected, paired, true, false = iterators([1, 2, 3], triple, even, {'strict': True})
    assert visited == []
    assert list(selected) == [6]
    assert visited == [1, 2, 3]
    assert list(paired) == [(1, 1), (2, 2), (3, 3)]
    assert (true, false) == (True, False)

    @jit
    def strict_zip(left, right):
        return list(zip(left, right, strict=True))

    with pytest.raises(ValueError, match='shorter'):
        strict_zip([1, 2], [3])


def test_min_max_sorted_and_sum_accept_standard_options():
    def key(value):
        return -value

    compiled = jit_from_source('''
        def select(values, options):
            return (min(values, key=key, default=None), max(*values, **options),
                    sorted(*[values], **options), sum(values, start=10))
    ''', namespace={'key': key})
    assert compiled([2, 3], {'key': key}) == (3, 2, [3, 2], 15)

    @jit
    def compare(left, right):
        return min(left, right), max(left, right)

    result = compare(2, 3.0)
    assert type(result[0]) is int and type(result[1]) is float
    assert all(math.isnan(value) for value in compare(float('nan'), 1.0))
    assert compare(1.0, float('nan')) == (1.0, 1.0)


def test_introspection_and_io_builtins_use_python_objects(capsys):
    class Holder:
        value = 7

    @jit
    def describe(value: object, function):
        print(getattr(value, 'value'), getattr(value, 'missing', 4), sep=':', end='!')
        return (hasattr(value, 'value'), hasattr(value, 'missing'), callable(function),
                isinstance(value, type(value)), issubclass(type(value), object))

    assert describe(Holder(), abs) == (True, False, True, True, True)
    assert capsys.readouterr().out == '7:4!'


def test_frame_builtins_and_shadowed_builtins_follow_python_scope():
    @jit
    def frame(value):
        local_value = value + 2
        return eval('value + local_value'), locals()['local_value']

    assert frame(5) == (12, 7)
    assert runtime_stats(frame)['backend'] == 'cpython'

    @jit
    def shadow(round, value):
        return round(value)

    assert shadow(str, 123) == '123'
    custom = jit_from_source('def format_number(value): return round(value)', namespace={'round': str})
    assert custom(123) == '123'
