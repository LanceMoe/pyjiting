from collections import UserDict
from typing import Dict

import numpy as np
import pytest

from pyjiting import inspect_specializations, jit, jit_from_source


def test_displays_constructors_and_comprehensions_preserve_insertion_order():
    @jit
    def collect(incoming: Dict[str, object], rows):
        code = 'outside'
        result = {code: name if keep else None for keep, pairs in rows for code, name in pairs}
        display = {'first': 1, **incoming, 'last': 3, **result}
        kwargs = dict([('first', 0)], **incoming, extra=True)
        return display, kwargs, dict(*[incoming]), code

    source = {'first': 2, 'middle': False}
    rows = [(True, [('name', '日本語')]), (False, [('last', 'discarded')])]
    display, kwargs, copied, outer = collect(source, rows)
    assert list(display.items()) == [('first', 2), ('middle', False), ('last', None), ('name', '日本語')]
    assert list(kwargs.items()) == [('first', 2), ('middle', False), ('extra', True)]
    assert display['middle'] is False and kwargs['extra'] is True
    assert copied == source and copied is not source and outer == 'outside'


def test_keys_values_and_inplace_updates_keep_object_identity():
    @jit
    def change(data: dict, key, extras) -> dict:
        existing = data[key, 1]
        data[key, 1] += 2
        data[()] = [existing]
        data[key,] = data[()]
        del data[()]
        data |= extras
        return data

    value = object()
    data = {('name', 1): 5, 'kept': value}
    assert change(data, 'name', [('new', 9)]) is data
    assert list(data.items()) == [(('name', 1), 7), ('kept', value), (('name',), [5]), ('new', 9)]
    assert data['kept'] is value


def test_standard_methods_and_views_match_python():
    def methods(data, more):
        initial = data.copy()
        keys, values, items = data.keys(), data.values(), data.items()
        proxy = keys.mapping
        initial_values = list(values)
        data.update([('next', 3)], extra=4, **more)
        child = data.setdefault('child', [])
        child.append(data.get('missing', 7))
        removed = data.pop('next')
        none = data.pop('missing', None)
        entry = data.popitem()
        backwards = list(reversed(items))
        matching = keys & {'base', 'extra', 'missing'}
        data.clear()
        return initial, initial_values, removed, none, entry, backwards, matching, proxy, keys, values, items

    compiled = jit(methods)
    first, second = {'base': 1}, {'base': 1}
    actual, expected = compiled(first, {'end': 5}), methods(second, {'end': 5})
    assert actual[:8] == expected[:8] and first == second == {}
    first['later'] = 10
    _, keys, values, items = actual[-4:]
    assert list(keys) == ['later'] and list(values) == [10] and list(items) == [('later', 10)]
    assert actual[-4]['later'] == 10

    @jit
    def consume(keys, values, items):
        return list(keys), list(values), list(items)

    assert consume(keys, values, items) == (['later'], [10], [('later', 10)])


def test_fromkeys_defaults_and_merge_keep_shared_values():
    @jit
    def merge(left, right, defaults={}):
        created = dict.fromkeys(left, [])
        instance_created = left.fromkeys(('a', 'b'), created)
        merged = left | right
        defaults.update(merged)
        return created, instance_created, merged, defaults, left == right, len(merged), bool(merged)

    left, right = {'a': 1, 'b': 2}, {'a': 3, 'c': 4}
    created, instance_created, merged, defaults, equal, size, truth = merge(left, right)
    assert created['a'] is created['b'] and instance_created['a'] is created
    assert list(merged.items()) == [('a', 3), ('b', 2), ('c', 4)]
    assert defaults == merged and defaults is not merged
    assert merge({'last': 5}, right)[3] is defaults and defaults['last'] == 5
    assert equal is False and size == 3 and truth is True
    assert list(left.items()) == [('a', 1), ('b', 2)]
    assert len(inspect_specializations(merge)) == 1


def test_mapping_protocol_and_numpy_values_interoperate():
    @jit
    def build(mapping, array):
        data = {**mapping, 'data': array}
        data[1:4] = 7
        return data, data[1:4]

    array = np.arange(3)
    data, value = build(UserDict({'a': 1}), array)
    assert data['a'] == 1 and data['data'] is array and value == 7


def test_dict_comprehension_evaluates_key_then_value():
    @jit
    def build(operands):
        return {operands.pop(): operands.pop() for _ in range(2)}

    assert build([14, 'second', 10, 'first']) == {'first': 10, 'second': 14}


@pytest.mark.parametrize(('statement', 'error'), [
    ('data[99]', KeyError),
    ('data[[]] = 4', TypeError),
    ('del data[99]', KeyError),
    ('data.popitem()', KeyError),
    ('data.update([(1, 2), (3,)])', ValueError),
    ('{**[(1, 2)]}', TypeError),
    ('dict(**{1: 2})', TypeError),
    ('dict(name=1, **{"name": 2})', TypeError),
    ('data.update(**{"name": 2}, name=1)', TypeError),
    ('data.update(**[("name", 2)])', TypeError),
    ('data | [(1, 2)]', TypeError),
])
def test_dict_errors_and_partial_mutation_follow_python(statement, error):
    source = f'def f(data):\n    {statement}\n    data[20] = 1'
    compiled = jit_from_source(source)
    namespace = {}
    exec(source, namespace)
    expected, actual = {}, {}
    with pytest.raises(error) as python_error:
        namespace['f'](expected)
    with pytest.raises(error) as jit_error:
        compiled(actual)
    assert jit_error.value.args == python_error.value.args
    assert actual == expected


def test_mutation_during_view_iteration_propagates_even_with_fallback():
    @jit(fallback=True)
    def edit(data):
        data['before'] = 1
        for key in data.keys():
            data[key + 'new'] = data[key]
        return data

    data = {'a': 1}
    with pytest.raises(RuntimeError, match='changed size'):
        edit(data)
    assert list(data) == ['a', 'before', 'anew']
