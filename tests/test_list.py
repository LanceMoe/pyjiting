from copy import deepcopy

import numpy as np
import pytest

from pyjiting import clear_cache, inspect_specializations, jit, jit_from_source


def test_mixed_and_cyclic_lists_preserve_identity_across_jit_calls():
    @jit
    def append_value(values, item):
        values.append(item)
        return values

    @jit
    def mutate(values: list[object], alias: list[object]) -> list[object]:
        alias.append([None, True, '日本語'])
        result = append_value(values, alias)
        return result

    marker = object()
    values = [marker]
    assert mutate(values, values) is values
    assert values[0] is marker
    assert values[1] == [None, True, '日本語']
    nested = values[1]
    assert isinstance(nested, list) and nested[1] is True
    assert values[2] is values
    assert mutate([], []) is not values
    assert len(inspect_specializations(mutate)) == 1


def test_decorated_list_defaults_keep_python_identity_across_calls():
    @jit
    def append(values=[]):
        values.append(len(values))
        return values

    first = append()
    assert first == [0]
    assert append() is first
    assert first == [0, 1]
    assert append(values=[]) == [0]


def test_mutations_and_shallow_copies_match_python():
    def update(values, replacement):
        alias = values
        first = values.copy()
        values[-1] = [11]
        values[1:3] = replacement
        del values[::2]
        alias += (8, 9)
        alias *= 2
        alias[0] += [42]
        alias.insert(1, None)
        alias.remove(None)
        popped = alias.pop(-1)
        alias.reverse()
        old = [*first, popped, *alias[1:3]]
        return old, alias

    arguments = ([[1], [2], [3], [4]], [[5], [6], [7]])
    original = deepcopy(arguments)
    updated = deepcopy(arguments)
    first_child = updated[0][0]
    replacement_child = updated[1][0]
    expected, expected_alias = update(*original)
    actual, actual_alias = jit(update)(*updated)
    assert updated == original
    assert (actual, actual_alias) == (expected, expected_alias)
    assert actual_alias is updated[0]
    assert actual[0] is first_child
    assert actual_alias[-1] is replacement_child


def test_comprehensions_unpacking_and_list_iterators_match_python():
    def make_rows(rows):
        number = 'outer'
        result = [(number, first + i, last) for number, (first, *rest, last) in enumerate(rows)
                  if first >= 0 for i in range(len(rest))]
        return result, number, list(reversed(range(4)))

    rows = [[2, 3, 4, 5], [-2, 0, 4], [1, 2]]
    assert jit(make_rows)(rows) == make_rows(rows)


def test_iterator_observes_in_place_changes_and_boxed_locals_keep_their_types():
    def total(values):
        result = 0
        for item in values:
            if item is None:
                continue
            if item == 1:
                values.append(2.5)
            result += item
        else:
            result += 0.5
        return result

    compiled = jit(total)
    for values in ([1, None, 2, 3], [], [None], [2.5, 3]):
        original, updated = values.copy(), values.copy()
        assert compiled(updated) == total(original)
        assert updated == original


def test_sort_and_standard_list_methods_match_python():
    def reorder(values, pivot):
        values.sort(key=lambda item: abs(item - pivot), reverse=True)
        ranking = sorted(values, reverse=True)
        metrics = [values.count(3), values.index(3, 1), len(values), min(values),
                   max(values), sum(values, 10), any(values), all(values)]
        snapshot = values[::-1]
        values.extend([1, 2])
        values.clear()
        return ranking, metrics, snapshot

    original = [7, 3, 0, -2, 3]
    updated = original.copy()
    actual = jit(reorder)(updated, 2)
    assert actual == reorder(original, 2)
    assert actual[1][-2] is True
    assert actual[1][-1] is False
    assert updated == original == []


def test_sort_keys_in_a_native_callee_survive_cache_clearing():
    ordered = jit_from_source('''
        def ordered(values, key):
            values.sort(key=distance, reverse=True)
            return sorted(values, key=key)
    ''', namespace={'distance': lambda item: abs(item - 4)})
    caller = jit_from_source('''
        def caller(values, key):
            return ordered(values, key)
    ''', namespace={'ordered': ordered})

    def key(item):
        return -(item % 2)

    values = [6, 3, 1, 4]
    expected_input = sorted(values, key=lambda item: abs(item - 4), reverse=True)
    expected_output = sorted(expected_input, key=key)
    for _ in range(2):
        sample = values.copy()
        assert caller(sample, key) == expected_output
        assert sample == expected_input
        clear_cache(ordered)


def test_list_index_and_reordered_loop_bounds_are_evaluated_once():
    def consume(values):
        return [number for number in range(values.pop())]

    values = [5, 2]
    assert jit(consume)(values) == [0, 1]
    assert values == [5]


def test_list_interoperates_with_numpy_arguments_and_conditional_items():
    compiled = jit_from_source('''
        def f(array):
            values = list(array)
            converted = [value if value > 0 else None for value in values]
            return converted, [array]
    ''')
    array = np.arange(-2, 4, dtype=np.int32)[::-2]
    converted, nested = compiled(array)
    assert converted == [3, 1, None]
    assert type(converted[0]) is array.dtype.type
    assert nested[0] is array


def test_list_operators_use_python_values_and_short_circuit():
    checked = jit_from_source('''
        def test(xs, ys):
            return [xs + ys, 2 * xs, xs * -1, 2 in xs, 3 not in xs, xs < ys, xs != ys,
                    xs and xs[-1], ys or xs, [] and ys[999]]
    ''')
    xs, ys = [1, 2], [3]
    result = checked(xs, ys)
    assert result == [[1, 2, 3], [1, 2, 1, 2], [], True, True, True, True, 2, ys, []]
    assert result[3:7] == [True] * 4 and all(value is True for value in result[3:7])
    assert result[8] is ys

    empty = jit_from_source('def empty(values):\n    return not values')
    assert empty([]) is True
    assert empty(xs) is False


@pytest.mark.parametrize(
    ('operation', 'error'),
    [
        ('return values[99]', IndexError),
        ('return values[::0]', ValueError),
        ('values[::2] = [1, 2, 3, 4]', ValueError),
        ('values.remove(99)', ValueError),
        ('values.sort()', TypeError),
        ('first, second = values', ValueError),
        ('del values[99]', IndexError),
    ],
)
def test_list_errors_propagate_with_python_messages_and_no_fallback(operation, error):
    source = f'def modify(values):\n    {operation}\n    values.append("unreachable")'
    native = jit_from_source(source)
    namespace = {}
    exec(source, namespace)
    python = namespace['modify']
    original = [1, 'x', 2]
    updated = original.copy()
    with pytest.raises(error) as expected:
        python(original)
    with pytest.raises(error) as actual:
        native(updated)
    assert str(actual.value) == str(expected.value)
    assert updated == original


def test_list_mutation_failure_does_not_reexecute_with_fallback():
    @jit(fallback=True)
    def modify(values):
        values.append(1)
        return values[99]

    values = []
    with pytest.raises(IndexError):
        modify(values)
    assert values == [1]
