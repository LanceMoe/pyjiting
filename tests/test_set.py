from typing import FrozenSet, Set

import pytest

from pyjiting import inspect_specializations, jit, jit_from_source


def test_displays_comprehensions_and_boxed_iteration():
    @jit
    def collect(rows):
        item = 'outer'
        values = {item for key, rest in rows if key > 0 for item in rest if item is not None}
        all_values = {None, *values, True, ('日本語', 2)}
        return all_values, item, set(range(2)), frozenset(), tuple(sorted(values)), not set()

    values, outer, numbers, frozen, ordered, empty = collect([(1, [3, None, 3]), (-1, [2])])
    assert values == {None, 3, True, ('日本語', 2)}
    assert any(item is True for item in values)
    assert (outer, numbers, frozen, ordered, empty) == ('outer', {0, 1}, frozenset(), (3,), True)
    assert type(frozen) is frozenset and empty is True


def test_operators_preserve_identity_and_mixed_result_types():
    @jit
    def operate(left: Set[int], right: FrozenSet[int]):
        original = left
        results = [left | right, left & right, left - right, left ^ right,
                   right | left, right & left, right - left, right ^ left]
        relations = [left <= right, left < right, right >= left, right > left,
                     left == right, left != right, 2 in left, 5 not in left]
        left |= right
        left &= {1, 3, 4}
        left -= {3}
        left ^= {4, 5}
        return results, relations, left is original, left, right

    left, right = {1, 2}, frozenset({2, 3, 4})
    results, relations, same, updated, frozen = operate(left, right)
    assert results == [{1, 2, 3, 4}, {2}, {1}, {1, 3, 4},
                       {1, 2, 3, 4}, {2}, {3, 4}, {1, 3, 4}]
    assert [type(item) for item in results] == [set] * 4 + [frozenset] * 4
    assert relations == [False, False, False, False, False, True, True, True]
    assert all(type(item) is bool for item in relations)
    assert same is True and updated is left and frozen is right
    assert left == {1, 5}
    assert len(inspect_specializations(operate)) == 1


def test_standard_methods_accept_iterables_and_preserve_python_objects():
    @jit
    def mutate(values, groups):
        snapshot = values.copy()
        values.add(8)
        values.update(*groups)
        values.intersection_update([1, 2, 3, 4, 8], range(1, 9))
        values.difference_update([3], (8,))
        values.symmetric_difference_update([2, 9])
        values.discard(33)
        values.remove(9)
        removed = values.pop()
        values.clear()
        return snapshot, removed

    @jit
    def read(values):
        return (values.union(*[[2], [3]]), values.intersection([1, 2], [1]), values.difference([1], (8,)),
                values.symmetric_difference([2]), values.isdisjoint([8]),
                values.issubset([1, 2]), values.issuperset([1]))

    values = {1}
    snapshot, removed = mutate(values, ([2, 3], [3]))
    assert snapshot == {1} and snapshot is not values and removed == 1 and not values
    for container_type in (set, frozenset):
        result = read(container_type({1}))
        assert result == ({1, 2, 3}, {1}, set(), {1, 2}, True, True, True)
        assert all(type(item) is container_type for item in result[:4])


def test_custom_members_and_hash_semantics_survive_python_bridge():
    @jit
    def inspect(values, other):
        first = next(iter(values))
        if {1} in other:
            other.remove({1})
        immutable = frozenset(values)
        return first, immutable, hash(immutable)

    item = object()
    values, nested = {item}, {frozenset({1})}
    member, immutable, hashed = inspect(values, nested)
    assert member is item and immutable == values and hashed == hash(frozenset(values))
    assert not nested


@pytest.mark.parametrize(('statement', 'expected'), [
    ('values.add([])', TypeError),
    ('values.remove(10)', KeyError),
    ('values | [1]', TypeError),
    ('values.pop()', KeyError),
    ('for value in {1}:\n        values.add(3)', None),
    ('for value in values:\n        values.add(3)', RuntimeError),
])
def test_set_failures_follow_python_and_do_not_restart(statement, expected):
    source = f'def f(values):\n    {statement}\n    values.add(20)\n    return values'
    compiled = jit_from_source(source)
    namespace = {}
    exec(source, namespace)
    python_values = {1} if expected is RuntimeError else set()
    compiled_values = python_values.copy()
    if expected is None:
        assert compiled(compiled_values) == namespace['f'](python_values)
    else:
        with pytest.raises(expected) as python_error:
            namespace['f'](python_values)
        with pytest.raises(expected) as compiled_error:
            compiled(compiled_values)
        assert compiled_error.value.args == python_error.value.args
    assert compiled_values == python_values


def test_set_display_evaluates_items_before_hashing():
    build = jit_from_source('def build(events):\n    return {[], events.pop()}')

    events = [1]
    with pytest.raises(TypeError, match='unhashable'):
        build(events)
    assert events == []
