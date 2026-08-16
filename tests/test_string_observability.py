from pyjiting import jit, runtime_stats


def test_string_runtime_callbacks_are_observable_by_operation():
    @jit
    def transform(value):
        return value.upper()

    before = runtime_stats()['string_callbacks'].get('upper', 0)
    assert transform('Abc') == 'ABC'
    after = runtime_stats()['string_callbacks'].get('upper', 0)
    assert after == before + 1


def test_unicode_comparison_is_native_and_does_not_enter_python_runtime():
    @jit
    def compare(left, right):
        return (left == right) + (left < right) * 2 + (left > right) * 4

    before = runtime_stats()['string_callbacks'].get('compare', 0)
    pairs = (('', ''), ('a', 'aa'), ('你A🙂', '你A𐐷'), ('a\0b', 'a\0c'), ('ß', 'ss'))
    for left, right in pairs:
        expected = (left == right) + (left < right) * 2 + (left > right) * 4
        assert compare(left, right) == expected
    after = runtime_stats()['string_callbacks'].get('compare', 0)
    assert after == before
