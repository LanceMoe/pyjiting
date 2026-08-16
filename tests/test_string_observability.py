from pyjiting import jit, runtime_stats


def test_string_runtime_callbacks_are_observable_by_operation():
    @jit
    def transform(value):
        return value.upper()

    before = runtime_stats()['string_callbacks'].get('upper', 0)
    assert transform('Abc') == 'ABC'
    after = runtime_stats()['string_callbacks'].get('upper', 0)
    assert after == before + 1
