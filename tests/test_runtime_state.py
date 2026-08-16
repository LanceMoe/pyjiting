import pytest

from pyjiting import clear_cache, jit, runtime_stats
from pyjiting.errors import InferError, SpecializationLimitError


@jit(max_specializations=1)
def limited_identity(value):
    return value


@jit
def cacheable_increment(value):
    return value + 1


def test_per_function_specialization_limit_is_enforced():
    assert limited_identity(3) == 3
    with pytest.raises(SpecializationLimitError, match=r'specialization limit \(1\)'):
        limited_identity(3.5)


def test_runtime_stats_and_function_cache_clear_are_observable_and_safe():
    clear_cache(cacheable_increment)
    before = runtime_stats()

    assert cacheable_increment(4) == 5
    compiled = runtime_stats()
    assert compiled['specializations'] == before['specializations'] + 1
    assert compiled['compile_misses'] == before['compile_misses'] + 1

    assert clear_cache(cacheable_increment) == 1
    cleared = runtime_stats()
    assert cleared['specializations'] == before['specializations']
    assert cleared['retained_modules'] == compiled['retained_modules']

    assert cacheable_increment(9) == 10
    recompiled = runtime_stats()
    assert recompiled['retained_modules'] == compiled['retained_modules'] + 1


def test_cache_configuration_validation_and_clear_target_validation():
    def plain(value):
        return value

    with pytest.raises(ValueError, match='positive integer or None'):
        jit(plain, max_specializations=0)
    with pytest.raises(TypeError, match='expects a @jit function'):
        clear_cache(lambda: None)


def test_deterministic_compilation_failures_are_cached_and_clearable():
    @jit
    def invalid(value):
        return missing(value)  # pyright: ignore[reportUndefinedVariable]

    clear_cache(invalid)
    before = runtime_stats()['compile_failures']
    with pytest.raises(InferError, match='not registered'):
        invalid(3)
    with pytest.raises(InferError, match='not registered'):
        invalid(3)
    after = runtime_stats()
    assert after['compile_failures'] == before + 1
    assert after['failure_cache_hits'] >= 1
    assert clear_cache(invalid) == 1
    with pytest.raises(InferError, match='not registered'):
        invalid(3)
    assert runtime_stats()['compile_failures'] == before + 2
