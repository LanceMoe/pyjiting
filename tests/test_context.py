import pytest

from pyjiting import JITContext, runtime_stats
from pyjiting.errors import InferError, RuntimeClosedError, RuntimeResourceError


def test_context_owns_an_isolated_engine_and_rejects_calls_after_close():
    context = JITContext()

    @context.jit
    def increment(value):
        return value + 1

    assert increment(4) == 5
    assert context.stats()['specializations'] == 1
    assert runtime_stats(increment)['specializations'] == 1
    context.close()
    assert context.stats()['closed'] is True
    with pytest.raises(RuntimeClosedError):
        increment(5)


def test_context_manager_and_resource_budgets():
    with JITContext(max_specializations=1, max_modules=1) as context:
        @context.jit(max_specializations=None)
        def identity(value):
            return value

        assert identity(3) == 3
        with pytest.raises(RuntimeResourceError, match='specialization limit'):
            identity(3.5)

    with pytest.raises(RuntimeClosedError):
        context.jit(lambda value: value)


def test_jit_calls_cannot_cross_contexts():
    first = JITContext()
    second = JITContext()

    @first.jit
    def callee(value):
        return value + 1

    @second.jit
    def caller(value):
        return callee(value)

    with pytest.raises(InferError, match='different runtime contexts'):
        caller(3)
