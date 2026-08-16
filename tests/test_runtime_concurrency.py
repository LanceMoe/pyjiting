from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from pyjiting import JITContext, clear_cache, jit, runtime_stats
from pyjiting.errors import RuntimeResourceError
from pyjiting.main import LLVMCodeGen


def test_concurrent_warmup_of_one_signature_compiles_once_and_records_wait(monkeypatch):
    @jit
    def kernel(value):
        return value + 1

    clear_cache(kernel)
    entered = threading.Event()
    release = threading.Event()
    original_visit = LLVMCodeGen.visit

    def controlled_visit(self, node):
        if getattr(node, 'fname', None) == 'kernel' and not entered.is_set():
            entered.set()
            assert release.wait(2)
        return original_visit(self, node)

    monkeypatch.setattr(LLVMCodeGen, 'visit', controlled_visit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(kernel.specialize, 3)
        assert entered.wait(2)
        second = pool.submit(kernel.warmup, 4)
        release.set()
        first.result(timeout=3)
        second.result(timeout=3)

    stats = runtime_stats(kernel)
    assert stats['specializations'] == 1
    assert stats['signatures'][0]['compile_count'] == 1
    assert runtime_stats()['compile_waits'] >= 1


def test_context_specialization_budget_is_atomic_under_concurrency():
    context = JITContext(max_specializations=1)

    @context.jit(max_specializations=None)
    def identity(value):
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(identity, value) for value in (3, 3.5)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except RuntimeResourceError:
            outcomes.append('limited')

    assert len([value for value in outcomes if value == 'limited']) == 1
    assert context.stats()['specializations'] == 1


def test_context_cannot_close_while_compilation_state_is_active():
    context = JITContext()
    state = context._state
    state.compilation_states[('test',)] = object()
    with pytest.raises(RuntimeError, match='compilation is active'):
        context.close()
    state.compilation_states.clear()
    context.close()
