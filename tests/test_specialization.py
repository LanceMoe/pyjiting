import os
import subprocess
import sys

from pyjiting import jit
from pyjiting.main import function_cache


def test_same_signature_is_compiled_once():
    @jit
    def cached(value):
        return value + 1

    before = len(function_cache)
    assert cached(4) == 5
    after_first = len(function_cache)
    assert cached(5) == 6
    assert len(function_cache) == after_first
    assert after_first == before + 1


class FirstKernel:
    @staticmethod
    def compute(value):
        return value + 1


class SecondKernel:
    @staticmethod
    def compute(value):
        return value + 2


first_compute = jit(FirstKernel.compute)
second_compute = jit(SecondKernel.compute)


def test_same_named_functions_from_different_scopes_do_not_share_machine_code():
    before = len(function_cache)
    assert first_compute(4) == 5
    assert second_compute(4) == 6
    assert len(function_cache) == before + 2


def closure_factory(offset):
    @jit
    def add_offset(value):
        return value + offset
    return add_offset


def test_factory_closures_have_isolated_compilation_units_and_machine_code():
    first = closure_factory(1)
    second = closure_factory(2)

    assert first(10) == 11
    assert second(10) == 12
    assert first.__pyjiting_tree__.compilation_unit_id != second.__pyjiting_tree__.compilation_unit_id
    assert first.__pyjiting_tree__.symbol != second.__pyjiting_tree__.symbol
    assert first.__pyjiting_tree__.semantic_fingerprint != second.__pyjiting_tree__.semantic_fingerprint


def test_mangler_is_stable_across_hash_seeds():
    code = (
        'from pyjiting.ll_types import mangler; '
        'from pyjiting.types import int64_t, double64_t; '
        "print(mangler('sample', [int64_t, double64_t]))"
    )
    outputs = []
    for seed in ('1', '987654'):
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = seed
        outputs.append(subprocess.check_output([sys.executable, '-c', code], env=env, text=True).strip())
    assert outputs[0] == outputs[1] == 'sample__i64_f64'
