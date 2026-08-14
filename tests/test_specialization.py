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
