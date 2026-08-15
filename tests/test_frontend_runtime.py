import warnings

import pytest

from pyjiting import jit, jit_from_source
from pyjiting.errors import CompileError, InferError


def test_explicit_source_entrypoint_compiles_dynamic_functions():
    add = jit_from_source('''
        def add(left, right):
            return left + right
    ''')

    assert add(3, 4) == 7
    assert jit.from_source is jit_from_source


def test_source_retrieval_failure_has_a_compile_error_with_remediation():
    namespace = {'__name__': 'dynamic_test'}
    exec('def generated(value):\n    return value + 1', namespace)

    with pytest.raises(CompileError, match=r'use jit\.from_source'):
        jit(namespace['generated'])


def test_fallback_mode_runs_the_original_function_and_warns_once():
    namespace = {'__name__': 'dynamic_test'}
    exec('def generated(value=3):\n    return [value]', namespace)

    fallback = jit(namespace['generated'], fallback=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        assert fallback() == [3]
        assert fallback(value=4) == [4]

    assert len(captured) == 1
    assert 'pyjiting fallback' in str(captured[0].message)


def test_unannotated_mutual_recursion_reports_a_compilation_error():
    namespace = {}
    even = jit_from_source('''
        def even(value):
            if value == 0:
                return True
            return odd(value - 1)
    ''', namespace=namespace)
    odd = jit_from_source('''
        def odd(value):
            if value == 0:
                return False
            return even(value - 1)
    ''', namespace=namespace)
    namespace.update(even=even, odd=odd)

    with pytest.raises(InferError, match='mutual recursion is not supported'):
        even(4)


def test_annotated_mutual_recursion_is_rejected_before_mcjit_finalization():
    namespace = {}
    even = jit_from_source('''
        def even(value: int) -> bool:
            if value == 0:
                return True
            return odd(value - 1)
    ''', namespace=namespace)
    odd = jit_from_source('''
        def odd(value: int) -> bool:
            if value == 0:
                return False
            return even(value - 1)
    ''', namespace=namespace)
    namespace.update(even=even, odd=odd)

    with pytest.raises(InferError, match='mutual recursion is not supported'):
        even(20)
