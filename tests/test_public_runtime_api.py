# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

import inspect

import pytest

from pyjiting import (clear_cache, get_llvm_ir, inspect_specializations, jit,
                       runtime_stats)
from pyjiting.errors import CompileError


def test_keyword_and_default_arguments_share_one_specialization():
    @jit
    def combine(left: int, right: int = 2) -> int:
        return left * 10 + right

    clear_cache(combine)
    assert combine(3) == 32
    assert combine(left=3, right=4) == 34
    assert combine(right=5, left=3) == 35
    assert runtime_stats(combine)['specializations'] == 1
    assert runtime_stats(combine)['calls'] == 3
    assert str(inspect.signature(combine)) == '(left: int, right: int = 2) -> int'


def test_python_signature_reports_invalid_calls():
    @jit
    def add(left, right):
        return left + right

    with pytest.raises(TypeError):
        add(left=1)
    with pytest.raises(TypeError):
        add(1, left=2, right=3)
    with pytest.raises(TypeError):
        add(left=1, right=2, extra=3)


def test_mutable_or_unsupported_default_is_rejected():
    def invalid(value=[]):
        return 0

    with pytest.raises(CompileError, match='unsupported default value.*value'):
        jit(invalid)


def test_specialize_compiles_without_executing_and_exposes_readonly_metadata():
    @jit
    def pure(value):
        return value + 1

    clear_cache(pure)
    metadata = pure.specialize(4)
    assert metadata['argument_types'] == ('Int64',)
    assert metadata['return_type'] == 'Int64'
    assert metadata['compile_time_ns'] > 0
    assert runtime_stats(pure)['calls'] == 0
    with pytest.raises(TypeError):
        metadata['calls'] = 3
    assert pure(4) == 5
    assert runtime_stats(pure)['calls'] == 1


def test_specialization_inspection_and_ir_query():
    @jit
    def square(value):
        return value * value

    clear_cache(square)
    unoptimized = get_llvm_ir(square, 3)
    optimized = get_llvm_ir(square, 3, optimized=True)
    entries = inspect_specializations(square)

    assert 'define' in unoptimized
    assert 'define' in optimized
    assert len(entries) == 1
    assert entries[0]['native_symbol'] in unoptimized
