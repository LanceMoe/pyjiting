# pyright: reportAttributeAccessIssue=false

import pytest

from pyjiting import ast as core
from pyjiting.errors import CompileError
from pyjiting.parser import ASTVisitor
from pyjiting.types import int64_t


def test_parser_keeps_source_location_and_core_index_shape():
    tree = ASTVisitor()('''
        def kernel(values, row, col):
            values[row, col] = 3
            return values[row, col]
    ''')

    store, result = tree.body
    assert isinstance(store, core.StoreIndex)
    assert len(store.indices) == 2
    assert store.lineno == 3
    assert isinstance(result.value, core.Index)
    assert len(result.value.indices) == 2


def test_parser_lowers_multiple_assignment_through_one_hidden_temporary():
    tree = ASTVisitor()('''
        def duplicate(value):
            left = right = value
            return left + right
    ''')

    assignments, result = tree.body
    temporary, left, right = assignments
    assert isinstance(temporary, core.Assign)
    assert temporary.ref == '__assign_tmp'
    assert isinstance(left.value, core.Var) and left.value.id == '__assign_tmp'
    assert isinstance(right.value, core.Var) and right.value.id == '__assign_tmp'
    assert isinstance(result, core.Return)


def test_ndarray_annotation_defers_element_type_to_runtime_specialization():
    tree = ASTVisitor()('''
        import numpy as np
        def total(values: np.ndarray) -> int:
            return values[0]
    ''')

    assert tree.args[0].annotation is None
    assert tree.return_annotation == int64_t


@pytest.mark.parametrize('statement', ['break', 'continue'])
def test_parser_rejects_loop_control_outside_a_loop(statement):
    source = 'def invalid():\n    ' + statement
    with pytest.raises(CompileError, match=statement + ' outside loop'):
        ASTVisitor()(source)


def test_parser_rejects_unknown_annotations_with_a_location():
    with pytest.raises(CompileError, match=r'line 2:.*unsupported annotation'):
        ASTVisitor()('''
            def bad(value: complex):
                return value
        ''')


@pytest.mark.parametrize(
    ('source', 'message'),
    [
        ('def defaulted(value=1):\n    return value', 'default, keyword-only, and variadic parameters'),
        ('def keyword_only(*, value):\n    return value', 'default, keyword-only, and variadic parameters'),
        ('def variadic(*values):\n    return 0', 'default, keyword-only, and variadic parameters'),
        ('def caller(value):\n    return callee(value=value)', 'keyword arguments are not supported'),
    ],
)
def test_parser_rejects_parameter_and_call_forms_outside_the_subset(source, message):
    with pytest.raises(CompileError, match=message):
        ASTVisitor()(source)
