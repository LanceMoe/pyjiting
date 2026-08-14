import pytest

from conftest import infer_function
from pyjiting.errors import InferError
from pyjiting.types import (bool_t, double64_t, float32_t, int32_t, int64_t,
                            make_array_type)


@pytest.mark.parametrize(
    ('arg_types', 'operand_type', 'result_type'),
    [
        ([bool_t, bool_t], int64_t, int64_t),
        ([int32_t, int64_t], int64_t, int64_t),
        ([int32_t, float32_t], float32_t, float32_t),
        ([int64_t, float32_t], double64_t, double64_t),
        ([float32_t, double64_t], double64_t, double64_t),
    ],
)
def test_numeric_promotion_matrix(arg_types, operand_type, result_type):
    tree, signature = infer_function('''
        def add(left, right):
            return left + right
    ''', arg_types)

    expression = tree.body[0].value
    assert expression.operand_type == operand_type
    assert expression.type == result_type
    assert signature.return_type == result_type


def test_assignment_rejects_narrowing():
    with pytest.raises(InferError, match='cannot use Int64 where Int32 is required'):
        infer_function('''
            def narrow(value):
                target: int32 = value
                return target
        ''', [int64_t])


def test_array_annotation_is_specialized_from_the_call():
    tree, signature = infer_function('''
        import numpy as np
        def first(values: np.ndarray):
            return values[0]
    ''', [make_array_type(float32_t)])

    assert tree.args[0].type == make_array_type(float32_t)
    assert signature.return_type == float32_t
