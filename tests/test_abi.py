# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

import ctypes

import numpy as np

from pyjiting.codegen import array_type, ir_i64
from pyjiting.ll_types import wrap_ndarray, wrap_type


def test_ndarray_descriptor_uses_fixed_windows_safe_fields():
    descriptor = wrap_type(array_type(ir_i64).pointee)

    assert [name for name, _ in descriptor._fields_] == ['data', 'ndim', 'shape', 'strides']
    assert descriptor.ndim.offset == ctypes.sizeof(ctypes.c_void_p)
    assert descriptor.shape.offset > descriptor.ndim.offset
    assert descriptor.strides.offset > descriptor.shape.offset


def test_ndarray_wrapper_preserves_element_strides_for_views():
    values = np.arange(8, dtype=np.int64)[::-2]
    data, ndim, shape, strides = wrap_ndarray(values)

    assert data
    assert ndim == 1
    assert shape[0] == 4
    assert strides[0] == -2
