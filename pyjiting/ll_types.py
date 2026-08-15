# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalSubscript=false

import ctypes
from typing import Any

import numpy as np
from llvmlite import ir

from .string_runtime import (StringDescriptor, StringPointer, begin_call, end_call,
                             make_string, to_python)


ERROR_DIVISION_BY_ZERO = 1
ERROR_RANGE_STEP_ZERO = 2
ERROR_ARRAY_DIMENSION_MISMATCH = 3
ERROR_INDEX_OUT_OF_BOUNDS = 4
ERROR_SLICE_STEP_ZERO = 5
ERROR_ORD_LENGTH = 6
ERROR_CHR_RANGE = 7


_scalar_ctypes = {1: ctypes.c_int8, 8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64}
_numpy_ctypes = {np.dtype(np.int32): ctypes.c_int32, np.dtype(np.int64): ctypes.c_int64, np.dtype(np.float32): ctypes.c_float, np.dtype(np.float64): ctypes.c_double}


def type_repr(ty):
    from .types import GenericType, array_t
    if isinstance(ty, GenericType) and ty.a == array_t: return f'arr_{type_repr(ty.b)}'
    return {'Int32': 'i32', 'Int64': 'i64', 'Bool': 'bool', 'Float': 'f32', 'Double': 'f64',
            'String': 'str', 'Void': 'void'}.get(str(ty), str(ty).lower())


def mangler(fname, signature): return fname + '__' + '_'.join(type_repr(ty) for ty in signature)


def wrap_type(llvm_type) -> Any:
    if isinstance(llvm_type, ir.IntType): return _scalar_ctypes[llvm_type.width]
    if isinstance(llvm_type, ir.DoubleType): return ctypes.c_double
    if isinstance(llvm_type, ir.FloatType): return ctypes.c_float
    if isinstance(llvm_type, ir.VoidType): return None
    if isinstance(llvm_type, ir.PointerType): return ctypes.POINTER(wrap_type(llvm_type.pointee))
    if isinstance(llvm_type, ir.IdentifiedStructType):
        if llvm_type.name == 'pyjiting.string': return StringDescriptor
        cached = getattr(llvm_type, '_pyjiting_ctype', None)
        if cached is not None: return cached
        fields = [('data', wrap_type(llvm_type.elements[0])), ('ndim', ctypes.c_int64), ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64))]
        ctype = type(llvm_type.name.replace('.', '_'), (ctypes.Structure,), {'_fields_': fields})
        setattr(llvm_type, '_pyjiting_ctype', ctype)
        return ctype
    raise RuntimeError(f'Unknown LLVM type {llvm_type}')


def wrap_ndarray(value):
    dtype = np.dtype(value.dtype)
    if dtype not in _numpy_ctypes: raise TypeError(f'unsupported ndarray dtype {dtype}')
    data = value.ctypes.data_as(ctypes.POINTER(_numpy_ctypes[dtype]))
    shape = (ctypes.c_int64 * value.ndim)(*value.shape)
    strides = (ctypes.c_int64 * value.ndim)(*(stride // value.dtype.itemsize for stride in value.strides))
    return data, value.ndim, shape, strides


def wrap_arg(arg, value):
    if isinstance(value, np.ndarray):
        data, ndim, shape, strides = wrap_ndarray(value)
        return arg._type_(data, ndim, shape, strides)
    if isinstance(value, str): return make_string(value)
    return value


def wrap_function(func, engine):
    args, ret_type = func.type.pointee.args, func.type.pointee.return_type
    cfunc = ctypes.CFUNCTYPE(wrap_type(ret_type), *(wrap_type(arg) for arg in args))(engine.get_function_address(func.name))
    cfunc.__name__ = func.name
    return cfunc


def dispatcher(fn, user_arg_count):
    def call(*args):
        begin_call()
        try:
            error = ctypes.c_int32(0)
            values = [wrap_arg(arg, value) for arg, value in zip(fn._argtypes_[:user_arg_count], args)]
            result = fn(*values, ctypes.byref(error))
            if error.value == ERROR_DIVISION_BY_ZERO: raise ZeroDivisionError('division by zero')
            if error.value == ERROR_RANGE_STEP_ZERO: raise ValueError('range() arg 3 must not be zero')
            if error.value == ERROR_ARRAY_DIMENSION_MISMATCH:
                raise ValueError('array index count does not match array dimensions')
            if error.value == ERROR_INDEX_OUT_OF_BOUNDS: raise IndexError('index out of range')
            if error.value == ERROR_SLICE_STEP_ZERO: raise ValueError('slice step cannot be zero')
            if error.value == ERROR_ORD_LENGTH: raise TypeError('ord() expected a character')
            if error.value == ERROR_CHR_RANGE: raise ValueError('chr() arg not in range(0x110000)')
            return to_python(result) if fn._restype_ == StringPointer else result
        finally:
            end_call()
    call.__name__ = fn.__name__
    return call


def wrap_module(sig, llfunc, engine): return dispatcher(wrap_function(llfunc, engine), len(sig))
