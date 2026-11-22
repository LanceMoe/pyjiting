from functools import reduce
from typing import Any, Union

from llvmlite.ir.types import FunctionType, PointerType, Type

'''
Basic types and type constructors
'''


class VarType(Type):
    def __init__(self, s):
        self.s = s

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if isinstance(other, VarType):
            return (self.s == other.s)
        else:
            return False

    def __str__(self):
        return self.s


class BaseType(Type):
    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        if isinstance(other, BaseType):
            return (self.s == other.s)
        else:
            return False

    def __hash__(self):
        return hash(self.s)

    def __str__(self):
        return self.s


class GenericType(Type):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if isinstance(other, GenericType):
            return (self.a == other.a) & (self.b == other.b)
        else:
            return False

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return str(self.a) + ' ' + str(self.b)


class FuncType(FunctionType):
    def __str__(self):
        return str(self.args) + ' -> ' + str(self.return_type)


class TupleType(Type):
    def __init__(self, elements):
        self.elements = tuple(elements)

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elements == other.elements

    def __hash__(self):
        return hash(('Tuple', self.elements))

    def __str__(self):
        return 'Tuple[' + ', '.join(map(str, self.elements)) + ']'


CoreType = Union[GenericType, BaseType, FuncType, TupleType, VarType]

int32_t = BaseType('Int32')
int64_t = BaseType('Int64')
bool_t = BaseType('Bool')
float32_t = BaseType('Float')
double64_t = BaseType('Double')
void_t = BaseType('Void')
array_t = BaseType('Array')
shape_t = BaseType('Shape')
str_t = BaseType('String')
list_t = BaseType('List')
set_t = BaseType('Set')
frozenset_t = BaseType('FrozenSet')
dict_t = BaseType('Dict')
tuple_t = BaseType('PythonTuple')
complex_t = BaseType('Complex')
bytes_t = BaseType('Bytes')
bytearray_t = BaseType('ByteArray')
memoryview_t = BaseType('MemoryView')
range_t = BaseType('Range')
slice_t = BaseType('Slice')
none_t = BaseType('None')
object_t = BaseType('Object')

BOXED_TYPES = {
    list: list_t, set: set_t, frozenset: frozenset_t, dict: dict_t, tuple: tuple_t,
    complex: complex_t, bytes: bytes_t, bytearray: bytearray_t, memoryview: memoryview_t,
    range: range_t, slice: slice_t, type(None): none_t,
}
PYTHON_TYPES = {
    **{core_type: python_type for python_type, core_type in BOXED_TYPES.items()},
    str_t: str, bool_t: bool, int32_t: int, int64_t: int, float32_t: float, double64_t: float,
    object_t: object,
}
BOXED_CORE_TYPES = frozenset(BOXED_TYPES.values()) | {object_t}

ptr_t = PointerType


def make_array_type(t): return GenericType(array_t, t)


int32_array_t = make_array_type(int32_t)
int64_array_t = make_array_type(int64_t)
double64_array_t = make_array_type(double64_t)
float32_array_t = make_array_type(float32_t)


def ftv(x) -> set:
    # ftv: free type variables
    if isinstance(x, BaseType):
        return set()
    elif isinstance(x, GenericType):
        return ftv(x.a) | ftv(x.b)
    elif isinstance(x, FuncType):
        return reduce(set.union, set(map(ftv, x.args))) | ftv(x.return_type)
    elif isinstance(x, TupleType):
        return reduce(set.union, map(ftv, x.elements), set())
    elif isinstance(x, VarType):
        return set([x])
    raise TypeError(f'unsupported core type: {type(x).__name__}')


def is_array(ty: Union[GenericType, Any]) -> bool:
    return isinstance(ty, GenericType) and ty.a == array_t


def is_tuple(ty):
    return isinstance(ty, TupleType)


def contains_array(ty):
    """Return whether a type contains an ndarray at any structural depth."""
    return is_array(ty) or (is_tuple(ty) and any(contains_array(element) for element in ty.elements))


numeric_types = {bool_t, int32_t, int64_t, float32_t, double64_t}
integer_types = {bool_t, int32_t, int64_t}
float_types = {float32_t, double64_t}


def is_numeric(ty):
    return ty in numeric_types


def is_integer(ty):
    return ty in integer_types


def is_float(ty):
    return ty in float_types


def is_string(ty):
    return ty == str_t


def is_boxed(ty):
    return ty in BOXED_CORE_TYPES


def is_boxable(ty):
    return is_boxed(ty) or is_numeric(ty) or is_string(ty) or (
        is_tuple(ty) and all(is_boxable(element) for element in ty.elements))


def is_truthy_type(ty):
    return is_numeric(ty) or is_string(ty) or is_tuple(ty) or is_boxed(ty)


def promote_numeric(left, right):
    """Return the language-level common numeric type for two scalar types."""
    if not is_numeric(left) or not is_numeric(right):
        return None
    if left == double64_t or right == double64_t:
        return double64_t
    if left == float32_t or right == float32_t:
        return double64_t if left == int64_t or right == int64_t else float32_t
    if left == int64_t or right == int64_t:
        return int64_t
    if left == int32_t or right == int32_t:
        return int32_t
    return int64_t


def can_widen(actual, expected):
    if actual == expected:
        return True
    if actual == bool_t and expected in (int32_t, int64_t, float32_t, double64_t):
        return True
    if actual == int32_t and expected in (int64_t, float32_t, double64_t):
        return True
    if actual == int64_t and expected == double64_t:
        return True
    if actual == float32_t and expected == double64_t:
        return True
    return False
