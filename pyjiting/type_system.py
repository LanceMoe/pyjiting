
from functools import reduce
from typing import Any, Union

### == Type System ==


class VarType(object):
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


class BaseType(object):
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


class GenericType(object):
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


class FuncType(object):
    def __init__(self, argtys, retty):
        assert isinstance(argtys, list)
        self.argtys = argtys
        self.retty = retty

    def __eq__(self, other):
        if isinstance(other, FuncType):
            return (self.argtys == other.argtys) & (self.retty == other.retty)
        else:
            return False

    def __str__(self):
        return str(self.argtys) + ' -> ' + str(self.retty)


CoreType = Union[GenericType, BaseType, FuncType, VarType]

int32_t = BaseType('Int32')
int64_t = BaseType('Int64')
float32_t = BaseType('Float')
double64_t = BaseType('Double')
void_t = BaseType('Void')
array_t = BaseType('Array')

def make_array_type(t): return GenericType(array_t, t)

int32_array_t = make_array_type(int32_t)
int64_array_t = make_array_type(int64_t)
double64_array_t = make_array_type(double64_t)


def ftv(x) -> set:
    # ftv: free type variables
    if isinstance(x, BaseType):
        return set()
    elif isinstance(x, GenericType):
        return ftv(x.a) | ftv(x.b)
    elif isinstance(x, FuncType):
        return reduce(set.union, set(map(ftv, x.argtys))) | ftv(x.retty)
    elif isinstance(x, VarType):
        return set([x])


def is_array(ty: Union[GenericType, Any]) -> bool:
    return isinstance(ty, GenericType) and ty.a == array_t
