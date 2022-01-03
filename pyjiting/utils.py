from collections import deque
from typing import Union

from .infer import InferError, InfiniteType
from .types import CoreType, GenericType, BaseType, FuncType, VarType, ftv

### == Constraint Solver ==


def empty() -> dict:
    return {}


def apply(s: dict, t: CoreType) -> CoreType:
    if isinstance(t, BaseType):
        return t
    elif isinstance(t, GenericType):
        return GenericType(apply(s, t.a), apply(s, t.b))
    elif isinstance(t, FuncType):
        args = [apply(s, a) for a in t.args]
        return_type = apply(s, t.return_type)
        return FuncType(args=args, return_type=return_type)
    elif isinstance(t, VarType):
        return s.get(t.s, t)


def apply_list(s: dict, xs: list) -> list:
    return [(apply(s, x), apply(s, y)) for (x, y) in xs]


def unify(x: CoreType, y: CoreType) -> dict:
    if isinstance(x, GenericType) and isinstance(y, GenericType):
        s1 = unify(x.a, y.a)
        s2 = unify(apply(s1, x.b), apply(s1, y.b))
        return compose(s2, s1)
    elif isinstance(x, BaseType) and isinstance(y, BaseType) and (x == y):
        return empty()
    elif isinstance(x, FuncType) and isinstance(y, FuncType):
        if len(x.args) != len(y.args):
            raise RuntimeError('Wrong number of arguments')
        s1 = solve(list(zip(x.args, y.args)))
        s2 = unify(apply(s1, x.return_type), apply(s1, y.return_type))
        return compose(s2, s1)
    elif isinstance(x, VarType):
        return bind(x.s, y)
    elif isinstance(y, VarType):
        return bind(y.s, x)
    else:
        raise InferError(x, y)


def solve(xs: list):
    mgu = empty()
    cs = deque(xs)
    while len(cs):
        (a, b) = cs.pop()
        s = unify(a, b)
        mgu = compose(s, mgu)
        cs = deque(apply_list(s, list(cs)))
    return mgu


def bind(n, x):
    if x == n:
        return empty()
    elif occurs_check(n, x):
        raise InfiniteType(n, x)
    else:
        return dict([(n, x)])


def occurs_check(n, x) -> bool:
    return n in ftv(x)


def union(s1: dict, s2: dict) -> dict:
    nenv = s1.copy()
    nenv.update(s2)
    return nenv


def compose(s1: dict, s2: dict) -> dict:
    s3 = dict((t, apply(s1, u)) for t, u in s2.items())
    return union(s1, s3)
