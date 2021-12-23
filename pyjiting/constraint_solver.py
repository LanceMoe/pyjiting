from collections import deque

from .type_inference import InferError, InfiniteType
from .type_system import TemplateType, BaseType, FuncType, VarType, ftv

### == Constraint Solver ==


def empty():
    return {}


def apply(s, t):
    if isinstance(t, BaseType):
        return t
    elif isinstance(t, TemplateType):
        return TemplateType(apply(s, t.a), apply(s, t.b))
    elif isinstance(t, FuncType):
        argtys = [apply(s, a) for a in t.argtys]
        retty = apply(s, t.retty)
        return FuncType(argtys, retty)
    elif isinstance(t, VarType):
        return s.get(t.s, t)


def apply_list(s, xs):
    return [(apply(s, x), apply(s, y)) for (x, y) in xs]


def unify(x, y):
    if isinstance(x, TemplateType) and isinstance(y, TemplateType):
        s1 = unify(x.a, y.a)
        s2 = unify(apply(s1, x.b), apply(s1, y.b))
        return compose(s2, s1)
    elif isinstance(x, BaseType) and isinstance(y, BaseType) and (x == y):
        return empty()
    elif isinstance(x, FuncType) and isinstance(y, FuncType):
        if len(x.argtys) != len(y.argtys):
            return Exception('Wrong number of arguments')
        s1 = solve(zip(x.argtys, y.argtys))
        s2 = unify(apply(s1, x.retty), apply(s1, y.retty))
        return compose(s2, s1)
    elif isinstance(x, VarType):
        return bind(x.s, y)
    elif isinstance(y, VarType):
        return bind(y.s, x)
    else:
        raise InferError(x, y)


def solve(xs):
    mgu = empty()
    cs = deque(xs)
    while len(cs):
        (a, b) = cs.pop()
        s = unify(a, b)
        mgu = compose(s, mgu)
        cs = deque(apply_list(s, cs))
    return mgu


def bind(n, x):
    if x == n:
        return empty()
    elif occurs_check(n, x):
        raise InfiniteType(n, x)
    else:
        return dict([(n, x)])


def occurs_check(n, x):
    return n in ftv(x)


def union(s1, s2):
    nenv = s1.copy()
    nenv.update(s2)
    return nenv


def compose(s1, s2):
    s3 = dict((t, apply(s1, u)) for t, u in s2.items())
    return union(s1, s3)
