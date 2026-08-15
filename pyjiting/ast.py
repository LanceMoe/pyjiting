import ast
import copy


class Node(ast.AST):
    _fields = ()

    def __init__(self, source=None):
        if source is not None:
            for name in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                if hasattr(source, name):
                    setattr(self, name, getattr(source, name))
        self.type = None

    def __deepcopy__(self, memo):
        cloned = type(self).__new__(type(self))
        memo[id(self)] = cloned
        for name, value in self.__dict__.items():
            setattr(cloned, name, value if name == 'namespace' else copy.deepcopy(value, memo))
        return cloned


class Var(Node):
    _fields = ('id',)
    def __init__(self, id, annotation=None, source=None):
        super().__init__(source); self.id = id; self.annotation = annotation


class Assign(Node):
    _fields = ('ref', 'value')
    def __init__(self, ref, value, annotation=None, source=None):
        super().__init__(source); self.ref = ref; self.value = value; self.annotation = annotation


class UnpackAssign(Node):
    _fields = ('refs', 'value')
    def __init__(self, refs, value, source=None):
        super().__init__(source); self.refs, self.value = refs, value


class StoreIndex(Node):
    _fields = ('value', 'indices', 'rhs')
    def __init__(self, value, indices, rhs, source=None):
        super().__init__(source); self.value = value; self.indices = indices; self.rhs = rhs


class AugStoreIndex(Node):
    _fields = ('value', 'indices', 'fn', 'rhs')
    def __init__(self, value, indices, fn, rhs, source=None):
        super().__init__(source); self.value, self.indices, self.fn, self.rhs = value, indices, fn, rhs
        self.operand_type = None


class Return(Node):
    _fields = ('value',)
    def __init__(self, value, source=None): super().__init__(source); self.value = value


class Loop(Node):
    _fields = ('var', 'begin', 'end', 'body', 'step', 'orelse')
    def __init__(self, var, begin, end, body, step, orelse=None, source=None):
        super().__init__(source); self.var, self.begin, self.end = var, begin, end; self.body, self.step, self.orelse = body, step, orelse or []


class ForEach(Node):
    _fields = ('var', 'iterable', 'body', 'orelse')
    def __init__(self, var, iterable, body, orelse=None, source=None):
        super().__init__(source); self.var, self.iterable = var, iterable; self.body, self.orelse = body, orelse or []


class If(Node):
    _fields = ('test', 'body', 'orelse')
    def __init__(self, test, body, orelse, source=None): super().__init__(source); self.test, self.body, self.orelse = test, body, orelse


class While(If):
    pass


class Compare(Node):
    _fields = ('left', 'ops', 'comparators')
    def __init__(self, left, ops, comparators, source=None): super().__init__(source); self.left, self.ops, self.comparators = left, ops, comparators


class CallFunc(Node):
    _fields = ('fn', 'args')
    def __init__(self, fn, args, source=None): super().__init__(source); self.fn, self.args = fn, args


class Fun(Node):
    _fields = ('fname', 'args', 'body')
    def __init__(self, fname, args, body, return_annotation=None, source=None):
        super().__init__(); self.fname, self.args, self.body, self.return_annotation = fname, args, body, return_annotation
        self.symbol = fname
        if source is not None:
            for name in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                if hasattr(source, name):
                    setattr(self, name, getattr(source, name))


class LitInt(Node):
    _fields = ('n',)
    def __init__(self, n, source=None, literal_type=None):
        super().__init__(source); self.n, self.literal_type = int(n), literal_type


class LitFloat(Node):
    _fields = ('n',)
    def __init__(self, n, source=None, literal_type=None):
        super().__init__(source); self.n, self.literal_type = float(n), literal_type


class LitBool(Node):
    _fields = ('n',)
    def __init__(self, n, source=None): super().__init__(source); self.n = bool(n)


class LitStr(Node):
    _fields = ('value',)
    def __init__(self, value, source=None): super().__init__(source); self.value = str(value)


class LitTuple(Node):
    _fields = ('elements',)
    def __init__(self, elements, source=None): super().__init__(source); self.elements = elements


class Prim(Node):
    _fields = ('fn', 'args')
    def __init__(self, fn, args, source=None): super().__init__(source); self.fn, self.args, self.operand_type = fn, args, None


class Index(Node):
    _fields = ('value', 'indices')
    def __init__(self, value, indices, source=None): super().__init__(source); self.value = value; self.indices = indices if isinstance(indices, list) else [indices]


class Slice(Node):
    _fields = ('lower', 'upper', 'step')
    def __init__(self, lower, upper, step, source=None):
        super().__init__(source); self.lower, self.upper, self.step = lower, upper, step


class Expr(Node):
    _fields = ('value',)
    def __init__(self, value, source=None): super().__init__(source); self.value = value


class Noop(Node): pass
class Break(Node): pass
class Continue(Node): pass


def integer_constant_value(node):
    if isinstance(node, LitInt):
        return node.n
    if isinstance(node, Prim) and node.fn == 'neg#' and len(node.args) == 1:
        value = integer_constant_value(node.args[0])
        return -value if value is not None else None
    return None


PRIM_OPS = {
    ast.Add: 'add#', ast.Sub: 'sub#', ast.Mult: 'mult#', ast.Div: 'div#', ast.FloorDiv: 'floordiv#', ast.Mod: 'mod#', ast.Pow: 'pow#',
    ast.Eq: 'eq#', ast.NotEq: 'ne#', ast.Lt: 'lt#', ast.LtE: 'le#', ast.Gt: 'gt#', ast.GtE: 'ge#',
    ast.In: 'in#', ast.NotIn: 'notin#',
}
ARITHMETIC_OPS = {'add#', 'sub#', 'mult#', 'div#', 'floordiv#', 'mod#', 'pow#'}
COMPARISON_OPS = {'eq#', 'ne#', 'lt#', 'le#', 'gt#', 'ge#'}
MEMBERSHIP_OPS = {'in#', 'notin#'}
