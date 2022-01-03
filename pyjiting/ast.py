import ast

### == Core Language ==


class Var(ast.AST):
    _fields = ['id', 'type']

    def __init__(self, id, type=None):
        self.id = id
        self.type = type


class Assign(ast.AST):
    _fields = ['ref', 'value', 'type']

    def __init__(self, ref, value, type=None):
        self.ref = ref
        self.value = value
        self.type = type


class Return(ast.AST):
    _fields = ['value']

    def __init__(self, value):
        self.value = value


class Loop(ast.AST):
    _fields = ['var', 'begin', 'end', 'body']

    def __init__(self, var, begin, end, body):
        self.var = var
        self.begin = begin
        self.end = end
        self.body = body


class If(ast.AST):
    _fields = ['test', 'body', 'orelse']

    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse


class Compare(ast.AST):
    _fields = ['left', 'ops', 'comparators']

    def __init__(self, left, ops, comparators):
        self.left = left
        self.ops = ops
        self.comparators = comparators


class App(ast.AST):
    _fields = ['fn', 'args']

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args


class Fun(ast.AST):
    _fields = ['fname', 'args', 'body']

    def __init__(self, fname, args, body):
        self.fname = fname
        self.args = args
        self.body = body


class LitInt(ast.AST):
    _fields = ['n']

    def __init__(self, n, type=None):
        self.n = int(n)
        self.type = type


class LitFloat(ast.AST):
    _fields = ['n']

    def __init__(self, n, type=None):
        self.n = float(n)
        self.type = None


class LitBool(ast.AST):
    _fields = ['n']

    def __init__(self, n):
        self.n = bool(n)


class Prim(ast.AST):
    _fields = ['fn', 'args']

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args


class Const(ast.AST):
    _fields = ['value']

    def __init__(self, value):
        self.value = value


class Index(ast.AST):
    _fields = ['value', 'ix']

    def __init__(self, value, ix):
        self.value = value
        self.ix = ix


class Noop(ast.AST):
    _fields = []

class Break(ast.AST):
    _fields = []


PRIM_OPS = {
    ast.Add: 'add#',
    ast.Mult: 'mult#',
    ast.Sub: 'sub#',
    ast.Div: 'div#',
    ast.Pow: 'pow#',
    ast.Mod: 'mod#',
    ast.And: 'and#',
    ast.Or: 'or#',
    ast.Eq: 'eq#',
    ast.NotEq: 'ne#',
    ast.Lt: 'lt#',
    ast.LtE: 'le#',
    ast.Gt: 'gt#',
    ast.GtE: 'ge#'
}

LLVM_PRIM_OPS = list(PRIM_OPS.values())
