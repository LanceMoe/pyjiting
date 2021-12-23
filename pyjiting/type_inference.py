
import ast
import string

from .core_lang import LLVM_PRIM_OPS
from .type_system import *

### == Type Inference ==


def naming():
    k = 0
    while True:
        for a in string.ascii_lowercase:
            yield f'\{a}{str(k)}' if (k > 0) else a
        k = k+1


class TypeInferencer:

    def __init__(self):
        self.constraints = []
        self.env = {}
        self.names = naming()

    def fresh(self):
        return VarType('$' + next(self.names))  # New meta type variable.

    def visit(self, node):
        name = f'visit_{type(node).__name__}'
        print(name)
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

    def visit_Fun(self, node):
        self.argtys = [self.fresh() for v in node.args]
        self.retty = VarType('$retty')
        for (arg, ty) in zip(node.args, self.argtys):
            arg.type = ty
            self.env[arg.id] = ty
        list(map(self.visit, node.body))
        return FuncType(self.argtys, self.retty)

    def visit_NoneType(self, node):
        return None

    def visit_Noop(self, node):
        return None

    def visit_If(self, node):
        list(map(self.visit, node.body))

    def visit_LitInt(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_LitFloat(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_Assign(self, node):
        ty = self.visit(node.value)
        if node.ref in self.env:
            # Subsequent uses of a variable must have the same type.
            self.constraints += [(ty, self.env[node.ref])]
        self.env[node.ref] = ty
        node.type = ty
        return None

    def visit_Index(self, node):
        tv = self.fresh()
        ty = self.visit(node.value)
        ixty = self.visit(node.ix)
        self.constraints += [(ty, make_array_type(tv)), (ixty, int64_t)]
        return tv

    def visit_Prim(self, node):
        if node.fn == 'shape#':
            return make_array_type(int64_t)
        elif node.fn in LLVM_PRIM_OPS:
            tya = self.visit(node.args[0])
            tyb = self.visit(node.args[1])
            self.constraints += [(tya, tyb)]
            return tyb
        else:
            raise NotImplementedError(ast.dump(node))

    def visit_Var(self, node):
        ty = self.env[node.id]
        node.type = ty
        print(ast.dump(node))
        return ty

    def visit_Return(self, node):
        ty = self.visit(node.value)
        self.constraints += [(ty, self.retty)]

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return int64_t
        elif isinstance(node.value, float):
            return float32_t
        raise NotImplementedError(node.value)

    def visit_Loop(self, node):
        self.env[node.var.id] = int64_t
        varty = self.visit(node.var)
        begin = self.visit(node.begin)
        end = self.visit(node.end)
        self.constraints += [(varty, int64_t), (
            begin, int64_t), (end, int64_t)]
        list(map(self.visit, node.body))

    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))


class UnderDetermined(Exception):
    def __str__(self):
        return 'The types in the function are not fully determined by the input types. Add annotations.'


class InferError(Exception):
    def __init__(self, ty1, ty2):
        self.ty1 = ty1
        self.ty2 = ty2

    def __str__(self):
        return '\n'.join([
            'Type mismatch: ',
            'Given: ', '\t' + str(self.ty1),
            'Expected: ', '\t' + str(self.ty2)
        ])


class InfiniteType(Exception):
    def __init__(self, ty1, ty2):
        self.ty1 = ty1
        self.ty2 = ty2

    def __str__(self):
        return '\n'.join([
            'Type mismatch: ',
            'Given: ', '\t' + str(self.ty1),
            'Expected: ', '\t' + str(self.ty2)
        ])