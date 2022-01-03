
import ast
import inspect
import types
from textwrap import dedent

from .ast import (PRIM_OPS, App, Assign, Break, Compare, Const, Fun, If, Index,
                  LitBool, LitFloat, LitInt, Loop, Noop, Prim, Return, Var)
from .types import *

'''
Parse a Python function into pyjiting CoreAST.
'''


def get_type_hint(var):
    if hasattr(var, 'annotation') and hasattr(var.annotation, 'id'):
        ty = var.annotation.id
        if ty == 'int64':
            return int64_t
        elif ty == 'float':
            return double64_t
        elif ty == 'bool':
            return int64_t
        return None
    return None


class ASTVisitor(ast.NodeVisitor):

    def __init__(self):
        pass

    def __call__(self, source):
        if isinstance(source, types.ModuleType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.FunctionType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.LambdaType):
            source = dedent(inspect.getsource(source))
        elif isinstance(source, str):
            source = dedent(source)
        else:
            raise NotImplementedError(ast.dump(source))

        self._source = source
        self._ast = ast.parse(source)
        return self.visit(self._ast)

    def visit_Module(self, node):
        body = list(map(self.visit, node.body))
        return body[0]

    def visit_Name(self, node):
        return Var(node.id)

    def visit_Num(self, node):
        if isinstance(node.n, float):
            return LitFloat(node.n)
        else:
            return LitInt(node.n)

    def visit_Bool(self, node):
        return LitBool(node.n)

    def visit_Call(self, node):
        name = self.visit(node.func)
        args = list(map(self.visit, node.args))
        return App(name, args)

    def visit_BinOp(self, node):
        op_str = node.op.__class__
        a = self.visit(node.left)
        b = self.visit(node.right)
        opname = PRIM_OPS[op_str]
        return Prim(opname, [a, b])

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        var = node.targets[0].id
        value = self.visit(node.value)
        return Assign(var, value, get_type_hint(var))

    def visit_FunctionDef(self, node):
        stmts = list(node.body)
        stmts = list(map(self.visit, stmts))
        args = [Var(a.arg, get_type_hint(a)) for a in node.args.args]
        res = Fun(node.name, args, stmts)
        return res

    def visit_Pass(self, node):
        return Noop()

    def visit_Break(self, node):
        return Break()

    def visit_Return(self, node):
        value = self.visit(node.value)
        return Return(value)

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, bool):
            return LitBool(val)
        elif isinstance(val, int):
            return LitInt(val)
        elif isinstance(val, float):
            return LitFloat(val)
        raise NotImplementedError(node)

    def visit_Attribute(self, node):
        if node.attr == 'shape':
            value = self.visit(node.value)
            return Prim('shape#', [value])
        else:
            raise NotImplementedError(ast.dump(node))

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.slice:
                value = self.visit(node.value)
                ix = self.visit(node.slice)
                return Index(value, ix)
        elif isinstance(node.ctx, ast.Store):
            raise NotImplementedError(ast.dump(node))

    def visit_int(self, node):
        return LitInt(node, type=int64_t)

    def visit_For(self, node):
        target = self.visit(node.target)
        stmts = list(map(self.visit, node.body))
        if node.iter.func.id in ['xrange', 'range']:
            args = list(map(self.visit, node.iter.args))
        else:
            raise RuntimeError('Loop must be over range')

        start = 0
        stop = 0
        step = Const(1)
        if len(args) == 1:   # range(stop)
            start = Const(0)
            stop = args[0]
        elif len(args) == 2:  # range(start,stop)
            start = args[0]
            stop = args[1]
        elif len(args) == 3:  # range(start,stop,step)
            start = args[0]
            stop = args[1]
            step = args[2]
        return Loop(target, start, stop, stmts, step)

    def visit_If(self, node):
        test = self.visit(node.test)
        body = list(map(self.visit, node.body))
        orelse = list(map(self.visit, node.orelse))
        return If(test, body, orelse)

    def visit_Compare(self, node):
        def visit_op(sub_node):
            op_str = sub_node.__class__
            opname = PRIM_OPS[op_str]
            return opname
        left = self.visit(node.left)
        ops = list(map(visit_op, node.ops))
        comparators = list(map(self.visit, node.comparators))
        return Compare(left, ops, comparators)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim('add#', [Var(ref), value]))
        if isinstance(node.op, ast.Mult):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim('mult#', [Var(ref), value]))
        else:
            raise NotImplementedError(ast.dump(node))

    def visit_Constant(self, node):
        return Const(node.value)

    def visit_Expr(self, node):
        return None

    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))
