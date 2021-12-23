
import ast
import inspect
import types
from textwrap import dedent

from .core_lang import (PRIM_OPS, App, Assign, Fun, Index, LitBool, LitFloat,
                        LitInt, Loop, Noop, Prim, Return, Var)
from .type_system import *

### == Core Translator ==


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
        return Assign(var, value)

    def visit_FunctionDef(self, node):
        stmts = list(node.body)
        stmts = list(map(self.visit, stmts))
        args = [Var(a.arg) for a in node.args.args]
        res = Fun(node.name, args, stmts)
        return res

    def visit_Pass(self, node):
        return Noop()

    def visit_Return(self, node):
        value = self.visit(node.value)
        return Return(value)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return LitBool(node.value)
        elif isinstance(node.value, int):
            return LitInt(node.value)
        elif isinstance(node.value, float):
            return LitFloat(node.value)
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
        if node.iter.func.id in {'xrange', 'range'}:
            args = list(map(self.visit, node.iter.args))
        else:
            raise Exception('Loop must be over range')

        if len(args) == 1:   # xrange(n)
            return Loop(target, LitInt(0, type=int64_t), args[0], stmts)
        elif len(args) == 2:  # xrange(n,m)
            return Loop(target, args[0], args[1], stmts)

    def visit_If(self, node):
        print('visit_If')
        print(ast.dump(node))
        # target = self.visit(node.target)
        # stmts = list(map(self.visit, node.body))
        # if node.iter.func.id in {'xrange', 'range'}:
        #     args = list(map(self.visit, node.iter.args))
        # else:
        #     raise Exception('Loop must be over range')

        # if len(args) == 1:   # xrange(n)
        #     return Loop(target, LitInt(0, type=int64), args[0], stmts)
        # elif len(args) == 2:  # xrange(n,m)
        #     return Loop(target, args[0], args[1], stmts)
        return node

    def visit_Compare(self, node):
        print('visit_Compare')
        print(ast.dump(node))
        return node

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim('add#', [Var(ref), value]))
        if isinstance(node.op, ast.Mul):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim('mult#', [Var(ref), value]))
        else:
            raise NotImplementedError(ast.dump(node))

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return LitInt(node.value)
        elif isinstance(node.value, float):
            return LitFloat(node.value)
        elif isinstance(node.value, bool):
            return LitBool(node.value)
        raise NotImplementedError(ast.dump(node))

    # def visit_Expr(self, node):
    #     return

    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))
