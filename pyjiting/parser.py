import ast
import inspect
import types
from textwrap import dedent

from . import ast as core
from .errors import CompileError
from .types import bool_t, double64_t, float32_t, int32_t, int64_t


def get_type_hint(annotation):
    """Map supported syntax or evaluated annotations to a Core scalar type."""
    if annotation is None:
        return None
    if isinstance(annotation, str):
        try:
            annotation = ast.parse(annotation, mode='eval').body
        except SyntaxError:
            return None
    if annotation is int:
        return int64_t
    if annotation is float:
        return double64_t
    if annotation is bool:
        return bool_t
    name = getattr(annotation, '__name__', None)
    if name is not None:
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(name)
    if isinstance(annotation, ast.Name):
        return {'int': int64_t, 'int64': int64_t, 'int32': int32_t,
                'float': double64_t, 'float64': double64_t, 'float32': float32_t,
                'bool': bool_t}.get(annotation.id)
    if isinstance(annotation, ast.Attribute):
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(annotation.attr)
    return None


class ASTVisitor(ast.NodeVisitor):
    def __call__(self, source):
        self._evaluated_annotations = {}
        if isinstance(source, (types.FunctionType, types.LambdaType, types.ModuleType)):
            if isinstance(source, (types.FunctionType, types.LambdaType)):
                try:
                    self._evaluated_annotations = inspect.get_annotations(source, eval_str=True)
                except (NameError, TypeError, ValueError):
                    # The source AST remains useful when a forward reference cannot
                    # be resolved in the function's defining module.
                    self._evaluated_annotations = {}
            source = dedent(inspect.getsource(source))
        elif isinstance(source, str):
            source = dedent(source)
        else:
            raise CompileError(f'expected function or source string, got {type(source).__name__}')
        return self.visit(ast.parse(source))

    def visit_Module(self, node):
        functions = [item for item in node.body if isinstance(item, ast.FunctionDef)]
        if len(functions) != 1:
            raise CompileError('source must contain exactly one function', node)
        return self.visit(functions[0])

    def visit_FunctionDef(self, node):
        args = []
        for arg in node.args.args:
            annotation = self._evaluated_annotations.get(arg.arg, arg.annotation)
            hint = get_type_hint(annotation)
            if arg.annotation is not None and hint is None:
                raise CompileError(f'unsupported annotation for {arg.arg}', arg.annotation)
            args.append(core.Var(arg.arg, hint, arg))
        return_annotation = self._evaluated_annotations.get('return', node.returns)
        hint = get_type_hint(return_annotation)
        if node.returns is not None and hint is None:
            raise CompileError('unsupported return annotation', node.returns)
        return core.Fun(node.name, args, [self.visit(stmt) for stmt in node.body], hint, node)

    def visit_Name(self, node): return core.Var(node.id, source=node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool): return core.LitBool(node.value, node)
        if isinstance(node.value, int): return core.LitInt(node.value, node)
        if isinstance(node.value, float): return core.LitFloat(node.value, node)
        raise CompileError(f'unsupported constant {node.value!r}', node)

    def visit_Return(self, node):
        if node.value is None: raise CompileError('bare return is not supported', node)
        return core.Return(self.visit(node.value), node)

    def _assign_target(self, target, value, annotation, node):
        if isinstance(target, ast.Name): return core.Assign(target.id, value, annotation, node)
        if isinstance(target, ast.Subscript):
            base, indices = self._subscript_parts(target)
            return core.StoreIndex(base, indices, value, node)
        raise CompileError('unsupported assignment target', target)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        if len(node.targets) == 1: return self._assign_target(node.targets[0], value, None, node)
        temp = '__assign_tmp'
        return [core.Assign(temp, value, source=node)] + [self._assign_target(target, core.Var(temp, source=node), None, node) for target in node.targets]

    def visit_AnnAssign(self, node):
        if node.value is None: raise CompileError('annotation without a value is not supported', node)
        hint = get_type_hint(node.annotation)
        if hint is None: raise CompileError('unsupported variable annotation', node.annotation)
        return self._assign_target(node.target, self.visit(node.value), hint, node)

    def visit_AugAssign(self, node):
        opname = core.PRIM_OPS.get(type(node.op))
        if opname is None: raise CompileError('unsupported augmented assignment operator', node)
        if isinstance(node.target, ast.Name):
            ref = core.Var(node.target.id, source=node.target)
            return core.Assign(node.target.id, core.Prim(opname, [ref, self.visit(node.value)], node), source=node)
        if isinstance(node.target, ast.Subscript):
            value, indices = self._subscript_parts(node.target)
            return core.AugStoreIndex(value, indices, opname, self.visit(node.value), node)
        raise CompileError('unsupported augmented assignment target', node.target)

    def visit_BinOp(self, node):
        opname = core.PRIM_OPS.get(type(node.op))
        if opname is None: raise CompileError('unsupported binary operator', node)
        return core.Prim(opname, [self.visit(node.left), self.visit(node.right)], node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub): return core.Prim('neg#', [self.visit(node.operand)], node)
        if isinstance(node.op, ast.Not): return core.Prim('not#', [self.visit(node.operand)], node)
        raise CompileError('unsupported unary operator', node)

    def visit_BoolOp(self, node):
        opname = 'and#' if isinstance(node.op, ast.And) else 'or#'
        value = self.visit(node.values[0])
        for next_value in node.values[1:]: value = core.Prim(opname, [value, self.visit(next_value)], node)
        return value

    def visit_Compare(self, node):
        ops = [core.PRIM_OPS.get(type(op)) for op in node.ops]
        if any(op is None for op in ops): raise CompileError('unsupported comparison operator', node)
        return core.Compare(self.visit(node.left), ops, [self.visit(value) for value in node.comparators], node)

    def visit_If(self, node): return core.If(self.visit(node.test), [self.visit(x) for x in node.body], [self.visit(x) for x in node.orelse], node)
    def visit_While(self, node): return core.While(self.visit(node.test), [self.visit(x) for x in node.body], [self.visit(x) for x in node.orelse], node)

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id not in ('range', 'xrange'):
            raise CompileError('for loops must iterate a named variable over range()', node)
        args = [self.visit(arg) for arg in node.iter.args]
        if not 1 <= len(args) <= 3: raise CompileError('range() expects one to three arguments', node.iter)
        begin, end, step = core.LitInt(0, node), args[0], core.LitInt(1, node)
        if len(args) >= 2: begin, end = args[0], args[1]
        if len(args) == 3: step = args[2]
        return core.Loop(core.Var(node.target.id, source=node.target), begin, end, [self.visit(x) for x in node.body], step, [self.visit(x) for x in node.orelse], node)

    def visit_Break(self, node): return core.Break(node)
    def visit_Continue(self, node): return core.Continue(node)
    def visit_Pass(self, node): return core.Noop(node)
    def visit_Expr(self, node): return core.Expr(self.visit(node.value), node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name): raise CompileError('only calls to named functions are supported', node)
        return core.CallFunc(self.visit(node.func), [self.visit(arg) for arg in node.args], node)

    def visit_Attribute(self, node):
        if node.attr == 'shape': return core.Prim('shape#', [self.visit(node.value)], node)
        raise CompileError('unsupported attribute access', node)

    def _subscript_parts(self, node):
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return self.visit(node.value), [self.visit(element) for element in elements]

    def visit_Subscript(self, node):
        value, indices = self._subscript_parts(node)
        return core.Index(value, indices, node)

    def generic_visit(self, node): raise CompileError(f'unsupported syntax: {type(node).__name__}', node)
