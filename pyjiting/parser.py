import ast
import inspect
import math
import types
from textwrap import dedent

from . import ast as core
from .errors import CompileError
from .intrinsics import MATH_CONSTANTS, MATH_FUNCTIONS, STRING_METHODS
from .types import bool_t, double64_t, float32_t, int32_t, int64_t, str_t


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
    if annotation is str:
        return str_t
    name = getattr(annotation, '__name__', None)
    if name is not None:
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(name)
    if isinstance(annotation, ast.Name):
        return {'int': int64_t, 'int64': int64_t, 'int32': int32_t,
                'float': double64_t, 'float64': double64_t, 'float32': float32_t,
                'bool': bool_t, 'str': str_t}.get(annotation.id)
    if isinstance(annotation, ast.Attribute):
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(annotation.attr)
    return None


def is_dynamic_array_annotation(annotation):
    """Return whether an annotation deliberately defers ndarray dtype to a call."""
    if isinstance(annotation, str):
        try:
            annotation = ast.parse(annotation, mode='eval').body
        except SyntaxError:
            return False
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == 'ndarray'
    return getattr(annotation, '__name__', None) == 'ndarray'


class ASTVisitor(ast.NodeVisitor):
    def __call__(self, source):
        self._evaluated_annotations = {}
        self._constants = {}
        self._local_names = set()
        self._loop_depth = 0
        if isinstance(source, (types.FunctionType, types.LambdaType, types.ModuleType)):
            if isinstance(source, (types.FunctionType, types.LambdaType)):
                try:
                    self._evaluated_annotations = inspect.get_annotations(source, eval_str=True)
                except (NameError, TypeError, ValueError):
                    # The source AST remains useful when a forward reference cannot
                    # be resolved in the function's defining module.
                    self._evaluated_annotations = {}
                closure = inspect.getclosurevars(source)
                self._constants = {**closure.globals, **closure.nonlocals}
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
        if node.args.defaults or node.args.kw_defaults or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise CompileError('default, keyword-only, and variadic parameters are not supported', node.args)
        args = []
        self._local_names = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args]}
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                self._local_names.add(item.id)
        for arg in [*node.args.posonlyargs, *node.args.args]:
            annotation = self._evaluated_annotations.get(arg.arg, arg.annotation)
            hint = get_type_hint(annotation)
            if arg.annotation is not None and hint is None and not is_dynamic_array_annotation(annotation):
                raise CompileError(f'unsupported annotation for {arg.arg}', arg.annotation)
            args.append(core.Var(arg.arg, hint, arg))
        return_annotation = self._evaluated_annotations.get('return', node.returns)
        hint = get_type_hint(return_annotation)
        if node.returns is not None and hint is None:
            raise CompileError('unsupported return annotation', node.returns)
        return core.Fun(node.name, args, [self.visit(stmt) for stmt in node.body], hint, node)

    def _literal(self, value, node):
        try:
            import numpy as np
            value_type = type(value)
            if value_type is np.bool_: return core.LitBool(value, node)
            if value_type is np.int32: return core.LitInt(value, node, int32_t)
            if value_type is np.int64: return core.LitInt(value, node, int64_t)
            if value_type is np.float32: return core.LitFloat(value, node, float32_t)
            if value_type is np.float64: return core.LitFloat(value, node, double64_t)
        except ImportError:
            pass
        if isinstance(value, bool): return core.LitBool(value, node)
        if isinstance(value, int):
            if not -(1 << 63) <= value < (1 << 63):
                raise CompileError('integer constant is outside the supported Int64 range', node)
            return core.LitInt(value, node)
        if isinstance(value, float): return core.LitFloat(value, node)
        if isinstance(value, str): return core.LitStr(value, node)
        raise CompileError(f'global {getattr(node, "id", "value")!r} is not an immutable supported constant', node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self._local_names and node.id in self._constants:
            return self._literal(self._constants[node.id], node)
        return core.Var(node.id, source=node)

    def visit_Constant(self, node):
        return self._literal(node.value, node)

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

    def _visit_loop_body(self, statements):
        self._loop_depth += 1
        try:
            return [self.visit(statement) for statement in statements]
        finally:
            self._loop_depth -= 1

    def visit_While(self, node): return core.While(self.visit(node.test), self._visit_loop_body(node.body), [self.visit(x) for x in node.orelse], node)

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise CompileError('for loop targets must be named variables', node.target)
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id in ('range', 'xrange')):
            return core.ForEach(core.Var(node.target.id, source=node.target), self.visit(node.iter),
                                self._visit_loop_body(node.body), [self.visit(x) for x in node.orelse], node)
        args = [self.visit(arg) for arg in node.iter.args]
        if not 1 <= len(args) <= 3: raise CompileError('range() expects one to three arguments', node.iter)
        begin, end, step = core.LitInt(0, node), args[0], core.LitInt(1, node)
        if len(args) >= 2: begin, end = args[0], args[1]
        if len(args) == 3: step = args[2]
        return core.Loop(core.Var(node.target.id, source=node.target), begin, end, self._visit_loop_body(node.body), step, [self.visit(x) for x in node.orelse], node)

    def visit_Break(self, node):
        if self._loop_depth == 0: raise CompileError('break outside loop', node)
        return core.Break(node)

    def visit_Continue(self, node):
        if self._loop_depth == 0: raise CompileError('continue outside loop', node)
        return core.Continue(node)
    def visit_Pass(self, node): return core.Noop(node)
    def visit_Expr(self, node): return core.Expr(self.visit(node.value), node)

    def visit_Call(self, node):
        if node.keywords:
            raise CompileError('keyword arguments are not supported', node)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                module_value = self._constants.get(module_name, math if module_name == 'math' else None)
            else:
                module_value = None
            if module_value is math and module_name not in self._local_names:
                if node.func.attr not in MATH_FUNCTIONS:
                    raise CompileError(f'unsupported math function {node.func.attr!r}', node.func)
                return core.CallFunc(core.Var(f'math.{node.func.attr}', source=node.func),
                                     [self.visit(arg) for arg in node.args], node)
            if node.func.attr not in STRING_METHODS:
                raise CompileError(f'unsupported method {node.func.attr!r}', node.func)
            return core.CallFunc(core.Var(f'str.{node.func.attr}', source=node.func),
                                 [self.visit(node.func.value), *[self.visit(arg) for arg in node.args]], node)
        if not isinstance(node.func, ast.Name): raise CompileError('only calls to named functions are supported', node)
        return core.CallFunc(core.Var(node.func.id, source=node.func),
                             [self.visit(arg) for arg in node.args], node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            module_name = node.value.id
            module_value = self._constants.get(module_name, math if module_name == 'math' else None)
            if module_value is math and module_name not in self._local_names and node.attr in MATH_CONSTANTS:
                return core.LitFloat(getattr(math, node.attr), node, double64_t)
        if node.attr == 'shape': return core.Prim('shape#', [self.visit(node.value)], node)
        raise CompileError('unsupported attribute access', node)

    def _subscript_parts(self, node):
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return self.visit(node.value), [self.visit(element) for element in elements]

    def visit_Slice(self, node):
        return core.Slice(self.visit(node.lower) if node.lower else None,
                          self.visit(node.upper) if node.upper else None,
                          self.visit(node.step) if node.step else None, node)

    def visit_Subscript(self, node):
        value, indices = self._subscript_parts(node)
        return core.Index(value, indices, node)

    def generic_visit(self, node): raise CompileError(f'unsupported syntax: {type(node).__name__}', node)
