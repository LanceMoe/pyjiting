import ast
import builtins
import inspect
import math
import types
import typing
from collections.abc import Callable
from textwrap import dedent

from . import ast as core
from .errors import CompileError
from .intrinsics import (DICT_METHODS, FRAME_BUILTINS, FUNCTION_INTRINSICS, LIST_METHODS,
                         MATH_CONSTANTS, MATH_FUNCTIONS, OBJECT_BUILTINS,
                         OBJECT_CLASS_METHODS, OBJECT_CONSTRUCTORS, OBJECT_METHODS,
                         SET_METHODS)
from .types import (BOXED_TYPES, TupleType, bool_t, dict_t, double64_t, float32_t, frozenset_t,
                    int32_t, int64_t, list_t, none_t, object_t, set_t, str_t, tuple_t, void_t)


_CONTAINER_HINTS = {
    **{kind.__name__: ty for kind, ty in BOXED_TYPES.items()},
    'List': list_t, 'Set': set_t, 'FrozenSet': frozenset_t, 'Dict': dict_t, 'Tuple': tuple_t,
}
_BOXED_METHODS = LIST_METHODS | SET_METHODS | DICT_METHODS
_TYPE_NAMES = {kind.__name__: kind for kind in (*BOXED_TYPES, int, float, bool, str, object)}
_PYTHON_STATEMENTS = (
    ast.Assert, ast.AsyncFor, ast.AsyncFunctionDef, ast.AsyncWith, ast.Await,
    ast.ClassDef, ast.FunctionDef, ast.GeneratorExp, ast.Global, ast.Import,
    ast.ImportFrom, ast.Lambda, ast.Match, ast.NamedExpr, ast.Nonlocal, ast.Raise, ast.Try,
    ast.TryStar, ast.TypeAlias, ast.With, ast.Yield, ast.YieldFrom,
)


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
    if annotation is object or annotation is typing.Any:
        return object_t
    origin = typing.get_origin(annotation)
    for python_type, container_type in BOXED_TYPES.items():
        if annotation is python_type or origin is python_type and origin is not tuple:
            return container_type
    if origin in (types.UnionType, typing.Union):
        return object_t if all(get_type_hint(member) is not None for member in typing.get_args(annotation)) else None
    if annotation is Callable or origin is Callable:
        return object_t
    if origin is tuple:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple_t if get_type_hint(args[0]) is not None else None
        elements = [get_type_hint(arg) for arg in args]
        return TupleType(elements) if all(element is not None for element in elements) else None
    name = getattr(annotation, '__name__', None)
    if name is not None:
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(name)
    if isinstance(annotation, ast.Name):
        return {'int': int64_t, 'int64': int64_t, 'int32': int32_t,
                'float': double64_t, 'float64': double64_t, 'float32': float32_t,
                'bool': bool_t, 'str': str_t, 'None': none_t, **_CONTAINER_HINTS,
                'Callable': object_t, 'Any': object_t, 'object': object_t}.get(annotation.id)
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return none_t
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return object_t if all(get_type_hint(member) is not None for member in
                               (annotation.left, annotation.right)) else None
    if isinstance(annotation, ast.Attribute):
        if isinstance(annotation.value, ast.Name) and annotation.value.id == 'typing':
            return _CONTAINER_HINTS.get(annotation.attr, object_t if annotation.attr == 'Any' else None)
        return {'int32': int32_t, 'int64': int64_t, 'float32': float32_t,
                'float64': double64_t, 'bool_': bool_t}.get(annotation.attr)
    if isinstance(annotation, ast.Subscript):
        name = getattr(annotation.value, 'id', getattr(annotation.value, 'attr', None))
        members = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if name in ('Optional', 'Union'):
            return object_t if all(get_type_hint(member) is not None for member in members) else None
        container = get_type_hint(annotation.value)
        if container == tuple_t:
            if len(members) == 2 and isinstance(members[1], ast.Constant) and members[1].value is Ellipsis:
                return tuple_t if get_type_hint(members[0]) is not None else None
            hints = [get_type_hint(member) for member in members]
            return TupleType(hints) if all(hint is not None for hint in hints) else None
        if container in (list_t, set_t, frozenset_t, dict_t):
            return container
    if isinstance(annotation, ast.Subscript) and (
        isinstance(annotation.value, ast.Name) and annotation.value.id == 'Callable'
        or isinstance(annotation.value, ast.Attribute) and annotation.value.attr == 'Callable'
    ):
        return object_t
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
    def __call__(self, source, *, namespace=None):
        self._evaluated_annotations = {}
        self._constants = {}
        self._bindings = {}
        self._local_names = set()
        self._comp_locals = {}
        self._temporary_count = 0
        self._loop_depth = 0
        self._closure_cells = {}
        self._namespace = namespace if namespace is not None else {}
        if isinstance(source, (types.FunctionType, types.LambdaType, types.ModuleType)):
            if isinstance(source, (types.FunctionType, types.LambdaType)):
                try:
                    self._evaluated_annotations = inspect.get_annotations(source, eval_str=True)
                except (NameError, TypeError, ValueError):
                    # The source AST remains useful when a forward reference cannot
                    # be resolved in the function's defining module.
                    self._evaluated_annotations = {}
                self._closure_cells = dict(zip(source.__code__.co_freevars, source.__closure__ or ()))
                self._namespace = source.__globals__
                self._bindings = {name: self._namespace[name] for name in source.__code__.co_names
                                  if name in self._namespace}
                for name, cell in self._closure_cells.items():
                    try:
                        self._bindings[name] = cell.cell_contents
                    except ValueError:
                        # A method's __class__ cell is empty until the class is created.
                        pass
                self._constants = self._bindings.copy()
            try:
                source = dedent(inspect.getsource(source))
            except (OSError, TypeError) as error:
                name = getattr(source, '__qualname__', repr(source))
                raise CompileError(
                    f'cannot retrieve source for {name!r}; use jit.from_source(...) '
                    'for dynamic functions') from error
        elif isinstance(source, str):
            source = dedent(source)
        else:
            raise CompileError(f'expected function or source string, got {type(source).__name__}')
        return self.visit(ast.parse(source))

    def visit_Module(self, node):
        functions = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            raise CompileError('source must contain exactly one function', node)
        definition = functions[0]
        function = self._visit_function(definition)
        function.definition = definition
        function.source_imports = [item for item in node.body if isinstance(item, (ast.Import, ast.ImportFrom))]
        function.closure_cells = self._closure_cells
        return function

    def _visit_function(self, node):
        positional = [*node.args.posonlyargs, *node.args.args]
        parameters = [*positional, *filter(None, (node.args.vararg,)),
                      *node.args.kwonlyargs, *filter(None, (node.args.kwarg,))]
        self._local_names = {arg.arg for arg in parameters}
        # Comprehension targets belong to their own scope.
        def local_assignments(item):
            if isinstance(item, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                return
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                self._local_names.add(item.id)
            for child in ast.iter_child_nodes(item):
                local_assignments(child)

        for statement in node.body:
            local_assignments(statement)
        self._function_name = node.name
        self._direct_positional_count = len(positional) if len(parameters) == len(positional) else None
        python_features = set()
        if isinstance(node, ast.AsyncFunctionDef):
            python_features.add('AsyncFunctionDef')
        if node.type_params:
            python_features.add('type parameters')
        for statement in node.body:
            for item in ast.walk(statement):
                if isinstance(item, _PYTHON_STATEMENTS):
                    python_features.add(type(item).__name__)
                elif (isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                      and item.id in FRAME_BUILTINS and self._is_builtin(item.id)):
                    python_features.add(item.id)
                elif isinstance(item, (ast.Attribute, ast.Name)) and isinstance(item.ctx, ast.Del):
                    python_features.add('del name or attribute')
                elif isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store):
                    python_features.add('attribute assignment')
                elif isinstance(item, ast.AnnAssign) and item.value is None:
                    python_features.add('annotation without a value')
        if python_features:
            function = core.Fun(node.name, [core.Var(arg.arg, source=arg) for arg in parameters], [], source=node)
            function.python_features = tuple(sorted(python_features))
            return function
        args = []
        for arg in parameters:
            if arg is node.args.vararg or arg is node.args.kwarg:
                args.append(core.Var(arg.arg, tuple_t if arg is node.args.vararg else dict_t, arg))
                continue
            annotation = self._evaluated_annotations.get(arg.arg, arg.annotation)
            if arg.annotation is not None and annotation is None:
                annotation = type(None)
            hint = get_type_hint(annotation)
            if arg.annotation is not None and hint is None and not is_dynamic_array_annotation(annotation):
                raise CompileError(f'unsupported annotation for {arg.arg}', arg.annotation)
            args.append(core.Var(arg.arg, hint, arg))
        return_annotation = self._evaluated_annotations.get('return', node.returns)
        if node.returns is not None and return_annotation is None:
            return_annotation = type(None)
        hint = get_type_hint(return_annotation)
        if node.returns is not None and hint is None:
            raise CompileError('unsupported return annotation', node.returns)
        function = core.Fun(node.name, args, [self.visit(stmt) for stmt in node.body],
                            void_t if hint == none_t else hint, node)
        function.direct_positional_count = self._direct_positional_count
        return function

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
        if value is None: return core.LitNone(node)
        if isinstance(value, (complex, bytes, range, slice)) or value is Ellipsis:
            return core.LitObject(value, node)
        if isinstance(value, tuple): return core.LitTuple([self._literal(item, node) for item in value], node)
        raise CompileError(f'global {getattr(node, "id", "value")!r} is not an immutable supported constant', node)

    def visit_Name(self, node):
        if node.id in self._comp_locals:
            return core.Var(self._comp_locals[node.id], source=node)
        if isinstance(node.ctx, ast.Load) and node.id not in self._local_names and node.id in self._constants:
            if any(self._constants[node.id] is kind for kind in _TYPE_NAMES.values()):
                return core.LitObject(self._constants[node.id], node)
            if callable(self._constants[node.id]):
                return core.Global(node.id, node)
            return self._literal(self._constants[node.id], node)
        if isinstance(node.ctx, ast.Load) and node.id in _TYPE_NAMES and self._is_builtin(node.id):
            return core.LitObject(_TYPE_NAMES[node.id], node)
        if (isinstance(node.ctx, ast.Load) and node.id not in self._local_names
                and (node.id in self._namespace or hasattr(builtins, node.id))):
            return core.Global(node.id, node)
        return core.Var(node.id, source=node)

    def visit_Constant(self, node):
        if node.value is None: return core.LitNone(node)
        return self._literal(node.value, node)

    def visit_JoinedStr(self, node):
        return core.JoinedStr([self.visit(part) for part in node.values], node)

    def visit_FormattedValue(self, node):
        return core.FormattedValue(self.visit(node.value),
                                   self.visit(node.format_spec) if node.format_spec is not None else None,
                                   node.conversion, node)

    def visit_Tuple(self, node): return core.LitTuple(self._items(node.elts), node)

    def _items(self, elements):
        return [core.Starred(self.visit(element.value), element) if isinstance(element, ast.Starred)
                else self.visit(element) for element in elements]

    def visit_List(self, node): return core.LitList(self._items(node.elts), node)
    def visit_Set(self, node): return core.LitSet(self._items(node.elts), node)

    def visit_Dict(self, node):
        entries = [core.Starred(self.visit(value), value) if key is None
                   else core.LitTuple([self.visit(key), self.visit(value)], key)
                   for key, value in zip(node.keys, node.values)]
        return core.LitDict(entries, node)

    def visit_Return(self, node):
        if node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None):
            return core.Return(None, node)
        return core.Return(self.visit(node.value), node)

    def _assign_target(self, target, value, annotation, node):
        if isinstance(target, ast.Name): return core.Assign(target.id, value, annotation, node)
        if isinstance(target, (ast.Tuple, ast.List)):
            if annotation is not None: raise CompileError('unpacking annotations are not supported', target)
            star = None
            elements = []
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    if star is not None:
                        raise CompileError('multiple starred unpacking targets', target)
                    star = index
                    element = element.value
                elements.append(element)
            if all(isinstance(element, ast.Name) for element in elements):
                return core.UnpackAssign([element.id for element in elements], value, node, star=star)
            refs = [self._temporary('unpack') for _ in elements]
            unpack = core.UnpackAssign(refs, value, node, star=star)
            return [unpack, *[self._assign_target(element, core.Var(ref, source=element), None, node)
                              for ref, element in zip(refs, elements)]]
        if isinstance(target, ast.Subscript):
            base, indices = self._subscript_parts(target)
            return core.StoreIndex(base, indices, value, target)
        raise CompileError('unsupported assignment target', target)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        if len(node.targets) == 1: return self._assign_target(node.targets[0], value, None, node)
        temp = self._temporary('assign')
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
            value = core.Prim(opname, [ref, self.visit(node.value)], node)
            value.inplace = True
            return core.Assign(node.target.id, value, source=node)
        if isinstance(node.target, ast.Subscript):
            value, indices = self._subscript_parts(node.target)
            return core.AugStoreIndex(value, indices, opname, self.visit(node.value), node.target)
        raise CompileError('unsupported augmented assignment target', node.target)

    def visit_Delete(self, node):
        targets = []
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                raise CompileError('del requires a subscript', target)
            value, indices = self._subscript_parts(target)
            targets.append(core.DeleteIndex(value, indices, target))
        return targets

    def visit_BinOp(self, node):
        opname = core.PRIM_OPS.get(type(node.op))
        if opname is None: raise CompileError('unsupported binary operator', node)
        return core.Prim(opname, [self.visit(node.left), self.visit(node.right)], node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub): return core.Prim('neg#', [self.visit(node.operand)], node)
        if isinstance(node.op, ast.UAdd): return core.Prim('pos#', [self.visit(node.operand)], node)
        if isinstance(node.op, ast.Invert): return core.Prim('invert#', [self.visit(node.operand)], node)
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

    def visit_IfExp(self, node):
        return core.IfExp(self.visit(node.test), self.visit(node.body), self.visit(node.orelse), node)

    def _visit_loop_body(self, statements):
        self._loop_depth += 1
        try:
            return [self.visit(statement) for statement in statements]
        finally:
            self._loop_depth -= 1

    def visit_While(self, node): return core.While(self.visit(node.test), self._visit_loop_body(node.body), [self.visit(x) for x in node.orelse], node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            variable = core.Var(node.target.id, source=node.target)
            body = lambda: self._visit_loop_body(node.body)
        else:
            variable = core.Var(self._temporary('item'), source=node.target)

            def body():
                unpack = self._assign_target(node.target, variable, None, node)
                return [unpack, *self._visit_loop_body(node.body)]

        return self._for_loop(variable, node.iter, body, node.orelse, node)

    def _for_loop(self, variable, iterable, body, orelse, node):
        if not (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name)
                and iterable.func.id in ('range', 'xrange')
                and iterable.func.id not in self._local_names
                and iterable.func.id not in self._comp_locals
                and (iterable.func.id == 'xrange' or self._is_builtin('range'))
                and not iterable.keywords
                and not any(isinstance(arg, ast.Starred) for arg in iterable.args)):
            values = self.visit(iterable)
            return core.ForEach(variable, values, body(), [self.visit(x) for x in orelse], node)
        args = [self.visit(arg) for arg in iterable.args]
        if not 1 <= len(args) <= 3:
            raise CompileError('range() expects one to three arguments', iterable)
        begin, end, step = core.LitInt(0, node), args[0], core.LitInt(1, node)
        if len(args) >= 2: begin, end = args[0], args[1]
        if len(args) == 3: step = args[2]
        return core.Loop(variable, begin, end, body(), step, [self.visit(x) for x in orelse], node)

    def _temporary(self, role):
        self._temporary_count += 1
        return f'${role}.{self._temporary_count}'

    def visit_ListComp(self, node):
        return self._comprehension(node, core.LitList([], node), 'append', [node.elt])

    def visit_SetComp(self, node):
        return self._comprehension(node, core.LitSet([], node), 'add', [node.elt])

    def visit_DictComp(self, node):
        return self._comprehension(node, core.LitDict([], node), '__setitem__', [node.key, node.value])

    def _comprehension(self, node, initial, method, elements):
        outer = self._comp_locals.copy()
        result_name = self._temporary('comprehension')
        result = core.Var(result_name, source=node)

        def clause(index):
            if index == len(node.generators):
                call = core.MethodCall(result, method, [self.visit(item) for item in elements], [], [], node)
                return core.Expr(call, node)
            generator = node.generators[index]
            if generator.is_async:
                raise CompileError('async comprehensions are not supported', node)
            item_name = self._temporary('item')

            def body():
                if isinstance(generator.target, ast.Name):
                    self._comp_locals[generator.target.id] = item_name
                    unpack = []
                else:
                    target = self._comp_target(generator.target)
                    unpack = [self._assign_target(target, core.Var(item_name, source=node), None, node)]
                conditions = [self.visit(test) for test in generator.ifs]
                inner = clause(index + 1)
                for condition in reversed(conditions):
                    inner = core.If(condition, [inner], [], condition)
                return unpack + [inner]

            return self._for_loop(core.Var(item_name, source=node), generator.iter, body, [], node)

        try:
            assign = core.Assign(result_name, initial, source=node)
            return core.Comprehension([assign, clause(0)], result, node)
        finally:
            self._comp_locals = outer

    def _comp_target(self, node):
        if isinstance(node, ast.Name):
            name = self._temporary('item')
            self._comp_locals[node.id] = name
            return ast.copy_location(ast.Name(id=name, ctx=ast.Store()), node)
        if isinstance(node, (ast.Tuple, ast.List)):
            return ast.copy_location(type(node)(elts=[self._comp_target(x) for x in node.elts], ctx=ast.Store()), node)
        if isinstance(node, ast.Starred):
            return ast.copy_location(ast.Starred(value=self._comp_target(node.value), ctx=ast.Store()), node)
        raise CompileError('unsupported comprehension target', node)

    def visit_Break(self, node):
        if self._loop_depth == 0: raise CompileError('break outside loop', node)
        return core.Break(node)

    def visit_Continue(self, node):
        if self._loop_depth == 0: raise CompileError('continue outside loop', node)
        return core.Continue(node)
    def visit_Pass(self, node): return core.Noop(node)
    def visit_Expr(self, node): return core.Expr(self.visit(node.value), node)

    def _is_builtin(self, name):
        if name in self._local_names or name in self._comp_locals:
            return False
        builtin = getattr(builtins, name)
        return self._bindings.get(name, self._namespace.get(name, builtin)) is builtin

    def visit_Call(self, node):
        names, values = self._keywords(node)
        expanded = any(isinstance(arg, ast.Starred) for arg in node.args)
        class_method = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner = node.func.value.id
            qualified = f'{owner}.{node.func.attr}'
            if (qualified in OBJECT_CLASS_METHODS and owner not in self._local_names
                    and self._is_builtin(owner)):
                class_method = qualified
        args = self._items(node.args)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                module_value = self._constants.get(module_name, math if module_name == 'math' else None)
            else:
                module_value = None
            if module_value is math and module_name not in self._local_names and not names and not expanded:
                if node.func.attr not in MATH_FUNCTIONS:
                    raise CompileError(f'unsupported math function {node.func.attr!r}', node.func)
                return core.CallFunc(core.Var(f'math.{node.func.attr}', source=node.func), args, node)
            if class_method:
                return core.MethodCall(None, class_method, args, names, values, node)
            if (node.func.attr == 'fromkeys' and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'dict' and 'dict' not in self._local_names
                    and self._is_builtin('dict')):
                return core.MethodCall(None, 'dict.fromkeys', args, names, values, node)
            receiver = (core.Global(module_name, node.func.value)
                        if module_value is math and module_name not in self._local_names
                        else self.visit(node.func.value))
            if node.func.attr in _BOXED_METHODS | OBJECT_METHODS:
                return core.MethodCall(receiver, node.func.attr, args, names, values, node)
            return core.Invoke(core.Attribute(receiver, node.func.attr, node.func), args, names, values, node)
        if not isinstance(node.func, ast.Name):
            return core.Invoke(self.visit(node.func), args, names, values, node)
        name = node.func.id
        if name in self._local_names or name in self._comp_locals:
            return core.Invoke(self.visit(node.func), args, names, values, node)
        if name in FUNCTION_INTRINSICS:
            if not self._is_builtin(name):
                return core.Invoke(core.Global(name, node.func), args, names, values, node)
            if name in ('sorted', 'dict') or names or expanded:
                return core.MethodCall(None, name, args, names, values, node)
        elif (names or expanded or hasattr(builtins, name)
              or name == self._function_name and len(args) != self._direct_positional_count):
            return core.Invoke(core.Global(name, node.func), args, names, values, node)
        return core.CallFunc(core.Var(name, source=node.func), args, node)

    def _keywords(self, node):
        return [option.arg for option in node.keywords], [
            core.Global(option.value.id, option.value)
            if option.arg == 'key' and isinstance(option.value, ast.Name)
            and option.value.id not in self._local_names and option.value.id not in self._comp_locals
            else self.visit(option.value)
            for option in node.keywords
        ]

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            module_name = node.value.id
            module_value = self._constants.get(module_name, math if module_name == 'math' else None)
            if module_value is math and module_name not in self._local_names and node.attr in MATH_CONSTANTS:
                return core.LitFloat(getattr(math, node.attr), node, double64_t)
        if node.attr == 'shape': return core.Prim('shape#', [self.visit(node.value)], node)
        return core.Attribute(self.visit(node.value), node.attr, node)

    def _subscript_parts(self, node):
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return self.visit(node.value), self._items(elements)

    def visit_Slice(self, node):
        return core.Slice(self.visit(node.lower) if node.lower else None,
                          self.visit(node.upper) if node.upper else None,
                          self.visit(node.step) if node.step else None, node)

    def visit_Subscript(self, node):
        value, indices = self._subscript_parts(node)
        return core.Index(value, indices, node)

    def generic_visit(self, node): raise CompileError(f'unsupported syntax: {type(node).__name__}', node)
