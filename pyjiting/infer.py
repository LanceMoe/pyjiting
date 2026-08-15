from . import ast as core
from .errors import InferError
from .intrinsics import STRING_INTRINSICS
from .types import (FuncType, bool_t, can_widen, double64_t, int64_t, is_array,
                    is_integer, is_numeric, is_string, is_truthy_type,
                    promote_numeric, shape_t, str_t, void_t)


class UnderDetermined(InferError):
    pass


class TypeInferencer:
    """Infer one monomorphic specialization and annotate Core AST nodes in place."""

    def __init__(self, arg_types=None, registry=None, jit_resolver=None):
        self.arg_types, self.registry = arg_types, registry or {}
        self.jit_resolver = jit_resolver
        self.env, self.return_type, self.org_func_name = {}, None, None

    def visit(self, node):
        method = getattr(self, f'visit_{type(node).__name__}', None)
        if method is None: raise InferError(f'no type rule for {type(node).__name__}', node)
        return method(node)

    def _coerce(self, actual, expected, node):
        if actual == expected: return expected
        if can_widen(actual, expected): return expected
        raise InferError(f'cannot use {actual} where {expected} is required', node)

    def _numeric(self, node, division=False):
        left, right = self.visit(node.args[0]), self.visit(node.args[1])
        common = promote_numeric(left, right)
        if common is None: raise InferError(f'{node.fn} requires numeric operands, got {left} and {right}', node)
        node.operand_type = common
        node.type = double64_t if division and is_integer(common) else common
        return node.type

    def visit_Fun(self, node):
        self.org_func_name = node.fname
        if self.arg_types is None:
            if any(arg.annotation is None for arg in node.args):
                raise UnderDetermined('function arguments require a call specialization or annotations', node)
            self.arg_types = [arg.annotation for arg in node.args]
        if len(self.arg_types) != len(node.args): raise InferError('wrong number of arguments', node)
        for arg, actual in zip(node.args, self.arg_types):
            ty = self._coerce(actual, arg.annotation, arg) if arg.annotation else actual
            arg.type, self.env[arg.id] = ty, ty
        for stmt in node.body: self._visit_statement(stmt)
        if self.return_type is None: self.return_type = void_t
        if node.return_annotation: self.return_type = self._coerce(self.return_type, node.return_annotation, node)
        if self.return_type != void_t and not self._always_returns(node.body):
            raise InferError('non-Void function has a path without return', node)
        return FuncType(args=self.arg_types, return_type=self.return_type)

    def _always_returns(self, statements):
        for statement in statements:
            if isinstance(statement, list):
                if self._always_returns(statement): return True
            elif isinstance(statement, core.Return):
                return True
            elif isinstance(statement, core.If):
                if statement.orelse and self._always_returns(statement.body) and self._always_returns(statement.orelse):
                    return True
        return False

    def _visit_statement(self, stmt):
        if isinstance(stmt, list):
            for item in stmt: self._visit_statement(item)
        else: self.visit(stmt)

    def visit_LitInt(self, node): node.type = int64_t; return node.type
    def visit_LitFloat(self, node): node.type = double64_t; return node.type
    def visit_LitBool(self, node): node.type = bool_t; return node.type
    def visit_LitStr(self, node): node.type = str_t; return node.type

    def visit_Var(self, node):
        if node.id not in self.env: raise InferError(f'unknown variable {node.id}', node)
        node.type = self.env[node.id]; return node.type

    def visit_Assign(self, node):
        value_ty = self.visit(node.value); expected = node.annotation or self.env.get(node.ref)
        node.type = self._coerce(value_ty, expected, node) if expected else value_ty
        self.env[node.ref] = node.type

    def visit_StoreIndex(self, node):
        array_ty = self.visit(node.value)
        if not is_array(array_ty): raise InferError('subscript assignment requires an array', node.value)
        for index in node.indices: self._coerce(self.visit(index), int64_t, index)
        node.type = self._coerce(self.visit(node.rhs), array_ty.b, node.rhs)

    def visit_AugStoreIndex(self, node):
        array_ty = self.visit(node.value)
        if not is_array(array_ty): raise InferError('subscript assignment requires an array', node.value)
        for index in node.indices: self._coerce(self.visit(index), int64_t, index)
        rhs_ty = self.visit(node.rhs)
        common = promote_numeric(array_ty.b, rhs_ty)
        if common is None: raise InferError(f'{node.fn} requires numeric operands', node)
        exponent = core.integer_constant_value(node.rhs) if node.fn == 'pow#' else None
        if node.fn == 'pow#' and is_integer(array_ty.b) and is_integer(rhs_ty) and exponent is None:
            raise InferError('integer power requires a constant exponent', node)
        result = double64_t if node.fn == 'div#' and is_integer(common) else common
        if node.fn == 'pow#' and exponent is not None and exponent < 0: result = double64_t
        node.operand_type = common
        node.type = self._coerce(result, array_ty.b, node)

    def visit_Index(self, node):
        value_ty = self.visit(node.value)
        if is_string(value_ty):
            if len(node.indices) != 1: raise InferError('string expects one index', node)
            index = node.indices[0]
            if isinstance(index, core.Slice):
                for bound in (index.lower, index.upper):
                    if bound is not None: self._coerce(self.visit(bound), int64_t, bound)
                if index.step is not None:
                    self._coerce(self.visit(index.step), int64_t, index.step)
                    if core.integer_constant_value(index.step) != 1:
                        raise InferError('string slices only support a step of 1', index.step)
                setattr(index, 'type', str_t); node.type = str_t; return str_t
            self._coerce(self.visit(index), int64_t, index); node.type = str_t; return str_t
        if value_ty == shape_t:
            if len(node.indices) != 1: raise InferError('shape expects one index', node)
            self._coerce(self.visit(node.indices[0]), int64_t, node.indices[0]); node.type = int64_t; return node.type
        if not is_array(value_ty): raise InferError('subscript requires an array', node.value)
        for index in node.indices: self._coerce(self.visit(index), int64_t, index)
        node.type = value_ty.b; return node.type

    def visit_Prim(self, node):
        if node.fn == 'shape#':
            if not is_array(self.visit(node.args[0])): raise InferError('shape requires an array', node)
            node.type = shape_t; return node.type
        if node.fn in core.ARITHMETIC_OPS:
            left = self.visit(node.args[0]); right = self.visit(node.args[1])
            if node.fn == 'add#' and is_string(left) and is_string(right):
                node.operand_type = node.type = str_t; return str_t
            if node.fn == 'mult#' and ((is_string(left) and is_integer(right)) or
                                       (is_integer(left) and is_string(right))):
                node.operand_type = str_t; node.type = str_t; return str_t
            if node.fn == 'div#': return self._numeric(node, True)
            if node.fn == 'pow#':
                left, right = self.visit(node.args[0]), self.visit(node.args[1])
                exponent = core.integer_constant_value(node.args[1])
                if is_integer(left) and is_integer(right) and exponent is None:
                    raise InferError('integer power requires a constant exponent', node)
                common = promote_numeric(left, right)
                if common is None: raise InferError('pow requires numeric operands', node)
                node.operand_type = common
                node.type = double64_t if exponent is not None and exponent < 0 else common
                return node.type
            common = promote_numeric(left, right)
            if common is None: raise InferError(f'{node.fn} requires numeric operands, got {left} and {right}', node)
            node.operand_type = node.type = common; return common
        if node.fn == 'neg#':
            ty = self.visit(node.args[0])
            if not is_numeric(ty): raise InferError('unary minus requires a numeric value', node)
            node.type = ty; return ty
        if node.fn == 'not#':
            ty = self.visit(node.args[0])
            if not is_truthy_type(ty): raise InferError('not requires a scalar value', node)
            node.type = bool_t; return bool_t
        if node.fn in ('and#', 'or#'):
            left, right = self.visit(node.args[0]), self.visit(node.args[1])
            if is_string(left) and is_string(right):
                node.operand_type = node.type = str_t; return str_t
            common = promote_numeric(left, right)
            if common is None: raise InferError(f'{node.fn} operands need a common scalar type', node)
            node.operand_type, node.type = common, common; return common
        raise InferError(f'unknown primitive {node.fn}', node)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if not (is_string(left) and is_string(right)) and promote_numeric(left, right) is None:
                raise InferError(f'{op} requires comparable values of compatible types', node)
            left = right
        node.type = bool_t; return bool_t

    def visit_If(self, node):
        test_ty = self.visit(node.test)
        if not is_truthy_type(test_ty): raise InferError(f'cannot use {test_ty} as a condition', node.test)
        before = self.env.copy()
        for stmt in node.body: self._visit_statement(stmt)
        body_env = self.env.copy(); self.env = before.copy()
        for stmt in node.orelse: self._visit_statement(stmt)
        else_env = self.env.copy()
        for name in body_env.keys() & else_env.keys():
            if body_env[name] == else_env[name]: self.env[name] = body_env[name]

    def visit_While(self, node):
        test_ty = self.visit(node.test)
        if not is_truthy_type(test_ty): raise InferError(f'cannot use {test_ty} as a condition', node.test)
        before = self.env.copy()
        for stmt in node.body: self._visit_statement(stmt)
        for stmt in node.orelse: self._visit_statement(stmt)
        self.env = {name: ty for name, ty in self.env.items() if name in before}

    def visit_Loop(self, node):
        if core.integer_constant_value(node.step) == 0:
            raise InferError('range() arg 3 must not be zero', node.step)
        for value in (node.begin, node.end, node.step): self._coerce(self.visit(value), int64_t, value)
        before = self.env.copy(); self.env[node.var.id] = int64_t; node.var.type = int64_t
        for stmt in node.body: self._visit_statement(stmt)
        for stmt in node.orelse: self._visit_statement(stmt)
        self.env = {name: ty for name, ty in self.env.items() if name in before}

    def visit_ForEach(self, node):
        iterable = self.visit(node.iterable)
        if is_array(iterable): item_type = iterable.b
        elif is_string(iterable): item_type = str_t
        else: raise InferError('for iteration requires an array or string', node.iterable)
        before = self.env.copy(); self.env[node.var.id] = item_type; node.var.type = item_type
        for stmt in node.body: self._visit_statement(stmt)
        for stmt in node.orelse: self._visit_statement(stmt)
        self.env = {name: ty for name, ty in self.env.items() if name in before}

    def visit_Return(self, node):
        value_ty = self.visit(node.value)
        self.return_type = value_ty if self.return_type is None else self._coerce(value_ty, self.return_type, node)

    def visit_CallFunc(self, node):
        arg_types = [self.visit(arg) for arg in node.args]
        if node.fn.id == 'len':
            if len(arg_types) != 1 or not (is_string(arg_types[0]) or is_array(arg_types[0])):
                raise InferError('len expects one string or array argument', node)
            node.type = int64_t; return int64_t
        if node.fn.id == 'abs':
            if len(arg_types) != 1 or not is_numeric(arg_types[0]):
                raise InferError('abs expects one numeric argument', node)
            node.type = arg_types[0]; return node.type
        if node.fn.id in ('min', 'max'):
            if len(arg_types) != 2:
                raise InferError(f'{node.fn.id} expects two arguments', node)
            common = promote_numeric(arg_types[0], arg_types[1])
            if common is None: raise InferError(f'{node.fn.id} expects numeric arguments', node)
            node.operand_type = node.type = common; return common
        if node.fn.id in STRING_INTRINSICS and node.fn.id in ('str.startswith', 'str.endswith'):
            if arg_types != [str_t, str_t]: raise InferError(f'{node.fn.id[4:]} expects one string argument', node)
            node.type = bool_t; return bool_t
        if node.fn.id in STRING_INTRINSICS and node.fn.id in ('str.find', 'str.count'):
            if arg_types != [str_t, str_t]: raise InferError(f'{node.fn.id[4:]} expects one string argument', node)
            node.type = int64_t; return int64_t
        if node.fn.id == self.org_func_name:
            if self.return_type is None: raise InferError('recursive return type needs an earlier return', node)
            node.type = self.return_type; return node.type
        if self.jit_resolver is not None:
            signature, symbol = self.jit_resolver(node.fn.id, arg_types)
            if signature is not None:
                node.jit_signature, node.jit_symbol = signature, symbol
                node.type = signature.return_type; return node.type
        signature = self.registry.get(node.fn.id)
        if signature is None: raise InferError(f'function {node.fn.id!r} is not registered', node)
        if len(signature.args) != len(arg_types): raise InferError('wrong number of callback arguments', node)
        for actual, expected in zip(arg_types, signature.args): self._coerce(actual, expected, node)
        node.type = signature.return_type; return node.type

    def visit_Expr(self, node): return self.visit(node.value)
    def visit_Noop(self, node): return void_t
    def visit_Break(self, node): return void_t
    def visit_Continue(self, node): return void_t
