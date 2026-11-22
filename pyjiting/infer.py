from . import ast as core
from .errors import InferError
from .intrinsics import (LIST_INTEGER_METHODS, LIST_RETURNING_METHODS, MATH_INTRINSICS,
                         STRING_INTRINSICS, STRING_PREDICATES, STRING_TRANSFORMS)
from .types import (FuncType, TupleType, bool_t, can_widen, contains_array,
                    double64_t, float32_t, int32_t, int64_t, is_array, is_integer,
                    is_boxable, is_boxed, is_numeric, is_string, is_truthy_type, is_tuple,
                    list_t, object_t, promote_numeric, shape_t, str_t, void_t)


class UnderDetermined(InferError):
    pass


class TypeInferencer:
    """Infer one monomorphic specialization and annotate Core AST nodes in place."""

    def __init__(self, arg_types=None, registry=None, jit_resolver=None, reg_resolver=None):
        self.arg_types, self.registry = arg_types, registry or {}
        self.jit_resolver = jit_resolver
        self.reg_resolver = reg_resolver
        self.env, self.return_type, self.org_func_name = {}, None, None
        self.annotations = {}

    def visit(self, node):
        method = getattr(self, f'visit_{type(node).__name__}', None)
        if method is None: raise InferError(f'no type rule for {type(node).__name__}', node)
        return method(node)

    def _coerce(self, actual, expected, node):
        if actual == expected: return expected
        if can_widen(actual, expected): return expected
        if expected == object_t and is_boxable(actual): return expected
        if actual == object_t and (is_numeric(expected) or expected in (list_t, str_t)):
            return expected
        raise InferError(f'cannot use {actual} where {expected} is required', node)

    def _boxed_types(self, types, node):
        for ty in types:
            if not (is_boxable(ty) or is_array(ty)):
                raise InferError(f'cannot use {ty} as a Python object', node)

    def _join(self, left, right):
        if left == right: return left
        if (is_boxed(left) or is_boxed(right)) and is_boxable(left) and is_boxable(right):
            return object_t
        return promote_numeric(left, right)

    def _assignment_type(self, name, actual, expected, node):
        if expected is None: return actual
        if name not in self.annotations and (is_boxed(actual) or is_boxed(expected)):
            common = self._join(actual, expected)
            if common is not None: return common
        return self._coerce(actual, expected, node)

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
            if arg.annotation is not None:
                self.annotations[arg.id] = arg.annotation
        for stmt in node.body: self._visit_statement(stmt)
        if self.return_type is None: self.return_type = void_t
        if node.return_annotation: self.return_type = self._coerce(self.return_type, node.return_annotation, node)
        if contains_array(self.return_type):
            raise InferError('ndarray return values are not supported', node)
        if self.return_type != void_t and not self._always_returns(node.body):
            raise InferError('non-Void function has a path without return', node)
        return FuncType(args=self.arg_types, return_type=self.return_type)

    def _always_returns(self, statements):
        for statement in statements:
            if isinstance(statement, list):
                if self._always_returns(statement): return True
            elif isinstance(statement, core.Return):
                return True
            elif (isinstance(statement, core.While) and isinstance(statement.test, core.LitBool)
                  and statement.test.n and self._always_returns(statement.body)):
                return True
            elif isinstance(statement, core.If):
                if statement.orelse and self._always_returns(statement.body) and self._always_returns(statement.orelse):
                    return True
        return False

    def _visit_statement(self, stmt):
        if isinstance(stmt, list):
            for item in stmt: self._visit_statement(item)
        else: self.visit(stmt)

    def visit_LitInt(self, node): node.type = node.literal_type or int64_t; return node.type
    def visit_LitFloat(self, node): node.type = node.literal_type or double64_t; return node.type
    def visit_LitBool(self, node): node.type = bool_t; return node.type
    def visit_LitStr(self, node): node.type = str_t; return node.type
    def visit_LitNone(self, node): node.type = object_t; return node.type

    def visit_LitList(self, node):
        self._boxed_types([self.visit(item) for item in node.elements], node)
        node.type = list_t
        return node.type

    def visit_Starred(self, node):
        node.type = self.visit(node.value)
        return node.type

    def visit_ListComp(self, node):
        before = self.env.copy()
        for statement in node.body:
            self._visit_statement(statement)
        self.visit(node.result)
        self.env = before
        node.type = list_t
        return list_t

    def visit_SortKey(self, node):
        self._boxed_types([self.visit(item) for item in node.values], node)
        node.type = object_t
        return object_t
    def visit_LitTuple(self, node):
        node.type = TupleType([self.visit(element) for element in node.elements])
        return node.type

    def visit_Var(self, node):
        if node.id not in self.env: raise InferError(f'unknown variable {node.id}', node)
        node.type = self.env[node.id]; return node.type

    def visit_Assign(self, node):
        value_ty = self.visit(node.value); expected = node.annotation or self.env.get(node.ref)
        if value_ty == void_t: raise InferError('cannot assign a Void value', node)
        if node.annotation is not None:
            self.annotations[node.ref] = node.annotation
        node.type = self._assignment_type(node.ref, value_ty, expected, node)
        self.env[node.ref] = node.type

    def visit_UnpackAssign(self, node):
        value_ty = self.visit(node.value)
        node.boxed = is_boxed(value_ty) or node.star is not None and is_boxable(value_ty)
        if node.boxed:
            source_types = [object_t] * len(node.refs)
            if node.star is not None:
                source_types[node.star] = list_t
        elif not is_tuple(value_ty):
            raise InferError('unpacking requires a tuple or list value', node.value)
        elif len(node.refs) != len(value_ty.elements):
            raise InferError(f'cannot unpack tuple of length {len(value_ty.elements)} into {len(node.refs)} names', node)
        else:
            source_types = list(value_ty.elements)
        node.source_types, node.ref_types = source_types, []
        for name, ty in zip(node.refs, source_types):
            target = self._assignment_type(name, ty, self.env.get(name), node)
            self.env[name] = target; node.ref_types.append(target)
        node.type = value_ty

    def visit_StoreIndex(self, node):
        array_ty = self.visit(node.value)
        if is_boxed(array_ty):
            self._boxed_subscript(node)
            node.type = self.visit(node.rhs)
            self._boxed_types([node.type], node.rhs)
            return
        if not is_array(array_ty): raise InferError('subscript assignment requires an array', node.value)
        for index in node.indices: self._coerce(self.visit(index), int64_t, index)
        node.type = self._coerce(self.visit(node.rhs), array_ty.b, node.rhs)

    def visit_AugStoreIndex(self, node):
        array_ty = self.visit(node.value)
        if is_boxed(array_ty):
            self._boxed_subscript(node)
            self._boxed_types([self.visit(node.rhs)], node.rhs)
            node.type = object_t
            return
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

    def _boxed_subscript(self, node):
        if len(node.indices) != 1:
            raise InferError('list expects one index', node)
        index = node.indices[0]
        if isinstance(index, core.Slice):
            for bound in (index.lower, index.upper, index.step):
                if bound is not None:
                    self._boxed_types([self.visit(bound)], bound)
        else:
            self._boxed_types([self.visit(index)], index)

    def visit_DeleteIndex(self, node):
        if not is_boxed(self.visit(node.value)):
            raise InferError('subscript deletion requires a list', node.value)
        self._boxed_subscript(node)
        return void_t

    def visit_Index(self, node):
        value_ty = self.visit(node.value)
        if is_boxed(value_ty):
            self._boxed_subscript(node)
            node.type = list_t if value_ty == list_t and isinstance(node.indices[0], core.Slice) else object_t
            return node.type
        if is_tuple(value_ty):
            if len(node.indices) != 1: raise InferError('tuple expects one index', node)
            index = core.integer_constant_value(node.indices[0])
            if index is None: raise InferError('tuple index must be a compile-time integer', node.indices[0])
            if index < 0: index += len(value_ty.elements)
            if not 0 <= index < len(value_ty.elements): raise InferError('tuple index out of range', node.indices[0])
            node.tuple_index = index; node.type = value_ty.elements[index]; return node.type
        if is_string(value_ty):
            if len(node.indices) != 1: raise InferError('string expects one index', node)
            index = node.indices[0]
            if isinstance(index, core.Slice):
                for bound in (index.lower, index.upper):
                    if bound is not None: self._coerce(self.visit(bound), int64_t, bound)
                if index.step is not None:
                    self._coerce(self.visit(index.step), int64_t, index.step)
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
            if is_boxed(left) or is_boxed(right):
                self._boxed_types([left, right], node)
                node.operand_type = object_t
                node.type = list_t if (
                    node.fn == 'add#' and left == list_t and (right == list_t or node.inplace)
                    or node.fn == 'mult#' and (
                        left == list_t and is_integer(right) or right == list_t and is_integer(left))
                ) else object_t
                return node.type
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
            if is_boxed(ty):
                node.operand_type = object_t
                node.type = object_t
                return node.type
            if not is_numeric(ty): raise InferError('unary minus requires a numeric value', node)
            node.type = ty; return ty
        if node.fn == 'not#':
            ty = self.visit(node.args[0])
            if not is_truthy_type(ty): raise InferError('not requires a scalar value', node)
            node.type = object_t if is_boxed(ty) else bool_t
            return node.type
        if node.fn in ('and#', 'or#'):
            left, right = self.visit(node.args[0]), self.visit(node.args[1])
            if is_boxed(left) or is_boxed(right):
                self._boxed_types([left, right], node)
                node.operand_type = node.type = self._join(left, right)
                return node.type
            if is_string(left) and is_string(right):
                node.operand_type = node.type = str_t; return str_t
            if is_tuple(left) and left == right:
                node.operand_type = node.type = left; return left
            common = promote_numeric(left, right)
            if common is None: raise InferError(f'{node.fn} operands need a common scalar type', node)
            node.operand_type, node.type = common, common; return common
        raise InferError(f'unknown primitive {node.fn}', node)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        boxed_comparison = False
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if is_boxed(left) or is_boxed(right):
                self._boxed_types([left, right], node)
                boxed_comparison = True
                left = right
                continue
            if op in ('is#', 'isnot#'):
                raise InferError('identity comparison requires a list or boxed value', node)
            if op in core.MEMBERSHIP_OPS:
                if not (is_string(left) and is_string(right)):
                    raise InferError(f'{op} requires string operands', node)
                left = right
                continue
            if not (is_string(left) and is_string(right)) and promote_numeric(left, right) is None:
                raise InferError(f'{op} requires comparable values of compatible types', node)
            left = right
        node.type = object_t if boxed_comparison else bool_t
        return node.type

    def visit_If(self, node):
        test_ty = self.visit(node.test)
        if not is_truthy_type(test_ty): raise InferError(f'cannot use {test_ty} as a condition', node.test)
        before = self.env.copy()
        for stmt in node.body: self._visit_statement(stmt)
        body_env = self.env.copy(); self.env = before.copy()
        for stmt in node.orelse: self._visit_statement(stmt)
        else_env = self.env.copy()
        merged = before.copy()
        for name in body_env.keys() & else_env.keys():
            left, right = body_env[name], else_env[name]
            if not node.orelse and not (is_boxed(left) or is_boxed(right)):
                continue
            common = self._join(left, right)
            if common is None:
                raise InferError(f'incompatible branch types for {name}: {left} and {right}', node)
            if common != object_t:
                self._retarget_assignments(node.body, name, common)
                self._retarget_assignments(node.orelse, name, common)
            merged[name] = common
        self.env = merged

    def visit_IfExp(self, node):
        test_type = self.visit(node.test)
        if not is_truthy_type(test_type):
            raise InferError(f'cannot use {test_type} as a condition', node.test)
        left, right = self.visit(node.body), self.visit(node.orelse)
        node.type = self._join(left, right)
        if node.type in (None, void_t, shape_t) or is_array(node.type):
            raise InferError(f'incompatible conditional expression types {left} and {right}', node)
        return node.type

    def _retarget_assignments(self, statements, name, target):
        for statement in statements:
            if isinstance(statement, list):
                self._retarget_assignments(statement, name, target)
            elif isinstance(statement, core.Assign) and statement.ref == name:
                statement.type = target
            elif isinstance(statement, core.UnpackAssign) and name in statement.refs:
                statement.ref_types[statement.refs.index(name)] = target
            elif isinstance(statement, (core.If, core.While, core.Loop, core.ForEach)):
                self._retarget_assignments(statement.body, name, target)
                self._retarget_assignments(statement.orelse, name, target)

    def visit_While(self, node):
        def test():
            test_ty = self.visit(node.test)
            if not is_truthy_type(test_ty):
                raise InferError(f'cannot use {test_ty} as a condition', node.test)
        self._loop_body(node, test=test)

    def visit_Loop(self, node):
        if core.integer_constant_value(node.step) == 0:
            raise InferError('range() arg 3 must not be zero', node.step)
        for value in (node.begin, node.end, node.step): self._coerce(self.visit(value), int64_t, value)
        self._loop_body(node, int64_t)

    def visit_ForEach(self, node):
        iterable = self.visit(node.iterable)
        if is_array(iterable): item_type = iterable.b
        elif is_string(iterable): item_type = str_t
        elif is_boxed(iterable): item_type = object_t
        else: raise InferError('for iteration requires an array, list, or string', node.iterable)
        self._loop_body(node, item_type)

    def _loop_body(self, node, item_type=None, test=None):
        # A local can become boxed on a back edge, e.g. total += item in a
        # mixed numeric list. Iterate to a stable set of boxed locals.
        before = self.env.copy()
        entry = before.copy()
        return_type = self.return_type
        while True:
            self.env = entry.copy()
            self.return_type = return_type
            if test is not None:
                test()
            if item_type is not None:
                name = node.var.id
                target = self._coerce(item_type, self.annotations[name], node.var) if name in self.annotations else item_type
                self.env[name] = node.var.type = target
                node.item_type = item_type
            for stmt in node.body:
                self._visit_statement(stmt)
            promotions = {name for name, current in entry.items() if current != object_t
                          and name in self.env and self._join(current, self.env[name]) == object_t}
            if not promotions:
                break
            for name in promotions:
                entry[name] = object_t
        for name in before:
            if entry[name] == object_t:
                self.env[name] = object_t
        for stmt in node.orelse:
            self._visit_statement(stmt)
        self.env = {name: self.env[name] for name in before}

    def visit_Return(self, node):
        value_ty = void_t if node.value is None else self.visit(node.value)
        if self.return_type is None:
            self.return_type = value_ty
        elif self.return_type != value_ty:
            if value_ty == void_t and is_boxed(self.return_type) or self.return_type == void_t and is_boxed(value_ty):
                common = object_t
            else:
                common = self._join(self.return_type, value_ty)
            if common is None:
                raise InferError(f'incompatible return types {self.return_type} and {value_ty}', node)
            self.return_type = common

    def visit_CallFunc(self, node):
        arg_types = [self.visit(arg) for arg in node.args]
        name = node.fn.id
        always_boxed = ('list', 'range', 'slice', 'enumerate', 'zip', 'reversed')
        optionally_boxed = ('sum', 'any', 'all', 'len', 'min', 'max', 'abs', 'ord', 'chr',
                            'int', 'float', 'str', 'bool')
        if name in always_boxed or name in optionally_boxed and any(is_boxed(ty) for ty in arg_types):
            self._boxed_types(arg_types, node)
            node.list_runtime = name
            node.type = {'list': list_t, 'len': int64_t, 'ord': int64_t}.get(name, object_t)
            return node.type
        if node.fn.id == 'len':
            if len(arg_types) != 1 or not (is_string(arg_types[0]) or is_array(arg_types[0]) or is_tuple(arg_types[0])):
                raise InferError('len expects one string, tuple, or array argument', node)
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
        if node.fn.id == 'ord':
            if arg_types != [str_t]: raise InferError('ord expects one string argument', node)
            node.type = int64_t; return int64_t
        if node.fn.id == 'chr':
            if len(arg_types) != 1 or not is_integer(arg_types[0]):
                raise InferError('chr expects one integer argument', node)
            node.type = str_t; return str_t
        if node.fn.id in MATH_INTRINSICS:
            if len(arg_types) != 1 or not is_numeric(arg_types[0]):
                raise InferError(f'{node.fn.id} expects one numeric argument', node)
            node.operand_type = double64_t
            node.type = bool_t if node.fn.id in ('math.isnan', 'math.isinf', 'math.isfinite') else double64_t
            return node.type
        if node.fn.id in ('sum', 'any', 'all'):
            if len(arg_types) != 1 or not is_array(arg_types[0]):
                raise InferError(f'{node.fn.id} expects one array argument', node)
            element = arg_types[0].b
            if node.fn.id in ('any', 'all'):
                node.type = bool_t
            elif element in (int32_t, bool_t):
                node.type = int64_t
            elif element == float32_t:
                node.type = double64_t
            else:
                node.type = element
            return node.type
        if node.fn.id in STRING_INTRINSICS and node.fn.id in ('str.startswith', 'str.endswith'):
            if arg_types != [str_t, str_t]: raise InferError(f'{node.fn.id[4:]} expects one string argument', node)
            node.type = bool_t; return bool_t
        if node.fn.id in STRING_INTRINSICS and node.fn.id in ('str.find', 'str.count'):
            if arg_types != [str_t, str_t]: raise InferError(f'{node.fn.id[4:]} expects one string argument', node)
            node.type = int64_t; return int64_t
        if node.fn.id in STRING_TRANSFORMS:
            if arg_types != [str_t]: raise InferError(f'{node.fn.id[4:]} expects no arguments', node)
            node.type = str_t; return str_t
        if node.fn.id == 'str.replace':
            if arg_types != [str_t, str_t, str_t]:
                raise InferError('replace expects two string arguments', node)
            node.type = str_t; return str_t
        if node.fn.id in STRING_PREDICATES:
            if arg_types != [str_t]: raise InferError(f'{node.fn.id[4:]} expects no arguments', node)
            node.type = bool_t; return bool_t
        if node.fn.id == self.org_func_name:
            if self.return_type is None: raise InferError('recursive return type needs an earlier return', node)
            node.type = self.return_type; return node.type
        if self.jit_resolver is not None:
            signature, symbol = self.jit_resolver(node.fn.id, arg_types)
            if signature is not None:
                node.jit_signature, node.jit_symbol = signature, symbol
                node.type = signature.return_type; return node.type
        if self.reg_resolver is not None:
            signature, registration_id = self.reg_resolver(node.fn.id)
            if signature is not None:
                if len(signature.args) != len(arg_types): raise InferError('wrong number of callback arguments', node)
                for actual, expected in zip(arg_types, signature.args): self._coerce(actual, expected, node)
                node.registered_id = registration_id
                node.type = signature.return_type; return node.type
        signature = self.registry.get(node.fn.id)
        if signature is None: raise InferError(f'function {node.fn.id!r} is not registered', node)
        if len(signature.args) != len(arg_types): raise InferError('wrong number of callback arguments', node)
        for actual, expected in zip(arg_types, signature.args): self._coerce(actual, expected, node)
        node.type = signature.return_type; return node.type

    def visit_MethodCall(self, node):
        value_type = self.visit(node.value) if node.value is not None else None
        args = [self.visit(arg) for arg in node.args]
        keywords = [self.visit(arg) for arg in node.keyword_values]
        if value_type == str_t and node.name == 'count':
            if args != [str_t] or keywords:
                raise InferError('count expects one string argument', node)
            node.type = int64_t
            return node.type
        if value_type is not None and not is_boxed(value_type):
            raise InferError(f'{node.name} requires a list', node.value)
        self._boxed_types(args + keywords, node)
        if node.name == 'sorted' or node.name in LIST_RETURNING_METHODS and value_type == list_t:
            node.type = list_t
        elif node.name in LIST_INTEGER_METHODS and value_type == list_t:
            node.type = int64_t
        else:
            node.type = object_t
        return node.type

    def visit_Expr(self, node): return self.visit(node.value)
    def visit_Noop(self, node): return void_t
    def visit_Break(self, node): return void_t
    def visit_Continue(self, node): return void_t
