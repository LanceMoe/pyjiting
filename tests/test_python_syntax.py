# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import asyncio
from contextlib import contextmanager
import inspect

import pytest

from pyjiting import JITContext, get_llvm_ir, jit, jit_from_source, runtime_stats
from pyjiting.errors import CompileError, InferError, RuntimeClosedError


def test_f_string_keeps_python_formatting_and_evaluation_order_in_llvm():
    class Value:
        def __repr__(self):
            events.append('repr')
            return 'café'

        def __format__(self, spec):
            events.append(('format', spec))
            return 'formatted'

    def render(value: object, width, mark):
        return f"{{{mark(value)!a:{mark(width)}}}} {value:*>4} {width=} {value!s} {value!r}"

    events = []

    def mark(value):
        events.append('width' if isinstance(value, int) else 'value')
        return value

    value = Value()
    expected = render(value, 12, mark)
    expected_events = events.copy()
    events.clear()
    compiled = jit(render)
    assert compiled(value, 12, mark) == expected
    assert events == expected_events
    assert runtime_stats(compiled)['backend'] == 'llvm'
    assert 'define' in get_llvm_ir(compiled, value, 12, mark)


def test_generator_expression_is_lazy_and_crosses_the_llvm_boundary():
    @jit
    def squares(values, log):
        return (log(value) for value in values if value > 1)

    @jit
    def take_first(values, log):
        return next(squares(values, log))

    events = []

    def log(value):
        events.append(value)
        return value * value

    stream = squares([1, 2, 3], log)
    assert inspect.isgenerator(stream)
    assert events == []
    assert next(stream) == 4
    assert events == [2]
    assert list(stream) == [9]
    assert take_first([1, 4, 5], log) == 16
    assert events == [2, 3, 4]
    assert runtime_stats(take_first)['backend'] == 'llvm'
    stats = runtime_stats(squares)
    assert (stats['backend'], stats['calls'], stats['specializations']) == ('cpython', 2, 0)
    assert 'GeneratorExp' in stats['python_features']
    with pytest.raises(CompileError, match='no LLVM IR'):
        get_llvm_ir(squares, [], log)


def test_yield_send_delegation_and_finally_match_python():
    @jit
    def stream(events):
        try:
            received = yield 'first'
            yield from (received, received * 2)
            return 'done'
        finally:
            events.append('cleanup')

    events = []
    first = stream(events)
    assert next(first) == 'first'
    assert first.send(3) == 3
    first.close()
    assert events == ['cleanup']

    second = stream(events)
    assert next(second) == 'first'
    assert second.send(5) == 5
    assert next(second) == 10
    with pytest.raises(StopIteration) as finished:
        next(second)
    assert finished.value.value == 'done'
    assert events == ['cleanup', 'cleanup']


def test_try_except_else_finally_and_with_preserve_cleanup_and_suppression():
    @contextmanager
    def managed(events, suppress):
        events.append('enter')
        try:
            yield
        except LookupError:
            events.append('exit with exception')
            if not suppress:
                raise
        else:
            events.append('exit normally')

    @jit
    def run(events, fail, suppress):
        try:
            with managed(events, suppress):
                if fail:
                    raise KeyError('missing')
                events.append('body')
        except KeyError as error:
            events.append(error.args[0])
        else:
            events.append('else')
        finally:
            events.append('finally')

    for fail, suppress, expected in [
        (True, True, ['enter', 'exit with exception', 'else', 'finally']),
        (True, False, ['enter', 'exit with exception', 'missing', 'finally']),
        (False, False, ['enter', 'body', 'exit normally', 'else', 'finally']),
    ]:
        events = []
        run(events, fail, suppress)
        assert events == expected


def test_parameter_binding_and_forwarding_work_from_both_entrypoints():
    def pack(head=1, /, *tail: int, scale=2, **extra: int):
        return head * scale, tail, extra

    decorated = jit(pack)
    source = jit_from_source('''
        def pack(head=1, /, *tail: int, scale=2, **extra: int):
            return head * scale, tail, extra
    ''')
    for compiled in (decorated, source):
        assert compiled(head=9) == pack(head=9)
        assert compiled(3, 4, 5, scale=6, head=7, color=8) == pack(3, 4, 5, scale=6, head=7, color=8)
        assert compiled.specialize(2, 3, scale=4)['argument_types'] == (
            'Int64', 'PythonTuple', 'Int64', 'Dict')
        assert runtime_stats(compiled)['backend'] == 'llvm'

    @jit
    def forward(values, options):
        return decorated(*values, **options)

    @jit
    def positional():
        return decorated(3, 4, 5)

    assert forward((3, 4), {'scale': 5, 'item': 6}) == pack(3, 4, scale=5, item=6)
    assert positional() == pack(3, 4, 5)
    with pytest.raises(TypeError, match='keywords must be strings'):
        forward((3,), {1: 2})

    def keyword_only(*, value):
        return value

    strict = jit(keyword_only)
    assert strict(value=4) == 4
    with pytest.raises(TypeError):
        strict(4)
    with pytest.raises(TypeError):
        strict()


def test_source_defaults_run_once_and_nested_keyword_calls_propagate_errors():
    events = []

    def default():
        events.append('default')
        return []

    namespace = {'default': default}
    append = jit_from_source('''
        def append(value, *, values=default()):
            values.append(value)
            return values
    ''', namespace=namespace)
    invoke = jit_from_source('''
        def invoke(value, options):
            return append(value=value, **options)
    ''', namespace=namespace)
    first = append(value=1)
    assert invoke(2, {}) is first
    assert first == [1, 2]
    with pytest.raises(TypeError, match="multiple values.*'value'"):
        invoke(3, {'value': 4})
    assert events == ['default']


def test_computed_calls_look_up_the_callable_before_evaluating_arguments():
    class Receiver:
        @property
        def action(self):
            events.append('lookup')
            return lambda argument: argument

    @jit
    def run(receiver: object, evaluate):
        return receiver.action(evaluate())

    events = []

    def evaluate():
        events.append('argument')
        return 3

    assert run(Receiver(), evaluate) == 3
    assert events == ['lookup', 'argument']


def test_expanded_calls_keep_python_evaluation_order_on_error():
    def expanded(target, evaluate):
        return target(**{'item': 1}, item=evaluate('item'), other=evaluate('other'))

    def starred(target, items, evaluate):
        return target(*items, other=evaluate('other'))

    events = []

    def evaluate(label):
        events.append(label)
        return 4

    def target(*args, **kwargs):
        raise AssertionError('invalid arguments should stop the call')

    for function, arguments in [(expanded, (target, evaluate)), (starred, (target, 42, evaluate))]:
        with pytest.raises(TypeError):
            function(*arguments)
        expected = events.copy()
        events.clear()
        with pytest.raises(TypeError):
            jit(function)(*arguments)
        assert events == expected
        events.clear()


def test_python_backend_keeps_context_lifetime_and_zero_argument_super():
    class Parent:
        def answer(self):
            return 42

    class Child(Parent):
        @jit
        def answer(self):
            return super().answer()

    assert Child().answer() == 42
    context = JITContext()
    run = context.from_source('''
        async def run(value):
            match value:
                case [left, right]:
                    return left + right
                case _:
                    return None
    ''')
    assert inspect.iscoroutinefunction(run)
    assert asyncio.run(run([1, 2])) == 3
    assert asyncio.run(run(1)) is None
    context.close()
    with pytest.raises(RuntimeClosedError):
        run([])


def test_keyword_calls_cannot_cross_llvm_contexts():
    with JITContext() as context:
        callee = context.from_source('def callee(*, value): return value')

        @jit
        def call(value):
            return callee(value=value)

        with pytest.raises(InferError, match='different runtime contexts'):
            call(1)
