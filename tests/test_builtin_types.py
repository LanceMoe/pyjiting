from typing import Optional

import pytest

from pyjiting import jit, jit_from_source


def test_complex_arithmetic_and_conversion_match_python():
    def operations(value: complex, scalar: float):
        z = complex(real=scalar, imag=-2)
        return (value.real, value.imag, z + value, z / value, 1 / value, value ** z,
                -value, +value, value.conjugate(), abs(value), bool(value), value == 2j)

    native = jit(operations)
    assert native(2j, 3.0) == operations(2j, 3.0)
    with pytest.raises(ZeroDivisionError):
        native(0j, 3.0)
    with pytest.raises(TypeError, match='not supported'):
        jit_from_source('def f(a, b):\n    return a < b')(1j, 2j)


def test_binary_construction_mutation_and_encoding():
    @jit
    def packet(buffer: bytearray, label: str):
        literal = b'\x00\xff'
        suffix = bytes.fromhex('ca fe')
        other = bytearray.fromhex('01 02')
        buffer[:] = literal
        buffer.extend(suffix)
        del buffer[:1]
        buffer += other
        buffer *= 2
        encoded = label.encode(encoding='utf-8')
        return buffer, bytes(buffer), buffer[::-1], encoded.decode('utf-8'), suffix.hex('-'), 255 in literal

    original = bytearray(b'old')
    changed, contents, reversed_bytes, decoded, separator, found = packet(original, '日本語')
    assert changed is original and contents == b'\xff\xca\xfe\x01\x02' * 2
    assert reversed_bytes == bytearray(contents[::-1])
    assert decoded == '日本語' and separator == 'ca-fe' and found is True

    @jit
    def binary_number(value):
        encoded = value.to_bytes(length=2, byteorder='big')
        return encoded, (int.from_bytes(encoded, byteorder='big') << 8) | (~0 & 255)

    assert binary_number(0xCAFE) == (b'\xca\xfe', 0xCAFEFF)


def test_memoryview_mutation_cast_and_properties_match_python():
    @jit
    def inspect_buffer(view: memoryview, selection: slice):
        window = view[selection]
        window[0] = 65
        table = view.cast('B', shape=[2, 2])
        table[1, 1] = 66
        return view, window, table.shape, table[1, 1], table.obj, view.readonly, window.tobytes()

    data = bytearray(b'1234')
    view = memoryview(data)
    returned, window, shape, last, owner, readonly, snapshot = inspect_buffer(view, slice(1, None, 2))
    assert returned is view and window.obj is data and owner is data and readonly is False
    assert data == bytearray(b'1A3B') and shape == (2, 2) and last == 66 and snapshot == b'AB'
    with pytest.raises(TypeError, match='read-only'):
        inspect_buffer(memoryview(b'1234'), slice(1, None, 2))

    release = jit_from_source('def close(view):\n    view.release()\n    return view.tobytes()')
    with pytest.raises(ValueError, match='released memoryview'):
        release(view)


def test_range_and_slice_preserve_python_indices_and_loop_order():
    @jit
    def read(interval: range, selector: slice):
        items = [index for index in interval[selector]]
        return interval, selector, interval.start, interval.stop, interval.step, selector.indices(len(interval)), items

    interval, selector = range(10, -2, -2), slice(None, None, -2)
    actual, selected, start, stop, step, indices, items = read(interval, selector)
    assert actual is interval and selected is selector and (start, stop, step) == (10, -2, -2)
    assert indices == selector.indices(len(interval)) and items == list(interval[selector])

    @jit
    def make():
        progression = range(3, 15, 3)
        return (progression.count(6), progression.index(9), progression[-1],
                list(reversed(progression)), 4 not in progression, slice(2))

    assert make() == (1, 2, 12, [12, 9, 6, 3], True, slice(2))


def test_none_optional_and_bool_preserve_python_values_at_boundaries():
    @jit
    def lookup(value: Optional[int] = None) -> int | None:
        if value is not None:
            return value + 1

    @jit
    def flags(value: None):
        return value is None, (True, not value)

    both = jit_from_source('def select(flag):\n    if flag:\n        return None\n    return 5')
    assert lookup() is None and lookup(5) == 6
    state, pair = flags(None)
    assert state is True and pair[0] is True and pair[1] is True
    assert both(True) is None and both(False) == 5

    truth = jit_from_source('def f(value):\n    return value > 2')
    assert truth(3) is True and truth(0) is False

    @jit
    def mixed(flag) -> int | str:
        if flag:
            return 'yes'
        return 0

    assert mixed(True) == 'yes' and mixed(False) == 0


def test_tuple_iteration_algebra_and_structural_identity():
    @jit
    def inspect_tuple(values, index):
        alias = values
        local = ('local', 1)
        shifted = (*values[1:], 7)
        return (alias is values, local is local, shifted, values[index], values.count('x'),
                values + ('more',), values * 2, values < ('z',), 'x' in values,
                [value for value in values], values)

    original = ('x', 2)
    same, local, shifted, indexed, count, joined, repeated, less, contains, items, returned = inspect_tuple(original, -1)
    assert same is local is less is contains is True
    assert shifted == (2, 7) and indexed == 2 and count == 1 and items == ['x', 2]
    assert joined == ('x', 2, 'more') and repeated == ('x', 2, 'x', 2) and returned is original


def test_strings_can_round_trip_through_extended_methods():
    @jit
    def transform(text: str):
        pieces = text.split(',', maxsplit=1)
        formatted = '{first}:{second}'.format(first=pieces[0], second=pieces[1].strip())
        return pieces, formatted, text.startswith(('a', 'b')), text.replace(',', ';', 1)

    assert transform('a, bé') == (['a', ' bé'], 'a:bé', True, 'a; bé')


def test_object_annotation_preserves_custom_instance_across_jit_calls():
    @jit
    def identity(value: object) -> object:
        return value

    @jit
    def use(value: object):
        return identity(value), type(value), isinstance(value, bytes)

    value = object()
    result, kind, is_bytes = use(value)
    assert result is value and kind is object and is_bytes is False
