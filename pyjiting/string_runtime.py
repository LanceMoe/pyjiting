import ctypes
import threading


class StringDescriptor(ctypes.Structure):
    pass


StringDescriptor._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_uint32)),
    ('length', ctypes.c_int64),
]
StringPointer = ctypes.POINTER(StringDescriptor)

_state = threading.local()
_callbacks = {}
_literals = {}


def begin_call():
    stack = getattr(_state, 'arenas', None)
    if stack is None:
        stack = _state.arenas = []
    stack.append([])


def end_call():
    _state.arenas.pop()


def _arena():
    arenas = getattr(_state, 'arenas', None)
    if not arenas:
        raise RuntimeError('string value created outside a JIT dispatch')
    return arenas[-1]


def make_string(value):
    codepoints = [ord(char) for char in value]
    buffer = (ctypes.c_uint32 * max(1, len(codepoints)))(*codepoints)
    descriptor = StringDescriptor(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32)), len(codepoints))
    pointer = ctypes.pointer(descriptor)
    _arena().append((buffer, descriptor, pointer))
    return pointer


def literal_address(value):
    cached = _literals.get(value)
    if cached is None:
        codepoints = [ord(char) for char in value]
        buffer = (ctypes.c_uint32 * max(1, len(codepoints)))(*codepoints)
        descriptor = StringDescriptor(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32)), len(codepoints))
        pointer = ctypes.pointer(descriptor)
        cached = _literals[value] = (buffer, descriptor, pointer)
    return ctypes.cast(cached[2], ctypes.c_void_p).value


def to_python(pointer):
    if not pointer:
        return ''
    value = pointer.contents
    return ''.join(chr(value.data[index]) for index in range(value.length))


def _binary_string(fn):
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer, StringPointer)

    @callback_type
    def callback(left, right):
        return ctypes.cast(make_string(fn(to_python(left), to_python(right))), ctypes.c_void_p).value
    return callback


def _query(fn):
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int64, StringPointer, StringPointer)

    @callback_type
    def callback(left, right):
        return int(fn(to_python(left), to_python(right)))
    return callback


def _repeat():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer, ctypes.c_int64)

    @callback_type
    def callback(value, count):
        return ctypes.cast(make_string(to_python(value) * count), ctypes.c_void_p).value
    return callback


def _index():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32))

    @callback_type
    def callback(value, index, error):
        text = to_python(value)
        try:
            result = text[index]
        except IndexError:
            error[0] = 4
            result = ''
        return ctypes.cast(make_string(result), ctypes.c_void_p).value
    return callback


def _slice():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64)

    @callback_type
    def callback(value, has_lower, lower, has_upper, upper):
        start = lower if has_lower else None
        stop = upper if has_upper else None
        return ctypes.cast(make_string(to_python(value)[start:stop]), ctypes.c_void_p).value
    return callback


def callback_address(name):
    if not _callbacks:
        _callbacks.update({
            'concat': _binary_string(lambda left, right: left + right),
            'repeat': _repeat(),
            'index': _index(),
            'slice': _slice(),
            'compare': _query(lambda left, right: (left > right) - (left < right)),
            'startswith': _query(str.startswith),
            'endswith': _query(str.endswith),
            'find': _query(str.find),
            'count': _query(str.count),
        })
    return ctypes.cast(_callbacks[name], ctypes.c_void_p).value
