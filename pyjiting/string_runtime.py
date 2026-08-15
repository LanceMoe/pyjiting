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
_allocator = None


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


def keep_alive(*values):
    _arena().extend(values)


def allocation_address():
    global _allocator
    if _allocator is None:
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64)

        @callback_type
        def allocate(size):
            buffer = ctypes.create_string_buffer(max(1, size))
            keep_alive(buffer)
            return ctypes.addressof(buffer)
        _allocator = allocate
    return ctypes.cast(_allocator, ctypes.c_void_p).value


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


def _unary_string(fn):
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer)

    @callback_type
    def callback(value):
        return ctypes.cast(make_string(fn(to_python(value))), ctypes.c_void_p).value
    return callback


def _unary_query(fn):
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int64, StringPointer)

    @callback_type
    def callback(value):
        return int(fn(to_python(value)))
    return callback


def _replace():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, StringPointer, StringPointer, StringPointer)

    @callback_type
    def callback(value, old, new):
        result = to_python(value).replace(to_python(old), to_python(new))
        return ctypes.cast(make_string(result), ctypes.c_void_p).value
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
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.POINTER(ctypes.c_int32))

    @callback_type
    def callback(value, has_lower, lower, has_upper, upper, has_step, step, error):
        start = lower if has_lower else None
        stop = upper if has_upper else None
        stride = step if has_step else None
        if stride == 0:
            error[0] = 5
            result = ''
        else:
            result = to_python(value)[start:stop:stride]
        return ctypes.cast(make_string(result), ctypes.c_void_p).value
    return callback


def _ord():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int64, StringPointer, ctypes.POINTER(ctypes.c_int32))

    @callback_type
    def callback(value, error):
        text = to_python(value)
        if len(text) != 1:
            error[0] = 6
            return 0
        return ord(text)
    return callback


def _chr():
    callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32))

    @callback_type
    def callback(value, error):
        if not 0 <= value <= 0x10FFFF:
            error[0] = 7
            result = ''
        else:
            result = chr(value)
        return ctypes.cast(make_string(result), ctypes.c_void_p).value
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
            'contains': _query(lambda needle, value: needle in value),
            'upper': _unary_string(str.upper),
            'lower': _unary_string(str.lower),
            'strip': _unary_string(str.strip),
            'lstrip': _unary_string(str.lstrip),
            'rstrip': _unary_string(str.rstrip),
            'replace': _replace(),
            'isalpha': _unary_query(str.isalpha),
            'isalnum': _unary_query(str.isalnum),
            'isdigit': _unary_query(str.isdigit),
            'isspace': _unary_query(str.isspace),
            'ord': _ord(),
            'chr': _chr(),
        })
    return ctypes.cast(_callbacks[name], ctypes.c_void_p).value
