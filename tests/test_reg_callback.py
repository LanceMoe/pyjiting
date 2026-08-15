import gc

import numpy as np
import pytest

from pyjiting import jit, reg


@reg
def scale_float(value: np.float32, factor: np.float32) -> np.float32:
    return np.float32(value * factor)


@reg
def bool_to_int(value: bool) -> int:
    return 7 if value else 3


@jit
def callback_float(value):
    return scale_float(value, value)


@jit
def callback_bool(value):
    return bool_to_int(value)


class CallbackFailure(RuntimeError):
    pass


@reg
def fail_callback(value: int) -> int:
    raise CallbackFailure(f'callback failed for {value}')


@jit
def callback_failure(value):
    return fail_callback(value) + 100


def test_registered_callbacks_use_their_annotations_and_survive_gc():
    assert callback_float(np.float32(1.25)) == np.float32(1.5625)
    gc.collect()
    assert callback_float(np.float32(2.0)) == np.float32(4.0)
    assert callback_bool(True) == 7
    assert callback_bool(False) == 3


def test_registered_callback_exception_is_reraised_and_stops_native_execution():
    with pytest.raises(CallbackFailure, match='callback failed for 7'):
        callback_failure(7)


def callback_factory(result):
    @reg
    def shared(value: int) -> int:
        return result + value

    @jit
    def caller(value):
        return shared(value)

    return caller


def test_same_named_registered_callbacks_are_bound_by_identity():
    first = callback_factory(10)
    second = callback_factory(20)

    assert first(1) == 11
    assert second(1) == 21
