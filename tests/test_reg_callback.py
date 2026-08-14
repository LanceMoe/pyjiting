import gc

import numpy as np

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


def test_registered_callbacks_use_their_annotations_and_survive_gc():
    assert callback_float(np.float32(1.25)) == np.float32(1.5625)
    gc.collect()
    assert callback_float(np.float32(2.0)) == np.float32(4.0)
    assert callback_bool(True) == 7
    assert callback_bool(False) == 3
