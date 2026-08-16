from pyjiting import jit, reg


@jit
def typed_jit(value: int, scale: int = 2) -> int:
    return value * scale


@reg
def typed_reg(value: int) -> int:
    return value + 1


result_jit: int = typed_jit(value=3)
result_reg: int = typed_reg(3)

# These ignores are intentionally absent: Pyright must validate both decorators.
typed_jit('bad')  # pyright: ignore[reportArgumentType]
typed_reg('bad')  # pyright: ignore[reportArgumentType]
