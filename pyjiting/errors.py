class CompileError(Exception):
    """Raised when source is outside pyjiting's supported Python subset."""

    def __init__(self, message, node=None):
        if node is not None and hasattr(node, "lineno"):
            message = f"line {node.lineno}:{getattr(node, 'col_offset', 0)}: {message}"
        super().__init__(message)


class InferError(CompileError):
    """Raised when a program cannot be assigned a valid static type."""


class CodegenError(CompileError):
    """Raised when a typed program cannot be lowered to LLVM IR."""
