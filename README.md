**Languages:** [English](README.md) · [简体中文](README.zh.md) · [日本語](README.ja.md)

# Pyjiting

Pyjiting is an experimental Python JIT compiler, which is the product of my undergraduate thesis. The goal is to implement a light-weight miniature general-purpose Python JIT compiler, using LLVM (via [llvmlite](https://llvmlite.readthedocs.io/)) as the backend.

```python
from pyjiting import jit

@jit
def fib(x):
    if x < 3:
        return 1
    return fib(x-1) + fib(x-2)

print(fib(40))  # ~40x faster than CPython
```

## How it works

Pyjiting compiles a function to native code when it is first called. A function decorated with `@jit` goes through the following pipeline:

```
 Python source
      │  ast.parse + ASTVisitor (pyjiting/parser.py)
      ▼
 Core AST (pyjiting/ast.py)
      │  Hindley-Milner style type inference (pyjiting/infer.py)
      ▼
 Type constraints (pyjiting/types.py, pyjiting/utils.py)
      │  unified with the runtime argument types on each call
      ▼
 Specialized AST ──► LLVM IR codegen (pyjiting/codegen.py)
                          │
                          ▼
              LLVM opt (O3 + loop vectorization, new pass manager)
                          │
                          ▼
              MCJIT compilation ──► ctypes callback (pyjiting/ll_types.py)
```

1. **Parsing** — the function source is parsed into a small Core AST (`pyjiting/parser.py`).
2. **Type inference** — a Hindley-Milner style inferencer collects type constraints (`pyjiting/infer.py`).
3. **Specialization** — when the function is called, the actual argument types (`int`, `float`, `numpy.ndarray`, …) are unified with the inferred constraints to obtain a concrete monomorphic signature (`pyjiting/main.py`).
4. **Codegen** — LLVM IR is generated for that signature and compiled to native code with the LLVM optimizer and MCJIT engine.
5. **Caching** — each specialized signature is compiled only once and cached by its mangled name; subsequent calls with the same types dispatch directly to the native code.

## Supported subset

| Area | Supported behavior | Regression coverage |
|---|---|---|
| Specialization | Separate native specialization per scalar/array signature, deterministic names, isolated LLVM modules and same-name function isolation | `tests/test_specialization.py` |
| Scalars | Bool, Int32, Int64, Float32 and Float64; deterministic widening and fixed-width integer wraparound | `tests/test_infer.py`, `tests/test_arith.py` |
| Arithmetic | `+ - * / // % **`, unary `-`, Python floor/mod signs, constant integer powers, NaN-aware scalar truthiness | `tests/test_arith.py` |
| Control flow | `if`, `while`, `for range`, `break`, `continue`, negative/dynamic steps, nested loops and loop `else` | `tests/test_control_flow.py`, `tests/test_runtime_errors.py` |
| Arrays | int32/int64/float32/float64 ndarrays; strided multidimensional reads/writes, shape indexing, transposed/sliced/F-order/negative-stride views | `tests/test_array.py`, `tests/test_abi.py` |
| Annotations and callbacks | Scalar annotations, deferred `np.ndarray` dtype specialization, and persistent annotated `@reg` callbacks | `tests/test_parser.py`, `tests/test_reg_callback.py` |
| Validation | Parser source locations, inference rules and LLVM verification for generated modules | `tests/test_parser.py`, `tests/test_codegen.py` |

### Numeric and array semantics

pyjiting uses fixed-width Int32/Int64 values, not arbitrary-precision Python integers. Mixed scalar operations use deterministic promotion: int32 widens to int64 as needed, float32 widens to float64 when combined with int64 or float64, and `/` produces a floating result. Assignment allows widening only, so a Python `int` must be explicitly converted to `np.int32` before storing into an int32 array.

Integer arithmetic follows two's-complement fixed-width behavior. In particular, the minimum signed integer divided by -1 remains the minimum signed integer, rather than attempting arbitrary-precision promotion.

Arrays use a stable `data/ndim/shape/strides` ABI. Element reads and writes support multidimensional, transposed, sliced and negative-stride NumPy views. Bounds checks, array creation, broadcasting and whole-array ufunc operations are deliberately not supported.
The number of indices must match the runtime array dimensionality; a mismatch raises `ValueError`.

Each specialization uses a private LLVM symbol derived from the decorated Python function identity plus its type signature. Two functions with the same short name therefore cannot share a cached native implementation by accident.
## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- llvmlite >= 0.49.0
- numpy >= 2.5.2

Project dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Use
`uv` to create the environment and run all project commands; `requirements.txt` is
not part of the development workflow.

## Development with uv

```bash
uv sync --extra dev
uv run pytest -q
uv run pyright
```

`uv sync` creates or updates the local `.venv` from the locked dependency set. The
project is installed into that environment, so examples can be run directly with
`uv run`.

Build distributable artifacts with:

```bash
uv build
```

The source distribution and wheel are written to `dist/`.

## Usage

### JIT compile a function

```python
from pyjiting import jit

@jit
def is_prime(x):
    if x < 2:
        return 0
    i = 2
    while i * i <= x:
        if x % i == 0:
            return 0
        i = i + 1
    return 1

is_prime(169941229)   # returns 1 (compiled for int)
```

### Dynamic typing

The same function can be specialized for different argument types. Each new type combination triggers one extra compilation, then it is cached:

```python
@jit
def test(a, b):
    return a + b

print(test(114, 514))    # 628   (int64 specialization)
print(test(11.4, 51.4))  # 62.8  (double specialization)
```

### `@jit` and `@reg`

`@jit` marks a function as a compilation target. Its source is parsed when the decorator runs, then it is specialized and compiled to native code on the first call for each argument-type signature. Use it for the numerical and control-flow-heavy part of a workload.

`@reg` marks an ordinary Python function as a callback that JIT-compiled code may invoke. It is not compiled by pyjiting: the native function crosses the ctypes callback boundary to execute the original Python implementation. Use it for reporting, logging, or existing Python-only operations. Registered functions need supported scalar parameter and return annotations: `int`, `float`, `bool`, `np.int32`, `np.int64`, `np.float32`, or `np.float64`. They must not raise an exception across the callback boundary.

| Decorator | Role | Runs as | Typical use |
|---|---|---|---|
| `@jit` | Compiles a function | LLVM-generated native code | Numeric kernels, loops, recursion |
| `@reg` | Registers a callable for JIT code | Original Python function through ctypes | Logging, reporting, Python-only integration |

### Call Python code from JITed functions

Plain Python functions can be called from native code when registered with `@reg` and fully annotated with supported scalar types:
```python
from pyjiting import jit, reg

@reg
def report(x: int) -> int:
    print(f'{x} is prime!')
    return 0

@jit
def find_primes(n):
    for i in range(2, n):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime == True:
            report(i)
    return 0
```

### Run the examples

```bash
uv run examples/example_fib.py
uv run examples/example_find_primes.py
uv run examples/example_is_prime.py
uv run examples/example_loop.py
uv run examples/example_while.py
uv run examples/example_generic.py
uv run examples/example_bool_ops.py
uv run examples/example_float_math.py
uv run examples/example_array_2d.py
uv run examples/example_annotations.py
uv run examples/example_mixed_types.py
```

## Performance

Run the focused benchmark with `uv run benchmarks/benchmark.py`. It reports cold compilation plus execution, a warm cached call, and an equivalent CPython loop. The workload uses a dynamic modular reduction so its loop body cannot be reduced to a closed-form counter calculation.

You can find the source code of these test samples in the `examples/` directory.

```
My test environment:
CPU: 13th Gen Intel(R) Core(TM) i5-13600K@5.4Ghz
Memory: 64GB
OS: Windows 11 64bit (10.0.26200)
Python 3.12.11 64bit
LLVMLite 0.49.0
NumPy 2.5.2
```

| Benchmark | JIT | CPython | Speedup |
|---|---:|---:|---:|
| `fib(40)` | 126.2 ms | 5056.4 ms | ~40x |
| `find_primes(100000)` | 909.4 ms | 11984.9 ms | ~13.2x |
| `is_prime(169941229)` | 340.5 ms | 4462.4 ms | ~13.1x |
| `loop(100000000)` | ~0 ms | 1871.1 ms | ∞ (see note below) |
| `test_while(100000000)` | ~0 ms | 1669.1 ms | ∞ (see note below) |

Raw output:

```
fib_jit(40) = 102334155 (cost time: 126.23047828674316 ms)
fib_nojit(40) = 102334155 (cost time: 5056.420564651489 ms)
rate: 40.057049876380916

find_primes_jit(100000) = 0 (cost time: 909.3630313873291 ms)
find_primes_nojit(100000) = 0 (cost time: 11984.926223754883 ms)
rate: 13.179473774594307

is_prime_jit(169941229) = True (cost time: 340.4872417449951 ms)
is_prime_nojit(169941229) = True (cost time: 4462.382793426514 ms)
rate: 13.105873719546224

loop_jit(100000000) = 200000000 (cost time: 0.0 ms)
loop_nojit(100000000) = 200000000 (cost time: 1871.0994720458984 ms)
rate: Infinite

test_while_jit(100000000) = 100000000 (cost time: 0.0 ms)
test_while_nojit(100000000) = 100000000 (cost time: 1669.1172122955322 ms)
rate: Infinite
```

> Note: the trivial `loop`/`while` benchmarks only count a counter, so LLVM optimizes them away entirely at `-O3` — hence the "Infinite" speedup. The other benchmarks do real work.

## Project layout

```
pyjiting/
├── __init__.py   # exports jit, reg
├── main.py       # @jit / @reg decorators, specialization & call caching
├── parser.py     # Python AST -> Core AST
├── ast.py        # Core AST node definitions
├── infer.py      # Hindley-Milner style type inference
├── codegen.py    # Core AST -> LLVM IR
├── ll_types.py   # ctypes wrappers, name mangling, argument conversion
├── types.py      # type system (BaseType, VarType, FuncType, ArrayType, ...)
└── utils.py      # unification / substitution utilities
```

## Limitations

This is a research/educational project, not a production JIT. Among others:

- No garbage collection integration; only a small statically-typed subset of Python is supported.
- `for` loops must iterate over `range`; a constant zero step is rejected during compilation.
- Runtime division by zero raises `ZeroDivisionError`; a dynamic zero range step raises `ValueError` through the JIT error-status ABI.
- A non-void JIT function must return on every control-flow path.
- Integer power requires a compile-time constant exponent because a dynamic negative exponent has no single static return type.
- Registered (`@reg`) functions need supported scalar annotations and must not raise across the callback boundary.
- Default, keyword-only, variadic and keyword call arguments are rejected. JIT calls accept exactly their declared positional argument count.
- Python strings, containers, unpacking assignment, arbitrary objects, bounds checks and array-wide NumPy operations are unsupported.
# Special thanks

Inspired by [numpile](https://dev.stephendiehl.com/numpile/) tutorial and continue to work on this basis.

I am deeply grateful to professor Mr Takeshi Ogasawara gave me many inspirations and appropriate advice.
