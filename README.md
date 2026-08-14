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

## Features

- [x] LLVM backend via llvmlite, optimized at `-O3` with loop vectorization (new pass manager).
- [x] Automatic type specialization on first call; a separate native binary is compiled per argument-type combination (e.g. `int` vs `float`).
- [x] Arithmetic (`+ - * / %` …), comparison and boolean operators on scalars.
- [x] Control flow: `if` expressions, `for` loops over `range` (with `break`), and `while` loops.
- [x] Recursion (self-calls compile to native calls).
- [x] Calling ordinary Python functions from JITed native code through libffi-style callbacks — register them with the `@reg` decorator and annotate their types (see `example_find_primes.py`).
- [x] Scalar types: `int` (i64), `float` (f64), `bool`; NumPy array types with element access.

## Requirements

- Python >= 3.12
- llvmlite >= 0.44 (new LLVM pass manager API)
- numpy

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

### Call Python code from JITed functions

Plain Python functions can be called from native code. They must be registered with `@reg` and fully type-annotated (`int` / `float`):

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

### Run the benchmarks

```bash
uv run example_fib.py
uv run example_find_primes.py
uv run example_is_prime.py
uv run example_loop.py
uv run example_while.py
uv run example_generic.py
```

## Performance

You can find the source code of these test samples in the root directory.

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
- `for` loops must iterate over `range`; `pow` (`**`) is not yet implemented in codegen.
- Registered (`@reg`) functions must be annotated with `int`/`float` types only.
- Type inference must be able to determine all types from the arguments, otherwise `UnderDetermined` is raised.

# Special thanks

Inspired by [numpile](https://dev.stephendiehl.com/numpile/) tutorial and continue to work on this basis.

I am deeply grateful to professor Mr Takeshi Ogasawara gave me many inspirations and appropriate advice.
