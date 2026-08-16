**Languages:** [English](README.md) · [简体中文](README.zh.md) · [日本語](README.ja.md)

# Pyjiting

Pyjiting 是一个实验性的 Python JIT 编译器，也是我的本科毕业论文成果。本项目旨在基于 LLVM 后端（通过 [llvmlite](https://llvmlite.readthedocs.io/)），实现一个轻量、小巧的通用 Python JIT 编译器。

```python
from pyjiting import jit

@jit
def fib(x):
    if x < 3:
        return 1
    return fib(x-1) + fib(x-2)

print(fib(40))  # 比 CPython 快约 40 倍

```

## 工作原理

Pyjiting 在函数首次被调用时将其 JIT 编译为原生机器码。被 `@jit` 装饰的函数会经过以下流水线：

```
 Python 源码
      │  ast.parse + ASTVisitor (pyjiting/parser.py)
      ▼
 Core AST (pyjiting/ast.py)
      │  Hindley-Milner 风格的类型推导 (pyjiting/infer.py)
      ▼
 类型约束 (pyjiting/types.py, pyjiting/utils.py)
      │  每次调用时与运行时参数类型进行合一（Unification）
      ▼
 特化后的 AST ──► LLVM IR 代码生成 (pyjiting/codegen.py)
                          │
                          ▼
              LLVM 优化（O3 + 循环向量化，新版 Pass Manager）
                          │
                          ▼
              MCJIT 编译 ──► ctypes 回调 (pyjiting/ll_types.py)

```

1. **解析（Parsing）** —— 函数源码被解析并精简为自定义的 Core AST（`pyjiting/parser.py`）。
2. **类型推导（Type Inference）** —— 基于 Hindley-Milner 算法的类型推导器收集并求解类型约束（`pyjiting/infer.py`）。
3. **特化（Specialization）** —— 函数调用时，将实参类型（`int`、`float`、`numpy.ndarray` 等）与推导得到的约束合一，求得具体的单态（Monomorphic）签名（`pyjiting/main.py`）。
4. **代码生成（Codegen）** —— 针对该签名生成 LLVM IR，并通过 LLVM 优化器与 MCJIT 引擎编译为机器码。
5. **缓存（Caching）** —— 每种特化签名仅编译一次，并基于修饰名（Mangled Name）进行缓存；后续相同类型的调用会直接派发至已编译的原生代码执行。

## 支持的语言子集

| 领域 | 支持的行为 | 回归测试覆盖 |
| --- | --- | --- |
| 特化 | 针对标量/数组/字符串签名生成独立原生代码、确定性命名、并发编译隔离与原生 JIT 函数互调 | `tests/test_specialization.py`、`tests/test_extensions.py` |
| 标量 | `bool`、`int32`、`int64`、`float32`、`float64`；确定性的类型提升与定宽整数回绕（Wrap-around） | `tests/test_infer.py`、`tests/test_arith.py` |
| 算术 | `+ - * / // % **`、一元 `-`、符合 Python 规范的 floor/mod 符号语义、常量整数幂、适配 NaN 的标量真值判定 | `tests/test_arith.py` |
| 控制流 | `if`、`while`、`for in range(...)`、一维数组/字符串迭代、`break`、`continue`、循环 `else` | `tests/test_control_flow.py`、`tests/test_extensions.py` |
| 字符串 | Unicode 值、比较/成员判断、完整切片、拼接/重复、大小写/去空白/替换、字符谓词、搜索及 `ord`/`chr` | `tests/test_string.py`、`tests/test_string_phase2.py` |
| Tuple | 固定长度异构字面量/参数/返回值、嵌套、注解、常量索引、`len`、真值及定长名称解包 | `tests/test_tuple_phase4.py` |
| 数组 | 四种数值 dtype；安全索引、多维 Strided 读写、一维迭代及多维全量 `sum`/`any`/`all` 标量归约 | `tests/test_array.py`、`tests/test_extensions.py`、`tests/test_numeric_phase3.py` |
| 内建函数 | 标量/字符串内建函数，以及原生 `math` 三角、开方、指数、对数、分类与常量 | `tests/test_extensions.py`、`tests/test_string_phase2.py`、`tests/test_numeric_phase3.py` |
| 常量 | 在应用 `@jit` 时捕获不可变的标量/字符串全局及闭包值 | `tests/test_numeric_phase3.py` |
| 注解与回调 | 标量/字符串注解、`np.ndarray` dtype 惰性特化、带注解的 `@reg` 回调 | `tests/test_parser.py`、`tests/test_reg_callback.py`、`tests/test_extensions.py` |
| 验证机制 | 解析器源码位置映射、推导规则校验、LLVM 生成模块合法性验证 | `tests/test_parser.py`、`tests/test_codegen.py` |

### 数值与数组语义

Pyjiting 采用定宽的 Int32/Int64，而非 Python 原生的任意精度大整数。标量混合运算遵循确定性的类型提升规则：`int32` 按需提升为 `int64`；`float32` 与 `int64` 或 `float64` 混合时提升为 `float64`；`/` 除法始终返回浮点数。赋值操作仅允许安全的类型拓宽，因此将原生 Python `int` 赋给 int32 数组前，必须显式转换为 `np.int32`。

整数运算严格遵循二进制补码的定宽语义。特别地，最小有符号负数除以 `-1` 时不会自动提升精度，而是直接发生溢出并保留为最小有符号负数。

数组使用 descriptor v2 ABI：`data`、`ndim`、`shape`、以字节表示的 `strides`、`itemsize` 以及 NumPy flags。元素和 shape 索引支持负数并进行边界检查，支持转置、切片、负 stride、字节 stride 以及未对齐 view；读写使用对齐安全的访问。向只读数组实际写入会抛出 `ValueError("assignment destination is read-only")`；越界抛出 `IndexError`，索引维度数不匹配则抛出 `ValueError`。数组创建、广播、数组返回及整组 ufunc 运算仍不支持。
任意维度 Strided 数组支持原生全量 `sum`、`any` 和 `all`；int32/float32 求和提升为 int64/float64，空数组身份值与 Python 一致。axis 归约仍不支持。

不可变的数值/字符串全局与闭包值会在应用 `@jit` 时冻结为 Core 字面量。原生 `math` 支持 `sin`、`cos`、`sqrt`、`exp`、`log`、`log2`、`log10`、浮点分类及常量；domain/range 异常沿 JIT 错误 ABI 传播。

字符串采用带长度的 UTF-32 ABI，可正确处理 Unicode 码点和内嵌 NUL。临时值与返回值由单次调用 arena 管理。切片支持省略、正数、负数和动态非零步长；成员判断、Unicode 大小写转换、去空白、替换、字符谓词及 `ord`/`chr` 在静态子集内遵循 Python 语义。

Tuple 是不可变的固定长度结构类型，元素类型会进入特化签名和名称修饰。原生值通过指向形状专属结构的指针传递，并由单次调用 arena 保活，从而避开平台相关的聚合返回 ABI；Python 与 JIT-to-JIT 边界支持嵌套数值/字符串 tuple。

每个特化版本均使用由 Python 函数本体（Identity）及其类型签名派生出的独立 LLVM 符号。因此，即使存在同名的不同函数，也不会发生错误的本地缓存共享。

普通 `@jit` wrapper 通过原 Python 签名支持位置/关键字混合调用和不可变默认参数。`compiled.specialize(*args)` 可以只编译而不执行；`runtime_stats(compiled)`、`inspect_specializations(compiled)` 和 `get_llvm_ir(compiled, *args)` 分别提供函数级指标、特化列表和开发诊断 IR。统计包含编译等待、每个签名的编译次数、轻量失败详情、字符串 runtime 调用和注册回调调用。`JITContext` 为 Notebook 和长期服务提供独立 engine、模块/特化预算、显式关闭语义，并清理通过该 context 注册的回调。

启用 `fallback=True` 时，前端、类型推断或 codegen 不支持的路径会执行原 Python 函数。`FallbackWarning` 提供 `function`、`reason` 和 `error_type` 字段；`fallback_warning` 可设为默认的 `"once"`、`"always"` 或 `"ignore"`。函数特化上限、context 资源上限、已关闭 runtime 和 LLVM 内部错误绝不会回退。字符串索引和比较使用原生 UTF-32 路径；`upper()`、`lower()` 等复杂 Unicode 转换仍进入 Python 字符串 runtime，并可通过 `runtime_stats()['string_callbacks']` 观察。

需要生成数组结果的 kernel 应由调用者传入 output，而不是返回内部 descriptor，参见 `examples/example_array_output.py`。shape/dtype 必须兼容且 output 必须可写；独立输出、精确 in-place 和共享 base 的不重叠 view 受支持。部分重叠输入/输出 view 不在契约内，因为顺序写入可能改变后续读取。

## 环境要求

* Python >= 3.12
* [uv](https://docs.astral.sh/uv/)
* llvmlite >= 0.49.0
* numpy >= 2.5.2

项目依赖已在 `pyproject.toml` 中声明并通过 `uv.lock` 锁定。推荐使用 `uv` 管理虚拟环境并执行项目指令（开发工作流无需使用 `requirements.txt`）。

## 基于 uv 的开发流程

```bash
uv sync --extra dev
uv run coverage run -m pytest -q
uv run coverage report
uv run pyright

```

`uv sync` 会根据锁定的依赖自动创建或更新本地 `.venv`。项目将以可编辑模式（Editable Mode）安装至该环境中，可以直接使用 `uv run` 运行示例脚本。

构建发布产物：

```bash
uv build

```

源码包（sdist）与 wheel 文件将输出至 `dist/` 目录。

## 使用指南

### 基础 JIT 编译

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

is_prime(169941229)   # 返回 1（针对 int 特化编译）

```

### 动态类型支持（多态调用）

同一个函数会根据不同的实参类型组合自动生成各自的特化版本。仅在遇到未出现过的类型组合时触发一次编译，之后直接读取缓存运行：

```python
@jit
def test(a, b):
    return a + b

print(test(114, 514))    # 628   (int64 特化)
print(test(11.4, 51.4))  # 62.8  (double/float64 特化)

```

### `@jit` 与 `@reg` 装饰器

* **`@jit`**：将函数标记为 JIT 编译目标。源码在函数定义时完成 AST 解析，并在首次调用时根据参数类型签名完成特化并编译为原生代码。适用于计算密集、循环或递归繁重的高频逻辑。
* **`@reg`**：将普通 Python 函数注册为可供 JIT 代码调用的回调函数（Callback）。该函数本身不会被 Pyjiting 编译，而是在原生执行期间跨越 ctypes 边界回调原 Python 函数。适用于日志记录、结果打印或与现有纯 Python 库集成。注册函数必须提供完整的标量或字符串类型注解（`int`、`float`、`bool`、`str`、`np.int32`、`np.int64`、`np.float32`、`np.float64`）；同名回调会按具体 callable 隔离。回调异常会由 dispatch frame 捕获并在最外层 Python 调用点重新抛出，原生用户逻辑不会使用占位返回值继续执行。

| 装饰器 | 作用 | 运行方式 | 适用场景 |
| --- | --- | --- | --- |
| `@jit` | 编译为原生机器码 | LLVM 生成的本地二进制代码 | 数值计算内核、核心循环、递归 |
| `@reg` | 注册供 JIT 调用的 Callable | 跨 ctypes 边界执行原 Python 实现 | 日志打印、终端展示、纯 Python 库调用 |

### 在 JIT 函数中调用 Python 回调

只要使用 `@reg` 注册并为参数及返回值提供完整的标量类型注解，即可从 JIT 编译的原生代码中无缝调用 Python 函数：

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

### 运行示例代码

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

## 性能测试

运行 `uv run benchmarks/benchmark.py --repeat 20` 可执行多轮冷启动、热调用、CPython 与 NumPy 对照测试；添加 `--json result.json` 可保存原始样本和环境元数据。脚本会校验每次结果并报告中位数、最小值和标准差。冷调用包含特化与 LLVM 编译，热调用使用已有原生特化。

测试代码位于 `examples/` 目录下。

```
测试环境：
CPU: 13th Gen Intel(R) Core(TM) i5-13600K @ 5.4GHz
内存: 64GB
操作系统: Windows 11 64-bit (10.0.26200)
Python 3.12.11 64-bit
LLVMLite 0.49.0
NumPy 2.5.2

```

| 基准测试 | JIT 耗时 | CPython 耗时 | 加速比 |
| --- | --- | --- | --- |
| `fib(40)` | 126.2 ms | 5056.4 ms | ~40.1x |
| `find_primes(100000)` | 909.4 ms | 11984.9 ms | ~13.2x |
| `is_prime(169941229)` | 340.5 ms | 4462.4 ms | ~13.1x |
| `loop(100000000)` | ~0 ms | 1871.1 ms | ∞（※见下方注记） |
| `test_while(100000000)` | ~0 ms | 1669.1 ms | ∞（※见下方注记） |

基准测试原始输出：

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

> **※注记**：简单的 `loop` / `while` 基准仅对计数器进行单纯累加，LLVM 在 `-O3` 优化下会通过常量折叠等将循环体完全消除（因此呈现为 "Infinite" 极限加速）。其余基准测试均执行了实质性的有效计算。

## 项目结构

```
pyjiting/
├── __init__.py   # 导出 jit、reg 接口
├── main.py       # @jit / @reg 装饰器、特化派发与调用缓存
├── parser.py     # Python AST -> Core AST 转换
├── ast.py        # Core AST 节点定义
├── infer.py      # Hindley-Milner 风格的类型推导实现
├── codegen.py    # Core AST -> LLVM IR 代码生成
├── ll_types.py   # ctypes 封装、符号修饰（Name Mangling）、参数封送
├── types.py      # 类型系统（BaseType、VarType、FuncType、ArrayType 等）
└── utils.py      # 合一（Unification）及类型替换等实用工具

```

## 局限性与已知约束

本项目主要用于学术研究与教学探索，并非面向生产环境的成熟 JIT 编译器。目前存在的限制如下：

* 未与垃圾回收机制（GC）集成；仅支持静态类型化的小型 Python 语法子集。
* `for` 支持 `range`、字符串和一维数组；多维数组迭代暂不支持。编译期判定 range 步长为常量 `0` 时将报错。
* 运行时除以零将抛出 `ZeroDivisionError`；动态步长为 `0` 的 `range` 将通过 JIT 错误 ABI 抛出 `ValueError`。
* 带有非 `None` 声明的 JIT 函数必须在所有控制流分支上均显式返回值（Return）。
* 整数幂运算（`**`）的指数必须为编译期常量（因动态负指数无法推导确定单一的静态返回类型）。
* 使用 `@reg` 注册的回调函数必须提供完整的标量或字符串类型注解。回调异常会在最外层 Python 调用点重新抛出，而不会跨越 ctypes ABI。
* 普通 `@jit` 支持不可变默认参数和外层关键字实参；仍不支持仅限关键字参数和可变长参数（`*args`, `**kwargs`），JIT 函数内部调用仍只支持位置实参。
* 暂不支持可变全局、List/Dict、星号或嵌套解包目标、动态 tuple 索引、tuple 修改/比较/迭代、任意 Python 对象、多维数组迭代、axis 及整组 NumPy 矢量化操作。

# 特别致谢

本项目受 [numpile](https://dev.stephendiehl.com/numpile/) 教程启发，并在此基础上继续完成。

衷心感谢小笠原武史教授（Prof. Takeshi Ogasawara）给予我的诸多启发与悉心指导。
