# Changelog

All notable changes to Pyjiting are documented in this file.

## 0.3.0 - 2026-08-16

### Usability

- Added Python signature binding for keyword calls and immutable default arguments.
- Added explicit specialization warmup, per-function statistics, specialization
  inspection and LLVM IR diagnostics.
- Preserved decorated callable types through `ParamSpec` overloads.
- Added structured fallback warnings with configurable once/always/ignore policies,
  while keeping resource and internal runtime failures non-fallbackable.

### Runtime

- Added isolated `JITContext` engines with close semantics and specialization/module
  budgets.
- Added deterministic compilation-failure caching and observable string runtime
  callback counters.
- Added compile-wait, registered-callback and lightweight failure-detail statistics.
- Made context resource budgets atomic across in-flight compilations and released
  context-owned registered callbacks on close.
- Lowered Unicode string comparisons to native UTF-32 lexicographic loops.
- Added multidimensional whole-array `sum`, `any` and `all` over arbitrary NumPy
  strides.

### Quality

- Added repeatable cold/warm benchmarks with JSON output and NumPy comparisons.
- Added scalar break-even reporting and separate native/runtime string benchmark cases.
- Added Hypothesis differential tests, branch coverage configuration and CI artifacts.
- Added a tag-driven PyPI Trusted Publishing workflow and single-source package version.
- Documented and tested caller-owned ndarray output, in-place, and aliasing behavior.

## 0.2.0 - 2026-08-15

### Safety

- Changed the ndarray ABI to preserve byte strides, alignment-safe loads/stores,
  item size and NumPy flags.
- Added native writeability guards so read-only arrays cannot be mutated.
- Rejected ndarray return values instead of exposing internal ctypes pointers.
- Added checked propagation for Python callback, string runtime and allocation
  failures.

### Correctness

- Isolated every `@jit` decoration as its own compilation unit, fixing closure
  and hot-reload cache collisions.
- Bound `@reg` callbacks by registration identity instead of short name.
- Added numeric branch type joins, `None`/Void returns and constant-true loop
  return analysis.
- Rejected mutual recursion before unsafe MCJIT symbol finalization.

### Runtime

- Added per-function specialization limits, `runtime_stats()` and
  `clear_cache()`.
- Split per-specialization compilation coordination from the MCJIT engine lock.
- Lowered string indexing and iteration directly over the UTF-32 descriptor.
- Added `jit.from_source()` / `jit_from_source()` and optional Python fallback.

### Quality

- Added cross-platform GitHub Actions for Python 3.12 and 3.13.
- Added safety, cache, callback, frontend, runtime and differential regressions.
- Expanded package metadata and synchronized legacy requirements.

## 0.1.0

- Initial research release with scalar, string, tuple and ndarray specialization.
