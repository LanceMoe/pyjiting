# Security policy

## Supported versions

Pyjiting is an experimental native-code compiler. Only the latest released
version receives safety fixes.

## Reporting a vulnerability

Please report memory-safety, code-generation or ABI issues privately through
GitHub's security advisory feature for the repository. Include a minimal input,
platform details, Python/llvmlite/NumPy versions and whether the issue reproduces
in a fresh process.

Do not use Pyjiting as a sandbox for untrusted Python source. Compiled functions
execute native code in the current process and intentionally have no isolation
boundary.
