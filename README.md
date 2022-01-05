# Pyjiting

Pyjiting is a experimental Python-JIT compiler, which is the product of my undergraduate thesis. The goal is to implement a light-weight miniature general-purpose Python JIT compiler.

## Functions that have been implemented so far

1. Use llvmlite as Backend, support Python 3.9 and up, fix some errors in the numpile tutorial.
2. Support calls the Python function(code in `.py`, need manually register, just add `@reg`) from the JITed function(llvm binary) based on TypeHints. (See `example_find_primes.py`)
3. Implement basic functions, such as Compare operators, Mathematical operators, etc.
4. Implement `if` expressions.
5. Implement `for` expressions, and allow `break`.
6. Implement `while` expressions.
7. Implement recursion.


## Performance

You can find the source code of these test samples in the root directory.

```
My test environment:
CPU: i7-8700K@4.8Ghz
Memory: 32GB DDR4 3200Mhz
OS: Windows 10 21H2 64bit
Python 3.9.9 64bit
LLVMLite 0.37.0
```

```
fib_jit(40) = 102334155 (cost time: 216.74108505249023 ms)
fib_nojit(40) = 102334155 (cost time: 27159.422159194946 ms)
rate: 125.31


find_primes_jit(100000) = 0 (cost time: 3669.861316680908 ms)
find_primes_nojit(100000) = 0 (cost time: 38279.1805267334 ms)
rate: 10.43

is_prime_jit(169941229) = True (cost time: 979.4812202453613 ms)
is_prime_nojit(169941229) = True (cost time: 13560.30797958374 ms)
rate: 13.84

loop_jit(100000000) = 200000000 (cost time: 0.0 ms)
loop_nojit(100000000) = 200000000 (cost time: 7256.179332733154 ms)
rate: Infinite

test_while_jit(100000000) = 100000000 (cost time: 0.0 ms)
test_while_nojit(100000000) = 100000000 (cost time: 7198.246292114258 ms)
rate: Infinite
```

# Special thanks

Inspired by [numpile](https://dev.stephendiehl.com/numpile/) tutorial and continue to work on this basis.

I am deeply grateful to professor Mr Takeshi Ogasawara gave me many inspirations and appropriate advice.
