# Pyjiting

Pyjiting is a experimental Python-JIT compiler, which is the product of my undergraduate thesis. The goal is to implement a light-weight miniature general-purpose Python JIT compiler.

## Functions that have been implemented so far

1. Backend uses llvmlite, support Python 3.9 and up, fix some errors in the numpile tutorial.
2. Manually register the function of calling Python in the JITed function based on TypeHints.
3. Function calls Python code(Requires manual registration, add `@reg`) also can be JIT-compiled.
4. Implement basic functions, such as Compare operators, Mathematical operators, etc.
5. Implement `if` expressions.
6. Implement `for` expressions, and allow `break`.
7. Implement recursion.


## Performance

You can find the source code of these test samples in the root directory.

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
```

# Special thanks

Inspired by [numpile](https://dev.stephendiehl.com/numpile/) tutorial and continue to work on this basis.

I am deeply grateful to professor Mr Takeshi Ogasawara gave me many inspirations and appropriate advice.
