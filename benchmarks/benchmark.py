"""Repeatable cold/warm benchmark with optional JSON output."""

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns

import llvmlite
import numpy as np

from pyjiting import clear_cache, jit


@jit
def modular_sum(limit):
    total = 0
    for value in range(limit):
        total += value % 97
    return total


def modular_sum_python(limit):
    total = 0
    for value in range(limit):
        total += value % 97
    return total


@jit
def strided_sum(values):
    return sum(values)


def measure(function, repeat, expected=None):
    samples = []
    for _ in range(repeat):
        started = perf_counter_ns()
        result = function()
        elapsed = perf_counter_ns() - started
        if expected is None:
            expected = result
        elif result != expected:
            raise AssertionError(f'benchmark result changed: {result!r} != {expected!r}')
        samples.append(elapsed)
    return {
        'result': float(expected) if isinstance(expected, np.floating) else expected,
        'median_ns': int(statistics.median(samples)),
        'min_ns': min(samples),
        'stdev_ns': int(statistics.stdev(samples)) if len(samples) > 1 else 0,
        'samples_ns': samples,
    }


def run(repeat):
    limit = 100_000
    clear_cache(modular_sum)
    expected_scalar = modular_sum_python(limit)
    cold = measure(lambda: modular_sum(limit), 1, expected_scalar)
    warm = measure(lambda: modular_sum(limit), repeat, expected_scalar)
    python = measure(lambda: modular_sum_python(limit), repeat, expected_scalar)

    values = np.arange(200_000, dtype=np.float64)[::-3]
    clear_cache(strided_sum)
    expected_array = np.sum(values)
    array_cold = measure(lambda: strided_sum(values), 1, expected_array)
    array_warm = measure(lambda: strided_sum(values), repeat, expected_array)
    numpy = measure(lambda: np.sum(values), repeat, expected_array)

    return {
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'processor': platform.processor(),
            'llvmlite': llvmlite.__version__,
            'numpy': np.__version__,
        },
        'repeat': repeat,
        'cases': {
            'scalar_cold': cold,
            'scalar_warm': warm,
            'scalar_cpython': python,
            'strided_array_cold': array_cold,
            'strided_array_warm': array_warm,
            'strided_array_numpy': numpy,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error('--repeat must be at least 2')
    report = run(args.repeat)
    for name, result in report['cases'].items():
        print(f"{name}: median={result['median_ns'] / 1e6:.3f} ms "
              f"min={result['min_ns'] / 1e6:.3f} ms result={result['result']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
