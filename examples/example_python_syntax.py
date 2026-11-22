from pyjiting import jit, jit_from_source, runtime_stats


@jit
def rounded(values, *, digits=2):
    return (round(value, digits) for value in values)


@jit
def report(*values, **options):
    rows = list(rounded(values, **options))
    highest = max(rows, default=0.0)
    return f"{len(rows)} readings, highest {highest:.2f}"


label = jit_from_source("""
    def label(value, /, *, width=5):
        return f'{value:>{width}}'
""")

assert label('ok', width=4) == '  ok'
print(report(2.675, 4.126, digits=2))
print(runtime_stats(report)['backend'], runtime_stats(rounded)['backend'])
