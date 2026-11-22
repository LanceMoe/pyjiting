from pyjiting import jit


@jit
def add_new_tags(seen: set[str], candidates: list[str]) -> set[str]:
    normalized = {tag.strip().lower() for tag in candidates}
    added = normalized - seen
    seen |= added
    return added


seen = {'python'}
added = add_new_tags(seen, [' Python ', ' LLVM ', 'llvm', 'JIT'])
assert seen == {'python', 'llvm', 'jit'}
print(sorted(added))
