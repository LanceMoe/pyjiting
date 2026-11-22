from pyjiting import jit


@jit
def count_votes(totals: dict[str, int], votes: list[str]) -> dict[str, int]:
    for name in votes:
        totals[name] = totals.get(name, 0) + 1
    return {name: count for name, count in totals.items() if count >= 2}


totals = {'Ada': 1}
leaders = count_votes(totals, ['Ada', 'Lin', 'Lin', 'Sam'])
assert totals == {'Ada': 2, 'Lin': 2, 'Sam': 1}
print(leaders)
