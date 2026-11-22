from pyjiting import jit


@jit
def clean_scores(scores):
    cleaned = [score for score in scores if score is not None]
    cleaned.sort(reverse=True)
    scores[:] = cleaned
    return scores


scores = [12, None, 7, 20, None]
result = clean_scores(scores)

assert result is scores
assert scores == [20, 12, 7]
print(scores)
