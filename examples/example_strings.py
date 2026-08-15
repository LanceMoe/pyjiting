from pyjiting import jit


@jit
def summarize(text: str, needle: str) -> str:
    if text.startswith(needle):
        return '[' + text + '] contains ' + needle * text.count(needle)
    if needle in text:
        return text.strip().replace(needle, needle.upper())
    return text[::-1]


print(summarize('你好你好', '你好'))
print(summarize('pyjiting', 'jit'))
print(summarize('unicode', 'xyz'))
