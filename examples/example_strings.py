from pyjiting import jit


@jit
def summarize(text: str, needle: str) -> str:
    if text.startswith(needle):
        return '[' + text + '] contains ' + needle * text.count(needle)
    return text[1:-1]


print(summarize('你好你好', '你好'))
print(summarize('pyjiting', 'jit'))
