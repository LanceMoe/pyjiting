from pyjiting import autojit


@autojit
def add():
    return True

print(add())
