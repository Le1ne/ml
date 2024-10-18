def find_modified_max_argmax(L, f):
    t = [f(x) for x in L if type(x) == int]
    if t:
        m = max(t)
        return m, t.index(m)
    return ()
