import os
import sys
import warnings

_here = os.path.dirname(os.path.abspath(__file__))
_lt_path = os.environ.get("LT_PATH", os.path.join(_here, "..", "python"))
sys.path.insert(0, _lt_path)

import litetorch as lt
import torch

fails = 0


def check(name, cond):
    global fails
    if cond:
        print("PASS:", name)
    else:
        print("FAIL:", name)
        fails += 1


def access_grad(t):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = t.grad
    return g, rec


x = lt.tensor([2.0])
x.requires_grad = True
y = x * x
g, rec = access_grad(y)
check("non-leaf grad None before backward, warns once", g is None and len(rec) == 1)
y.backward()
check("leaf x.grad == [4.0]", x.grad.to_vector() == [4.0])
g, rec = access_grad(y)
check("non-leaf grad None after backward, warns once", g is None and len(rec) == 1)
check("warning text mentions non-leaf", len(rec) == 1 and "not a leaf Tensor" in str(rec[0].message))

z = lt.tensor([2.0])
z.requires_grad = True
w = z * z
w.retain_grad()
w.backward()
g, rec = access_grad(w)
check("retained non-leaf grad == [1.0], no warning", g is not None and g.to_vector() == [1.0] and len(rec) == 0)
check("leaf z.grad == [4.0]", z.grad.to_vector() == [4.0])

s = lt.tensor([2.0])
s.requires_grad = True
s.backward()
check("leaf scalar own backward grad == [1.0]", s.grad.to_vector() == [1.0])

a = lt.tensor([2.0])
b = a * a
g, rec = access_grad(b)
check("no-requires-grad non-leaf grad None, no warning", g is None and len(rec) == 0)

m = lt.tensor([2.0])
m.requires_grad = True
n = m * m + m
n.backward()
check("multi-branch leaf accumulation == [5.0]", m.grad.to_vector() == [5.0])

c = lt.tensor([2.0])
c.requires_grad = True
d = c * c
d.backward(create_graph=True)
check("create_graph leaf grad == [4.0]", c.grad.to_vector() == [4.0])
g, rec = access_grad(d)
check("create_graph non-leaf grad None", g is None)

e = lt.tensor([2.0])
e.requires_grad = True
f = e * e
f.backward()
try:
    f.backward()
    check("second backward without retain_graph throws", False)
except RuntimeError as ex:
    check("second backward without retain_graph throws", "retain_graph" in str(ex))

h = lt.tensor([2.0])
h.requires_grad = True
k = h * h
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    k.grad = lt.tensor([7.0])
    got = k.grad
check("manual .grad assignment round-trips", got is not None and got.to_vector() == [7.0])

xt = torch.tensor([2.0], requires_grad=True)
yt = xt * xt
yt.backward()
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    tg = yt.grad
check("torch reference: non-leaf grad None + warns", tg is None and len(rec) == 1)

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
