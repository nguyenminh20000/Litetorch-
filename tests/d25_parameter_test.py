import sys
import litetorch as lt

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    return cond

ok = True
t = lt.tensor([1.0, 2.0, 3.0])
try:
    p = lt.nn.Parameter(t)
    ok &= check("Parameter(tensor, requires_grad=True default) constructs", True)
    ok &= check("values equal input", p.to_vector() == [1.0, 2.0, 3.0])
    ok &= check("requires_grad is True", bool(p.requires_grad))
except Exception as e:
    ok &= check("Parameter(tensor, requires_grad=True default) constructs", False)
    print("  raised:", e)

t2 = lt.tensor([4.0, 5.0])
try:
    p2 = lt.nn.Parameter(t2, False)
    ok &= check("Parameter(tensor, requires_grad=False) constructs", True)
    ok &= check("values equal input", p2.to_vector() == [4.0, 5.0])
    ok &= check("requires_grad is False", not bool(p2.requires_grad))
except Exception as e:
    ok &= check("Parameter(tensor, requires_grad=False) constructs", False)
    print("  raised:", e)

t3 = lt.tensor([1, 2], dtype=lt.DataType.INT64)
try:
    p3 = lt.nn.Parameter(t3, False)
    ok &= check("Parameter(INT64 tensor, requires_grad=False) constructs", True)
    ok &= check("INT64 values equal input", p3.to_vector() == [1.0, 2.0])
    ok &= check("INT64 requires_grad is False", not bool(p3.requires_grad))
except Exception as e:
    ok &= check("Parameter(INT64 tensor, requires_grad=False) constructs", False)
    print("  raised:", e)

try:
    b = lt.nn.Buffer(lt.tensor([7.0, 8.0]))
    ok &= check("Buffer(tensor) constructs", True)
    ok &= check("Buffer values equal input", b.to_vector() == [7.0, 8.0])
    ok &= check("Buffer requires_grad is False", not bool(b.requires_grad))
except Exception as e:
    ok &= check("Buffer(tensor) constructs", False)
    print("  raised:", e)

try:
    p = lt.nn.Parameter(lt.tensor([1.0, 2.0]))
    x = lt.tensor([3.0, 4.0])
    y = (p * x).sum()
    y.backward()
    ok &= check("backward populates .grad", p.grad is not None and p.grad.to_vector() == [3.0, 4.0])
except Exception as e:
    ok &= check("backward populates .grad", False)
    print("  raised:", e)

try:
    p = lt.nn.Parameter(lt.tensor([1.0, 2.0]))
    lin = lt.nn.Linear(2, 2)
    lin.register_parameter("weight", p)
    names = [n for n, _ in lin.named_parameters()]
    ok &= check("Parameter registers into module parameters()", "weight" in names)
except Exception as e:
    ok &= check("Parameter registers into module parameters()", False)
    print("  raised:", e)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
