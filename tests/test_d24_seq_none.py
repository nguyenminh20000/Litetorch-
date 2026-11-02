import sys

def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'} {name} {extra}")
    return cond

passed = 0
failed = 0

def run(name, cond, extra=""):
    global passed, failed
    if check(name, cond, extra):
        passed += 1
    else:
        failed += 1

import litetorch as lt

print("== baseline: Sequential(Linear x3) ==")
seq = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
x = lt.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
run("baseline params 6", len(seq.parameters()) == 6, f"got {len(seq.parameters())}")
run("baseline state_dict 6 keys", len(seq.state_dict()) == 6, f"got {len(seq.state_dict())}")
run("baseline named_children 3", len(seq.named_children()) == 3)
run("baseline forward ok", seq(x) is not None)

print("== setattr(seq, '1', None) ==")
setattr(seq, '1', None)
try:
    _g1 = getattr(seq, '1')
    run("getattr 1 is None", _g1 is None, f"got {_g1!r}")
except Exception as e:
    run("getattr 1 is None", False, f"raised {type(e).__name__}: {e}")
run("hasattr 1 True", hasattr(seq, '1'))
run("named_children skips None", [k for k, _ in seq.named_children()] == ['0', '2'],
    f"got {[k for k, _ in seq.named_children()]}")
run("params 4 after None", len(seq.parameters()) == 4, f"got {len(seq.parameters())}")
run("state_dict 4 keys after None", len(seq.state_dict()) == 4, f"got {len(seq.state_dict())}")
try:
    out = seq(x)
    run("forward raises on None member", False, f"silently returned shape {list(out.shape)}")
except Exception as e:
    run("forward raises on None member", True, f"raised {type(e).__name__}: {e}")

print("== add() after None ==")
new = lt.nn.Linear(4, 4)
seq.add(new)
run("params 6 after add (lin0+lin2+new)", len(seq.parameters()) == 6,
    f"got {len(seq.parameters())}")
run("named_children names [0,2,3] after add", [k for k, _ in seq.named_children()] == ['0', '2', '3'],
    f"got {[k for k, _ in seq.named_children()]}")
try:
    out = seq(x)
    run("forward still raises after add (None slot kept)", False,
        f"silently returned shape {list(out.shape)}")
except Exception as e:
    run("forward still raises after add (None slot kept)", True, f"raised {type(e).__name__}")

print("== reassign None slot with a module ==")
seq2 = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
setattr(seq2, '1', None)
fill = lt.nn.Linear(4, 4)
setattr(seq2, '1', fill)
run("reassign fills slot: params 6", len(seq2.parameters()) == 6, f"got {len(seq2.parameters())}")
run("reassign fills slot: named_children [0,1,2]",
    [k for k, _ in seq2.named_children()] == ['0', '1', '2'],
    f"got {[k for k, _ in seq2.named_children()]}")
run("reassign fills slot: getattr is fill", getattr(seq2, '1') is fill)
try:
    out = seq2(x)
    run("forward ok after fill", out is not None)
except Exception as e:
    run("forward ok after fill", False, f"raised {type(e).__name__}: {e}")

print("== reassign None slot with non-module must raise TypeError ==")
seq3 = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
setattr(seq3, '1', None)
try:
    setattr(seq3, '1', 42)
    run("non-module reassign raises TypeError", False, f"silently set to {getattr(seq3, '1')!r}")
except TypeError as e:
    run("non-module reassign raises TypeError", True, f"raised TypeError: {e}")
except Exception as e:
    run("non-module reassign raises TypeError", False, f"raised {type(e).__name__}: {e}")

print("== delattr after None removes the slot ==")
seq4 = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
setattr(seq4, '1', None)
try:
    delattr(seq4, '1')
    run("delattr after None succeeds", True)
except Exception as e:
    run("delattr after None succeeds", False, f"raised {type(e).__name__}: {e}")
run("hasattr False after del", not hasattr(seq4, '1'))
try:
    getattr(seq4, '1')
    run("getattr after del raises AttributeError", False)
except AttributeError:
    run("getattr after del raises AttributeError", True)
except Exception as e:
    run("getattr after del raises AttributeError", False, f"raised {type(e).__name__}")
run("params 4 after del", len(seq4.parameters()) == 4, f"got {len(seq4.parameters())}")
run("named_children [0,2] after del", [k for k, _ in seq4.named_children()] == ['0', '2'],
    f"got {[k for k, _ in seq4.named_children()]}")

print("== delattr then add replaces (no-regression) ==")
seq5 = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
delattr(seq5, '1')
seq5.add(lt.nn.Linear(4, 4))
run("delattr+add names [0,2]", [k for k, _ in seq5.named_children()] == ['0', '2'],
    f"got {[k for k, _ in seq5.named_children()]}")
run("delattr+add params 4", len(seq5.parameters()) == 4, f"got {len(seq5.parameters())}")
try:
    out = seq5(x)
    run("delattr+add forward ok", out is not None)
except Exception as e:
    run("delattr+add forward ok", False, f"raised {type(e).__name__}: {e}")

print("== None then delattr then add replaces (guards position-erase) ==")
seq6 = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.Linear(4, 4), lt.nn.Linear(4, 4)])
setattr(seq6, '1', None)
try:
    delattr(seq6, '1')
    run("none+del delattr succeeds", True)
except Exception as e:
    run("none+del delattr succeeds", False, f"raised {type(e).__name__}: {e}")
seq6.add(lt.nn.Linear(4, 4))
run("none+del+add names [0,2]", [k for k, _ in seq6.named_children()] == ['0', '2'],
    f"got {[k for k, _ in seq6.named_children()]}")
run("none+del+add params 4", len(seq6.parameters()) == 4, f"got {len(seq6.parameters())}")
try:
    out = seq6(x)
    run("none+del+add forward ok", out is not None)
except Exception as e:
    run("none+del+add forward ok", False, f"raised {type(e).__name__}: {e}")

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
