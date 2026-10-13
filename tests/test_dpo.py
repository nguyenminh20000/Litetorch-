import math
import sys
sys.path.insert(0, "python")

import litetorch as lt
from litetorch import nn
from litetorch.nn import functional as F

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dpo_ref(pc, pr, rc, rr, beta):
    return [-math.log(sigmoid(beta * (a - b - (c - d))))
            for a, b, c, d in zip(pc, pr, rc, rr)]

def orpo_ref(pc, pr):
    def log_odds(lp):
        p = math.exp(lp)
        return lp - math.log(max(1e-7, 1.0 - p))
    return [-math.log(sigmoid(log_odds(a) - log_odds(b)))
            for a, b in zip(pc, pr)]

print("== DPO loss vs reference ==")
pc = [-1.2, -2.5, -0.8, -3.1]
pr = [-2.0, -1.5, -2.2, -1.9]
rc = [-1.0, -2.0, -1.5, -2.8]
rr = [-1.8, -1.2, -2.0, -1.5]
beta = 0.5
ref = dpo_ref(pc, pr, rc, rr, beta)

tpc = lt.tensor(pc, requires_grad=True)
tpr = lt.tensor(pr, requires_grad=True)
trc = lt.tensor(rc)
trr = lt.tensor(rr)

loss = F.dpo_loss(tpc, tpr, trc, trr, beta=beta)
check("dpo mean", approx(loss.item(), sum(ref) / len(ref), 1e-4))

loss_none = F.dpo_loss(tpc, tpr, trc, trr, beta=beta, reduction="none")
check("dpo none[0]", approx(loss_none[0].item(), ref[0], 1e-4))
check("dpo none[3]", approx(loss_none[3].item(), ref[3], 1e-4))

loss_sum = F.dpo_loss(tpc, tpr, trc, trr, beta=beta, reduction="sum")
check("dpo sum", approx(loss_sum.item(), sum(ref), 1e-4))

print("== DPOLoss module ==")
mod = nn.DPOLoss(beta=beta)
mloss = mod(tpc, tpr, trc, trr)
check("module == functional", approx(mloss.item(), loss.item(), 1e-6))
check("module attrs", mod.beta == beta and mod.reference_free == False)

print("== gradients ==")
tpc2 = lt.tensor(pc, requires_grad=True)
tpr2 = lt.tensor(pr, requires_grad=True)
l2 = F.dpo_loss(tpc2, tpr2, trc, trr, beta=beta)
l2.backward()
check("chosen grad exists", tpc2.grad is not None)
check("rejected grad exists", tpr2.grad is not None)
check("chosen grad negative (loss decreases as chosen logp rises)",
      tpc2.grad[0].item() < 0)
check("rejected grad positive", tpr2.grad[0].item() > 0)
eps = 1e-4
tpc3 = lt.tensor([pc[0] + eps] + pc[1:], requires_grad=False)
tpr3 = lt.tensor(pr, requires_grad=False)
lp = F.dpo_loss(tpc3, tpr3, trc, trr, beta=beta).item()
tpc4 = lt.tensor([pc[0] - eps] + pc[1:], requires_grad=False)
lm = F.dpo_loss(tpc4, tpr3, trc, trr, beta=beta).item()
fd = (lp - lm) / (2 * eps)
check("grad matches finite diff", approx(tpc2.grad[0].item(), fd, 1e-2))

print("== reference-free (ORPO-style) ==")
oref = orpo_ref(pc, pr)
oloss = F.dpo_loss(tpc, tpr, reference_free=True)
check("orpo mean", approx(oloss.item(), sum(oref) / len(oref), 1e-4))
oloss_none = F.dpo_loss(tpc, tpr, reference_free=True, reduction="none")
check("orpo none[1]", approx(oloss_none[1].item(), oref[1], 1e-4))
omod = nn.DPOLoss(reference_free=True)
check("orpo module == functional", approx(omod(tpc, tpr).item(), oloss.item(), 1e-6))
tpc5 = lt.tensor(pc, requires_grad=True)
ol = F.dpo_loss(tpc5, tpr, reference_free=True)
ol.backward()
check("orpo grad flows", tpc5.grad is not None and tpc5.grad[0].item() < 0)

print("== error cases ==")
def expect_throw(name, fn):
    try:
        fn()
        check(name, False)
    except Exception:
        check(name, True)

expect_throw("missing ref throws", lambda: F.dpo_loss(tpc, tpr, beta=beta))
expect_throw("bad reduction throws",
             lambda: F.dpo_loss(tpc, tpr, trc, trr, beta=beta, reduction="bad"))
expect_throw("bad beta throws",
             lambda: F.dpo_loss(tpc, tpr, trc, trr, beta=0.0))
expect_throw("module bad beta throws", lambda: nn.DPOLoss(beta=-1.0))
expect_throw("single-arg forward throws", lambda: mod(tpc))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
