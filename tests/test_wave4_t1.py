import math
import sys
sys.path.insert(0, "python")

import litetorch as lt
from litetorch import nn, optim, distributions

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

print("== RNN/LSTM/GRU __call__ tuple ==")
rnn = nn.RNN(4, 8, num_layers=2)
x = lt.randn([5, 3, 4])
out = rnn(x)
check("rnn call returns tuple", isinstance(out, tuple) and len(out) == 2)
check("rnn output shape", list(out[0].shape) == [5, 3, 8])
check("rnn h_n shape", list(out[1].shape) == [2, 3, 8])

lstm = nn.LSTM(4, 8, num_layers=2)
out = lstm(x)
check("lstm call returns tuple", isinstance(out, tuple) and len(out) == 2)
check("lstm output shape", list(out[0].shape) == [5, 3, 8])
check("lstm h_n/c_n nested", isinstance(out[1], tuple) and len(out[1]) == 2)
check("lstm h_n shape", list(out[1][0].shape) == [2, 3, 8])
check("lstm c_n shape", list(out[1][1].shape) == [2, 3, 8])

gru = nn.GRU(4, 8, num_layers=2)
out = gru(x)
check("gru call returns tuple", isinstance(out, tuple) and len(out) == 2)
check("gru output shape", list(out[0].shape) == [5, 3, 8])
check("gru h_n shape", list(out[1].shape) == [2, 3, 8])

print("== Normal.log_prob broadcast ==")
loc = lt.tensor([0.0])
scale = lt.tensor([1.0])
n = distributions.Normal(loc, scale)
vals = lt.tensor([0.0, 1.0, -1.0])
lp = n.log_prob(vals)
check("scalar params vector input shape", list(lp.shape) == [3])
expected_0 = -0.5 * math.log(2 * math.pi)
check("log_prob(0) value", approx(lp[0].item(), expected_0, 1e-5))
expected_1 = -0.5 - 0.5 * math.log(2 * math.pi)
check("log_prob(1) value", approx(lp[1].item(), expected_1, 1e-5))

loc2 = lt.tensor([0.0, 1.0, 2.0])
n2 = distributions.Normal(loc2, scale)
lp2 = n2.log_prob(vals)
check("vector params shape", list(lp2.shape) == [3])

print("== Adadelta ==")
p = lt.tensor([1.0, 2.0, 3.0], requires_grad=True)
opt = optim.Adadelta([p], lr=1.0, rho=0.9)
p.grad = lt.tensor([0.1, 0.2, 0.3])
opt.step()
check("adadelta updates params", not approx(p[0].item(), 1.0))
check("adadelta get_lr", approx(opt.get_lr(), 1.0))

print("== Adamax ==")
p2 = lt.tensor([1.0, 2.0, 3.0], requires_grad=True)
opt2 = optim.Adamax([p2], lr=0.002)
p2.grad = lt.tensor([0.1, 0.2, 0.3])
before = p2[0].item()
opt2.step()
check("adamax updates params", not approx(p2[0].item(), before))
check("adamax get_lr", approx(opt2.get_lr(), 0.002))

print("== Schedulers ==")
p3 = lt.tensor([1.0], requires_grad=True)
sgd = optim.SGD([p3], lr=0.1)

sched = optim.lr_scheduler.LinearLR(sgd, start_factor=0.5, end_factor=1.0, total_iters=4)
sched.step()
check("linearlr step1", approx(sgd.get_lr(), 0.1 * 0.625, 1e-6))
for _ in range(3):
    sched.step()
check("linearlr final", approx(sgd.get_lr(), 0.1, 1e-6))

sgd2 = optim.SGD([lt.tensor([1.0], requires_grad=True)], lr=0.1)
sched2 = optim.lr_scheduler.PolynomialLR(sgd2, total_iters=4, power=2.0)
sched2.step()
check("polyllr step1", approx(sgd2.get_lr(), 0.1 * 0.5625, 1e-6))

sgd3 = optim.SGD([lt.tensor([1.0], requires_grad=True)], lr=0.1)
sched3 = optim.lr_scheduler.CosineAnnealingWarmRestarts(sgd3, T_0=4, T_mult=1)
lrs = []
for _ in range(5):
    sched3.step()
    lrs.append(sgd3.get_lr())
check("cawr restart", approx(lrs[3], 0.1, 1e-6))
check("cawr decay", lrs[1] < lrs[0] and lrs[1] > 0)

sgd4 = optim.SGD([lt.tensor([1.0], requires_grad=True)], lr=0.1)
sched4 = optim.lr_scheduler.LambdaLR(sgd4, lr_lambda=lambda epoch: 0.9 ** epoch)
sched4.step()
check("lambdalr", approx(sgd4.get_lr(), 0.09, 1e-6))

sgd5 = optim.SGD([lt.tensor([1.0], requires_grad=True)], lr=0.1)
sched5 = optim.lr_scheduler.OneCycleLR(sgd5, max_lr=0.1, total_steps=10)
init_lr = sgd5.get_lr()
check("onecycle init", approx(init_lr, 0.1 / 25.0, 1e-9))
for _ in range(3):
    sched5.step()
mid_lr = sgd5.get_lr()
check("onecycle peak near max", mid_lr <= 0.1 and mid_lr > init_lr)
for _ in range(7):
    sched5.step()
final_lr = sgd5.get_lr()
check("onecycle final small", final_lr < 0.01)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
