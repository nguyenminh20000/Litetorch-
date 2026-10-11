import litetorch as lt
import math

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        print(f"[PASS] {name}")
        passed += 1
    else:
        print(f"[FAIL] {name}")
        failed += 1

cpu = lt.Device("cpu")

t = lt.Tensor.zeros([256, 512], cpu)
lt.nn.init.kaiming_uniform_(t)
v = t.to_vector()
fan_in = 512
bound = math.sqrt(6.0 / fan_in)
check("kaiming_uniform range", all(-bound - 1e-6 <= x <= bound + 1e-6 for x in v))
check("kaiming_uniform nonzero", any(abs(x) > 1e-9 for x in v))

t = lt.Tensor.zeros([256, 512], cpu)
lt.nn.init.kaiming_normal_(t)
v = t.to_vector()
mean = sum(v) / len(v)
std_exp = math.sqrt(2.0 / 512)
var = sum((x - mean) ** 2 for x in v) / len(v)
check("kaiming_normal std", abs(math.sqrt(var) - std_exp) < 0.05 * std_exp)

t = lt.Tensor.zeros([256, 512], cpu)
lt.nn.init.xavier_uniform_(t)
v = t.to_vector()
bound = math.sqrt(6.0 / (256 + 512))
check("xavier_uniform range", all(-bound - 1e-6 <= x <= bound + 1e-6 for x in v))

t = lt.Tensor.zeros([1000], cpu)
lt.nn.init.normal_(t, mean=5.0, std=2.0)
v = t.to_vector()
m = sum(v) / len(v)
check("normal_ mean", abs(m - 5.0) < 0.3)

t = lt.Tensor.zeros([100], cpu)
lt.nn.init.zeros_(t)
check("zeros_", all(x == 0.0 for x in t.to_vector()))
lt.nn.init.ones_(t)
check("ones_", all(x == 1.0 for x in t.to_vector()))
lt.nn.init.fill_(t, 3.5)
check("fill_", all(x == 3.5 for x in t.to_vector()))

lin = lt.nn.Linear(128, 64)
w = lin.weight.to_vector()
bound = math.sqrt(6.0 / 128)
check("Linear kaiming init range", all(-bound - 1e-6 <= x <= bound + 1e-6 for x in w))

p = [lt.Tensor.zeros([10], cpu)]
opt = lt.optim.AdamW(p, lr=0.001)
sched = lt.optim.lr_scheduler.LinearWarmup(opt, warmup_steps=100, total_steps=1000)
lrs = []
for _ in range(1000):
    sched.step()
    lrs.append(opt.get_lr())
check("warmup step1", abs(lrs[0] - 0.001 * 1 / 100) < 1e-9)
check("warmup step100 = base", abs(lrs[99] - 0.001) < 1e-9)
check("warmup monotonic up", all(lrs[i] <= lrs[i+1] + 1e-12 for i in range(99)))
check("decay monotonic down", all(lrs[i] >= lrs[i+1] - 1e-12 for i in range(100, 999)))
check("final lr = 0", abs(lrs[999]) < 1e-9)
mid = lrs[550]
exp = 0.001 * (1 - 451 / 900)
check("decay midpoint", abs(mid - exp) < 1e-9)

print(f"== SUMMARY: {passed} PASS, {failed} FAIL ==")
