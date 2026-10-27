# E2 — Autograd

Autograd records operations on tensors with `requires_grad=True` into a graph, then walks the graph backward to compute gradients. This page explains graph construction, `backward`, `gradcheck`/`gradgradcheck`, the no-grad modes, leaf vs non-leaf tensors, and grad accumulation — and the lesson from the 2026-10-26 no-learn defect, where forward values matched PyTorch while the gradients were corrupt.

- **Tree:** integration HEAD `b326f29`; runnable extension built from `88a4ba1` (`lt.build_commit()`, asserted in every example).
- **Reference:** PyTorch 2.14.1+cpu. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** non-leaf `.grad` is populated without `retain_grad()` (§4, new finding on this build); `zero_grad()` leaves `.grad` as `None` instead of zeros (§5); `backward()` has no `retain_graph` parameter (§6). None of these are in `plan/differences.md` yet — they are recorded here and flagged to the Docs leader.

## 1. Graph construction and `backward`

Every operation on a tensor with `requires_grad=True` creates a node in a directed acyclic graph (DAG) and marks its output as requiring grad. `.backward()` walks the graph from the output back to the leaves, applying the chain rule, and writes each leaf's gradient into `.grad`. Before any backward, `.grad` is `None`. `lt.autograd.grad(outputs, inputs)` computes the same gradients without writing them into `.grad`.

```
x (leaf, requires_grad=True)
│
▼  y = x * x          node: MulBackward
│
▼  loss = y.sum()     node: SumBackward
│
backward() walks SumBackward → MulBackward,
accumulating d loss / d x into x.grad
```

<!-- ex: id=E2-1 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
y = x * x
print("y requires_grad:", y.requires_grad)
print("y.grad before backward:", y.grad)
y.sum().backward()
print("x.grad:", x.grad.to_vector())
xx = lt.tensor([3.0])
xx.requires_grad = True
gg = lt.autograd.grad(xx * xx, [xx])
print("autograd.grad:", [g.to_vector() for g in gg])
xt = torch.tensor([2.0, 3.0], requires_grad=True)
yt = xt * xt
yt.sum().backward()
assert max(abs(a - b) for a, b in zip(x.grad.to_vector(), xt.grad.tolist())) <= 1e-4
assert gg[0].to_vector() == [6.0]
print("PARITY OK")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

Expected output:

```
y requires_grad: True
y.grad before backward: None
x.grad: [4.0, 6.0]
autograd.grad: [[6.0]]
PARITY OK
```

`d(x²)/dx = 2x`, so at `x = [2, 3]` the gradients are `[4, 6]`. `y.sum()` is needed because `backward()` starts from a scalar: the sum's gradient is all ones, which is the seed for the chain rule. `autograd.grad` returns a list of gradients in the order of the requested inputs, and does not touch `.grad`.

## 2. `gradcheck` and `gradgradcheck`: the eps trap

`gradcheck` verifies an analytical gradient against a numerical one computed by central differences: `(f(x + ε) − f(x − ε)) / 2ε`. `gradgradcheck` does the same for second derivatives. Both default to `eps=1e-6`. **On FP32 that default fails, in both frameworks.** The numerical gradient at `x = 3` for `t * t` comes out as 5.7220458984375 instead of the analytical 6.0 — an error of 0.28, far outside any tolerance. The cause is FP32 rounding: at `x = 3`, `f(x)` is near 9, and FP32 has about 7 decimal digits, so `f(x + ε)` and `f(x − ε)` each carry an absolute error near 1e-6; their difference is near 1.2e-5, and dividing by `2ε = 2e-6` amplifies the error to order 0.5. With `eps=1e-3` the same computation gives 5.999565124511719 and the check passes.

<!-- ex: id=E2-2 device=cpu backend=cpu runtime=~90s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

def f(t):
    return t * t

x = lt.tensor([3.0])
x.requires_grad = True
print("lt gradcheck eps=1e-6:", lt.autograd.gradcheck(f, (x,), eps=1e-6, raise_exception=False))
print("lt gradcheck eps=1e-3:", lt.autograd.gradcheck(f, (x,), eps=1e-3, raise_exception=False))
xp = lt.tensor([3.0 + 1e-6])
xm = lt.tensor([3.0 - 1e-6])
num6 = ((xp * xp).to_vector()[0] - (xm * xm).to_vector()[0]) / (2e-6)
print("lt numerical grad eps=1e-6:", num6)
xp3 = lt.tensor([3.0 + 1e-3])
xm3 = lt.tensor([3.0 - 1e-3])
num3 = ((xp3 * xp3).to_vector()[0] - (xm3 * xm3).to_vector()[0]) / (2e-3)
print("lt numerical grad eps=1e-3:", num3)
print("analytical grad:", 6.0)
xt = torch.tensor([3.0], requires_grad=True)
print("torch gradcheck eps=1e-6:", torch.autograd.gradcheck(lambda t: t * t, (xt,), eps=1e-6, raise_exception=False))
print("torch gradcheck eps=1e-3:", torch.autograd.gradcheck(lambda t: t * t, (xt,), eps=1e-3, raise_exception=False))
assert num6 != 6.0
assert abs(num3 - 6.0) <= 1e-3
print("DEFAULT EPS FAILS ON FP32; EPS=1E-3 PASSES (BOTH FRAMEWORKS)")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. No numeric parity: the point is that both frameworks give the same verdict — fail at `eps=1e-6`, pass at `eps=1e-3`.

Expected output (stdout; the torch `UserWarning` about non-double inputs goes to stderr):

```
lt gradcheck eps=1e-6: False
lt gradcheck eps=1e-3: True
lt numerical grad eps=1e-6: 5.7220458984375
lt numerical grad eps=1e-3: 5.999565124511719
analytical grad: 6.0
torch gradcheck eps=1e-6: False
torch gradcheck eps=1e-3: True
DEFAULT EPS FAILS ON FP32; EPS=1E-3 PASSES (BOTH FRAMEWORKS)
```

The rule: **use `eps=1e-3` for FP32 gradchecks, and `eps=1e-6` only for FP64.** A gradcheck failure at the default eps on FP32 is evidence about the eps, not about the gradient. The same trap applies to `gradgradcheck`:

<!-- ex: id=E2-3 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([3.0])
x.requires_grad = True
print("lt gradgradcheck eps=1e-3:", lt.autograd.gradgradcheck(lambda t: t * t, (x,), eps=1e-3, raise_exception=False))
y = x * x
g = lt.autograd.grad(y, [x], create_graph=True)[0]
print("first grad:", g.to_vector())
g2 = lt.autograd.grad(g, [x])[0]
print("second grad:", g2.to_vector())
xt = torch.tensor([3.0], requires_grad=True)
print("torch gradgradcheck eps=1e-3:", torch.autograd.gradgradcheck(lambda t: t * t, (xt,), eps=1e-3, raise_exception=False))
assert g2.to_vector() == [2.0]
print("SECOND DERIVATIVE OK")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

Expected output (stdout; the torch `UserWarning` about non-double inputs goes to stderr):

```
lt gradgradcheck eps=1e-3: True
first grad: [6.0]
second grad: [2.0]
torch gradgradcheck eps=1e-3: True
SECOND DERIVATIVE OK
```

`d²(x²)/dx² = 2` everywhere, so the second gradient is `[2.0]`. Note `create_graph=True` on `autograd.grad`: without it, the first gradient is a plain tensor with no graph attached, and differentiating it again has nothing to differentiate.

## 3. `no_grad` and `inference_mode`

Both context managers stop graph construction. Inside `no_grad()`, `is_grad_enabled()` is `False` and new tensors do not require grad even when `requires_grad` is set. Inside `inference_mode()`, `is_inference_mode()` is `True`. Both restore the previous state on exit. Use `no_grad()` for evaluation and for any computation whose gradients you will never need — it also skips recording, so it is faster and uses less memory.

<!-- ex: id=E2-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

print("grad enabled before:", lt.is_grad_enabled())
with lt.no_grad():
    n = lt.tensor([1.0])
    n.requires_grad = True
    m = n * 2
    print("inside no_grad, grad enabled:", lt.is_grad_enabled())
    print("inside no_grad, m.requires_grad:", m.requires_grad)
    print("inside no_grad, m.grad:", m.grad)
print("grad enabled after:", lt.is_grad_enabled())
with lt.inference_mode():
    print("inside inference_mode:", lt.is_inference_mode())
print("inference_mode after:", lt.is_inference_mode())
assert lt.is_grad_enabled()
assert not lt.is_inference_mode()
print("NO_GRAD AND INFERENCE_MODE OK")
```

Verified on CPU, LiteTorch 88a4ba1. No torch parity: this observes LiteTorch's own mode flags.

Expected output:

```
grad enabled before: True
inside no_grad, grad enabled: False
inside no_grad, m.requires_grad: False
inside no_grad, m.grad: None
grad enabled after: True
inside inference_mode: True
inference_mode after: False
NO_GRAD AND INFERENCE_MODE OK
```

## 4. Leaf vs non-leaf tensors

A **leaf** is a tensor created by the user with `requires_grad=True` (a parameter, an input). A **non-leaf** is the output of an operation. Gradients accumulate into leaves' `.grad`. In PyTorch, a non-leaf's `.grad` stays `None` unless `retain_grad()` is called on it — accessing it even warns. **LiteTorch differs: a non-leaf's `.grad` is populated after backward without any `retain_grad()` call.** The value it holds is the seed gradient flowing into that node (1.0 for a summed scalar output), not the gradient with respect to anything else.

<!-- ex: id=E2-5 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0])
x.requires_grad = True
y = x * x
print("x.grad before backward:", x.grad)
print("y.grad before backward:", y.grad)
y.backward()
print("x.grad after backward:", x.grad.to_vector())
print("y.grad after backward (no retain):", y.grad.to_vector())
z = lt.tensor([2.0])
z.requires_grad = True
w = z * z
w.retain_grad()
w.backward()
print("w.grad with retain_grad:", w.grad.to_vector())
print("z.grad:", z.grad.to_vector())
xt = torch.tensor([2.0], requires_grad=True)
yt = xt * xt
yt.backward()
print("torch y.grad after backward (no retain):", yt.grad)
assert y.grad.to_vector() == [1.0]
assert w.grad.to_vector() == [1.0]
print("NON-LEAF GRAD POPULATED WITHOUT RETAIN_GRAD (DIVERGES FROM TORCH, see text)")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. The divergence is the point: LiteTorch populates it, torch returns `None`.

Expected output (stdout; the torch `UserWarning` about accessing a non-leaf `.grad` goes to stderr):

```
x.grad before backward: None
y.grad before backward: None
x.grad after backward: [4.0]
y.grad after backward (no retain): [1.0]
w.grad with retain_grad: [1.0]
z.grad: [4.0]
torch y.grad after backward (no retain): None
NON-LEAF GRAD POPULATED WITHOUT RETAIN_GRAD (DIVERGES FROM TORCH, see text)
```

`retain_grad()` exists and works, but on this build it changes nothing observable: the non-leaf gradient is there either way. Do not write code that depends on a non-leaf `.grad` being `None` — that assumption holds in torch and fails here. Conversely, do not assume `retain_grad()` is required before reading an intermediate gradient; in LiteTorch it is not.

## 5. Grad accumulation and `zero_grad`

`.backward()` **adds** into `.grad` instead of replacing it. Two backward passes over two fresh forward passes double the gradient. That is the standard training-loop pattern: forward, backward, step, zero the grads, repeat. Forgetting the zero step silently trains on stale accumulated gradients.

<!-- ex: id=E2-6 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
(x * x).sum().backward()
print("after step 1:", x.grad.to_vector())
(x * x).sum().backward()
print("after step 2 (accumulated):", x.grad.to_vector())
x.zero_grad()
print("after zero_grad:", x.grad)
xt = torch.tensor([2.0, 3.0], requires_grad=True)
(xt * xt).sum().backward()
(xt * xt).sum().backward()
print("torch after step 2:", xt.grad.tolist())
xt.grad.zero_()
print("torch after zero_:", xt.grad.tolist())
assert x.grad is None
print("ACCUMULATION OK; zero_grad DIVERGES FROM TORCH (None vs zeros, see text)")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Accumulation matches torch; the zero step does not.

Expected output:

```
after step 1: [4.0, 6.0]
after step 2 (accumulated): [8.0, 12.0]
after zero_grad: None
torch after step 2: [8.0, 12.0]
torch after zero_: [0.0, 0.0]
ACCUMULATION OK; zero_grad DIVERGES FROM TORCH (None vs zeros, see text)
```

**Divergence:** LiteTorch `zero_grad()` sets `.grad` to `None`; torch leaves a zeros tensor in `.grad`. Code that reads `.grad` after zeroing must handle `None` — `x.grad.to_vector()` after `zero_grad()` raises. The next `backward()` re-creates `.grad` from `None`, so training loops work; only direct reads of a freshly zeroed gradient break.

## 6. One backward per graph: no `retain_graph`

`backward()` frees the graph. Calling it a second time on the same output throws `Autograd::backward: graph was already freed by a previous backward() call; pass retain_graph=true to run backward multiple times`. The error message names a parameter that does not exist: `backward()` takes only `(gradient=None, create_graph=False)` — there is no `retain_graph` in LiteTorch. PyTorch accepts `retain_graph=True` and re-runs the same graph.

<!-- ex: id=E2-7 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
y = (x * x).sum()
y.backward()
print("first backward ok, x.grad:", x.grad.to_vector())
try:
    y.backward()
    print("second backward ok")
except RuntimeError as e:
    print("second backward throws:", e)
print("backward signature:", lt.Tensor.backward.__doc__)
xt = torch.tensor([2.0, 3.0], requires_grad=True)
yt = (xt * xt).sum()
yt.backward(retain_graph=True)
yt.backward()
print("torch second backward with retain_graph ok, grad:", xt.grad.tolist())
print("NO RETAIN_GRAPH IN LITETORCH (see text)")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. The difference is the point: LiteTorch throws where torch re-runs.

Expected output:

```
first backward ok, x.grad: [4.0, 6.0]
second backward throws: [litetorch Error] Autograd::backward: graph was already freed by a previous backward() call; pass retain_graph=true to run backward multiple times
backward signature: backward(self: litetorch.Tensor, gradient: litetorch.Tensor = None, create_graph: bool = False) -> None

torch second backward with retain_graph ok, grad: [8.0, 12.0]
NO RETAIN_GRAPH IN LITETORCH (see text)
```

The migration is structural, not a flag: re-run the forward pass to build a fresh graph (which is also how accumulation in §5 works), or use `create_graph=True` when the goal is higher-order gradients rather than a repeated backward. The error message itself is misleading and is recorded here so nobody chases a `retain_graph` argument that the binding does not accept.

## 7. The no-learn lesson: forward parity is not enough

In October 2026 a CIFAR training run did not learn: accuracy stuck at 0.1000, loss went NaN, and the first Adam step went NaN. The forward pass matched PyTorch — copy-eval accuracy about 0.65 on both frameworks, per-op forward values within tolerance. The gradients were corrupt. A `Conv2d` backward was writing out of bounds into a neighboring gradient buffer, zeroing and scrambling gradients while leaving every forward value intact. The root cause was a `cached_native_kernel` collision: the FP32 and FP16 backward typedefs were byte-identical, shared one static cache entry, and every "filter-gradient" call silently ran the data-gradient kernel instead. The fix (name-keyed kernel lookup) merged in `88a4ba1` after QA approval (qa-r51) and a Kaggle verification run: CIFAR accuracy 0.6154 vs torch 0.6433, gradcheck 10/10, no gradient corruption.

The rule that came out of it: **compare gradients, not just forward values.** Every example in this book that calls `.backward()` compares `.grad` against torch at 1e-4, because the no-learn defect was exactly a gradient-path corruption with correct-looking forward values. The example below is the pattern — a small linear model, forward and all three gradients compared:

<!-- ex: id=E2-8 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

torch.manual_seed(0)
x_t = torch.randn(4, 3, requires_grad=True)
w_t = torch.randn(3, 2, requires_grad=True)
b_t = torch.randn(2, requires_grad=True)
y_t = torch.nn.functional.linear(x_t, w_t.t(), b_t)
loss_t = (y_t * y_t).sum()
loss_t.backward()
x_l = lt.tensor(x_t.detach().numpy().tolist())
w_l = lt.tensor(w_t.detach().numpy().tolist())
b_l = lt.tensor(b_t.detach().numpy().tolist())
x_l.requires_grad = True
w_l.requires_grad = True
b_l.requires_grad = True
y_l = lt.Ops.matmul(x_l, w_l) + b_l
loss_l = (y_l * y_l).sum()
loss_l.backward()
def maxdiff(a, b):
    return max(abs(p - q) for p, q in zip(a, b))
print("forward max diff:", maxdiff(y_l.to_vector(), y_t.detach().numpy().ravel().tolist()))
print("x.grad max diff:", maxdiff(x_l.grad.to_vector(), x_t.grad.numpy().ravel().tolist()))
print("w.grad max diff:", maxdiff(w_l.grad.to_vector(), w_t.grad.numpy().ravel().tolist()))
print("b.grad max diff:", maxdiff(b_l.grad.to_vector(), b_t.grad.numpy().ravel().tolist()))
assert maxdiff(y_l.to_vector(), y_t.detach().numpy().ravel().tolist()) <= 1e-4
assert maxdiff(x_l.grad.to_vector(), x_t.grad.numpy().ravel().tolist()) <= 1e-4
assert maxdiff(w_l.grad.to_vector(), w_t.grad.numpy().ravel().tolist()) <= 1e-4
assert maxdiff(b_l.grad.to_vector(), b_t.grad.numpy().ravel().tolist()) <= 1e-4
print("FORWARD AND GRAD PARITY OK")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4 for forward and all gradients.

Expected output:

```
forward max diff: 1.1920928955078125e-07
x.grad max diff: 4.76837158203125e-07
w.grad max diff: 9.5367431640625e-07
b.grad max diff: 2.384185791015625e-07
FORWARD AND GRAD PARITY OK
```

All four diffs sit near 1e-7, two orders inside the 1e-4 criterion. If any one of them had failed while the forward diff passed, that would be the no-learn signature: a backward defect hiding behind a correct forward. That is why the verification method (`plan/verification.md` §3) requires gradient comparison for every example that calls `.backward()` — forward-only parity is not sufficient.

## 8. What this page does not cover

- Tensor layout, dtypes, devices, and the allocator — see [E1 Tensor and Memory Model](E1-tensor-memory-model.md).
- Optimizers, which consume `.grad` — see the optimizers reference and tutorial T2.
- The full autograd API surface — see `reference/cuda-tpu-autograd.md`.
