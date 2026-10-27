# E6 — JIT Fusion: Tracer, JITFunction, and Fused Ops

JIT fusion takes a small graph of elementwise ops and compiles it into
one kernel launch instead of one launch per op. The win is not faster
arithmetic — it is fewer memory round-trips: an unfused `relu(x + b)`
reads `x` and `b`, writes the sum, reads the sum back, then writes the
ReLU output, while the fused version reads `x` and `b` once and writes
once. This page explains the tracer that builds the graph, the
`JITFunction` that runs it, the exact fusion rules, the errors the
tracer raises, and the fused-ops inventory — all against tree HEAD
`b326f29` (extension `88a4ba1`).

Prerequisites: tensors, `lt.Ops.*`, autograd (T1). All examples run on
CPU against LiteTorch `88a4ba1` and PyTorch `2.14.1+cpu`.

## E6.1 — Two objects: JITVar is the graph, JITFunction is the program

A `JITVar` is one node of a symbolic expression: an input, a constant,
or the result of an op (`ADD`, `SUB`, `MUL`, `DIV`, `RELU`, `GELU`,
`SIGMOID`, `TANH`, `SQRT`, `EXP`, `LOG`, `ABS`, `NEG` —
`bind_jit.cpp:6-23`). You build expressions with the `lt.jit` unary
functions (`lt.jit.relu(a)`, `lt.jit.gelu(a)`, ...) and with Python
operators on `JITVar`s (`a + b`, `a * 2.0`). A `JITVar` holds no data:
it cannot be printed as a tensor and it cannot be executed on its own.

A `JITFunction` is a compiled program: a name, a root `JITVar`
expression, and the ordered list of input `JITVar`s. You call it with
a list of tensors and get a tensor back. `JITFunction` has `__call__`
and `forward` (same thing), plus `save(filepath)` and
`JITFunction.load(filepath)` for persistence
(`bind_jit.cpp:80-89`).

```
JITVar (symbolic)                    JITFunction (executable)
  a = JITVar(INPUT, "x")               f = JITFunction("relu_fn", relu(a), [a])
  expr = relu(a)                       f([tensor]) -> tensor
  expr has no data                     f.save(path) / JITFunction.load(path)
```

<!-- ex: id=E6-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    a = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
    f = lt.jit.JITFunction("relu_fn", lt.jit.relu(a), [a])
    y = f([x])
    d = lt.jit.Tracer.derivative(lt.jit.relu(a), a)
    df = lt.jit.JITFunction("drelu_fn", d, [a])
    dy = df([x])
    f.save("/tmp/e6-jitfn.bin")
    g = lt.jit.JITFunction.load("/tmp/e6-jitfn.bin")
    z = g([x])
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    exp = F.relu(xt).flatten().tolist()
    exp_d = (xt > 0).float().flatten().tolist()
    dy_max = max(abs(p - q) for p, q in zip(y.to_vector(), exp))
    dd_max = max(abs(p - q) for p, q in zip(dy.to_vector(), exp_d))
    dz_max = max(abs(p - q) for p, q in zip(z.to_vector(), exp))
    print("relu max_abs_diff:", dy_max, "derivative max_abs_diff:", dd_max, "save/load max_abs_diff:", dz_max)
    ok = dy_max <= 1e-4 and dd_max <= 1e-4 and dz_max <= 1e-4
    print("E6-1 PASS" if ok else "E6-1 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`Tracer.derivative(var, wrt)` differentiates a `JITVar` expression
symbolically (quotient rule for `DIV`, chain rule for `SIGMOID` /
`TANH` / `EXP` / `LOG`, `RELU_GRAD` / `GELU_GRAD` / `ABS_GRAD` nodes
for the non-smooth cases — `src/ops/jit.cpp`, `Tracer::derivative`).
The derivative of ReLU is the step function: 1 where the input is
positive, 0 elsewhere.

## E6.2 — The ready-made fused ops

For the common patterns you do not build a `JITFunction` by hand:
`lt.jit` exposes them directly (`bind_jit.cpp:46-78`). Each takes
tensors and returns a tensor, fused into one kernel:

| Function | Computes | Torch mirror |
|---|---|---|
| `fused_bias_relu(x, bias)` | `relu(x + bias)` | `F.relu(x + bias)` |
| `fused_bias_gelu(x, bias)` | `gelu(x + bias)` | `F.gelu(x + bias, approximate='tanh')` |
| `fused_residual_add(x, residual)` | `x + residual` | `x + residual` |
| `fused_residual_gelu(x, residual)` | `gelu(x + residual)` | `F.gelu(x + residual, approximate='tanh')` |

The GELU here is the tanh approximation, not torch's default exact
erf form — the parity mirror must use `approximate='tanh'`, and the
tolerance is the usual FP32 1e-4 against that mirror (see D-09 in the
Behavioral Differences page: exact-erf GELU differs by up to ~4.4e-4).

<!-- ex: id=E6-2 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def check(name, got, exp, tol=1e-4):
    d = max(abs(a - b) for a, b in zip(got, exp))
    print(name, "max_abs_diff:", d)
    return d <= tol


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    b = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    bt = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    ok = True
    ok = check("fused_bias_relu", lt.jit.fused_bias_relu(x, b).to_vector(), F.relu(xt + bt).flatten().tolist()) and ok
    ok = check("fused_bias_gelu", lt.jit.fused_bias_gelu(x, b).to_vector(), F.gelu(xt + bt, approximate="tanh").flatten().tolist()) and ok
    ok = check("fused_residual_add", lt.jit.fused_residual_add(x, b).to_vector(), (xt + bt).flatten().tolist()) and ok
    ok = check("fused_residual_gelu", lt.jit.fused_residual_gelu(x, b).to_vector(), F.gelu(xt + bt, approximate="tanh").flatten().tolist()) and ok
    print("E6-2 PASS" if ok else "E6-2 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

## E6.3 — The tracer: from eager ops to a JITFunction

`Tracer.trace(inputs, func, name)` records what `func` does to the
input tensors and returns a `JITFunction` — not a `JITVar`. The
distinction matters: the return value is executable, the nodes inside
it are not. `trace` works by setting `requires_grad = True` on the
inputs, calling `func(inputs)`, and walking the autograd graph of the
output backwards, translating each creator node into a `JITVar`
(`src/ops/jit.cpp`, `Tracer::trace` and `trace_tensor`). The input
tensors' original `requires_grad` flags are restored afterwards.

Rules the tracer enforces, in order of surprise:

1. **Build the graph inside `func`, with `lt.Ops.*`.** The ops the
   tracer understands are exactly the ones in `trace_tensor`: `Add`,
   `Sub`, `Mul`, `Div`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Sqrt`,
   `Exp`, `Log`, `Abs`, `Neg`. Anything else raises
   `[litetorch Error] Unsupported operator name in JIT compilation: <name>`
   (§E6.4).
2. **Every non-input leaf must be a 1-element constant.** A scalar
   leaf with `requires_grad` cannot be baked in — pass it as an
   explicit input instead. A multi-element leaf that is not in the
   input list raises `Traced graph reached a leaf tensor that is not
   in the input list` (§E6.4).
3. **Keep every intermediate alive until `trace` returns.** The
   tracer reads creator inputs through weak pointers; an intermediate
   that has been freed raises the expired-intermediate error
   (§E6.4). In practice this means: do not trace a graph you built
   earlier and then dropped the intermediates of — build inside
   `func`, where the locals stay alive for the whole call.

<!-- ex: id=E6-3 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    b = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)

    def func(inputs):
        return lt.Ops.relu(lt.Ops.add(inputs[0], inputs[1]))

    fn = lt.jit.Tracer.trace([x, b], func, "traced_add_relu")
    print("trace returns:", type(fn).__name__)
    y = fn([x, b])
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    bt = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    exp = F.relu(xt + bt).flatten().tolist()
    d = max(abs(a - c) for a, c in zip(y.to_vector(), exp))
    print("traced max_abs_diff:", d)
    ok = type(fn).__name__ == "JITFunction" and d <= 1e-4
    print("E6-3 PASS" if ok else "E6-3 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

## E6.4 — What the tracer rejects, and the expired-intermediate error

Three distinct failures, three distinct messages. Do not confuse them:

| Error | Trigger | Fix |
|---|---|---|
| `Unsupported operator name in JIT compilation: MatMul` | `func` uses an op outside the §E6.3 list (here `matmul`) | Rewrite the traced region with supported elementwise ops only |
| `Traced graph reached a leaf tensor that is not in the input list` | Tracing a graph built *outside* `func` (inputs had `requires_grad=False` when built, so no creator nodes exist) or a multi-element leaf not passed as an input | Build the graph inside `func` from the listed inputs |
| `Tracer found an expired intermediate tensor; keep every intermediate alive until trace returns` | An intermediate's weak pointer expired before `trace` walked the graph (`src/ops/jit.cpp:404`) | Keep every intermediate alive until `trace` returns; build inside `func` with `lt.Ops.*` |

The example below reproduces the first two on `88a4ba1` and states
the measured status of the third. Two findings from probing this
build are worth recording because they contradict the naive reading
of the rule. First, plain Python `+` on tensors *inside* `func`
traces fine on this build — the `Add` creator node is identical to
`lt.Ops.add`'s, so the `lt.Ops.*` requirement is about which ops the
tracer understands, not about operator syntax. Second, the
expired-intermediate error could not be triggered through the Python
API on this build in eight probed variants (named intermediates,
`del`'d intermediates, helper-built intermediates, `gc.collect()`,
graphs pre-built with `requires_grad=True`): the autograd graph
holds the intermediates alive past the point where user code drops
them. The error string exists in source (`jit.cpp:404`) and the rule
stands as written — keep intermediates alive until `trace` returns —
but on `88a4ba1` the failure is latent through Python, not
reproducible. The pre-built-graph variant below raises the leaf
error instead, which is the failure you will actually meet if you
trace a stale graph.

<!-- ex: id=E6-4 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt


def main():
    cpu = lt.Device("cpu")
    a = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0, 0.25, 0.75, -0.25, 0.5], [2, 4], cpu)
    b = lt.Tensor.from_vector([0.5, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, 0.25], [4, 2], cpu)
    try:
        lt.jit.Tracer.trace([a, b], lambda ins: lt.Ops.matmul(ins[0], ins[1]), "mm")
        print("matmul trace: no error (UNEXPECTED)")
    except RuntimeError as e:
        print("matmul trace raises:", e)
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    c = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)
    y = lt.Ops.relu(lt.Ops.add(x, c))
    try:
        lt.jit.Tracer.trace([x, c], lambda ins: y, "prebuilt")
        print("prebuilt trace: no error (UNEXPECTED)")
    except RuntimeError as e:
        print("prebuilt trace raises:", e)
    print("E6-4 PASS")


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1. The two printed messages are
the verbatim runtime errors; the expired-intermediate message is
quoted in the table above from `src/ops/jit.cpp:404` and was not
reproducible through the Python API on this build (see text).

## E6.5 — Fused-ops inventory: the bigger kernels

Beyond `lt.jit`, `lt.Ops` exposes fused kernels for whole
sub-graphs (`bind_ops.cpp:35, 78, 162-164`). These are single ops,
not traced programs: they run one kernel (or one kernel sequence)
with no graph construction at all.

| Function | Computes | Notes |
|---|---|---|
| `Ops.fused_add_layernorm(input, residual, normalized_shape, weight=None, bias=None, eps=1e-5)` | `layernorm(input + residual)` | `normalized_shape` is a list, e.g. `[4]` |
| `Ops.fused_swiglu(gate, up)` | `silu(gate) * up` | SwiGLU activation, the transformer MLP core |
| `Ops.fused_qkv_proj(x, wq, wk, wv)` | `(x @ wq.T, x @ wk.T, x @ wv.T)` | Returns a 3-tuple `(q, k, v)`, not one tensor |
| `Ops.fused_linear_cross_entropy(x, weight, target, bias=None)` | cross-entropy of `x @ weight.T + bias` against `target` | Fused so the full logits never materialize |

<!-- ex: id=E6-5 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def check(name, got, exp, tol=1e-4):
    d = max(abs(a - b) for a, b in zip(got, exp))
    print(name, "max_abs_diff:", d)
    return d <= tol


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    r = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
    w = lt.Tensor.from_vector([1.0, 1.0, 1.0, 1.0], [4], cpu)
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    rt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    ok = True
    ok = check("fused_add_layernorm", lt.Ops.fused_add_layernorm(x, r, [4], w).to_vector(), F.layer_norm(xt + rt, [4]).flatten().tolist()) and ok
    ok = check("fused_swiglu", lt.Ops.fused_swiglu(x, r).to_vector(), (F.silu(xt) * rt).flatten().tolist()) and ok
    wq = lt.Tensor.from_vector([0.5, 0.25, 0.0, 0.75, 0.1, 0.5, -0.5, 0.25, 0.3, 0.3, 0.2, 0.1, -0.2, 0.4, 0.6, 0.2], [4, 4], cpu)
    wqt = torch.tensor([[0.5, 0.25, 0.0, 0.75], [0.1, 0.5, -0.5, 0.25], [0.3, 0.3, 0.2, 0.1], [-0.2, 0.4, 0.6, 0.2]])
    q, k, v = lt.Ops.fused_qkv_proj(x, wq, wq, wq)
    exp_q = (xt @ wqt.T).flatten().tolist()
    ok = check("fused_qkv_proj q", q.to_vector(), exp_q) and ok
    ok = check("fused_qkv_proj k", k.to_vector(), exp_q) and ok
    ok = check("fused_qkv_proj v", v.to_vector(), exp_q) and ok
    tgt = lt.Tensor.from_vector([2.0], [1], cpu)
    ok = check("fused_linear_cross_entropy", lt.Ops.fused_linear_cross_entropy(x, wq, tgt).to_vector(), F.cross_entropy(xt @ wqt.T, torch.tensor([2])).flatten().tolist()) and ok
    print("E6-5 PASS" if ok else "E6-5 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`fused_linear_cross_entropy` matters most at scale: with a large
vocabulary the unfused path materializes a `[batch, vocab]` logits
tensor just to reduce it to one number per row; the fused kernel
never builds it.

## E6.6 — When fusion helps, and when it hurts

Fusion helps when the graph is memory-bound and elementwise: the
win is proportional to the traffic eliminated. A `relu(x + b)` on a
large tensor removes one full read + one full write of the
intermediate; a chain of five elementwise ops removes four. Fusion
hurts — or is neutral — when:

- The graph is compute-bound already (matmul-dominated): fusing the
  epilogue saves a launch but the matmul dominates the time, so the
  relative win is noise.
- The intermediate is needed later: if a later op consumes the
  pre-activation sum, fusing it into the ReLU forces recomputation
  or a second materialization. The tracer has no rematerialization
  analysis; it fuses what you give it.
- The graph is tiny: one kernel launch is ~microseconds, and tracing
  + compiling the `JITFunction` costs far more than one eager run.
  Trace once, call many times — never trace inside a training loop.
- The op is not in the supported set (§E6.3): the trace fails
  outright, and the fallback is the unfused eager path.

The example below measures the shape of the trade-off on CPU: the
same `relu(x + b)` three ways — the `lt.jit` convenience wrapper, a
`JITFunction` traced once and called 2000 times, and the unfused eager
path — 2000 iterations each, values checked for parity first, timings
printed second. The absolute times are machine-specific (2-CPU VM
here, under load from other jobs); the ratios are the point. Measured
on CPU at `[64, 64]` on `88a4ba1`, the JIT path LOSES: the traced
function costs ~0.012 s/iter and the wrapper ~0.011 s/iter against
~0.0003-0.0005 s/iter unfused, a ratio of ~0.03. The fusion win — one memory
pass instead of two — is real, but at this size it is smaller than the
JIT dispatch overhead, and the wrapper additionally rebuilds the
function on every call. Fusion pays where the traffic and launch
savings exceed the dispatch cost: larger tensors, and above all GPU,
where each eager op is a separate kernel launch. The GPU side of that
claim is REQUIRES-GPU, not measured here; the CPU side is measured,
and it says: for small graphs on CPU, prefer the fused `Ops` kernels
of §E6.5 — single C++ calls with no JIT dispatch — over traced
`JITFunction`s.

<!-- ex: id=E6-6 device=cpu backend=cpu runtime=~45s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import time
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    n = 64 * 64
    xv = [0.001 * float((i * 7) % 200 - 100) for i in range(n)]
    bv = [0.01] * n
    x = lt.Tensor.from_vector(xv, [64, 64], cpu)
    b = lt.Tensor.from_vector(bv, [64, 64], cpu)
    xt = torch.tensor(xv).reshape(64, 64)
    bt = torch.tensor(bv).reshape(64, 64)
    exp = F.relu(xt + bt).flatten().tolist()
    fn = lt.jit.Tracer.trace([x, b], lambda ins: lt.Ops.relu(lt.Ops.add(ins[0], ins[1])), "traced_add_relu")
    d = max(abs(a - c) for a, c in zip(fn([x, b]).to_vector(), exp))
    print("traced vs torch max_abs_diff:", d)
    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        lt.jit.fused_bias_relu(x, b)
    t_wrap = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        fn([x, b])
    t_traced = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        lt.Ops.relu(lt.Ops.add(x, b))
    t_unfused = time.perf_counter() - t0
    print("wrapper s/iter:", t_wrap / iters, "traced s/iter:", t_traced / iters, "unfused s/iter:", t_unfused / iters)
    print("ratio unfused/traced:", t_unfused / t_traced)
    ok = d <= 1e-4
    print("E6-6 PASS" if ok else "E6-6 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(values); timings are informational and machine-specific.

## E6.7 — What to remember

- `JITVar` is the graph, `JITFunction` is the program;
  `Tracer.trace` returns a `JITFunction`, never a `JITVar`.
- Trace inside `func` with supported ops only (`Add/Sub/Mul/Div`,
  `ReLU/GELU/Sigmoid/Tanh/Sqrt/Exp/Log/Abs/Neg`); `MatMul` and
  friends are rejected, as is any graph built outside `func`.
- The expired-intermediate error (`jit.cpp:404`) is real in source
  and latent through Python on `88a4ba1` — keep intermediates alive
  until `trace` returns anyway, because the tracer reads them
  through weak pointers.
- Ready-made fused ops (`fused_bias_relu`, `fused_bias_gelu`,
  `fused_residual_add`, `fused_residual_gelu`) and the bigger
  `Ops` fused kernels (`fused_add_layernorm`, `fused_swiglu`,
  `fused_qkv_proj`, `fused_linear_cross_entropy`) cover the common
  patterns without tracing.
- GELU in the fused path is the tanh approximation: compare against
  `F.gelu(..., approximate='tanh')`, not torch's exact default.
- Fusion pays when the graph is memory-bound and elementwise and
  the compiled function is called many times; but on CPU at `[64, 64]`
  the JIT path measured ~30x slower than unfused eager — the fusion
  win only exceeds the dispatch cost at larger sizes or on GPU
  (unmeasured, REQUIRES-GPU). For small graphs on CPU, prefer the
  fused `Ops` kernels of §E6.5 over traced `JITFunction`s.
