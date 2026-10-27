# Tutorial T1 — 60-Minute Blitz: Tensors, Autograd, Training

Learn LiteTorch by doing: create tensors, differentiate through them, build a
model, train it, save it, and move it between devices. Every example below is
self-contained and runnable on CPU. Total time: about 60 minutes.

What you will learn:

- How to create tensors and run ops on them.
- How autograd records a graph and computes gradients.
- How to build a model from `lt.nn.Module` and `lt.nn.Sequential`.
- How to run a training loop and read the loss curve.
- How to save and reload a model, and how devices and dtypes work.

Prerequisites: LiteTorch installed (`pip install litetorch`) and importable.
All examples run on CPU against LiteTorch 88a4ba1; each one states its
PyTorch parity result underneath.

> PyTorch differences touched in this tutorial: D-03 (`.to()` takes a
> `Device`, no string form), D-04 (factory functions take a single shape
> list), D-02 (`.to(device)` silently falls back to CPU when the accelerator
> is unavailable — always check `t.device` afterwards).

## T1.1 — Create your first tensors

A tensor is an n-dimensional array with a shape, a dtype, and a device.
Build one from a Python list with `Tensor.from_vector`, or use a factory.
Factory functions take a single shape list: `lt.randn([2, 3])`, never
`lt.randn(2, 3)` (D-04).

<!-- ex: id=T1.1-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False)
b = lt.Tensor.from_vector([0.5, 1.5, -1.0, 2.0], [2, 2], device, False)

c = lt.Ops.matmul(a, b)
print("a @ b:", c.to_vector())
print("shape:", c.shape, "dtype:", c.dtype, "device:", c.device)

r = lt.randn([2, 3])
print("randn shape:", r.shape)

z = lt.zeros([2, 2])
print("zeros:", z.to_vector())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4 (matmul
matches `torch.matmul` on the same inputs).

## T1.2 — Autograd in five lines

Set `requires_grad=True` when you create a tensor, run ops on it, call
`.backward()` on a scalar loss, and read `.grad`. LiteTorch records a
tape-based DAG during the forward pass and walks it backwards.

<!-- ex: id=T1.2-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([2.0, 3.0], [2], device, True)
w = lt.Tensor.from_vector([4.0, 5.0], [2], device, True)

y = lt.Ops.mul(x, w)
loss = lt.Ops.sum(y)
loss.backward()

print("y:", y.to_vector())
print("loss:", loss.item())
print("x.grad:", x.grad.to_vector())
print("w.grad:", w.grad.to_vector())
```

Expected output: `y = [8, 15]`, `loss = 23`, `x.grad = [4, 5]`,
`w.grad = [2, 3]`. Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1,
parity 1e-4 (values and gradients match `torch` on the same inputs).

Note: elementwise and shape ops live under `litetorch.Ops` (for example
`lt.Ops.relu(t)`), not as `Tensor` methods — `t.relu()` does not exist
(D-17). `t ** 2.0` works for a scalar exponent; there is no `.pow()`
method and no tensor-exponent form (D-05).

## T1.3 — Build a model

Subclass `lt.nn.Module` for a custom block, or compose layers with
`lt.nn.Sequential`. A module owns its parameters; `parameters()` returns
them, `to(device)` moves them.

<!-- ex: id=T1.3-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt


class MultiLayerPerceptron(lt.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = lt.nn.Linear(in_features, hidden_features, True)
        self.relu = lt.nn.ReLU()
        self.fc2 = lt.nn.Linear(hidden_features, out_features, True)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        return self.fc2.forward(h)

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def to(self, device):
        self.fc1.to(device)
        self.fc2.to(device)


device = lt.auto_device()
model = MultiLayerPerceptron(4, 8, 2)
model.to(device)

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
print("custom model output shape:", model.forward(x).shape)
print("parameter count:", len(model.parameters()))
```

The same network as a `Sequential` is shorter and equivalent:

<!-- ex: id=T1.3-2 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
print("sequential output shape:", model.forward(x).shape)
print("parameter count:", len(model.parameters()))
```

Verified on CPU, LiteTorch 88a4ba1. No torch parity claim: the weights are
randomly initialized, so only shapes and parameter counts are asserted.

## T1.4 — Train it: the training loop

The loop is always the same five steps: zero the gradients, forward,
loss, backward, optimizer step. Seed first with `lt.manual_seed` so the
run is repeatable. The loader runs with `shuffle=False` so the printed
losses are reproducible run to run; enable `shuffle=True` in real
training. With the seed and data below, the average loss falls
from 1.0306 in epoch 1 to 0.2142 in epoch 10.

<!-- ex: id=T1.4-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(42)
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x_train = lt.Tensor.from_vector([
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    5.0, 6.0, 7.0, 8.0,
    8.0, 7.0, 6.0, 5.0
], [4, 4], device, False)

y_train = lt.Tensor.from_vector([
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0
], [4, 2], device, False)

dataset = lt.data.TensorDataset(x_train, y_train)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)

for epoch in range(10):
    loader.reset()
    total_loss = 0.0
    steps = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        optimizer.zero_grad()
        preds = model.forward(bx)
        loss = loss_fn.forward(preds, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    print("Epoch %d, Loss: %.4f" % (epoch + 1, total_loss / max(steps, 1)))
```

Verified on CPU, LiteTorch 88a4ba1. Loss decreases every epoch
(1.0306, 0.7767, 0.5915, 0.4568, 0.3630, 0.3002, 0.2597, 0.2345, 0.2197,
0.2142). No pointwise torch parity claim: training dynamics depend on
the random initialization, which differs across frameworks.

## T1.5 — Save and reload a checkpoint

`lt.save` writes a model (or optimizer) to a `.lt` file; `lt.load`
restores it into a fresh model of the same shape. Reloading must
reproduce the original outputs exactly.

<!-- ex: id=T1.5-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(42)
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
before = model.forward(x).to_vector()

lt.save(model, "/tmp/tut_model.lt")

model2 = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model2.to(device)
lt.load(model2, "/tmp/tut_model.lt")
after = model2.forward(x).to_vector()

print("max abs diff after reload:", max(abs(a - b) for a, b in zip(before, after)))
```

Verified on CPU, LiteTorch 88a4ba1: max abs diff is exactly 0.0.

## T1.6 — Devices and dtypes

`lt.auto_device()` picks the best available device (CPU here).
`.to()` takes a `Device` object — `model.to("cuda")` raises a TypeError
(D-03). And if the requested accelerator is unavailable, `.to()` falls
back to CPU silently (D-02): always check `t.device` afterwards.

<!-- ex: id=T1.6-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

print("auto device:", lt.auto_device())
print("cuda available:", lt.cuda.is_available())

device = lt.auto_device()
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False)

gpu = lt.Device("cuda")
y = x.to(gpu)
print("requested cuda, got device:", y.device)

half = x.cast(lt.DataType.FP16)
print("cast dtype:", half.dtype)
print("back to FP32:", half.cast(lt.DataType.FP32).dtype)
```

Verified on CPU, LiteTorch 88a4ba1. On this CPU-only machine
`lt.cuda.is_available()` is False and the `.to("cuda")` request lands
on CPU — the silent fallback of D-02, demonstrated rather than hidden.
On a GPU machine the same code returns a GPU tensor; the device check
after `.to()` is what tells you which.
