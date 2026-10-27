# Tutorial T2 — Train a Classifier

Train a small MLP to separate two classes, end to end: build the data,
load it in batches, train with cross-entropy loss, and measure accuracy.
The data here is synthetic (two Gaussian blobs, seeded) so the tutorial
runs anywhere on CPU in seconds; the same loop trains a real MNIST/CIFAR
model once you swap in a real dataset.

What you will learn:

- How to build a `TensorDataset` and iterate it with a `DataLoader`.
- How to train a classifier with `CrossEntropyLoss` and `AdamW`.
- How to measure accuracy and read the loss curve.
- How the same code moves to GPU.

Prerequisites: Tutorial T1. All examples run on CPU against LiteTorch
88a4ba1; each one states its PyTorch parity result underneath.

> PyTorch differences touched in this tutorial: D-12 (use
> `lt.Ops.sum(t, [0, 1])` for multi-dim reductions — `Tensor.sum` takes a
> single int dim), D-09 (GELU is tanh-approximation only; compare against
> `F.gelu(x, approximate='tanh')`).

## T2.1 — Build the data

Two classes, four features each. Class 0 is centered at 0, class 1 at
1.5, both with unit Gaussian noise. Seeded with `random.seed(0)`, so the
dataset is identical on every run.

<!-- ex: id=T2.1-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import random
import litetorch as lt

device = lt.auto_device()

random.seed(0)
n = 200
xs = []
ys = []
for i in range(n):
    c = i % 2
    xs.extend([random.gauss(c * 1.5, 1.0) for _ in range(4)])
    ys.append(float(c))

x = lt.Tensor.from_vector(xs, [n, 4], device, False)
y = lt.Tensor.from_vector(ys, [n], device, False)

dataset = lt.data.TensorDataset(x, y)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=False, device=device)

print("dataset size:", dataset.size())
print("batches per epoch:", len(loader))

loader.reset()
bx, by = loader.next()
print("first batch x shape:", bx.shape, "y shape:", by.shape)
```

Verified on CPU, LiteTorch 88a4ba1: 200 samples, 13 batches of 16
(12 full batches plus a final batch of 8), first batch shapes `[16, 4]`
and `[16]`.

## T2.2 — Train the classifier

MLP `4 -> 16 -> 2` with ReLU, `CrossEntropyLoss`, `AdamW` at lr 0.05,
15 epochs. Seeded with `lt.manual_seed(7)` so the initialization is
repeatable, and the loader runs with `shuffle=False` so the printed
losses are reproducible run to run. The average loss falls from 0.4016
in epoch 1 to 0.1075 in epoch 15, and training accuracy reaches 0.96.

<!-- ex: id=T2.2-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import random
import litetorch as lt

lt.manual_seed(7)
random.seed(0)
device = lt.auto_device()

n = 200
xs = []
ys = []
for i in range(n):
    c = i % 2
    xs.extend([random.gauss(c * 1.5, 1.0) for _ in range(4)])
    ys.append(float(c))

x = lt.Tensor.from_vector(xs, [n, 4], device, False)
y = lt.Tensor.from_vector(ys, [n], device, False)

dataset = lt.data.TensorDataset(x, y)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=False, device=device)

model = lt.nn.Sequential([
    lt.nn.Linear(4, 16, True),
    lt.nn.ReLU(),
    lt.nn.Linear(16, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
loss_fn = lt.nn.CrossEntropyLoss()

for epoch in range(15):
    loader.reset()
    total_loss = 0.0
    batches = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        optimizer.zero_grad()
        logits = model.forward(bx)
        loss = loss_fn.forward(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    print("Epoch %d, Loss: %.4f" % (epoch + 1, total_loss / batches))

logits = model.forward(x)
lv = logits.to_vector()
correct = 0
for i in range(n):
    pred = 0 if lv[2 * i] > lv[2 * i + 1] else 1
    if pred == int(ys[i]):
        correct += 1
print("train accuracy:", correct / n)
```

Verified on CPU, LiteTorch 88a4ba1: final loss 0.1075, train accuracy
0.96. The run is deterministic with these seeds. A shuffled order or a
different seed lands nearby — final loss ~0.10–0.13, accuracy
~0.94–0.96 — as measured on a shuffled run of the same setup. No
pointwise torch parity claim: training dynamics depend on the random
initialization.

## T2.3 — Check the loss against PyTorch

The classifier above is only trustworthy if the loss itself is right.
`CrossEntropyLoss` values and gradients match `torch.nn.functional.
cross_entropy` at 1e-4 on the same inputs.

<!-- ex: id=T2.3-1 device=cpu backend=cpu runtime=~5.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

logits_lt = lt.Tensor.from_vector([2.0, 1.0, 0.1, 0.5, 2.5, 0.3], [2, 3], device, True)
target_lt = lt.Tensor.from_vector([0.0, 2.0], [2], device, False)

loss_fn = lt.nn.CrossEntropyLoss()
loss_lt = loss_fn.forward(logits_lt, target_lt)
loss_lt.backward()

logits_t = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
loss_t = torch.nn.functional.cross_entropy(logits_t, torch.tensor([0, 2]))
loss_t.backward()

print("lt loss:", loss_lt.item())
print("torch loss:", loss_t.item())
print("lt grad:", logits_lt.grad.to_vector())
print("torch grad:", logits_t.grad.flatten().tolist())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4:
loss 1.418540 on both sides, gradients equal within 1e-4.

## T2.4 — Move the same loop to GPU (REQUIRES-GPU)

Nothing in the loop changes except the device: build the model and the
data on the GPU device and run the identical code. Do not run this
block on a CPU-only machine — `.to("cuda")` silently falls back to CPU
(D-02), so the block below checks the device and stops loudly if no GPU
is present.

<!-- ex: id=T2.4-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->
```python skip-run

import litetorch as lt

gpu = lt.Device("cuda")
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 16, True),
    lt.nn.ReLU(),
    lt.nn.Linear(16, 2, True)
])
model.to(gpu)
params = model.parameters()
if params[0].device.type != lt.DeviceType.GPU:
    raise RuntimeError("no GPU device: refusing to run the GPU variant on CPU")

print("training on:", params[0].device)
```

REQUIRES-GPU — verified on Kaggle (2x T4) in B5, not on CPU. The CPU
run of T2.2 is the same loop on the same data.
