# Quickstart Guide

This guide is the 60-minute entry point to LiteTorch. In about an hour you will create tensors, differentiate through them, build a neural network module, load data in batches, train it, and save a checkpoint. Every example is self-contained and runnable: copy a block, run it, and compare your output with the expected output printed below it.

Prerequisites: a working install ([Installation](installation.md)). All examples below run on CPU; on a GPU machine the same code runs on the GPU via `lt.auto_device()`.

A note on reproducibility: the training example seeds the RNG with `lt.manual_seed(42)` and runs the DataLoader with `shuffle=False`, so the loss values below are exact. In real training use `shuffle=True`; the per-epoch values then vary run to run (the DataLoader shuffle uses its own RNG, which `lt.manual_seed` does not control), but the loss still decreases.

---

## 3.1. Tensor operations and autograd

A `Tensor` holds data on a device; `requires_grad=True` asks autograd to record the operations so `.backward()` can compute gradients.

<!-- ex: id=3.1-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
b = lt.Tensor.from_vector([0.5, 1.5, -1.0, 2.0], [2, 2], device, True)

c = lt.Ops.matmul(a, b)
loss = lt.Ops.sum(c)

loss.backward()

print("Matrix multiplication result c:", c.to_vector())
print("Loss value:", loss.item())
print("Gradient da/dc:", a.grad.to_vector())
print("Gradient db/dc:", b.grad.to_vector())
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27 vs torch 2.14.1+cpu: values, loss and gradients match exactly, max diff 0.0).

Expected output:

```
Matrix multiplication result c: [-1.5, 5.5, -2.5, 12.5]
Loss value: 14.0
Gradient da/dc: [2.0, 1.0, 2.0, 1.0]
Gradient db/dc: [4.0, 4.0, 6.0, 6.0]
```

What happened: `c = a @ b`, `loss = sum(c) = 14.0`. The gradient of the loss with respect to `a` is a matrix of the row-sums of `b` broadcast across columns, and with respect to `b` it is the column-sums of `a` broadcast across rows. If you change `b`, both the loss and the gradients change — try it.

---

## 3.2. Building neural network modules

Subclass `lt.nn.Module` to build a modular architecture. A module owns its parameters, moves them with `.to(device)`, and computes its output in `.forward(x)`.

<!-- ex: id=3.2-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

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
        out = self.fc2.forward(h)
        return out

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def to(self, device):
        self.fc1.to(device)
        self.fc2.to(device)

model = MultiLayerPerceptron(64, 128, 10)
print("Parameters:", len(model.parameters()))
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27: parameter count matches torch, 4 == 4).

Expected output:

```
Parameters: 4
```

Four parameter tensors: the weight and bias of each of the two `Linear` layers. `ReLU` has no parameters.

Or compose layers with `lt.nn.Sequential` — the same model in four lines:

<!-- ex: id=3.2-2 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(64, 128, True),
    lt.nn.ReLU(),
    lt.nn.Dropout(0.1),
    lt.nn.Linear(128, 10, True)
])

model.to(device)
print("Parameters:", len(model.parameters()))
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27: parameter count matches torch, 4 == 4).

Expected output:

```
Parameters: 4
```

`Dropout` has no parameters, and it is active only in training mode — it is a no-op at inference. The rest of this guide uses `Sequential` for brevity; the reference pages document every module class.

---

## 3.3. Managing datasets with TensorDataset and DataLoader

`TensorDataset` pairs feature and label tensors; `DataLoader` batches them, shuffles them, and moves batches to the device.

<!-- ex: id=3.3-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

x_data = [float(i % 10) for i in range(100 * 64)]
y_data = [float(i % 2) for i in range(100 * 1)]

x_tensor = lt.Tensor.from_vector(x_data, [100, 64], device, False)
y_tensor = lt.Tensor.from_vector(y_data, [100, 1], device, False)

dataset = lt.data.TensorDataset(x_tensor, y_tensor)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=True, device=device)

loader.reset()
batch = loader.next()
bx, by = batch
print("Batch x shape:", bx.shape, "Batch y shape:", by.shape)
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27: first-batch shapes match torch, [16, 64] / [16, 1]).

Expected output:

```
Batch x shape: [16, 64] Batch y shape: [16, 1]
```

`loader.next()` returns `None` when the epoch is exhausted; call `loader.reset()` to start a new epoch. With `shuffle=True` the batch order differs every epoch.

---

## 3.4. Complete training loop

This example trains the tiny model from §3.2 on four synthetic samples for 10 epochs and prints the average loss per epoch. The loss must decrease — if it does not, something is wrong with the setup, not the model.

<!-- ex: id=3.4-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

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

    avg_loss = total_loss / max(steps, 1)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27 vs torch 2.14.1+cpu: 10-epoch AdamW trajectory matches with identical init weights, max per-epoch diff 1.38e-07).

Expected output (exact with `lt.manual_seed(42)` and `shuffle=False` — the DataLoader shuffle uses its own RNG, so with `shuffle=True` the per-epoch values vary run to run while the loss still decreases):

```
Epoch 1, Loss: 1.0306
Epoch 2, Loss: 0.7767
Epoch 3, Loss: 0.5915
Epoch 4, Loss: 0.4568
Epoch 5, Loss: 0.3630
Epoch 6, Loss: 0.3002
Epoch 7, Loss: 0.2597
Epoch 8, Loss: 0.2345
Epoch 9, Loss: 0.2197
Epoch 10, Loss: 0.2142
```

The loss falls from 1.03 to 0.21 over 10 epochs. The three calls that must happen in this order every step are `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — forgetting `zero_grad()` accumulates gradients across steps and the loss will not behave.

---

## 3.5. Model and optimizer checkpointing

Save a model and its optimizer, then load them into fresh instances and confirm the outputs are identical. This block is self-contained: it builds, trains one step, saves, reloads, and compares.

<!-- ex: id=3.5-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

lt.manual_seed(7)
device = lt.auto_device()

def build_model():
    m = lt.nn.Sequential([
        lt.nn.Linear(4, 8, True),
        lt.nn.ReLU(),
        lt.nn.Linear(8, 2, True)
    ])
    m.to(device)
    return m

model = build_model()
optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
y = lt.Tensor.from_vector([1.0, 0.0], [1, 2], device, False)

optimizer.zero_grad()
loss = loss_fn.forward(model.forward(x), y)
loss.backward()
optimizer.step()
print("Loss before save:", loss.item())

before = model.forward(x).to_vector()
lt.save(model, "model_checkpoint.lt")
lt.save(optimizer, "optim_checkpoint.lt")

model2 = build_model()
optimizer2 = lt.optim.AdamW(model2.parameters(), lr=0.01, weight_decay=0.001)
lt.load(model2, "model_checkpoint.lt")
lt.load(optimizer2, "optim_checkpoint.lt")

after = model2.forward(x).to_vector()
print("Outputs match:", before == after)
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27 vs torch 2.14.1+cpu: one-step loss bit-identical, 0.21786627173423767; save/load round-trip exact in both).

Expected output (exact with `lt.manual_seed(7)`):

```
Loss before save: 0.21786627173423767
Outputs match: True
```

The checkpoint files `model_checkpoint.lt` and `optim_checkpoint.lt` are written to the current working directory. Loading into a freshly built model of the same architecture restores the trained weights exactly — `before == after` must be `True`.

---

## What to read next

- **Tensor and autograd in depth**: Explanations — tensor and autograd (in progress).
- **Every op and module**: API reference (in progress) — start with the Tensor class and `nn` modules.
- **Train a real classifier**: Tutorial T2 (in progress) — MNIST / CIFAR with expected accuracy.
- **Train on 2 GPUs**: How-to H1 (in progress).
- **Something failed**: Troubleshooting and FAQ (in progress).
- **Coming from PyTorch**: Migration guide from PyTorch (in progress).
