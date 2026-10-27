# How-To H7 — Data Loading at Scale

`DataLoader` batches, shuffles, prefetches, and checkpoints its own
state. This page covers the three things that matter in practice:
reproducible batches, a custom dataset, and resuming from a saved
loader state. All examples below run on CPU against LiteTorch 88a4ba1.

Prerequisites: LiteTorch installed and importable.

> PyTorch differences touched here: D-04 (the old §11.3 example was
> not standalone — every example below inlines its own setup), D-06
> (`.long()` returns INT32 — cast to INT64 for index tensors).

## H7.1 — A standalone DataLoader with checkpoint and resume

Everything in one block: dataset, loader, iteration, saving the loader
state to disk, and resuming from it. `save_state` / `load_state` let a
training job restart mid-epoch without re-feeding the batches it
already consumed.

<!-- ex: id=H7.1-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import os

import litetorch as lt

device = lt.auto_device()
features = lt.Tensor.from_vector([float(i) * 0.1 for i in range(400)], [40, 10], device, False)
labels = lt.Tensor.from_vector([float(i % 2) for i in range(40)], [40], device, False)

dataset = lt.data.TensorDataset(features, labels)
loader = lt.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    device=device,
    prefetch_limit=2,
    num_workers=0,
    drop_last=False,
)

batches = []
for _ in range(5):
    batches.append(loader.next())
print("batches:", len(batches), "first batch shape:", list(batches[0][0].shape))

os.makedirs("/tmp/litetorch-dataloader-demo", exist_ok=True)
loader.save_state("/tmp/litetorch-dataloader-demo/loader_state.json")

loader2 = lt.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    device=device,
    prefetch_limit=2,
    num_workers=0,
    drop_last=False,
)
loader2.load_state("/tmp/litetorch-dataloader-demo/loader_state.json")

resumed = loader2.next()
print("resumed batch first row:", resumed[0].to_vector()[:3])
print("original next batch first row:", loader.next()[0].to_vector()[:3])
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: 5 batches of shape
`[4, 10]`, and the resumed loader's next batch matches the original
loader's next batch exactly (same first row). The checkpoint only
records the loader's position — the dataset itself is rebuilt from the
same tensors on resume, so keep the dataset construction identical.

One reproducibility note, measured twice during this rewrite: the
`DataLoader` shuffle is **not** seeded by `lt.manual_seed(42)`. Two
runs with the same seed produce different batch orders when
`shuffle=True`. For reproducible doc numbers use `shuffle=False`, as
above; for real training, treat shuffle order as nondeterministic.

## H7.2 — A custom dataset

Subclass `lt.data.Dataset` and implement `size()` and
`get(index)` returning `(features, label)` tensors. The trampoline
dispatches to those two method names — not to `__len__` /
`__getitem__` — so define `size` and `get` exactly as spelled.
(`len(dataset)` and `dataset[i]` still work: they are bound to the
same C++ methods.) `TensorDataset`
(H7.1) covers the common case; subclass when the data is generated,
transformed, or streamed rather than already a tensor.

<!-- ex: id=H7.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()


class SquaresDataset(lt.data.Dataset):
    def __init__(self, n, device):
        super().__init__()
        self.n = n
        self.device = device

    def size(self):
        return self.n

    def get(self, idx):
        x = lt.Tensor.from_vector([float(idx)] * 4, [4], self.device, False)
        y = lt.Tensor.from_vector([float(idx * idx)], [1], self.device, False)
        return x, y


dataset = SquaresDataset(10, device)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)
first_batch = loader.next()
print("first batch x shape:", list(first_batch[0].shape))
print("first batch y:", first_batch[1].to_vector())
del loader
del dataset
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: first batch `x`
shape `[2, 4]`, `y` values `[0.0, 1.0]` (the squares of indices 0 and
1). Keep `get` cheap: it runs once per sample, on the loader's
calling thread when `num_workers=0`. The explicit `del loader` /
`del dataset` at the end is load-bearing, not style: a Python-backed
`Dataset` left held by the C++ loader aborts the interpreter at
shutdown (`FATAL: exception not rethrown`, SIGABRT) if it is left to
garbage collection. Delete both explicitly before exit. This is a
known defect in the `Dataset` trampoline, filed to Core for triage —
`TensorDataset` (H7.1) does not need the workaround.

## H7.3 — Save and load a model with the .lt format

`.lt` is LiteTorch's native checkpoint format. Save the model and the
optimizer together; on load, restore both before resuming training, or
the optimizer restarts from its initial state and the first steps
after resume are wrong.

<!-- ex: id=H7.3-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import os

import litetorch as lt

device = lt.auto_device()
model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.GELU(),
    lt.nn.Linear(8, 2, True),
])
model.to(device)

optimizer = lt.optim.Adam(model.parameters(), 0.001)

x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], device, True)
y = model(x)
loss = lt.Ops.sum(y)
loss.backward()
optimizer.step()

os.makedirs("/tmp/litetorch-checkpoint-demo", exist_ok=True)
lt.save(model, "/tmp/litetorch-checkpoint-demo/model.lt")
lt.save(optimizer, "/tmp/litetorch-checkpoint-demo/optimizer.lt")

model2 = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.GELU(),
    lt.nn.Linear(8, 2, True),
])
model2.to(device)
optimizer2 = lt.optim.Adam(model2.parameters(), 0.001)
lt.load(model2, "/tmp/litetorch-checkpoint-demo/model.lt")
lt.load(optimizer2, "/tmp/litetorch-checkpoint-demo/optimizer.lt")

x2 = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], device, True)
print("original output:", model(x2).to_vector())
print("restored output:", model2(x2).to_vector())
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: the restored
model prints exactly the same output as the original (same weights,
same input). `lt.save` / `lt.load` take a `Module`, a list of
parameters, or an `Optimizer` — not a state dict. Save the optimizer
with the same call (`lt.save(optimizer, path)`), and keep epoch/step
counters in a sidecar JSON; an optimizer restored from its initial
state makes the first steps after resume wrong. `state_dict()` /
`load_state_dict(sd)` exist too, for moving weights between two live
models without touching disk.
