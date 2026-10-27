# Data, Trainer, and Serialization — API Reference

`litetorch.data` holds datasets, samplers, and the `DataLoader`; `litetorch` top level holds `Trainer`, `TrainerConfig`, the `save` / `load` family, `AsyncCheckpointSaver`, and the RNG functions.

- **Source of truth:** `src/bindings/bind_data.cpp`, `include/litetorch/data.h`, `include/litetorch/trainer.h`, `src/serialization/optimizer.cpp`, `src/serialization/parameters.cpp`, `src/serialization/safetensors.cpp`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline; B5 re-verifies at head). Examples run against the Core D3 extension of 88a4ba1.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** data classes live under `lt.data.*`, not at the top level (`lt.data.TensorDataset`, not `lt.TensorDataset`). Python `Dataset` subclasses must define `size()` / `get(idx)` — the trampoline dispatches to those, not to `__len__` / `__getitem__` — and a loader holding a Python-backed dataset (or a Python `collate_fn`) aborts the interpreter at shutdown unless the loader is deleted explicitly (H7.2 defect, filed to Core). `lt.save` / `lt.load` are in-place and type-dispatched; `save_safetensors` / `AsyncCheckpointSaver.save_async` take a `dict`, not a module. `save_optimizer_state` supports Adam, AdamW, SGD, RMSprop only.

## `data.Dataset`

- **Signature:** `data.Dataset() -> Dataset`
- **Source:** `src/bindings/bind_data.cpp:8-13`
- **Methods:** `size()` — number of items; `get(index)` — returns an `(x, y)` tuple of tensors; `__len__` (alias of `size`); `__getitem__` (alias of `get`).
- **Subclassing:** define `size()` and `get(idx)` in the subclass. Defining only `__len__` / `__getitem__` does **not** work: the C++ trampoline dispatches to `size` / `get`, so a subclass that defines only the dunder methods raises at load time. The bound `__len__` / `__getitem__` on the base class call `size` / `get`, so `len(ds)` and `ds[i]` work once the subclass defines them.
- **Shutdown hazard (H7.2):** a `DataLoader` holding a Python-backed `Dataset` (or a Python `collate_fn`) aborts the interpreter at exit (`FATAL: exception not rethrown`, SIGABRT) unless the loader and dataset are deleted explicitly with `del loader` / `del dataset` before the interpreter shuts down. Filed to Core; see the `DataLoader` entry for the workaround.
- **PyTorch:** `torch.utils.data.Dataset` — PyTorch subclasses define `__len__` / `__getitem__`; LiteTorch requires `size` / `get` instead.

<!-- ex: id=r11-dataset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
class DS(lt.data.Dataset):
    def size(self):
        return 3
    def get(self, idx):
        return (lt.Tensor.from_vector([float(idx)], [1], cpu), lt.Tensor.from_vector([float(idx * 2)], [1], cpu))
ds = DS()
print(len(ds), ds[1][0].to_vector(), ds.get(1)[1].to_vector())
dl = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
for bx, by in dl:
    print(bx.to_vector(), by.to_vector())
del dl, ds
```

## `data.TensorDataset`

- **Signature:** `data.TensorDataset(x, y) -> TensorDataset`
- **Source:** `src/bindings/bind_data.cpp:15-20`
- **Parameters:** `x` — feature tensor, shape `[n, ...]`. `y` — label tensor, shape `[n, ...]`.
- **Attributes:** `x`, `y` — read/write tensor attributes.
- **Methods:** `size()` (returns `n`), `get(index)` (returns the `index`-th row pair), `__len__`, `__getitem__`.
- **Edge cases:** `x` and `y` must share the same first dimension; a mismatch is not checked at construction and fails at `get` time.
- **PyTorch:** `torch.utils.data.TensorDataset` — equivalent, except LiteTorch takes exactly two tensors (`x`, `y`), not a varargs list.

<!-- ex: id=r11-tensordataset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
print(ds.size(), ds.get(1)[0].to_vector(), ds[1][1].to_vector())
```

## Samplers (`data.Sampler`, `data.SequentialSampler`, `data.RandomSampler`, `data.BatchSampler`)

- **Signatures:** `data.SequentialSampler(n) -> SequentialSampler`; `data.RandomSampler(n, replacement=False, num_samples=0, seed=0) -> RandomSampler`; `data.BatchSampler(sampler, batch_size, drop_last=False) -> BatchSampler`
- **Source:** `src/bindings/bind_data.cpp:22-43`
- **Methods:** `indices()` — returns the list of indices (a Python list of ints); `__len__` — number of indices (for `BatchSampler`, the number of batches). `BatchSampler` also exposes `batches()` (list of index lists) and read-only `batch_size` / `drop_last` properties.
- **Parameters:** `RandomSampler`: `n` — dataset size; `replacement` — sample with replacement, default False; `num_samples` — samples to draw, default 0 (meaning `n`); `seed` — RNG seed, default 0 (0 means unseeded). `BatchSampler`: `sampler` — the underlying sampler; `batch_size` — indices per batch; `drop_last` — drop a final partial batch, default False.
- **Edge cases:** samplers are **not iterable** from Python — there is no `__iter__`; use `indices()` / `batches()` or pass the sampler to a `DataLoader`. `RandomSampler` with `seed=0` draws a different order on every call.
- **PyTorch:** `torch.utils.data.SequentialSampler` / `RandomSampler` / `BatchSampler` — same semantics; PyTorch samplers are iterable, LiteTorch samplers are not.

<!-- ex: id=r11-samplers device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
s = lt.data.SequentialSampler(5)
r = lt.data.RandomSampler(5, seed=7)
print(s.indices(), len(s))
print(r.indices())
b = lt.data.BatchSampler(s, 2)
print(b.batches(), len(b))
```

## `data.Subset` and `data.random_split`

- **Signatures:** `data.Subset(dataset, indices) -> Subset`; `data.random_split(dataset, lengths, seed=0) -> list[Dataset]`
- **Source:** `src/bindings/bind_data.cpp:45-54`
- **Parameters:** `Subset`: `dataset` — the underlying dataset; `indices` — list of indices to keep. `random_split`: `lengths` — list of split sizes summing to `len(dataset)`; `seed` — shuffle seed, default 0.
- **Methods:** `Subset` exposes `size()`, `get(index)`, `__len__`, `__getitem__` (indices are remapped through the list).
- **Edge cases:** `random_split` with `seed=0` shuffles unseeded. `lengths` must sum exactly to the dataset size.
- **PyTorch:** `torch.utils.data.Subset` / `random_split` — equivalent.

<!-- ex: id=r11-subset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
sub = lt.data.Subset(ds, [0, 2])
print(sub.size(), sub.get(1)[0].to_vector())
parts = lt.data.random_split(ds, [3, 1], seed=3)
print([p.size() for p in parts])
```

## `data.DataLoader`

- **Signature:** `data.DataLoader(dataset, batch_size, shuffle=True, device=None, prefetch_limit=2, num_workers=0, drop_last=False, sampler=None, collate_fn=None) -> DataLoader`
- **Source:** `src/bindings/bind_data.cpp:56-112`, `include/litetorch/data.h:109-114`
- **Parameters:** `dataset` — a `Dataset`. `batch_size` — items per batch (required). `shuffle` — shuffle each epoch, default True. `device` — target device for batches, default None (CPU). `prefetch_limit` — batches prefetched ahead, default 2. `num_workers` — background prefetch threads, default 0 (batches are built on the calling thread; `num_workers > 0` spawns worker threads that prefetch batches). `drop_last` — drop a final partial batch, default False. `sampler` — a `Sampler` overriding the shuffle order (when given, `shuffle` is ignored). `collate_fn` — a Python callable taking a list of `(x, y)` pairs and returning one `(batch_x, batch_y)` tuple.
- **Methods:** `reset()` — restart the epoch; `next()` — returns the next `(batch_x, batch_y)` tuple or `None` at the end; `__iter__` / `__next__` — iterating resets and yields batches until `StopIteration`; `__len__` / `num_batches()` — number of batches; `save_state(filepath)` / `load_state(filepath)` — persist and restore the iteration position (resume mid-epoch).
- **Attributes:** `dataset`, `batch_size`, `shuffle` — read/write.
- **Edge cases:** `shuffle=True` is **not** seeded by `lt.manual_seed` — the epoch order varies between runs; use `shuffle=False` (or an explicit seeded `RandomSampler`) for reproducible examples. A loader holding a Python-backed `Dataset` or a Python `collate_fn` aborts the interpreter at shutdown (SIGABRT, `FATAL: exception not rethrown`) unless the loader is deleted explicitly before exit — `del loader` (and `del dataset`) at the end of the program. Batches are built on the device passed in `device`; the default collate stacks rows into `[batch, ...]` tensors.
- **PyTorch:** `torch.utils.data.DataLoader` — same role; PyTorch defaults `shuffle=False` and `num_workers=0`, LiteTorch defaults `shuffle=True`. PyTorch `next()` is via the iterator protocol only; LiteTorch also exposes a direct `next()` method returning `None` at the end.

<!-- ex: id=r11-dataloader device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
dl = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
print(dl.num_batches())
for bx, by in dl:
    print(bx.shape, by.to_vector())
dl2 = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
print(dl2.next()[1].to_vector())
path = os.path.join(tempfile.mkdtemp(), "dlstate")
dl2.save_state(path)
dl3 = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
dl3.load_state(path)
print(dl3.next()[1].to_vector())
```

## `save` / `load`

- **Signatures:** `save(obj, filepath) -> None`; `load(obj, filepath) -> None`
- **Source:** `src/bindings/bind_data.cpp:141-169`
- **Parameters:** `obj` — for `save`: an `nn.Module`, a list of parameter tensors, or an `Optimizer`; for `load`: an `nn.Module` or a list of parameter tensors to load **into** (the object must already exist with the right shapes). `filepath` — destination path.
- **Behavior:** `save` dispatches on the object type: modules save their parameters, lists save the tensors, optimizers save optimizer state. `load` is in-place — it does not return the object. The format is LiteTorch's own binary format (magic + CRC32), not `torch.save` pickles.
- **Edge cases:** `save` of a bare tensor throws (`save expects a Module, list of Tensor parameters, or Optimizer`); wrap it in a list. `load` into a module with a different parameter count throws. Files are not portable to PyTorch.
- **PyTorch:** `torch.save` / `torch.load` — same role, different format; `torch.load` returns the object, LiteTorch `load` mutates in place.

<!-- ex: id=r11-save-load device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "m.lt")
lt.save(m, path)
m2 = lt.nn.Linear(4, 2)
lt.load(m2, path)
print([p.to_vector() for p in m2.parameters()] == [p.to_vector() for p in m.parameters()])
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
lt.save([t], path)
t2 = lt.Tensor.from_vector([0.0, 0.0, 0.0], [3], cpu)
lt.load([t2], path)
print(t2.to_vector())
```

## `save_parameters` / `load_parameters`

- **Signatures:** `save_parameters(params, filepath) -> None`; `load_parameters(params, filepath) -> None`
- **Source:** `src/bindings/bind_data.cpp:114-115`, `src/serialization/parameters.cpp`
- **Parameters:** `params` — a sequence of parameter tensors (e.g. `module.parameters()`), not a module. `filepath` — destination path.
- **Behavior:** the low-level form of `save` / `load` for parameter lists. `load_parameters` is in-place.
- **Edge cases:** passing a module instead of a parameter list throws a type error.
- **PyTorch:** `torch.save(module.state_dict(), ...)` / `module.load_state_dict(torch.load(...))` — same role, different format.

<!-- ex: id=r11-save-parameters device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "p.lt")
lt.save_parameters(m.parameters(), path)
m2 = lt.nn.Linear(4, 2)
lt.load_parameters(m2.parameters(), path)
print([p.to_vector() for p in m2.parameters()] == [p.to_vector() for p in m.parameters()])
```

## `save_safetensors` / `load_safetensors`

- **Signatures:** `save_safetensors(tensors, filepath) -> None`; `load_safetensors(filepath) -> dict[str, Tensor]`
- **Source:** `src/bindings/bind_data.cpp:116-127`, `src/serialization/safetensors.cpp`
- **Parameters:** `tensors` — a **dict** mapping names to tensors (not a module). `filepath` — destination path (conventionally `.safetensors`).
- **Behavior:** writes the safetensors format (compatible with Hugging Face `safetensors`). `load_safetensors` returns a dict of tensors keyed by the saved names.
- **Edge cases:** passing a module instead of a dict throws. Names are stored sorted; the returned dict is keyed by the saved names.
- **PyTorch:** `safetensors.torch.save_file` / `load_file` — same format and role.

<!-- ex: id=r11-safetensors device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "m.safetensors")
lt.save_safetensors({"weight": m.parameters()[0], "bias": m.parameters()[1]}, path)
d = lt.load_safetensors(path)
print(sorted(d.keys()), d["weight"].to_vector() == m.parameters()[0].to_vector())
```

## `save_optimizer_state` / `load_optimizer_state`

- **Signatures:** `save_optimizer_state(optimizer, filepath) -> None`; `load_optimizer_state(optimizer, filepath) -> None`
- **Source:** `src/bindings/bind_data.cpp:129-130`, `src/serialization/optimizer.cpp:37`
- **Parameters:** `optimizer` — the optimizer whose moments to persist. `filepath` — destination path.
- **Behavior:** binary format with a per-type magic (`LTAD` Adam, `LTAW` AdamW, `LTSG` SGD, `LTRP` RMSprop) plus CRC32. `load_optimizer_state` is in-place and checks that the file type matches the optimizer type.
- **Edge cases:** only **Adam, AdamW, SGD, RMSprop** are supported. Every other optimizer (NAdam, Adagrad, Adamax, Adadelta, ZeRO3Optimizer, AdamW8bit, AdamWFP8) throws `Unsupported optimizer type for saving state` on save. Loading a state saved from a different optimizer type throws. `Adagrad` and `NAdam` have their own `state_dict()` / `load_state_dict()` instead.
- **PyTorch:** `torch.save(optimizer.state_dict(), ...)` — PyTorch supports every optimizer; LiteTorch supports four.

<!-- ex: id=r11-optimizer-state device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
adam = lt.optim.Adam([w], lr=0.01)
adam.step()
path = os.path.join(tempfile.mkdtemp(), "opt.lt")
lt.save_optimizer_state(adam, path)
w2 = lt.Tensor.from_vector([1.0], [1], cpu, True)
adam2 = lt.optim.Adam([w2], lr=0.01)
lt.load_optimizer_state(adam2, path)
print("loaded")
try:
    lt.save_optimizer_state(lt.optim.NAdam([w2], lr=0.002), path)
except Exception as e:
    print(type(e).__name__, str(e)[:60])
```

## `AsyncCheckpointSaver`

- **Signature:** `AsyncCheckpointSaver() -> AsyncCheckpointSaver`
- **Source:** `src/bindings/bind_data.cpp:131-139`
- **Methods:** `save_async(tensors, filepath)` — queues a checkpoint write on a background thread; `tensors` is a **dict** of name → tensor. `wait()` — blocks until all queued saves complete. `is_saving()` — True while a save is in flight.
- **Behavior:** writes the safetensors format asynchronously; the caller must call `wait()` before reading the file or exiting, or the file may be missing or partial.
- **Edge cases:** `save_async` takes a dict, not a module. There is no callback on completion — poll `is_saving()` or call `wait()`. Intermittent shutdown hang observed: in the verification batch, 2 of ~46 runs hung for over 60 s after `wait()` returned and had to be killed (suspected background-thread teardown race); delete the saver explicitly with `del saver` before exit as a precaution.
- **PyTorch:** no equivalent in `torch` (async checkpointing exists in `torch.distributed.checkpoint`).

<!-- ex: id=r11-async-saver device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
saver = lt.AsyncCheckpointSaver()
path = os.path.join(tempfile.mkdtemp(), "a.safetensors")
saver.save_async({"t": t}, path)
print(saver.is_saving())
saver.wait()
print(os.path.exists(path), saver.is_saving())
del saver
```

## `TrainerConfig`

- **Signature:** `TrainerConfig() -> TrainerConfig`
- **Source:** `src/bindings/bind_data.cpp:191-197`, `include/litetorch/trainer.h:25-29`
- **Fields (all read/write):** `ckpt_dir` — checkpoint directory, default `"checkpoints"`. `save_every_steps` — steps between checkpoints, default 1000. `max_checkpoints` — newest checkpoints kept, default 5. `async_save` — write checkpoints on a background thread, default True. `elastic` — use the elastic consistent-checkpoint path (distributed only), default False.
- **PyTorch:** no equivalent class (`torch` has no built-in trainer; this is closest to a Hugging Face `TrainingArguments` subset).

<!-- ex: id=r11-trainerconfig device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cfg = lt.TrainerConfig()
print(cfg.ckpt_dir, cfg.save_every_steps, cfg.max_checkpoints, cfg.async_save, cfg.elastic)
cfg.ckpt_dir = "my-checkpoints"
cfg.max_checkpoints = 3
print(cfg.ckpt_dir, cfg.max_checkpoints)
```

## `Trainer`

- **Signature:** `Trainer(model, optimizer, config=TrainerConfig()) -> Trainer`
- **Source:** `src/bindings/bind_data.cpp:199-210`, `src/trainer/trainer.cpp`
- **Parameters:** `model` — an `nn.Module`. `optimizer` — an optimizer. `config` — a `TrainerConfig`, default-constructed when omitted.
- **Methods:** `train_step()` — one optimizer step plus `zero_grad` (see the warning below); `save_checkpoint(step)` — writes `step-<n>.safetensors` into `config.ckpt_dir`; `load_latest()` — loads the newest checkpoint into the model, returns its step or -1 when none exists; `garbage_collect()` — deletes all but the `max_checkpoints` newest checkpoints; `latest_step()` — newest checkpoint step, or -1; `rank()` — process rank (0 when not distributed); `world_size()` — 1 when not distributed; `is_distributed()` — True only under a distributed process group; `wait_pending_saves()` — blocks until async checkpoint writes finish.
- **Warning — `train_step()` does NOT run forward/backward.** It only calls `optimizer.step()` followed by `zero_grad()`. The caller must run the forward pass, compute the loss, call `loss.backward()`, and set gradients before calling `train_step()`. Calling `train_step()` on a model with no gradients is a silent no-op.
- **Edge cases:** `save_checkpoint` is asynchronous when `config.async_save` is True (the default) — the file appears only after `wait_pending_saves()`; check the directory before that and it looks empty even though `latest_step()` already reports the step. `load_latest()` returns -1 (not an exception) when no checkpoint exists. `rank()`, `world_size()`, `is_distributed()` are methods, not properties. In distributed mode `save_checkpoint` uses the elastic consistent-checkpoint path when `config.elastic` is True.
- **PyTorch:** no equivalent class in `torch` (closest: a hand-written training loop, or Hugging Face `Trainer`).

<!-- ex: id=r11-trainer device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
opt = lt.optim.SGD(m.parameters(), lr=0.1)
cfg = lt.TrainerConfig()
cfg.ckpt_dir = os.path.join(tempfile.mkdtemp(), "ckpt")
tr = lt.Trainer(m, opt, cfg)
print(tr.rank(), tr.world_size(), tr.is_distributed())
for p in m.parameters():
    p.grad = lt.Tensor.from_vector([0.1] * p.numel(), p.shape, cpu)
tr.train_step()
tr.save_checkpoint(1)
tr.wait_pending_saves()
print(sorted(os.listdir(cfg.ckpt_dir)), tr.latest_step(), tr.load_latest())
```

## RNG (`manual_seed`, `get_rng_state`, `set_rng_state`, `randn`, `ones`, `zeros`, `arange`)

- **Signatures:** `manual_seed(seed) -> None`; `get_rng_state() -> object`; `set_rng_state(state) -> None`; `randn(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `ones(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `zeros(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `arange(start, end, step=1, device=CPU, dtype=FP32) -> Tensor`
- **Source:** `src/bindings/bind_data.cpp:171-184`
- **Behavior:** `manual_seed` seeds the global CPU RNG; `get_rng_state` / `set_rng_state` save and restore it, so a sequence drawn after `set_rng_state(state)` reproduces the sequence drawn after `get_rng_state()`. `randn` draws standard-normal values.
- **Edge cases:** `manual_seed` does **not** seed `DataLoader(shuffle=True)` ordering — loader shuffling uses its own RNG. `arange` with integer dtype produces integer values.
- **PyTorch:** `torch.manual_seed` / `torch.get_rng_state` / `torch.set_rng_state` / `torch.randn` / `torch.ones` / `torch.zeros` / `torch.arange` — equivalent (note LiteTorch spells it `arange`, PyTorch `arange` too; the dtype argument order differs).

<!-- ex: id=r11-rng device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(42)
a = lt.randn([3], cpu)
st = lt.get_rng_state()
b = lt.randn([3], cpu)
lt.set_rng_state(st)
c = lt.randn([3], cpu)
print(b.to_vector() == c.to_vector())
print(lt.ones([2], cpu).to_vector(), lt.zeros([2], cpu).to_vector(), lt.arange(0, 4, 2, cpu).to_vector())
```
