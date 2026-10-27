# Ops — Reduction, Shape, and Indexing — API Reference

`litetorch.Ops.*` reduction, shape-manipulation, and indexing functions. The matching `Tensor` methods (`sum`, `mean`, `max`, `argmax`, `argmin`, `topk`, `view`, `reshape`, `permute`, `expand`, `gather`) are documented in `tensor.md`.

- **Source of truth:** `src/bindings/bind_ops.cpp:12-18, 43-57, 114-141` (bindings), `include/litetorch/ops.h:21-22, 41-68, 204-222` (C++ declarations).
- **Tree:** agent-v1@02cc271a / local HEAD 88a4ba1 (docs baseline; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-12 (single-int dim on Tensor methods vs dim lists on `Ops.*`).

## Reduction

### `Ops.sum` / `Ops.mean` / `Ops.max`

- **Signatures:** `sum(a) -> Tensor`; `sum(a, dim, keepdim=False) -> Tensor`; `sum(a, dim_list, keepdim=False) -> Tensor` (and the same three forms for `mean` and `max`)
- **Source:** `src/bindings/bind_ops.cpp:12-17, 121-125`
- **Parameters:** `dim` — a single int, or a list of ints for the third overload. Unlike the `Tensor` methods, the `Ops.*` forms DO accept a list of dims (D-12).
- **Returns:** reduced tensor; `keepdim=True` keeps reduced axes as size 1.
- **PyTorch:** `torch.sum(a, dim)` — equivalent, including tuple dims.

<!-- ex: id=r4-sum-mean-max device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.Ops.sum(a).to_vector(), lt.Ops.sum(a, [0, 1]).to_vector())
print(lt.Ops.mean(a, 0).to_vector(), lt.Ops.max(a, 1, True).to_vector())
```

### `Ops.min` / `Ops.amax` / `Ops.amin`

- **Signatures:** `min(a) -> Tensor`; `amax(a, dim, keepdim=False) -> Tensor`; `amin(a, dim, keepdim=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:114-118`
- **Parameters:** `dim` — a single int or a list of ints (the binding parses both).
- **Returns:** the minimum / max over the given dims. Note `Ops.min` reduces over ALL elements; there is no `min(a, dim)` overload — use `amin` for dim-wise minima.
- **PyTorch:** `torch.min(a)` / `torch.amax(a, dim)` / `torch.amin(a, dim)` — equivalent.

<!-- ex: id=r4-min-amax device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.Ops.min(a).to_vector(), lt.Ops.amax(a, 0).to_vector(), lt.Ops.amin(a, [0, 1]).to_vector())
```

### `Ops.max_with_indices`

- **Signature:** `max_with_indices(a, dim, keepdim=False) -> (Tensor, Tensor)`
- **Source:** `src/bindings/bind_ops.cpp:18`
- **Returns:** `(values, indices)` tuple for the max along `dim`.
- **PyTorch:** `torch.max(a, dim)` — equivalent values.

### `Ops.argmax` / `Ops.argmin`

- **Signatures:** `argmax(a, dim, keepdim=False) -> Tensor`; `argmin(a, dim, keepdim=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:55-56`
- **Behavior:** `dim` is required here (unlike the `Tensor` methods, where it defaults to -1).
- **PyTorch:** `torch.argmax(a, dim, keepdim)` — equivalent.

### `Ops.topk`

- **Signature:** `topk(a, k, dim=-1, largest=True, sorted=True) -> (Tensor, Tensor)`
- **Source:** `src/bindings/bind_ops.cpp:57`
- **Returns:** `(values, indices)` tuple.
- **PyTorch:** `torch.topk(a, k, dim, largest, sorted)` — equivalent.

<!-- ex: id=r4-arg-topk device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 3.0, 2.0, 4.0], [2, 2], cpu)
print(lt.Ops.max_with_indices(a, 1)[0].to_vector())
print(lt.Ops.argmax(a, 1).to_vector(), lt.Ops.argmin(a, 0).to_vector())
print(lt.Ops.topk(a, 1, dim=1)[0].to_vector())
```

### `Ops.sort` / `Ops.argsort`

- **Signatures:** `sort(a, dim=-1, descending=False) -> (Tensor, Tensor)`; `argsort(a, dim=-1, descending=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:133-134`
- **Returns:** `sort` returns `(values, indices)`; `argsort` returns indices only.
- **PyTorch:** `torch.sort(a, dim, descending)` / `torch.argsort(a, dim, descending)` — equivalent.

### `Ops.cumsum`

- **Signature:** `cumsum(a, dim) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:135`
- **Returns:** cumulative sum along `dim` (same shape, same dtype).
- **PyTorch:** `torch.cumsum(a, dim)` — equivalent.

<!-- ex: id=r4-sort-cumsum device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([3.0, 1.0, 2.0], [3], cpu)
print(lt.Ops.sort(a)[0].to_vector(), lt.Ops.argsort(a).to_vector())
print(lt.Ops.cumsum(a, 0).to_vector())
```

## Shape

### `Ops.cat`

- **Signature:** `cat(tensors, dim=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:43`
- **Parameters:** `tensors` — a list of tensors; all dims except `dim` must match.
- **PyTorch:** `torch.cat(tensors, dim)` — equivalent.

### `Ops.stack`

- **Signature:** `stack(tensors, dim=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:53`
- **Parameters:** `tensors` — a list of same-shape tensors; a new dimension is inserted at `dim`.
- **PyTorch:** `torch.stack(tensors, dim)` — equivalent.

<!-- ex: id=r4-cat-stack device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.Ops.cat([a, b]).to_vector(), lt.Ops.cat([a, b], 0).to_vector())
print(lt.Ops.stack([a, b]).shape, lt.Ops.stack([a, b], 1).shape)
```

### `Ops.chunk` / `Ops.split`

- **Signatures:** `chunk(input, chunks, dim=0) -> list[Tensor]`; `split(input, sizes, dim=-1) -> list[Tensor]`
- **Source:** `src/bindings/bind_ops.cpp:44-45`
- **Parameters:** `chunk` splits into `chunks` roughly equal parts; `split` splits into the exact `sizes` (list of ints summing to `shape[dim]`).
- **PyTorch:** `torch.chunk(input, chunks, dim)` / `torch.split(input, sizes, dim)` — equivalent.

### `Ops.squeeze` / `Ops.unsqueeze`

- **Signatures:** `squeeze(a, dim=None) -> Tensor`; `unsqueeze(a, dim) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:46-50`
- **Behavior:** `squeeze` with no `dim` removes all size-1 axes; with `dim` removes that axis only if it is size 1. `unsqueeze` inserts a size-1 axis at `dim`.
- **PyTorch:** `torch.squeeze(a, dim)` / `torch.unsqueeze(a, dim)` — equivalent.

<!-- ex: id=r4-chunk-squeeze device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
print([t.to_vector() for t in lt.Ops.chunk(a, 2)])
print([t.to_vector() for t in lt.Ops.split(a, [1, 3])])
b = lt.Tensor.from_vector([1.0, 2.0], [1, 2], cpu)
print(lt.Ops.squeeze(b).shape, lt.Ops.unsqueeze(a, 0).shape)
```

### `Ops.permute` / `Ops.expand`

- **Signatures:** `permute(a, dims) -> Tensor`; `expand(a, shape) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:51-52`
- **Behavior:** same as the `Tensor` methods (see `tensor.md`).
- **PyTorch:** `torch.permute(a, dims)` / `a.expand(shape)` — equivalent.

### `Ops.flatten`

- **Signature:** `flatten(a, start_dim=0, end_dim=-1) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:138`
- **Returns:** the dims from `start_dim` to `end_dim` collapsed into one.
- **PyTorch:** `torch.flatten(a, start_dim, end_dim)` — equivalent.

### `Ops.repeat`

- **Signature:** `repeat(a, repeats) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:139`
- **Parameters:** `repeats` — a list of ints, one per dim (extra leading entries add new dims).
- **PyTorch:** `a.repeat(repeats)` — equivalent.

### `Ops.narrow`

- **Signature:** `narrow(a, dim, start, length) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:140`
- **Returns:** a view of `length` elements starting at `start` along `dim`.
- **PyTorch:** `torch.narrow(a, dim, start, length)` — equivalent.

### `Ops.t`

- **Signature:** `t(a) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:141`
- **Behavior:** transposes a 1-D or 2-D tensor (1-D is returned unchanged).
- **PyTorch:** `torch.t(a)` — equivalent.

### `Ops.triu` / `Ops.tril`

- **Signatures:** `triu(a, diagonal=0) -> Tensor`; `tril(a, diagonal=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:131-132`
- **Behavior:** zeroes below/above the given diagonal.
- **PyTorch:** `torch.triu(a, diagonal)` / `torch.tril(a, diagonal)` — equivalent.

<!-- ex: id=r4-shape-rest device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.Ops.permute(a, [1, 0]).to_vector(), lt.Ops.expand(a, [2, 2]).shape)
print(lt.Ops.flatten(a).to_vector(), lt.Ops.repeat(a, [1, 2]).shape)
print(lt.Ops.narrow(a, 0, 0, 1).to_vector(), lt.Ops.t(a).to_vector())
print(lt.Ops.triu(a).to_vector(), lt.Ops.tril(a).to_vector())
```

## Indexing

### `Ops.gather`

- **Signature:** `gather(input, dim, index) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:54`
- **Behavior:** same as the `Tensor.gather` method (see `tensor.md`).
- **PyTorch:** `torch.gather(input, dim, index)` — equivalent.

### `Ops.scatter` / `Ops.scatter_add`

- **Signatures:** `scatter(input, dim, index, src) -> Tensor`; `scatter_add(input, dim, index, src) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:136-137`
- **Behavior:** `scatter` writes `src` values into `input` at `index` positions (out-of-place copy); `scatter_add` adds them instead. `index` dtype must be integer.
- **PyTorch:** `torch.scatter(input, dim, index, src)` / `torch.scatter_add(input, dim, index, src)` — equivalent.

<!-- ex: id=r4-scatter device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
base = lt.Tensor.zeros([2, 2], cpu)
idx = lt.Tensor.from_vector([1.0, 0.0], [1, 2], cpu, False, lt.DataType.INT32)
src = lt.Tensor.from_vector([9.0, 8.0], [1, 2], cpu)
print(lt.Ops.scatter(base, 1, idx, src).to_vector())
print(lt.Ops.scatter_add(base, 1, idx, src).to_vector())
print(lt.Ops.gather(lt.Ops.scatter(base, 1, idx, src), 1, idx).to_vector())
```
