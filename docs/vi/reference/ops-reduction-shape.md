# Ops — Rút Gọn, Shape và Chỉ Mục — Tài Liệu Tham Khảo API

Các hàm `litetorch.Ops.*` rút gọn, thao tác shape và chỉ mục. Các phương thức `Tensor` tương ứng (`sum`, `mean`, `max`, `argmax`, `argmin`, `topk`, `view`, `reshape`, `permute`, `expand`, `gather`) được ghi trong `tensor.md`.

- **Nguồn chân lý:** `src/bindings/bind_ops.cpp:12-18, 43-57, 114-141` (binding), `include/litetorch/ops.h:21-22, 41-68, 204-222` (khai báo C++).
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 88a4ba1 (mốc tài liệu; B5 kiểm tra lại ở head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Tương đương PyTorch là FP32 1e-4 trừ khi header ghi khác.
- **Khác biệt liên quan:** D-12 (dim đơn kiểu int trên phương thức Tensor so với danh sách dim trên `Ops.*`).

## Rút gọn

### `Ops.sum` / `Ops.mean` / `Ops.max`

- **Chữ ký:** `sum(a) -> Tensor`; `sum(a, dim, keepdim=False) -> Tensor`; `sum(a, dim_list, keepdim=False) -> Tensor` (và ba dạng tương tự cho `mean` và `max`)
- **Nguồn:** `src/bindings/bind_ops.cpp:12-17, 121-125`
- **Tham số:** `dim` — một số nguyên duy nhất, hoặc danh sách các số nguyên cho dạng nạp chồng thứ ba. Khác với phương thức `Tensor`, các dạng `Ops.*` CÓ chấp nhận danh sách dim (D-12).
- **Trả về:** tensor đã rút gọn; `keepdim=True` giữ các trục đã rút gọn ở kích thước 1.
- **PyTorch:** `torch.sum(a, dim)` — tương đương, kể cả dim dạng tuple.

<!-- ex: id=r4-sum-mean-max device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.Ops.sum(a).to_vector(), lt.Ops.sum(a, [0, 1]).to_vector())
print(lt.Ops.mean(a, 0).to_vector(), lt.Ops.max(a, 1, True).to_vector())
```

### `Ops.min` / `Ops.amax` / `Ops.amin`

- **Chữ ký:** `min(a) -> Tensor`; `amax(a, dim, keepdim=False) -> Tensor`; `amin(a, dim, keepdim=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:114-118`
- **Tham số:** `dim` — một số nguyên duy nhất hoặc danh sách các số nguyên (binding phân tích cả hai).
- **Trả về:** giá trị nhỏ nhất / lớn nhất trên các dim đã cho. Lưu ý `Ops.min` rút gọn trên TẤT CẢ phần tử; không có dạng nạp chồng `min(a, dim)` — dùng `amin` cho min theo chiều.
- **PyTorch:** `torch.min(a)` / `torch.amax(a, dim)` / `torch.amin(a, dim)` — tương đương.

<!-- ex: id=r4-min-amax device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.Ops.min(a).to_vector(), lt.Ops.amax(a, 0).to_vector(), lt.Ops.amin(a, [0, 1]).to_vector())
```

### `Ops.max_with_indices`

- **Chữ ký:** `max_with_indices(a, dim, keepdim=False) -> (Tensor, Tensor)`
- **Nguồn:** `src/bindings/bind_ops.cpp:18`
- **Trả về:** tuple `(values, indices)` cho max dọc theo `dim`.
- **PyTorch:** `torch.max(a, dim)` — giá trị tương đương.

### `Ops.argmax` / `Ops.argmin`

- **Chữ ký:** `argmax(a, dim, keepdim=False) -> Tensor`; `argmin(a, dim, keepdim=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:55-56`
- **Hành vi:** `dim` là bắt buộc ở đây (khác phương thức `Tensor`, nơi nó mặc định -1).
- **PyTorch:** `torch.argmax(a, dim, keepdim)` — tương đương.

### `Ops.topk`

- **Chữ ký:** `topk(a, k, dim=-1, largest=True, sorted=True) -> (Tensor, Tensor)`
- **Nguồn:** `src/bindings/bind_ops.cpp:57`
- **Trả về:** tuple `(values, indices)`.
- **PyTorch:** `torch.topk(a, k, dim, largest, sorted)` — tương đương.

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

- **Chữ ký:** `sort(a, dim=-1, descending=False) -> (Tensor, Tensor)`; `argsort(a, dim=-1, descending=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:133-134`
- **Trả về:** `sort` trả về `(values, indices)`; `argsort` chỉ trả về chỉ mục.
- **PyTorch:** `torch.sort(a, dim, descending)` / `torch.argsort(a, dim, descending)` — tương đương.

### `Ops.cumsum`

- **Chữ ký:** `cumsum(a, dim) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:135`
- **Trả về:** tổng tích lũy dọc theo `dim` (cùng shape, cùng kiểu).
- **PyTorch:** `torch.cumsum(a, dim)` — tương đương.

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

- **Chữ ký:** `cat(tensors, dim=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:43`
- **Tham số:** `tensors` — danh sách các tensor; mọi chiều trừ `dim` phải khớp.
- **PyTorch:** `torch.cat(tensors, dim)` — tương đương.

### `Ops.stack`

- **Chữ ký:** `stack(tensors, dim=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:53`
- **Tham số:** `tensors` — danh sách các tensor cùng shape; một chiều mới được chèn tại `dim`.
- **PyTorch:** `torch.stack(tensors, dim)` — tương đương.

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

- **Chữ ký:** `chunk(input, chunks, dim=0) -> list[Tensor]`; `split(input, sizes, dim=-1) -> list[Tensor]`
- **Nguồn:** `src/bindings/bind_ops.cpp:44-45`
- **Tham số:** `chunk` chia thành `chunks` phần gần bằng nhau; `split` chia thành các `sizes` chính xác (danh sách số nguyên có tổng bằng `shape[dim]`).
- **PyTorch:** `torch.chunk(input, chunks, dim)` / `torch.split(input, sizes, dim)` — tương đương.

### `Ops.squeeze` / `Ops.unsqueeze`

- **Chữ ký:** `squeeze(a, dim=None) -> Tensor`; `unsqueeze(a, dim) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:46-50`
- **Hành vi:** `squeeze` không `dim` loại bỏ mọi trục kích thước 1; với `dim` chỉ loại bỏ trục đó nếu nó có kích thước 1. `unsqueeze` chèn trục kích thước 1 tại `dim`.
- **PyTorch:** `torch.squeeze(a, dim)` / `torch.unsqueeze(a, dim)` — tương đương.

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

- **Chữ ký:** `permute(a, dims) -> Tensor`; `expand(a, shape) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:51-52`
- **Hành vi:** giống các phương thức `Tensor` (xem `tensor.md`).
- **PyTorch:** `torch.permute(a, dims)` / `a.expand(shape)` — tương đương.

### `Ops.flatten`

- **Chữ ký:** `flatten(a, start_dim=0, end_dim=-1) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:138`
- **Trả về:** các chiều từ `start_dim` tới `end_dim` gộp thành một.
- **PyTorch:** `torch.flatten(a, start_dim, end_dim)` — tương đương.

### `Ops.repeat`

- **Chữ ký:** `repeat(a, repeats) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:139`
- **Tham số:** `repeats` — danh sách số nguyên, một cho mỗi chiều (mục dẫn đầu thêm sẽ tạo chiều mới).
- **PyTorch:** `a.repeat(repeats)` — tương đương.

### `Ops.narrow`

- **Chữ ký:** `narrow(a, dim, start, length) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:140`
- **Trả về:** view gồm `length` phần tử bắt đầu tại `start` dọc theo `dim`.
- **PyTorch:** `torch.narrow(a, dim, start, length)` — tương đương.

### `Ops.t`

- **Chữ ký:** `t(a) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:141`
- **Hành vi:** chuyển vị tensor 1 chiều hoặc 2 chiều (1 chiều trả về không đổi).
- **PyTorch:** `torch.t(a)` — tương đương.

### `Ops.triu` / `Ops.tril`

- **Chữ ký:** `triu(a, diagonal=0) -> Tensor`; `tril(a, diagonal=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:131-132`
- **Hành vi:** đưa về 0 phía dưới/trên đường chéo đã cho.
- **PyTorch:** `torch.triu(a, diagonal)` / `torch.tril(a, diagonal)` — tương đương.

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

## Chỉ mục

### `Ops.gather`

- **Chữ ký:** `gather(input, dim, index) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:54`
- **Hành vi:** giống phương thức `Tensor.gather` (xem `tensor.md`).
- **PyTorch:** `torch.gather(input, dim, index)` — tương đương.

### `Ops.scatter` / `Ops.scatter_add`

- **Chữ ký:** `scatter(input, dim, index, src) -> Tensor`; `scatter_add(input, dim, index, src) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:136-137`
- **Hành vi:** `scatter` ghi giá trị `src` vào `input` tại vị trí `index` (bản sao, không tại chỗ); `scatter_add` cộng thay vì ghi. `index` phải có kiểu số nguyên.
- **PyTorch:** `torch.scatter(input, dim, index, src)` / `torch.scatter_add(input, dim, index, src)` — tương đương.

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
