# nn modules — Tài liệu tham khảo API

Mọi lớp đều nằm dưới `litetorch.nn`. Các chữ ký lấy từ các binding pybind11 (`src/bindings/bind_nn_base.cpp`, `bind_nn_layers.cpp`, `bind_nn_recurrent.cpp`, `bind_nn_utils.cpp`); binding là nguồn chân lý. Mỗi ví dụ đều tự chứa, chạy trên CPU và mang header xác minh (xem kế hoạch xác minh, `plan/verification.md` §4).

Hai quy tắc về dạng lời gọi áp dụng cho mọi module nhiều đối số trong file này. Thứ nhất, `__call__()` chỉ định tuyến `forward(x)` một tensor: module nào có forward nhận hai tensor trở lên — `MultiHeadAttention`, `Bilinear`, `TripletMarginWithDistanceLoss`, `GaussianNLLLoss`, `CTCLoss`, `DPOLoss` và mọi loss hai đối số — phải được gọi qua `.forward(...)` một cách rõ ràng; gọi với hai đối số sẽ gây `TypeError`. Thứ hai, `Module.to()` chỉ nhận một `litetorch.Device`, không bao giờ nhận chuỗi (D-03).
## Container và lớp cơ sở
### `nn.Module`

- **Chữ ký:** `Module()`
- **Nguồn:** `bind_nn_base.cpp:6`
- **Mô tả:** Lớp cơ sở cho mọi module. Kế thừa lớp này và ghi đè forward(). Giữ các parameter, buffer và module con; điều khiển chế độ train()/eval(), .to(device) và lưu/tải state_dict().
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** forward() is virtual in C++; a Python subclass must define forward(). .to() takes a litetorch.Device only, no string form (D-03). CRITICAL (new deviation, D-24 candidate): assigning a submodule as a Python attribute (self.linear = lt.nn.Linear(4, 4)) does NOT register it — parameters() returns 0 and state_dict() is empty. Register children explicitly with add_module('linear', module); after add_module, parameters() finds them. Torch registers attribute-assigned submodules automatically.
- **PyTorch:** torch.nn.Module — equivalent core API. LiteTorch adds compile(), save_pretrained()/load_pretrained(). Python subclassing works through the PyModule trampoline — with one major exception (see the edge case).
- **Ví dụ:**

<!-- ex: id=r7-module-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

class TwoLayer(lt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = lt.nn.Linear(4, 4)
        self.activation = lt.nn.ReLU()
    def forward(self, x):
        return self.activation(self.linear(x))

m = TwoLayer()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("parameters after attribute assignment:", len(m.parameters()))
m.add_module("linear2", lt.nn.Linear(4, 4))
print("parameters after add_module:", len(m.parameters()))
```
### `nn.Sequential`

- **Chữ ký:** `Sequential() | Sequential(modules=[...])`
- **Nguồn:** `bind_nn_base.cpp:296`
- **Mô tả:** Container có thứ tự chứa các module, áp dụng lần lượt theo thứ tự. Nhận danh sách module, hoặc rỗng rồi dùng .add().
- **Tham số:** Overloads: `Sequential()` or `Sequential(modules=[...])` where modules is a list of modules.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Two constructor overloads: Sequential() and Sequential([modules]).
- **PyTorch:** torch.nn.Sequential — equivalent. LiteTorch also accepts .add(module) after construction.
- **Ví dụ:**

<!-- ex: id=r7-sequential-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.ReLU()])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ModuleList`

- **Chữ ký:** `ModuleList(modules)`
- **Nguồn:** `bind_nn_layers.cpp:85`
- **Mô tả:** Container dạng danh sách chứa các module. Hỗ trợ chỉ mục và lặp; các module được đăng ký làm module con nên parameters() tìm thấy chúng.
- **Tham số:** `modules` (required).
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** A plain Python list does NOT register parameters; use ModuleList (or ModuleDict) inside a custom Module.
- **PyTorch:** torch.nn.ModuleList — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-module-list-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ModuleList([lt.nn.Linear(4, 4), lt.nn.ReLU()])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
for layer in m:
    x = layer(x)
print("output shape:", list(x.shape))
```
### `nn.ModuleDict`

- **Chữ ký:** `ModuleDict()`
- **Nguồn:** `bind_nn_layers.cpp:107`
- **Mô tả:** Container dạng dict chứa các module, khóa bằng chuỗi. Hỗ trợ keys(), values(), items(), __contains__, __iter__.
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Add entries with m[key] = module or put(key, module); add_module() does NOT add to the dict (getitem then raises KeyError). Supports keys(), get(key), __len__, __contains__, __iter__; no values()/items().
- **PyTorch:** torch.nn.ModuleDict — equivalent in effect, but entries are added with put(key, module) or m[key] = module, NOT with add_module() (which registers a child but does not add it to the dict). No values() or items() methods are bound.
- **Ví dụ:**

<!-- ex: id=r7-module-dict-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ModuleDict()
m["linear"] = lt.nn.Linear(4, 4)
m["relu"] = lt.nn.ReLU()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m["relu"](m["linear"](x))
print("output shape:", list(y.shape))
print("keys:", list(m.keys()))
```
### `nn.Parameter`

- **Chữ ký:** `Parameter(data, requires_grad=true)`
- **Nguồn:** `bind_nn_layers.cpp:70`
- **Mô tả:** Một tensor được đăng ký làm parameter có thể học. Tạo từ một tensor có sẵn; requires_grad là True.
- **Tham số:** `data` (required), `requires_grad` = true.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** BROKEN on 88a4ba1: Parameter(tensor) always throws 'copy_: a leaf Tensor that requires grad is being used in an in-place operation', even for requires_grad=False inputs (the binding copy_()s into the Parameter itself). No workaround exists; obtain parameters from modules (for example Linear.weight) instead. Flagged for Core.
- **PyTorch:** torch.nn.Parameter — equivalent in intent, but see the edge case: construction currently throws on this build.
- **Ví dụ:**

<!-- ex: id=r7-parameter-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Linear(4, 4)
print("parameter types:", [type(p).__name__ for p in m.parameters()])
print("requires_grad:", [p.requires_grad for p in m.parameters()])
print("weight shape:", list(m.weight.shape))
print("Parameter(tensor) construction currently throws on this build; see the edge case above.")
```
### `nn.Buffer`

- **Chữ ký:** `Buffer(data)`
- **Nguồn:** `bind_nn_layers.cpp:78`
- **Mô tả:** Một tensor được đăng ký làm buffer bền vững, không học được (ví dụ thống kê running). Di chuyển cùng .to(device) nhưng không phải parameter.
- **Tham số:** `data` (required).
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** The binding copies the input tensor; later mutation of the source does not affect the buffer.
- **PyTorch:** torch.nn.Buffer — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-buffer-1 device=cpu backend=cpu runtime=~0.7s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [2, 2], device)
b = lt.nn.Buffer(t)
print("buffer shape:", list(b.shape))
print("buffer requires_grad:", b.requires_grad)
```
### `nn.Identity`

- **Chữ ký:** `Identity()`
- **Nguồn:** `bind_nn_recurrent.cpp:247`
- **Mô tả:** Trả về đầu vào không thay đổi. Hữu ích làm lớp giữ chỗ.
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Takes no constructor arguments.
- **PyTorch:** torch.nn.Identity — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-identity-1 device=cpu backend=cpu runtime=~1.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Identity()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Flatten`

- **Chữ ký:** `Flatten()`
- **Nguồn:** `bind_nn_base.cpp:191`
- **Mô tả:** Làm phẳng mọi chiều trừ chiều batch: (N, ...) -> (N, prod).
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** No start_dim/end_dim: always flattens from dimension 1.
- **PyTorch:** torch.nn.Flatten — equivalent for the default case, but LiteTorch takes NO start_dim/end_dim arguments; the flatten range is fixed.
- **Ví dụ:**

<!-- ex: id=r7-flatten-1 device=cpu backend=cpu runtime=~0.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Flatten()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 3, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Unflatten`

- **Chữ ký:** `Unflatten(dim, unflattened_size)`
- **Nguồn:** `bind_nn_recurrent.cpp:251`
- **Mô tả:** Mở phẳng một chiều thành nhiều chiều: (N, d, ...) -> (N, *unflattened_size, ...) tại dim đã cho.
- **Tham số:** `dim` (required), `unflattened_size` (required).
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** dim is a single int64; unflattened_size is a list of ints whose product must equal the size of dim.
- **PyTorch:** torch.nn.Unflatten — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-unflatten-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Unflatten(1, [2, 2])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Dropout`

- **Chữ ký:** `Dropout(p=0.5)`
- **Nguồn:** `bind_nn_base.cpp:160`
- **Mô tả:** Đặt ngẫu nhiên mỗi phần tử về 0 với xác suất p khi huấn luyện; là identity ở chế độ eval. Đầu ra được nhân 1/(1-p) khi huấn luyện.
- **Tham số:** `p` = 0.5f.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** In training mode the output is stochastic: seed with lt.manual_seed() for reproducibility, or call eval() for a deterministic pass. Default p=0.5 matches torch.
- **PyTorch:** torch.nn.Dropout — equivalent, including the default p=0.5.
- **Ví dụ:**

<!-- ex: id=r7-dropout-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Dropout(0.5)
m.eval()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Linear
### `nn.Linear`

- **Chữ ký:** `Linear(in_features, out_features, bias=true)`
- **Nguồn:** `bind_nn_base.cpp:19`
- **Mô tả:** Áp dụng y = x @ W.T + b. Weight có shape (out_features, in_features); bias có shape (out_features,) trừ khi bias=False.
- **Tham số:** `in_features` (required), `out_features` (required), `bias` = true.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** Input (..., in_features) -> (..., out_features). Weights are initialized randomly; use nn.init.* to re-initialize.
- **PyTorch:** torch.nn.Linear — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-linear-1 device=cpu backend=cpu runtime=~0.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Linear(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Bilinear`

- **Chữ ký:** `Bilinear(in1_features, in2_features, out_features, bias=true)`
- **Nguồn:** `bind_nn_utils.cpp:153`
- **Mô tả:** Dạng song tuyến y = x1 @ W @ x2 + b với weight 3 chiều (out_features, in1_features, in2_features). Nhận HAI đầu vào.
- **Tham số:** `in1_features` (required), `in2_features` (required), `out_features` (required), `bias` = true.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** Two-input module: __call__() only routes the single-tensor forward, so call .forward(x1, x2) explicitly (see the multi-argument call note).
- **PyTorch:** torch.nn.Bilinear — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-bilinear-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Bilinear(4, 3, 2)
x1 = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
x2 = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
y = m.forward(x1, x2)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ColumnParallelLinear`

- **Chữ ký:** `ColumnParallelLinear(in_features, out_features, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:49`
- **Mô tả:** Linear song song theo tensor, chia weight theo các cột đầu ra (kiểu Megatron). Dùng trong tensor parallelism; trên một thiết bị thì hoạt động như Linear.
- **Tham số:** `in_features` (required), `out_features` (required), `has_bias` = true.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** Parameter name is has_bias (not bias). On a single process the output equals Linear with the same weight.
- **PyTorch:** torch does not ship this; Megatron-LM ColumnParallelLinear is the reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-column-parallel-linear-1 device=cpu backend=cpu runtime=~0.8s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ColumnParallelLinear(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.RowParallelLinear`

- **Chữ ký:** `RowParallelLinear(in_features, out_features, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:57`
- **Mô tả:** Linear song song theo tensor, chia weight theo các hàng đầu vào (kiểu Megatron); thường đứng sau ColumnParallelLinear và all-reduce các đầu ra từng phần.
- **Tham số:** `in_features` (required), `out_features` (required), `has_bias` = true.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** Parameter name is has_bias (not bias).
- **PyTorch:** Megatron-LM RowParallelLinear is the reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-row-parallel-linear-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.RowParallelLinear(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.VocabParallelEmbedding`

- **Chữ ký:** `VocabParallelEmbedding(vocab_size, embed_dim, padding_idx=-1)`
- **Nguồn:** `bind_nn_base.cpp:81`
- **Mô tả:** Embedding song song theo tensor, chia từ vựng cho các rank.
- **Tham số:** `vocab_size` (required), `embed_dim` (required), `padding_idx` = -1.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** BROKEN on 88a4ba1: forward() throws '[litetorch Error] data_ptr(): tensor dtype is not FP32' for both INT64 and FP32 index tensors (the lookup reads indices through data_ptr(), D-01). Construction works; forward does not. Flagged for Core.
- **PyTorch:** Megatron-LM VocabParallelEmbedding is the reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-vocab-parallel-embedding-1 device=cpu backend=cpu runtime=~0.7s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.VocabParallelEmbedding(10, 4)
print("constructed:", type(m).__name__)
print("forward() currently throws on this build: data_ptr() requires FP32 (D-01); see the edge case above.")
```
### `nn.MoELinear`

- **Chữ ký:** `MoELinear(in_features, out_features, num_experts, top_k)`
- **Nguồn:** `bind_nn_base.cpp:39`
- **Mô tả:** Linear mixture-of-experts: định tuyến mỗi token tới top_k trong số num_experts expert.
- **Tham số:** `in_features` (required), `out_features` (required), `num_experts` (required), `top_k` (required).
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** num_experts and top_k have no defaults. Input (..., in_features) -> (..., out_features).
- **PyTorch:** No direct torch.nn equivalent (torch has no built-in MoE linear); vLLM/Megatron MoE layers are the conceptual reference.
- **Ví dụ:**

<!-- ex: id=r7-mo-e-linear-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MoELinear(4, 3, num_experts=2, top_k=1)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.QLoRALinear`

- **Chữ ký:** `QLoRALinear(in_features, out_features, r=8, lora_alpha=16.0, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:28`
- **Mô tả:** Linear LoRA lượng tử hóa: weight gốc 4-bit bị đóng băng cộng với adapter hạng thấp có thể học (hạng r, hệ số lora_alpha).
- **Tham số:** `in_features` (required), `out_features` (required), `r` = 8, `lora_alpha` = 16.0f, `has_bias` = true.
- **Đầu vào:** `(..., in_features)` → **Đầu ra:** `(..., out_features)`
- **Trường hợp biên:** r=8 and lora_alpha=16.0 by default; has_bias=True by default.
- **PyTorch:** No direct torch.nn equivalent; bitsandbytes QLoRA is the conceptual reference.
- **Ví dụ:**

<!-- ex: id=r7-q-lo-ra-linear-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.QLoRALinear(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Tích chập và pooling
### `nn.Conv1d`

- **Chữ ký:** `Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Nguồn:** `bind_nn_layers.cpp:254`
- **Mô tả:** Tích chập 1 chiều trên (N, C_in, L). kernel_size/stride/padding nhận int hoặc tuple một phần tử.
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** No dilation/groups arguments (the binding takes scalar stride/padding only). Weight shape (C_out, C_in, K).
- **PyTorch:** torch.nn.Conv1d — equivalent for scalar args, except LiteTorch has no dilation or groups arguments.
- **Ví dụ:**

<!-- ex: id=r7-conv1d-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Conv1d(3, 4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 3, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Conv2d`

- **Chữ ký:** `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:93`
- **Mô tả:** Tích chập 2 chiều trên (N, C_in, H, W). kernel_size/stride/padding nhận int hoặc danh sách 2 phần tử.
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** No dilation/groups arguments. Weight shape (C_out, C_in, K_h, K_w).
- **PyTorch:** torch.nn.Conv2d — equivalent for scalar/list args, except LiteTorch has no dilation or groups arguments. The fused variant conv2d_relu lives under Ops, not nn.
- **Ví dụ:**

<!-- ex: id=r7-conv2d-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Conv2d(3, 4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 3, 8, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Conv3d`

- **Chữ ký:** `Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:115`
- **Mô tả:** Tích chập 3 chiều trên (N, C_in, D, H, W). kernel_size/stride/padding chỉ là int vô hướng.
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Scalar stride/padding only; no tuple form (unlike Conv2d). Weight shape (C_out, C_in, K, K, K).
- **PyTorch:** torch.nn.Conv3d — equivalent for scalar args, except LiteTorch has no dilation or groups arguments.
- **Ví dụ:**

<!-- ex: id=r7-conv3d-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Conv3d(2, 3, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ConvTranspose1d`

- **Chữ ký:** `ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Nguồn:** `bind_nn_recurrent.cpp:140`
- **Mô tả:** Tích chập chuyển vị 1 chiều (upsampling có thể học) trên (N, C_in, L).
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Unlike the forward convs, this class DOES take groups and dilation. Output length (L-1)*stride - 2*padding + kernel_size + output_padding.
- **PyTorch:** torch.nn.ConvTranspose1d — equivalent, including groups and dilation arguments (unlike Conv1d).
- **Ví dụ:**

<!-- ex: id=r7-conv-transpose1d-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ConvTranspose1d(2, 3, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ConvTranspose2d`

- **Chữ ký:** `ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Nguồn:** `bind_nn_recurrent.cpp:151`
- **Mô tả:** Tích chập chuyển vị 2 chiều (upsampling có thể học) trên (N, C_in, H, W).
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Unlike Conv2d, this class DOES take groups and dilation.
- **PyTorch:** torch.nn.ConvTranspose2d — equivalent, including groups and dilation arguments.
- **Ví dụ:**

<!-- ex: id=r7-conv-transpose2d-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ConvTranspose2d(2, 3, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ConvTranspose3d`

- **Chữ ký:** `ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Nguồn:** `bind_nn_recurrent.cpp:162`
- **Mô tả:** Tích chập chuyển vị 3 chiều (upsampling có thể học) trên (N, C_in, D, H, W).
- **Tham số:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Unlike Conv3d, this class DOES take groups and dilation.
- **PyTorch:** torch.nn.ConvTranspose3d — equivalent, including groups and dilation arguments.
- **Ví dụ:**

<!-- ex: id=r7-conv-transpose3d-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ConvTranspose3d(2, 3, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 2, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.MaxPool2d`

- **Chữ ký:** `MaxPool2d(kernel_size, stride=-1, padding=0)`
- **Nguồn:** `bind_nn_base.cpp:140`
- **Mô tả:** Max pooling 2 chiều trên (N, C, H, W). stride mặc định là -1, nghĩa là stride = kernel_size.
- **Tham số:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** stride=-1 means stride equals kernel_size. No ceil_mode / return_indices / dilation.
- **PyTorch:** torch.nn.MaxPool2d — equivalent. LiteTorch has no ceil_mode, return_indices, or dilation arguments.
- **Ví dụ:**

<!-- ex: id=r7-max-pool2d-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MaxPool2d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.MaxPool3d`

- **Chữ ký:** `MaxPool3d(kernel_size, stride=-1, padding=0)`
- **Nguồn:** `bind_nn_base.cpp:147`
- **Mô tả:** Max pooling 3 chiều trên (N, C, D, H, W). stride mặc định là -1, nghĩa là stride = kernel_size.
- **Tham số:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** stride=-1 means stride equals kernel_size. No ceil_mode / return_indices / dilation.
- **PyTorch:** torch.nn.MaxPool3d — equivalent. LiteTorch has no ceil_mode, return_indices, or dilation arguments.
- **Ví dụ:**

<!-- ex: id=r7-max-pool3d-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MaxPool3d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.AvgPool2d`

- **Chữ ký:** `AvgPool2d(kernel_size, stride=-1, padding=0)`
- **Nguồn:** `bind_nn_layers.cpp:276`
- **Mô tả:** Average pooling 2 chiều trên (N, C, H, W). stride mặc định là -1, nghĩa là stride = kernel_size.
- **Tham số:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** stride=-1 means stride equals kernel_size. No ceil_mode / count_include_pad.
- **PyTorch:** torch.nn.AvgPool2d — equivalent for the basic case. LiteTorch has no ceil_mode or count_include_pad arguments.
- **Ví dụ:**

<!-- ex: id=r7-avg-pool2d-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.AvgPool2d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.AdaptiveAvgPool2d`

- **Chữ ký:** `AdaptiveAvgPool2d(output_height, output_width)`
- **Nguồn:** `bind_nn_base.cpp:154`
- **Mô tả:** Adaptive average pooling 2 chiều về kích thước đầu ra cố định, cho bằng hai int riêng biệt (output_height, output_width).
- **Tham số:** `output_height` (required), `output_width` (required).
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Takes (output_height, output_width) as two ints, not an output_size tuple.
- **PyTorch:** torch.nn.AdaptiveAvgPool2d — equivalent in effect, but torch takes a single output_size (int or tuple); LiteTorch takes two separate int arguments.
- **Ví dụ:**

<!-- ex: id=r7-adaptive-avg-pool2d-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.AdaptiveAvgPool2d(2, 2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 3, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Fold`

- **Chữ ký:** `Fold(output_size, kernel_size, dilation=[1, padding=[0, stride=[1)`
- **Nguồn:** `bind_nn_utils.cpp:157`
- **Mô tả:** Kết hợp mảng các block cục bộ trượt thành một tensor lớn chứa chúng (nghịch đảo của Unfold).
- **Tham số:** `output_size` (required), `kernel_size` (required), `dilation` = std::vector<int64_t>{1, `padding` = std::vector<int64_t>{0, `stride` = std::vector<int64_t>{1.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Input is (N, C * prod(kernel_size), L) where L is the number of sliding blocks; output_size and kernel_size are lists.
- **PyTorch:** torch.nn.Fold — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-fold-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Fold(output_size=[8, 8], kernel_size=[3, 3])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4], [1, 9, 36], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Unfold`

- **Chữ ký:** `Unfold(kernel_size, dilation=[1, padding=[0, stride=[1)`
- **Nguồn:** `bind_nn_utils.cpp:162`
- **Mô tả:** Trích xuất các block cục bộ trượt từ tensor đầu vào theo batch (im2col).
- **Tham số:** `kernel_size` (required), `dilation` = std::vector<int64_t>{1, `padding` = std::vector<int64_t>{0, `stride` = std::vector<int64_t>{1.
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Output is (N, C * prod(kernel_size), L). kernel_size is a list; dilation/padding/stride are lists with defaults.
- **PyTorch:** torch.nn.Unfold — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-unfold-1 device=cpu backend=cpu runtime=~1.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Unfold(kernel_size=[3, 3])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 1, 8, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.PixelShuffle`

- **Chữ ký:** `PixelShuffle(upscale_factor)`
- **Nguồn:** `bind_nn_recurrent.cpp:237`
- **Mô tả:** Sắp xếp lại (N, C*r^2, H, W) thành (N, C, H*r, W*r) để upsampling sub-pixel.
- **Tham số:** `upscale_factor` (required).
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Input channels must be divisible by upscale_factor^2.
- **PyTorch:** torch.nn.PixelShuffle — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-pixel-shuffle-1 device=cpu backend=cpu runtime=~0.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.PixelShuffle(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 4, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.PixelUnshuffle`

- **Chữ ký:** `PixelUnshuffle(downscale_factor)`
- **Nguồn:** `bind_nn_recurrent.cpp:242`
- **Mô tả:** Nghịch đảo của PixelShuffle: sắp xếp lại (N, C, H*r, W*r) thành (N, C*r^2, H, W).
- **Tham số:** `downscale_factor` (required).
- **Đầu vào:** `(N, C_in, ...) — see the example` → **Đầu ra:** `(N, C_out, ...) — see the example`
- **Trường hợp biên:** Input height and width must be divisible by downscale_factor.
- **PyTorch:** torch.nn.PixelUnshuffle — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-pixel-unshuffle-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.PixelUnshuffle(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 1, 4, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Chuẩn hóa
### `nn.LayerNorm`

- **Chữ ký:** `LayerNorm(normalized_shape, eps=1e-5)`
- **Nguồn:** `bind_nn_base.cpp:208`
- **Mô tả:** Chuẩn hóa trên các chiều cuối normalized_shape: y = (x - mean) / sqrt(var + eps) * weight + bias.
- **Tham số:** `normalized_shape` (required), `eps` = 1e-5f.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** normalized_shape MUST be a list of ints (vector<int64_t>); a bare int is rejected. eps defaults to 1e-5.
- **PyTorch:** torch.nn.LayerNorm — equivalent. LiteTorch requires normalized_shape as a list; torch also accepts a bare int.
- **Ví dụ:**

<!-- ex: id=r7-layer-norm-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.LayerNorm(normalized_shape=[4])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.RMSNorm`

- **Chữ ký:** `RMSNorm(normalized_shape, eps=1e-5)`
- **Nguồn:** `bind_nn_base.cpp:249`
- **Mô tả:** Chuẩn hóa RMS trên các chiều cuối normalized_shape, không trừ mean: y = x / sqrt(mean(x^2) + eps) * weight.
- **Tham số:** `normalized_shape` (required), `eps` = 1e-5f.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** normalized_shape MUST be a list of ints. No bias term.
- **PyTorch:** torch has no RMSNorm module (it is torch.nn.functional.rms_norm in recent versions). LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-rms-norm-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.RMSNorm(normalized_shape=[4])
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.BatchNorm1d`

- **Chữ ký:** `BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Nguồn:** `bind_nn_layers.cpp:239`
- **Mô tả:** Batch normalization trên (N, C) hoặc (N, C, L). Dùng thống kê batch khi huấn luyện, thống kê running khi eval.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm1d — equivalent, including the default eps=1e-5, momentum=0.1, affine=True, track_running_stats=True.
- **Ví dụ:**

<!-- ex: id=r7-batch-norm1d-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BatchNorm1d(4)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.BatchNorm2d`

- **Chữ ký:** `BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Nguồn:** `bind_nn_base.cpp:125`
- **Mô tả:** Batch normalization trên (N, C, H, W). Dùng thống kê batch khi huấn luyện, thống kê running khi eval.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm2d — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-batch-norm2d-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BatchNorm2d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.BatchNorm3d`

- **Chữ ký:** `BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Nguồn:** `bind_nn_recurrent.cpp:200`
- **Mô tả:** Batch normalization trên (N, C, D, H, W). Dùng thống kê batch khi huấn luyện, thống kê running khi eval.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm3d — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-batch-norm3d-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BatchNorm3d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 2, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.SyncBatchNorm`

- **Chữ ký:** `SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Nguồn:** `bind_nn_recurrent.cpp:211`
- **Mô tả:** Batch normalization đồng bộ thống kê giữa các process khi huấn luyện phân tán.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** Channel BatchNorm: the input must be rank 3 (N, C, L) — a rank-2 (N, C) input raises 'channel BatchNorm: bad input rank or channels'. On a single process it behaves like BatchNorm1d. Cross-process sync requires a process group (see the distributed reference).
- **PyTorch:** torch.nn.SyncBatchNorm — equivalent in intent.
- **Ví dụ:**

<!-- ex: id=r7-sync-batch-norm-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SyncBatchNorm(4)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.GroupNorm`

- **Chữ ký:** `GroupNorm(num_groups, num_channels, eps=1e-5, affine=true)`
- **Nguồn:** `bind_nn_layers.cpp:227`
- **Mô tả:** Chuẩn hóa theo nhóm kênh: num_channels được chia thành num_groups nhóm, mỗi nhóm chuẩn hóa độc lập.
- **Tham số:** `num_groups` (required), `num_channels` (required), `eps` = 1e-5f, `affine` = true.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** num_channels must be divisible by num_groups. affine=True by default.
- **PyTorch:** torch.nn.GroupNorm — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-group-norm-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GroupNorm(2, 4)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.InstanceNorm1d`

- **Chữ ký:** `InstanceNorm1d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:173`
- **Mô tả:** Instance normalization trên (N, C, L): mỗi lát (N, C) được chuẩn hóa độc lập.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm1d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Ví dụ:**

<!-- ex: id=r7-instance-norm1d-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.InstanceNorm1d(4)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4, 3], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.InstanceNorm2d`

- **Chữ ký:** `InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:182`
- **Mô tả:** Instance normalization trên (N, C, H, W): mỗi lát (N, C) được chuẩn hóa độc lập.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm2d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Ví dụ:**

<!-- ex: id=r7-instance-norm2d-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.InstanceNorm2d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.InstanceNorm3d`

- **Chữ ký:** `InstanceNorm3d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:191`
- **Mô tả:** Instance normalization trên (N, C, D, H, W): mỗi lát (N, C) được chuẩn hóa độc lập.
- **Tham số:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Đầu vào:** `(N, C, ...) normalized over the feature dims` → **Đầu ra:** `(N, C, ...), same shape`
- **Trường hợp biên:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm3d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Ví dụ:**

<!-- ex: id=r7-instance-norm3d-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.InstanceNorm3d(2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 2, 2, 2, 2], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Hàm kích hoạt
### `nn.ReLU`

- **Chữ ký:** `ReLU()`
- **Nguồn:** `bind_nn_base.cpp:165`
- **Mô tả:** ReLU: max(0, x) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Gradient is 0 for x < 0 and 1 for x > 0.
- **PyTorch:** torch.nn.ReLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-re-lu-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ReLU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ReLU6`

- **Chữ ký:** `ReLU6()`
- **Nguồn:** `bind_nn_utils.cpp:102`
- **Mô tả:** ReLU giới hạn ở 6: min(max(0, x), 6) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters.
- **PyTorch:** torch.nn.ReLU6 — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-re-lu6-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ReLU6()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.LeakyReLU`

- **Chữ ký:** `LeakyReLU(negative_slope=0.01)`
- **Nguồn:** `bind_nn_base.cpp:169`
- **Mô tả:** Leaky ReLU: x khi x >= 0, negative_slope * x trong các trường hợp còn lại.
- **Tham số:** `negative_slope` = 0.01f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** negative_slope defaults to 0.01.
- **PyTorch:** torch.nn.LeakyReLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-leaky-re-lu-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.LeakyReLU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Sigmoid`

- **Chữ ký:** `Sigmoid()`
- **Nguồn:** `bind_nn_base.cpp:174`
- **Mô tả:** Sigmoid: 1 / (1 + exp(-x)) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Output range (0, 1).
- **PyTorch:** torch.nn.Sigmoid — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-sigmoid-1 device=cpu backend=cpu runtime=~0.8s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Sigmoid()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Tanh`

- **Chữ ký:** `Tanh()`
- **Nguồn:** `bind_nn_base.cpp:178`
- **Mô tả:** Tang hyperbolic theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Output range (-1, 1).
- **PyTorch:** torch.nn.Tanh — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-tanh-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Tanh()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.GELU`

- **Chữ ký:** `GELU()`
- **Nguồn:** `bind_nn_base.cpp:182`
- **Mô tả:** GELU. LiteTorch chỉ dùng xấp xỉ tanh; không có đối số approximate.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** GELU is tanh-approximation only (D-09). Compare against F.gelu(approximate='tanh') at parity time, or allow ~5e-4 tolerance against the torch default.
- **PyTorch:** torch.nn.GELU — differs: torch defaults to the exact erf form. LiteTorch matches torch.nn.functional.gelu(x, approximate='tanh') (D-09, max diff vs exact erf ~4.4e-4).
- **Ví dụ:**

<!-- ex: id=r7-gelu-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GELU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.SiLU`

- **Chữ ký:** `SiLU()`
- **Nguồn:** `bind_nn_layers.cpp:312`
- **Mô tả:** SiLU: x * sigmoid(x) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Also bound as a Tensor method (.silu()).
- **PyTorch:** torch.nn.SiLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-si-lu-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SiLU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.ELU`

- **Chữ ký:** `ELU(alpha=1.0)`
- **Nguồn:** `bind_nn_recurrent.cpp:220`
- **Mô tả:** ELU: x khi x > 0, alpha * (exp(x) - 1) trong các trường hợp còn lại.
- **Tham số:** `alpha` = 1.0f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** alpha defaults to 1.0.
- **PyTorch:** torch.nn.ELU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-elu-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.ELU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.CELU`

- **Chữ ký:** `CELU(alpha=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:110`
- **Mô tả:** CELU: x khi x > 0, alpha * (exp(x / alpha) - 1) trong các trường hợp còn lại.
- **Tham số:** `alpha` = 1.0f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** alpha defaults to 1.0.
- **PyTorch:** torch.nn.CELU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-celu-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.CELU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.SELU`

- **Chữ ký:** `SELU()`
- **Nguồn:** `bind_nn_utils.cpp:114`
- **Mô tả:** SELU với hệ số scale cố định (lambda ~1.0507) và alpha (~1.6733).
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Designed for self-normalizing networks; use with the matching initialization.
- **PyTorch:** torch.nn.SELU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-selu-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SELU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Mish`

- **Chữ ký:** `Mish()`
- **Nguồn:** `bind_nn_utils.cpp:117`
- **Mô tả:** Mish: x * tanh(softplus(x)) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters.
- **PyTorch:** torch.nn.Mish — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-mish-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Mish()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Softplus`

- **Chữ ký:** `Softplus(beta=1.0, threshold=20.0)`
- **Nguồn:** `bind_nn_utils.cpp:120`
- **Mô tả:** Softplus: (1/beta) * log(1 + exp(beta * x)), giới hạn ở threshold để ổn định số học.
- **Tham số:** `beta` = 1.0f, `threshold` = 20.0f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** beta defaults to 1.0, threshold to 20.0.
- **PyTorch:** torch.nn.Softplus — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-softplus-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Softplus()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Softshrink`

- **Chữ ký:** `Softshrink(lambd=0.5)`
- **Nguồn:** `bind_nn_utils.cpp:125`
- **Mô tả:** Soft shrinkage: x - lambd khi x > lambd, x + lambd khi x < -lambd, còn lại là 0.
- **Tham số:** `lambd` = 0.5f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** lambd defaults to 0.5.
- **PyTorch:** torch.nn.Softshrink — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-softshrink-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Softshrink()
x = lt.Tensor.from_vector([-1.0, -0.3, 0.0, 0.3, 1.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Softsign`

- **Chữ ký:** `Softsign()`
- **Nguồn:** `bind_nn_utils.cpp:136`
- **Mô tả:** Softsign: x / (1 + |x|) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters. Output range (-1, 1).
- **PyTorch:** torch.nn.Softsign — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-softsign-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Softsign()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Hardshrink`

- **Chữ ký:** `Hardshrink(lambd=0.5)`
- **Nguồn:** `bind_nn_utils.cpp:129`
- **Mô tả:** Hard shrinkage: x khi |x| > lambd, còn lại là 0.
- **Tham số:** `lambd` = 0.5f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** lambd defaults to 0.5.
- **PyTorch:** torch.nn.Hardshrink — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-hardshrink-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Hardshrink()
x = lt.Tensor.from_vector([-1.0, -0.3, 0.0, 0.3, 1.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Hardtanh`

- **Chữ ký:** `Hardtanh(min_val=-1.0, max_val=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:139`
- **Mô tả:** Hard tanh: x được kẹp trong [min_val, max_val].
- **Tham số:** `min_val` = -1.0f, `max_val` = 1.0f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** min_val defaults to -1.0, max_val to 1.0.
- **PyTorch:** torch.nn.Hardtanh — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-hardtanh-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Hardtanh()
x = lt.Tensor.from_vector([-2.0, -0.5, 0.0, 0.5, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Threshold`

- **Chữ ký:** `Threshold(threshold, value)`
- **Nguồn:** `bind_nn_utils.cpp:144`
- **Mô tả:** Threshold: x khi x > threshold, value trong các trường hợp còn lại.
- **Tham số:** `threshold` (required), `value` (required).
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** threshold and value are required (no defaults).
- **PyTorch:** torch.nn.Threshold — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-threshold-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Threshold(0.0, -1.0)
x = lt.Tensor.from_vector([-1.0, 0.0, 1.0], [3], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Tanhshrink`

- **Chữ ký:** `Tanhshrink()`
- **Nguồn:** `bind_nn_utils.cpp:133`
- **Mô tả:** Tanh shrinkage: x - tanh(x) theo từng phần tử.
- **Tham số:** None.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** No parameters.
- **PyTorch:** torch.nn.Tanhshrink — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-tanhshrink-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Tanhshrink()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.PReLU`

- **Chữ ký:** `PReLU(num_parameters=1, init=0.25f)`
- **Nguồn:** `bind_nn_recurrent.cpp:225`
- **Mô tả:** PReLU: x khi x >= 0, a * x trong các trường hợp còn lại, với a là parameter có thể học (một cho mỗi kênh hoặc một giá trị chung).
- **Tham số:** `num_parameters` = 1, `init` = 0.25f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** num_parameters defaults to 1, init to 0.25.
- **PyTorch:** torch.nn.PReLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-p-re-lu-1 device=cpu backend=cpu runtime=~0.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.PReLU()
x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.RReLU`

- **Chữ ký:** `RReLU(lower=0.125f, upper=0.3333333333333333f)`
- **Nguồn:** `bind_nn_utils.cpp:105`
- **Mô tả:** RReLU: hệ số dốc âm được lấy mẫu đều từ [lower, upper] khi huấn luyện và cố định bằng (lower + upper) / 2 khi eval.
- **Tham số:** `lower` = 0.125f, `upper` = 0.3333333333333333f.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** lower defaults to 0.125, upper to 1/3. Stochastic in training mode: use eval() for a deterministic pass.
- **PyTorch:** torch.nn.RReLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-r-re-lu-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.RReLU()
m.eval()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.GLU`

- **Chữ ký:** `GLU(dim=-1)`
- **Nguồn:** `bind_nn_recurrent.cpp:232`
- **Mô tả:** GLU: tách đầu vào làm đôi theo dim và trả về a * sigmoid(b).
- **Tham số:** `dim` = -1.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** dim defaults to -1. The size of dim must be even.
- **PyTorch:** torch.nn.GLU — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-glu-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GLU()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.LogSoftmax`

- **Chữ ký:** `LogSoftmax(dim=-1)`
- **Nguồn:** `bind_nn_layers.cpp:316`
- **Mô tả:** Log-softmax theo dim: log(softmax(x)).
- **Tham số:** `dim` = -1.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** dim defaults to -1. Also bound as a Tensor method (.log_softmax()).
- **PyTorch:** torch.nn.LogSoftmax — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-log-softmax-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.LogSoftmax(dim=-1)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.Softmax`

- **Chữ ký:** `Softmax(dim=-1)`
- **Nguồn:** `bind_nn_base.cpp:186`
- **Mô tả:** Softmax theo dim.
- **Tham số:** `dim` = -1.
- **Đầu vào:** `(*)` → **Đầu ra:** `(*), same shape`
- **Trường hợp biên:** dim defaults to -1.
- **PyTorch:** torch.nn.Softmax — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-softmax-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Softmax(dim=-1)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Embedding
### `nn.Embedding`

- **Chữ ký:** `Embedding(num_embeddings, embedding_dim, padding_idx=py::none()`
- **Nguồn:** `bind_nn_base.cpp:195`
- **Mô tả:** Bảng tra cứu: ánh xạ ID token nguyên thành vector dày đặc có kích thước embedding_dim.
- **Tham số:** `num_embeddings` (required), `embedding_dim` (required), `padding_idx` = py::none(.
- **Đầu vào:** `(N,) integer tensor` → **Đầu ra:** `(N, embedding_dim) for Embedding; (1, embedding_dim) for EmbeddingBag (mean mode)`
- **Trường hợp biên:** Input must be an integer tensor (INT64). The weight is a learnable parameter of shape (num_embeddings, embedding_dim).
- **PyTorch:** torch.nn.Embedding — equivalent. padding_idx defaults to no padding (None).
- **Ví dụ:**

<!-- ex: id=r7-embedding-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Embedding(10, 4)
ids = lt.tensor([1, 2, 3, 4], dtype=lt.DataType.INT64)
y = m(ids)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.EmbeddingBag`

- **Chữ ký:** `EmbeddingBag(num_embeddings, embedding_dim, mode="mean")`
- **Nguồn:** `bind_nn_recurrent.cpp:129`
- **Mô tả:** Embedding bag: cộng hoặc lấy trung bình các embedding của một túi ID token (mode 'mean' mặc định).
- **Tham số:** `num_embeddings` (required), `embedding_dim` (required), `mode` = "mean".
- **Đầu vào:** `(N,) integer tensor` → **Đầu ra:** `(N, embedding_dim) for Embedding; (1, embedding_dim) for EmbeddingBag (mean mode)`
- **Trường hợp biên:** mode defaults to 'mean'. Input is a 1-D index tensor; offsets are not exposed in this binding.
- **PyTorch:** torch.nn.EmbeddingBag — equivalent for the default (offsets-free) call.
- **Ví dụ:**

<!-- ex: id=r7-embedding-bag-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.EmbeddingBag(10, 4)
ids = lt.tensor([1, 2, 3], dtype=lt.DataType.INT64)
y = m(ids)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Hồi quy
### `nn.RNNCell`

- **Chữ ký:** `RNNCell(input_size, hidden_size, bias=true, nonlinearity="tanh")`
- **Nguồn:** `bind_nn_recurrent.cpp:6`
- **Mô tả:** Một bước của RNN thường: h' = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh).
- **Tham số:** `input_size` (required), `hidden_size` (required), `bias` = true, `nonlinearity` = "tanh".
- **Đầu vào:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Đầu ra:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Trường hợp biên:** nonlinearity defaults to 'tanh' ('relu' is the other option). forward(x) uses a zero hidden state; forward(x, h) continues from h.
- **PyTorch:** torch.nn.RNNCell — equivalent. The single-argument forward uses a zero initial hidden state.
- **Ví dụ:**

<!-- ex: id=r7-rnn-cell-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.RNNCell(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
h = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
h2 = m.forward(x, h)
print("output shape:", list(h2.shape))
print("output:", h2.to_vector()[:8])
```
### `nn.LSTMCell`

- **Chữ ký:** `LSTMCell(input_size, hidden_size, bias=true)`
- **Nguồn:** `bind_nn_recurrent.cpp:22`
- **Mô tả:** Một bước của LSTM: trả về (h', c') khi cho đầu vào x và (h, c) trước đó.
- **Tham số:** `input_size` (required), `hidden_size` (required), `bias` = true.
- **Đầu vào:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Đầu ra:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Trường hợp biên:** forward(x) uses zero (h, c); forward(x, h, c) continues from the given state. The 1-arg call returns a (h, c) tuple.
- **PyTorch:** torch.nn.LSTMCell — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-lstm-cell-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.LSTMCell(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
h = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
c = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
h2, c2 = m.forward(x, h, c)
print("h shape:", list(h2.shape))
print("c shape:", list(c2.shape))
```
### `nn.GRUCell`

- **Chữ ký:** `GRUCell(input_size, hidden_size, bias=true)`
- **Nguồn:** `bind_nn_recurrent.cpp:35`
- **Mô tả:** Một bước của GRU: h' = (1 - z) * n + z * h.
- **Tham số:** `input_size` (required), `hidden_size` (required), `bias` = true.
- **Đầu vào:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Đầu ra:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Trường hợp biên:** forward(x) uses a zero hidden state; forward(x, h) continues from h.
- **PyTorch:** torch.nn.GRUCell — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-gru-cell-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GRUCell(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
h = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
h2 = m.forward(x, h)
print("output shape:", list(h2.shape))
print("output:", h2.to_vector()[:8])
```
### `nn.RNN`

- **Chữ ký:** `RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:50`
- **Mô tả:** RNN thường nhiều lớp trên một chuỗi. forward(x) trả về (output, h_n); forward_full(x, h0) nhận trạng thái khởi tạo rõ ràng.
- **Tham số:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `nonlinearity` = "tanh", `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Đầu vào:** `(seq, batch, input_size)` → **Đầu ra:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Trường hợp biên:** batch_first defaults to False. forward() returns a tuple (output, h_n) — unpack it. dropout applies between layers only when num_layers > 1.
- **PyTorch:** torch.nn.RNN — equivalent, including the (output, h_n) return. batch_first defaults to False: input is (seq, batch, input_size).
- **Ví dụ:**

<!-- ex: id=r7-rnn-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.RNN(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [3, 2, 4], device)
out, h = m(x)
print("output shape:", list(out.shape))
print("h shape:", list(h.shape))
```
### `nn.LSTM`

- **Chữ ký:** `LSTM(input_size, hidden_size, num_layers=1, bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:68`
- **Mô tả:** LSTM nhiều lớp trên một chuỗi. forward(x) trả về (output, (h_n, c_n)).
- **Tham số:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Đầu vào:** `(seq, batch, input_size)` → **Đầu ra:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Trường hợp biên:** batch_first defaults to False. forward() returns (output, (h_n, c_n)) — unpack the nested tuple.
- **PyTorch:** torch.nn.LSTM — equivalent, including the nested (h_n, c_n) return. batch_first defaults to False.
- **Ví dụ:**

<!-- ex: id=r7-lstm-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.LSTM(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [3, 2, 4], device)
out, (h, c) = m(x)
print("output shape:", list(out.shape))
print("h shape:", list(h.shape))
print("c shape:", list(c.shape))
```
### `nn.GRU`

- **Chữ ký:** `GRU(input_size, hidden_size, num_layers=1, bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:87`
- **Mô tả:** GRU nhiều lớp trên một chuỗi. forward(x) trả về (output, h_n).
- **Tham số:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Đầu vào:** `(seq, batch, input_size)` → **Đầu ra:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Trường hợp biên:** batch_first defaults to False. forward() returns a tuple (output, h_n) — unpack it.
- **PyTorch:** torch.nn.GRU — equivalent. batch_first defaults to False.
- **Ví dụ:**

<!-- ex: id=r7-gru-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GRU(4, 3)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [3, 2, 4], device)
out, h = m(x)
print("output shape:", list(out.shape))
print("h shape:", list(h.shape))
```
## Attention, transformer và GPT
### `nn.MultiHeadAttention`

- **Chữ ký:** `MultiHeadAttention(embed_dim, num_heads, dropout=0.0, bias=true, kdim=-1, vdim=-1, batch_first=false)`
- **Nguồn:** `bind_nn_base.cpp:218`
- **Mô tả:** Attention đa đầu: softmax(QK^T / sqrt(d_k)) V với các projection đã học. Lời gọi 3 đối số phải đi qua .forward(q, k, v).
- **Tham số:** `embed_dim` (required), `num_heads` (required), `dropout` = 0.0, `bias` = true, `kdim` = -1, `vdim` = -1, `batch_first` = false.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** CRITICAL: __call__() only routes the single-tensor forward, so m(q, k, v) raises TypeError. Call m.forward(q, k, v) instead. batch_first defaults to False, so inputs are (seq, batch, embed_dim). kdim/vdim default to -1 (same as embed_dim).
- **PyTorch:** torch.nn.MultiheadAttention — equivalent in effect, but the call shape differs: torch allows m(q, k, v) while LiteTorch requires m.forward(q, k, v) (see the edge case). D-20: there is no F.scaled_dot_product_attention; use Ops.flash_attention for the fused kernel.
- **Ví dụ:**

<!-- ex: id=r7-multi-head-attention-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MultiHeadAttention(embed_dim=8, num_heads=2)
q = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
k = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
v = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
result = m.forward(q, k, v)
out = result[0] if isinstance(result, tuple) else result
print("output shape:", list(out.shape))
print("output:", out.to_vector()[:8])
```
### `nn.TPAttention`

- **Chữ ký:** `TPAttention(embed_dim, num_heads, n_kv_heads=-1, has_bias=true)`
- **Nguồn:** `bind_nn_base.cpp:66`
- **Mô tả:** Attention đa đầu song song theo tensor, hỗ trợ grouped-query (n_kv_heads có thể khác num_heads).
- **Tham số:** `embed_dim` (required), `num_heads` (required), `n_kv_heads` = -1, `has_bias` = true.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** n_kv_heads defaults to -1 (same as num_heads). Input is (seq, batch, embed_dim).
- **PyTorch:** No direct torch.nn equivalent; Megatron-style grouped-query attention is the reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-tp-attention-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.TPAttention(embed_dim=8, num_heads=2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
result = m.forward(x)
out = result[0] if isinstance(result, tuple) else result
print("output shape:", list(out.shape))
print("output:", out.to_vector()[:8])
```
### `nn.TransformerEncoderLayer`

- **Chữ ký:** `TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=false)`
- **Nguồn:** `bind_nn_recurrent.cpp:104`
- **Mô tả:** Một lớp encoder transformer: self-attention, feed-forward, kết nối residual và layer norm.
- **Tham số:** `d_model` (required), `nhead` (required), `dim_feedforward` = 2048, `dropout` = 0.1f, `activation` = "relu", `batch_first` = false.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** d_model and nhead are required; dim_feedforward defaults to 2048, dropout to 0.1, activation to 'relu', batch_first to False.
- **PyTorch:** torch.nn.TransformerEncoderLayer — equivalent for the basic case.
- **Ví dụ:**

<!-- ex: id=r7-transformer-encoder-layer-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.TransformerEncoderLayer(d_model=8, nhead=2)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.TransformerDecoderLayer`

- **Chữ ký:** `TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=2048)`
- **Nguồn:** `bind_nn_base.cpp:242`
- **Mô tả:** Một lớp decoder transformer: masked self-attention, cross-attention, feed-forward, residual và norm.
- **Tham số:** `embed_dim` (required), `num_heads` (required), `dim_feedforward` = 2048.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** DIVERGES from torch (D-29 candidate): two-argument forward differs from torch with identical weights (max diff 0.1245, eval mode). embed_dim and num_heads are required (unlike TransformerEncoderLayer, which takes d_model/nhead — the two layers use different parameter names); dim_feedforward defaults to 2048. The single-argument call runs self-attention only; use .forward(tgt, memory) for the full cross-attention path.
- **PyTorch:** torch.nn.TransformerDecoderLayer — DIVERGES (new deviation, D-29 candidate): with identical weights copied from LiteTorch, the two-argument forward returns different outputs (max diff 0.1245 in eval mode on the reference inputs). The cause is not yet isolated — possibly a different norm placement or cross-attention detail. The single-argument call was not compared apples-to-apples (LiteTorch runs self-attention only; torch needs a memory argument).
- **Ví dụ:**

<!-- ex: id=r7-transformer-decoder-layer-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.TransformerDecoderLayer(embed_dim=8, num_heads=2)
tgt = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
memory = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
y = m.forward(tgt, memory)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.TransformerEncoder`

- **Chữ ký:** `TransformerEncoder(encoder_layer, num_layers, norm=nullptr)`
- **Nguồn:** `bind_nn_recurrent.cpp:117`
- **Mô tả:** Chồng gồm num_layers lớp encoder, tùy chọn có normalization cuối cùng.
- **Tham số:** `encoder_layer` (required), `num_layers` (required), `norm` = nullptr.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** Takes an encoder_layer instance, num_layers, and an optional norm (default None).
- **PyTorch:** torch.nn.TransformerEncoder — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-transformer-encoder-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

layer = lt.nn.TransformerEncoderLayer(d_model=8, nhead=2)
m = lt.nn.TransformerEncoder(layer, num_layers=1)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 2, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.GPTBlock`

- **Chữ ký:** `GPTBlock(d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912, rope_theta=500000.0)`
- **Nguồn:** `bind_nn_base.cpp:258`
- **Mô tả:** Một block decoder GPT (attention GQA + MLP SwiGLU + RMSNorm) với các mặc định của mô hình 2B của LiteTorch.
- **Tham số:** `d_model` = 2560, `n_q_heads` = 20, `n_kv_heads` = 5, `d_ff` = 6912, `rope_theta` = 500000.0f.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** Defaults are the 2B-model sizes (d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912) — pass small values on CPU.
- **PyTorch:** No direct torch.nn equivalent; a GPT decoder block in HF Transformers is the conceptual reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-gpt-block-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GPTBlock(d_model=64, n_q_heads=4, n_kv_heads=2, d_ff=128)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 4, 64], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `nn.GPT2B`

- **Chữ ký:** `GPT2B(vocab_size=50257, d_model=2560, n_layers=28, n_q_heads=20, n_kv_heads=5, d_ff=6912, rope_theta=500000.0)`
- **Nguồn:** `bind_nn_base.cpp:271`
- **Mô tả:** Mô hình GPT 2B tham số của LiteTorch: embedding token, n_layers block GPT, norm cuối và LM head tied.
- **Tham số:** `vocab_size` = 50257, `d_model` = 2560, `n_layers` = 28, `n_q_heads` = 20, `n_kv_heads` = 5, `d_ff` = 6912, `rope_theta` = 500000.0f.
- **Đầu vào:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Đầu ra:** `(seq, batch, embed_dim)`
- **Trường hợp biên:** Defaults are the full 2B sizes (vocab_size=50257, d_model=2560, n_layers=28) — pass small values on CPU. Input must be a 2-D INT64 tensor of token IDs (batch, seq); a 1-D input raises 'Shape mismatch in view'. Methods: forward(ids) -> logits, stage_forward(input, stage), tie_weights(), pipeline_to(d0, d1, split=-1).
- **PyTorch:** No direct torch.nn equivalent; a GPT-style decoder in HF Transformers is the conceptual reference. LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-gpt2-b-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GPT2B(vocab_size=100, d_model=64, n_layers=1, n_q_heads=4, n_kv_heads=2, d_ff=128)
ids = lt.tensor([[1, 2, 3, 4]], dtype=lt.DataType.INT64)
y = m(ids)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
## Hàm mất mát
### `nn.MSELoss`

- **Chữ ký:** `MSELoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:126`
- **Mô tả:** Mean squared error: mean((input - target)^2) theo mặc định.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** reduction defaults to 'mean' ('sum' and 'none' are the other options).
- **PyTorch:** torch.nn.MSELoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-mse-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MSELoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.L1Loss`

- **Chữ ký:** `L1Loss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:219`
- **Mô tả:** Mean absolute error: mean(|input - target|) theo mặc định.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.L1Loss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-l1-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.L1Loss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.SmoothL1Loss`

- **Chữ ký:** `SmoothL1Loss(beta=1.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:332`
- **Mô tả:** Loss Smooth L1 (dạng Huber): bậc hai dưới beta, tuyến tính trên beta. beta mặc định là 1.0.
- **Tham số:** `beta` = 1.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** beta defaults to 1.0; reduction defaults to 'mean'.
- **PyTorch:** torch.nn.SmoothL1Loss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-smooth-l1-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SmoothL1Loss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.HuberLoss`

- **Chữ ký:** `HuberLoss(delta=1.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:342`
- **Mô tả:** Loss Huber: 0.5 * (input - target)^2 dưới delta, delta * (|input - target| - 0.5 * delta) trên delta.
- **Tham số:** `delta` = 1.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** delta defaults to 1.0; reduction defaults to 'mean'.
- **PyTorch:** torch.nn.HuberLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-huber-loss-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.HuberLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.CrossEntropyLoss`

- **Chữ ký:** `CrossEntropyLoss(weight=py::none(, ignore_index=-100, reduction="mean", label_smoothing=0.0)`
- **Nguồn:** `bind_nn_layers.cpp:198`
- **Mô tả:** Cross entropy giữa logits và chỉ mục lớp: -log(softmax(logits)[target]).
- **Tham số:** `weight` = py::none(, `ignore_index` = -100, `reduction` = "mean", `label_smoothing` = 0.0f.
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Torch requires int64 targets, so build the target with dtype INT32 (for example lt.tensor([1, 0], dtype=lt.DataType.INT32)). Note the D-06 interplay: after the D-06 fix .long() returns INT64, which this loss rejects — use INT32 explicitly.
- **PyTorch:** torch.nn.CrossEntropyLoss — equivalent in value, but the target dtype differs (see the edge case).
- **Ví dụ:**

<!-- ex: id=r7-cross-entropy-loss-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.CrossEntropyLoss()
logits = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
target = lt.tensor([1, 0], dtype=lt.DataType.INT32)
y = m.forward(logits, target)
print("loss:", y.to_vector())
```
### `nn.NLLLoss`

- **Chữ ký:** `NLLLoss(weight=py::none(, ignore_index=-100, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:283`
- **Mô tả:** Negative log likelihood: -input[target] với input là log-probabilities.
- **Tham số:** `weight` = py::none(, `ignore_index` = -100, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Pass log-probabilities (for example from .log_softmax(dim)) as input, and an INT32 target tensor.
- **PyTorch:** torch.nn.NLLLoss — equivalent in value, but the target dtype differs (see the edge case).
- **Ví dụ:**

<!-- ex: id=r7-nll-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.NLLLoss()
log_probs = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device).log_softmax(1)
target = lt.tensor([1, 0], dtype=lt.DataType.INT32)
y = m.forward(log_probs, target)
print("loss:", y.to_vector())
```
### `nn.BCELoss`

- **Chữ ký:** `BCELoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:213`
- **Mô tả:** Binary cross entropy giữa xác suất và target nhị phân.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Input must already be probabilities in [0, 1]; for raw logits use BCEWithLogitsLoss instead. reduction defaults to 'mean'.
- **PyTorch:** torch.nn.BCELoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-bce-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BCELoss()
y = m.forward(lt.Tensor.from_vector([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], [2, 3], device), lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.BCEWithLogitsLoss`

- **Chữ ký:** `BCEWithLogitsLoss(weight=py::none(, reduction="mean", pos_weight=py::none()`
- **Nguồn:** `bind_nn_layers.cpp:297`
- **Mô tả:** Binary cross entropy với logits: sigmoid + BCE trong một bước ổn định số học.
- **Tham số:** `weight` = py::none(, `reduction` = "mean", `pos_weight` = py::none(.
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** reduction defaults to 'mean'. pos_weight is not exposed in this binding.
- **PyTorch:** torch.nn.BCEWithLogitsLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-bce-with-logits-loss-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BCEWithLogitsLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.KLDivLoss`

- **Chữ ký:** `KLDivLoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:323`
- **Mô tả:** Divergence Kullback-Leibler giữa log-probabilities (input) và probabilities (target).
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Input must be log-probabilities; target must be probabilities. reduction defaults to 'mean'.
- **PyTorch:** torch.nn.KLDivLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-kl-div-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.KLDivLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device).log_softmax(1), lt.Tensor.from_vector([0.2, 0.5, 0.3, 0.4, 0.3, 0.3], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.HingeEmbeddingLoss`

- **Chữ ký:** `HingeEmbeddingLoss(margin=1.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:133`
- **Mô tả:** Hinge embedding loss cho học tương tự: max(0, margin - input) khi target == 1, input khi target == -1.
- **Tham số:** `margin` = 1.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** DIVERGES from torch (D-26 candidate): LiteTorch returns mean(max(0, margin - target * input)) over all elements (1.1750 on the example); torch returns 0.7000 on the same inputs. margin defaults to 1.0. Target values are 1 or -1.
- **PyTorch:** torch.nn.HingeEmbeddingLoss — DIVERGES (new deviation, D-26 candidate): LiteTorch computes mean(max(0, margin - target * input)) for every element — the classic SVM hinge form — while torch computes input when target == 1 and max(0, margin - input) when target == -1. On the reference example LiteTorch returns 1.1750 where torch returns 0.7000.
- **Ví dụ:**

<!-- ex: id=r7-hinge-embedding-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.HingeEmbeddingLoss()
x = lt.Tensor.from_vector([0.5, 1.5, 2.0, 0.2], [4], device)
target = lt.Tensor.from_vector([1.0, 1.0, -1.0, -1.0], [4], device)
y = m.forward(x, target)
print("loss:", y.to_vector())
```
### `nn.CosineEmbeddingLoss`

- **Chữ ký:** `CosineEmbeddingLoss(margin=0.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:139`
- **Mô tả:** Cosine embedding loss: 1 - cos(x1, x2) khi target == 1, max(0, cos(x1, x2) - margin) khi target == -1.
- **Tham số:** `margin` = 0.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Three inputs (x1, x2, target): call .forward(x1, x2, target) explicitly. margin defaults to 0.0.
- **PyTorch:** torch.nn.CosineEmbeddingLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-cosine-embedding-loss-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.CosineEmbeddingLoss()
x1 = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
x2 = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
target = lt.Tensor.from_vector([1.0, -1.0], [2], device)
y = m.forward(x1, x2, target)
print("loss:", y.to_vector())
```
### `nn.MarginRankingLoss`

- **Chữ ký:** `MarginRankingLoss(margin=0.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:146`
- **Mô tả:** Margin ranking loss: max(0, -target * (input1 - input2) + margin).
- **Tham số:** `margin` = 0.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Three inputs (input1, input2, target): call .forward(x1, x2, target) explicitly. margin defaults to 0.0.
- **PyTorch:** torch.nn.MarginRankingLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-margin-ranking-loss-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MarginRankingLoss()
x1 = lt.Tensor.from_vector([0.8, 0.3], [2], device)
x2 = lt.Tensor.from_vector([0.2, 0.6], [2], device)
target = lt.Tensor.from_vector([1.0, -1.0], [2], device)
y = m.forward(x1, x2, target)
print("loss:", y.to_vector())
```
### `nn.MultiMarginLoss`

- **Chữ ký:** `MultiMarginLoss(p=1, margin=1.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:152`
- **Mô tả:** Multi-class multi-margin loss cho phân loại.
- **Tham số:** `p` = 1, `margin` = 1.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** DIVERGES from torch (D-27 candidate): 1.0000 vs 0.7500 on the single-sample all-zeros case; 0.9667 vs 0.6000 on the two-sample example. p defaults to 1 (an int in this binding — p=2.0 raises TypeError). Target is a 1-D INT32 tensor of class indices (same INT32-target rule as CrossEntropyLoss).
- **PyTorch:** torch.nn.MultiMarginLoss — DIVERGES (new deviation, D-27 candidate): the per-sample normalization differs. On the single-sample all-zeros case LiteTorch returns 1.0000 where torch returns 0.7500 (torch divides the summed hinge terms by the number of classes; LiteTorch divides by classes - 1). On the two-sample example LiteTorch returns 0.9667 where torch returns 0.6000 (p=1); with p=2 LiteTorch returns 1.1700 where torch returns 0.7525.
- **Ví dụ:**

<!-- ex: id=r7-multi-margin-loss-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MultiMarginLoss()
logits = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
target = lt.tensor([1, 0], dtype=lt.DataType.INT32)
y = m.forward(logits, target)
print("loss:", y.to_vector())
```
### `nn.MultiLabelMarginLoss`

- **Chữ ký:** `MultiLabelMarginLoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:159`
- **Mô tả:** Multi-label margin loss cho phân loại đa nhãn.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Target is a 2-D INT32 tensor of label indices (same INT32-target rule as CrossEntropyLoss).
- **PyTorch:** torch.nn.MultiLabelMarginLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-multi-label-margin-loss-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MultiLabelMarginLoss()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
target = lt.Tensor.from_vector([1.0, 2.0, -1.0, -1.0, 0.0, 1.0, -1.0, -1.0], [2, 4], device)
y = m.forward(x, target)
print("loss:", y.to_vector())
```
### `nn.SoftMarginLoss`

- **Chữ ký:** `SoftMarginLoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:164`
- **Mô tả:** Soft margin loss: log(1 + exp(-target * input)) cho phân loại nhị phân với target trong {-1, 1}.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.SoftMarginLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-soft-margin-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SoftMarginLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.MultiLabelSoftMarginLoss`

- **Chữ ký:** `MultiLabelSoftMarginLoss(reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:169`
- **Mô tả:** Multi-label soft margin loss: trung bình theo nhãn của log(1 + exp(-target * input)) với target trong {0, 1}.
- **Tham số:** `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.MultiLabelSoftMarginLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-multi-label-soft-margin-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MultiLabelSoftMarginLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.TripletMarginLoss`

- **Chữ ký:** `TripletMarginLoss(margin=1.0, p=2, eps=1e-6, swap=false, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:174`
- **Mô tả:** Triplet margin loss: max(0, d(anchor, positive) - d(anchor, negative) + margin).
- **Tham số:** `margin` = 1.0f, `p` = 2, `eps` = 1e-6f, `swap` = false, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Three inputs (anchor, positive, negative): call .forward(a, p, n) explicitly. margin defaults to 1.0, p to 2.0.
- **PyTorch:** torch.nn.TripletMarginLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-triplet-margin-loss-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.TripletMarginLoss()
a = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
p = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
n = lt.Tensor.from_vector([0.9, -0.8, 0.7, -0.6, 0.5, -0.4], [2, 3], device)
y = m.forward(a, p, n)
print("loss:", y.to_vector())
```
### `nn.TripletMarginWithDistanceLoss`

- **Chữ ký:** `TripletMarginWithDistanceLoss(margin=1.0, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:180`
- **Mô tả:** Triplet margin loss với hàm khoảng cách tùy chỉnh.
- **Tham số:** `margin` = 1.0f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Three inputs (anchor, positive, negative): call .forward(a, p, n) explicitly. The distance function argument is not exposed; the default distance is used.
- **PyTorch:** torch.nn.TripletMarginWithDistanceLoss — equivalent in intent.
- **Ví dụ:**

<!-- ex: id=r7-triplet-margin-with-distance-loss-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.TripletMarginWithDistanceLoss()
a = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
p = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
n = lt.Tensor.from_vector([0.9, -0.8, 0.7, -0.6, 0.5, -0.4], [2, 3], device)
y = m.forward(a, p, n)
print("loss:", y.to_vector())
```
### `nn.PoissonNLLLoss`

- **Chữ ký:** `PoissonNLLLoss(log_input=true, full=false, eps=1e-8, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:186`
- **Mô tả:** Poisson negative log likelihood: input - target * log(input + eps).
- **Tham số:** `log_input` = true, `full` = false, `eps` = 1e-8f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** log_input defaults to True; full defaults to False. eps defaults to 1e-8.
- **PyTorch:** torch.nn.PoissonNLLLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-poisson-nll-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.PoissonNLLLoss()
y = m.forward(lt.Tensor.from_vector([0.5, 1.0, 1.5, 2.0, 0.8, 1.2], [2, 3], device), lt.Tensor.from_vector([1.0, 2.0, 1.0, 3.0, 1.0, 2.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.GaussianNLLLoss`

- **Chữ ký:** `GaussianNLLLoss(full=false, eps=1e-6, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:191`
- **Mô tả:** Gaussian negative log likelihood: 0.5 * (log(var) + (input - target)^2 / var).
- **Tham số:** `full` = false, `eps` = 1e-6f, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Three inputs (input, target, var): call .forward(input, target, var) explicitly. full defaults to False; eps defaults to 1e-6.
- **PyTorch:** torch.nn.GaussianNLLLoss — equivalent.
- **Ví dụ:**

<!-- ex: id=r7-gaussian-nll-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.GaussianNLLLoss()
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
target = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
var = lt.Tensor.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [2, 3], device)
y = m.forward(x, target, var)
print("loss:", y.to_vector())
```
### `nn.CTCLoss`

- **Chữ ký:** `CTCLoss(blank=0, reduction="mean", zero_infinity=false)`
- **Nguồn:** `bind_nn_layers.cpp:352`
- **Mô tả:** Loss CTC cho căn chỉnh chuỗi (nhận dạng giọng nói, OCR).
- **Tham số:** `blank` = 0, `reduction` = "mean", `zero_infinity` = false.
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** DIVERGES from torch under reduction='mean' (D-28 candidate): LiteTorch 4.0802 vs torch 1.3601 on the example; reduction='sum' matches exactly (8.1604). Four inputs (log_probs, targets, input_lengths, target_lengths): call .forward(lp, targets, in_lens, tgt_lens) explicitly. log_probs must be log-probabilities of shape (T, N, C). blank defaults to 0 — blank must not appear in the targets. Lengths are INT32 tensors on this build (the INT32 rule of CrossEntropyLoss applies here too).
- **PyTorch:** torch.nn.CTCLoss — DIVERGES under reduction='mean' (new deviation, D-28 candidate): LiteTorch returns the mean over the batch of the per-sample losses (4.0802 on the example), while torch divides each per-sample loss by its target length first (1.3601). LiteTorch's reduction='sum' matches torch's 'sum' exactly (8.1604 on the example).
- **Ví dụ:**

<!-- ex: id=r7-ctc-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.CTCLoss()
log_probs = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [5, 2, 4], device).log_softmax(2)
targets = lt.tensor([1, 2, 3, 1, 2, 3], dtype=lt.DataType.INT64)
input_lengths = lt.tensor([5, 5], dtype=lt.DataType.INT64)
target_lengths = lt.tensor([3, 3], dtype=lt.DataType.INT64)
y = m.forward(log_probs, targets, input_lengths, target_lengths)
print("loss:", y.to_vector())
```
### `nn.DPOLoss`

- **Chữ ký:** `DPOLoss(beta=0.1, reference_free=false, reduction="mean")`
- **Nguồn:** `bind_nn_layers.cpp:368`
- **Mô tả:** Loss DPO: -log(sigmoid(beta * (logp_chosen - logp_rejected))) với một số hạng từ mô hình tham chiếu.
- **Tham số:** `beta` = 0.1f, `reference_free` = false, `reduction` = "mean".
- **Đầu vào:** `(prediction, target) — see the example` → **Đầu ra:** `scalar tensor, shape [1]`
- **Trường hợp biên:** Four inputs (chosen, rejected, ref_chosen, ref_rejected): call .forward(c, r, rc, rr) explicitly. beta defaults to 0.1.
- **PyTorch:** No torch.nn equivalent (DPO is a training recipe, not a torch module). LiteTorch-only module.
- **Ví dụ:**

<!-- ex: id=r7-dpo-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.DPOLoss()
chosen = lt.Tensor.from_vector([-0.5, -0.3], [2], device)
rejected = lt.Tensor.from_vector([-1.2, -0.9], [2], device)
ref_chosen = lt.Tensor.from_vector([-0.6, -0.4], [2], device)
ref_rejected = lt.Tensor.from_vector([-1.0, -0.8], [2], device)
y = m.forward(chosen, rejected, ref_chosen, ref_rejected)
print("loss:", y.to_vector())
```
## Phục vụ và giải mã suy đoán
### `nn.GenerateConfig`

- **Chữ ký:** `GenerateConfig()`
- **Nguồn:** `bind_nn_utils.cpp:14`
- **Mô tả:** Cấu hình cho sinh văn bản: max_new_tokens, temperature, top_p, top_k, repetition_penalty và danh sách token dừng.
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** max_new_tokens defaults to 32. temperature defaults to 1.0. stop_tokens defaults to an empty list.
- **PyTorch:** No torch.nn equivalent; HF GenerationConfig is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-generate-config-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

cfg = lt.nn.GenerateConfig()
cfg.max_new_tokens = 2
cfg.temperature = 0.7
print("max_new_tokens:", cfg.max_new_tokens)
print("temperature:", cfg.temperature)
print("do_sample:", cfg.do_sample)
```
### `nn.GuidedDecoder`

- **Chữ ký:** `GuidedDecoder(vocabulary)`
- **Nguồn:** `bind_nn_utils.cpp:30`
- **Mô tả:** Giải mã có ràng buộc: che logits để quá trình sinh tuân theo một ngữ pháp (ví dụ JSON). Khởi tạo bằng một danh sách từ vựng; apply_mask(logits, prefix, target_pattern) là in-place.
- **Tham số:** `vocabulary` (required).
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Constructor takes a vocabulary (list of strings). apply_mask(logits, prefix, target_pattern) takes THREE arguments and modifies logits in place; the target_pattern is a regex-style pattern. The vocab attribute is readable/writable.
- **PyTorch:** No torch.nn equivalent; vLLM guided decoding is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-guided-decoder-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

dec = lt.nn.GuidedDecoder(["a", "b", "c"])
logits = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [1, 4], device)
result = dec.apply_mask(logits, "a", "a")
print("apply_mask returned:", result)
print("logits:", logits.to_vector())
```
### `nn.Request`

- **Chữ ký:** `Request(id, prompt, max_new_tokens=32)`
- **Nguồn:** `bind_nn_utils.cpp:35`
- **Mô tả:** Một yêu cầu phục vụ: id nguyên, danh sách token prompt và ngân sách max_tokens.
- **Tham số:** `id` (required), `prompt` (required), `max_new_tokens` = 32.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** id and prompt are required; max_new_tokens defaults to 32. Generated tokens accumulate in generated_tokens; is_finished and prefill_processed track progress.
- **PyTorch:** No torch.nn equivalent; vLLM Request is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-request-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

req = lt.nn.Request(1, [101, 205, 200, 102], 8)
print("id:", req.id)
print("prompt_tokens:", req.prompt_tokens)
print("max_new_tokens:", req.max_new_tokens)
print("is_finished:", req.is_finished)
```
### `nn.PrefixCache`

- **Chữ ký:** `PrefixCache.get() — no constructor`
- **Nguồn:** `bind_nn_utils.cpp:59`
- **Mô tả:** Cache tiền tố singleton cho các block KV, dùng chung giữa các request có cùng tiền tố prompt. Lấy instance bằng PrefixCache.get().
- **Tham số:** No constructor. Use the static `PrefixCache.get()`.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** No public constructor: use the static PrefixCache.get(). put(prefix, blocks) stores blocks for a token prefix; get_cache(prefix) retrieves them; clear() empties the cache.
- **PyTorch:** No torch.nn equivalent; vLLM prefix caching is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-prefix-cache-1 device=cpu backend=cpu runtime=~0.3s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

cache = lt.nn.PrefixCache.get()
t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
cache.put([1, 2], [t])
blocks = cache.get_cache([1, 2])
print("cached blocks:", len(blocks))
cache.clear()
print("after clear:", len(cache.get_cache([1, 2])))
```
### `nn.ServingScheduler`

- **Chữ ký:** `ServingScheduler(num_blocks, block_size, max_num_seqs, max_num_batched_tokens)`
- **Nguồn:** `bind_nn_utils.cpp:45`
- **Mô tả:** Bộ lập lịch phục vụ batching liên tục: nhận request, cấp phát block KV và chạy các bước prefill + decode. step(model) trả về các request đã hoàn thành trong bước đó.
- **Tham số:** `num_blocks` (required), `block_size` (required), `max_num_seqs` (required), `max_num_batched_tokens` (required).
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** The model must map token IDs to logits: its first layer must be an Embedding (the scheduler feeds a [1, chunk] tensor of token IDs and reads a [1, seq, vocab] output). A bare Linear model fails with 'Matrix inner dimensions must match'.
- **PyTorch:** No torch.nn equivalent; vLLM's scheduler is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-serving-scheduler-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
model = lt.nn.Sequential([
    lt.nn.Embedding(1000, 32),
    lt.nn.Linear(32, 1000, True),
])
model.to(device)
scheduler = lt.nn.ServingScheduler(
    num_blocks=16,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=128
)
scheduler.add_request(lt.nn.Request(1, [101, 205, 200, 102], 8))
scheduler.add_request(lt.nn.Request(2, [101, 759, 102], 8))
done = []
for step in range(20):
    finished = scheduler.step(model)
    done.extend(finished)
    if not scheduler.running_queue and not scheduler.pending_queue:
        break
print("steps:", step + 1, "finished requests:", len(done))
for req in done:
    print("request", req.id, "generated tokens:", req.generated_tokens)
```
### `nn.SpeculativeEngine`

- **Chữ ký:** `SpeculativeEngine(...)`
- **Nguồn:** `bind_nn_utils.cpp:65`
- **Mô tả:** Engine giải mã suy đoán: một mô hình draft nhỏ đề xuất token và một mô hình target lớn hơn xác minh chúng. generate(prompt, max_new_tokens, lookahead=4) chạy một request.
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Constructor takes (draft_model, target_model) — two shared_ptr modules. generate(prompt, max_new_tokens, lookahead=4) returns the generated token list.
- **PyTorch:** No torch.nn equivalent; vLLM speculative decoding is the conceptual reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-speculative-engine-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
draft = lt.nn.Sequential([lt.nn.Embedding(100, 32), lt.nn.Linear(32, 100, True)])
target = lt.nn.Sequential([lt.nn.Embedding(100, 32), lt.nn.Linear(32, 100, True)])
engine = lt.nn.SpeculativeEngine(draft, target)
tokens = engine.generate([1, 2, 3], 2)
print("generated tokens:", tokens)
```
### `nn.MedusaEngine`

- **Chữ ký:** `MedusaEngine(...)`
- **Nguồn:** `bind_nn_utils.cpp:79`
- **Mô tả:** Engine giải mã suy đoán Medusa: các MedusaHead đề xuất thêm token cho mỗi vị trí để xác minh song song. Khởi tạo bằng (base_model, medusa_heads).
- **Tham số:** None.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Constructor takes (base_model, medusa_heads) where medusa_heads is a list of MedusaHead instances. generate(prompt, max_new_tokens, lookahead=2) runs one request.
- **PyTorch:** No torch.nn equivalent; the Medusa paper (Cai et al., 2024) is the reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-medusa-engine-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
base = lt.nn.Sequential([lt.nn.Embedding(100, 32), lt.nn.Linear(32, 100, True)])
heads = [lt.nn.MedusaHead(100, 100)]
engine = lt.nn.MedusaEngine(base, heads)
tokens = engine.generate([1, 2, 3], 2)
print("generated tokens:", tokens)
```
### `nn.MedusaHead`

- **Chữ ký:** `MedusaHead(int, int)`
- **Nguồn:** `bind_nn_utils.cpp:71`
- **Mô tả:** Head dự đoán Medusa: ánh xạ hidden state thành các dự đoán token thêm, một head cho mỗi vị trí suy đoán.
- **Tham số:** Positional `(int, int)` — (in_features, out_features); no keyword argument names are bound.
- **Đầu vào:** `varies — see the example` → **Đầu ra:** `varies — see the example`
- **Trường hợp biên:** Takes two positional ints (in_features, out_features); no keyword argument names are bound.
- **PyTorch:** No torch.nn equivalent; the Medusa paper (Cai et al., 2024) is the reference. LiteTorch-only class.
- **Ví dụ:**

<!-- ex: id=r7-medusa-head-1 device=cpu backend=cpu runtime=~0.8s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MedusaHead(8, 4)
x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 8], device)
y = m(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```

### `nn.generate(model, prompt_tokens, config=GenerateConfig())`

- **Chữ ký:** `generate(model, prompt_tokens, config=GenerateConfig())`
- **Nguồn:** `bind_nn_utils.cpp:27`
- **Mô tả:** Chạy sinh văn bản greedy với một mô hình ánh xạ ID token thành logits (Embedding rồi Linear). Trả về danh sách token đã sinh, không bao gồm prompt.
- **Tham số:** `model` (bắt buộc), `prompt_tokens` (bắt buộc, danh sách int), `config` = `GenerateConfig()` (xem GenerateConfig ở trên; `max_new_tokens` mặc định là 32).
- **Đầu vào:** `(model, [int, ...])` → **Đầu ra:** `[int, ...]` token đã sinh.
- **Trường hợp biên:** Mô hình phải ánh xạ ID token thành logits: lớp đầu tiên phải là Embedding. Một mô hình Linear trần sẽ thất bại. Quá trình sinh là greedy trừ khi cấu hình bật sampling.
- **PyTorch:** Không có tương đương trong torch.nn; `generate()` của HF là tham chiếu khái niệm. Hàm chỉ có ở LiteTorch.
- **Ví dụ:**

<!-- ex: id=r7-generate-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
model = lt.nn.Sequential([
    lt.nn.Embedding(100, 32),
    lt.nn.Linear(32, 100, True),
])
model.to(device)
cfg = lt.nn.GenerateConfig()
cfg.max_new_tokens = 2
tokens = lt.nn.generate(model, [1, 2, 3], cfg)
print("generated tokens:", tokens)
```
