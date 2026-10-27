# nn.functional, nn.init, nn.utils — Tài liệu tham khảo API

Các hàm dưới `litetorch.nn.functional`, `litetorch.nn.init` và `litetorch.nn.utils`. Các chữ ký lấy từ các binding pybind11 (`src/bindings/bind_nn_utils.cpp`); binding là nguồn chân lý. Mỗi ví dụ đều tự chứa, chạy trên CPU và mang header xác minh (xem kế hoạch xác minh, `plan/verification.md` §4).

Lưu ý: `nn.utils` có năm hàm. `clip_value` KHÔNG được bind ở đây — chỉ có `clip_grad_value_`; hàm cắt theo norm, `clip_grad_norm_`, nằm dưới `Ops`.
## nn.functional
### `F.dropout`

- **Chữ ký:** `dropout(x, p=0.5, training=True)`
- **Nguồn:** `bind_nn_utils.cpp:171`
- **Mô tả:** Dropout dạng hàm: đặt ngẫu nhiên mỗi phần tử về 0 với xác suất p khi training là True; là identity trong các trường hợp còn lại. Đầu ra được nhân 1/(1-p) khi huấn luyện.
- **Trường hợp biên:** training defaults to True, so a bare call is stochastic: pass training=False (or seed with lt.manual_seed()) for a deterministic pass. The module form is nn.Dropout.
- **PyTorch:** torch.nn.functional.dropout — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-dropout-1 device=cpu backend=cpu runtime=~0.6s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = lt.nn.functional.dropout(x, p=0.5, training=False)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.linear`

- **Chữ ký:** `linear(x, weight, bias=None)`
- **Nguồn:** `bind_nn_utils.cpp:173`
- **Mô tả:** Linear dạng hàm: y = x @ weight.T + bias.
- **Trường hợp biên:** bias defaults to None (no bias). weight shape is (out_features, in_features).
- **PyTorch:** torch.nn.functional.linear — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-linear-1 device=cpu backend=cpu runtime=~1.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
weight = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4], [3, 4], device)
bias = lt.Tensor.from_vector([0.1, 0.2, 0.3], [3], device)
y = lt.nn.functional.linear(x, weight, bias)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.pad`

- **Chữ ký:** `pad(x, pad, mode='constant', value=0.0)`
- **Nguồn:** `bind_nn_utils.cpp:175`
- **Mô tả:** Đệm một tensor. pad là danh sách các cặp (trước, sau) cho mỗi chiều, bắt đầu từ chiều cuối cùng.
- **Trường hợp biên:** mode defaults to 'constant' with value 0.0. The pad list is ordered from the last dimension backward: [1, 1, 0, 0] pads the last dimension by 1 on each side and leaves the next dimension unchanged.
- **PyTorch:** torch.nn.functional.pad — equivalent for the constant mode.
- **Ví dụ:**

<!-- ex: id=r8-functional-pad-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
y = lt.nn.functional.pad(x, [1, 1, 0, 0])
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.interpolate`

- **Chữ ký:** `interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False)`
- **Nguồn:** `bind_nn_utils.cpp:177`
- **Mô tả:** Thay đổi kích thước một tensor. Cho size (danh sách kích thước đầu ra) hoặc scale_factor (float hoặc danh sách).
- **Trường hợp biên:** size and scale_factor are both optional but one must be given. mode defaults to 'nearest'; align_corners defaults to False.
- **PyTorch:** torch.nn.functional.interpolate — equivalent for the nearest mode.
- **Ví dụ:**

<!-- ex: id=r8-functional-interpolate-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [1, 1, 4, 4], device)
y = lt.nn.functional.interpolate(x, size=[8, 8])
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.nll_loss`

- **Chữ ký:** `nll_loss(input, target, weight=None, reduction='mean', ignore_index=-100)`
- **Nguồn:** `bind_nn_utils.cpp:201`
- **Mô tả:** Negative log likelihood: -input[target] với input là log-probabilities.
- **Trường hợp biên:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Pass log-probabilities (for example from .log_softmax(dim)) as input, and an INT32 target tensor.
- **PyTorch:** torch.nn.functional.nll_loss — equivalent in value, but the target dtype differs (see the edge case).
- **Ví dụ:**

<!-- ex: id=r8-functional-nll_loss-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

log_probs = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device).log_softmax(1)
target = lt.tensor([1, 0], dtype=lt.DataType.INT32)
y = lt.nn.functional.nll_loss(log_probs, target)
print("loss:", y.to_vector())
```
### `F.bce_with_logits`

- **Chữ ký:** `bce_with_logits(input, target, reduction='mean')`
- **Nguồn:** `bind_nn_utils.cpp:204`
- **Mô tả:** Binary cross entropy với logits: sigmoid + BCE trong một bước ổn định số học.
- **Trường hợp biên:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.functional.binary_cross_entropy_with_logits — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-bce_with_logits-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device)
target = lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3], device)
y = lt.nn.functional.bce_with_logits(x, target)
print("loss:", y.to_vector())
```
### `F.normalize`

- **Chữ ký:** `normalize(x, p=2.0, dim=1, eps=1e-12)`
- **Nguồn:** `bind_nn_utils.cpp:206`
- **Mô tả:** Chuẩn hóa L_p theo dim: x / max(||x||_p, eps).
- **Trường hợp biên:** p defaults to 2.0, dim to 1, eps to 1e-12.
- **PyTorch:** torch.nn.functional.normalize — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-normalize-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = lt.nn.functional.normalize(x, p=2.0, dim=1)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.glu`

- **Chữ ký:** `glu(x, dim=-1)`
- **Nguồn:** `bind_nn_utils.cpp:208`
- **Mô tả:** Gated linear unit: tách đầu vào làm đôi theo dim và trả về a * sigmoid(b).
- **Trường hợp biên:** dim defaults to -1. The size of dim must be even. The module form is nn.GLU.
- **PyTorch:** torch.nn.functional.glu — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-glu-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4], device)
y = lt.nn.functional.glu(x, dim=-1)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.elu`

- **Chữ ký:** `elu(x, alpha=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:210`
- **Mô tả:** Exponential linear unit: x khi x > 0, alpha * (exp(x) - 1) trong các trường hợp còn lại.
- **Trường hợp biên:** alpha defaults to 1.0. The module form is nn.ELU.
- **PyTorch:** torch.nn.functional.elu — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-functional-elu-1 device=cpu backend=cpu runtime=~0.2s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

x = lt.Tensor.from_vector([-2.0, -1.0, 0.0, 1.0, 2.0], [5], device)
y = lt.nn.functional.elu(x)
print("output shape:", list(y.shape))
print("output:", y.to_vector()[:8])
```
### `F.dpo_loss`

- **Chữ ký:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps=None, ref_rejected_logps=None, beta=0.1, reference_free=False, reduction='mean')`
- **Nguồn:** `bind_nn_utils.cpp:214`
- **Mô tả:** Loss DPO: -log(sigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))).
- **Trường hợp biên:** ref_chosen_logps and ref_rejected_logps are required unless reference_free=True — omitting them raises ValueError. beta defaults to 0.1.
- **PyTorch:** No torch.nn.functional equivalent (DPO is a training recipe, not a torch function). LiteTorch-only function. The module form is nn.DPOLoss.
- **Ví dụ:**

<!-- ex: id=r8-functional-dpo_loss-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

chosen = lt.Tensor.from_vector([-0.5, -0.3], [2], device)
rejected = lt.Tensor.from_vector([-1.2, -0.9], [2], device)
ref_chosen = lt.Tensor.from_vector([-0.6, -0.4], [2], device)
ref_rejected = lt.Tensor.from_vector([-1.0, -0.8], [2], device)
y = lt.nn.functional.dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
print("loss:", y.to_vector())
```
## nn.init
### `init.kaiming_uniform_`

- **Chữ ký:** `kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='relu')`
- **Nguồn:** `bind_nn_utils.cpp:86`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ U(-bound, bound), với bound được tính từ fan-in (hoặc fan-out) theo công thức Kaiming/He. Trả về tensor.
- **Trường hợp biên:** a defaults to 0.0 (the negative slope of the rectifier), mode to 'fan_in', nonlinearity to 'relu'. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.kaiming_uniform_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-kaiming_uniform_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.kaiming_uniform_(t)
print("values:", t.to_vector()[:4])
```
### `init.kaiming_normal_`

- **Chữ ký:** `kaiming_normal_(tensor, a=0.0, mode='fan_in', nonlinearity='relu')`
- **Nguồn:** `bind_nn_utils.cpp:87`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ N(0, std), với std được tính từ fan-in (hoặc fan-out) theo công thức Kaiming/He. Trả về tensor.
- **Trường hợp biên:** a defaults to 0.0, mode to 'fan_in', nonlinearity to 'relu'. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.kaiming_normal_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-kaiming_normal_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.kaiming_normal_(t)
print("values:", t.to_vector()[:4])
```
### `init.xavier_uniform_`

- **Chữ ký:** `xavier_uniform_(tensor, gain=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:88`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ U(-a, a), với a = gain * sqrt(6 / (fan_in + fan_out)). Trả về tensor.
- **Trường hợp biên:** gain defaults to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.xavier_uniform_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-xavier_uniform_-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.xavier_uniform_(t)
print("values:", t.to_vector()[:4])
```
### `init.xavier_normal_`

- **Chữ ký:** `xavier_normal_(tensor, gain=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:89`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ N(0, std), với std = gain * sqrt(2 / (fan_in + fan_out)). Trả về tensor.
- **Trường hợp biên:** gain defaults to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.xavier_normal_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-xavier_normal_-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.xavier_normal_(t)
print("values:", t.to_vector()[:4])
```
### `init.normal_`

- **Chữ ký:** `normal_(tensor, mean=0.0, std=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:90`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ N(mean, std). Trả về tensor.
- **Trường hợp biên:** mean defaults to 0.0, std to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.normal_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-normal_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.normal_(t, mean=0.0, std=1.0)
print("values:", t.to_vector()[:4])
```
### `init.uniform_`

- **Chữ ký:** `uniform_(tensor, a=0.0, b=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:91`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ U(a, b). Trả về tensor.
- **Trường hợp biên:** a defaults to 0.0, b to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.uniform_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-uniform_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.uniform_(t, a=0.0, b=1.0)
print("values:", t.to_vector()[:4])
```
### `init.zeros_`

- **Chữ ký:** `zeros_(tensor)`
- **Nguồn:** `bind_nn_utils.cpp:92`
- **Mô tả:** Điền tại chỗ một tensor bằng số 0. Trả về tensor.
- **Trường hợp biên:** In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.zeros_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-zeros_-1 device=cpu backend=cpu runtime=~0.3s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.zeros_(t)
print("values:", t.to_vector())
```
### `init.ones_`

- **Chữ ký:** `ones_(tensor)`
- **Nguồn:** `bind_nn_utils.cpp:93`
- **Mô tả:** Điền tại chỗ một tensor bằng số 1. Trả về tensor.
- **Trường hợp biên:** In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.ones_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-ones_-1 device=cpu backend=cpu runtime=~0.4s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.ones_(t)
print("values:", t.to_vector())
```
### `init.fill_`

- **Chữ ký:** `fill_(tensor, value)`
- **Nguồn:** `bind_nn_utils.cpp:94`
- **Mô tả:** Điền tại chỗ một tensor bằng một hằng số. Trả về tensor.
- **Trường hợp biên:** value is required (no default). In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.constant_ is the torch counterpart (LiteTorch names it fill_).
- **Ví dụ:**

<!-- ex: id=r8-init-fill_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.fill_(t, 3.5)
print("values:", t.to_vector())
```
### `init.trunc_normal_`

- **Chữ ký:** `trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)`
- **Nguồn:** `bind_nn_utils.cpp:96`
- **Mô tả:** Điền tại chỗ một tensor bằng các giá trị lấy mẫu từ phân phối chuẩn cắt cụt N(mean, std), giới hạn trong [a, b]. Trả về tensor.
- **Trường hợp biên:** mean defaults to 0.0, std to 1.0, a to -2.0, b to 2.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.trunc_normal_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-trunc_normal_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.trunc_normal_(t)
print("values:", t.to_vector()[:4])
```
### `init.orthogonal_`

- **Chữ ký:** `orthogonal_(tensor, gain=1.0)`
- **Nguồn:** `bind_nn_utils.cpp:98`
- **Mô tả:** Điền tại chỗ một tensor 2 chiều bằng một ma trận trực giao (tùy chọn có hệ số scale). Trả về tensor.
- **Trường hợp biên:** gain defaults to 1.0. The tensor must be 2-D with at least as many rows as columns (or vice versa). In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.orthogonal_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-init-orthogonal_-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

lt.manual_seed(0)
t = lt.Tensor.from_vector([0.0] * 16, [4, 4], device)
lt.nn.init.orthogonal_(t)
print("values:", t.to_vector()[:4])
```
## nn.utils
### `utils.clip_grad_value_`

- **Chữ ký:** `clip_grad_value_(parameters, clip_value)`
- **Nguồn:** `bind_nn_utils.cpp:7`
- **Mô tả:** Cắt gradient của một danh sách parameter tại chỗ về [-clip_value, clip_value].
- **Trường hợp biên:** parameters is a list of tensors; clip_value is required. Operates on .grad — parameters without gradients are skipped. Note: clip_grad_norm_ is NOT under nn.utils; it lives under Ops.
- **PyTorch:** torch.nn.utils.clip_grad_value_ — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-utils-clip_grad_value_-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Linear(4, 3)
params = list(m.parameters())
lt.nn.utils.clip_grad_value_(params, 1.0)
print("clipped", len(params), "parameters")
```
### `utils.parameters_to_vector`

- **Chữ ký:** `parameters_to_vector(parameters)`
- **Nguồn:** `bind_nn_utils.cpp:8`
- **Mô tả:** Làm phẳng một danh sách parameter thành một vector 1 chiều duy nhất.
- **Trường hợp biên:** parameters is a list of tensors. The vector length is the sum of the parameter sizes.
- **PyTorch:** torch.nn.utils.parameters_to_vector — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-utils-parameters_to_vector-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Linear(4, 3)
vec = lt.nn.utils.parameters_to_vector(list(m.parameters()))
print("vector shape:", list(vec.shape))
print("values:", vec.to_vector()[:4])
```
### `utils.vector_to_parameters`

- **Chữ ký:** `vector_to_parameters(vec, parameters)`
- **Nguồn:** `bind_nn_utils.cpp:9`
- **Mô tả:** Sao chép một vector phẳng 1 chiều trở lại vào một danh sách parameter (nghịch đảo của parameters_to_vector).
- **Trường hợp biên:** The vector length must equal the total parameter size.
- **PyTorch:** torch.nn.utils.vector_to_parameters — equivalent.
- **Ví dụ:**

<!-- ex: id=r8-utils-vector_to_parameters-1 device=cpu backend=cpu runtime=~0.6s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.Linear(4, 3)
params = list(m.parameters())
vec = lt.nn.utils.parameters_to_vector(params)
lt.nn.utils.vector_to_parameters(vec, params)
print("round-trip ok, vector shape:", list(vec.shape))
```
### `utils.weight_norm`

- **Chữ ký:** `weight_norm(v, g, dim=0)`
- **Nguồn:** `bind_nn_utils.cpp:10`
- **Mô tả:** Weight normalization: tham số hóa lại một weight thành g * v / ||v|| theo dim. Trả về tensor weight đã chuẩn hóa.
- **Trường hợp biên:** v is the direction tensor, g the scale tensor; dim defaults to 0.
- **PyTorch:** torch.nn.utils.weight_norm is a module wrapper in torch; LiteTorch exposes the raw reparameterization as a function.
- **Ví dụ:**

<!-- ex: id=r8-utils-weight_norm-1 device=cpu backend=cpu runtime=~0.5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

v = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4], [3, 4], device)
g = lt.Tensor.from_vector([1.0, 1.0, 1.0], [3, 1], device)
w = lt.nn.utils.weight_norm(v, g)
print("weight shape:", list(w.shape))
print("values:", w.to_vector()[:4])
```
### `utils.spectral_norm`

- **Chữ ký:** `spectral_norm(weight, n_power_iterations=1)`
- **Nguồn:** `bind_nn_utils.cpp:11`
- **Mô tả:** Spectral normalization: thay đổi tỉ lệ một weight theo giá trị kỳ dị lớn nhất (ước lượng bằng power iteration). Trả về tensor weight đã chuẩn hóa.
- **Trường hợp biên:** n_power_iterations defaults to 1.
- **PyTorch:** torch.nn.utils.spectral_norm is a module wrapper in torch; LiteTorch exposes the raw normalization as a function.
- **Ví dụ:**

<!-- ex: id=r8-utils-spectral_norm-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

w = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4], [3, 4], device)
wn = lt.nn.utils.spectral_norm(w)
print("weight shape:", list(wn.shape))
print("values:", wn.to_vector()[:4])
```
