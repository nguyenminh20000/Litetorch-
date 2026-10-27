# Optimizer, LR Scheduler và AMP — Tài liệu tham khảo API

`litetorch.optim` chứa tất cả optimizer, learning-rate scheduler và `GradScaler` cơ sở; `litetorch.amp` chứa `GradScaler` của AMP, `AutocastGuard` và `autocast`.

- **Nguồn xác thực:** `src/bindings/bind_optim.cpp` (optimizer, scheduler, `optim.GradScaler`), `src/bindings/bind_amp.cpp` (AMP), `include/litetorch/optim.h`, `include/litetorch/amp.h`.
- **Cây mã:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline của tài liệu; B5 xác minh lại tại head). Các ví dụ chạy trên extension D3 của Core tại commit 88a4ba1.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Tính tương đương với PyTorch là FP32 1e-4 trừ khi header ghi khác. Optimizer nhận một danh sách Python các tensor tham số; gradient phải được đặt vào `p.grad` trước `step()` (hoặc sinh ra bởi `loss.backward()`).
- **Điểm khác biệt liên quan:** NAdam lệch khỏi PyTorch (đã đo, xem mục `NAdam` — nguyên nhân gốc trong `src/optim/nadam.cpp`: `mu_product` là tỉ số `mu_next/mu`, không phải tích lũy thừa như PyTorch giữ, và `step_size` chia cho `bias_correction1`, điều PyTorch không làm với NAdam). `save_optimizer_state` / `load_optimizer_state` chỉ hỗ trợ Adam, AdamW, SGD, RMSprop và ném lỗi với mọi optimizer khác. `AdamW8bit` / `AdamWFP8` / `LinearWarmup` không có đối tác trong PyTorch. Xem `plan/differences.md`.

## `Optimizer` (lớp cơ sở)

- **Chữ ký:** `Optimizer(params) -> Optimizer`
- **Nguồn:** `src/bindings/bind_optim.cpp:6`
- **Tham số:** `params` — danh sách các tensor tham số.
- **Phương thức:** `step()` — một bước cập nhật, bỏ qua các tham số không có `.grad`; `zero_grad()` — xóa mọi gradient của tham số; `get_lr()` — tốc độ học hiện tại của nhóm tham số đầu tiên; `set_lr(new_lr)` — đặt tốc độ học cho mọi nhóm tham số.
- **Trường hợp biên:** `step()` trên optimizer mà các tham số không có gradient là no-op. Không có `state_dict()` / `load_state_dict()` trên lớp cơ sở — chỉ `Adagrad` và `NAdam` có (xem dưới). `param_groups` chỉ có trên `Adagrad` và `NAdam`.
- **PyTorch:** `torch.optim.Optimizer` — API cốt lõi tương đương, trừ việc PyTorch có `state_dict()` / `load_state_dict()` / `param_groups` trên mọi optimizer.

<!-- ex: id=r9-base device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.SGD([w], lr=0.1)
opt.step()
print(w.to_vector(), opt.get_lr())
opt.set_lr(0.05)
opt.zero_grad()
print(w.grad, opt.get_lr())
```

## `SGD`

- **Chữ ký:** `SGD(params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False) -> SGD`
- **Nguồn:** `src/bindings/bind_optim.cpp:12-13`
- **Tham số:** `lr` — tốc độ học, mặc định 0.01. `momentum` — hệ số momentum, mặc định 0.0, khoảng [0, 1). `dampening` — hệ số giảm chấn cho momentum, mặc định 0.0. `weight_decay` — phạt L2, mặc định 0.0. `nesterov` — momentum Nesterov, mặc định False (không có tác dụng khi `momentum == 0`).
- **Công thức cập nhật:** `v = momentum * v + (1 - dampening) * g`; `p -= lr * (v + weight_decay * p)`; với Nesterov, gradient dùng là `g + momentum * v`.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** `nesterov=True` với `momentum=0` tương đương SGD thường. Tham số không có `.grad` bị bỏ qua.
- **PyTorch:** `torch.optim.SGD` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-sgd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.SGD([w], lr=0.1, momentum=0.9)
opt.step()
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt.step()
print(w.to_vector())
```

## `Adam`

- **Chữ ký:** `Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> Adam`
- **Nguồn:** `src/bindings/bind_optim.cpp:24-25`
- **Tham số:** `lr` — mặc định 0.001. `beta1` — hệ số suy giảm của moment bậc nhất, mặc định 0.9. `beta2` — hệ số suy giảm của moment bậc hai, mặc định 0.999. `eps` — hằng số ổn định mẫu số, mặc định 1e-8. `weight_decay` — phạt L2 cộng vào gradient (không tách rời), mặc định 0.0. Lưu ý hai beta là hai đối số từ khóa riêng biệt, không phải tuple `betas` như trong PyTorch.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** không có tùy chọn `amsgrad`. Không có `state_dict()` — dùng `lt.save_optimizer_state` / `lt.load_optimizer_state` (Adam được hỗ trợ).
- **PyTorch:** `torch.optim.Adam` — giống hệt về số (đã xác minh 1e-7); phía torch truyền `betas=(beta1, beta2)`.

<!-- ex: id=r9-adam device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.Adam([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `AdamW`

- **Chữ ký:** `AdamW(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, offload_to_cpu=False) -> AdamW`
- **Nguồn:** `src/bindings/bind_optim.cpp:36-37`
- **Tham số:** `lr` — mặc định 0.001. `beta1` — mặc định 0.9. `beta2` — mặc định 0.999. `eps` — mặc định 1e-8. `weight_decay` — weight decay tách rời, mặc định 0.01 (khác với 0.0 của Adam). `offload_to_cpu` — giữ trạng thái optimizer trên CPU, mặc định False.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ. Có các bộ đếm tĩnh `fused_launch_count`, `single_launch_count`, `reset_launch_counts` (cho đường fused CUDA launch).
- **Trường hợp biên:** weight decay tách rời nghĩa là `p -= lr * weight_decay * p` diễn ra độc lập với gradient. Không có tùy chọn `amsgrad`.
- **PyTorch:** `torch.optim.AdamW` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-adamw device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.AdamW([w], lr=0.01, weight_decay=0.01)
opt.step()
print(w.to_vector())
```

## `AdamW8bit`

- **Chữ ký:** `AdamW8bit(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> AdamW8bit`
- **Nguồn:** `src/bindings/bind_optim.cpp:51-52`
- **Tham số:** mặc định giống `AdamW`, trừ `weight_decay` mặc định 0.0.
- **Hành vi:** AdamW mà moment bậc nhất và bậc hai được lưu ở dạng lượng tử hóa 8-bit theo block (kích thước block 2048, một tỉ lệ FP32 cho mỗi block). Bộ nhớ cho mỗi tham số khoảng 2 byte thay vì 8 cho hai moment. Bước cập nhật được tính từ moment đã giải lượng tử hóa, nên kết quả gần AdamW nhưng không giống từng bit.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** sai số lượng tử hóa tăng khi moment của các tham số trải rộng trên dải động lớn trong một block 2048 phần tử. Không hỗ trợ `save_optimizer_state` (ném `Unsupported optimizer type`).
- **PyTorch:** không có đối tác trong `torch.optim` (AdamW 8-bit chỉ có trong thư viện bên thứ ba như bitsandbytes). So sánh với `torch.optim.AdamW` với dung sai lỏng.

<!-- ex: id=r9-adamw8bit device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.AdamW8bit([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `AdamWFP8`

- **Chữ ký:** `AdamWFP8(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> AdamWFP8`
- **Nguồn:** `src/bindings/bind_optim.cpp:60-61`
- **Tham số:** mặc định giống `AdamW8bit`.
- **Hành vi:** AdamW mà các moment được lưu ở FP8 với lịch sử amax để co tỉ lệ. Cùng mức bộ nhớ như `AdamW8bit` (2 byte cho mỗi tham số cho hai moment) với lược đồ lượng tử hóa khác.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** FP8 có dải động hẹp hơn định dạng block 8-bit của `AdamW8bit`; các cú nhảy gradient lớn có thể tràn dải FP8 và bão hòa moment. Không hỗ trợ `save_optimizer_state`.
- **PyTorch:** không có đối tác trong `torch.optim`.

<!-- ex: id=r9-adamwfp8 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.AdamWFP8([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `RMSprop`

- **Chữ ký:** `RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0) -> RMSprop`
- **Nguồn:** `src/bindings/bind_optim.cpp:69-70`
- **Tham số:** `lr` — mặc định 0.01. `alpha` — hằng số làm mượt cho trung bình bình phương gradient, mặc định 0.99. `eps` — mặc định 1e-8. `weight_decay` — phạt L2, mặc định 0.0.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** không có tùy chọn `momentum` hay `centered` (cả hai đều có trong PyTorch). Không có `state_dict()`.
- **PyTorch:** `torch.optim.RMSprop` với `momentum=0, centered=False` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-rmsprop device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.RMSprop([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `Adadelta`

- **Chữ ký:** `Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0) -> Adadelta`
- **Nguồn:** `src/bindings/bind_optim.cpp:78-79`
- **Tham số:** `lr` — mặc định 1.0 (Adadelta được thiết kế để chạy ở lr 1.0). `rho` — tốc độ suy giảm cho trung bình bình phương gradient và bình phương bước cập nhật, mặc định 0.9. `eps` — mặc định 1e-6. `weight_decay` — mặc định 0.0.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** bước cập nhật không phụ thuộc tỉ lệ gradient (tỉ số của hai đại lượng RMS đang chạy), nên bước đầu tiên nhỏ và kích thước bước hiệu dụng do `lr * RMS(update) / RMS(grad)` quyết định.
- **PyTorch:** `torch.optim.Adadelta` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-adadelta device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.Adadelta([w], lr=1.0)
opt.step()
print(w.to_vector())
```

## `Adamax`

- **Chữ ký:** `Adamax(params, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> Adamax`
- **Nguồn:** `src/bindings/bind_optim.cpp:88-89`
- **Tham số:** `lr` — mặc định 0.002. `beta1` — mặc định 0.9. `beta2` — mặc định 0.999. `eps` — mặc định 1e-8. `weight_decay` — mặc định 0.0.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** moment bậc hai là chuẩn vô cực (`max(beta2 * v, |g|)`), nên mẫu số không bao giờ nhỏ hơn gradient lớn nhất đã thấy. Không hỗ trợ `save_optimizer_state`.
- **PyTorch:** `torch.optim.Adamax` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-adamax device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.Adamax([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `Adagrad`

- **Chữ ký:** `Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10) -> Adagrad`
- **Nguồn:** `src/bindings/bind_optim.cpp:182-183`
- **Tham số:** `lr` — mặc định 0.01. `lr_decay` — suy giảm tốc độ học, mặc định 0.0. `weight_decay` — mặc định 0.0. `eps` — mặc định 1e-10.
- **Phương thức:** `step()`, `zero_grad()`, `get_lr()`, `set_lr(new_lr)`, cộng `add_param_group(params, lr, weight_decay=0.0)`, `param_groups()` (trả về danh sách dict), `state_dict()` và `load_state_dict(state_dict)` — một trong hai optimizer duy nhất có các phương thức này (optimizer còn lại là `NAdam`).
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** tổng bình phương gradient tích lũy chỉ tăng, nên kích thước bước hiệu dụng giảm dần theo quá trình huấn luyện; dùng `lr_decay` để giảm tốc độ học thêm. Không hỗ trợ `save_optimizer_state` — dùng `state_dict()` / `load_state_dict()` thay thế.
- **PyTorch:** `torch.optim.Adagrad` — giống hệt về số (đã xác minh 1e-7).

<!-- ex: id=r9-adagrad device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.Adagrad([w], lr=0.01)
opt.step()
print(w.to_vector(), opt.state_dict().keys())
```

## `NAdam`

- **Chữ ký:** `NAdam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=0.004) -> NAdam`
- **Nguồn:** `src/bindings/bind_optim.cpp:261-262`, cài đặt `src/optim/nadam.cpp`
- **Tham số:** `lr` — mặc định 0.002. `betas` — tuple `(beta1, beta2)`, mặc định (0.9, 0.999). `eps` — mặc định 1e-8. `weight_decay` — mặc định 0.0. `momentum_decay` — suy giảm của lịch momentum Nesterov, mặc định 0.004.
- **Phương thức:** `step()`, `zero_grad()`, `get_lr()`, `set_lr(new_lr)`, cộng `add_param_group(params, lr, weight_decay=0.0)`, `param_groups()`, `state_dict()`, `load_state_dict(state_dict)`.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** `step()` trên tensor GPU ném lỗi `NAdam: GPU step not implemented, use CPU tensors`. Tham số phải liên tục (contiguous). Không hỗ trợ `save_optimizer_state`.
- **PyTorch:** `torch.optim.NAdam` — **LỆCH**. Đã đo trên CPU, `w=[1.0]`, `grad=0.5`, `lr=0.01`, ba bước: LiteTorch `[0.9350, 0.8961, 0.8658]` so với PyTorch `[0.9894, 0.9816, 0.9743]`; một bước ở `lr=0.1` cho LiteTorch `0.3501` so với PyTorch `0.8944`. Nguyên nhân gốc trong `src/optim/nadam.cpp`: LiteTorch tính `mu_product = mu_next / mu` (một tỉ số) và `step_size = lr / bias_correction1`; PyTorch giữ `mu_product` là tích lũy thừa của mọi giá trị `mu` từ trước đến nay (bắt đầu từ 1) và áp dụng bước cập nhật dưới dạng hai số hạng, `-lr * (1 - mu) / (1 - mu_product) * grad / denom` và `-lr * mu_next / (1 - mu_product * mu_next) * exp_avg / denom`, không chia `lr` cho `bias_correction1`. Bước đầu tiên của LiteTorch vì vậy lớn hơn khoảng 6.5 lần. Không chuyển các run huấn luyện NAdam từ PyTorch sang `NAdam` của LiteTorch cho đến khi lỗi này được sửa; dùng `Adam` thay thế. Đã báo cho Core dưới dạng defect (xem `plan/differences.md`).

<!-- ex: id=r9-nadam device=cpu backend=cpu runtime=~1s torch-parity=parity-fail-nadam status=parity-fail-nadam -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.NAdam([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## `ZeRO3Optimizer`

- **Chữ ký:** `ZeRO3Optimizer(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> ZeRO3Optimizer`
- **Nguồn:** `src/bindings/bind_optim.cpp:97-98`, `include/litetorch/zero3_optimizer.h`
- **Tham số:** mặc định giống `Adam` (`lr` 0.001, `beta1` 0.9, `beta2` 0.999, `eps` 1e-8, `weight_decay` 0.0).
- **Hành vi:** optimizer dựa trên AdamW cho huấn luyện sharded ZeRO-3. Nó giữ một bản sao master FP32 của mỗi tham số và chạy bước cập nhật AdamW trên các trọng số master, để huấn luyện mixed-precision không làm mất bước cập nhật do làm tròn FP16.
- **Trả về:** không có. `step()` cập nhật tham số tại chỗ.
- **Trường hợp biên:** không có `get_lr()` / `set_lr()` được bind (chỉ `step()` và `zero_grad()`). Không hỗ trợ `save_optimizer_state`. Trên một tiến trình đơn không có phân tán, nó hoạt động như AdamW với trọng số master.
- **PyTorch:** không có đối tác trong `torch.optim` (ZeRO-3 là tính năng của DeepSpeed).

<!-- ex: id=r9-zero3 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.ZeRO3Optimizer([w], lr=0.01)
opt.step()
print(w.to_vector())
```

## LR scheduler

Mọi scheduler đều là lớp dưới `lt.optim` và hầu hết cũng có dưới dạng hàm factory trong `lt.optim.lr_scheduler`. Dạng factory nhận cùng đối số nhưng là hàm thường; `LinearWarmup` chỉ tồn tại dưới dạng lớp trong `lt.optim.lr_scheduler`. Scheduler giữ tham chiếu tới optimizer (`keep_alive`), nên optimizer phải sống lâu hơn scheduler. Mọi scheduler đều có `step()`; `CosineAnnealingWarmRestarts`, `LinearLR`, `PolynomialLR`, `OneCycleLR`, `LambdaLR`, `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau` còn có `get_last_lr()`. `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau` có `state_dict()` / `load_state_dict(state_dict)`; các scheduler khác không có.

### Thứ tự gọi step

Gọi `optimizer.step()` trước, rồi `scheduler.step()` một lần cho mỗi bước huấn luyện (một lần cho mỗi epoch với lịch theo epoch). `ReduceLROnPlateau.step(metric)` là ngoại lệ: gọi một lần cho mỗi lần validation, truyền metric được theo dõi. Gọi `scheduler.step()` trước `optimizer.step()` đầu tiên sẽ phát cảnh báo trong PyTorch và lặng lẽ dịch lịch trong LiteTorch — luôn step optimizer trước.

### `StepLR`

- **Chữ ký:** `StepLR(optimizer, step_size, gamma=0.1) -> StepLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:107-108`; factory `src/bindings/bind_optim.cpp:426`
- **Tham số:** `step_size` — số bước giữa các lần giảm (bắt buộc, không có mặc định). `gamma` — hệ số giảm nhân, mặc định 0.1.
- **Hành vi:** `lr *= gamma` sau mỗi `step_size` bước.
- **PyTorch:** `torch.optim.lr_scheduler.StepLR` — lịch giống hệt.

<!-- ex: id=r9-steplr device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.StepLR(opt, step_size=2, gamma=0.5)
lrs = []
for i in range(5):
    opt.step()
    sch.step()
    lrs.append(opt.get_lr())
print(lrs)
```

### `CosineAnnealingLR`

- **Chữ ký:** `CosineAnnealingLR(optimizer, T_max, eta_min=0.0) -> CosineAnnealingLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:114-115`; factory `src/bindings/bind_optim.cpp:429`
- **Tham số:** `T_max` — số bước cho nửa chu kỳ cosine (bắt buộc). `eta_min` — tốc độ học tối thiểu, mặc định 0.0.
- **Hành vi:** `lr = eta_min + 0.5 * (lr0 - eta_min) * (1 + cos(pi * t / T_max))`.
- **PyTorch:** `torch.optim.lr_scheduler.CosineAnnealingLR` — lịch giống hệt.

<!-- ex: id=r9-cosine device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.CosineAnnealingLR(opt, T_max=4)
lrs = []
for i in range(5):
    opt.step()
    sch.step()
    lrs.append(opt.get_lr())
print(lrs)
```

### `CosineAnnealingWarmRestarts`

- **Chữ ký:** `CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0) -> CosineAnnealingWarmRestarts`
- **Nguồn:** `src/bindings/bind_optim.cpp:121-122`; factory `src/bindings/bind_optim.cpp:447`
- **Tham số:** `T_0` — số bước trong chu kỳ restart đầu tiên (bắt buộc). `T_mult` — hệ số nhân độ dài chu kỳ sau mỗi lần restart, mặc định 1. `eta_min` — tốc độ học tối thiểu, mặc định 0.0.
- **Hành vi:** suy giảm cosine trong mỗi chu kỳ, đặt lại tốc độ học cơ sở tại mỗi lần restart; độ dài chu kỳ tăng theo `T_mult` mỗi lần.
- **PyTorch:** `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` — lịch giống hệt.

<!-- ex: id=r9-coswarm device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
lrs = []
for i in range(7):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_last_lr(), 6))
print(lrs)
```

### `LinearLR`

- **Chữ ký:** `LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5) -> LinearLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:130-131`; factory `src/bindings/bind_optim.cpp:441`
- **Tham số:** `start_factor` — hệ số nhân tốc độ học ban đầu, mặc định 1/3. `end_factor` — hệ số nhân cuối, mặc định 1.0. `total_iters` — số bước để hệ số di chuyển tuyến tính, mặc định 5.
- **Hành vi:** hệ số di chuyển tuyến tính từ `start_factor` đến `end_factor` trong `total_iters` bước, rồi giữ ở `end_factor`.
- **PyTorch:** `torch.optim.lr_scheduler.LinearLR` — lịch giống hệt.

<!-- ex: id=r9-linearlr device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.LinearLR(opt, start_factor=0.5, end_factor=1.0, total_iters=4)
lrs = []
for i in range(5):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_last_lr(), 6))
print(lrs)
```

### `PolynomialLR`

- **Chữ ký:** `PolynomialLR(optimizer, total_iters=5, power=1.0) -> PolynomialLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:139-140`; factory `src/bindings/bind_optim.cpp:444`
- **Tham số:** `total_iters` — số bước để tốc độ học suy giảm, mặc định 5. `power` — số mũ đa thức, mặc định 1.0 (suy giảm tuyến tính).
- **Hành vi:** `lr = lr0 * (1 - t / total_iters) ** power`; giữ ở 0 sau `total_iters` bước.
- **PyTorch:** `torch.optim.lr_scheduler.PolynomialLR` — lịch giống hệt.

<!-- ex: id=r9-poly device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.PolynomialLR(opt, total_iters=4, power=2.0)
lrs = []
for i in range(5):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_last_lr(), 6))
print(lrs)
```

### `OneCycleLR`

- **Chữ ký:** `OneCycleLR(optimizer, max_lr, total_steps=-1, epochs=-1, steps_per_epoch=-1, pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=10000.0) -> OneCycleLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:147-148`; factory `src/bindings/bind_optim.cpp:450`
- **Tham số:** `max_lr` — tốc độ học đỉnh (bắt buộc). `total_steps` — tổng số bước (mặc định -1, nghĩa là suy ra từ `epochs * steps_per_epoch`). `epochs` / `steps_per_epoch` — cách khác để chỉ định tổng số bước. `pct_start` — tỉ lệ chu kỳ dành cho pha tăng, mặc định 0.3. `anneal_strategy` — `"cos"` hoặc `"linear"`, mặc định `"cos"`. `div_factor` — tốc độ học ban đầu là `max_lr / div_factor`, mặc định 25.0. `final_div_factor` — tốc độ học cuối là `max_lr / (div_factor * final_div_factor)`, mặc định 10000.0.
- **Hành vi:** tốc độ học tăng từ `max_lr / div_factor` lên `max_lr` trong `pct_start` đầu của chu kỳ, rồi anneal về gần 0.
- **Trường hợp biên:** phải cho đúng một trong hai: `total_steps`, hoặc (`epochs` và `steps_per_epoch`); dạng factory yêu cầu `total_steps` ở vị trí và không có tham số `epochs` / `steps_per_epoch`. **Lịch chậm hơn PyTorch một bước.** Đã đo với `max_lr=0.1, total_steps=10, pct_start=0.3` (tốc độ học cơ sở 0.004): LiteTorch báo `[0.028, 0.076, 0.1, 0.09504846, 0.08117457, 0.0611262, 0.0388742, 0.01882583, 0.00495194, 0.0000004]` sau 10 lần gọi `step()`, PyTorch `[0.052, 0.1, 0.09504846, 0.08117457, 0.0611262, 0.0388742, 0.01882583, 0.00495194, 0.0000004, 0.00495194]`. Từ lần gọi thứ ba trở đi LiteTorch bằng lần gọi trước của PyTorch; trên đoạn tăng LiteTorch cần 3 lần gọi để chạm `max_lr` trong khi PyTorch cần 2; sau lần gọi cuối LiteTorch chạm sàn (`max_lr / (div_factor * final_div_factor)`) trong khi PyTorch phát lại giá trị trước. Cùng một đường cong, lệch pha một bước — đừng cho rằng tốc độ học từng bước giống hệt khi chuyển đổi.
- **PyTorch:** `torch.optim.lr_scheduler.OneCycleLR` — lịch giống hệt.

<!-- ex: id=r9-onecycle device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.004)
sch = lt.optim.OneCycleLR(opt, max_lr=0.1, total_steps=10)
lrs = []
for i in range(10):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_last_lr(), 6))
print(lrs)
```

### `LambdaLR`

- **Chữ ký:** `LambdaLR(optimizer, lr_lambda) -> LambdaLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:155-156`; factory `src/bindings/bind_optim.cpp:438`
- **Tham số:** `lr_lambda` — một callable Python `epoch -> hệ số nhân` (bắt buộc).
- **Hành vi:** `lr = base_lr * lr_lambda(step)` tại mỗi bước.
- **PyTorch:** `torch.optim.lr_scheduler.LambdaLR` — lịch giống hệt.

<!-- ex: id=r9-lambda device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.LambdaLR(opt, lambda epoch: 0.9 ** epoch)
lrs = []
for i in range(4):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_last_lr(), 6))
print(lrs)
```

### `MultiStepLR`

- **Chữ ký:** `MultiStepLR(optimizer, milestones, gamma=0.1) -> MultiStepLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:352-353`; factory `src/bindings/bind_optim.cpp:435`
- **Tham số:** `milestones` — danh sách các chỉ số bước mà tốc độ học giảm (bắt buộc). `gamma` — hệ số giảm, mặc định 0.1.
- **Hành vi:** `lr *= gamma` mỗi khi số bước chạm một milestone.
- **Phương thức:** `step()`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **PyTorch:** `torch.optim.lr_scheduler.MultiStepLR` — lịch giống hệt.

<!-- ex: id=r9-multistep device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.MultiStepLR(opt, milestones=[2, 4], gamma=0.5)
lrs = []
for i in range(6):
    opt.step()
    sch.step()
    lrs.append(opt.get_lr())
print(lrs)
```

### `ExponentialLR`

- **Chữ ký:** `ExponentialLR(optimizer, gamma) -> ExponentialLR`
- **Nguồn:** `src/bindings/bind_optim.cpp:380-381`; factory `src/bindings/bind_optim.cpp:432`
- **Tham số:** `gamma` — hệ số giảm nhân cho mỗi bước (bắt buộc, không có mặc định).
- **Hành vi:** `lr *= gamma` mỗi bước.
- **Phương thức:** `step()`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **PyTorch:** `torch.optim.lr_scheduler.ExponentialLR` — lịch giống hệt.

<!-- ex: id=r9-explr device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.ExponentialLR(opt, gamma=0.9)
lrs = []
for i in range(4):
    opt.step()
    sch.step()
    lrs.append(opt.get_lr())
print(lrs)
```

### `ReduceLROnPlateau`

- **Chữ ký:** `ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8) -> ReduceLROnPlateau`
- **Nguồn:** `src/bindings/bind_optim.cpp:399-400`; factory `src/bindings/bind_optim.cpp:426`
- **Tham số:** `mode` — `"min"` (giảm khi metric ngừng giảm) hoặc `"max"`, mặc định `"min"`. `factor` — hệ số giảm, mặc định 0.1. `patience` — số lần gọi không cải thiện trước khi giảm, mặc định 10. `threshold` — ngưỡng cải thiện, mặc định 1e-4. `threshold_mode` — `"rel"` hoặc `"abs"`, mặc định `"rel"`. `cooldown` — số bước chờ sau một lần giảm, mặc định 0. `min_lr` — giới hạn dưới, mặc định 0.0. `eps` — mức giảm tối thiểu áp dụng, mặc định 1e-8.
- **Hành vi:** gọi `step(metric)` một lần cho mỗi lần validation; tốc độ học nhân với `factor` khi metric không cải thiện trong `patience` lần gọi.
- **Phương thức:** `step(metric)`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **Trường hợp biên:** khác các scheduler còn lại, `step()` nhận một đối số `metric` bắt buộc. Không gọi nó cho mỗi bước huấn luyện.
- **PyTorch:** `torch.optim.lr_scheduler.ReduceLROnPlateau` — hành vi giống hệt.

<!-- ex: id=r9-plateau device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.ReduceLROnPlateau(opt, patience=2, factor=0.5)
for metric in [1.0, 0.9, 0.9, 0.9, 0.9]:
    sch.step(metric)
    print(opt.get_lr())
```

### `LinearWarmup`

- **Chữ ký:** `lt.optim.lr_scheduler.LinearWarmup(optimizer, warmup_steps, total_steps, min_lr=0.0) -> LinearWarmup`
- **Nguồn:** `src/bindings/bind_optim.cpp:174-175` (chỉ dạng lớp, trong `lr_scheduler`; không có `lt.optim.LinearWarmup` và không có dạng factory)
- **Tham số:** `warmup_steps` — số bước warmup tuyến tính (bắt buộc). `total_steps` — tổng số bước (bắt buộc). `min_lr` — sàn, mặc định 0.0.
- **Hành vi:** tốc độ học tăng tuyến tính từ `min_lr` lên tốc độ cơ sở trong `warmup_steps`, rồi giảm tuyến tính về `min_lr` tại `total_steps`.
- **Phương thức:** `step()`, `get_lr()`.
- **PyTorch:** không có scheduler tương đương trong `torch.optim.lr_scheduler`.

<!-- ex: id=r9-linearwarmup device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.lr_scheduler.LinearWarmup(opt, warmup_steps=2, total_steps=6)
lrs = []
for i in range(6):
    opt.step()
    sch.step()
    lrs.append(round(sch.get_lr(), 6))
print(lrs)
```

### Factory `lr_scheduler`

- **Chữ ký:** `lt.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)`, `.CosineAnnealingLR(optimizer, T_max, eta_min=0.0)`, `.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0)`, `.LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5)`, `.PolynomialLR(optimizer, total_iters=5, power=1.0)`, `.OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=10000.0)`, `.LambdaLR(optimizer, lr_lambda)`, `.MultiStepLR(optimizer, milestones, gamma=0.1)`, `.ExponentialLR(optimizer, gamma)`, `.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8)`
- **Nguồn:** `src/bindings/bind_optim.cpp:426-453`
- **Hành vi:** các hàm factory trả về cùng đối tượng scheduler như các lớp dưới `lt.optim`. Khác biệt so với dạng lớp: factory `OneCycleLR` yêu cầu `total_steps` ở vị trí và không có tham số `epochs` / `steps_per_epoch`.
- **PyTorch:** `torch.optim.lr_scheduler` — cùng tên, cùng lịch.

<!-- ex: id=r9-lrsched-factories device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
opt = lt.optim.SGD([w], lr=0.1)
sch = lt.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
opt.step()
sch.step()
print(opt.get_lr())
```

## `optim.GradScaler`

- **Chữ ký:** `optim.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True) -> GradScaler`
- **Nguồn:** `src/bindings/bind_optim.cpp:455-456`
- **Tham số:** `init_scale` — tỉ lệ loss ban đầu, mặc định 65536.0. `growth_factor` — hệ số nhân tỉ lệ sau một khoảng sạch, mặc định 2.0. `backoff_factor` — hệ số nhân tỉ lệ sau tràn số, mặc định 0.5. `growth_interval` — số bước sạch liên tiếp trước khi tăng, mặc định 2000. `enabled` — có kích hoạt co tỉ lệ hay không, mặc định True.
- **Phương thức:** `scale(loss)` — trả về loss nhân với tỉ lệ hiện tại; `scale_loss(loss)` — bí danh có cùng hành vi, chỉ có trên `optim.GradScaler` (không có trên `amp.GradScaler`); `unscale_(optimizer)` — chia gradient của optimizer cho tỉ lệ hiện tại, tại chỗ; `step(optimizer)` — chạy `optimizer.step()` trừ khi phát hiện tràn số, khi đó bước bị bỏ qua; `update()` — tăng hoặc giảm tỉ lệ tùy có tràn số hay không; `get_scale()`; `set_scale(scale)`; `is_enabled()`.
- **Trả về:** `scale()` trả về một tensor đã co tỉ lệ mới; `step()` không trả về gì.
- **Trường hợp biên:** `step()` sau `update()` mà không có `scale()` mới dùng trạng thái tràn số cuối cùng. Co tỉ lệ là no-op khi `enabled=False`.
- **PyTorch:** `torch.amp.GradScaler` — cùng hình dạng API; LiteTorch không có `get_growth_tracker()` / `set_growth_tracker()`.

<!-- ex: id=r9-optim-gs device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
loss = lt.Tensor.from_vector([2.0], [1], cpu)
gs = lt.optim.GradScaler(init_scale=4.0)
print(gs.scale_loss(loss).to_vector(), gs.get_scale())
```

## AMP (`litetorch.amp`)

### `amp.GradScaler`

- **Chữ ký:** `amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True) -> GradScaler`
- **Nguồn:** `src/bindings/bind_amp.cpp:6-7`
- **Hành vi:** scaler của AMP, lớp con của `optim.GradScaler` với cùng constructor và phương thức trừ `scale_loss` (không có ở đây). Đây là scaler dùng với `amp.autocast`; `optim.GradScaler` là cùng engine nhưng không gắn với AMP.
- **PyTorch:** `torch.amp.GradScaler` — cùng vai trò.

### `amp.autocast`

- **Chữ ký:** `amp.autocast(device_type="cuda", dtype=None, enabled=True)` — context manager
- **Nguồn:** `src/bindings/bind_amp.cpp:21-22`
- **Tham số:** `device_type` — `"cuda"` (mặc định) hoặc `"cpu"`. `dtype` — kiểu dữ liệu có độ chính xác thấp mục tiêu; khi `None`, CPU mặc định BF16 và CUDA mặc định FP16. `enabled` — mặc định True.
- **Hành vi:** trong context, các phép toán chạy ở kiểu dữ liệu autocast; `is_autocast_enabled()` trả về True và `get_autocast_dtype()` trả về kiểu dữ liệu đang hoạt động. Khi thoát, trạng thái trước đó được khôi phục.
- **Trường hợp biên:** `device_type` mặc định là `"cuda"` — trên máy chỉ có CPU truyền `"cpu"` tường minh. Không có kiểm tra kiểu dữ liệu tại điểm vào: truyền một kiểu dữ liệu không hỗ trợ cho thiết bị được chấp nhận lặng lẽ cho đến khi một op chạy.
- **PyTorch:** `torch.autocast` — cùng hình dạng; autocast CPU của PyTorch cũng mặc định BF16.

### `amp.AutocastGuard`

- **Chữ ký:** `amp.AutocastGuard(enabled=True, dtype=DataType.FP16) -> AutocastGuard`
- **Nguồn:** `src/bindings/bind_amp.cpp:16-17`
- **Tham số:** `enabled` — mặc định True. `dtype` — mặc định FP16.
- **Phương thức:** các phương thức tĩnh `is_enabled()` và `get_dtype()` báo cáo trạng thái autocast toàn cục; `amp.is_autocast_enabled()` và `amp.get_autocast_dtype()` là bí danh ở cấp module cho cùng hai hàm.
- **PyTorch:** không có đối tác trực tiếp (PyTorch chỉ có trạng thái qua `torch.is_autocast_enabled()`).

<!-- ex: id=r9-amp device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
opt = lt.optim.SGD([w], lr=0.1)
gs = lt.amp.GradScaler(init_scale=4.0)
x = lt.Tensor.from_vector([2.0], [1], cpu, True)
with lt.amp.autocast("cpu"):
    print(lt.amp.is_autocast_enabled(), lt.amp.get_autocast_dtype())
    loss = (x * x).sum()
scaled = gs.scale(loss)
scaled.backward()
gs.unscale_(opt)
gs.step(opt)
gs.update()
print(w.to_vector(), gs.get_scale())
print(lt.amp.is_autocast_enabled())
```

<!-- ex: id=r9-autocastguard device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
g = lt.amp.AutocastGuard(True, lt.DataType.FP16)
print(lt.amp.AutocastGuard.is_enabled(), lt.amp.AutocastGuard.get_dtype())
print(lt.amp.is_autocast_enabled(), lt.amp.get_autocast_dtype())
```

## Các chế độ lỗi của AMP

1. **Tràn số làm bỏ qua bước.** Nếu bất kỳ gradient nào là inf hoặc NaN sau `unscale_`, `scaler.step(optimizer)` lặng lẽ bỏ qua bước cập nhật và `update()` giảm tỉ lệ một nửa. Dấu hiệu là `get_scale()` giảm; đừng cho rằng bước đã diễn ra.
2. **Step mà không `unscale_` sẽ áp dụng gradient đã co tỉ lệ.** Gọi `scaler.step(optimizer)` mà không `unscale_` trước sẽ cập nhật với gradient nhân với tỉ lệ (mặc định 65536 lần) — phân kỳ ngay lập tức. Luôn chạy theo thứ tự `scale` → `backward` → `unscale_` → `step` → `update`.
3. **Autocast CPU là BF16, không phải FP16.** `autocast("cpu")` không có dtype dùng BF16; kiểu dữ liệu mặc định của `AutocastGuard` là FP16. Đừng cho rằng mặc định của guard khớp với mặc định của autocast.
4. **`autocast()` mặc định CUDA.** Gọi `lt.amp.autocast()` không có đối số trên máy chỉ có CPU sẽ chọn loại thiết bị `"cuda"`; truyền `"cpu"` tường minh.
5. **NAdam không có bước GPU.** `NAdam.step()` trên tensor GPU ném lỗi; chuyển tham số sang CPU hoặc dùng `Adam`.
