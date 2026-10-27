# Chuyển Đổi từ PyTorch sang LiteTorch

Hướng dẫn này chuyển đổi mã PyTorch sang LiteTorch. LiteTorch phản chiếu phần lớn bề mặt API của PyTorch — `Tensor`, `nn.Module`, `nn.Linear`, optimizer, `DataLoader` — nhưng ánh xạ không phải 1:1. Mọi hàng dưới đây đều dựa trên tham chiếu API (trích dẫn theo từng mục); mọi hàng được đánh dấu **different** (khác) hoặc **absent** (vắng mặt) đều liên kết tới trang Khác Biệt Hành Vi, nơi khác biệt được ghi lại kèm số liệu.

- **Cơ sở:** cây HEAD `b326f29`; các ví dụ chạy trên extension D3 của Core `build_commit 88a4ba1`; PyTorch tham chiếu torch 2.14.1+cpu.
- **Cách đọc bảng:** `identical` (giống hệt) = cùng tên, cùng ngữ nghĩa (trong dung sai parity FP32 1e-4). `renamed` (đổi tên) = cùng ngữ nghĩa, tên khác hoặc dạng gọi khác. `different` (khác) = cùng tên, ngữ nghĩa khác — đọc mục D liên kết trước khi chuyển đổi. `absent` (vắng mặt) = không có tương đương; ghi chú đưa ra cách thay thế hoặc nói rõ là không có.

## Quy trình chuyển đổi

1. **Ánh xạ API** theo các bảng dưới đây; sửa mọi hàng `absent` / `different` trước — đó là những hàng thất bại lặng lẽ, không phải những hàng báo lỗi.
2. **Sửa mã thiết bị.** Tạo `litetorch.Device("cuda")` một cách tường minh (D-03); kiểm tra `t.device` sau mỗi `.to()` (D-02 — rơi về CPU lặng lẽ là lỗi lặng lẽ phổ biến nhất trong một lần chuyển đổi).
3. **Sửa mã module.** Đăng ký submodule bằng `add_module()` (D-24); gọi module đa đối số qua `.forward(...)` (D-22); đừng tạo `Parameter` trực tiếp (D-25).
4. **Sửa mã dữ liệu.** Subclass `Dataset` định nghĩa `size()`/`get()` (D-37); dùng `shuffle=False` cho các lần chạy tái lập (D-46); dùng target loss INT32 (D-23).
5. **Xác minh bằng parity, không bằng đường cong loss.** So sánh giá trị VÀ gradient với torch ở FP32 1e-4 (`plan/verification.md` §3). Khuyết điểm no-learn ngày 2026-10-25/26 là lỗi đường backward với giá trị forward trông đúng — parity chỉ-forward là không đủ với bất cứ mã nào gọi `.backward()`.

## Ví dụ chuyển đổi: một vòng lặp huấn luyện nhỏ

Cùng một MLP 2 lớp, cùng dữ liệu, cùng 5 bước SGD — một lần trong torch, một lần trong LiteTorch. Trọng số được sao chép từ LiteTorch sang torch trước khi huấn luyện, nên cả hai phía bắt đầu từ tham số giống hệt.

<!-- ex: id=mig-port-1 device=cpu backend=cpu runtime=~106s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn as tnn

lt.manual_seed(42)
cpu = lt.Device("cpu")
X = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0]
Y = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
x_lt = lt.Tensor.from_vector(X, [4, 4], cpu, False)
y_lt = lt.Tensor.from_vector(Y, [4, 2], cpu, False)
model_lt = lt.nn.Sequential([lt.nn.Linear(4, 8, True), lt.nn.ReLU(), lt.nn.Linear(8, 2, True)])
opt_lt = lt.optim.SGD(model_lt.parameters(), lr=0.1)
loss_lt = lt.nn.MSELoss()

x_t = torch.tensor(X).reshape(4, 4)
y_t = torch.tensor(Y).reshape(4, 2)
model_t = tnn.Sequential(tnn.Linear(4, 8), tnn.ReLU(), tnn.Linear(8, 2))
with torch.no_grad():
    for p_lt, p_t in zip(model_lt.parameters(), model_t.parameters()):
        p_t.copy_(torch.tensor(p_lt.to_vector()).reshape(list(p_t.shape)))
opt_t = torch.optim.SGD(model_t.parameters(), lr=0.1)
loss_t = tnn.MSELoss()

for step in range(5):
    opt_lt.zero_grad()
    pred_lt = model_lt.forward(x_lt)
    l_lt = loss_lt.forward(pred_lt, y_lt)
    l_lt.backward()
    opt_lt.step()
    opt_t.zero_grad()
    pred_t = model_t(x_t)
    l_t = loss_t(pred_t, y_t)
    l_t.backward()
    opt_t.step()

print("loss cuoi lt:", l_lt.to_vector()[0])
print("loss cuoi torch:", l_t.item())
maxdiff = 0.0
for p_lt, p_t in zip(model_lt.parameters(), model_t.parameters()):
    v = p_lt.to_vector()
    t = p_t.detach().flatten().tolist()
    maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(v, t)))
print("chenh lech trong so lon nhat:", maxdiff)
```

Đo trên extension xác minh: loss cuối lt 0.23475296795368195, loss cuối torch 0.23475295305252075, chênh lệch trọng số lớn nhất 5.96e-08 — trong dung sai parity FP32 1e-4.

## Chọn thiết bị

Torch chọn thiết bị bằng `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. LiteTorch có `auto_device()` cho cùng mục đích — nhưng `auto_device()` có thể trả về thiết bị CUDA rồi lặng lẽ rơi về CPU (D-02), nên luôn xác minh.

<!-- ex: id=mig-device-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

device = lt.auto_device()
print("auto_device:", device)
t = lt.Tensor.from_vector([1.0, 2.0], [2], device, False)
print("thiet bi cua tensor:", t.device)
print("cuda kha dung:", lt.cuda.is_available())
if str(t.device) != str(device):
    print("CANH BAO: thiet bi khong khop — roi ve lang le (D-02)")
```

## Tạo tensor

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.tensor(list)` | `lt.Tensor.from_vector(list, shape, device, requires_grad)` | đổi tên; shape tường minh, kiểu dữ liệu suy ra là FP32 |
| `torch.zeros(2, 3)` / `torch.zeros([2, 3])` | `lt.zeros([2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.ones(2, 3)` | `lt.ones([2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.randn(2, 3)` | `lt.randn([2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.rand(2, 3)` | `lt.rand([2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.empty(2, 3)` | `lt.empty([2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.full((2, 3), 1.0)` | `lt.full([2, 3], 1.0)` | khác: chỉ một dãy shape (D-04) |
| `torch.arange(0, 5)` | `lt.Tensor.arange(0, 5)` | khác: chỉ số nguyên; không có bước float (D-15) |
| `torch.linspace(0, 1, 5)` | `lt.linspace(0, 1, 5)` | giống hệt |
| `torch.logspace(0, 2, 5)` | `lt.logspace(0, 2, 5)` | giống hệt |
| `torch.eye(3)` | `lt.eye(3)` | giống hệt |
| `torch.randint(0, 10, (2, 3))` | `lt.randint(0, 10, [2, 3])` | khác: chỉ một dãy shape (D-04) |
| `torch.randperm(5)` | `lt.randperm(5)` | giống hệt |
| `torch.zeros_like(t)` v.v. | `lt.zeros_like(t)` / `ones_like` / `empty_like` / `full_like` / `rand_like` / `randn_like` | giống hệt |
| `torch.manual_seed(42)` | `lt.manual_seed(42)` | giống hệt với RNG tensor; KHÔNG gieo hạt cho shuffle của DataLoader (D-46) |
| `torch.get_rng_state()` / `set_rng_state()` | `lt.get_rng_state()` / `lt.set_rng_state()` | giống hệt |

## Phương thức và thuộc tính Tensor

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `t.shape` / `t.dtype` / `t.device` / `t.numel()` / `t.dim()` | tên giống hệt | giống hệt |
| `t.to("cuda")` / `t.to("cpu")` / `t.to(torch.float32)` | `t.to(lt.Device("cuda"))` / `t.to(lt.DataType.FP32)` | khác: không có dạng chuỗi (D-03); rơi về CPU lặng lẽ (D-02) |
| `t.cuda()` / `t.cpu()` | `t.cuda()` / `t.cpu()` | khác: `.cuda()` lặng lẽ trả về CPU khi không có GPU (D-02) |
| `t.half()` / `t.float()` / `t.double()` / `t.int()` / `t.long()` | tên giống hệt | giống hệt (`.long()` là INT64 kể từ bản sửa D-06) |
| `t.bool()` / `t.byte()` / `t.char()` / `t.short()` | vắng mặt | dùng `t.cast(lt.DataType.BOOL)` v.v. (D-13) |
| `t.numpy()` | `t.numpy()` | khác: từ chối UINT8 / INT64 / BOOL; BF16 lặng lẽ ép sang FP32; chỉ tensor CPU (D-14) |
| `t.item()` / `t.tolist()` | `t.item()` / `t.tolist()` / `t.to_vector()` | khác: trả về float; mất chính xác int64 trên 2^53 (D-18) |
| `t.data_ptr()` | vắng mặt trong Python | chỉ C++, và báo lỗi trên tensor không phải FP32 (D-01) |
| `t[mask]` / `t[idx]` / `t[:, 1]` | vắng mặt | chỉ int/slice trên dim 0; dùng `Ops.gather`, `index_select`, `slice` (D-11) |
| `t.sum(dim=(0, 1))` | `lt.Ops.sum(t, [0, 1])` | khác: phương thức Tensor chỉ nhận một dim int (D-12) |
| `t.pow(exp)` / `t ** u` | vắng mặt | `Ops.pow(t, float_exp)`; số mũ dạng tensor không hỗ trợ (D-05) |
| `t.exp()` / `t.log()` / `t.sqrt()` / `t.relu()` / `t.sigmoid()` / `t.tanh()` / `t.abs()` / `t.squeeze()` / `t.unsqueeze()` | vắng mặt dưới dạng phương thức | dùng `lt.Ops.*` (D-17); chỉ `log_softmax` và `silu` được bind làm phương thức |
| `t.view(shape)` / `t.reshape(shape)` / `t.permute(...)` / `t.transpose(a, b)` / `t.expand(...)` / `t.detach()` / `t.clone()` / `t.contiguous()` | tên giống hệt | giống hệt |
| `t.backward()` / `t.zero_grad()` / `t.retain_grad()` / `t.register_hook(...)` | tên giống hệt | giống hệt |
| `t.copy_(src)` / `t.add_(other)` | tên giống hệt | giống hệt |
| `t.requires_grad` / `t.grad` | tên giống hệt | giống hệt |

## Op (`torch.*` / `F.*` → `litetorch.Ops.*`)

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.add/sub/mul/div` / `t + u` | `lt.Ops.add/sub/mul/div` / `t + u` | khác: tensor int + vô hướng int trong Python nâng lên FP32 (D-08); toán int trên GPU chạy qua FP32 (D-21) |
| `torch.matmul` / `t @ u` | `lt.Ops.matmul` / `t @ u` | giống hệt |
| `torch.bmm` | `lt.Ops.bmm` | giống hệt |
| `torch.einsum` | vắng mặt | không có tương đương |
| `torch.softmax` / `log_softmax` | `lt.Ops.softmax` / `lt.Ops.log_softmax` | giống hệt |
| `F.gelu` | `lt.Ops.gelu` | khác: chỉ xấp xỉ tanh, không có đối số `approximate` (D-09) |
| `F.silu` / `F.sigmoid` / `F.tanh` / `F.relu` / `F.leaky_relu` | `lt.Ops.silu` / `sigmoid` / `tanh` / `relu` / `leaky_relu` | giống hệt |
| `F.scaled_dot_product_attention` | vắng mặt | dùng `lt.Ops.flash_attention(q, k, v)`; không có đối số scale/dropout (D-20) |
| `torch.clamp(t, min, max)` | `lt.Ops.clamp(t, min, max)` | khác: chỉ biên vô hướng (D-16) |
| `torch.where` | `lt.Ops.where` | giống hệt |
| `torch.cat` / `torch.stack` / `torch.chunk` / `torch.split` | `lt.Ops.cat` / `stack` / `chunk` / `split` | giống hệt |
| `torch.gather` / `torch.scatter` | `lt.Ops.gather` / `scatter` / `scatter_add` | giống hệt |
| `torch.topk` / `torch.sort` / `torch.argsort` / `torch.argmax` / `torch.argmin` | `lt.Ops.topk` / `sort` / `argsort` / `argmax` / `argmin` | giống hệt |
| `torch.sum` / `mean` / `max` / `min` / `cumsum` | `lt.Ops.sum` / `mean` / `max` / `min` / `cumsum` | giống hệt (chấp nhận đa dim — cách khắc phục D-12) |
| `torch.nn.functional.cross_entropy` | `lt.Ops.cross_entropy_loss` | khác: target INT64 bị từ chối, dùng INT32 (D-23) |
| `torch.nn.functional.mse_loss` / `l1_loss` / `bce_loss` / `nll_loss` / `bce_with_logits` | `lt.Ops.mse_loss` / `l1_loss` / `bce_loss` / `nll_loss` / `bce_with_logits_loss` | giống hệt với đầu vào FP32 |
| `torch.conv2d` / `F.conv2d` | `lt.Ops.conv2d` | khác: chỉ stride/padding vô hướng; không có dilation/groups |
| `F.max_pool2d` / `F.avg_pool2d` / `F.adaptive_avg_pool2d` | `lt.Ops.max_pool2d` / `avg_pool2d` / `adaptive_avg_pool2d` | giống hệt |
| `F.layer_norm` / `F.batch_norm` | `lt.Ops.layer_norm` / `batch_norm2d` | giống hệt |
| `torch.allclose` / `isclose` | `lt.Ops.allclose` / `isclose` | giống hệt |
| `torch.utils.clip_grad_norm_` | `lt.Ops.clip_grad_norm_` | giống hệt |

## Mạng nơ-ron (`torch.nn` → `litetorch.nn`)

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| Subclass `nn.Module` | subclass `lt.nn.Module` | khác: gán thuộc tính KHÔNG đăng ký submodule — dùng `add_module()` (D-24); `parameters()` phải thu thập tường minh trong subclass (xem T1) |
| `model(x)` | `model.forward(x)` | khác: `__call__` chỉ đi qua forward một tensor; module đa đối số phải dùng `.forward(...)` (D-22) |
| `nn.Sequential` / `ModuleList` / `ModuleDict` | tên giống hệt | khác: mục `ModuleDict` đi qua `m[key] = module` / `put()`, không phải `add_module()`; không có `values()`/`items()` |
| `nn.Parameter(tensor)` | `lt.nn.Parameter(tensor)` | HỎNG trên 88a4ba1 — luôn báo lỗi (D-25); lấy tham số từ module thay thế |
| `nn.Linear` / `Bilinear` | tên giống hệt | giống hệt (Linear); Bilinear đa đối số — dùng `.forward()` (D-22) |
| `nn.Conv1d/2d/3d` / `ConvTranspose1d/2d/3d` | tên giống hệt | giống hệt |
| `nn.MaxPool2d/3d` / `AvgPool2d` / `AdaptiveAvgPool2d` / `Fold` / `Unfold` / `PixelShuffle` / `PixelUnshuffle` | tên giống hệt | giống hệt |
| `nn.LayerNorm` / `RMSNorm` / `BatchNorm1d/2d/3d` / `GroupNorm` / `InstanceNorm1d/2d/3d` | tên giống hệt | giống hệt; `SyncBatchNorm` chỉ theo kênh: đầu vào rank-2 (N, C) báo lỗi, rank-3 (N, C, L) chạy được |
| `nn.ReLU/ReLU6/LeakyReLU/Sigmoid/Tanh/GELU/SiLU/ELU/CELU/SELU/Mish/Softplus/Softshrink/Softsign/Hardshrink/Hardtanh/Threshold/Tanhshrink/PReLU/RReLU/GLU/LogSoftmax/Softmax` | tên giống hệt | giống hệt (GELU chỉ tanh, D-09) |
| `nn.Embedding` / `EmbeddingBag` | tên giống hệt | giống hệt; forward của `VocabParallelEmbedding` HỎNG (D-39) |
| `nn.RNN/RNNCell/LSTM/LSTMCell/GRU/GRUCell` | tên giống hệt | giống hệt |
| `nn.MultiHeadAttention` | tên giống hệt | khác: đa đối số — dùng `.forward(q, k, v)` (D-22) |
| `nn.TransformerEncoderLayer` / `TransformerDecoderLayer` / `TransformerEncoder` | tên giống hệt | khác: đầu ra decoder khác torch 0.12–0.14 (D-29); parity encoder chỉ ở chế độ eval; hàm tạo decoder là `(embed_dim, num_heads, dim_feedforward=2048)` |
| `nn.GPTBlock` / `nn.GPT2B` | tên giống hệt | block GPT chỉ có trong LiteTorch; GPT2B đã được bind |
| `nn.MSELoss` / `L1Loss` / `BCELoss` / `BCEWithLogitsLoss` / `NLLLoss` / `CrossEntropyLoss` / `HuberLoss` / `SmoothL1Loss` / `KLDivLoss` / `CTCLoss` / `DPOLoss` / `TripletMarginLoss` / `HingeEmbeddingLoss` / `MultiMarginLoss` / `GaussianNLLLoss` / `PoissonNLLLoss` / `MarginRankingLoss` / `MultiLabelMarginLoss` / `SoftMarginLoss` / `CosineEmbeddingLoss` | tên giống hệt | khác: target loss phải là INT32 (D-23); công thức HingeEmbeddingLoss khác (D-26); chuẩn hóa MultiMarginLoss khác, `p` là int (D-27); giảm trung bình CTCLoss khác (D-28) |
| `F.linear` / `F.dropout` / `F.pad` / `F.interpolate` / `F.nll_loss` / `F.bce_with_logits` / `F.normalize` / `F.glu` / `F.elu` / `F.dpo_loss` | `lt.nn.functional.*` (cùng tên) | giống hệt với 10 hàm này; mọi `F.*` khác đều vắng mặt — dùng `Ops.*` hoặc module |
| `nn.init.kaiming_uniform_` / `xavier_uniform_` / `normal_` / `uniform_` / `zeros_` / `ones_` / `fill_` / `trunc_normal_` / `orthogonal_` | `lt.nn.init.*` (cùng tên) | giống hệt |
| `nn.utils.clip_grad_value_` / `parameters_to_vector` / `vector_to_parameters` / `weight_norm` / `spectral_norm` | `lt.nn.utils.*` (cùng tên) | giống hệt; `clip_grad_norm_` nằm dưới `Ops` |
| `model.state_dict()` / `load_state_dict(sd)` | tên giống hệt | giống hệt |
| `model.to("cuda")` | `model.to(lt.Device("cuda"))` | khác: chỉ Device, không có dạng chuỗi (D-03) |

## Optimizer và scheduler

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.optim.SGD` | `lt.optim.SGD(params, lr=0.01, momentum=0.0, ...)` | giống hệt (đã xác minh 1e-7) |
| `torch.optim.Adam` / `AdamW` | `lt.optim.Adam` / `AdamW` | giống hệt |
| `torch.optim.RMSprop` / `Adadelta` / `Adamax` / `Adagrad` | tên giống hệt | giống hệt |
| `torch.optim.NAdam` | `lt.optim.NAdam` | HỎNG — bước đầu tiên lớn hơn ~6.5 lần (D-30); dùng Adam/AdamW |
| `torch.optim.LBFGS` | vắng mặt | không có tương đương |
| Biến thể 8-bit / FP8 của `torch.optim.AdamW` | `lt.optim.AdamW8bit` / `AdamWFP8` | chỉ có trong LiteTorch; giống hệt torch AdamW tới 6 chữ số thập phân trên thăm dò 1 bước |
| `torch.optim.lr_scheduler.StepLR` / `CosineAnnealingLR` / `CosineAnnealingWarmRestarts` / `LinearLR` / `PolynomialLR` / `LambdaLR` / `MultiStepLR` / `ExponentialLR` / `ReduceLROnPlateau` | tên giống hệt dưới `lt.optim` | giống hệt (tất cả đã xác minh khớp chính xác với torch) |
| `torch.optim.lr_scheduler.OneCycleLR` | `lt.optim.OneCycleLR` | khác: lệch pha một bước (D-31) |
| `torch.cuda.amp.GradScaler` | `lt.optim.GradScaler` / `lt.amp.GradScaler` | giống hệt (bỏ qua tràn đã xác minh) |
| `torch.cuda.amp.autocast` | `lt.amp.autocast` / `lt.amp.AutocastGuard` | giống hệt |

## Nạp dữ liệu

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.utils.data.Dataset` | `lt.data.Dataset` | khác: subclass định nghĩa `size()` / `get(idx)`, KHÔNG phải `__len__` / `__getitem__` (D-37) |
| `torch.utils.data.TensorDataset` | `lt.data.TensorDataset` | giống hệt |
| `torch.utils.data.DataLoader` | `lt.data.DataLoader` | khác: `shuffle=True` không được gieo hạt bởi `manual_seed` (D-46); `collate_fn` Python kích hoạt lỗi abort khi tắt của D-37 |
| `torch.utils.data.Sampler` / `SequentialSampler` / `RandomSampler` / `BatchSampler` | `lt.data.Sampler` v.v. | khác: sampler không thể lặp — chỉ `indices()` / `batches()` |
| `torch.utils.data.Subset` / `random_split` | `lt.data.Subset` / `lt.data.random_split` | giống hệt |
| `torch.save(state_dict, path)` | `lt.save(module, path)` | khác: nhận Module/list/Optimizer, không phải state dict (D-45) |
| `torch.load(path)` | `lt.load(...)` | khác: in-place, điều phối theo kiểu; tensor trần báo lỗi (D-45) |
| `torch.save` trạng thái optimizer | `lt.save_optimizer_state` | khác: chỉ Adam/AdamW/SGD/RMSprop (D-45) |
| `safetensors.torch.save_file` | `lt.save_safetensors(dict)` / `lt.load_safetensors` | khác: nhận dict, không phải module (D-45) |
| `transformers.Trainer` | `lt.Trainer` | khác: `train_step()` không chạy forward/backward; checkpoint là async theo mặc định (D-44) |

## Huấn luyện phân tán

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.distributed.init_process_group` | `lt.distributed.init_process_group` | giống hệt (phải gọi trước tiên) |
| `torch.distributed.destroy_process_group` | vắng mặt | dùng `lt.distributed.shutdown()` |
| `torch.distributed.barrier` / `all_reduce` / `all_gather` / `reduce_scatter` / `broadcast` / `all_to_all` | tên giống hệt | giống hệt |
| `torch.nn.parallel.DistributedDataParallel` | `lt.distributed.DistributedDataParallel` | HỎNG — backward nhân đôi một lát gradient (D-33); đừng huấn luyện với DDP |
| `torch.distributed.fsdp.FullyShardedDataParallel` | vắng mặt dưới dạng lớp | dùng `lt.distributed.fsdp_fully_shard(module)`; FSDP shard tham số + gradient, KHÔNG phải trạng thái optimizer; phương thức `FSDPUnit` không tới được từ Python |
| `torch.distributed.tensor` / `DeviceMesh` / `DTensor` | `lt.distributed.DeviceMesh` / `DTensor` / `Placement` | giống hệt về đại thể; xem `reference/distributed.md` |
| Backend NCCL của `torch.distributed` | `lt.distributed.NCCLBridge` | dlopen thuần (`libnccl.so` → `librccl.so`), fail-soft; override `LITETORCH_NCCL_LIB` là patch đã kiểm chứng trên Kaggle, chưa merge (dist-w15) |
| `torchrun` / rendezvous | `lt.distributed.RendezvousStore` / `RendezvousClient` | khác: join thread `run()` sau `stop()` nếu không tiến trình abort khi thoát (D-35) |

## AMP, JIT, lượng tử hóa, serving

| PyTorch | LiteTorch | Ghi chú |
|---|---|---|
| `torch.cuda.amp.autocast` / `GradScaler` | `lt.amp.autocast` / `AutocastGuard` / `GradScaler` | giống hệt |
| `torch.jit.script` / `torch.jit.trace` | `lt.jit.Tracer.trace` → `JITFunction` | khác: `trace` trả về `JITFunction`, không phải `JITVar`; trung gian `+` thuần báo lỗi intermediate đã hết hạn (dùng `lt.Ops.*`) |
| `torch.jit.fuse` / op fused | `lt.jit.fused_bias_relu` / `fused_bias_gelu` / `fused_residual_add` / `fused_residual_gelu` | khác: op bias fused yêu cầu shape giống hệt (không broadcasting) |
| `torch.quantization.quantize_dynamic` | `lt.quantization.quantize_dynamic_linear` / `quantize_dynamic_lstm` / `quantize_dynamic_gru` | đổi tên; hình dạng API khác |
| `torch.quantization.prepare` / `convert` / QAT | `lt.quantization.prepare` / `convert` / `PreparedModel` / `QATLinear` / `Calibrator` / `PTQConfig` | đổi tên; `calibrate` nhận callable không đối số `data_fn`, không phải list |
| `torch.onnx.export` | `lt.onnx.export` / `export_dynamic` / `export_onnx_text` / `export_onnx_binary` | đổi tên |
| `transformers.AutoModel.from_pretrained` | `lt.hf.from_pretrained` | đổi tên; `hf.load_config`, `hf.remap_state_dict`, `hf.weight_name_map` |
| `torch.cuda.CUDAGraph` | `lt.inference.CudaGraph` / `capture_cuda_graph` | đổi tên |
| `torch.nn.utils.prune` | `lt.prune.l1_unstructured` / `random_unstructured` / `remove` / `is_pruned` | đổi tên |
| `torch.profiler` | `lt.profiler.enable` / `disable` / `report` / `record` / `profile` | đổi tên |
| `torch.hub.load_state_dict_from_url` | `lt.hub.load_state_dict_from_url` / `download_url_to_file` | giống hệt |
| Serving kiểu vLLM | `lt.Ops.paged_attention` (6 đối số vị trí), `ServingScheduler.step(model)`, `PrefixCache` (dạng list), `MedusaHead`, `SpeculativeEngine` | chỉ có trong LiteTorch; không có tương đương torch |
| `torchvision` / `torchaudio` / `torchtext` | `ltvision` / `ltaudio` / `lttext` / `ltmobile` / `ltgeometric` / `ltrec` / `ltrrl` | đổi tên; API khác — xem `reference/subpackages.md` |

## API C++

API C++ (`include/litetorch/`, xem `reference/cpp-api.md`) là bề mặt riêng với quy tắc riêng: `Tensor::data_ptr()` báo lỗi trên tensor không phải FP32 (D-01), `Ops::reduce_broadcast` / `Ops::from_data` chỉ có trong C++, và bố cục build CMake được ghi trong tham chiếu C++. Không có bảng ánh xạ libtorch của PyTorch — chuyển đổi mã C++ từng hàm một theo `reference/cpp-api.md`.

## Danh sách kiểm tra lỗi thường gặp

Trước khi chạy một lần huấn luyện đã chuyển đổi, kiểm tra những mục này — đây là những lỗi không báo lỗi:

1. Thiết bị: xác minh `t.device` sau `.to()` (D-02). Một lần "GPU" lặng lẽ chạy trên CPU là lỗi lặng lẽ phổ biến nhất.
2. Đăng ký module: `add_module()` hoặc ghi đè `parameters()` tường minh (D-24). `parameters()` trống nghĩa là optimizer bước qua khoảng trống.
3. Target loss: INT32, không phải INT64 (D-23). `.long()` nay trả về INT64 (D-06) — và các hàm loss từ chối nó.
4. Gradient: so sánh gradient, không chỉ giá trị forward (bài học no-learn). DDP nhân đôi một lát gradient (D-33); NAdam lệch ~6.5 lần (D-30).
5. DataLoader: `shuffle=False` để tái lập (D-46); `size()`/`get()` cho dataset tùy chỉnh (D-37).
6. GELU: dung sai 5e-4 so với torch mặc định, hoặc so sánh với `approximate='tanh'` (D-09).
7. gradcheck: truyền `eps=1e-3` cho FP32 (D-34).
