# Migrating from PyTorch to LiteTorch

This guide ports PyTorch code to LiteTorch. LiteTorch mirrors much of the PyTorch API surface — `Tensor`, `nn.Module`, `nn.Linear`, optimizers, `DataLoader` — but the mapping is not 1:1. Every row below is grounded in the API reference (cited per section); every row marked **different** or **absent** links to the Behavioral Differences page, where the deviation is documented with numbers.

- **Baseline:** tree HEAD `b326f29`; examples run against the Core D3 extension `build_commit 88a4ba1`; PyTorch reference torch 2.14.1+cpu.
- **Reading the tables:** `identical` = same name, same semantics (within FP32 1e-4 parity). `renamed` = same semantics, different name or call shape. `different` = same name, different semantics — read the linked D-entry before porting. `absent` = no equivalent; the notes give the replacement or state that none exists.

## Porting workflow

1. **Map the API** with the tables below; fix every `absent` / `different` row first — those are the rows that fail silently, not the ones that raise.
2. **Fix the device code.** Construct `litetorch.Device("cuda")` explicitly (D-03); check `t.device` after every `.to()` (D-02 — the silent CPU fallback is the most common silent failure in a port).
3. **Fix the module code.** Register submodules with `add_module()` (D-24); call multi-arg modules via `.forward(...)` (D-22); do not construct `Parameter` directly (D-25).
4. **Fix the data code.** `Dataset` subclasses define `size()`/`get()` (D-37); use `shuffle=False` for reproducible runs (D-46); use INT32 loss targets (D-23).
5. **Verify with parity, not with loss curves.** Compare values AND grads against torch at FP32 1e-4 (`plan/verification.md` §3). The no-learn defect of 2026-10-25/26 was a gradient-path corruption with correct-looking forward values — forward-only parity is not sufficient for any code that calls `.backward()`.

## Worked port: a small training loop

The same 2-layer MLP, the same data, the same 5 SGD steps — once in torch, once in LiteTorch. Weights are copied from LiteTorch into torch before training, so both sides start from identical parameters.

<!-- ex: id=mig-port-1 device=cpu backend=cpu runtime=~93s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
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

print("lt final loss:", l_lt.to_vector()[0])
print("torch final loss:", l_t.item())
maxdiff = 0.0
for p_lt, p_t in zip(model_lt.parameters(), model_t.parameters()):
    v = p_lt.to_vector()
    t = p_t.detach().flatten().tolist()
    maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(v, t)))
print("max weight diff:", maxdiff)
```

Measured on the verification extension: lt final loss 0.23475296795368195, torch final loss 0.23475295305252075, max weight diff 5.96e-08 — inside the FP32 1e-4 parity tolerance.

## Device selection

Torch picks a device with `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. LiteTorch has `auto_device()` for the same purpose — but `auto_device()` can return a CUDA device that then silently falls back to CPU (D-02), so always verify.

<!-- ex: id=mig-device-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

device = lt.auto_device()
print("auto_device:", device)
t = lt.Tensor.from_vector([1.0, 2.0], [2], device, False)
print("tensor device:", t.device)
print("cuda available:", lt.cuda.is_available())
if str(t.device) != str(device):
    print("WARNING: device mismatch — silent fallback (D-02)")
```

## Tensor creation

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.tensor(list)` | `lt.Tensor.from_vector(list, shape, device, requires_grad)` | renamed; shape is explicit, dtype inferred as FP32 |
| `torch.zeros(2, 3)` / `torch.zeros([2, 3])` | `lt.zeros([2, 3])` | different: single shape sequence only (D-04) |
| `torch.ones(2, 3)` | `lt.ones([2, 3])` | different: single shape sequence only (D-04) |
| `torch.randn(2, 3)` | `lt.randn([2, 3])` | different: single shape sequence only (D-04) |
| `torch.rand(2, 3)` | `lt.rand([2, 3])` | different: single shape sequence only (D-04) |
| `torch.empty(2, 3)` | `lt.empty([2, 3])` | different: single shape sequence only (D-04) |
| `torch.full((2, 3), 1.0)` | `lt.full([2, 3], 1.0)` | different: single shape sequence only (D-04) |
| `torch.arange(0, 5)` | `lt.Tensor.arange(0, 5)` | different: integer-only; no float step (D-15) |
| `torch.linspace(0, 1, 5)` | `lt.linspace(0, 1, 5)` | identical |
| `torch.logspace(0, 2, 5)` | `lt.logspace(0, 2, 5)` | identical |
| `torch.eye(3)` | `lt.eye(3)` | identical |
| `torch.randint(0, 10, (2, 3))` | `lt.randint(0, 10, [2, 3])` | different: single shape sequence only (D-04) |
| `torch.randperm(5)` | `lt.randperm(5)` | identical |
| `torch.zeros_like(t)` etc. | `lt.zeros_like(t)` / `ones_like` / `empty_like` / `full_like` / `rand_like` / `randn_like` | identical |
| `torch.manual_seed(42)` | `lt.manual_seed(42)` | identical for tensor RNG; does NOT seed DataLoader shuffle (D-46) |
| `torch.get_rng_state()` / `set_rng_state()` | `lt.get_rng_state()` / `lt.set_rng_state()` | identical |

## Tensor methods and attributes

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `t.shape` / `t.dtype` / `t.device` / `t.numel()` / `t.dim()` | identical names | identical |
| `t.to("cuda")` / `t.to("cpu")` / `t.to(torch.float32)` | `t.to(lt.Device("cuda"))` / `t.to(lt.DataType.FP32)` | different: no string form (D-03); silent CPU fallback (D-02) |
| `t.cuda()` / `t.cpu()` | `t.cuda()` / `t.cpu()` | different: `.cuda()` silently returns CPU when no GPU (D-02) |
| `t.half()` / `t.float()` / `t.double()` / `t.int()` / `t.long()` | identical names | identical (`.long()` is INT64 since the D-06 fix) |
| `t.bool()` / `t.byte()` / `t.char()` / `t.short()` | absent | use `t.cast(lt.DataType.BOOL)` etc. (D-13) |
| `t.numpy()` | `t.numpy()` | different: rejects UINT8 / INT64 / BOOL; BF16 silently cast to FP32; CPU tensors only (D-14) |
| `t.item()` / `t.tolist()` | `t.item()` / `t.tolist()` / `t.to_vector()` | different: return float; int64 precision loss above 2^53 (D-18) |
| `t.data_ptr()` | absent in Python | C++-only, and throws on non-FP32 (D-01) |
| `t[mask]` / `t[idx]` / `t[:, 1]` | absent | dim-0 int/slice only; use `Ops.gather`, `index_select`, `slice` (D-11) |
| `t.sum(dim=(0, 1))` | `lt.Ops.sum(t, [0, 1])` | different: Tensor methods take a single int dim (D-12) |
| `t.pow(exp)` / `t ** u` | absent | `Ops.pow(t, float_exp)`; tensor exponents unsupported (D-05) |
| `t.exp()` / `t.log()` / `t.sqrt()` / `t.relu()` / `t.sigmoid()` / `t.tanh()` / `t.abs()` / `t.squeeze()` / `t.unsqueeze()` | absent as methods | use `lt.Ops.*` (D-17); only `log_softmax` and `silu` are bound as methods |
| `t.view(shape)` / `t.reshape(shape)` / `t.permute(...)` / `t.transpose(a, b)` / `t.expand(...)` / `t.detach()` / `t.clone()` / `t.contiguous()` | identical names | identical |
| `t.backward()` / `t.zero_grad()` / `t.retain_grad()` / `t.register_hook(...)` | identical names | identical |
| `t.copy_(src)` / `t.add_(other)` | identical names | identical |
| `t.requires_grad` / `t.grad` | identical names | identical |

## Ops (`torch.*` / `F.*` → `litetorch.Ops.*`)

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.add/sub/mul/div` / `t + u` | `lt.Ops.add/sub/mul/div` / `t + u` | different: int tensor + Python int scalar promotes to FP32 (D-08); GPU int math runs through FP32 (D-21) |
| `torch.matmul` / `t @ u` | `lt.Ops.matmul` / `t @ u` | identical |
| `torch.bmm` | `lt.Ops.bmm` | identical |
| `torch.einsum` | absent | no equivalent |
| `torch.softmax` / `log_softmax` | `lt.Ops.softmax` / `lt.Ops.log_softmax` | identical |
| `F.gelu` | `lt.Ops.gelu` | different: tanh approximation only, no `approximate` argument (D-09) |
| `F.silu` / `F.sigmoid` / `F.tanh` / `F.relu` / `F.leaky_relu` | `lt.Ops.silu` / `sigmoid` / `tanh` / `relu` / `leaky_relu` | identical |
| `F.scaled_dot_product_attention` | absent | use `lt.Ops.flash_attention(q, k, v)`; no scale/dropout args (D-20) |
| `torch.clamp(t, min, max)` | `lt.Ops.clamp(t, min, max)` | different: scalar bounds only (D-16) |
| `torch.where` | `lt.Ops.where` | identical |
| `torch.cat` / `torch.stack` / `torch.chunk` / `torch.split` | `lt.Ops.cat` / `stack` / `chunk` / `split` | identical |
| `torch.gather` / `torch.scatter` | `lt.Ops.gather` / `scatter` / `scatter_add` | identical |
| `torch.topk` / `torch.sort` / `torch.argsort` / `torch.argmax` / `torch.argmin` | `lt.Ops.topk` / `sort` / `argsort` / `argmax` / `argmin` | identical |
| `torch.sum` / `mean` / `max` / `min` / `cumsum` | `lt.Ops.sum` / `mean` / `max` / `min` / `cumsum` | identical (multi-dim accepted — D-12 workaround) |
| `torch.nn.functional.cross_entropy` | `lt.Ops.cross_entropy_loss` | different: INT64 targets rejected, use INT32 (D-23) |
| `torch.nn.functional.mse_loss` / `l1_loss` / `bce_loss` / `nll_loss` / `bce_with_logits` | `lt.Ops.mse_loss` / `l1_loss` / `bce_loss` / `nll_loss` / `bce_with_logits_loss` | identical for FP32 inputs |
| `torch.conv2d` / `F.conv2d` | `lt.Ops.conv2d` | different: scalar stride/padding only; no dilation/groups |
| `F.max_pool2d` / `F.avg_pool2d` / `F.adaptive_avg_pool2d` | `lt.Ops.max_pool2d` / `avg_pool2d` / `adaptive_avg_pool2d` | identical |
| `F.layer_norm` / `F.batch_norm` | `lt.Ops.layer_norm` / `batch_norm2d` | identical |
| `torch.allclose` / `isclose` | `lt.Ops.allclose` / `isclose` | identical |
| `torch.utils.clip_grad_norm_` | `lt.Ops.clip_grad_norm_` | identical |

## Neural networks (`torch.nn` → `litetorch.nn`)

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `nn.Module` subclass | `lt.nn.Module` subclass | different: attribute assignment does NOT register submodules — use `add_module()` (D-24); `parameters()` must be collected explicitly in subclasses (see T1) |
| `model(x)` | `model.forward(x)` | different: `__call__` routes only the single-tensor forward; multi-arg modules must use `.forward(...)` (D-22) |
| `nn.Sequential` / `ModuleList` / `ModuleDict` | identical names | different: `ModuleDict` entries go via `m[key] = module` / `put()`, not `add_module()`; no `values()`/`items()` |
| `nn.Parameter(tensor)` | `lt.nn.Parameter(tensor)` | BROKEN on 88a4ba1 — always throws (D-25); obtain parameters from modules instead |
| `nn.Linear` / `Bilinear` | identical names | identical (Linear); Bilinear is multi-arg — use `.forward()` (D-22) |
| `nn.Conv1d/2d/3d` / `ConvTranspose1d/2d/3d` | identical names | identical |
| `nn.MaxPool2d/3d` / `AvgPool2d` / `AdaptiveAvgPool2d` / `Fold` / `Unfold` / `PixelShuffle` / `PixelUnshuffle` | identical names | identical |
| `nn.LayerNorm` / `RMSNorm` / `BatchNorm1d/2d/3d` / `GroupNorm` / `InstanceNorm1d/2d/3d` | identical names | identical; `SyncBatchNorm` is channel-only: rank-2 (N, C) raises, rank-3 (N, C, L) works |
| `nn.ReLU/ReLU6/LeakyReLU/Sigmoid/Tanh/GELU/SiLU/ELU/CELU/SELU/Mish/Softplus/Softshrink/Softsign/Hardshrink/Hardtanh/Threshold/Tanhshrink/PReLU/RReLU/GLU/LogSoftmax/Softmax` | identical names | identical (GELU is tanh-only, D-09) |
| `nn.Embedding` / `EmbeddingBag` | identical names | identical; `VocabParallelEmbedding` forward is BROKEN (D-39) |
| `nn.RNN/RNNCell/LSTM/LSTMCell/GRU/GRUCell` | identical names | identical |
| `nn.MultiHeadAttention` | identical name | different: multi-arg — use `.forward(q, k, v)` (D-22) |
| `nn.TransformerEncoderLayer` / `TransformerDecoderLayer` / `TransformerEncoder` | identical names | different: decoder output differs from torch by 0.12–0.14 (D-29); encoder parity is eval-mode only; decoder ctor is `(embed_dim, num_heads, dim_feedforward=2048)` |
| `nn.GPTBlock` / `nn.GPT2B` | identical names | LiteTorch-only GPT blocks; GPT2B is bound |
| `nn.MSELoss` / `L1Loss` / `BCELoss` / `BCEWithLogitsLoss` / `NLLLoss` / `CrossEntropyLoss` / `HuberLoss` / `SmoothL1Loss` / `KLDivLoss` / `CTCLoss` / `DPOLoss` / `TripletMarginLoss` / `HingeEmbeddingLoss` / `MultiMarginLoss` / `GaussianNLLLoss` / `PoissonNLLLoss` / `MarginRankingLoss` / `MultiLabelMarginLoss` / `SoftMarginLoss` / `CosineEmbeddingLoss` | identical names | different: loss targets must be INT32 (D-23); HingeEmbeddingLoss formula differs (D-26); MultiMarginLoss normalization differs, `p` is an int (D-27); CTCLoss mean differs (D-28) |
| `F.linear` / `F.dropout` / `F.pad` / `F.interpolate` / `F.nll_loss` / `F.bce_with_logits` / `F.normalize` / `F.glu` / `F.elu` / `F.dpo_loss` | `lt.nn.functional.*` (same names) | identical for these 10; every other `F.*` is absent — use `Ops.*` or the module |
| `nn.init.kaiming_uniform_` / `xavier_uniform_` / `normal_` / `uniform_` / `zeros_` / `ones_` / `fill_` / `trunc_normal_` / `orthogonal_` | `lt.nn.init.*` (same names) | identical |
| `nn.utils.clip_grad_value_` / `parameters_to_vector` / `vector_to_parameters` / `weight_norm` / `spectral_norm` | `lt.nn.utils.*` (same names) | identical; `clip_grad_norm_` lives under `Ops` |
| `model.state_dict()` / `load_state_dict(sd)` | identical names | identical |
| `model.to("cuda")` | `model.to(lt.Device("cuda"))` | different: Device only, no string form (D-03) |

## Optimizers and schedulers

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.optim.SGD` | `lt.optim.SGD(params, lr=0.01, momentum=0.0, ...)` | identical (verified 1e-7) |
| `torch.optim.Adam` / `AdamW` | `lt.optim.Adam` / `AdamW` | identical |
| `torch.optim.RMSprop` / `Adadelta` / `Adamax` / `Adagrad` | identical names | identical |
| `torch.optim.NAdam` | `lt.optim.NAdam` | BROKEN — first-step update ~6.5x too large (D-30); use Adam/AdamW |
| `torch.optim.LBFGS` | absent | no equivalent |
| `torch.optim.AdamW` 8-bit / FP8 variants | `lt.optim.AdamW8bit` / `AdamWFP8` | LiteTorch-only; identical to torch AdamW to 6 decimals on the 1-step probe |
| `torch.optim.lr_scheduler.StepLR` / `CosineAnnealingLR` / `CosineAnnealingWarmRestarts` / `LinearLR` / `PolynomialLR` / `LambdaLR` / `MultiStepLR` / `ExponentialLR` / `ReduceLROnPlateau` | identical names under `lt.optim` | identical (all verified to match torch exactly) |
| `torch.optim.lr_scheduler.OneCycleLR` | `lt.optim.OneCycleLR` | different: one-step phase shift (D-31) |
| `torch.cuda.amp.GradScaler` | `lt.optim.GradScaler` / `lt.amp.GradScaler` | identical (overflow-skip verified) |
| `torch.cuda.amp.autocast` | `lt.amp.autocast` / `lt.amp.AutocastGuard` | identical |

## Data loading

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.utils.data.Dataset` | `lt.data.Dataset` | different: subclass defines `size()` / `get(idx)`, NOT `__len__` / `__getitem__` (D-37) |
| `torch.utils.data.TensorDataset` | `lt.data.TensorDataset` | identical |
| `torch.utils.data.DataLoader` | `lt.data.DataLoader` | different: `shuffle=True` is not seeded by `manual_seed` (D-46); a Python `collate_fn` triggers the D-37 shutdown abort |
| `torch.utils.data.Sampler` / `SequentialSampler` / `RandomSampler` / `BatchSampler` | `lt.data.Sampler` etc. | different: samplers are not iterable — `indices()` / `batches()` only |
| `torch.utils.data.Subset` / `random_split` | `lt.data.Subset` / `lt.data.random_split` | identical |
| `torch.save(state_dict, path)` | `lt.save(module, path)` | different: takes a Module/list/Optimizer, not a state dict (D-45) |
| `torch.load(path)` | `lt.load(...)` | different: in-place, type-dispatched; a bare tensor throws (D-45) |
| `torch.save` optimizer state | `lt.save_optimizer_state` | different: Adam/AdamW/SGD/RMSprop only (D-45) |
| `safetensors.torch.save_file` | `lt.save_safetensors(dict)` / `lt.load_safetensors` | different: takes a dict, not a module (D-45) |
| `transformers.Trainer` | `lt.Trainer` | different: `train_step()` runs no forward/backward; checkpoints are async by default (D-44) |

## Distributed

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.distributed.init_process_group` | `lt.distributed.init_process_group` | identical (must be called first) |
| `torch.distributed.destroy_process_group` | absent | use `lt.distributed.shutdown()` |
| `torch.distributed.barrier` / `all_reduce` / `all_gather` / `reduce_scatter` / `broadcast` / `all_to_all` | identical names | identical |
| `torch.nn.parallel.DistributedDataParallel` | `lt.distributed.DistributedDataParallel` | FIXED (D-33) — pre-fix backward doubled one grad slice; fixed by dist-w20 (@39e53ee0), Track 5 gate PASS 2026-10-31 |
| `torch.distributed.fsdp.FullyShardedDataParallel` | absent as a class | use `lt.distributed.fsdp_fully_shard(module)`; FSDP shards params + grads, NOT optimizer states; `FSDPUnit` methods are unreachable from Python |
| `torch.distributed.tensor` / `DeviceMesh` / `DTensor` | `lt.distributed.DeviceMesh` / `DTensor` / `Placement` | identical in outline; see `reference/distributed.md` |
| `torch.distributed` NCCL backend | `lt.distributed.NCCLBridge` | plain dlopen (`libnccl.so` → `librccl.so`), fail-soft; the `LITETORCH_NCCL_LIB` override is a Kaggle-verified patch, not yet merged (dist-w15) |
| `torchrun` / rendezvous | `lt.distributed.RendezvousStore` / `RendezvousClient` | different: join the `run()` thread after `stop()` or the process aborts at exit (D-35) |

## AMP, JIT, quantization, serving

| PyTorch | LiteTorch | Notes |
|---|---|---|
| `torch.cuda.amp.autocast` / `GradScaler` | `lt.amp.autocast` / `AutocastGuard` / `GradScaler` | identical |
| `torch.jit.script` / `torch.jit.trace` | `lt.jit.Tracer.trace` → `JITFunction` | different: `trace` returns `JITFunction`, not `JITVar`; plain `+` intermediates raise the expired-intermediate error (use `lt.Ops.*`) |
| `torch.jit.fuse` / fused ops | `lt.jit.fused_bias_relu` / `fused_bias_gelu` / `fused_residual_add` / `fused_residual_gelu` | different: fused bias ops require identical shapes (no broadcasting) |
| `torch.quantization.quantize_dynamic` | `lt.quantization.quantize_dynamic_linear` / `quantize_dynamic_lstm` / `quantize_dynamic_gru` | renamed; different API shape |
| `torch.quantization.prepare` / `convert` / QAT | `lt.quantization.prepare` / `convert` / `PreparedModel` / `QATLinear` / `Calibrator` / `PTQConfig` | renamed; `calibrate` takes a zero-arg callable `data_fn`, not a list |
| `torch.onnx.export` | `lt.onnx.export` / `export_dynamic` / `export_onnx_text` / `export_onnx_binary` | renamed |
| `transformers.AutoModel.from_pretrained` | `lt.hf.from_pretrained` | renamed; `hf.load_config`, `hf.remap_state_dict`, `hf.weight_name_map` |
| `torch.cuda.CUDAGraph` | `lt.inference.CudaGraph` / `capture_cuda_graph` | renamed |
| `torch.nn.utils.prune` | `lt.prune.l1_unstructured` / `random_unstructured` / `remove` / `is_pruned` | renamed |
| `torch.profiler` | `lt.profiler.enable` / `disable` / `report` / `record` / `profile` | renamed |
| `torch.hub.load_state_dict_from_url` | `lt.hub.load_state_dict_from_url` / `download_url_to_file` | identical |
| vLLM-style serving | `lt.Ops.paged_attention` (6 positional args), `ServingScheduler.step(model)`, `PrefixCache` (list form), `MedusaHead`, `SpeculativeEngine` | LiteTorch-only; no torch equivalent |
| `torchvision` / `torchaudio` / `torchtext` | `ltvision` / `ltaudio` / `lttext` / `ltmobile` / `ltgeometric` / `ltrec` / `ltrrl` | renamed; different APIs — see `reference/subpackages.md` |

## C++ API

The C++ API (`include/litetorch/`, see `reference/cpp-api.md`) is a separate surface with its own rules: `Tensor::data_ptr()` throws on non-FP32 (D-01), `Ops::reduce_broadcast` / `Ops::from_data` are C++-only, and the CMake build layout is documented in the C++ reference. There is no PyTorch C++ (libtorch) mapping table — port C++ code function by function against `reference/cpp-api.md`.

## Common pitfalls checklist

Before running a ported training job, check these — they are the failures that do not raise:

1. Device: verify `t.device` after `.to()` (D-02). A "GPU" job silently running on CPU is the most common silent failure.
2. Module registration: `add_module()` or an explicit `parameters()` override (D-24). An empty `parameters()` means the optimizer steps over nothing.
3. Loss targets: INT32, not INT64 (D-23). `.long()` now returns INT64 (D-06) — and the losses reject it.
4. Gradients: compare grads, not just forward values (the no-learn lesson). DDP doubles a grad slice (D-33); NAdam is ~6.5x off (D-30).
5. DataLoader: `shuffle=False` for reproducibility (D-46); `size()`/`get()` for custom datasets (D-37).
6. GELU: tolerance 5e-4 against torch default, or compare against `approximate='tanh'` (D-09).
7. gradcheck: pass `eps=1e-3` for FP32 (D-34).
