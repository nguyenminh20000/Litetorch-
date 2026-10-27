# nn modules — API reference

All classes live under `litetorch.nn`. Signatures are taken from the pybind11 bindings (`src/bindings/bind_nn_base.cpp`, `bind_nn_layers.cpp`, `bind_nn_recurrent.cpp`, `bind_nn_utils.cpp`); the binding is the source of truth. Every example is self-contained, runs on CPU, and carries a verification header (see the verification plan, `plan/verification.md` §4).

Two call-shape rules apply to every multi-argument module in this file. First, `__call__()` only routes the single-tensor `forward(x)`: a module whose forward takes two or more tensors — `MultiHeadAttention`, `Bilinear`, `TripletMarginWithDistanceLoss`, `GaussianNLLLoss`, `CTCLoss`, `DPOLoss`, and every two-argument loss — must be called through `.forward(...)` explicitly; calling it with two arguments raises `TypeError`. Second, `Module.to()` takes a `litetorch.Device` only, never a string (D-03).
## Containers and base
### `nn.Module`

- **Signature:** `Module()`
- **Source:** `bind_nn_base.cpp:6`
- **Description:** Base class for all modules. Subclass it and override forward(). Holds parameters, buffers, and child modules; drives train()/eval() mode, .to(device), and state_dict() save/load.
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** forward() is virtual in C++; a Python subclass must define forward(). .to() takes a litetorch.Device only, no string form (D-03). CRITICAL (new deviation, D-24 candidate): assigning a submodule as a Python attribute (self.linear = lt.nn.Linear(4, 4)) does NOT register it — parameters() returns 0 and state_dict() is empty. Register children explicitly with add_module('linear', module); after add_module, parameters() finds them. Torch registers attribute-assigned submodules automatically.
- **PyTorch:** torch.nn.Module — equivalent core API. LiteTorch adds compile(), save_pretrained()/load_pretrained(). Python subclassing works through the PyModule trampoline — with one major exception (see the edge case).
- **Example:**

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

- **Signature:** `Sequential() | Sequential(modules=[...])`
- **Source:** `bind_nn_base.cpp:296`
- **Description:** Ordered container of modules, applied in order. Accepts a list of modules, or empty plus .add().
- **Parameters:** Overloads: `Sequential()` or `Sequential(modules=[...])` where modules is a list of modules.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Two constructor overloads: Sequential() and Sequential([modules]).
- **PyTorch:** torch.nn.Sequential — equivalent. LiteTorch also accepts .add(module) after construction.
- **Example:**

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

- **Signature:** `ModuleList(modules)`
- **Source:** `bind_nn_layers.cpp:85`
- **Description:** List container of modules. Indexing and iteration work; modules are registered as children so parameters() finds them.
- **Parameters:** `modules` (required).
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** A plain Python list does NOT register parameters; use ModuleList (or ModuleDict) inside a custom Module.
- **PyTorch:** torch.nn.ModuleList — equivalent.
- **Example:**

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

- **Signature:** `ModuleDict()`
- **Source:** `bind_nn_layers.cpp:107`
- **Description:** Dict container of modules keyed by string. Supports keys(), values(), items(), __contains__, __iter__.
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Add entries with m[key] = module or put(key, module); add_module() does NOT add to the dict (getitem then raises KeyError). Supports keys(), get(key), __len__, __contains__, __iter__; no values()/items().
- **PyTorch:** torch.nn.ModuleDict — equivalent in effect, but entries are added with put(key, module) or m[key] = module, NOT with add_module() (which registers a child but does not add it to the dict). No values() or items() methods are bound.
- **Example:**

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

- **Signature:** `Parameter(data, requires_grad=true)`
- **Source:** `bind_nn_layers.cpp:70`
- **Description:** A tensor registered as a learnable parameter. Created from an existing tensor; requires_grad is True.
- **Parameters:** `data` (required), `requires_grad` = true.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** BROKEN on 88a4ba1: Parameter(tensor) always throws 'copy_: a leaf Tensor that requires grad is being used in an in-place operation', even for requires_grad=False inputs (the binding copy_()s into the Parameter itself). No workaround exists; obtain parameters from modules (for example Linear.weight) instead. Flagged for Core.
- **PyTorch:** torch.nn.Parameter — equivalent in intent, but see the edge case: construction currently throws on this build.
- **Example:**

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

- **Signature:** `Buffer(data)`
- **Source:** `bind_nn_layers.cpp:78`
- **Description:** A tensor registered as a persistent non-learnable buffer (for example running statistics). Moves with .to(device) but is not a parameter.
- **Parameters:** `data` (required).
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** The binding copies the input tensor; later mutation of the source does not affect the buffer.
- **PyTorch:** torch.nn.Buffer — equivalent.
- **Example:**

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

- **Signature:** `Identity()`
- **Source:** `bind_nn_recurrent.cpp:247`
- **Description:** Returns its input unchanged. Useful as a placeholder layer.
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Takes no constructor arguments.
- **PyTorch:** torch.nn.Identity — equivalent.
- **Example:**

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

- **Signature:** `Flatten()`
- **Source:** `bind_nn_base.cpp:191`
- **Description:** Flattens all dimensions except the batch dimension: (N, ...) -> (N, prod).
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** No start_dim/end_dim: always flattens from dimension 1.
- **PyTorch:** torch.nn.Flatten — equivalent for the default case, but LiteTorch takes NO start_dim/end_dim arguments; the flatten range is fixed.
- **Example:**

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

- **Signature:** `Unflatten(dim, unflattened_size)`
- **Source:** `bind_nn_recurrent.cpp:251`
- **Description:** Unflattens one dimension into several: (N, d, ...) -> (N, *unflattened_size, ...) at the given dim.
- **Parameters:** `dim` (required), `unflattened_size` (required).
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** dim is a single int64; unflattened_size is a list of ints whose product must equal the size of dim.
- **PyTorch:** torch.nn.Unflatten — equivalent.
- **Example:**

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

- **Signature:** `Dropout(p=0.5)`
- **Source:** `bind_nn_base.cpp:160`
- **Description:** Randomly zeroes each element with probability p during training; identity in eval mode. Output is scaled by 1/(1-p) in training.
- **Parameters:** `p` = 0.5f.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** In training mode the output is stochastic: seed with lt.manual_seed() for reproducibility, or call eval() for a deterministic pass. Default p=0.5 matches torch.
- **PyTorch:** torch.nn.Dropout — equivalent, including the default p=0.5.
- **Example:**

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

- **Signature:** `Linear(in_features, out_features, bias=true)`
- **Source:** `bind_nn_base.cpp:19`
- **Description:** Applies y = x @ W.T + b. Weight shape (out_features, in_features); bias shape (out_features,) unless bias=False.
- **Parameters:** `in_features` (required), `out_features` (required), `bias` = true.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** Input (..., in_features) -> (..., out_features). Weights are initialized randomly; use nn.init.* to re-initialize.
- **PyTorch:** torch.nn.Linear — equivalent.
- **Example:**

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

- **Signature:** `Bilinear(in1_features, in2_features, out_features, bias=true)`
- **Source:** `bind_nn_utils.cpp:153`
- **Description:** Bilinear form y = x1 @ W @ x2 + b with a 3-D weight (out_features, in1_features, in2_features). Takes TWO inputs.
- **Parameters:** `in1_features` (required), `in2_features` (required), `out_features` (required), `bias` = true.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** Two-input module: __call__() only routes the single-tensor forward, so call .forward(x1, x2) explicitly (see the multi-argument call note).
- **PyTorch:** torch.nn.Bilinear — equivalent.
- **Example:**

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

- **Signature:** `ColumnParallelLinear(in_features, out_features, has_bias=true)`
- **Source:** `bind_nn_base.cpp:49`
- **Description:** Tensor-parallel linear that shards the weight along output columns (Megatron-style). Used under tensor parallelism; on a single device it behaves like Linear.
- **Parameters:** `in_features` (required), `out_features` (required), `has_bias` = true.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** Parameter name is has_bias (not bias). On a single process the output equals Linear with the same weight.
- **PyTorch:** torch does not ship this; Megatron-LM ColumnParallelLinear is the reference. LiteTorch-only module.
- **Example:**

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

- **Signature:** `RowParallelLinear(in_features, out_features, has_bias=true)`
- **Source:** `bind_nn_base.cpp:57`
- **Description:** Tensor-parallel linear that shards the weight along input rows (Megatron-style); typically follows a ColumnParallelLinear and all-reduces the partial outputs.
- **Parameters:** `in_features` (required), `out_features` (required), `has_bias` = true.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** Parameter name is has_bias (not bias).
- **PyTorch:** Megatron-LM RowParallelLinear is the reference. LiteTorch-only module.
- **Example:**

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

- **Signature:** `VocabParallelEmbedding(vocab_size, embed_dim, padding_idx=-1)`
- **Source:** `bind_nn_base.cpp:81`
- **Description:** Tensor-parallel embedding that shards the vocabulary across ranks.
- **Parameters:** `vocab_size` (required), `embed_dim` (required), `padding_idx` = -1.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** BROKEN on 88a4ba1: forward() throws '[litetorch Error] data_ptr(): tensor dtype is not FP32' for both INT64 and FP32 index tensors (the lookup reads indices through data_ptr(), D-01). Construction works; forward does not. Flagged for Core.
- **PyTorch:** Megatron-LM VocabParallelEmbedding is the reference. LiteTorch-only module.
- **Example:**

<!-- ex: id=r7-vocab-parallel-embedding-1 device=cpu backend=cpu runtime=~0.7s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.VocabParallelEmbedding(10, 4)
print("constructed:", type(m).__name__)
print("forward() currently throws on this build: data_ptr() requires FP32 (D-01); see the edge case above.")
```
### `nn.MoELinear`

- **Signature:** `MoELinear(in_features, out_features, num_experts, top_k)`
- **Source:** `bind_nn_base.cpp:39`
- **Description:** Mixture-of-experts linear: routes each token to its top_k of num_experts expert linears.
- **Parameters:** `in_features` (required), `out_features` (required), `num_experts` (required), `top_k` (required).
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** num_experts and top_k have no defaults. Input (..., in_features) -> (..., out_features).
- **PyTorch:** No direct torch.nn equivalent (torch has no built-in MoE linear); vLLM/Megatron MoE layers are the conceptual reference.
- **Example:**

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

- **Signature:** `QLoRALinear(in_features, out_features, r=8, lora_alpha=16.0, has_bias=true)`
- **Source:** `bind_nn_base.cpp:28`
- **Description:** Quantized LoRA linear: a frozen 4-bit base weight plus trainable low-rank adapters (rank r, scaling lora_alpha).
- **Parameters:** `in_features` (required), `out_features` (required), `r` = 8, `lora_alpha` = 16.0f, `has_bias` = true.
- **Input:** `(..., in_features)` → **Output:** `(..., out_features)`
- **Edge cases:** r=8 and lora_alpha=16.0 by default; has_bias=True by default.
- **PyTorch:** No direct torch.nn equivalent; bitsandbytes QLoRA is the conceptual reference.
- **Example:**

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
## Convolution and pooling
### `nn.Conv1d`

- **Signature:** `Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Source:** `bind_nn_layers.cpp:254`
- **Description:** 1-D convolution over (N, C_in, L). kernel_size/stride/padding accept an int or a single-element tuple.
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** No dilation/groups arguments (the binding takes scalar stride/padding only). Weight shape (C_out, C_in, K).
- **PyTorch:** torch.nn.Conv1d — equivalent for scalar args, except LiteTorch has no dilation or groups arguments.
- **Example:**

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

- **Signature:** `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Source:** `bind_nn_base.cpp:93`
- **Description:** 2-D convolution over (N, C_in, H, W). kernel_size/stride/padding accept an int or a 2-element list.
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** No dilation/groups arguments. Weight shape (C_out, C_in, K_h, K_w).
- **PyTorch:** torch.nn.Conv2d — equivalent for scalar/list args, except LiteTorch has no dilation or groups arguments. The fused variant conv2d_relu lives under Ops, not nn.
- **Example:**

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

- **Signature:** `Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=true)`
- **Source:** `bind_nn_base.cpp:115`
- **Description:** 3-D convolution over (N, C_in, D, H, W). kernel_size/stride/padding are scalar ints only.
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `has_bias` = true.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Scalar stride/padding only; no tuple form (unlike Conv2d). Weight shape (C_out, C_in, K, K, K).
- **PyTorch:** torch.nn.Conv3d — equivalent for scalar args, except LiteTorch has no dilation or groups arguments.
- **Example:**

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

- **Signature:** `ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Source:** `bind_nn_recurrent.cpp:140`
- **Description:** 1-D transposed convolution (learned upsampling) over (N, C_in, L).
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Unlike the forward convs, this class DOES take groups and dilation. Output length (L-1)*stride - 2*padding + kernel_size + output_padding.
- **PyTorch:** torch.nn.ConvTranspose1d — equivalent, including groups and dilation arguments (unlike Conv1d).
- **Example:**

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

- **Signature:** `ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Source:** `bind_nn_recurrent.cpp:151`
- **Description:** 2-D transposed convolution (learned upsampling) over (N, C_in, H, W).
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Unlike Conv2d, this class DOES take groups and dilation.
- **PyTorch:** torch.nn.ConvTranspose2d — equivalent, including groups and dilation arguments.
- **Example:**

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

- **Signature:** `ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=true, dilation=1)`
- **Source:** `bind_nn_recurrent.cpp:162`
- **Description:** 3-D transposed convolution (learned upsampling) over (N, C_in, D, H, W).
- **Parameters:** `in_channels` (required), `out_channels` (required), `kernel_size` (required), `stride` = 1, `padding` = 0, `output_padding` = 0, `groups` = 1, `bias` = true, `dilation` = 1.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Unlike Conv3d, this class DOES take groups and dilation.
- **PyTorch:** torch.nn.ConvTranspose3d — equivalent, including groups and dilation arguments.
- **Example:**

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

- **Signature:** `MaxPool2d(kernel_size, stride=-1, padding=0)`
- **Source:** `bind_nn_base.cpp:140`
- **Description:** 2-D max pooling over (N, C, H, W). stride defaults to -1, which means stride = kernel_size.
- **Parameters:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** stride=-1 means stride equals kernel_size. No ceil_mode / return_indices / dilation.
- **PyTorch:** torch.nn.MaxPool2d — equivalent. LiteTorch has no ceil_mode, return_indices, or dilation arguments.
- **Example:**

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

- **Signature:** `MaxPool3d(kernel_size, stride=-1, padding=0)`
- **Source:** `bind_nn_base.cpp:147`
- **Description:** 3-D max pooling over (N, C, D, H, W). stride defaults to -1, which means stride = kernel_size.
- **Parameters:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** stride=-1 means stride equals kernel_size. No ceil_mode / return_indices / dilation.
- **PyTorch:** torch.nn.MaxPool3d — equivalent. LiteTorch has no ceil_mode, return_indices, or dilation arguments.
- **Example:**

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

- **Signature:** `AvgPool2d(kernel_size, stride=-1, padding=0)`
- **Source:** `bind_nn_layers.cpp:276`
- **Description:** 2-D average pooling over (N, C, H, W). stride defaults to -1, which means stride = kernel_size.
- **Parameters:** `kernel_size` (required), `stride` = -1, `padding` = 0.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** stride=-1 means stride equals kernel_size. No ceil_mode / count_include_pad.
- **PyTorch:** torch.nn.AvgPool2d — equivalent for the basic case. LiteTorch has no ceil_mode or count_include_pad arguments.
- **Example:**

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

- **Signature:** `AdaptiveAvgPool2d(output_height, output_width)`
- **Source:** `bind_nn_base.cpp:154`
- **Description:** 2-D adaptive average pooling to a fixed output size, given as two separate ints (output_height, output_width).
- **Parameters:** `output_height` (required), `output_width` (required).
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Takes (output_height, output_width) as two ints, not an output_size tuple.
- **PyTorch:** torch.nn.AdaptiveAvgPool2d — equivalent in effect, but torch takes a single output_size (int or tuple); LiteTorch takes two separate int arguments.
- **Example:**

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

- **Signature:** `Fold(output_size, kernel_size, dilation=[1, padding=[0, stride=[1)`
- **Source:** `bind_nn_utils.cpp:157`
- **Description:** Combines an array of sliding local blocks into a large containing tensor (inverse of Unfold).
- **Parameters:** `output_size` (required), `kernel_size` (required), `dilation` = std::vector<int64_t>{1, `padding` = std::vector<int64_t>{0, `stride` = std::vector<int64_t>{1.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Input is (N, C * prod(kernel_size), L) where L is the number of sliding blocks; output_size and kernel_size are lists.
- **PyTorch:** torch.nn.Fold — equivalent.
- **Example:**

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

- **Signature:** `Unfold(kernel_size, dilation=[1, padding=[0, stride=[1)`
- **Source:** `bind_nn_utils.cpp:162`
- **Description:** Extracts sliding local blocks from a batched input tensor (im2col).
- **Parameters:** `kernel_size` (required), `dilation` = std::vector<int64_t>{1, `padding` = std::vector<int64_t>{0, `stride` = std::vector<int64_t>{1.
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Output is (N, C * prod(kernel_size), L). kernel_size is a list; dilation/padding/stride are lists with defaults.
- **PyTorch:** torch.nn.Unfold — equivalent.
- **Example:**

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

- **Signature:** `PixelShuffle(upscale_factor)`
- **Source:** `bind_nn_recurrent.cpp:237`
- **Description:** Rearranges (N, C*r^2, H, W) to (N, C, H*r, W*r) for sub-pixel upsampling.
- **Parameters:** `upscale_factor` (required).
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Input channels must be divisible by upscale_factor^2.
- **PyTorch:** torch.nn.PixelShuffle — equivalent.
- **Example:**

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

- **Signature:** `PixelUnshuffle(downscale_factor)`
- **Source:** `bind_nn_recurrent.cpp:242`
- **Description:** Inverse of PixelShuffle: rearranges (N, C, H*r, W*r) to (N, C*r^2, H, W).
- **Parameters:** `downscale_factor` (required).
- **Input:** `(N, C_in, ...) — see the example` → **Output:** `(N, C_out, ...) — see the example`
- **Edge cases:** Input height and width must be divisible by downscale_factor.
- **PyTorch:** torch.nn.PixelUnshuffle — equivalent.
- **Example:**

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
## Normalization
### `nn.LayerNorm`

- **Signature:** `LayerNorm(normalized_shape, eps=1e-5)`
- **Source:** `bind_nn_base.cpp:208`
- **Description:** Normalizes over the last normalized_shape dimensions: y = (x - mean) / sqrt(var + eps) * weight + bias.
- **Parameters:** `normalized_shape` (required), `eps` = 1e-5f.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** normalized_shape MUST be a list of ints (vector<int64_t>); a bare int is rejected. eps defaults to 1e-5.
- **PyTorch:** torch.nn.LayerNorm — equivalent. LiteTorch requires normalized_shape as a list; torch also accepts a bare int.
- **Example:**

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

- **Signature:** `RMSNorm(normalized_shape, eps=1e-5)`
- **Source:** `bind_nn_base.cpp:249`
- **Description:** Root-mean-square normalization over the last normalized_shape dimensions, without mean centering: y = x / sqrt(mean(x^2) + eps) * weight.
- **Parameters:** `normalized_shape` (required), `eps` = 1e-5f.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** normalized_shape MUST be a list of ints. No bias term.
- **PyTorch:** torch has no RMSNorm module (it is torch.nn.functional.rms_norm in recent versions). LiteTorch-only module.
- **Example:**

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

- **Signature:** `BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Source:** `bind_nn_layers.cpp:239`
- **Description:** Batch normalization over (N, C) or (N, C, L). Uses batch statistics in training, running statistics in eval.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm1d — equivalent, including the default eps=1e-5, momentum=0.1, affine=True, track_running_stats=True.
- **Example:**

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

- **Signature:** `BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Source:** `bind_nn_base.cpp:125`
- **Description:** Batch normalization over (N, C, H, W). Uses batch statistics in training, running statistics in eval.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm2d — equivalent.
- **Example:**

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

- **Signature:** `BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Source:** `bind_nn_recurrent.cpp:200`
- **Description:** Batch normalization over (N, C, D, H, W). Uses batch statistics in training, running statistics in eval.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** In training mode the batch must have more than one sample per channel for a meaningful variance. Remember to call eval() at inference time.
- **PyTorch:** torch.nn.BatchNorm3d — equivalent.
- **Example:**

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

- **Signature:** `SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)`
- **Source:** `bind_nn_recurrent.cpp:211`
- **Description:** Batch normalization that synchronizes statistics across processes in distributed training.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = true, `track_running_stats` = true.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** Channel BatchNorm: the input must be rank 3 (N, C, L) — a rank-2 (N, C) input raises 'channel BatchNorm: bad input rank or channels'. On a single process it behaves like BatchNorm1d. Cross-process sync requires a process group (see the distributed reference).
- **PyTorch:** torch.nn.SyncBatchNorm — equivalent in intent.
- **Example:**

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

- **Signature:** `GroupNorm(num_groups, num_channels, eps=1e-5, affine=true)`
- **Source:** `bind_nn_layers.cpp:227`
- **Description:** Normalizes over groups of channels: num_channels is split into num_groups groups, each normalized independently.
- **Parameters:** `num_groups` (required), `num_channels` (required), `eps` = 1e-5f, `affine` = true.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** num_channels must be divisible by num_groups. affine=True by default.
- **PyTorch:** torch.nn.GroupNorm — equivalent.
- **Example:**

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

- **Signature:** `InstanceNorm1d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Source:** `bind_nn_recurrent.cpp:173`
- **Description:** Instance normalization over (N, C, L): each (N, C) slice is normalized independently.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm1d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Example:**

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

- **Signature:** `InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Source:** `bind_nn_recurrent.cpp:182`
- **Description:** Instance normalization over (N, C, H, W): each (N, C) slice is normalized independently.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm2d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Example:**

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

- **Signature:** `InstanceNorm3d(num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false)`
- **Source:** `bind_nn_recurrent.cpp:191`
- **Description:** Instance normalization over (N, C, D, H, W): each (N, C) slice is normalized independently.
- **Parameters:** `num_features` (required), `eps` = 1e-5f, `momentum` = 0.1f, `affine` = false, `track_running_stats` = false.
- **Input:** `(N, C, ...) normalized over the feature dims` → **Output:** `(N, C, ...), same shape`
- **Edge cases:** Defaults differ from BatchNorm: affine=False and track_running_stats=False.
- **PyTorch:** torch.nn.InstanceNorm3d — equivalent, including the defaults affine=False, track_running_stats=False.
- **Example:**

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
## Activation
### `nn.ReLU`

- **Signature:** `ReLU()`
- **Source:** `bind_nn_base.cpp:165`
- **Description:** Rectified linear unit: max(0, x), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Gradient is 0 for x < 0 and 1 for x > 0.
- **PyTorch:** torch.nn.ReLU — equivalent.
- **Example:**

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

- **Signature:** `ReLU6()`
- **Source:** `bind_nn_utils.cpp:102`
- **Description:** ReLU capped at 6: min(max(0, x), 6), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters.
- **PyTorch:** torch.nn.ReLU6 — equivalent.
- **Example:**

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

- **Signature:** `LeakyReLU(negative_slope=0.01)`
- **Source:** `bind_nn_base.cpp:169`
- **Description:** Leaky ReLU: x for x >= 0, negative_slope * x otherwise.
- **Parameters:** `negative_slope` = 0.01f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** negative_slope defaults to 0.01.
- **PyTorch:** torch.nn.LeakyReLU — equivalent.
- **Example:**

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

- **Signature:** `Sigmoid()`
- **Source:** `bind_nn_base.cpp:174`
- **Description:** Sigmoid: 1 / (1 + exp(-x)), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Output range (0, 1).
- **PyTorch:** torch.nn.Sigmoid — equivalent.
- **Example:**

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

- **Signature:** `Tanh()`
- **Source:** `bind_nn_base.cpp:178`
- **Description:** Hyperbolic tangent, elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Output range (-1, 1).
- **PyTorch:** torch.nn.Tanh — equivalent.
- **Example:**

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

- **Signature:** `GELU()`
- **Source:** `bind_nn_base.cpp:182`
- **Description:** Gaussian error linear unit. LiteTorch uses the tanh approximation only; there is no approximate argument.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** GELU is tanh-approximation only (D-09). Compare against F.gelu(approximate='tanh') at parity time, or allow ~5e-4 tolerance against the torch default.
- **PyTorch:** torch.nn.GELU — differs: torch defaults to the exact erf form. LiteTorch matches torch.nn.functional.gelu(x, approximate='tanh') (D-09, max diff vs exact erf ~4.4e-4).
- **Example:**

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

- **Signature:** `SiLU()`
- **Source:** `bind_nn_layers.cpp:312`
- **Description:** Sigmoid linear unit: x * sigmoid(x), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Also bound as a Tensor method (.silu()).
- **PyTorch:** torch.nn.SiLU — equivalent.
- **Example:**

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

- **Signature:** `ELU(alpha=1.0)`
- **Source:** `bind_nn_recurrent.cpp:220`
- **Description:** Exponential linear unit: x for x > 0, alpha * (exp(x) - 1) otherwise.
- **Parameters:** `alpha` = 1.0f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** alpha defaults to 1.0.
- **PyTorch:** torch.nn.ELU — equivalent.
- **Example:**

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

- **Signature:** `CELU(alpha=1.0)`
- **Source:** `bind_nn_utils.cpp:110`
- **Description:** Continuously differentiable exponential linear unit: x for x > 0, alpha * (exp(x / alpha) - 1) otherwise.
- **Parameters:** `alpha` = 1.0f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** alpha defaults to 1.0.
- **PyTorch:** torch.nn.CELU — equivalent.
- **Example:**

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

- **Signature:** `SELU()`
- **Source:** `bind_nn_utils.cpp:114`
- **Description:** Scaled exponential linear unit with fixed scale (lambda ~1.0507) and alpha (~1.6733).
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Designed for self-normalizing networks; use with the matching initialization.
- **PyTorch:** torch.nn.SELU — equivalent.
- **Example:**

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

- **Signature:** `Mish()`
- **Source:** `bind_nn_utils.cpp:117`
- **Description:** Mish: x * tanh(softplus(x)), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters.
- **PyTorch:** torch.nn.Mish — equivalent.
- **Example:**

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

- **Signature:** `Softplus(beta=1.0, threshold=20.0)`
- **Source:** `bind_nn_utils.cpp:120`
- **Description:** Softplus: (1/beta) * log(1 + exp(beta * x)), capped at threshold for numerical stability.
- **Parameters:** `beta` = 1.0f, `threshold` = 20.0f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** beta defaults to 1.0, threshold to 20.0.
- **PyTorch:** torch.nn.Softplus — equivalent.
- **Example:**

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

- **Signature:** `Softshrink(lambd=0.5)`
- **Source:** `bind_nn_utils.cpp:125`
- **Description:** Soft shrinkage: x - lambd for x > lambd, x + lambd for x < -lambd, else 0.
- **Parameters:** `lambd` = 0.5f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** lambd defaults to 0.5.
- **PyTorch:** torch.nn.Softshrink — equivalent.
- **Example:**

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

- **Signature:** `Softsign()`
- **Source:** `bind_nn_utils.cpp:136`
- **Description:** Softsign: x / (1 + |x|), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters. Output range (-1, 1).
- **PyTorch:** torch.nn.Softsign — equivalent.
- **Example:**

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

- **Signature:** `Hardshrink(lambd=0.5)`
- **Source:** `bind_nn_utils.cpp:129`
- **Description:** Hard shrinkage: x for |x| > lambd, else 0.
- **Parameters:** `lambd` = 0.5f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** lambd defaults to 0.5.
- **PyTorch:** torch.nn.Hardshrink — equivalent.
- **Example:**

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

- **Signature:** `Hardtanh(min_val=-1.0, max_val=1.0)`
- **Source:** `bind_nn_utils.cpp:139`
- **Description:** Hard tanh: x clamped to [min_val, max_val].
- **Parameters:** `min_val` = -1.0f, `max_val` = 1.0f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** min_val defaults to -1.0, max_val to 1.0.
- **PyTorch:** torch.nn.Hardtanh — equivalent.
- **Example:**

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

- **Signature:** `Threshold(threshold, value)`
- **Source:** `bind_nn_utils.cpp:144`
- **Description:** Threshold: x for x > threshold, value otherwise.
- **Parameters:** `threshold` (required), `value` (required).
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** threshold and value are required (no defaults).
- **PyTorch:** torch.nn.Threshold — equivalent.
- **Example:**

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

- **Signature:** `Tanhshrink()`
- **Source:** `bind_nn_utils.cpp:133`
- **Description:** Tanh shrinkage: x - tanh(x), elementwise.
- **Parameters:** None.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** No parameters.
- **PyTorch:** torch.nn.Tanhshrink — equivalent.
- **Example:**

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

- **Signature:** `PReLU(num_parameters=1, init=0.25f)`
- **Source:** `bind_nn_recurrent.cpp:225`
- **Description:** Parametric ReLU: x for x >= 0, a * x otherwise, where a is a learnable parameter (one per channel or a single shared value).
- **Parameters:** `num_parameters` = 1, `init` = 0.25f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** num_parameters defaults to 1, init to 0.25.
- **PyTorch:** torch.nn.PReLU — equivalent.
- **Example:**

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

- **Signature:** `RReLU(lower=0.125f, upper=0.3333333333333333f)`
- **Source:** `bind_nn_utils.cpp:105`
- **Description:** Randomized leaky ReLU: the negative slope is sampled uniformly from [lower, upper] in training and fixed to (lower + upper) / 2 in eval.
- **Parameters:** `lower` = 0.125f, `upper` = 0.3333333333333333f.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** lower defaults to 0.125, upper to 1/3. Stochastic in training mode: use eval() for a deterministic pass.
- **PyTorch:** torch.nn.RReLU — equivalent.
- **Example:**

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

- **Signature:** `GLU(dim=-1)`
- **Source:** `bind_nn_recurrent.cpp:232`
- **Description:** Gated linear unit: splits the input in half along dim and returns a * sigmoid(b).
- **Parameters:** `dim` = -1.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** dim defaults to -1. The size of dim must be even.
- **PyTorch:** torch.nn.GLU — equivalent.
- **Example:**

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

- **Signature:** `LogSoftmax(dim=-1)`
- **Source:** `bind_nn_layers.cpp:316`
- **Description:** Log-softmax along dim: log(softmax(x)).
- **Parameters:** `dim` = -1.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** dim defaults to -1. Also bound as a Tensor method (.log_softmax()).
- **PyTorch:** torch.nn.LogSoftmax — equivalent.
- **Example:**

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

- **Signature:** `Softmax(dim=-1)`
- **Source:** `bind_nn_base.cpp:186`
- **Description:** Softmax along dim.
- **Parameters:** `dim` = -1.
- **Input:** `(*)` → **Output:** `(*), same shape`
- **Edge cases:** dim defaults to -1.
- **PyTorch:** torch.nn.Softmax — equivalent.
- **Example:**

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

- **Signature:** `Embedding(num_embeddings, embedding_dim, padding_idx=py::none()`
- **Source:** `bind_nn_base.cpp:195`
- **Description:** Lookup table: maps integer token IDs to dense vectors of size embedding_dim.
- **Parameters:** `num_embeddings` (required), `embedding_dim` (required), `padding_idx` = py::none(.
- **Input:** `(N,) integer tensor` → **Output:** `(N, embedding_dim) for Embedding; (1, embedding_dim) for EmbeddingBag (mean mode)`
- **Edge cases:** Input must be an integer tensor (INT64). The weight is a learnable parameter of shape (num_embeddings, embedding_dim).
- **PyTorch:** torch.nn.Embedding — equivalent. padding_idx defaults to no padding (None).
- **Example:**

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

- **Signature:** `EmbeddingBag(num_embeddings, embedding_dim, mode="mean")`
- **Source:** `bind_nn_recurrent.cpp:129`
- **Description:** Embedding bag: sums or averages the embeddings of a bag of token IDs (mode 'mean' by default).
- **Parameters:** `num_embeddings` (required), `embedding_dim` (required), `mode` = "mean".
- **Input:** `(N,) integer tensor` → **Output:** `(N, embedding_dim) for Embedding; (1, embedding_dim) for EmbeddingBag (mean mode)`
- **Edge cases:** mode defaults to 'mean'. Input is a 1-D index tensor; offsets are not exposed in this binding.
- **PyTorch:** torch.nn.EmbeddingBag — equivalent for the default (offsets-free) call.
- **Example:**

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
## Recurrent
### `nn.RNNCell`

- **Signature:** `RNNCell(input_size, hidden_size, bias=true, nonlinearity="tanh")`
- **Source:** `bind_nn_recurrent.cpp:6`
- **Description:** One step of a vanilla RNN: h' = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh).
- **Parameters:** `input_size` (required), `hidden_size` (required), `bias` = true, `nonlinearity` = "tanh".
- **Input:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Output:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Edge cases:** nonlinearity defaults to 'tanh' ('relu' is the other option). forward(x) uses a zero hidden state; forward(x, h) continues from h.
- **PyTorch:** torch.nn.RNNCell — equivalent. The single-argument forward uses a zero initial hidden state.
- **Example:**

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

- **Signature:** `LSTMCell(input_size, hidden_size, bias=true)`
- **Source:** `bind_nn_recurrent.cpp:22`
- **Description:** One step of an LSTM: returns (h', c') given input x and previous (h, c).
- **Parameters:** `input_size` (required), `hidden_size` (required), `bias` = true.
- **Input:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Output:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Edge cases:** forward(x) uses zero (h, c); forward(x, h, c) continues from the given state. The 1-arg call returns a (h, c) tuple.
- **PyTorch:** torch.nn.LSTMCell — equivalent.
- **Example:**

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

- **Signature:** `GRUCell(input_size, hidden_size, bias=true)`
- **Source:** `bind_nn_recurrent.cpp:35`
- **Description:** One step of a GRU: h' = (1 - z) * n + z * h.
- **Parameters:** `input_size` (required), `hidden_size` (required), `bias` = true.
- **Input:** `(batch, input_size), (batch, hidden_size) [+ (batch, hidden_size) for LSTM]` → **Output:** `(batch, hidden_size) [+ (batch, hidden_size) for LSTM]`
- **Edge cases:** forward(x) uses a zero hidden state; forward(x, h) continues from h.
- **PyTorch:** torch.nn.GRUCell — equivalent.
- **Example:**

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

- **Signature:** `RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Source:** `bind_nn_recurrent.cpp:50`
- **Description:** Multi-layer vanilla RNN over a sequence. forward(x) returns (output, h_n); forward_full(x, h0) takes an explicit initial state.
- **Parameters:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `nonlinearity` = "tanh", `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Input:** `(seq, batch, input_size)` → **Output:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Edge cases:** batch_first defaults to False. forward() returns a tuple (output, h_n) — unpack it. dropout applies between layers only when num_layers > 1.
- **PyTorch:** torch.nn.RNN — equivalent, including the (output, h_n) return. batch_first defaults to False: input is (seq, batch, input_size).
- **Example:**

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

- **Signature:** `LSTM(input_size, hidden_size, num_layers=1, bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Source:** `bind_nn_recurrent.cpp:68`
- **Description:** Multi-layer LSTM over a sequence. forward(x) returns (output, (h_n, c_n)).
- **Parameters:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Input:** `(seq, batch, input_size)` → **Output:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Edge cases:** batch_first defaults to False. forward() returns (output, (h_n, c_n)) — unpack the nested tuple.
- **PyTorch:** torch.nn.LSTM — equivalent, including the nested (h_n, c_n) return. batch_first defaults to False.
- **Example:**

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

- **Signature:** `GRU(input_size, hidden_size, num_layers=1, bias=true, batch_first=false, dropout=0.0, bidirectional=false)`
- **Source:** `bind_nn_recurrent.cpp:87`
- **Description:** Multi-layer GRU over a sequence. forward(x) returns (output, h_n).
- **Parameters:** `input_size` (required), `hidden_size` (required), `num_layers` = 1, `bias` = true, `batch_first` = false, `dropout` = 0.0f, `bidirectional` = false.
- **Input:** `(seq, batch, input_size)` → **Output:** `(seq, batch, hidden_size) output + (num_layers, batch, hidden_size) hidden [+ cell state for LSTM]`
- **Edge cases:** batch_first defaults to False. forward() returns a tuple (output, h_n) — unpack it.
- **PyTorch:** torch.nn.GRU — equivalent. batch_first defaults to False.
- **Example:**

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
## Attention, transformer, and GPT
### `nn.MultiHeadAttention`

- **Signature:** `MultiHeadAttention(embed_dim, num_heads, dropout=0.0, bias=true, kdim=-1, vdim=-1, batch_first=false)`
- **Source:** `bind_nn_base.cpp:218`
- **Description:** Multi-head self/cross attention: softmax(QK^T / sqrt(d_k)) V with learned projections. The 3-argument call must go through .forward(q, k, v).
- **Parameters:** `embed_dim` (required), `num_heads` (required), `dropout` = 0.0, `bias` = true, `kdim` = -1, `vdim` = -1, `batch_first` = false.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** CRITICAL: __call__() only routes the single-tensor forward, so m(q, k, v) raises TypeError. Call m.forward(q, k, v) instead. batch_first defaults to False, so inputs are (seq, batch, embed_dim). kdim/vdim default to -1 (same as embed_dim).
- **PyTorch:** torch.nn.MultiheadAttention — equivalent in effect, but the call shape differs: torch allows m(q, k, v) while LiteTorch requires m.forward(q, k, v) (see the edge case). D-20: there is no F.scaled_dot_product_attention; use Ops.flash_attention for the fused kernel.
- **Example:**

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

- **Signature:** `TPAttention(embed_dim, num_heads, n_kv_heads=-1, has_bias=true)`
- **Source:** `bind_nn_base.cpp:66`
- **Description:** Tensor-parallel multi-head attention with grouped-query support (n_kv_heads may differ from num_heads).
- **Parameters:** `embed_dim` (required), `num_heads` (required), `n_kv_heads` = -1, `has_bias` = true.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** n_kv_heads defaults to -1 (same as num_heads). Input is (seq, batch, embed_dim).
- **PyTorch:** No direct torch.nn equivalent; Megatron-style grouped-query attention is the reference. LiteTorch-only module.
- **Example:**

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

- **Signature:** `TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=false)`
- **Source:** `bind_nn_recurrent.cpp:104`
- **Description:** One transformer encoder layer: self-attention, feed-forward, residual connections, and layer norms.
- **Parameters:** `d_model` (required), `nhead` (required), `dim_feedforward` = 2048, `dropout` = 0.1f, `activation` = "relu", `batch_first` = false.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** d_model and nhead are required; dim_feedforward defaults to 2048, dropout to 0.1, activation to 'relu', batch_first to False.
- **PyTorch:** torch.nn.TransformerEncoderLayer — equivalent for the basic case.
- **Example:**

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

- **Signature:** `TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=2048)`
- **Source:** `bind_nn_base.cpp:242`
- **Description:** One transformer decoder layer: masked self-attention, cross-attention, feed-forward, residuals, and norms.
- **Parameters:** `embed_dim` (required), `num_heads` (required), `dim_feedforward` = 2048.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** DIVERGES from torch (D-29 candidate): two-argument forward differs from torch with identical weights (max diff 0.1245, eval mode). embed_dim and num_heads are required (unlike TransformerEncoderLayer, which takes d_model/nhead — the two layers use different parameter names); dim_feedforward defaults to 2048. The single-argument call runs self-attention only; use .forward(tgt, memory) for the full cross-attention path.
- **PyTorch:** torch.nn.TransformerDecoderLayer — DIVERGES (new deviation, D-29 candidate): with identical weights copied from LiteTorch, the two-argument forward returns different outputs (max diff 0.1245 in eval mode on the reference inputs). The cause is not yet isolated — possibly a different norm placement or cross-attention detail. The single-argument call was not compared apples-to-apples (LiteTorch runs self-attention only; torch needs a memory argument).
- **Example:**

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

- **Signature:** `TransformerEncoder(encoder_layer, num_layers, norm=nullptr)`
- **Source:** `bind_nn_recurrent.cpp:117`
- **Description:** Stack of num_layers encoder layers, optionally with a final normalization.
- **Parameters:** `encoder_layer` (required), `num_layers` (required), `norm` = nullptr.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** Takes an encoder_layer instance, num_layers, and an optional norm (default None).
- **PyTorch:** torch.nn.TransformerEncoder — equivalent.
- **Example:**

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

- **Signature:** `GPTBlock(d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912, rope_theta=500000.0)`
- **Source:** `bind_nn_base.cpp:258`
- **Description:** One GPT decoder block (GQA attention + SwiGLU MLP + RMSNorm) with the LiteTorch 2B-model defaults.
- **Parameters:** `d_model` = 2560, `n_q_heads` = 20, `n_kv_heads` = 5, `d_ff` = 6912, `rope_theta` = 500000.0f.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** Defaults are the 2B-model sizes (d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912) — pass small values on CPU.
- **PyTorch:** No direct torch.nn equivalent; a GPT decoder block in HF Transformers is the conceptual reference. LiteTorch-only module.
- **Example:**

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

- **Signature:** `GPT2B(vocab_size=50257, d_model=2560, n_layers=28, n_q_heads=20, n_kv_heads=5, d_ff=6912, rope_theta=500000.0)`
- **Source:** `bind_nn_base.cpp:271`
- **Description:** The LiteTorch 2B-parameter GPT model: token embedding, n_layers GPT blocks, final norm, and a tied LM head.
- **Parameters:** `vocab_size` = 50257, `d_model` = 2560, `n_layers` = 28, `n_q_heads` = 20, `n_kv_heads` = 5, `d_ff` = 6912, `rope_theta` = 500000.0f.
- **Input:** `(seq, batch, embed_dim) [+ (seq, batch, embed_dim) key/value for attention]` → **Output:** `(seq, batch, embed_dim)`
- **Edge cases:** Defaults are the full 2B sizes (vocab_size=50257, d_model=2560, n_layers=28) — pass small values on CPU. Input must be a 2-D INT64 tensor of token IDs (batch, seq); a 1-D input raises 'Shape mismatch in view'. Methods: forward(ids) -> logits, stage_forward(input, stage), tie_weights(), pipeline_to(d0, d1, split=-1).
- **PyTorch:** No direct torch.nn equivalent; a GPT-style decoder in HF Transformers is the conceptual reference. LiteTorch-only module.
- **Example:**

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
## Losses
### `nn.MSELoss`

- **Signature:** `MSELoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:126`
- **Description:** Mean squared error: mean((input - target)^2) by default.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** reduction defaults to 'mean' ('sum' and 'none' are the other options).
- **PyTorch:** torch.nn.MSELoss — equivalent.
- **Example:**

<!-- ex: id=r7-mse-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MSELoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.L1Loss`

- **Signature:** `L1Loss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:219`
- **Description:** Mean absolute error: mean(|input - target|) by default.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.L1Loss — equivalent.
- **Example:**

<!-- ex: id=r7-l1-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.L1Loss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.SmoothL1Loss`

- **Signature:** `SmoothL1Loss(beta=1.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:332`
- **Description:** Smooth L1 (Huber-like) loss: quadratic below beta, linear above. beta defaults to 1.0.
- **Parameters:** `beta` = 1.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** beta defaults to 1.0; reduction defaults to 'mean'.
- **PyTorch:** torch.nn.SmoothL1Loss — equivalent.
- **Example:**

<!-- ex: id=r7-smooth-l1-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SmoothL1Loss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.HuberLoss`

- **Signature:** `HuberLoss(delta=1.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:342`
- **Description:** Huber loss: 0.5 * (input - target)^2 below delta, delta * (|input - target| - 0.5 * delta) above.
- **Parameters:** `delta` = 1.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** delta defaults to 1.0; reduction defaults to 'mean'.
- **PyTorch:** torch.nn.HuberLoss — equivalent.
- **Example:**

<!-- ex: id=r7-huber-loss-1 device=cpu backend=cpu runtime=~0.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.HuberLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.CrossEntropyLoss`

- **Signature:** `CrossEntropyLoss(weight=py::none(, ignore_index=-100, reduction="mean", label_smoothing=0.0)`
- **Source:** `bind_nn_layers.cpp:198`
- **Description:** Cross entropy between logits and class indices: -log(softmax(logits)[target]).
- **Parameters:** `weight` = py::none(, `ignore_index` = -100, `reduction` = "mean", `label_smoothing` = 0.0f.
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Torch requires int64 targets, so build the target with dtype INT32 (for example lt.tensor([1, 0], dtype=lt.DataType.INT32)). Note the D-06 interplay: after the D-06 fix .long() returns INT64, which this loss rejects — use INT32 explicitly.
- **PyTorch:** torch.nn.CrossEntropyLoss — equivalent in value, but the target dtype differs (see the edge case).
- **Example:**

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

- **Signature:** `NLLLoss(weight=py::none(, ignore_index=-100, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:283`
- **Description:** Negative log likelihood: -input[target] where input is log-probabilities.
- **Parameters:** `weight` = py::none(, `ignore_index` = -100, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Pass log-probabilities (for example from .log_softmax(dim)) as input, and an INT32 target tensor.
- **PyTorch:** torch.nn.NLLLoss — equivalent in value, but the target dtype differs (see the edge case).
- **Example:**

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

- **Signature:** `BCELoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:213`
- **Description:** Binary cross entropy between probabilities and binary targets.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Input must already be probabilities in [0, 1]; for raw logits use BCEWithLogitsLoss instead. reduction defaults to 'mean'.
- **PyTorch:** torch.nn.BCELoss — equivalent.
- **Example:**

<!-- ex: id=r7-bce-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BCELoss()
y = m.forward(lt.Tensor.from_vector([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], [2, 3], device), lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.BCEWithLogitsLoss`

- **Signature:** `BCEWithLogitsLoss(weight=py::none(, reduction="mean", pos_weight=py::none()`
- **Source:** `bind_nn_layers.cpp:297`
- **Description:** Binary cross entropy with logits: sigmoid + BCE in one numerically stable step.
- **Parameters:** `weight` = py::none(, `reduction` = "mean", `pos_weight` = py::none(.
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** reduction defaults to 'mean'. pos_weight is not exposed in this binding.
- **PyTorch:** torch.nn.BCEWithLogitsLoss — equivalent.
- **Example:**

<!-- ex: id=r7-bce-with-logits-loss-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.BCEWithLogitsLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.KLDivLoss`

- **Signature:** `KLDivLoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:323`
- **Description:** Kullback-Leibler divergence between log-probabilities (input) and probabilities (target).
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Input must be log-probabilities; target must be probabilities. reduction defaults to 'mean'.
- **PyTorch:** torch.nn.KLDivLoss — equivalent.
- **Example:**

<!-- ex: id=r7-kl-div-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.KLDivLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device).log_softmax(1), lt.Tensor.from_vector([0.2, 0.5, 0.3, 0.4, 0.3, 0.3], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.HingeEmbeddingLoss`

- **Signature:** `HingeEmbeddingLoss(margin=1.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:133`
- **Description:** Hinge embedding loss for similarity learning: max(0, margin - input) when target == 1, input when target == -1.
- **Parameters:** `margin` = 1.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** DIVERGES from torch (D-26 candidate): LiteTorch returns mean(max(0, margin - target * input)) over all elements (1.1750 on the example); torch returns 0.7000 on the same inputs. margin defaults to 1.0. Target values are 1 or -1.
- **PyTorch:** torch.nn.HingeEmbeddingLoss — DIVERGES (new deviation, D-26 candidate): LiteTorch computes mean(max(0, margin - target * input)) for every element — the classic SVM hinge form — while torch computes input when target == 1 and max(0, margin - input) when target == -1. On the reference example LiteTorch returns 1.1750 where torch returns 0.7000.
- **Example:**

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

- **Signature:** `CosineEmbeddingLoss(margin=0.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:139`
- **Description:** Cosine embedding loss: 1 - cos(x1, x2) when target == 1, max(0, cos(x1, x2) - margin) when target == -1.
- **Parameters:** `margin` = 0.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Three inputs (x1, x2, target): call .forward(x1, x2, target) explicitly. margin defaults to 0.0.
- **PyTorch:** torch.nn.CosineEmbeddingLoss — equivalent.
- **Example:**

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

- **Signature:** `MarginRankingLoss(margin=0.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:146`
- **Description:** Margin ranking loss: max(0, -target * (input1 - input2) + margin).
- **Parameters:** `margin` = 0.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Three inputs (input1, input2, target): call .forward(x1, x2, target) explicitly. margin defaults to 0.0.
- **PyTorch:** torch.nn.MarginRankingLoss — equivalent.
- **Example:**

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

- **Signature:** `MultiMarginLoss(p=1, margin=1.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:152`
- **Description:** Multi-class multi-margin loss for classification.
- **Parameters:** `p` = 1, `margin` = 1.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** DIVERGES from torch (D-27 candidate): 1.0000 vs 0.7500 on the single-sample all-zeros case; 0.9667 vs 0.6000 on the two-sample example. p defaults to 1 (an int in this binding — p=2.0 raises TypeError). Target is a 1-D INT32 tensor of class indices (same INT32-target rule as CrossEntropyLoss).
- **PyTorch:** torch.nn.MultiMarginLoss — DIVERGES (new deviation, D-27 candidate): the per-sample normalization differs. On the single-sample all-zeros case LiteTorch returns 1.0000 where torch returns 0.7500 (torch divides the summed hinge terms by the number of classes; LiteTorch divides by classes - 1). On the two-sample example LiteTorch returns 0.9667 where torch returns 0.6000 (p=1); with p=2 LiteTorch returns 1.1700 where torch returns 0.7525.
- **Example:**

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

- **Signature:** `MultiLabelMarginLoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:159`
- **Description:** Multi-label margin loss for multi-label classification.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Target is a 2-D INT32 tensor of label indices (same INT32-target rule as CrossEntropyLoss).
- **PyTorch:** torch.nn.MultiLabelMarginLoss — equivalent.
- **Example:**

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

- **Signature:** `SoftMarginLoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:164`
- **Description:** Soft margin loss: log(1 + exp(-target * input)) for binary classification with targets in {-1, 1}.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.SoftMarginLoss — equivalent.
- **Example:**

<!-- ex: id=r7-soft-margin-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.SoftMarginLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.MultiLabelSoftMarginLoss`

- **Signature:** `MultiLabelSoftMarginLoss(reduction="mean")`
- **Source:** `bind_nn_layers.cpp:169`
- **Description:** Multi-label soft margin loss: mean over labels of log(1 + exp(-target * input)) with targets in {0, 1}.
- **Parameters:** `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.MultiLabelSoftMarginLoss — equivalent.
- **Example:**

<!-- ex: id=r7-multi-label-soft-margin-loss-1 device=cpu backend=cpu runtime=~0.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.MultiLabelSoftMarginLoss()
y = m.forward(lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [2, 3], device), lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.TripletMarginLoss`

- **Signature:** `TripletMarginLoss(margin=1.0, p=2, eps=1e-6, swap=false, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:174`
- **Description:** Triplet margin loss: max(0, d(anchor, positive) - d(anchor, negative) + margin).
- **Parameters:** `margin` = 1.0f, `p` = 2, `eps` = 1e-6f, `swap` = false, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Three inputs (anchor, positive, negative): call .forward(a, p, n) explicitly. margin defaults to 1.0, p to 2.0.
- **PyTorch:** torch.nn.TripletMarginLoss — equivalent.
- **Example:**

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

- **Signature:** `TripletMarginWithDistanceLoss(margin=1.0, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:180`
- **Description:** Triplet margin loss with a custom distance function.
- **Parameters:** `margin` = 1.0f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Three inputs (anchor, positive, negative): call .forward(a, p, n) explicitly. The distance function argument is not exposed; the default distance is used.
- **PyTorch:** torch.nn.TripletMarginWithDistanceLoss — equivalent in intent.
- **Example:**

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

- **Signature:** `PoissonNLLLoss(log_input=true, full=false, eps=1e-8, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:186`
- **Description:** Poisson negative log likelihood: input - target * log(input + eps).
- **Parameters:** `log_input` = true, `full` = false, `eps` = 1e-8f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** log_input defaults to True; full defaults to False. eps defaults to 1e-8.
- **PyTorch:** torch.nn.PoissonNLLLoss — equivalent.
- **Example:**

<!-- ex: id=r7-poisson-nll-loss-1 device=cpu backend=cpu runtime=~0.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

m = lt.nn.PoissonNLLLoss()
y = m.forward(lt.Tensor.from_vector([0.5, 1.0, 1.5, 2.0, 0.8, 1.2], [2, 3], device), lt.Tensor.from_vector([1.0, 2.0, 1.0, 3.0, 1.0, 2.0], [2, 3], device))
print("loss:", y.to_vector())
```
### `nn.GaussianNLLLoss`

- **Signature:** `GaussianNLLLoss(full=false, eps=1e-6, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:191`
- **Description:** Gaussian negative log likelihood: 0.5 * (log(var) + (input - target)^2 / var).
- **Parameters:** `full` = false, `eps` = 1e-6f, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Three inputs (input, target, var): call .forward(input, target, var) explicitly. full defaults to False; eps defaults to 1e-6.
- **PyTorch:** torch.nn.GaussianNLLLoss — equivalent.
- **Example:**

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

- **Signature:** `CTCLoss(blank=0, reduction="mean", zero_infinity=false)`
- **Source:** `bind_nn_layers.cpp:352`
- **Description:** Connectionist temporal classification loss for sequence alignment (speech, OCR).
- **Parameters:** `blank` = 0, `reduction` = "mean", `zero_infinity` = false.
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** DIVERGES from torch under reduction='mean' (D-28 candidate): LiteTorch 4.0802 vs torch 1.3601 on the example; reduction='sum' matches exactly (8.1604). Four inputs (log_probs, targets, input_lengths, target_lengths): call .forward(lp, targets, in_lens, tgt_lens) explicitly. log_probs must be log-probabilities of shape (T, N, C). blank defaults to 0 — blank must not appear in the targets. Lengths are INT32 tensors on this build (the INT32 rule of CrossEntropyLoss applies here too).
- **PyTorch:** torch.nn.CTCLoss — DIVERGES under reduction='mean' (new deviation, D-28 candidate): LiteTorch returns the mean over the batch of the per-sample losses (4.0802 on the example), while torch divides each per-sample loss by its target length first (1.3601). LiteTorch's reduction='sum' matches torch's 'sum' exactly (8.1604 on the example).
- **Example:**

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

- **Signature:** `DPOLoss(beta=0.1, reference_free=false, reduction="mean")`
- **Source:** `bind_nn_layers.cpp:368`
- **Description:** Direct preference optimization loss: -log(sigmoid(beta * (logp_chosen - logp_rejected))) with a reference-model term.
- **Parameters:** `beta` = 0.1f, `reference_free` = false, `reduction` = "mean".
- **Input:** `(prediction, target) — see the example` → **Output:** `scalar tensor, shape [1]`
- **Edge cases:** Four inputs (chosen, rejected, ref_chosen, ref_rejected): call .forward(c, r, rc, rr) explicitly. beta defaults to 0.1.
- **PyTorch:** No torch.nn equivalent (DPO is a training recipe, not a torch module). LiteTorch-only module.
- **Example:**

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
## Serving and speculative decoding
### `nn.GenerateConfig`

- **Signature:** `GenerateConfig()`
- **Source:** `bind_nn_utils.cpp:14`
- **Description:** Configuration for text generation: max_new_tokens, temperature, top_p, top_k, repetition_penalty, and the stop-token list.
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** max_new_tokens defaults to 32. temperature defaults to 1.0. stop_tokens defaults to an empty list.
- **PyTorch:** No torch.nn equivalent; HF GenerationConfig is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `GuidedDecoder(vocabulary)`
- **Source:** `bind_nn_utils.cpp:30`
- **Description:** Constrained decoding: masks logits so generation follows a grammar (for example JSON). Construct with a vocabulary list; apply_mask(logits, prefix, target_pattern) is in-place.
- **Parameters:** `vocabulary` (required).
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Constructor takes a vocabulary (list of strings). apply_mask(logits, prefix, target_pattern) takes THREE arguments and modifies logits in place; the target_pattern is a regex-style pattern. The vocab attribute is readable/writable.
- **PyTorch:** No torch.nn equivalent; vLLM guided decoding is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `Request(id, prompt, max_new_tokens=32)`
- **Source:** `bind_nn_utils.cpp:35`
- **Description:** One serving request: an integer id, a prompt token list, and a max_tokens budget.
- **Parameters:** `id` (required), `prompt` (required), `max_new_tokens` = 32.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** id and prompt are required; max_new_tokens defaults to 32. Generated tokens accumulate in generated_tokens; is_finished and prefill_processed track progress.
- **PyTorch:** No torch.nn equivalent; vLLM Request is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `PrefixCache.get() — no constructor`
- **Source:** `bind_nn_utils.cpp:59`
- **Description:** Singleton prefix cache for KV blocks, shared across requests with the same prompt prefix. Obtain the instance with PrefixCache.get().
- **Parameters:** No constructor. Use the static `PrefixCache.get()`.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** No public constructor: use the static PrefixCache.get(). put(prefix, blocks) stores blocks for a token prefix; get_cache(prefix) retrieves them; clear() empties the cache.
- **PyTorch:** No torch.nn equivalent; vLLM prefix caching is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `ServingScheduler(num_blocks, block_size, max_num_seqs, max_num_batched_tokens)`
- **Source:** `bind_nn_utils.cpp:45`
- **Description:** Continuous-batching serving scheduler: admits requests, allocates KV blocks, and runs prefill + decode steps. step(model) returns the requests that finished on that step.
- **Parameters:** `num_blocks` (required), `block_size` (required), `max_num_seqs` (required), `max_num_batched_tokens` (required).
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** The model must map token IDs to logits: its first layer must be an Embedding (the scheduler feeds a [1, chunk] tensor of token IDs and reads a [1, seq, vocab] output). A bare Linear model fails with 'Matrix inner dimensions must match'.
- **PyTorch:** No torch.nn equivalent; vLLM's scheduler is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `SpeculativeEngine(...)`
- **Source:** `bind_nn_utils.cpp:65`
- **Description:** Speculative decoding engine: a small draft model proposes tokens and a larger target model verifies them. generate(prompt, max_new_tokens, lookahead=4) runs one request.
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Constructor takes (draft_model, target_model) — two shared_ptr modules. generate(prompt, max_new_tokens, lookahead=4) returns the generated token list.
- **PyTorch:** No torch.nn equivalent; vLLM speculative decoding is the conceptual reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `MedusaEngine(...)`
- **Source:** `bind_nn_utils.cpp:79`
- **Description:** Medusa speculative decoding engine: MedusaHeads propose extra tokens per position for parallel verification. Construct with (base_model, medusa_heads).
- **Parameters:** None.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Constructor takes (base_model, medusa_heads) where medusa_heads is a list of MedusaHead instances. generate(prompt, max_new_tokens, lookahead=2) runs one request.
- **PyTorch:** No torch.nn equivalent; the Medusa paper (Cai et al., 2024) is the reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `MedusaHead(int, int)`
- **Source:** `bind_nn_utils.cpp:71`
- **Description:** Medusa prediction head: maps hidden states to extra token predictions, one head per speculative position.
- **Parameters:** Positional `(int, int)` — (in_features, out_features); no keyword argument names are bound.
- **Input:** `varies — see the example` → **Output:** `varies — see the example`
- **Edge cases:** Takes two positional ints (in_features, out_features); no keyword argument names are bound.
- **PyTorch:** No torch.nn equivalent; the Medusa paper (Cai et al., 2024) is the reference. LiteTorch-only class.
- **Example:**

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

- **Signature:** `generate(model, prompt_tokens, config=GenerateConfig())`
- **Source:** `bind_nn_utils.cpp:27`
- **Description:** Runs greedy text generation with a model that maps token IDs to logits (Embedding followed by Linear). Returns the generated token list, not including the prompt.
- **Parameters:** `model` (required), `prompt_tokens` (required, list of ints), `config` = `GenerateConfig()` (see GenerateConfig above; `max_new_tokens` defaults to 32).
- **Input:** `(model, [int, ...])` → **Output:** `[int, ...]` generated tokens.
- **Edge cases:** The model must map token IDs to logits: its first layer must be an Embedding. A bare Linear model fails. Generation is greedy unless the config enables sampling.
- **PyTorch:** No torch.nn equivalent; HF `generate()` is the conceptual reference. LiteTorch-only function.
- **Example:**

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
