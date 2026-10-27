# nn.functional, nn.init, nn.utils — API reference

Functions under `litetorch.nn.functional`, `litetorch.nn.init`, and `litetorch.nn.utils`. Signatures are taken from the pybind11 bindings (`src/bindings/bind_nn_utils.cpp`); the binding is the source of truth. Every example is self-contained, runs on CPU, and carries a verification header (see the verification plan, `plan/verification.md` §4).

Note: `nn.utils` has five functions. `clip_value` is NOT bound here — only `clip_grad_value_`; the norm-based clip, `clip_grad_norm_`, lives under `Ops`.
## nn.functional
### `F.dropout`

- **Signature:** `dropout(x, p=0.5, training=True)`
- **Source:** `bind_nn_utils.cpp:171`
- **Description:** Functional dropout: randomly zeroes each element with probability p when training is True; identity otherwise. Output is scaled by 1/(1-p) in training.
- **Edge cases:** training defaults to True, so a bare call is stochastic: pass training=False (or seed with lt.manual_seed()) for a deterministic pass. The module form is nn.Dropout.
- **PyTorch:** torch.nn.functional.dropout — equivalent.
- **Example:**

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

- **Signature:** `linear(x, weight, bias=None)`
- **Source:** `bind_nn_utils.cpp:173`
- **Description:** Functional linear: y = x @ weight.T + bias.
- **Edge cases:** bias defaults to None (no bias). weight shape is (out_features, in_features).
- **PyTorch:** torch.nn.functional.linear — equivalent.
- **Example:**

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

- **Signature:** `pad(x, pad, mode='constant', value=0.0)`
- **Source:** `bind_nn_utils.cpp:175`
- **Description:** Pads a tensor. pad is a list of (before, after) pairs per dimension, starting from the last dimension.
- **Edge cases:** mode defaults to 'constant' with value 0.0. The pad list is ordered from the last dimension backward: [1, 1, 0, 0] pads the last dimension by 1 on each side and leaves the next dimension unchanged.
- **PyTorch:** torch.nn.functional.pad — equivalent for the constant mode.
- **Example:**

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

- **Signature:** `interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False)`
- **Source:** `bind_nn_utils.cpp:177`
- **Description:** Resizes a tensor. Give size (a list of output sizes) or scale_factor (a float or a list).
- **Edge cases:** size and scale_factor are both optional but one must be given. mode defaults to 'nearest'; align_corners defaults to False.
- **PyTorch:** torch.nn.functional.interpolate — equivalent for the nearest mode.
- **Example:**

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

- **Signature:** `nll_loss(input, target, weight=None, reduction='mean', ignore_index=-100)`
- **Source:** `bind_nn_utils.cpp:201`
- **Description:** Negative log likelihood: -input[target] where input is log-probabilities.
- **Edge cases:** TARGET DTYPE (new deviation, D-23 candidate): targets must be INT32 on this build — an INT64 target throws 'cross_entropy_loss: target must have integer or float dtype'. Pass log-probabilities (for example from .log_softmax(dim)) as input, and an INT32 target tensor.
- **PyTorch:** torch.nn.functional.nll_loss — equivalent in value, but the target dtype differs (see the edge case).
- **Example:**

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

- **Signature:** `bce_with_logits(input, target, reduction='mean')`
- **Source:** `bind_nn_utils.cpp:204`
- **Description:** Binary cross entropy with logits: sigmoid + BCE in one numerically stable step.
- **Edge cases:** reduction defaults to 'mean'.
- **PyTorch:** torch.nn.functional.binary_cross_entropy_with_logits — equivalent.
- **Example:**

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

- **Signature:** `normalize(x, p=2.0, dim=1, eps=1e-12)`
- **Source:** `bind_nn_utils.cpp:206`
- **Description:** L_p normalization along dim: x / max(||x||_p, eps).
- **Edge cases:** p defaults to 2.0, dim to 1, eps to 1e-12.
- **PyTorch:** torch.nn.functional.normalize — equivalent.
- **Example:**

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

- **Signature:** `glu(x, dim=-1)`
- **Source:** `bind_nn_utils.cpp:208`
- **Description:** Gated linear unit: splits the input in half along dim and returns a * sigmoid(b).
- **Edge cases:** dim defaults to -1. The size of dim must be even. The module form is nn.GLU.
- **PyTorch:** torch.nn.functional.glu — equivalent.
- **Example:**

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

- **Signature:** `elu(x, alpha=1.0)`
- **Source:** `bind_nn_utils.cpp:210`
- **Description:** Exponential linear unit: x for x > 0, alpha * (exp(x) - 1) otherwise.
- **Edge cases:** alpha defaults to 1.0. The module form is nn.ELU.
- **PyTorch:** torch.nn.functional.elu — equivalent.
- **Example:**

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

- **Signature:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps=None, ref_rejected_logps=None, beta=0.1, reference_free=False, reduction='mean')`
- **Source:** `bind_nn_utils.cpp:214`
- **Description:** Direct preference optimization loss: -log(sigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))).
- **Edge cases:** ref_chosen_logps and ref_rejected_logps are required unless reference_free=True — omitting them raises ValueError. beta defaults to 0.1.
- **PyTorch:** No torch.nn.functional equivalent (DPO is a training recipe, not a torch function). LiteTorch-only function. The module form is nn.DPOLoss.
- **Example:**

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

- **Signature:** `kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='relu')`
- **Source:** `bind_nn_utils.cpp:86`
- **Description:** Fills a tensor in place with values sampled from U(-bound, bound), where bound is computed from the fan-in (or fan-out) by the Kaiming/He formula. Returns the tensor.
- **Edge cases:** a defaults to 0.0 (the negative slope of the rectifier), mode to 'fan_in', nonlinearity to 'relu'. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.kaiming_uniform_ — equivalent.
- **Example:**

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

- **Signature:** `kaiming_normal_(tensor, a=0.0, mode='fan_in', nonlinearity='relu')`
- **Source:** `bind_nn_utils.cpp:87`
- **Description:** Fills a tensor in place with values sampled from N(0, std), where std is computed from the fan-in (or fan-out) by the Kaiming/He formula. Returns the tensor.
- **Edge cases:** a defaults to 0.0, mode to 'fan_in', nonlinearity to 'relu'. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.kaiming_normal_ — equivalent.
- **Example:**

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

- **Signature:** `xavier_uniform_(tensor, gain=1.0)`
- **Source:** `bind_nn_utils.cpp:88`
- **Description:** Fills a tensor in place with values sampled from U(-a, a), where a = gain * sqrt(6 / (fan_in + fan_out)). Returns the tensor.
- **Edge cases:** gain defaults to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.xavier_uniform_ — equivalent.
- **Example:**

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

- **Signature:** `xavier_normal_(tensor, gain=1.0)`
- **Source:** `bind_nn_utils.cpp:89`
- **Description:** Fills a tensor in place with values sampled from N(0, std), where std = gain * sqrt(2 / (fan_in + fan_out)). Returns the tensor.
- **Edge cases:** gain defaults to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.xavier_normal_ — equivalent.
- **Example:**

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

- **Signature:** `normal_(tensor, mean=0.0, std=1.0)`
- **Source:** `bind_nn_utils.cpp:90`
- **Description:** Fills a tensor in place with values sampled from N(mean, std). Returns the tensor.
- **Edge cases:** mean defaults to 0.0, std to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.normal_ — equivalent.
- **Example:**

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

- **Signature:** `uniform_(tensor, a=0.0, b=1.0)`
- **Source:** `bind_nn_utils.cpp:91`
- **Description:** Fills a tensor in place with values sampled from U(a, b). Returns the tensor.
- **Edge cases:** a defaults to 0.0, b to 1.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.uniform_ — equivalent.
- **Example:**

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

- **Signature:** `zeros_(tensor)`
- **Source:** `bind_nn_utils.cpp:92`
- **Description:** Fills a tensor in place with zeros. Returns the tensor.
- **Edge cases:** In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.zeros_ — equivalent.
- **Example:**

<!-- ex: id=r8-init-zeros_-1 device=cpu backend=cpu runtime=~0.3s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.zeros_(t)
print("values:", t.to_vector())
```
### `init.ones_`

- **Signature:** `ones_(tensor)`
- **Source:** `bind_nn_utils.cpp:93`
- **Description:** Fills a tensor in place with ones. Returns the tensor.
- **Edge cases:** In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.ones_ — equivalent.
- **Example:**

<!-- ex: id=r8-init-ones_-1 device=cpu backend=cpu runtime=~0.4s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.ones_(t)
print("values:", t.to_vector())
```
### `init.fill_`

- **Signature:** `fill_(tensor, value)`
- **Source:** `bind_nn_utils.cpp:94`
- **Description:** Fills a tensor in place with a constant value. Returns the tensor.
- **Edge cases:** value is required (no default). In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.constant_ is the torch counterpart (LiteTorch names it fill_).
- **Example:**

<!-- ex: id=r8-init-fill_-1 device=cpu backend=cpu runtime=~0.5s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

t = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4], [4], device)
lt.nn.init.fill_(t, 3.5)
print("values:", t.to_vector())
```
### `init.trunc_normal_`

- **Signature:** `trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)`
- **Source:** `bind_nn_utils.cpp:96`
- **Description:** Fills a tensor in place with values sampled from a truncated normal distribution N(mean, std) clipped to [a, b]. Returns the tensor.
- **Edge cases:** mean defaults to 0.0, std to 1.0, a to -2.0, b to 2.0. In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.trunc_normal_ — equivalent.
- **Example:**

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

- **Signature:** `orthogonal_(tensor, gain=1.0)`
- **Source:** `bind_nn_utils.cpp:98`
- **Description:** Fills a 2-D tensor in place with an (optionally scaled) orthogonal matrix. Returns the tensor.
- **Edge cases:** gain defaults to 1.0. The tensor must be 2-D with at least as many rows as columns (or vice versa). In-place: the same tensor object is returned.
- **PyTorch:** torch.nn.init.orthogonal_ — equivalent.
- **Example:**

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

- **Signature:** `clip_grad_value_(parameters, clip_value)`
- **Source:** `bind_nn_utils.cpp:7`
- **Description:** Clips the gradients of a list of parameters in place to [-clip_value, clip_value].
- **Edge cases:** parameters is a list of tensors; clip_value is required. Operates on .grad — parameters without gradients are skipped. Note: clip_grad_norm_ is NOT under nn.utils; it lives under Ops.
- **PyTorch:** torch.nn.utils.clip_grad_value_ — equivalent.
- **Example:**

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

- **Signature:** `parameters_to_vector(parameters)`
- **Source:** `bind_nn_utils.cpp:8`
- **Description:** Flattens a list of parameters into a single 1-D vector.
- **Edge cases:** parameters is a list of tensors. The vector length is the sum of the parameter sizes.
- **PyTorch:** torch.nn.utils.parameters_to_vector — equivalent.
- **Example:**

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

- **Signature:** `vector_to_parameters(vec, parameters)`
- **Source:** `bind_nn_utils.cpp:9`
- **Description:** Copies a flat 1-D vector back into a list of parameters (the inverse of parameters_to_vector).
- **Edge cases:** The vector length must equal the total parameter size.
- **PyTorch:** torch.nn.utils.vector_to_parameters — equivalent.
- **Example:**

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

- **Signature:** `weight_norm(v, g, dim=0)`
- **Source:** `bind_nn_utils.cpp:10`
- **Description:** Weight normalization: reparameterizes a weight as g * v / ||v|| along dim. Returns the normalized weight tensor.
- **Edge cases:** v is the direction tensor, g the scale tensor; dim defaults to 0.
- **PyTorch:** torch.nn.utils.weight_norm is a module wrapper in torch; LiteTorch exposes the raw reparameterization as a function.
- **Example:**

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

- **Signature:** `spectral_norm(weight, n_power_iterations=1)`
- **Source:** `bind_nn_utils.cpp:11`
- **Description:** Spectral normalization: rescales a weight by its largest singular value (estimated by power iteration). Returns the normalized weight tensor.
- **Edge cases:** n_power_iterations defaults to 1.
- **PyTorch:** torch.nn.utils.spectral_norm is a module wrapper in torch; LiteTorch exposes the raw normalization as a function.
- **Example:**

<!-- ex: id=r8-utils-spectral_norm-1 device=cpu backend=cpu runtime=~0.4s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.Device("cpu")

w = lt.Tensor.from_vector([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, -0.2, 0.3, -0.4], [3, 4], device)
wn = lt.nn.utils.spectral_norm(w)
print("weight shape:", list(wn.shape))
print("values:", wn.to_vector()[:4])
```
