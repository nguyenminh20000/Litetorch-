# Optimizers, LR Schedulers, and AMP — API Reference

`litetorch.optim` holds all optimizers, learning-rate schedulers, and the base `GradScaler`; `litetorch.amp` holds the AMP `GradScaler`, `AutocastGuard`, and `autocast`.

- **Source of truth:** `src/bindings/bind_optim.cpp` (optimizers, schedulers, `optim.GradScaler`), `src/bindings/bind_amp.cpp` (AMP), `include/litetorch/optim.h`, `include/litetorch/amp.h`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline; B5 re-verifies at head). Examples run against the Core D3 extension of 88a4ba1.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise. Optimizers take a Python list of parameter tensors; gradients must be set on `p.grad` before `step()` (or produced by `loss.backward()`).
- **Related deviations:** NAdam diverges from PyTorch (measured, see `NAdam` — root cause in `src/optim/nadam.cpp`: `mu_product` is the ratio `mu_next/mu`, not the running product PyTorch keeps, and `step_size` divides by `bias_correction1`, which PyTorch does not do for NAdam). `save_optimizer_state` / `load_optimizer_state` support Adam, AdamW, SGD, RMSprop only and throw for every other optimizer. `AdamW8bit` / `AdamWFP8` / `LinearWarmup` have no PyTorch equivalent. See `plan/differences.md`.

## `Optimizer` (base class)

- **Signature:** `Optimizer(params) -> Optimizer`
- **Source:** `src/bindings/bind_optim.cpp:6`
- **Parameters:** `params` — list of `Tensor` parameters.
- **Methods:** `step()` — one update step, skips parameters with no `.grad`; `zero_grad()` — clears all parameter gradients; `get_lr()` — current learning rate of the first param group; `set_lr(new_lr)` — sets the learning rate of every param group.
- **Edge cases:** `step()` on an optimizer whose parameters have no gradients is a no-op. There is no `state_dict()` / `load_state_dict()` on the base class — only `Adagrad` and `NAdam` expose them (see below). `param_groups` is exposed only on `Adagrad` and `NAdam`.
- **PyTorch:** `torch.optim.Optimizer` — equivalent core API, except PyTorch exposes `state_dict()` / `load_state_dict()` / `param_groups` on every optimizer.

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

- **Signature:** `SGD(params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False) -> SGD`
- **Source:** `src/bindings/bind_optim.cpp:12-13`
- **Parameters:** `lr` — learning rate, default 0.01. `momentum` — momentum factor, default 0.0, range [0, 1). `dampening` — dampening for momentum, default 0.0. `weight_decay` — L2 penalty, default 0.0. `nesterov` — Nesterov momentum, default False (requires `momentum != 0` to have any effect).
- **Update:** `v = momentum * v + (1 - dampening) * g`; `p -= lr * (v + weight_decay * p)`; with Nesterov the gradient used is `g + momentum * v`.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** `nesterov=True` with `momentum=0` behaves like plain SGD. Parameters without `.grad` are skipped.
- **PyTorch:** `torch.optim.SGD` — numerically identical (verified 1e-7).

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

- **Signature:** `Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> Adam`
- **Source:** `src/bindings/bind_optim.cpp:24-25`
- **Parameters:** `lr` — default 0.001. `beta1` — first-moment decay, default 0.9. `beta2` — second-moment decay, default 0.999. `eps` — denominator stabilizer, default 1e-8. `weight_decay` — L2 penalty added to the gradient (not decoupled), default 0.0. Note the two betas are separate keyword arguments, not a `betas` tuple as in PyTorch.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** no `amsgrad` option. No `state_dict()` — use `lt.save_optimizer_state` / `lt.load_optimizer_state` (Adam is supported there).
- **PyTorch:** `torch.optim.Adam` — numerically identical (verified 1e-7); pass `betas=(beta1, beta2)` on the torch side.

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

- **Signature:** `AdamW(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, offload_to_cpu=False) -> AdamW`
- **Source:** `src/bindings/bind_optim.cpp:36-37`
- **Parameters:** `lr` — default 0.001. `beta1` — default 0.9. `beta2` — default 0.999. `eps` — default 1e-8. `weight_decay` — decoupled weight decay, default 0.01 (differs from Adam's 0.0). `offload_to_cpu` — keep optimizer states on CPU, default False.
- **Returns:** nothing. `step()` updates parameters in place. Exposes `fused_launch_count`, `single_launch_count`, `reset_launch_counts` (static counters for the fused CUDA launch path).
- **Edge cases:** decoupled decay means `p -= lr * weight_decay * p` happens independently of the gradient. No `amsgrad` option.
- **PyTorch:** `torch.optim.AdamW` — numerically identical (verified 1e-7).

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

- **Signature:** `AdamW8bit(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> AdamW8bit`
- **Source:** `src/bindings/bind_optim.cpp:51-52`
- **Parameters:** same defaults as `AdamW` except `weight_decay` defaults to 0.0.
- **Behavior:** AdamW where the first and second moments are stored in 8-bit block-quantized form (block size 2048, one FP32 scale per block). Memory per parameter is ~2 bytes instead of 8 for the two moments. The update is computed from dequantized moments, so results are close to AdamW but not bit-identical.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** quantization error grows for parameters whose moments span a large dynamic range within one 2048-element block. No `save_optimizer_state` support (throws `Unsupported optimizer type`).
- **PyTorch:** no equivalent in `torch.optim` (8-bit AdamW exists only in third-party libraries such as bitsandbytes). Compare against `torch.optim.AdamW` with a loose tolerance.

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

- **Signature:** `AdamWFP8(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> AdamWFP8`
- **Source:** `src/bindings/bind_optim.cpp:60-61`
- **Parameters:** same defaults as `AdamW8bit`.
- **Behavior:** AdamW where the moments are stored in FP8 with an amax history for scaling. Same memory profile as `AdamW8bit` (2 bytes per parameter for the two moments) with a different quantization scheme.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** FP8 has a narrower dynamic range than the 8-bit block format of `AdamW8bit`; large gradient spikes can overflow the FP8 range and saturate moments. No `save_optimizer_state` support.
- **PyTorch:** no equivalent in `torch.optim`.

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

- **Signature:** `RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0) -> RMSprop`
- **Source:** `src/bindings/bind_optim.cpp:69-70`
- **Parameters:** `lr` — default 0.01. `alpha` — smoothing constant for the squared-gradient average, default 0.99. `eps` — default 1e-8. `weight_decay` — L2 penalty, default 0.0.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** no `momentum` or `centered` options (both exist in PyTorch). No `state_dict()`.
- **PyTorch:** `torch.optim.RMSprop` with `momentum=0, centered=False` — numerically identical (verified 1e-7).

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

- **Signature:** `Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0) -> Adadelta`
- **Source:** `src/bindings/bind_optim.cpp:78-79`
- **Parameters:** `lr` — default 1.0 (Adadelta is designed to be run at lr 1.0). `rho` — decay rate for the squared-gradient and squared-update averages, default 0.9. `eps` — default 1e-6. `weight_decay` — default 0.0.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** the update is scale-free in the gradient (the ratio of two running RMS quantities), so the first step is small and the effective step size is governed by `lr * RMS(update) / RMS(grad)`.
- **PyTorch:** `torch.optim.Adadelta` — numerically identical (verified 1e-7).

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

- **Signature:** `Adamax(params, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> Adamax`
- **Source:** `src/bindings/bind_optim.cpp:88-89`
- **Parameters:** `lr` — default 0.002. `beta1` — default 0.9. `beta2` — default 0.999. `eps` — default 1e-8. `weight_decay` — default 0.0.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** the second moment is the infinity norm (`max(beta2 * v, |g|)`), so the denominator never shrinks below the largest gradient seen. No `save_optimizer_state` support.
- **PyTorch:** `torch.optim.Adamax` — numerically identical (verified 1e-7).

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

- **Signature:** `Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10) -> Adagrad`
- **Source:** `src/bindings/bind_optim.cpp:182-183`
- **Parameters:** `lr` — default 0.01. `lr_decay` — learning-rate decay, default 0.0. `weight_decay` — default 0.0. `eps` — default 1e-10.
- **Methods:** `step()`, `zero_grad()`, `get_lr()`, `set_lr(new_lr)`, plus `add_param_group(params, lr, weight_decay=0.0)`, `param_groups()` (returns a list of dicts), `state_dict()` and `load_state_dict(state_dict)` — one of only two optimizers exposing these (the other is `NAdam`).
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** the accumulated squared-gradient sum only grows, so the effective step size decays over training; use `lr_decay` to decay the learning rate as well. No `save_optimizer_state` support — use `state_dict()` / `load_state_dict()` instead.
- **PyTorch:** `torch.optim.Adagrad` — numerically identical (verified 1e-7).

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

- **Signature:** `NAdam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=0.004) -> NAdam`
- **Source:** `src/bindings/bind_optim.cpp:261-262`, implementation `src/optim/nadam.cpp`
- **Parameters:** `lr` — default 0.002. `betas` — a `(beta1, beta2)` tuple, default (0.9, 0.999). `eps` — default 1e-8. `weight_decay` — default 0.0. `momentum_decay` — decay of the Nesterov momentum schedule, default 0.004.
- **Methods:** `step()`, `zero_grad()`, `get_lr()`, `set_lr(new_lr)`, plus `add_param_group(params, lr, weight_decay=0.0)`, `param_groups()`, `state_dict()`, `load_state_dict(state_dict)`.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** `step()` on GPU tensors throws `NAdam: GPU step not implemented, use CPU tensors`. Parameters must be contiguous. No `save_optimizer_state` support.
- **PyTorch:** `torch.optim.NAdam` — **DIVERGES**. Measured on CPU, `w=[1.0]`, `grad=0.5`, `lr=0.01`, three steps: LiteTorch `[0.9350, 0.8961, 0.8658]` vs PyTorch `[0.9894, 0.9816, 0.9743]`; a single step at `lr=0.1` gives LiteTorch `0.3501` vs PyTorch `0.8944`. Root cause in `src/optim/nadam.cpp`: LiteTorch computes `mu_product = mu_next / mu` (a ratio) and `step_size = lr / bias_correction1`; PyTorch keeps `mu_product` as the running product of all `mu` values so far (starting at 1) and applies the update as two terms, `-lr * (1 - mu) / (1 - mu_product) * grad / denom` and `-lr * mu_next / (1 - mu_product * mu_next) * exp_avg / denom`, with no division of `lr` by `bias_correction1`. The LiteTorch first-step update is therefore about 6.5x too large. Do not port PyTorch NAdam training runs to LiteTorch `NAdam` until this is fixed; use `Adam` instead. Filed to Core as a defect (see `plan/differences.md`).

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

- **Signature:** `ZeRO3Optimizer(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0) -> ZeRO3Optimizer`
- **Source:** `src/bindings/bind_optim.cpp:97-98`, `include/litetorch/zero3_optimizer.h`
- **Parameters:** same defaults as `Adam` (`lr` 0.001, `beta1` 0.9, `beta2` 0.999, `eps` 1e-8, `weight_decay` 0.0).
- **Behavior:** AdamW-based optimizer for ZeRO-3 sharded training. It keeps an FP32 master copy of each parameter and runs the AdamW update on the master weights, so mixed-precision training does not lose the update in FP16 rounding.
- **Returns:** nothing. `step()` updates parameters in place.
- **Edge cases:** no `get_lr()` / `set_lr()` are bound (only `step()` and `zero_grad()`). No `save_optimizer_state` support. On a single process with no distribution it behaves like AdamW with master weights.
- **PyTorch:** no equivalent in `torch.optim` (ZeRO-3 is a DeepSpeed feature).

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

## LR schedulers

All schedulers are classes under `lt.optim` and most are also available as factory functions under `lt.optim.lr_scheduler`. The factory forms take the same arguments but are plain functions; `LinearWarmup` exists only as a class under `lt.optim.lr_scheduler`. Schedulers hold a reference to the optimizer (`keep_alive`), so the optimizer must outlive the scheduler. Every scheduler exposes `step()`; `CosineAnnealingWarmRestarts`, `LinearLR`, `PolynomialLR`, `OneCycleLR`, `LambdaLR`, `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau` also expose `get_last_lr()`. `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau` expose `state_dict()` / `load_state_dict(state_dict)`; the other schedulers do not.

### Step ordering

Call `optimizer.step()` first, then `scheduler.step()` once per training step (once per epoch for epoch-based schedules). `ReduceLROnPlateau.step(metric)` is the exception: call it once per validation, passing the monitored metric. Calling `scheduler.step()` before the first `optimizer.step()` emits a warning in PyTorch and silently shifts the schedule in LiteTorch — always step the optimizer first.

### `StepLR`

- **Signature:** `StepLR(optimizer, step_size, gamma=0.1) -> StepLR`
- **Source:** `src/bindings/bind_optim.cpp:107-108`; factory `src/bindings/bind_optim.cpp:426`
- **Parameters:** `step_size` — number of steps between decays (required, no default). `gamma` — multiplicative decay factor, default 0.1.
- **Behavior:** `lr *= gamma` every `step_size` steps.
- **PyTorch:** `torch.optim.lr_scheduler.StepLR` — identical schedule.

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

- **Signature:** `CosineAnnealingLR(optimizer, T_max, eta_min=0.0) -> CosineAnnealingLR`
- **Source:** `src/bindings/bind_optim.cpp:114-115`; factory `src/bindings/bind_optim.cpp:429`
- **Parameters:** `T_max` — number of steps for half a cosine cycle (required). `eta_min` — minimum learning rate, default 0.0.
- **Behavior:** `lr = eta_min + 0.5 * (lr0 - eta_min) * (1 + cos(pi * t / T_max))`.
- **PyTorch:** `torch.optim.lr_scheduler.CosineAnnealingLR` — identical schedule.

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

- **Signature:** `CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0) -> CosineAnnealingWarmRestarts`
- **Source:** `src/bindings/bind_optim.cpp:121-122`; factory `src/bindings/bind_optim.cpp:447`
- **Parameters:** `T_0` — steps in the first restart cycle (required). `T_mult` — cycle-length multiplier after each restart, default 1. `eta_min` — minimum learning rate, default 0.0.
- **Behavior:** cosine decay within each cycle, resetting to the base learning rate at each restart; cycle length grows by `T_mult` each time.
- **PyTorch:** `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` — identical schedule.

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

- **Signature:** `LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5) -> LinearLR`
- **Source:** `src/bindings/bind_optim.cpp:130-131`; factory `src/bindings/bind_optim.cpp:441`
- **Parameters:** `start_factor` — initial learning-rate multiplier, default 1/3. `end_factor` — final multiplier, default 1.0. `total_iters` — steps over which the multiplier moves linearly, default 5.
- **Behavior:** the multiplier moves linearly from `start_factor` to `end_factor` over `total_iters` steps, then stays at `end_factor`.
- **PyTorch:** `torch.optim.lr_scheduler.LinearLR` — identical schedule.

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

- **Signature:** `PolynomialLR(optimizer, total_iters=5, power=1.0) -> PolynomialLR`
- **Source:** `src/bindings/bind_optim.cpp:139-140`; factory `src/bindings/bind_optim.cpp:444`
- **Parameters:** `total_iters` — steps over which the rate decays, default 5. `power` — polynomial power, default 1.0 (linear decay).
- **Behavior:** `lr = lr0 * (1 - t / total_iters) ** power`; stays at 0 after `total_iters` steps.
- **PyTorch:** `torch.optim.lr_scheduler.PolynomialLR` — identical schedule.

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

- **Signature:** `OneCycleLR(optimizer, max_lr, total_steps=-1, epochs=-1, steps_per_epoch=-1, pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=10000.0) -> OneCycleLR`
- **Source:** `src/bindings/bind_optim.cpp:147-148`; factory `src/bindings/bind_optim.cpp:450`
- **Parameters:** `max_lr` — peak learning rate (required). `total_steps` — total number of steps (default -1, meaning derive from `epochs * steps_per_epoch`). `epochs` / `steps_per_epoch` — alternative way to specify the total. `pct_start` — fraction of the cycle spent increasing, default 0.3. `anneal_strategy` — `"cos"` or `"linear"`, default `"cos"`. `div_factor` — initial lr is `max_lr / div_factor`, default 25.0. `final_div_factor` — final lr is `max_lr / (div_factor * final_div_factor)`, default 10000.0.
- **Behavior:** lr rises from `max_lr / div_factor` to `max_lr` over the first `pct_start` of steps, then anneals to near zero.
- **Edge cases:** exactly one of `total_steps` or (`epochs` and `steps_per_epoch`) must be given; the factory form requires `total_steps` as a positional argument and has no `epochs` / `steps_per_epoch` parameters. **Schedule is one step later than PyTorch.** Measured with `max_lr=0.1, total_steps=10, pct_start=0.3` (base lr 0.004): LiteTorch reports `[0.028, 0.076, 0.1, 0.09504846, 0.08117457, 0.0611262, 0.0388742, 0.01882583, 0.00495194, 0.0000004]` over 10 `step()` calls, PyTorch `[0.052, 0.1, 0.09504846, 0.08117457, 0.0611262, 0.0388742, 0.01882583, 0.00495194, 0.0000004, 0.00495194]`. From the third call onward LiteTorch equals PyTorch's previous call; on the ramp LiteTorch takes 3 calls to reach `max_lr` where PyTorch takes 2; after the final call LiteTorch lands on the floor (`max_lr / (div_factor * final_div_factor)`) while PyTorch re-emits the previous value. Same curve, one-step phase shift — do not assume identical per-step learning rates when porting.
- **PyTorch:** `torch.optim.lr_scheduler.OneCycleLR` — identical schedule.

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

- **Signature:** `LambdaLR(optimizer, lr_lambda) -> LambdaLR`
- **Source:** `src/bindings/bind_optim.cpp:155-156`; factory `src/bindings/bind_optim.cpp:438`
- **Parameters:** `lr_lambda` — a Python callable `epoch -> multiplier` (required).
- **Behavior:** `lr = base_lr * lr_lambda(step)` at each step.
- **PyTorch:** `torch.optim.lr_scheduler.LambdaLR` — identical schedule.

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

- **Signature:** `MultiStepLR(optimizer, milestones, gamma=0.1) -> MultiStepLR`
- **Source:** `src/bindings/bind_optim.cpp:352-353`; factory `src/bindings/bind_optim.cpp:435`
- **Parameters:** `milestones` — list of step indices at which the rate decays (required). `gamma` — decay factor, default 0.1.
- **Behavior:** `lr *= gamma` each time the step count reaches a milestone.
- **Methods:** `step()`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **PyTorch:** `torch.optim.lr_scheduler.MultiStepLR` — identical schedule.

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

- **Signature:** `ExponentialLR(optimizer, gamma) -> ExponentialLR`
- **Source:** `src/bindings/bind_optim.cpp:380-381`; factory `src/bindings/bind_optim.cpp:432`
- **Parameters:** `gamma` — multiplicative decay per step (required, no default).
- **Behavior:** `lr *= gamma` every step.
- **Methods:** `step()`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **PyTorch:** `torch.optim.lr_scheduler.ExponentialLR` — identical schedule.

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

- **Signature:** `ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8) -> ReduceLROnPlateau`
- **Source:** `src/bindings/bind_optim.cpp:399-400`; factory `src/bindings/bind_optim.cpp:426`
- **Parameters:** `mode` — `"min"` (reduce when the metric stops decreasing) or `"max"`, default `"min"`. `factor` — decay factor, default 0.1. `patience` — number of non-improving calls before reducing, default 10. `threshold` — improvement threshold, default 1e-4. `threshold_mode` — `"rel"` or `"abs"`, default `"rel"`. `cooldown` — steps to wait after a reduction, default 0. `min_lr` — lower bound, default 0.0. `eps` — minimum decay applied, default 1e-8.
- **Behavior:** call `step(metric)` once per validation; the rate is multiplied by `factor` when the metric has not improved for `patience` calls.
- **Methods:** `step(metric)`, `get_last_lr()`, `state_dict()`, `load_state_dict(state_dict)`.
- **Edge cases:** unlike the other schedulers, `step()` takes a required `metric` argument. Do not call it per training step.
- **PyTorch:** `torch.optim.lr_scheduler.ReduceLROnPlateau` — identical behavior.

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

- **Signature:** `lt.optim.lr_scheduler.LinearWarmup(optimizer, warmup_steps, total_steps, min_lr=0.0) -> LinearWarmup`
- **Source:** `src/bindings/bind_optim.cpp:174-175` (class only, under `lr_scheduler`; there is no `lt.optim.LinearWarmup` and no factory form)
- **Parameters:** `warmup_steps` — steps of linear warmup (required). `total_steps` — total steps (required). `min_lr` — floor, default 0.0.
- **Behavior:** lr rises linearly from `min_lr` to the base rate over `warmup_steps`, then decays linearly to `min_lr` by `total_steps`.
- **Methods:** `step()`, `get_lr()`.
- **PyTorch:** no equivalent scheduler in `torch.optim.lr_scheduler`.

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

### `lr_scheduler` factories

- **Signature:** `lt.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)`, `.CosineAnnealingLR(optimizer, T_max, eta_min=0.0)`, `.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0)`, `.LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5)`, `.PolynomialLR(optimizer, total_iters=5, power=1.0)`, `.OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=10000.0)`, `.LambdaLR(optimizer, lr_lambda)`, `.MultiStepLR(optimizer, milestones, gamma=0.1)`, `.ExponentialLR(optimizer, gamma)`, `.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8)`
- **Source:** `src/bindings/bind_optim.cpp:426-453`
- **Behavior:** factory functions returning the same scheduler objects as the classes under `lt.optim`. Differences from the class forms: the `OneCycleLR` factory requires `total_steps` positionally and has no `epochs` / `steps_per_epoch` parameters.
- **PyTorch:** `torch.optim.lr_scheduler` — same names, same schedules.

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

- **Signature:** `optim.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True) -> GradScaler`
- **Source:** `src/bindings/bind_optim.cpp:455-456`
- **Parameters:** `init_scale` — initial loss scale, default 65536.0. `growth_factor` — scale multiplier on a clean interval, default 2.0. `backoff_factor` — scale multiplier after overflow, default 0.5. `growth_interval` — consecutive clean steps before growing, default 2000. `enabled` — whether scaling is active, default True.
- **Methods:** `scale(loss)` — returns the loss multiplied by the current scale; `scale_loss(loss)` — alias with the same behavior, present only on `optim.GradScaler` (not on `amp.GradScaler`); `unscale_(optimizer)` — divides the optimizer's gradients by the current scale, in place; `step(optimizer)` — runs `optimizer.step()` unless overflow was detected, in which case the step is skipped; `update()` — grows or backs off the scale based on whether overflow occurred; `get_scale()`; `set_scale(scale)`; `is_enabled()`.
- **Returns:** `scale()` returns a new scaled tensor; `step()` returns nothing.
- **Edge cases:** `step()` after `update()` without a new `scale()` call uses the last overflow state. Scaling is a no-op when `enabled=False`.
- **PyTorch:** `torch.amp.GradScaler` — same API shape; LiteTorch has no `get_growth_tracker()` / `set_growth_tracker()`.

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

- **Signature:** `amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True) -> GradScaler`
- **Source:** `src/bindings/bind_amp.cpp:6-7`
- **Behavior:** the AMP scaler, a subclass of `optim.GradScaler` with the same constructor and methods except `scale_loss` (absent here). This is the scaler to use with `amp.autocast`; `optim.GradScaler` is the same engine without the AMP affiliation.
- **PyTorch:** `torch.amp.GradScaler` — same role.

### `amp.autocast`

- **Signature:** `amp.autocast(device_type="cuda", dtype=None, enabled=True)` — context manager
- **Source:** `src/bindings/bind_amp.cpp:21-22`
- **Parameters:** `device_type` — `"cuda"` (default) or `"cpu"`. `dtype` — target low-precision dtype; when `None`, CPU defaults to BF16 and CUDA to FP16. `enabled` — default True.
- **Behavior:** inside the context, operations run in the autocast dtype; `is_autocast_enabled()` returns True and `get_autocast_dtype()` returns the active dtype. On exit the previous state is restored.
- **Edge cases:** the default `device_type` is `"cuda"` — on a CPU-only machine pass `"cpu"` explicitly. There is no dtype check at entry: passing an unsupported dtype for the device is silently accepted until an op runs.
- **PyTorch:** `torch.autocast` — same shape; PyTorch CPU autocast also defaults to BF16.

### `amp.AutocastGuard`

- **Signature:** `amp.AutocastGuard(enabled=True, dtype=DataType.FP16) -> AutocastGuard`
- **Source:** `src/bindings/bind_amp.cpp:16-17`
- **Parameters:** `enabled` — default True. `dtype` — default FP16.
- **Methods:** static `is_enabled()` and `get_dtype()` report the global autocast state; `amp.is_autocast_enabled()` and `amp.get_autocast_dtype()` are module-level aliases for the same two functions.
- **PyTorch:** no direct equivalent (PyTorch exposes the state only through `torch.is_autocast_enabled()`).

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

## AMP failure modes

1. **Overflow skips the step.** If any gradient is inf or NaN after `unscale_`, `scaler.step(optimizer)` silently skips the update and `update()` halves the scale. Check `get_scale()` dropping as the symptom; do not assume a step happened.
2. **Stepping without `unscale_` applies scaled gradients.** Calling `scaler.step(optimizer)` without `unscale_` first updates with gradients multiplied by the scale (default 65536x) — an immediate divergence. Always run `scale` → `backward` → `unscale_` → `step` → `update` in that order.
3. **CPU autocast is BF16, not FP16.** `autocast("cpu")` with no dtype uses BF16; the `AutocastGuard` default dtype is FP16. Do not assume the guard default matches the autocast default.
4. **`autocast()` defaults to CUDA.** Calling `lt.amp.autocast()` with no arguments on a CPU-only machine selects device type `"cuda"`; pass `"cpu"` explicitly.
5. **NAdam has no GPU step.** `NAdam.step()` on GPU tensors throws; move parameters to CPU or use `Adam`.
