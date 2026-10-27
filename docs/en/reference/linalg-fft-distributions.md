# Linear Algebra, FFT, and Distributions — API Reference

`litetorch.linalg` holds 28 linear-algebra functions, `litetorch.fft` holds 9 Fourier functions, and `litetorch.distributions` holds 20 probability-distribution classes plus `kl_divergence`.

- **Source of truth:** `src/bindings/bind_linalg.cpp`, `src/bindings/bind_fft.cpp`, `src/bindings/bind_distributions.cpp`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline; B5 re-verifies at head). Examples run against the Core D3 extension of 88a4ba1.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Dtype rules (D-07, FIXED):** `linalg` now follows PyTorch promotion: `solve` / `lstsq` / `vecdot` throw on integer, boolean, FP16, or mixed-dtype inputs; `cross` / `inner` / `vdot` reject booleans; `outer` / `kron` use full `result_type` promotion. All `linalg` ops are CPU-only (they go through `to_doubles` / `to_int64s` and `require_cpu`) — GPU tensors throw. See `plan/differences.md`.
- **Complex numbers:** LiteTorch has no complex dtype. FFT functions return real tensors with a trailing dimension of size 2 holding `[real, imag]`.

## `linalg.inv`

- **Signature:** `inv(A) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:6`
- **Parameters:** `A` — square FP32 matrix, shape `[n, n]`.
- **Returns:** the inverse of `A`, shape `[n, n]`, FP32.
- **Edge cases:** singular `A` produces inf/NaN, not an exception. Non-square `A` throws.
- **PyTorch:** `torch.linalg.inv` — identical (verified 1e-4).

<!-- ex: id=r10-inv device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.inv(a).to_vector())
```

## `linalg.solve`

- **Signature:** `solve(A, B) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:7`
- **Parameters:** `A` — square FP32 matrix `[n, n]`. `B` — right-hand side `[n]` or `[n, k]`.
- **Returns:** `X` with `A @ X == B`, same shape as `B`, FP32.
- **Edge cases:** throws on integer, boolean, FP16, or mixed-dtype inputs (D-07 fix). Singular `A` produces inf/NaN.
- **PyTorch:** `torch.linalg.solve` — identical (verified 1e-4).

<!-- ex: id=r10-solve device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
b = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(lt.linalg.solve(a, b).to_vector())
```

## `linalg.cholesky`

- **Signature:** `cholesky(A, upper=False) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:8`
- **Parameters:** `A` — symmetric positive-definite FP32 matrix `[n, n]`. `upper` — return the upper-triangular factor, default False (lower).
- **Returns:** the Cholesky factor `L` with `A == L @ L.T`, shape `[n, n]`.
- **Edge cases:** non-SPD `A` produces NaN (no exception).
- **PyTorch:** `torch.linalg.cholesky(A, upper=upper)` — identical (verified 1e-4).

<!-- ex: id=r10-cholesky device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cholesky(a).to_vector())
print(lt.linalg.cholesky(a, upper=True).to_vector())
```

## `linalg.cholesky_inverse`

- **Signature:** `cholesky_inverse(A) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:9`
- **Parameters:** `A` — symmetric positive-definite FP32 matrix `[n, n]`.
- **Returns:** the inverse of `A`, shape `[n, n]`, computed via its Cholesky factorization.
- **Edge cases:** **different contract from PyTorch.** `torch.cholesky_inverse(u)` takes the Cholesky *factor* `u` of `A`; LiteTorch takes the matrix `A` itself. `lt.linalg.cholesky_inverse(A)` equals `torch.linalg.inv(A)` (verified: `[0.6, -0.2, -0.2, 0.4]` for `A=[[2,1],[1,3]]`), while `lt.linalg.cholesky_inverse(lt.linalg.cholesky(A))` gives `[0.911, -0.407, -0.407, 0.815]` — neither `A^{-1}` nor `torch.cholesky_inverse(L)`. Porting PyTorch code directly gives wrong results; pass the matrix, not its factor.
- **PyTorch:** `torch.cholesky_inverse` — same name, different argument contract (see above).

<!-- ex: id=r10-cholesky-inverse device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cholesky_inverse(a).to_vector())
```

## `linalg.qr`

- **Signature:** `qr(A, mode="reduced") -> (Q, R)` or `R`
- **Source:** `src/bindings/bind_linalg.cpp:11-17`
- **Parameters:** `A` — FP32 matrix `[m, n]`. `mode` — `"reduced"` (default), `"complete"`, or `"r"` (returns only `R`, not a tuple).
- **Returns:** `(Q, R)` with `A == Q @ R`; `Q` is `[m, m]` for `"complete"`, `[m, min(m, n)]` for `"reduced"`; `R` is `[min(m, n), n]` (reduced) or `[m, n]` (complete).
- **Edge cases:** the signs of rows of `Q` (and corresponding columns of `R`) are not canonical — LiteTorch and PyTorch may return opposite signs. Both are valid factorizations; compare `Q @ R` against `A`, not `Q` directly. `mode="r"` changes the return type to a single tensor.
- **PyTorch:** `torch.linalg.qr` — same up to sign flips (verified via reconstruction 1e-4).

<!-- ex: id=r10-qr device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
q, r = lt.linalg.qr(a)
print(q.to_vector(), r.to_vector())
print(lt.linalg.qr(a, mode="r").to_vector())
```

## `linalg.svd`

- **Signature:** `svd(A, full_matrices=True) -> (U, S, Vh)`
- **Source:** `src/bindings/bind_linalg.cpp:18-23`
- **Parameters:** `A` — FP32 matrix `[m, n]`. `full_matrices` — return full `U`/`Vh`, default True.
- **Returns:** `(U, S, Vh)` with `A == U @ diag(S) @ Vh`; `S` is a 1-D tensor of singular values in descending order.
- **Edge cases:** singular vectors are unique only up to sign — LiteTorch and PyTorch may return opposite signs on `U` / `Vh` columns. Compare singular values and the reconstruction, not the raw vectors.
- **PyTorch:** `torch.linalg.svd` — same up to sign flips (verified via reconstruction 1e-4).

<!-- ex: id=r10-svd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
u, s, vh = lt.linalg.svd(a)
print(s.to_vector())
print(u.to_vector(), vh.to_vector())
```

## `linalg.eigh`

- **Signature:** `eigh(A, UPLO="L") -> (eigenvalues, eigenvectors)`
- **Source:** `src/bindings/bind_linalg.cpp:24-29`
- **Parameters:** `A` — symmetric FP32 matrix `[n, n]`. `UPLO` — `"L"` (default) or `"U"`, which triangle holds the data.
- **Returns:** `(w, V)`; `w` is a 1-D tensor of eigenvalues in ascending order, `V` holds eigenvectors as columns.
- **Edge cases:** eigenvector signs are not canonical (LiteTorch and PyTorch may differ by sign). Non-symmetric input uses only the `UPLO` triangle.
- **PyTorch:** `torch.linalg.eigh` — eigenvalues identical (verified 1e-4); eigenvectors up to sign.

<!-- ex: id=r10-eigh device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
w, v = lt.linalg.eigh(a)
print(w.to_vector(), v.to_vector())
```

## `linalg.eigvalsh`

- **Signature:** `eigvalsh(A, UPLO="L") -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:30`
- **Parameters:** `A` — symmetric FP32 matrix `[n, n]`. `UPLO` — `"L"` (default) or `"U"`.
- **Returns:** 1-D tensor of eigenvalues in ascending order.
- **PyTorch:** `torch.linalg.eigvalsh` — identical (verified 1e-4).

<!-- ex: id=r10-eigvalsh device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.eigvalsh(a).to_vector())
```

## `linalg.det`

- **Signature:** `det(A) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:31`
- **Parameters:** `A` — square FP32 matrix `[n, n]`.
- **Returns:** 0-D (scalar) tensor holding the determinant.
- **PyTorch:** `torch.linalg.det` — identical (verified 1e-4).

<!-- ex: id=r10-det device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.det(a).to_vector())
```

## `linalg.slogdet`

- **Signature:** `slogdet(A) -> (sign, logabsdet)`
- **Source:** `src/bindings/bind_linalg.cpp:32-37`
- **Parameters:** `A` — square FP32 matrix `[n, n]`.
- **Returns:** a 2-tuple of 0-D tensors: `sign` (+1, -1, or 0) and `logabsdet` (the natural log of `|det A|`).
- **PyTorch:** `torch.linalg.slogdet` — identical (verified 1e-4).

<!-- ex: id=r10-slogdet device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
sign, logabsdet = lt.linalg.slogdet(a)
print(sign.to_vector(), logabsdet.to_vector())
```

## `linalg.vector_norm`

- **Signature:** `vector_norm(x, ord=None) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:38-44`
- **Parameters:** `x` — FP32 vector. `ord` — norm order as a number, default `None` (2-norm). Accepts 0, 1, 2, inf, -inf, and other p values.
- **Returns:** 0-D tensor holding the norm.
- **PyTorch:** `torch.linalg.vector_norm` — identical (verified 1e-4).

<!-- ex: id=r10-vector-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vector_norm(x).to_vector())
print(lt.linalg.vector_norm(x, ord=1).to_vector())
```

## `linalg.matrix_norm`

- **Signature:** `matrix_norm(A, ord="fro") -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:45`
- **Parameters:** `A` — FP32 matrix. `ord` — a **string**: `"fro"` (default), `"nuc"`, `"1"`, `"2"`, `"inf"`, `"-inf"`, `"-1"`, `"-2"`. A numeric `ord` (e.g. `ord=1`) throws a type error — pass the string form.
- **Returns:** 0-D tensor holding the matrix norm.
- **PyTorch:** `torch.linalg.matrix_norm` — identical (verified 1e-4).

<!-- ex: id=r10-matrix-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_norm(a).to_vector())
print(lt.linalg.matrix_norm(a, ord="1").to_vector())
```

## `linalg.norm`

- **Signature:** `norm(x, ord=None, dim=None) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:46-60`
- **Parameters:** `x` — FP32 tensor. `ord` — norm order (number, or a string such as `"fro"` when `dim` is given). `dim` — dimension(s) to reduce over, default `None` (flattened vector norm).
- **Returns:** the norm; shape depends on `dim` (scalar when `dim` is `None`).
- **Edge cases:** a string `ord` with no `dim` throws (`linalg.norm: string ord needs dim`).
- **PyTorch:** `torch.linalg.norm` — identical (verified 1e-4).

<!-- ex: id=r10-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.norm(x).to_vector())
print(lt.linalg.norm(x, ord=1).to_vector())
```

## `linalg.matrix_rank`

- **Signature:** `matrix_rank(A, tol=-1.0, hermitian=False) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:111-112`
- **Parameters:** `A` — FP32 matrix. `tol` — singular-value threshold, default -1.0 (computed from the matrix size and dtype eps). `hermitian` — assume Hermitian, default False.
- **Returns:** 0-D tensor holding the rank as a float (e.g. `2.0`, not an int).
- **PyTorch:** `torch.linalg.matrix_rank` — same value; PyTorch returns a Python int, LiteTorch returns a 0-D tensor.

<!-- ex: id=r10-matrix-rank device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_rank(a).to_vector())
```

## `linalg.pinv`

- **Signature:** `pinv(A, rcond=1e-15) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:113`
- **Parameters:** `A` — FP32 matrix `[m, n]`. `rcond` — cutoff for small singular values, default 1e-15.
- **Returns:** the Moore-Penrose pseudoinverse, shape `[n, m]`.
- **PyTorch:** `torch.linalg.pinv` — identical (verified 1e-4).

<!-- ex: id=r10-pinv device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.pinv(a).to_vector())
```

## `linalg.lstsq`

- **Signature:** `lstsq(A, B, rcond=-1.0) -> (solution, residuals, rank, singular_values)`
- **Source:** `src/bindings/bind_linalg.cpp:114-120`
- **Parameters:** `A` — FP32 matrix `[m, n]`. `B` — right-hand side `[m]` or `[m, k]`. `rcond` — cutoff, default -1.0 (machine precision).
- **Returns:** a 4-tuple: `solution` (least-squares minimizer), `residuals` (0-D tensor, `0.0` when the system is solved exactly), `rank` (0-D tensor holding the rank as a float), `singular_values` (1-D tensor).
- **Edge cases:** throws on integer, boolean, FP16, or mixed-dtype inputs (D-07 fix). The return shape differs from PyTorch: PyTorch returns `residuals` as an empty tensor for exact fits and `rank` as a Python int; LiteTorch returns both as 0-D tensors (`0.0` and `2.0` in the example).
- **PyTorch:** `torch.linalg.lstsq` — same solution (verified 1e-4); see the return-shape note above.

<!-- ex: id=r10-lstsq device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
b = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
sol, res, rank, sv = lt.linalg.lstsq(a, b)
print(sol.to_vector(), res.to_vector(), rank.to_vector(), sv.to_vector())
```

## `linalg.lu`

- **Signature:** `lu(A) -> (P, L, U)`
- **Source:** `src/bindings/bind_linalg.cpp:121-126`
- **Parameters:** `A` — square FP32 matrix `[n, n]`.
- **Returns:** `(P, L, U)` with `P @ A == L @ U`; `P` is the permutation matrix, `L` unit lower-triangular, `U` upper-triangular.
- **PyTorch:** `torch.linalg.lu` — identical (verified 1e-4).

<!-- ex: id=r10-lu device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
p, l, u = lt.linalg.lu(a)
print(p.to_vector(), l.to_vector(), u.to_vector())
```

## `linalg.cross`

- **Signature:** `cross(a, b, dim=-1) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:127`
- **Parameters:** `a`, `b` — 3-element vectors (or batches of them) along `dim`. `dim` — default -1.
- **Returns:** the cross product, same shape as the inputs.
- **Edge cases:** rejects boolean inputs (D-07 fix).
- **PyTorch:** `torch.linalg.cross` — identical (verified 1e-4).

<!-- ex: id=r10-cross device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 0.0, 0.0], [3], cpu)
b = lt.Tensor.from_vector([0.0, 1.0, 0.0], [3], cpu)
print(lt.linalg.cross(a, b).to_vector())
```

## `linalg.vecdot`

- **Signature:** `vecdot(x, y, dim=-1) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:128`
- **Parameters:** `x`, `y` — FP32 vectors (or batches). `dim` — default -1.
- **Returns:** the dot product reduced over `dim`.
- **Edge cases:** throws on integer, boolean, FP16, or mixed-dtype inputs (D-07 fix).
- **PyTorch:** `torch.linalg.vecdot` — identical (verified 1e-4).

<!-- ex: id=r10-vecdot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
y = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vecdot(x, y).to_vector())
```

## `linalg.multi_dot`

- **Signature:** `multi_dot(tensors) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:129`
- **Parameters:** `tensors` — list of 1-D or 2-D FP32 tensors, in multiplication order.
- **Returns:** the chained matrix product, computed in the optimal parenthesization order.
- **PyTorch:** `torch.linalg.multi_dot` — identical (verified 1e-4).

<!-- ex: id=r10-multi-dot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.multi_dot([a, a]).to_vector())
```

## `linalg.diagonal`

- **Signature:** `diagonal(A, offset=0, dim1=-2, dim2=-1) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:130-131`
- **Parameters:** `A` — FP32 matrix. `offset` — diagonal offset, default 0. `dim1`, `dim2` — the two dimensions forming the matrix, defaults -2, -1.
- **Returns:** 1-D tensor holding the diagonal.
- **PyTorch:** `torch.diagonal` — identical (verified 1e-4).

<!-- ex: id=r10-diagonal device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.diagonal(a).to_vector())
```

## `linalg.trace`

- **Signature:** `trace(A) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:132`
- **Parameters:** `A` — square FP32 matrix `[n, n]`.
- **Returns:** 0-D tensor holding the sum of the diagonal.
- **PyTorch:** `torch.trace` — identical (verified 1e-4). Note `torch.linalg` has no `trace` in torch 2.14.1; use `torch.trace` for the comparison.

<!-- ex: id=r10-trace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.trace(a).to_vector())
```

## `linalg.matrix_power`

- **Signature:** `matrix_power(A, n) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:133`
- **Parameters:** `A` — square FP32 matrix `[n, n]`. `n` — integer power (required, no default); negative `n` inverts first.
- **Returns:** `A` raised to the power `n`, same shape.
- **PyTorch:** `torch.linalg.matrix_power` — identical (verified 1e-4).

<!-- ex: id=r10-matrix-power device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_power(a, 2).to_vector())
print(lt.linalg.matrix_power(a, -1).to_vector())
```

## `linalg.cond`

- **Signature:** `cond(A, p="2") -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:134`
- **Parameters:** `A` — FP32 matrix. `p` — norm order as a string: `"1"`, `"2"`, `"inf"`, `"fro"`, `"-1"`, `"-2"`, `"-inf"`, default `"2"`.
- **Returns:** 0-D tensor holding the condition number.
- **PyTorch:** `torch.linalg.cond` — identical (verified 1e-4).

<!-- ex: id=r10-cond device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cond(a).to_vector())
print(lt.linalg.cond(a, "inf").to_vector())
```

## `linalg.outer`

- **Signature:** `outer(a, b) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:135`
- **Parameters:** `a`, `b` — 1-D FP32 tensors.
- **Returns:** the outer product, shape `[len(a), len(b)]`.
- **Edge cases:** full `result_type` promotion across dtypes (D-07 fix).
- **PyTorch:** `torch.outer` — identical (verified 1e-4).

<!-- ex: id=r10-outer device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.outer(a, b).to_vector())
```

## `linalg.inner`

- **Signature:** `inner(a, b) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:136`
- **Parameters:** `a`, `b` — FP32 tensors; the last dimension is contracted.
- **Returns:** the inner product (scalar for 1-D inputs).
- **Edge cases:** rejects boolean inputs (D-07 fix).
- **PyTorch:** `torch.inner` — identical (verified 1e-4).

<!-- ex: id=r10-inner device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.inner(a, b).to_vector())
```

## `linalg.kron`

- **Signature:** `kron(a, b) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:137`
- **Parameters:** `a`, `b` — FP32 tensors of any shape.
- **Returns:** the Kronecker product.
- **Edge cases:** full `result_type` promotion across dtypes (D-07 fix).
- **PyTorch:** `torch.kron` — identical (verified 1e-4).

<!-- ex: id=r10-kron device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.kron(a, b).to_vector())
```

## `linalg.vdot`

- **Signature:** `vdot(a, b) -> Tensor`
- **Source:** `src/bindings/bind_linalg.cpp:138`
- **Parameters:** `a`, `b` — 1-D FP32 tensors.
- **Returns:** 0-D tensor holding the dot product.
- **Edge cases:** rejects boolean inputs (D-07 fix).
- **PyTorch:** `torch.vdot` — identical (verified 1e-4).

<!-- ex: id=r10-vdot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vdot(a, b).to_vector())
```

## FFT (`litetorch.fft`)

All FFT functions are CPU-only. `norm` is one of `"backward"` (default, no normalization on the forward transform), `"forward"`, or `"ortho"`. Complex results are returned as real tensors with a trailing dimension of size 2: `result[..., 0]` is the real part, `result[..., 1]` the imaginary part. `fft` / `ifft` expect that layout on input as well.

## `fft.fft`

- **Signature:** `fft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:6`
- **Parameters:** `x` — real FP32 tensor, or a complex-layout tensor with trailing dim 2. `n` — transform length, default -1 (input length). `dim` — dimension to transform, default -1. `norm` — default `"backward"`.
- **Returns:** complex-layout tensor, same shape as `x` except the transformed dim is replaced by `[n, 2]` when the input is real (an input of shape `[4]` yields shape `[4, 2]`).
- **PyTorch:** `torch.fft.fft` — same values (verified 1e-4); PyTorch returns a complex tensor, LiteTorch returns the `[..., 2]` real layout.

<!-- ex: id=r10-fft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.fft(x).shape)
print(lt.fft.fft(x).to_vector())
```

## `fft.ifft`

- **Signature:** `ifft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:8`
- **Parameters:** `x` — complex-layout tensor with trailing dim 2. `n`, `dim`, `norm` — as in `fft`.
- **Returns:** real tensor (not complex layout — the inverse of a real signal is real); `ifft(fft(x))` reproduces `x`.
- **PyTorch:** `torch.fft.ifft` — same values (verified 1e-4).

<!-- ex: id=r10-ifft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.ifft(lt.fft.fft(x)).to_vector())
```

## `fft.rfft`

- **Signature:** `rfft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:10`
- **Parameters:** `x` — real FP32 tensor. `n`, `dim`, `norm` — as in `fft`.
- **Returns:** complex-layout tensor with `n // 2 + 1` frequency bins (shape `[3, 2]` for an input of length 4).
- **PyTorch:** `torch.fft.rfft` — same values (verified 1e-4).

<!-- ex: id=r10-rfft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.rfft(x).shape)
print(lt.fft.rfft(x).to_vector())
```

## `fft.irfft`

- **Signature:** `irfft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:12`
- **Parameters:** `x` — complex-layout tensor with trailing dim 2. `n` — output length, default -1 (inferred as `2 * (bins - 1)`). `dim`, `norm` — as in `fft`.
- **Returns:** real tensor; `irfft(rfft(x))` reproduces `x`.
- **PyTorch:** `torch.fft.irfft` — same values (verified 1e-4).

<!-- ex: id=r10-irfft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.irfft(lt.fft.rfft(x)).to_vector())
```

## `fft.fftn`

- **Signature:** `fftn(x, s=[], dims=[], norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:14`
- **Parameters:** `x` — real FP32 tensor. `s` — transform lengths per dimension, default `[]` (input sizes). `dims` — dimensions to transform, default `[]` (all). `norm` — default `"backward"`.
- **Returns:** complex-layout tensor.
- **PyTorch:** `torch.fft.fftn` — same values (verified 1e-4).

<!-- ex: id=r10-fftn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [2, 2], cpu)
print(lt.fft.fftn(x).shape)
print(lt.fft.fftn(x).to_vector())
```

## `fft.ifftn`

- **Signature:** `ifftn(x, s=[], dims=[], norm="backward") -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:16`
- **Parameters:** `x` — complex-layout tensor. `s`, `dims`, `norm` — as in `fftn`.
- **Returns:** real tensor (not complex layout); `ifftn(fftn(x))` reproduces `x`.
- **PyTorch:** `torch.fft.ifftn` — same values (verified 1e-4).

<!-- ex: id=r10-ifftn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [2, 2], cpu)
print(lt.fft.ifftn(lt.fft.fftn(x)).to_vector())
```

## `fft.fftshift`

- **Signature:** `fftshift(x, dims=[]) -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:18`
- **Parameters:** `x` — FP32 tensor. `dims` — dimensions to shift, default `[]` (all).
- **Returns:** tensor with the zero-frequency component moved to the center.
- **PyTorch:** `torch.fft.fftshift` — identical (verified 1e-4).

<!-- ex: id=r10-fftshift device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.0, 1.0, 2.0, 3.0], [4], cpu)
print(lt.fft.fftshift(x).to_vector())
```

## `fft.ifftshift`

- **Signature:** `ifftshift(x, dims=[]) -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:20`
- **Parameters:** `x` — FP32 tensor. `dims` — dimensions to shift, default `[]` (all).
- **Returns:** the inverse of `fftshift`; `ifftshift(fftshift(x))` reproduces `x`.
- **PyTorch:** `torch.fft.ifftshift` — identical (verified 1e-4).

<!-- ex: id=r10-ifftshift device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.0, 1.0, 2.0, 3.0], [4], cpu)
print(lt.fft.ifftshift(lt.fft.fftshift(x)).to_vector())
```

## `fft.fftfreq`

- **Signature:** `fftfreq(n, d=1.0) -> Tensor`
- **Source:** `src/bindings/bind_fft.cpp:22`
- **Parameters:** `n` — window length (required). `d` — sample spacing, default 1.0.
- **Returns:** 1-D tensor of DFT sample frequencies.
- **PyTorch:** `torch.fft.fftfreq` — identical (verified 1e-4).

<!-- ex: id=r10-fftfreq device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
print(lt.fft.fftfreq(4).to_vector())
```

## Distributions (`litetorch.distributions`)

Every distribution class takes tensor parameters (scalars as 1-element tensors or tensors of any broadcastable shape). The base class `Distribution` exposes `sample(sample_shape=None)`, `log_prob(value)`, `entropy()`, `mean()`, `variance()`, and `batch_shape`. `sample([3])` on a scalar-parameter distribution returns shape `[3, 1]` (the sample shape is prepended to the batch shape). Use `lt.manual_seed(n)` for reproducible samples. All parameters are tensors — there are no scalar-float constructors.

### `distributions.Normal`

- **Signature:** `Normal(loc, scale) -> Normal`
- **Source:** `src/bindings/bind_distributions.cpp:24-25`
- **Parameters:** `loc` — mean tensor. `scale` — standard deviation tensor, must be positive.
- **Methods:** `sample(sample_shape=None)`, `log_prob(value)`, `entropy()`, `mean()`, `variance()`.
- **PyTorch:** `torch.distributions.Normal` — `log_prob`, `mean`, `variance`, `entropy` identical (verified 1e-4).

<!-- ex: id=r10-dist-normal device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
n = lt.distributions.Normal(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(n.log_prob(v).to_vector(), n.mean().to_vector(), n.variance().to_vector(), n.entropy().to_vector())
print(n.sample([3]).shape)
```

### `distributions.Uniform`

- **Signature:** `Uniform(low, high) -> Uniform`
- **Source:** `src/bindings/bind_distributions.cpp:26-27`
- **Parameters:** `low`, `high` — bounds, must satisfy `low < high`.
- **PyTorch:** `torch.distributions.Uniform` — `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-uniform device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
u = lt.distributions.Uniform(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([2.0], [1], cpu))
print(u.mean().to_vector(), u.variance().to_vector())
print(u.sample([3]).to_vector())
```

### `distributions.Exponential`

- **Signature:** `Exponential(rate) -> Exponential`
- **Source:** `src/bindings/bind_distributions.cpp:28-29`
- **Parameters:** `rate` — rate tensor, must be positive.
- **PyTorch:** `torch.distributions.Exponential` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-exponential device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
e = lt.distributions.Exponential(lt.Tensor.from_vector([2.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(e.log_prob(v).to_vector(), e.mean().to_vector(), e.variance().to_vector())
```

### `distributions.Laplace`

- **Signature:** `Laplace(loc, scale) -> Laplace`
- **Source:** `src/bindings/bind_distributions.cpp:30-31`
- **Parameters:** `loc` — location tensor. `scale` — scale tensor, must be positive.
- **PyTorch:** `torch.distributions.Laplace` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-laplace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
l = lt.distributions.Laplace(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(l.log_prob(v).to_vector(), l.mean().to_vector(), l.variance().to_vector())
```

### `distributions.Cauchy`

- **Signature:** `Cauchy(loc, scale) -> Cauchy`
- **Source:** `src/bindings/bind_distributions.cpp:32-33`
- **Parameters:** `loc` — location tensor. `scale` — scale tensor, must be positive.
- **Edge cases:** the mean and variance do not exist; `mean()` / `variance()` return NaN.
- **PyTorch:** `torch.distributions.Cauchy` — `log_prob` identical (verified 1e-4).

<!-- ex: id=r10-dist-cauchy device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
c = lt.distributions.Cauchy(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(c.log_prob(v).to_vector())
```

### `distributions.Beta`

- **Signature:** `Beta(alpha, beta) -> Beta`
- **Source:** `src/bindings/bind_distributions.cpp:34-35`
- **Parameters:** `alpha`, `beta` — concentration tensors, must be positive.
- **PyTorch:** `torch.distributions.Beta` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-beta device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
b = lt.distributions.Beta(lt.Tensor.from_vector([2.0], [1], cpu), lt.Tensor.from_vector([5.0], [1], cpu))
v = lt.Tensor.from_vector([0.3], [1], cpu)
print(b.log_prob(v).to_vector(), b.mean().to_vector(), b.variance().to_vector())
```

### `distributions.Gamma`

- **Signature:** `Gamma(alpha, beta) -> Gamma`
- **Source:** `src/bindings/bind_distributions.cpp:36-37`
- **Parameters:** `alpha` — shape tensor, must be positive. `beta` — rate tensor, must be positive. (LiteTorch follows the PyTorch convention: `beta` is the rate, not the scale.)
- **PyTorch:** `torch.distributions.Gamma` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-gamma device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
g = lt.distributions.Gamma(lt.Tensor.from_vector([2.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([1.5], [1], cpu)
print(g.log_prob(v).to_vector(), g.mean().to_vector(), g.variance().to_vector())
```

### `distributions.Dirichlet`

- **Signature:** `Dirichlet(concentration) -> Dirichlet`
- **Source:** `src/bindings/bind_distributions.cpp:38-39`
- **Parameters:** `concentration` — tensor of positive concentrations; its last dimension is the event dimension.
- **PyTorch:** `torch.distributions.Dirichlet` — `log_prob`, `mean` identical (verified 1e-4).

<!-- ex: id=r10-dist-dirichlet device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
d = lt.distributions.Dirichlet(lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu))
v = lt.Tensor.from_vector([0.2, 0.3, 0.5], [3], cpu)
print(d.log_prob(v).to_vector(), d.mean().to_vector())
```

### `distributions.Bernoulli`

- **Signature:** `Bernoulli(probs, from_logits=False) -> Bernoulli`
- **Source:** `src/bindings/bind_distributions.cpp:40-41`
- **Parameters:** `probs` — probability tensor in [0, 1], or logits when `from_logits=True`.
- **PyTorch:** `torch.distributions.Bernoulli` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-bernoulli device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
b = lt.distributions.Bernoulli(lt.Tensor.from_vector([0.3], [1], cpu))
v = lt.Tensor.from_vector([1.0], [1], cpu)
print(b.log_prob(v).to_vector(), b.mean().to_vector(), b.variance().to_vector())
print(b.sample([5]).to_vector())
```

### `distributions.Categorical`

- **Signature:** `Categorical(probs, from_logits=False) -> Categorical`
- **Source:** `src/bindings/bind_distributions.cpp:42-43`
- **Parameters:** `probs` — probability tensor over categories (last dimension), or logits when `from_logits=True`.
- **Returns:** samples are category indices as FP32 tensors (e.g. `2.0`), not integers.
- **PyTorch:** `torch.distributions.Categorical` — `log_prob` identical (verified 1e-4); PyTorch samples are int64 tensors, LiteTorch samples are FP32.

<!-- ex: id=r10-dist-categorical device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
c = lt.distributions.Categorical(lt.Tensor.from_vector([0.2, 0.5, 0.3], [3], cpu))
v = lt.Tensor.from_vector([1.0], [1], cpu)
print(c.log_prob(v).to_vector())
print(c.sample([5]).to_vector())
```

### `distributions.Poisson`

- **Signature:** `Poisson(rate) -> Poisson`
- **Source:** `src/bindings/bind_distributions.cpp:44-45`
- **Parameters:** `rate` — rate tensor, must be positive.
- **PyTorch:** `torch.distributions.Poisson` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-poisson device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
p = lt.distributions.Poisson(lt.Tensor.from_vector([2.0], [1], cpu))
v = lt.Tensor.from_vector([3.0], [1], cpu)
print(p.log_prob(v).to_vector(), p.mean().to_vector(), p.variance().to_vector())
```

### `distributions.Geometric`

- **Signature:** `Geometric(probs) -> Geometric`
- **Source:** `src/bindings/bind_distributions.cpp:46-47`
- **Parameters:** `probs` — success-probability tensor in (0, 1].
- **PyTorch:** `torch.distributions.Geometric` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-geometric device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
g = lt.distributions.Geometric(lt.Tensor.from_vector([0.3], [1], cpu))
v = lt.Tensor.from_vector([2.0], [1], cpu)
print(g.log_prob(v).to_vector(), g.mean().to_vector(), g.variance().to_vector())
```

### `distributions.Binomial`

- **Signature:** `Binomial(total_count, probs) -> Binomial`
- **Source:** `src/bindings/bind_distributions.cpp:48-49`
- **Parameters:** `total_count` — number of trials tensor. `probs` — success-probability tensor.
- **PyTorch:** `torch.distributions.Binomial` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-binomial device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
b = lt.distributions.Binomial(lt.Tensor.from_vector([10.0], [1], cpu), lt.Tensor.from_vector([0.3], [1], cpu))
v = lt.Tensor.from_vector([3.0], [1], cpu)
print(b.log_prob(v).to_vector(), b.mean().to_vector(), b.variance().to_vector())
```

### `distributions.NegativeBinomial`

- **Signature:** `NegativeBinomial(total_count, probs) -> NegativeBinomial`
- **Source:** `src/bindings/bind_distributions.cpp:50-51`
- **Parameters:** `total_count` — number of failures tensor. `probs` — success-probability tensor.
- **Edge cases:** **the `probs` parameter is swapped relative to PyTorch.** LiteTorch treats `probs` as the probability of *failure*, PyTorch as the probability of *success*. Measured: `NegativeBinomial(5, 0.3).log_prob(3)` is `-3.534541` in LiteTorch vs `-1.839945` in PyTorch; LiteTorch `mean` is `11.667` (torch `2.143`) and `variance` `38.889` (torch `3.061`). LiteTorch `NegativeBinomial(r, p)` equals PyTorch `NegativeBinomial(r, 1 - p)` exactly (`log_prob(3)` with `p=0.7` gives `-1.839945` in LiteTorch, matching torch with `p=0.3`). To port PyTorch code, pass `1 - probs`.
- **PyTorch:** `torch.distributions.NegativeBinomial` — **DIVERGES** as above; same name and constructor, opposite `probs` convention.

<!-- ex: id=r10-dist-negbinomial device=cpu backend=cpu runtime=~1s torch-parity=parity-fail-negbinomial status=parity-fail-negbinomial -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
b = lt.distributions.NegativeBinomial(lt.Tensor.from_vector([5.0], [1], cpu), lt.Tensor.from_vector([0.3], [1], cpu))
v = lt.Tensor.from_vector([3.0], [1], cpu)
print(b.log_prob(v).to_vector(), b.mean().to_vector(), b.variance().to_vector())
```

### `distributions.StudentT`

- **Signature:** `StudentT(df) -> StudentT`
- **Source:** `src/bindings/bind_distributions.cpp:52-53`
- **Parameters:** `df` — degrees-of-freedom tensor, must be positive.
- **PyTorch:** `torch.distributions.StudentT` — `log_prob` identical (verified 1e-4); `mean()` is NaN for `df <= 1`, `variance()` is NaN for `df <= 2`.

<!-- ex: id=r10-dist-studentt device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
t = lt.distributions.StudentT(lt.Tensor.from_vector([5.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(t.log_prob(v).to_vector(), t.mean().to_vector(), t.variance().to_vector())
```

### `distributions.Chi2`

- **Signature:** `Chi2(df) -> Chi2`
- **Source:** `src/bindings/bind_distributions.cpp:54-55`
- **Parameters:** `df` — degrees-of-freedom tensor, must be positive.
- **PyTorch:** `torch.distributions.Chi2` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-chi2 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
c = lt.distributions.Chi2(lt.Tensor.from_vector([3.0], [1], cpu))
v = lt.Tensor.from_vector([2.0], [1], cpu)
print(c.log_prob(v).to_vector(), c.mean().to_vector(), c.variance().to_vector())
```

### `distributions.LogNormal`

- **Signature:** `LogNormal(loc, scale) -> LogNormal`
- **Source:** `src/bindings/bind_distributions.cpp:56-57`
- **Parameters:** `loc` — mean of the underlying normal. `scale` — standard deviation of the underlying normal, must be positive.
- **PyTorch:** `torch.distributions.LogNormal` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-lognormal device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
l = lt.distributions.LogNormal(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([1.0], [1], cpu)
print(l.log_prob(v).to_vector(), l.mean().to_vector(), l.variance().to_vector())
```

### `distributions.Gumbel`

- **Signature:** `Gumbel(loc, scale) -> Gumbel`
- **Source:** `src/bindings/bind_distributions.cpp:58-59`
- **Parameters:** `loc` — location tensor. `scale` — scale tensor, must be positive.
- **PyTorch:** `torch.distributions.Gumbel` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-gumbel device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
g = lt.distributions.Gumbel(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
v = lt.Tensor.from_vector([0.5], [1], cpu)
print(g.log_prob(v).to_vector(), g.mean().to_vector(), g.variance().to_vector())
```

### `distributions.Weibull`

- **Signature:** `Weibull(scale, concentration) -> Weibull`
- **Source:** `src/bindings/bind_distributions.cpp:60-61`
- **Parameters:** `scale` — scale tensor, must be positive. `concentration` — shape tensor, must be positive.
- **PyTorch:** `torch.distributions.Weibull` — `log_prob`, `mean`, `variance` identical (verified 1e-4).

<!-- ex: id=r10-dist-weibull device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
w = lt.distributions.Weibull(lt.Tensor.from_vector([1.0], [1], cpu), lt.Tensor.from_vector([1.5], [1], cpu))
v = lt.Tensor.from_vector([1.0], [1], cpu)
print(w.log_prob(v).to_vector(), w.mean().to_vector(), w.variance().to_vector())
```

### `distributions.MultivariateNormal`

- **Signature:** `MultivariateNormal(loc, covariance_matrix) -> MultivariateNormal`
- **Source:** `src/bindings/bind_distributions.cpp:62-63`
- **Parameters:** `loc` — mean vector tensor `[n]`. `covariance_matrix` — SPD covariance tensor `[n, n]`.
- **PyTorch:** `torch.distributions.MultivariateNormal` — `log_prob`, `mean` identical (verified 1e-4).

<!-- ex: id=r10-dist-mvn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.distributions.MultivariateNormal(lt.Tensor.from_vector([0.0, 0.0], [2], cpu), lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu))
v = lt.Tensor.from_vector([0.5, -0.5], [2], cpu)
print(m.log_prob(v).to_vector(), m.mean().to_vector())
```

### `distributions.kl_divergence`

- **Signature:** `kl_divergence(p, q) -> Tensor`
- **Source:** `src/bindings/bind_distributions.cpp:64`
- **Parameters:** `p`, `q` — distribution objects.
- **Returns:** 0-D tensor holding `KL(p || q)`.
- **Edge cases:** only Normal-Normal and Exponential-Exponential pairs are supported; any other pair throws `kl_divergence: only Normal-Normal and Exponential-Exponential pairs supported`. There is no registered-pair table as in PyTorch — `Beta`, `Gamma`, `Dirichlet`, `Bernoulli`, `Categorical`, and all others throw.
- **PyTorch:** `torch.distributions.kl_divergence` — identical for the two supported pairs (verified 1e-4); PyTorch supports many more pairs.

<!-- ex: id=r10-kl device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
p = lt.distributions.Normal(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
q = lt.distributions.Normal(lt.Tensor.from_vector([1.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu))
print(lt.distributions.kl_divergence(p, q).to_vector())
try:
    lt.distributions.kl_divergence(p, lt.distributions.Uniform(lt.Tensor.from_vector([0.0], [1], cpu), lt.Tensor.from_vector([1.0], [1], cpu)))
except Exception as e:
    print(type(e).__name__, str(e)[:70])
```
