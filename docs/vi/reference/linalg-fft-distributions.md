# Đại số tuyến tính, FFT và Phân phối — Tài liệu tham khảo API

`litetorch.linalg` chứa 28 hàm đại số tuyến tính, `litetorch.fft` chứa 9 hàm Fourier, và `litetorch.distributions` chứa 20 lớp phân phối xác suất cộng `kl_divergence`.

- **Nguồn xác thực:** `src/bindings/bind_linalg.cpp`, `src/bindings/bind_fft.cpp`, `src/bindings/bind_distributions.cpp`.
- **Cây mã:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline của tài liệu; B5 xác minh lại tại head). Các ví dụ chạy trên extension D3 của Core tại commit 88a4ba1.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Tính tương đương với PyTorch là FP32 1e-4 trừ khi header ghi khác.
- **Quy tắc kiểu dữ liệu (D-07, ĐÃ SỬA):** `linalg` nay tuân theo promotion của PyTorch: `solve` / `lstsq` / `vecdot` ném lỗi với đầu vào nguyên, boolean, FP16 hoặc kiểu dữ liệu hỗn hợp; `cross` / `inner` / `vdot` từ chối boolean; `outer` / `kron` dùng promotion `result_type` đầy đủ. Mọi op `linalg` chỉ chạy trên CPU (chúng đi qua `to_doubles` / `to_int64s` và `require_cpu`) — tensor GPU ném lỗi. Xem `plan/differences.md`.
- **Số phức:** LiteTorch không có kiểu dữ liệu số phức. Các hàm FFT trả về tensor thực với chiều cuối có kích thước 2 chứa `[phần thực, phần ảo]`.

## `linalg.inv`

- **Chữ ký:** `inv(A) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:6`
- **Tham số:** `A` — ma trận vuông FP32, shape `[n, n]`.
- **Trả về:** nghịch đảo của `A`, shape `[n, n]`, FP32.
- **Trường hợp biên:** `A` suy biến cho ra inf/NaN, không phải ngoại lệ. `A` không vuông ném lỗi.
- **PyTorch:** `torch.linalg.inv` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-inv device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.inv(a).to_vector())
```

## `linalg.solve`

- **Chữ ký:** `solve(A, B) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:7`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`. `B` — vế phải `[n]` hoặc `[n, k]`.
- **Trả về:** `X` với `A @ X == B`, cùng shape với `B`, FP32.
- **Trường hợp biên:** ném lỗi với đầu vào nguyên, boolean, FP16 hoặc kiểu dữ liệu hỗn hợp (bản sửa D-07). `A` suy biến cho ra inf/NaN.
- **PyTorch:** `torch.linalg.solve` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-solve device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
b = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(lt.linalg.solve(a, b).to_vector())
```

## `linalg.cholesky`

- **Chữ ký:** `cholesky(A, upper=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:8`
- **Tham số:** `A` — ma trận FP32 đối xứng xác định dương `[n, n]`. `upper` — trả về thừa số tam giác trên, mặc định False (tam giác dưới).
- **Trả về:** thừa số Cholesky `L` với `A == L @ L.T`, shape `[n, n]`.
- **Trường hợp biên:** `A` không SPD cho ra NaN (không có ngoại lệ).
- **PyTorch:** `torch.linalg.cholesky(A, upper=upper)` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-cholesky device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cholesky(a).to_vector())
print(lt.linalg.cholesky(a, upper=True).to_vector())
```

## `linalg.cholesky_inverse`

- **Chữ ký:** `cholesky_inverse(A) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:9`
- **Tham số:** `A` — ma trận FP32 đối xứng xác định dương `[n, n]`.
- **Trả về:** nghịch đảo của `A`, shape `[n, n]`, tính qua phân tích Cholesky của nó.
- **Trường hợp biên:** **hợp đồng khác với PyTorch.** `torch.cholesky_inverse(u)` nhận thừa số Cholesky `u` của `A`; LiteTorch nhận chính ma trận `A`. `lt.linalg.cholesky_inverse(A)` bằng `torch.linalg.inv(A)` (đã xác minh: `[0.6, -0.2, -0.2, 0.4]` với `A=[[2,1],[1,3]]`), trong khi `lt.linalg.cholesky_inverse(lt.linalg.cholesky(A))` cho `[0.911, -0.407, -0.407, 0.815]` — không phải `A^{-1}` cũng không phải `torch.cholesky_inverse(L)`. Chuyển code PyTorch trực tiếp cho kết quả sai; truyền ma trận, không phải thừa số của nó.
- **PyTorch:** `torch.cholesky_inverse` — cùng tên, hợp đồng đối số khác (xem trên).

<!-- ex: id=r10-cholesky-inverse device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cholesky_inverse(a).to_vector())
```

## `linalg.qr`

- **Chữ ký:** `qr(A, mode="reduced") -> (Q, R)` hoặc `R`
- **Nguồn:** `src/bindings/bind_linalg.cpp:11-17`
- **Tham số:** `A` — ma trận FP32 `[m, n]`. `mode` — `"reduced"` (mặc định), `"complete"`, hoặc `"r"` (chỉ trả về `R`, không phải tuple).
- **Trả về:** `(Q, R)` với `A == Q @ R`; `Q` là `[m, m]` với `"complete"`, `[m, min(m, n)]` với `"reduced"`; `R` là `[min(m, n), n]` (reduced) hoặc `[m, n]` (complete).
- **Trường hợp biên:** dấu của các hàng của `Q` (và các cột tương ứng của `R`) không chuẩn tắc — LiteTorch và PyTorch có thể trả về dấu đối nhau. Cả hai đều là phân tích hợp lệ; so sánh `Q @ R` với `A`, không so sánh trực tiếp `Q`. `mode="r"` đổi kiểu trả về thành một tensor đơn.
- **PyTorch:** `torch.linalg.qr` — giống hệt trừ dấu (đã xác minh qua tái cấu trúc 1e-4).

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

- **Chữ ký:** `svd(A, full_matrices=True) -> (U, S, Vh)`
- **Nguồn:** `src/bindings/bind_linalg.cpp:18-23`
- **Tham số:** `A` — ma trận FP32 `[m, n]`. `full_matrices` — trả về `U`/`Vh` đầy đủ, mặc định True.
- **Trả về:** `(U, S, Vh)` với `A == U @ diag(S) @ Vh`; `S` là tensor 1-D các giá trị suy biến theo thứ tự giảm dần.
- **Trường hợp biên:** vector suy biến chỉ duy nhất tới dấu — LiteTorch và PyTorch có thể trả về dấu đối nhau trên các cột của `U` / `Vh`. So sánh giá trị suy biến và phép tái cấu trúc, không so sánh vector thô.
- **PyTorch:** `torch.linalg.svd` — giống hệt trừ dấu (đã xác minh qua tái cấu trúc 1e-4).

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

- **Chữ ký:** `eigh(A, UPLO="L") -> (eigenvalues, eigenvectors)`
- **Nguồn:** `src/bindings/bind_linalg.cpp:24-29`
- **Tham số:** `A` — ma trận FP32 đối xứng `[n, n]`. `UPLO` — `"L"` (mặc định) hoặc `"U"`, tam giác nào chứa dữ liệu.
- **Trả về:** `(w, V)`; `w` là tensor 1-D các trị riêng theo thứ tự tăng dần, `V` chứa các vector riêng theo cột.
- **Trường hợp biên:** dấu của vector riêng không chuẩn tắc (LiteTorch và PyTorch có thể lệch dấu). Đầu vào không đối xứng chỉ dùng tam giác `UPLO`.
- **PyTorch:** `torch.linalg.eigh` — trị riêng giống hệt (đã xác minh 1e-4); vector riêng giống trừ dấu.

<!-- ex: id=r10-eigh device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
w, v = lt.linalg.eigh(a)
print(w.to_vector(), v.to_vector())
```

## `linalg.eigvalsh`

- **Chữ ký:** `eigvalsh(A, UPLO="L") -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:30`
- **Tham số:** `A` — ma trận FP32 đối xứng `[n, n]`. `UPLO` — `"L"` (mặc định) hoặc `"U"`.
- **Trả về:** tensor 1-D các trị riêng theo thứ tự tăng dần.
- **PyTorch:** `torch.linalg.eigvalsh` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-eigvalsh device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.eigvalsh(a).to_vector())
```

## `linalg.det`

- **Chữ ký:** `det(A) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:31`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`.
- **Trả về:** tensor 0-D chứa định thức.
- **PyTorch:** `torch.linalg.det` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-det device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.det(a).to_vector())
```

## `linalg.slogdet`

- **Chữ ký:** `slogdet(A) -> (sign, logabsdet)`
- **Nguồn:** `src/bindings/bind_linalg.cpp:32-37`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`.
- **Trả về:** tuple 2 tensor 0-D: `sign` (+1, -1 hoặc 0) và `logabsdet` (log tự nhiên của `|det A|`).
- **PyTorch:** `torch.linalg.slogdet` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-slogdet device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
sign, logabsdet = lt.linalg.slogdet(a)
print(sign.to_vector(), logabsdet.to_vector())
```

## `linalg.vector_norm`

- **Chữ ký:** `vector_norm(x, ord=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:38-44`
- **Tham số:** `x` — vector FP32. `ord` — bậc chuẩn dưới dạng số, mặc định `None` (chuẩn 2). Chấp nhận 0, 1, 2, inf, -inf và các giá trị p khác.
- **Trả về:** tensor 0-D chứa chuẩn.
- **PyTorch:** `torch.linalg.vector_norm` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-vector-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vector_norm(x).to_vector())
print(lt.linalg.vector_norm(x, ord=1).to_vector())
```

## `linalg.matrix_norm`

- **Chữ ký:** `matrix_norm(A, ord="fro") -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:45`
- **Tham số:** `A` — ma trận FP32. `ord` — một **chuỗi**: `"fro"` (mặc định), `"nuc"`, `"1"`, `"2"`, `"inf"`, `"-inf"`, `"-1"`, `"-2"`. `ord` dạng số (ví dụ `ord=1`) sẽ ném lỗi kiểu — truyền dạng chuỗi.
- **Trả về:** tensor 0-D chứa chuẩn ma trận.
- **PyTorch:** `torch.linalg.matrix_norm` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-matrix-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_norm(a).to_vector())
print(lt.linalg.matrix_norm(a, ord="1").to_vector())
```

## `linalg.norm`

- **Chữ ký:** `norm(x, ord=None, dim=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:46-60`
- **Tham số:** `x` — tensor FP32. `ord` — bậc chuẩn (số, hoặc chuỗi như `"fro"` khi `dim` được cho). `dim` — chiều cần rút gọn, mặc định `None` (chuẩn vector trên tensor đã làm phẳng).
- **Trả về:** chuẩn; shape phụ thuộc `dim` (vô hướng khi `dim` là `None`).
- **Trường hợp biên:** `ord` dạng chuỗi mà không có `dim` sẽ ném lỗi (`linalg.norm: string ord needs dim`).
- **PyTorch:** `torch.linalg.norm` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.norm(x).to_vector())
print(lt.linalg.norm(x, ord=1).to_vector())
```

## `linalg.matrix_rank`

- **Chữ ký:** `matrix_rank(A, tol=-1.0, hermitian=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:111-112`
- **Tham số:** `A` — ma trận FP32. `tol` — ngưỡng giá trị suy biến, mặc định -1.0 (tính từ kích thước ma trận và eps của kiểu dữ liệu). `hermitian` — giả định Hermitian, mặc định False.
- **Trả về:** tensor 0-D chứa hạng dưới dạng số thực (ví dụ `2.0`, không phải int).
- **PyTorch:** `torch.linalg.matrix_rank` — cùng giá trị; PyTorch trả về int Python, LiteTorch trả về tensor 0-D.

<!-- ex: id=r10-matrix-rank device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_rank(a).to_vector())
```

## `linalg.pinv`

- **Chữ ký:** `pinv(A, rcond=1e-15) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:113`
- **Tham số:** `A` — ma trận FP32 `[m, n]`. `rcond` — ngưỡng cắt cho giá trị suy biến nhỏ, mặc định 1e-15.
- **Trả về:** nghịch đảo giả Moore-Penrose, shape `[n, m]`.
- **PyTorch:** `torch.linalg.pinv` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-pinv device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.pinv(a).to_vector())
```

## `linalg.lstsq`

- **Chữ ký:** `lstsq(A, B, rcond=-1.0) -> (solution, residuals, rank, singular_values)`
- **Nguồn:** `src/bindings/bind_linalg.cpp:114-120`
- **Tham số:** `A` — ma trận FP32 `[m, n]`. `B` — vế phải `[m]` hoặc `[m, k]`. `rcond` — ngưỡng cắt, mặc định -1.0 (độ chính xác máy).
- **Trả về:** tuple 4 phần tử: `solution` (nghiệm bình phương tối thiểu), `residuals` (tensor 0-D, `0.0` khi hệ được giải chính xác), `rank` (tensor 0-D chứa hạng dưới dạng số thực), `singular_values` (tensor 1-D).
- **Trường hợp biên:** ném lỗi với đầu vào nguyên, boolean, FP16 hoặc kiểu dữ liệu hỗn hợp (bản sửa D-07). Hình dạng trả về khác PyTorch: PyTorch trả về `residuals` là tensor rỗng khi giải chính xác và `rank` là int Python; LiteTorch trả về cả hai dưới dạng tensor 0-D (`0.0` và `2.0` trong ví dụ).
- **PyTorch:** `torch.linalg.lstsq` — cùng nghiệm (đã xác minh 1e-4); xem ghi chú về hình dạng trả về ở trên.

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

- **Chữ ký:** `lu(A) -> (P, L, U)`
- **Nguồn:** `src/bindings/bind_linalg.cpp:121-126`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`.
- **Trả về:** `(P, L, U)` với `P @ A == L @ U`; `P` là ma trận hoán vị, `L` tam giác dưới đơn vị, `U` tam giác trên.
- **PyTorch:** `torch.linalg.lu` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-lu device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
p, l, u = lt.linalg.lu(a)
print(p.to_vector(), l.to_vector(), u.to_vector())
```

## `linalg.cross`

- **Chữ ký:** `cross(a, b, dim=-1) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:127`
- **Tham số:** `a`, `b` — vector 3 phần tử (hoặc batch của chúng) dọc `dim`. `dim` — mặc định -1.
- **Trả về:** tích có hướng, cùng shape với đầu vào.
- **Trường hợp biên:** từ chối đầu vào boolean (bản sửa D-07).
- **PyTorch:** `torch.linalg.cross` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-cross device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 0.0, 0.0], [3], cpu)
b = lt.Tensor.from_vector([0.0, 1.0, 0.0], [3], cpu)
print(lt.linalg.cross(a, b).to_vector())
```

## `linalg.vecdot`

- **Chữ ký:** `vecdot(x, y, dim=-1) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:128`
- **Tham số:** `x`, `y` — vector FP32 (hoặc batch). `dim` — mặc định -1.
- **Trả về:** tích vô hướng rút gọn theo `dim`.
- **Trường hợp biên:** ném lỗi với đầu vào nguyên, boolean, FP16 hoặc kiểu dữ liệu hỗn hợp (bản sửa D-07).
- **PyTorch:** `torch.linalg.vecdot` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-vecdot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
y = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vecdot(x, y).to_vector())
```

## `linalg.multi_dot`

- **Chữ ký:** `multi_dot(tensors) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:129`
- **Tham số:** `tensors` — danh sách tensor FP32 1-D hoặc 2-D, theo thứ tự nhân.
- **Trả về:** tích ma trận chuỗi, tính theo thứ tự đặt dấu ngoặc tối ưu.
- **PyTorch:** `torch.linalg.multi_dot` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-multi-dot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.multi_dot([a, a]).to_vector())
```

## `linalg.diagonal`

- **Chữ ký:** `diagonal(A, offset=0, dim1=-2, dim2=-1) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:130-131`
- **Tham số:** `A` — ma trận FP32. `offset` — độ lệch đường chéo, mặc định 0. `dim1`, `dim2` — hai chiều tạo thành ma trận, mặc định -2, -1.
- **Trả về:** tensor 1-D chứa đường chéo.
- **PyTorch:** `torch.diagonal` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-diagonal device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.diagonal(a).to_vector())
```

## `linalg.trace`

- **Chữ ký:** `trace(A) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:132`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`.
- **Trả về:** tensor 0-D chứa tổng đường chéo.
- **PyTorch:** `torch.trace` — giống hệt (đã xác minh 1e-4). Lưu ý `torch.linalg` không có `trace` trong torch 2.14.1; dùng `torch.trace` để so sánh.

<!-- ex: id=r10-trace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.trace(a).to_vector())
```

## `linalg.matrix_power`

- **Chữ ký:** `matrix_power(A, n) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:133`
- **Tham số:** `A` — ma trận vuông FP32 `[n, n]`. `n` — số mũ nguyên (bắt buộc, không có mặc định); `n` âm sẽ nghịch đảo trước.
- **Trả về:** `A` lũy thừa `n`, cùng shape.
- **PyTorch:** `torch.linalg.matrix_power` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-matrix-power device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.matrix_power(a, 2).to_vector())
print(lt.linalg.matrix_power(a, -1).to_vector())
```

## `linalg.cond`

- **Chữ ký:** `cond(A, p="2") -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:134`
- **Tham số:** `A` — ma trận FP32. `p` — bậc chuẩn dưới dạng chuỗi: `"1"`, `"2"`, `"inf"`, `"fro"`, `"-1"`, `"-2"`, `"-inf"`, mặc định `"2"`.
- **Trả về:** tensor 0-D chứa số điều kiện.
- **PyTorch:** `torch.linalg.cond` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-cond device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 1.0, 1.0, 3.0], [2, 2], cpu)
print(lt.linalg.cond(a).to_vector())
print(lt.linalg.cond(a, "inf").to_vector())
```

## `linalg.outer`

- **Chữ ký:** `outer(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:135`
- **Tham số:** `a`, `b` — tensor FP32 1-D.
- **Trả về:** tích ngoài, shape `[len(a), len(b)]`.
- **Trường hợp biên:** promotion `result_type` đầy đủ giữa các kiểu dữ liệu (bản sửa D-07).
- **PyTorch:** `torch.outer` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-outer device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.outer(a, b).to_vector())
```

## `linalg.inner`

- **Chữ ký:** `inner(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:136`
- **Tham số:** `a`, `b` — tensor FP32; chiều cuối được co lại.
- **Trả về:** tích trong (vô hướng với đầu vào 1-D).
- **Trường hợp biên:** từ chối đầu vào boolean (bản sửa D-07).
- **PyTorch:** `torch.inner` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-inner device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.inner(a, b).to_vector())
```

## `linalg.kron`

- **Chữ ký:** `kron(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:137`
- **Tham số:** `a`, `b` — tensor FP32 với shape bất kỳ.
- **Trả về:** tích Kronecker.
- **Trường hợp biên:** promotion `result_type` đầy đủ giữa các kiểu dữ liệu (bản sửa D-07).
- **PyTorch:** `torch.kron` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-kron device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.kron(a, b).to_vector())
```

## `linalg.vdot`

- **Chữ ký:** `vdot(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_linalg.cpp:138`
- **Tham số:** `a`, `b` — tensor FP32 1-D.
- **Trả về:** tensor 0-D chứa tích vô hướng.
- **Trường hợp biên:** từ chối đầu vào boolean (bản sửa D-07).
- **PyTorch:** `torch.vdot` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-vdot device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.linalg.vdot(a, b).to_vector())
```

## FFT (`litetorch.fft`)

Mọi hàm FFT chỉ chạy trên CPU. `norm` là một trong `"backward"` (mặc định, không chuẩn hóa trên phép biến đổi thuận), `"forward"` hoặc `"ortho"`. Kết quả phức được trả về dưới dạng tensor thực với chiều cuối có kích thước 2: `result[..., 0]` là phần thực, `result[..., 1]` là phần ảo. `fft` / `ifft` cũng nhận đầu vào ở layout đó.

## `fft.fft`

- **Chữ ký:** `fft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:6`
- **Tham số:** `x` — tensor thực FP32, hoặc tensor layout phức với chiều cuối bằng 2. `n` — độ dài biến đổi, mặc định -1 (độ dài đầu vào). `dim` — chiều cần biến đổi, mặc định -1. `norm` — mặc định `"backward"`.
- **Trả về:** tensor layout phức, cùng shape với `x` trừ chiều được biến đổi thay bằng `[n, 2]` khi đầu vào là thực (đầu vào shape `[4]` cho shape `[4, 2]`).
- **PyTorch:** `torch.fft.fft` — cùng giá trị (đã xác minh 1e-4); PyTorch trả về tensor phức, LiteTorch trả về layout thực `[..., 2]`.

<!-- ex: id=r10-fft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.fft(x).shape)
print(lt.fft.fft(x).to_vector())
```

## `fft.ifft`

- **Chữ ký:** `ifft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:8`
- **Tham số:** `x` — tensor layout phức với chiều cuối bằng 2. `n`, `dim`, `norm` — như trong `fft`.
- **Trả về:** tensor thực (không phải layout phức — nghịch đảo của tín hiệu thực là thực); `ifft(fft(x))` tái tạo `x`.
- **PyTorch:** `torch.fft.ifft` — cùng giá trị (đã xác minh 1e-4).

<!-- ex: id=r10-ifft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.ifft(lt.fft.fft(x)).to_vector())
```

## `fft.rfft`

- **Chữ ký:** `rfft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:10`
- **Tham số:** `x` — tensor thực FP32. `n`, `dim`, `norm` — như trong `fft`.
- **Trả về:** tensor layout phức với `n // 2 + 1` bin tần số (shape `[3, 2]` với đầu vào dài 4).
- **PyTorch:** `torch.fft.rfft` — cùng giá trị (đã xác minh 1e-4).

<!-- ex: id=r10-rfft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.rfft(x).shape)
print(lt.fft.rfft(x).to_vector())
```

## `fft.irfft`

- **Chữ ký:** `irfft(x, n=-1, dim=-1, norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:12`
- **Tham số:** `x` — tensor layout phức với chiều cuối bằng 2. `n` — độ dài đầu ra, mặc định -1 (suy ra là `2 * (số bin - 1)`). `dim`, `norm` — như trong `fft`.
- **Trả về:** tensor thực; `irfft(rfft(x))` tái tạo `x`.
- **PyTorch:** `torch.fft.irfft` — cùng giá trị (đã xác minh 1e-4).

<!-- ex: id=r10-irfft device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [4], cpu)
print(lt.fft.irfft(lt.fft.rfft(x)).to_vector())
```

## `fft.fftn`

- **Chữ ký:** `fftn(x, s=[], dims=[], norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:14`
- **Tham số:** `x` — tensor thực FP32. `s` — độ dài biến đổi cho mỗi chiều, mặc định `[]` (kích thước đầu vào). `dims` — các chiều cần biến đổi, mặc định `[]` (tất cả). `norm` — mặc định `"backward"`.
- **Trả về:** tensor layout phức.
- **PyTorch:** `torch.fft.fftn` — cùng giá trị (đã xác minh 1e-4).

<!-- ex: id=r10-fftn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [2, 2], cpu)
print(lt.fft.fftn(x).shape)
print(lt.fft.fftn(x).to_vector())
```

## `fft.ifftn`

- **Chữ ký:** `ifftn(x, s=[], dims=[], norm="backward") -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:16`
- **Tham số:** `x` — tensor layout phức. `s`, `dims`, `norm` — như trong `fftn`.
- **Trả về:** tensor thực (không phải layout phức); `ifftn(fftn(x))` tái tạo `x`.
- **PyTorch:** `torch.fft.ifftn` — cùng giá trị (đã xác minh 1e-4).

<!-- ex: id=r10-ifftn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [2, 2], cpu)
print(lt.fft.ifftn(lt.fft.fftn(x)).to_vector())
```

## `fft.fftshift`

- **Chữ ký:** `fftshift(x, dims=[]) -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:18`
- **Tham số:** `x` — tensor FP32. `dims` — các chiều cần dịch, mặc định `[]` (tất cả).
- **Trả về:** tensor với thành phần tần số 0 chuyển về trung tâm.
- **PyTorch:** `torch.fft.fftshift` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-fftshift device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.0, 1.0, 2.0, 3.0], [4], cpu)
print(lt.fft.fftshift(x).to_vector())
```

## `fft.ifftshift`

- **Chữ ký:** `ifftshift(x, dims=[]) -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:20`
- **Tham số:** `x` — tensor FP32. `dims` — các chiều cần dịch, mặc định `[]` (tất cả).
- **Trả về:** nghịch đảo của `fftshift`; `ifftshift(fftshift(x))` tái tạo `x`.
- **PyTorch:** `torch.fft.ifftshift` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-ifftshift device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.0, 1.0, 2.0, 3.0], [4], cpu)
print(lt.fft.ifftshift(lt.fft.fftshift(x)).to_vector())
```

## `fft.fftfreq`

- **Chữ ký:** `fftfreq(n, d=1.0) -> Tensor`
- **Nguồn:** `src/bindings/bind_fft.cpp:22`
- **Tham số:** `n` — độ dài cửa sổ (bắt buộc). `d` — khoảng cách mẫu, mặc định 1.0.
- **Trả về:** tensor 1-D các tần số mẫu DFT.
- **PyTorch:** `torch.fft.fftfreq` — giống hệt (đã xác minh 1e-4).

<!-- ex: id=r10-fftfreq device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
print(lt.fft.fftfreq(4).to_vector())
```

## Phân phối (`litetorch.distributions`)

Mọi lớp phân phối nhận tham số dạng tensor (vô hướng dưới dạng tensor 1 phần tử hoặc tensor với shape bất kỳ có thể broadcast). Lớp cơ sở `Distribution` có `sample(sample_shape=None)`, `log_prob(value)`, `entropy()`, `mean()`, `variance()` và `batch_shape`. `sample([3])` trên phân phối có tham số vô hướng trả về shape `[3, 1]` (shape mẫu được đặt trước batch shape). Dùng `lt.manual_seed(n)` để mẫu tái lập được. Mọi tham số đều là tensor — không có constructor nhận số thực.

### `distributions.Normal`

- **Chữ ký:** `Normal(loc, scale) -> Normal`
- **Nguồn:** `src/bindings/bind_distributions.cpp:24-25`
- **Tham số:** `loc` — tensor trung bình. `scale` — tensor độ lệch chuẩn, phải dương.
- **Phương thức:** `sample(sample_shape=None)`, `log_prob(value)`, `entropy()`, `mean()`, `variance()`.
- **PyTorch:** `torch.distributions.Normal` — `log_prob`, `mean`, `variance`, `entropy` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Uniform(low, high) -> Uniform`
- **Nguồn:** `src/bindings/bind_distributions.cpp:26-27`
- **Tham số:** `low`, `high` — cận, phải thỏa `low < high`.
- **PyTorch:** `torch.distributions.Uniform` — `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Exponential(rate) -> Exponential`
- **Nguồn:** `src/bindings/bind_distributions.cpp:28-29`
- **Tham số:** `rate` — tensor tốc độ, phải dương.
- **PyTorch:** `torch.distributions.Exponential` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Laplace(loc, scale) -> Laplace`
- **Nguồn:** `src/bindings/bind_distributions.cpp:30-31`
- **Tham số:** `loc` — tensor vị trí. `scale` — tensor tỉ lệ, phải dương.
- **PyTorch:** `torch.distributions.Laplace` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Cauchy(loc, scale) -> Cauchy`
- **Nguồn:** `src/bindings/bind_distributions.cpp:32-33`
- **Tham số:** `loc` — tensor vị trí. `scale` — tensor tỉ lệ, phải dương.
- **Trường hợp biên:** trung bình và phương sai không tồn tại; `mean()` / `variance()` trả về NaN.
- **PyTorch:** `torch.distributions.Cauchy` — `log_prob` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Beta(alpha, beta) -> Beta`
- **Nguồn:** `src/bindings/bind_distributions.cpp:34-35`
- **Tham số:** `alpha`, `beta` — tensor nồng độ, phải dương.
- **PyTorch:** `torch.distributions.Beta` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Gamma(alpha, beta) -> Gamma`
- **Nguồn:** `src/bindings/bind_distributions.cpp:36-37`
- **Tham số:** `alpha` — tensor hình dạng, phải dương. `beta` — tensor tốc độ, phải dương. (LiteTorch theo quy ước của PyTorch: `beta` là tốc độ, không phải tỉ lệ.)
- **PyTorch:** `torch.distributions.Gamma` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Dirichlet(concentration) -> Dirichlet`
- **Nguồn:** `src/bindings/bind_distributions.cpp:38-39`
- **Tham số:** `concentration` — tensor nồng độ dương; chiều cuối là chiều sự kiện.
- **PyTorch:** `torch.distributions.Dirichlet` — `log_prob`, `mean` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Bernoulli(probs, from_logits=False) -> Bernoulli`
- **Nguồn:** `src/bindings/bind_distributions.cpp:40-41`
- **Tham số:** `probs` — tensor xác suất trong [0, 1], hoặc logits khi `from_logits=True`.
- **PyTorch:** `torch.distributions.Bernoulli` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Categorical(probs, from_logits=False) -> Categorical`
- **Nguồn:** `src/bindings/bind_distributions.cpp:42-43`
- **Tham số:** `probs` — tensor xác suất trên các hạng mục (chiều cuối), hoặc logits khi `from_logits=True`.
- **Trả về:** mẫu là chỉ số hạng mục dưới dạng tensor FP32 (ví dụ `2.0`), không phải số nguyên.
- **PyTorch:** `torch.distributions.Categorical` — `log_prob` giống hệt (đã xác minh 1e-4); mẫu của PyTorch là tensor int64, mẫu của LiteTorch là FP32.

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

- **Chữ ký:** `Poisson(rate) -> Poisson`
- **Nguồn:** `src/bindings/bind_distributions.cpp:44-45`
- **Tham số:** `rate` — tensor tốc độ, phải dương.
- **PyTorch:** `torch.distributions.Poisson` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Geometric(probs) -> Geometric`
- **Nguồn:** `src/bindings/bind_distributions.cpp:46-47`
- **Tham số:** `probs` — tensor xác suất thành công trong (0, 1].
- **PyTorch:** `torch.distributions.Geometric` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Binomial(total_count, probs) -> Binomial`
- **Nguồn:** `src/bindings/bind_distributions.cpp:48-49`
- **Tham số:** `total_count` — tensor số lần thử. `probs` — tensor xác suất thành công.
- **PyTorch:** `torch.distributions.Binomial` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `NegativeBinomial(total_count, probs) -> NegativeBinomial`
- **Nguồn:** `src/bindings/bind_distributions.cpp:50-51`
- **Tham số:** `total_count` — tensor số lần thất bại. `probs` — tensor xác suất thành công.
- **Trường hợp biên:** **tham số `probs` bị hoán đổi so với PyTorch.** LiteTorch coi `probs` là xác suất *thất bại*, PyTorch coi là xác suất *thành công*. Đã đo: `NegativeBinomial(5, 0.3).log_prob(3)` là `-3.534541` trong LiteTorch so với `-1.839945` trong PyTorch; `mean` của LiteTorch là `11.667` (torch `2.143`) và `variance` `38.889` (torch `3.061`). `NegativeBinomial(r, p)` của LiteTorch bằng đúng `NegativeBinomial(r, 1 - p)` của PyTorch (`log_prob(3)` với `p=0.7` cho `-1.839945` trong LiteTorch, khớp torch với `p=0.3`). Để chuyển code PyTorch, truyền `1 - probs`.
- **PyTorch:** `torch.distributions.NegativeBinomial` — **LỆCH** như trên; cùng tên và constructor, quy ước `probs` ngược nhau.

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

- **Chữ ký:** `StudentT(df) -> StudentT`
- **Nguồn:** `src/bindings/bind_distributions.cpp:52-53`
- **Tham số:** `df` — tensor bậc tự do, phải dương.
- **PyTorch:** `torch.distributions.StudentT` — `log_prob` giống hệt (đã xác minh 1e-4); `mean()` là NaN với `df <= 1`, `variance()` là NaN với `df <= 2`.

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

- **Chữ ký:** `Chi2(df) -> Chi2`
- **Nguồn:** `src/bindings/bind_distributions.cpp:54-55`
- **Tham số:** `df` — tensor bậc tự do, phải dương.
- **PyTorch:** `torch.distributions.Chi2` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `LogNormal(loc, scale) -> LogNormal`
- **Nguồn:** `src/bindings/bind_distributions.cpp:56-57`
- **Tham số:** `loc` — trung bình của phân phối chuẩn nền. `scale` — độ lệch chuẩn của phân phối chuẩn nền, phải dương.
- **PyTorch:** `torch.distributions.LogNormal` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Gumbel(loc, scale) -> Gumbel`
- **Nguồn:** `src/bindings/bind_distributions.cpp:58-59`
- **Tham số:** `loc` — tensor vị trí. `scale` — tensor tỉ lệ, phải dương.
- **PyTorch:** `torch.distributions.Gumbel` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `Weibull(scale, concentration) -> Weibull`
- **Nguồn:** `src/bindings/bind_distributions.cpp:60-61`
- **Tham số:** `scale` — tensor tỉ lệ, phải dương. `concentration` — tensor hình dạng, phải dương.
- **PyTorch:** `torch.distributions.Weibull` — `log_prob`, `mean`, `variance` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `MultivariateNormal(loc, covariance_matrix) -> MultivariateNormal`
- **Nguồn:** `src/bindings/bind_distributions.cpp:62-63`
- **Tham số:** `loc` — tensor vector trung bình `[n]`. `covariance_matrix` — tensor hiệp phương sai SPD `[n, n]`.
- **PyTorch:** `torch.distributions.MultivariateNormal` — `log_prob`, `mean` giống hệt (đã xác minh 1e-4).

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

- **Chữ ký:** `kl_divergence(p, q) -> Tensor`
- **Nguồn:** `src/bindings/bind_distributions.cpp:64`
- **Tham số:** `p`, `q` — đối tượng phân phối.
- **Trả về:** tensor 0-D chứa `KL(p || q)`.
- **Trường hợp biên:** chỉ cặp Normal-Normal và Exponential-Exponential được hỗ trợ; bất kỳ cặp nào khác đều ném lỗi `kl_divergence: only Normal-Normal and Exponential-Exponential pairs supported`. Không có bảng cặp đã đăng ký như trong PyTorch — `Beta`, `Gamma`, `Dirichlet`, `Bernoulli`, `Categorical` và mọi phân phối khác đều ném lỗi.
- **PyTorch:** `torch.distributions.kl_divergence` — giống hệt với hai cặp được hỗ trợ (đã xác minh 1e-4); PyTorch hỗ trợ nhiều cặp hơn.

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
