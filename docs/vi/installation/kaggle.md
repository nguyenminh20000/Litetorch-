# Cài đặt trên Kaggle

Kaggle là cách được hỗ trợ để chạy LiteTorch trên GPU (2x T4) và TPU (v5) mà không cần phần cứng local. Trang này bao quát thiết lập notebook, lấy mã nguồn, biên dịch, cổng kiểm tra tiền điều kiện GPU, và giữ session sống. Công việc GPU làm trước: hoàn thành mọi công việc GPU trước khi chạm tới TPU.

> Ghi chú phạm vi: các bước GPU trên Kaggle dưới đây (kiểm tra tiền điều kiện GPU, vị trí `liblitetorch_gpu.so`, số lượng thiết bị T4) được viết từ runbook Kaggle của dự án và lịch sử chạy đã kiểm chứng. Chúng không được chạy trên máy tài liệu này vì máy không có GPU; phần GPU là hạng mục kiểm tra của Batch B5 (một gói GPU cho mỗi batch tài liệu, do QA khởi chạy). Đoạn smoke test Python trên trang này chạy được trên CPU local và chạy không đổi trên Kaggle GPU.

---

## 1. Thiết lập notebook

1. Tạo một notebook Kaggle với accelerator **GPU** (2x T4) hoặc, đối với track TPU, accelerator **TPU v5**. Mỗi tài khoản chỉ cho phép một session GPU tương tác — trước khi khởi chạy, hãy xác nhận không có agent đồng đội nào đang chạy trên tài khoản đó; đừng bao giờ dừng session của đồng đội hay xóa cây `/kaggle/working` của họ.
2. Dùng dataset đi kèm (bundled), đừng bao giờ tải từ bên ngoài: các mirror dataset công khai chết mà không báo trước. Hãy ưu tiên dataset đi kèm của Kaggle và viết script để bỏ qua bước tải khi file đã tồn tại.
3. Đăng nhập dùng tài khoản Google đã lưu (cookie được giữ). Nếu màn hình xác minh 2 bước của Google ("Verify it's you") xuất hiện trên session mới: dừng lại tại màn hình đó và báo ngay cho commander — người dùng tự phê duyệt trên thiết bị của họ; đừng bao giờ tự động hóa bước này.

---

## 2. Lấy mã nguồn (chỉ qua GitHub)

Việc tải lên Kaggle qua trình duyệt đã bị loại bỏ. Input chạy được push lên GitHub trước; session Kaggle chỉ fetch và kiểm tra sha256 trước khi áp dụng. Đừng bao giờ tải mã nguồn lên qua hộp thoại upload.

```bash
cd /kaggle/working
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
git checkout agent-v1
```

Đối với gói chạy ví dụ tài liệu (một gói cho mỗi batch), hãy fetch nhánh `kaggle-run-inputs` và kiểm tra sha256 được liệt kê trong manifest của batch trước khi áp dụng:

```bash
cd /kaggle/working
git clone --branch kaggle-run-inputs https://github.com/nguyenminh20000/Litetorch-.git run-inputs
cd run-inputs
sha256sum <package-file>
```

`/kaggle/input` chỉ đọc: hãy sao chép cây mã nguồn sang `/kaggle/working` trước khi biên dịch, và đọc dataset từ `/kaggle/input/datasets/<owner>/<slug>/`.

---

## 3. Kiểm tra CUDA toolkit, rồi biên dịch

Hãy kiểm tra (và cài đặt nếu thiếu) CUDA toolkit một cách tường minh trước mọi lần chạy GPU CUDA. Nó tình cờ có sẵn trên các image GPU trước đây, nhưng điều đó không được đảm bảo.

```bash
nvcc --version
```

Biên dịch extension tại chỗ:

```bash
cd /kaggle/working/Litetorch-
pip install pybind11
python3 setup.py build_ext --inplace
```

Hai sự thật từ runbook:

1. **`setup.py` đặt `package_dir={"": "python"}`, nên file `.so` đã biên dịch nằm dưới `<tree>/python/`.** Mọi cell notebook và script đều phải đưa `<tree>/python` vào `sys.path` trước `import litetorch`.
2. **Trên image Kaggle hiện tại, `build_ext --inplace` ghi `liblitetorch_gpu.so` vào `build/lib.linux-x86_64-cpython-313/`** — phải sao chép nó sang cạnh extension dưới `python/` trước khi import hoạt động (bản sao dưới `python/` mới là thứ `dlopen` tìm thấy; bản sao dưới `/tmp` bị xóa khi session khởi động lại).

```bash
cp build/lib.linux-x86_64-cpython-313/liblitetorch_gpu.so python/
ls python/
```

---

## 4. Cổng kiểm tra tiền điều kiện GPU

Trước khi chạy bất kỳ ví dụ GPU nào, hãy khẳng định cả ba điều sau:

```bash
python3 -c "import sys; sys.path.insert(0, 'python'); import litetorch as lt; print(lt.get_backend_name()); print(lt.cuda.device_count())"
```

1. Dòng khởi tạo `Native GPU (CUDA/HIP)` xuất hiện (KHÔNG phải OpenCL) — điều này bắt được lỗi thiếu thư viện native.
2. `liblitetorch_gpu.so` nằm cạnh extension dưới `python/`.
3. `lt.cuda.device_count()` trả về số lượng kỳ vọng (2 trên image GPU Kaggle).

Nếu bất kỳ khẳng định nào thất bại, hãy dừng lại và báo cáo nó như một vấn đề harness / blocker, không phải như lỗi ví dụ.

---

## 5. Smoke test

Đoạn mã dưới đây tự chứa và chạy được trên CPU local cũng như trên Kaggle GPU mà không thay đổi. Trên GPU nó kiểm tra cùng đường dẫn import và phép matmul trên GPU; trên CPU nó kiểm tra việc cài đặt.

<!-- ex: id=2.8-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()
print("Backend dang su dung:", lt.get_backend_name())
print("Thiet bi:", device)
print("GPU kha dung:", lt.is_gpu_available())

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
print("Ket qua:", y.to_vector())
print("Tong:", lt.Ops.sum(y).item())
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27 so với torch 2.14.1+cpu: kết quả matmul và tổng khớp chính xác, max diff 0.0). Lần chạy GPU của cùng đoạn mã là hạng mục Batch B5.

Kết quả kỳ vọng trên CPU:

```
Backend dang su dung: CPU
Thiet bi: cpu:0
GPU kha dung: False
Ket qua: [7.0, 10.0, 15.0, 22.0]
Tong: 54.0
```

Trên session GPU Kaggle, ba dòng đầu sẽ là `CUDA` / `cuda:0` / `True`; kết quả và tổng giữ nguyên.

---

## 6. Giữ session sống

- **Hết thời gian chờ (~20–30 phút) sẽ xóa `/kaggle/working`.** Hãy giữ session sống (tương tác giao diện thật / "Continue editing" mỗi 5–6 phút) cho đến khi nhiệm vụ hoàn thành và đã báo cáo, và kiểm tra lại file `.so` cùng dữ liệu sau mỗi lần khởi động lại.
- **Đồng bộ draft của editor có thể âm thầm xóa `/kaggle/working` khi session khởi động lại** (`ConcurrencyViolation`). Sau mỗi lần khởi động lại, hãy tải lại editor cho đến khi nó báo "Draft saved", rồi chạy lại toàn bộ chuỗi. Hãy giữ chuỗi clone → patch → build → repro chạy lại được trong một lần.
- **Đừng bao giờ đóng session khi công việc chưa xong.** Session chết giữa chừng đã khiến dự án này mất nhiều lần chạy.
- **Đăng xuất khỏi Kaggle + Google khi công việc kết thúc.** Đừng bao giờ bỏ qua bước này.
- **TPU chỉ sau khi GPU xanh.** Gói kiểm tra TPU là một lần push `kaggle-run-inputs` riêng (một gói cho mỗi batch, giống như GPU).

---

## 7. Đọc tiếp ở đâu

- Lần chạy đầu tiên trong notebook: [Bắt đầu nhanh](../quickstart.md).
- Tham chiếu cài đặt đầy đủ: [Cài đặt](../installation.md).
- Workload huấn luyện và benchmark hiệu năng trên Kaggle: Tutorials T2–T3 và các how-to guide (đang viết).
