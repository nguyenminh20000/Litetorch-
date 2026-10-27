# Cài đặt trên Azure

Trang này trình bày cách chạy LiteTorch trên Microsoft Azure: máy ảo GPU (hướng chính), Azure Machine Learning compute (tóm tắt), và Azure Container Instances (tóm tắt, đã ngừng — không dùng cho triển khai mới). Các bước cài đặt cơ bản xem [Hướng dẫn cài đặt](../installation.md); về container nói chung xem [Cài đặt trên Docker](docker.md).

> Ghi chú trạng thái: mọi bước cấp phát cloud và mọi lần chạy GPU trên cloud trong trang này đều có `status=unverified`. Máy kiểm tra tài liệu này không có tài khoản Azure và không có thông tin đăng nhập Azure, nên không có bước cloud nào được cấp phát hay chạy tại đây. Máy này cũng không có binary Docker, nên Docker runtime cũng chưa được kiểm chứng tại đây. Các dữ kiện cloud trong trang này chỉ lấy từ tài liệu chính thức của Microsoft, trích dẫn theo từng khẳng định tại [Nguồn](#nguồn). Không nêu giá: giá thay đổi nhanh — hãy kiểm tra [trang giá Azure cho máy ảo Linux](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/) trước khi chọn size. Không nêu quota, tính khả dụng theo khu vực, hay SLA, trừ khi trang chính thức được trích dẫn có nêu.

---

## 1. Chọn hướng Azure nào

| Hướng | Dùng cho | Trạng thái trong trang này |
| :--- | :--- | :--- |
| Máy ảo GPU (dòng NC/ND) | Toàn quyền kiểm soát: huấn luyện, chạy đa GPU, build tùy chỉnh | Hướng chính, §2–§4 |
| Azure Machine Learning compute | Job huấn luyện được quản lý trên cluster GPU | Tóm tắt, §5 |
| Azure Container Instances với GPU | — | Đã ngừng 2025-07-14, không dùng, §6 |

---

## 2. Size máy ảo GPU: dòng NC và ND

Các họ máy ảo tăng tốc GPU của Azure là NC, ND, NG và NV ([Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)). Với LiteTorch, NC và ND là hai họ liên quan: NC được thiết kế cho workload tính toán chuyên sâu như huấn luyện mô hình AI và machine learning, còn ND cho deep learning, nghiên cứu AI và high-performance computing ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family), [ND family](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family)).

### 2.1. Dòng NC → GPU

| Dòng | GPU (theo tài liệu chính thức) | GPU trên mỗi máy ảo |
| :--- | :--- | :--- |
| NCads_H100_v5 | NVIDIA H100 NVL, 94 GB mỗi GPU | 1–2 |
| NCCads_H100_v5 | NVIDIA H100 NVL, 94 GB | 1 |
| NC_A100_v4 | NVIDIA A100 PCIe, 80 GB mỗi GPU | 1–4 |
| NCasT4_v3 | NVIDIA Tesla T4, 16 GB mỗi GPU | 1–4 |
| NCv3 | NVIDIA Tesla V100, 16 GB mỗi GPU | 1–4 |

Nguồn: [NC family VM size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family). NCads_H100_v5 là thế hệ hiện tại mà Microsoft nêu là nhận capacity mới: trang NC_A100_v4 ghi Microsoft "currently only deploying net new capacity for the latest generation of the NC product line, the NCads_H100_v5-series" ([NC_A100_v4 series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series)).

Hai cảnh báo từ các trang chính thức:

- **NCv3 đã ngừng.** Microsoft ngừng dòng NCv3-series vào ngày 30 tháng 9 năm 2025; xem trang retirement của dòng NCv3-series để biết khuyến nghị di chuyển ([NCv3 series](https://learn.microsoft.com/sl-si/azure/virtual-machines/sizes/gpu-accelerated/ncv3-series)). Không bắt đầu công việc mới trên NCv3.
- **Quota khởi đầu bằng 0 đối với NCv3.** Với NCv3, "the vCPU (core) quota in your subscription is initially set to 0 in each region. Request a vCPU quota increase for this series in an available region" ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family)). Nói chung hãy chuẩn bị yêu cầu tăng quota cho các dòng GPU; kiểm tra tính khả dụng theo khu vực trước khi lên kế hoạch chạy.

### 2.2. Dòng ND → GPU

| Dòng | GPU (theo tài liệu chính thức) | GPU trên mỗi máy ảo |
| :--- | :--- | :--- |
| ND_H100_v5 | NVIDIA H100 Tensor Core, 80 GB mỗi GPU | 8 |
| ND_A100_v4 | NVIDIA Ampere A100, 40 GB mỗi GPU | 8 |
| NDm_A100_v4 | NVIDIA Ampere A100, 80 GB mỗi GPU | 8 |
| NDv2 | NVIDIA Tesla V100 NVLink, 32 GB mỗi GPU | 8 |
| ND (v1) | NVIDIA Tesla P40, 24 GB mỗi GPU | 1–4 |
| ND_MI300X_v5 | AMD Instinct MI300X, 192 GB mỗi GPU | 8 |

Nguồn: [ND family VM size series](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family). ND_A100_v4, NDm_A100_v4 và ND_H100_v5 được xây dựng cho scale-up (8 GPU trên mỗi máy ảo) và scale-out (nhiều máy ảo qua InfiniBand, hỗ trợ GPU Direct RDMA và NCCL), theo cùng trang đó. ND_MI300X_v5 là phần cứng AMD: đó là hướng ROCm, không phải CUDA — các ghi chú CUDA trong trang này không áp dụng cho dòng đó.

### 2.3. Chọn size

- **Phát triển đơn GPU hoặc inference:** NCasT4_v3 (T4) hoặc NC_A100_v4 (A100). Tài liệu huấn luyện PyTorch của Azure Machine Learning dùng `Standard_NC4as_T4_v3` làm size GPU ví dụ ([Train deep learning PyTorch models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri)).
- **Huấn luyện đa GPU trên một máy ảo:** NC_A100_v4 (tối đa 4 GPU) hoặc ND_A100_v4 / ND_H100_v5 (8 GPU).
- **Huấn luyện phân tán đa máy ảo:** dòng ND với InfiniBand (ND_A100_v4, NDm_A100_v4, ND_H100_v5), mà trang ND chính thức mô tả là hỗ trợ clustering dựa trên NCCL.

Kiểm tra khu vực của bạn thực sự cung cấp những size nào trước khi cấp phát; trang tổng quan chính thức đưa ra lệnh Azure CLI cho việc đó:

<!-- ex: id=az-1 device=cpu backend=cpu runtime=~10s torch-parity=n/a status=unverified -->

```bash
az vm list-skus --location <region> --output table
```

Lệnh này được trích từ [Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu). Lệnh chưa được chạy tại đây: không có tài khoản Azure, không có thông tin đăng nhập.

---

## 3. Thiết lập máy ảo: image, driver, CUDA toolkit

### 3.1. Image cơ sở

Dùng image Ubuntu LTS. Trang driver chính thức liệt kê Ubuntu 24.04 LTS, 22.04 LTS và 20.04 LTS là các phiên bản hệ điều hành Linux được hỗ trợ cho GRID driver trên NCasT4_v3 ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Ubuntu 22.04 khớp với image Docker CUDA của LiteTorch ([Cài đặt trên Docker](docker.md)) và hướng dẫn cài đặt cơ bản ([Hướng dẫn cài đặt](../installation.md) §2.2).

### 3.2. Driver NVIDIA

Để dùng GPU trên máy ảo Azure N-series dùng GPU NVIDIA, bắt buộc phải cài driver GPU NVIDIA. NVIDIA GPU Driver Extension cài driver NVIDIA CUDA hoặc GRID phù hợp trên máy ảo N-series, và có thể cài đặt và quản lý qua Azure portal hoặc Azure PowerShell ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Hai giới hạn từ trang extension chính thức ([NVIDIA GPU Driver Extension — Azure Linux VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux)):

- Extension không tự động cập nhật driver sau khi cài; để chuyển sang driver mới hơn, tải và cài thủ công, hoặc gỡ rồi thêm lại extension.
- Secure Boot không được hỗ trợ khi dùng extension Windows hoặc Linux.

Chọn loại driver nào: NVIDIA phân phối CUDA driver cho các máy ảo NCasT4_v3, NC_A100_v4 và NCads_H100_v5; để cài CUDA driver, trang chính thức trỏ tới website của NVIDIA ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Trên NCasT4_v3, Azure NVIDIA GPU driver extension cài CUDA driver; với workload đồ họa/hiển thị trên cùng dòng đó, hãy cài thủ công GRID driver được Azure hỗ trợ ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family)). Với workload tính toán LiteTorch, cài CUDA driver, không phải GRID.

Kiểm tra driver sau khi cài:

<!-- ex: id=az-2 device=gpu backend=cuda runtime=~10s torch-parity=n/a status=unverified -->

```bash
nvidia-smi
```

Theo trang driver chính thức, nếu driver đã được cài, NVIDIA SMI liệt kê **GPU-Util** là N/A cho đến khi bạn chạy một workload GPU trên máy ảo ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Chưa chạy tại đây: không có máy ảo Azure nào.

### 3.3. CUDA toolkit — kiểm tra trước mọi lần chạy GPU

Chỉ có driver là chưa đủ. LiteTorch cần CUDA toolkit (`nvcc`, cuBLAS, cuBLASLt) để build thư viện GPU native: CUDA Toolkit 11.8 hoặc 12.x ([Hướng dẫn cài đặt](../installation.md) §2.1). Đây là quy tắc bắt buộc, không phải gợi ý: luôn kiểm tra (và cài nếu thiếu) CUDA toolkit một cách tường minh trước mọi lần chạy GPU CUDA. Thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ được build, và framework sẽ lặng lẽ fallback về CPU.

<!-- ex: id=az-3 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
nvcc --version
```

Nếu không tìm thấy `nvcc`, hãy cài CUDA Toolkit cho bản phân phối của bạn (ví dụ, network repository NVIDIA CUDA 12.x cho Ubuntu) và chạy lại `nvcc --version` cho đến khi nó in ra một phiên bản. Chỉ sau đó mới build hoặc chạy LiteTorch trên GPU. Chưa chạy tại đây: không có máy ảo Azure nào.

---

## 4. Cài LiteTorch trên máy ảo

Không lặp lại các bước cơ bản tại đây: cài LiteTorch đúng như trong hướng dẫn cơ bản — cài qua pip ([Hướng dẫn cài đặt](../installation.md) §2.2) hoặc build từ mã nguồn ([Hướng dẫn cài đặt](../installation.md) §2.4). Các điểm riêng cho máy ảo là:

1. **Vị trí extension.** `setup.py` đặt `package_dir={"": "python"}`, nên `build_ext --inplace` đặt extension có thể import được dưới `<tree>/python/`, không phải thư mục gốc của cây mã nguồn. Mọi script phải thêm `<tree>/python` vào `sys.path` trước `import litetorch` ([Hướng dẫn cài đặt](../installation.md) §2.4). Xác nhận các file đã build có ở đó:

<!-- ex: id=az-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=unverified -->

```bash
find python -maxdepth 1 -name "*.so"
```

2. **Cổng kiểm tra pre-flight.** Trước mọi lần chạy GPU, xác nhận dòng khởi động `Native GPU (CUDA/HIP)` xuất hiện khi import và `liblitetorch_gpu.so` nằm cạnh extension dưới `python/`. Nếu dòng khởi động vắng mặt, backend GPU đã không tải — kiểm tra `nvcc --version`, build lại, rồi kiểm tra lại. Đừng cho rằng GPU đang được dùng: `.to(device)` sẽ lặng lẽ fallback về CPU khi accelerator không khả dụng ([Hướng dẫn cài đặt](../installation.md) §2.6).
3. **Chuỗi device.** Chuỗi device chuẩn của LiteTorch là `gpu:N`, không phải `cuda:N`: `str(t.device)` in ra `gpu:0` cho GPU đầu tiên. Parser chấp nhận cả `cuda` và `gpu` làm đầu vào, nhưng đầu ra và `auto_device()` dùng `gpu:N`. Không viết `cuda:0` trong code LiteTorch và không so sánh với nó.

Đoạn kiểm tra (cùng code với [Hướng dẫn cài đặt](../installation.md) §2.6, chạy tại đây trên máy ảo):

<!-- ex: id=az-5 device=gpu backend=cuda runtime=~1s torch-parity=1e-4 status=unverified -->

```python
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Available: {is_gpu}")
print(f"Backend Name: {backend}")
print(f"Active Device: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Result shape:", y.shape)
print("Loss value:", loss.item())
print("Gradient:", x.grad.to_vector())
```

Đầu ra kỳ vọng trên một máy ảo GPU Azure hoạt động (các giá trị là chính xác cho input này; ba dòng đầu là cổng kiểm tra):

<!-- ex: id=az-5-out device=gpu backend=cuda runtime=n/a torch-parity=1e-4 status=unverified -->

```
GPU Available: True
Backend Name: CUDA
Active Device: gpu:0
Result shape: [2, 2]
Loss value: 54.0
Gradient: [7.0, 11.0, 9.0, 13.0]
```

Nếu `Active Device` in ra `cpu:0` trên một máy ảo mà bạn kỳ vọng có GPU, backend GPU đã không tải — kiểm tra lại §3.3, build lại, và xác nhận dòng khởi động `Native GPU (CUDA/HIP)`. Các giá trị loss và gradient ở trên là tham chiếu torch-parity từ lần chạy đã kiểm chứng trên CPU trong hướng dẫn cơ bản; bản thân lần chạy GPU vẫn chưa được kiểm chứng — chưa có lần chạy GPU Azure nào được thực hiện.

Khi xong việc, hãy dừng hoặc xóa các tài nguyên không còn cần. Đừng để một máy ảo GPU chạy không tải.

---

## 5. Azure Machine Learning compute và job (tóm tắt)

Azure Machine Learning quản lý compute thay cho bạn. Các loại compute được quản lý là compute cluster, serverless compute và compute instance ([Understand compute targets](https://learn.microsoft.com/da-dk/azure/machine-learning/concept-compute-target?view=azureml-api-2)):

- **Compute cluster:** cluster được quản lý hoàn toàn, tự động co giãn cho job huấn luyện; đơn node hoặc đa node, CPU và GPU. Đặt số node tối thiểu bằng 0 để tránh phí khi cluster không tải (cùng nguồn).
- **Compute instance:** cluster đơn node, máy phát triển Jupyter notebook gắn với workspace của bạn (cùng nguồn). Bật idle shutdown để tránh phí khi không tải (cùng nguồn).
- **Serverless compute:** chuyển việc quản lý vòng đời compute cho Azure Machine Learning thay vì tự tạo cluster (cùng nguồn).

Việc tạo cluster GPU tuân theo tài liệu huấn luyện PyTorch chính thức ([Train deep learning PyTorch models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri)), trong đó tạo một compute cluster Linux với size `Standard_NC4as_T4_v3` (size T4 trong §2.1). Rút gọn từ tài liệu đó, đã bỏ comment:

<!-- ex: id=az-6 device=gpu backend=cuda runtime=~10m torch-parity=n/a status=unverified -->

```python
from azure.ai.ml.entities import AmlCompute

gpu_cluster = AmlCompute(
    name="gpu-cluster",
    type="amlcompute",
    size="STANDARD_NC4AS_T4_V3",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=180,
    tier="Dedicated",
)
gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()
```

`ml_client` là client workspace trong cùng tài liệu (`MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)`). Tên tham số và giá trị ở trên được trích từ tài liệu chính thức; không thêm gì khác. Việc submit job chỉ trình bày ở mức tóm tắt: định nghĩa job huấn luyện (script huấn luyện, môi trường, và compute target `gpu-cluster`) rồi submit từ studio, Python SDK, hoặc Azure CLI, theo cùng tài liệu huấn luyện chính thức. Không đưa thêm flag CLI hay lệnh SDK nào ngoài tài liệu được trích dẫn.

Các ghi chú riêng của LiteTorch vẫn áp dụng nguyên vẹn trên Azure ML compute: kiểm tra CUDA toolkit trước mọi lần chạy GPU, vượt qua cổng pre-flight (dòng khởi động `Native GPU (CUDA/HIP)` + vị trí `.so`), và dùng chuỗi device `gpu:0`. Môi trường (container image hoặc conda environment) phải chứa CUDA toolkit và extension LiteTorch đã build dưới `<tree>/python/` trên `sys.path`, đúng như trong §4. Không bước nào trong số này đã được chạy trên Azure ML: không có workspace, không có tài khoản, không có thông tin đăng nhập.

---

## 6. Azure Container Instances (tóm tắt — đã ngừng)

Không dùng GPU resource của Azure Container Instances cho triển khai mới. Trang chính thức ghi: "This product is retired as of July 14, 2025" ([Deploy GPU-Enabled Container Instance](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-gpu)). Để lưu hồ sơ, tính năng đã ngừng này từng cung cấp:

- Chỉ phiên bản GPU V100, ánh xạ tới họ máy ảo NCv3; chỉ Linux; số GPU 1, 2 hoặc 4.
- Tài nguyên tối đa theo SKU: 1 GPU → 6 CPU / 112 GB bộ nhớ / 50 GB storage; 2 GPU → 12 CPU / 224 GB; 4 GPU → 24 CPU / 448 GB.
- Container instance với GPU resource được cấp sẵn NVIDIA CUDA driver và container runtime; trang đó hỗ trợ tới CUDA 11.
- Giới hạn (quota) mặc định của subscription cho GPU resource khác nhau theo phiên bản; giới hạn CPU mặc định cho phiên bản V100 khởi đầu bằng 0, và việc tăng quota cần yêu cầu Azure Support.

Tất cả nội dung trên được trích từ cùng trang chính thức đã ngừng đó, chỉ để lưu hồ sơ. Hãy dùng máy ảo GPU (§2–§4) hoặc Azure Machine Learning compute (§5) thay thế. Docker runtime dù sao cũng chưa được kiểm chứng trên máy này: không có binary Docker tại đây, nên không có lần build image hay chạy container nào trong hướng dẫn này được thực thi.

---

## 7. Xử lý sự cố

Hướng dẫn xử lý sự cố đầy đủ nằm trong [Xử lý sự cố](../troubleshooting.md). Các triệu chứng riêng cho Azure:

| Triệu chứng | Kiểm tra |
| :--- | :--- |
| `Active Device` in ra `cpu:0` trên máy ảo GPU | §3.3: `nvcc --version`; build lại; xác nhận dòng khởi động `Native GPU (CUDA/HIP)` (§4) |
| `nvidia-smi` lỗi hoặc không thấy GPU | §3.2: extension driver đã cài; đúng loại driver (CUDA, không phải GRID, cho compute) |
| `ModuleNotFoundError: No module named 'litetorch'` | `<tree>/python` chưa có trên `sys.path` (§4.1) |
| Size máy ảo không khả dụng, hoặc lỗi quota khi tạo | §2: `az vm list-skus` cho khu vực đó; yêu cầu tăng quota (quota NCv3 khởi đầu bằng 0 theo §2.1) |
| Triển khai GPU trên ACI thất bại | §6: tính năng đã ngừng (2025-07-14); dùng máy ảo hoặc Azure ML compute |

---

## Nguồn

Chỉ tài liệu chính thức của Microsoft. Đã truy cập và đọc ngày 2026-10-27. Không có giá, quota, tính khả dụng theo khu vực, hay SLA nào được nêu trong trang này ngoài những gì các trang này nêu.

- [Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu) — các họ GPU (NC, ND, NG, NV); lệnh `az vm list-skus`; trỏ tới trang giá.
- [NC family VM size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family) — ánh xạ GPU của NCads_H100_v5 / NCCads_H100_v5 / NCv3 / NCasT4_v3 / NC_A100_v4; quota NCv3 khởi đầu bằng 0; ghi chú driver CUDA so với GRID cho NCasT4_v3.
- [NC_A100_v4 size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series) — NCads_H100_v5 là thế hệ hiện tại nhận capacity mới.
- [NCv3 size series](https://learn.microsoft.com/sl-si/azure/virtual-machines/sizes/gpu-accelerated/ncv3-series) — NCv3 ngừng vào ngày 30 tháng 9 năm 2025.
- [ND family VM size series](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family) — ánh xạ GPU của ND / NDv2 / ND_A100_v4 / NDm_A100_v4 / ND_H100_v5 / ND_MI300X_v5; InfiniBand và scale-out NCCL.
- [Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup) — extension driver cài driver CUDA hoặc GRID; CUDA driver được phân phối cho NCasT4_v3, NC_A100_v4, NCads_H100_v5; kiểm tra bằng `nvidia-smi`; các phiên bản Ubuntu LTS được hỗ trợ.
- [NVIDIA GPU Driver Extension — Azure Linux VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux) — không tự động cập nhật driver; Secure Boot không được hỗ trợ với extension.
- [Train deep learning PyTorch models (SDK v2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri) — ví dụ cluster GPU `AmlCompute` (`STANDARD_NC4AS_T4_V3`, min 0, max 4, idle 180 s, Dedicated).
- [Understand compute targets](https://learn.microsoft.com/da-dk/azure/machine-learning/concept-compute-target?view=azureml-api-2) — compute cluster so với compute instance so với serverless compute; min node 0 và idle shutdown.
- [Deploy GPU-Enabled Container Instance](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-gpu) — ACI GPU ngừng 2025-07-14; chỉ V100, họ NCv3, chỉ Linux, 1/2/4 GPU, giới hạn tối đa theo SKU, hỗ trợ CUDA 11, quota mặc định 0.
- [Pricing — Linux Virtual Machines](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/) — kiểm tra giá hiện tại tại đây; trang này không trích giá nào.
