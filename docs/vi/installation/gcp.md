# Cài đặt trên Google Cloud Platform

Google Cloud Platform (GCP) cho LiteTorch một GPU mà không cần phần cứng tại chỗ: một máy ảo Compute Engine gắn GPU NVIDIA, hoặc một training job và endpoint được quản lý trên Vertex AI. Trang này trình bày cách chọn GPU, tạo máy ảo (image Deep Learning VM hoặc Ubuntu/Debian thuần cộng cài CUDA thủ công), cài đặt LiteTorch, và cổng kiểm tra pre-flight cho GPU.

> Lưu ý về trạng thái: mọi bước provisioning trên GCP và mọi lần chạy GPU trên GCP trong trang này đều là `status=unverified`. Máy ảo kiểm chứng tài liệu này không có tài khoản GCP và không có thông tin xác thực GCP, nên không có bước cloud nào trong trang này được thực thi. Các sự kiện về cloud được trích dẫn theo từng khẳng định tới tài liệu chính thức của Google Cloud; hãy kiểm tra trang chính thức được liên kết trước khi thực hiện, vì các loại GPU, họ image và quota đều thay đổi. Trang này không nêu giá — hãy xem [trang giá GPU của Google Cloud](https://cloud.google.com/compute/gpus-pricing).

---

## 1. Chọn GPU và dòng máy

Chỉ các dòng máy accelerator-optimized (A4X, A4, A3, A2, G4, G2) và dòng máy general-purpose N1 mới hỗ trợ GPU trên Compute Engine. Với các dòng máy A và G, GPU được gắn tự động vào instance; với N1, bạn gắn một mẫu GPU vào instance ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)).

| Dòng máy | Mẫu GPU | Chuỗi loại `accelerator` |
| :--- | :--- | :--- |
| A4X Max | NVIDIA GB300 Ultra Superchip | `nvidia-gb300` |
| A4X | NVIDIA GB200 Superchip | `nvidia-gb200` |
| A4 | NVIDIA B200 | `nvidia-b200` |
| A3 Ultra | NVIDIA H200 | `nvidia-h200-141gb` |
| A3 Mega | NVIDIA H100 | `nvidia-h100-mega-80gb` |
| A3 High / A3 Edge | NVIDIA H100 | `nvidia-h100-80gb` |
| A2 Ultra | NVIDIA A100 80GB | `nvidia-a100-80gb` |
| A2 Standard | NVIDIA A100 40GB | `nvidia-tesla-a100` |
| G4 | NVIDIA RTX PRO 6000 | `nvidia-rtx-pro-6000` |
| G2 | NVIDIA L4 | `nvidia-l4` |
| N1 (gắn thêm) | NVIDIA T4 | `nvidia-tesla-t4` |
| N1 (gắn thêm) | NVIDIA V100 | `nvidia-tesla-v100` |
| N1 (gắn thêm) | NVIDIA P4 | `nvidia-tesla-p4` |

Nguồn: [About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus). Hai cảnh báo từ cùng trang đó: NVIDIA T4 và NVIDIA P4 đang tiến tới end of support, và Compute Engine SLA chỉ bao phủ một instance có GPU gắn thêm khi mẫu GPU đó đã generally available.

Cách chọn:

- **L4 (G2)** — lựa chọn rẻ nhất trong thế hệ hiện tại cho phát triển, huấn luyện mô hình nhỏ, và suy luận single-host. G2 cũng hỗ trợ custom memory.
- **A100 (A2)** — huấn luyện các mô hình lớn hơn và công việc multi-GPU trên một host.
- **H100 / H200 / B200 (A3 / A4)** — huấn luyện mô hình lớn và công việc multi-host; các dòng này có giới hạn về consumption option và region, nên hãy đọc các giới hạn theo từng dòng trên [trang tổng quan tạo máy ảo](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus) trước khi lên kế hoạch.
- **T4 (N1)** — GPU gắn thêm rẻ nhất, và khớp nhất với track GPU trên Kaggle (2x T4). Hãy ưu tiên L4 thay vì T4 cho công việc mới vì T4 đang tiến tới end of support.

Không phải mẫu GPU nào cũng có ở mọi region và zone, và các project mới có một global GPU quota giới hạn tổng số GPU bạn có thể tạo trong bất kỳ zone được hỗ trợ nào. Khi yêu cầu GPU quota, bạn phải yêu cầu quota cho các loại GPU muốn dùng ở từng region cộng thêm một global quota ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)). Nếu tạo instance thất bại với lỗi quota, đó chính là nguyên nhân — hãy yêu cầu tăng quota; đó không phải lỗi của LiteTorch.

Spot VM có GPU có giá thấp hơn, nhưng máy ảo bị preempt khi host bảo trì và không thể tự khởi động lại ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)). Chỉ dùng Spot cho các lần chạy chịu được lỗi và có thể tạo lại.

---

## 2. Hướng A: Image Deep Learning VM (khuyến nghị)

Deep Learning VM (DLVM) image là image máy ảo được tối ưu cho data science và machine learning, với các ML framework và công cụ chính được cài sẵn, dùng được ngay trên các instance có GPU ([Introduction to Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/introduction)). Các image dựa trên Debian 11 và Ubuntu 22.04, gồm Python 3.10 với numpy, scipy, matplotlib, pandas, scikit-learn và các gói khác, môi trường JupyterLab, và — trên các image có GPU — NVIDIA driver, CUDA 11.x và 12.x, cuDNN, và NCCL 2.x (phiên bản tùy thuộc framework).

Các họ image hiện tại ([Choose an image](https://cloud.google.com/deep-learning-vm/docs/images)):

| Framework | Processor | Họ image |
| :--- | :--- | :--- |
| Base | GPU | `common-cu128-ubuntu-2204-nvidia-570` |
| PyTorch | GPU | `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` |

Ubuntu 22.04 là OS mặc định và các image đi kèm NVIDIA driver 570; các image Debian đều đã deprecated. Tên họ image thay đổi khi có phiên bản framework mới — luôn xác nhận họ hiện tại trên trang [Choose an image](https://cloud.google.com/deep-learning-vm/docs/images) trước khi tạo instance, và tham chiếu tới họ (không phải một image cố định) để nhận phiên bản mới nhất.

Tạo một DLVM instance gắn một T4 (mẫu theo [Create a Deep Learning VM instance from the command line](https://cloud.google.com/deep-learning-vm/docs/cli)):

<!-- ex: id=gcp-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
export IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
export ZONE="us-west1-b"
export INSTANCE_NAME="litetorch-gcp"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"
```

Các quy tắc trong lệnh đó, tất cả từ tài liệu DLVM chính thức:

- `--image-project` phải là `deeplearning-platform-release`.
- `--maintenance-policy` phải là `TERMINATE` — Compute Engine luôn dừng các instance có GPU gắn thêm khi có maintenance event trên host.
- `--accelerator` có dạng `type=TYPE,count=COUNT`; dùng các chuỗi loại trong bảng ở §1.
- `--metadata="install-nvidia-driver=True"` cài NVIDIA driver khi boot.
- Thay `us-west1-b` bằng một zone hỗ trợ loại GPU bạn chọn; không phải loại GPU nào cũng được hỗ trợ ở mọi region.

Giới hạn: không thể dùng Deep Learning VM Images làm boot disk cho các instance dùng dòng máy G2 ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)). Với L4, dùng Hướng B bên dưới.

---

## 3. Hướng B: Ubuntu/Debian thuần + cài CUDA thủ công

Chọn hướng này khi image DLVM không phù hợp: instance G2 (L4), một phiên bản CUDA mà image DLVM không có, hoặc một base image tối giản do bạn kiểm soát hoàn toàn. Bạn tự cài NVIDIA driver và CUDA Toolkit; với hầu hết các public image hoặc custom image, bạn cần cài cả hai ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)).

Tạo một instance G2 (L4) từ một image thuần (mẫu theo [Create a G2 or G4 instance](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series)):

<!-- ex: id=gcp-2 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
gcloud compute instances create litetorch-gcp-l4 \
  --machine-type=g2-standard-12 \
  --zone=us-west1-b \
  --boot-disk-size=200GB \
  --image-family=rocky-linux-8-optimized-gcp \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

Với instance G2, GPU được gắn tự động theo machine type, nên không có cờ `--accelerator`. `--maintenance-policy=TERMINATE` là bắt buộc với các instance GPU. Zone phải hỗ trợ GPU L4. `--image-family` ở trên là họ image ví dụ trong tài liệu G2 chính thức — bất kỳ OS image nào hỗ trợ GPU đều dùng được; Ubuntu 22.04 là lựa chọn phổ biến còn lại.

Sau đó, trên máy ảo, cài NVIDIA driver và CUDA Toolkit cho distribution của bạn và xác nhận cả hai trước khi làm bất cứ việc gì khác:

<!-- ex: id=gcp-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` xác minh driver nhìn thấy GPU; `nvcc --version` xác minh CUDA Toolkit. Nếu không tìm thấy `nvcc`, hãy cài CUDA Toolkit cho distribution của bạn và chạy lại cho tới khi nó in ra một phiên bản. Đây là quy tắc đứng của LiteTorch: luôn kiểm tra (và cài nếu thiếu) CUDA toolkit một cách tường minh trước bất kỳ lần chạy GPU CUDA nào — thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ được build và framework sẽ lặng lẽ fallback về CPU.

Khi nào chọn hướng nào:

| | Deep Learning VM (§2) | Image thuần + cài thủ công (§3) |
| :--- | :--- | :--- |
| Driver + CUDA | Cài sẵn | Bạn tự cài cả hai |
| Framework | PyTorch và các gói data-science cài sẵn | Chỉ những gì bạn cài |
| Chạy được trên G2 (L4) | Không | Có |
| Kiểm soát phiên bản CUDA | Không (image quyết định) | Có |
| Phù hợp nhất cho | Đường nhanh nhất tới một máy ảo GPU chạy được | L4, CUDA cố định, image tối giản |

---

## 4. Cài LiteTorch trên máy ảo

Khi máy ảo đã chạy và driver cùng CUDA Toolkit đã được xác nhận, việc cài LiteTorch giống như trên bất kỳ máy Linux nào. Đừng đọc lại các bước trùng lặp ở đây — hãy theo hướng dẫn chính:

- **Cài qua pip:** [Installation §2.2](../installation.md#22-linux-and-google-colab) (`pip install --upgrade litetorch`).
- **Build từ source:** [Installation §2.4](../installation.md#24-build-from-source) — clone, `pip install pybind11`, `python3 setup.py build_ext --inplace`.

<!-- ex: id=gcp-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
pip install pybind11
python3 setup.py build_ext --inplace
```

Hai sự kiện về build trong hướng dẫn chính gây ra hầu hết các lỗi lần đầu:

1. **`setup.py` đặt `package_dir={"": "python"}`, nên file `.so` được build nằm dưới `<tree>/python/`.** Mọi script phải thêm `<tree>/python` vào `sys.path` trước `import litetorch`, nếu không import sẽ lỗi `ModuleNotFoundError`.
2. **Trên các image GPU, `build_ext --inplace` có thể ghi `liblitetorch_gpu.so` vào `build/lib.linux-*/` thay vì cạnh extension.** Hãy copy nó tới cạnh extension dưới `python/` trước khi import — bản copy trong `python/` mới là bản `dlopen` tìm thấy:

<!-- ex: id=gcp-5 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
cp build/lib.linux-*/liblitetorch_gpu.so python/
ls python/
```

---

## 5. Cổng kiểm tra pre-flight cho GPU

Trước khi chạy bất kỳ ví dụ GPU nào trên máy ảo, hãy khẳng định cả ba điều — cùng cổng kiểm tra như track Kaggle:

<!-- ex: id=gcp-6 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
python3 -c "import sys; sys.path.insert(0, 'python'); import litetorch as lt; print(lt.get_backend_name()); print(lt.cuda.device_count())"
```

1. Dòng init `Native GPU (CUDA/HIP)` xuất hiện (KHÔNG phải OpenCL) — điều này bắt lỗi thiếu thư viện native.
2. `liblitetorch_gpu.so` nằm cạnh extension dưới `python/`.
3. `lt.cuda.device_count()` trả về số lượng mong đợi (1 trên các máy ảo một GPU ở §2–§3).

Nếu bất kỳ khẳng định nào thất bại, hãy dừng lại và báo cáo nó như một vấn đề harness / blocker, không phải như lỗi của ví dụ. Nguyên nhân thường gặp là bỏ qua bước kiểm tra `nvcc` ở §3: không có CUDA Toolkit lúc build nghĩa là không có thư viện GPU native, và `.to(device)` sau đó sẽ lặng lẽ fallback về CPU.

Chuỗi device: chuỗi device GPU chuẩn của LiteTorch là `gpu:N`, không phải `cuda:N` — `str(t.device)` là `gpu:0` trên GPU đầu tiên và `cpu:0` trên CPU. Parser cũng chấp nhận `cuda:0` làm đầu vào, nhưng đừng bao giờ mong đợi `cuda:0` trong đầu ra. Nếu smoke test bên dưới in ra `cpu:0` trong khi bạn mong đợi GPU, backend GPU đã không load.

---

## 6. Smoke test

Đoạn code dưới đây là tự chứa và là cùng smoke test như trang Kaggle. Trên máy ảo GPU của GCP, nó kiểm tra đường import và phép matmul trên GPU.

<!-- ex: id=gcp-7 device=gpu backend=cuda runtime=~1s torch-parity=1e-4 status=unverified -->

```python
import litetorch as lt

device = lt.auto_device()
print("Backend:", lt.get_backend_name())
print("Device:", device)
print("GPU available:", lt.is_gpu_available())

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
print("Result:", y.to_vector())
print("Sum:", lt.Ops.sum(y).item())
```

Đầu ra mong đợi trên máy ảo GPU:

```
Backend: CUDA
Device: gpu:0
GPU available: True
Result: [7.0, 10.0, 15.0, 22.0]
Sum: 54.0
```

(Tên backend là tên của backend CUDA; kết quả và tổng giống hệt trên CPU và GPU. Cùng đoạn code này đã được xác minh trên CPU tại chỗ — xem [Installation §2.6](../installation.md#26-verification-script) — nhưng lần chạy GPU của nó trên GCP vẫn là unverified cho tới khi một lần chạy GPU trên GCP thực thi nó.)

---

## 7. Vertex AI training và endpoints (ngắn gọn)

Để huấn luyện và phục vụ được quản lý thay vì một máy ảo tự quản lý, Vertex AI cung cấp custom training job và endpoint. Mục này chỉ là outline — chưa có API Vertex AI riêng cho LiteTorch, và chưa có bước nào ở đây được chạy.

Luồng chính thức ([Train and deploy your model](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/train-and-deploy-model), [Prediction classes](https://cloud.google.com/vertex-ai/docs/python-sdk/prediction-classes)) là:

1. **Custom training job** — một `CustomTrainingJob` nhận script huấn luyện của bạn (`script_path`), một training container image (`container_uri`), và `requirements` tùy chọn, sau đó `job.run(...)` bắt đầu huấn luyện và trả về một `Model` đã huấn luyện.
2. **Deploy lên endpoint** — trước khi nhận được online prediction, bạn phải deploy model lên một endpoint; việc deploy gắn tài nguyên máy vật lý với model để nó có thể phục vụ online prediction. `model.deploy(...)` tạo và trả về `Endpoint`.
3. **Predict** — `endpoint.predict(instances=...)` trả về dự đoán cho đầu vào mới.

Các lưu ý riêng cho LiteTorch, tất cả đều unverified:

- Vertex AI không có container huấn luyện hay dự đoán LiteTorch dựng sẵn. Cần một custom container — image CUDA trong [Installation on Docker](docker.md) là điểm khởi đầu để build một image như vậy, nhưng container, bước cài LiteTorch trong training script, và đường phục vụ đều chưa được build hay chạy.
- Các quy tắc LiteTorch tương tự áp dụng bên trong training container: kiểm tra CUDA Toolkit trước bất kỳ lần chạy GPU nào, xác nhận dòng init `Native GPU (CUDA/HIP)` và vị trí `.so` trong cổng pre-flight, và mong đợi chuỗi device `gpu:0`.
- Deep Learning VM Images và Deep Learning Containers cũng có thể dùng như một phần của công việc Vertex AI, ví dụ làm môi trường phát triển cho một hệ thống huấn luyện phân tán tự quản lý ([Use Deep Learning VM Images and Deep Learning Containers with Vertex AI](https://cloud.google.com/vertex-ai/docs/general/deep-learning)).

Đừng bịa thêm CLI flag hay SDK call của Vertex AI ngoài những gì các trang chính thức được trích dẫn cho thấy — nếu trang chính thức không cho thấy nó, nó chưa thuộc về một runbook.

---

## 8. Chi phí và dọn dẹp

- **Trang này không nêu giá.** Hãy xem [trang giá GPU của Google Cloud](https://cloud.google.com/compute/gpus-pricing) trước khi tạo bất cứ thứ gì; máy ảo GPU tính phí theo giây với mức tối thiểu 1 phút, và một instance nhàn rỗi nhưng vẫn RUNNING vẫn bị tính phí.
- **Dừng máy ảo khi xong việc** — `gcloud compute instances stop <name>` giữ lại boot disk và dừng tính phí cho máy ảo (tài liệu DLVM dùng lệnh này khi snapshot một instance). Xóa instance khi công việc hoàn tất để không còn gì tiếp tục bị tính phí.
- **Đừng giữ dữ liệu trên máy ảo.** Lưu dữ liệu không tạm thời trên persistent block storage (Persistent Disk / Hyperdisk) hoặc trong Cloud Storage, không phải trên local SSD của máy ảo — dữ liệu local SSD bị mất khi instance dừng ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)).

---

## 9. Troubleshooting

- `ModuleNotFoundError: No module named 'litetorch'` — `<tree>/python` không có trong `sys.path` (xem §4, sự kiện 1).
- Build in `BUILD OK` nhưng import thất bại — `setup.py` in `BUILD OK` ngay cả khi một số file compile lỗi. Cổng kiểm tra thực sự là file extension tồn tại dưới `<tree>/python/` và `import litetorch` thành công.
- Lần chạy GPU lặng lẽ dùng CPU — xác minh `nvcc --version`, build lại, và xác nhận dòng init `Native GPU (CUDA/HIP)` (§5). Nguyên nhân thường gặp trên image thuần (§3) là thiếu CUDA Toolkit lúc build.
- Tạo instance thất bại với lỗi quota — GPU quota (§1). Yêu cầu tăng quota cho loại GPU trong region cộng thêm global quota.
- Tạo instance thất bại với lỗi zone — loại GPU không có ở zone đó. Chọn một zone trong trang GPU regions and zones availability được liên kết từ [trang tổng quan tạo máy ảo](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).
- Tài liệu troubleshooting đầy đủ: [Troubleshooting](../troubleshooting.md).

---

## 10. Đi tiếp ở đâu

- Lần chạy đầu tiên trên máy ảo: [Quickstart](../quickstart.md).
- Tài liệu cài đặt đầy đủ: [Installation](../installation.md).
- Cài đặt container: [Installation on Docker](docker.md).
- Cài đặt notebook: [Installation on Kaggle](kaggle.md).

---

## Nguồn

Mọi sự kiện cloud trong trang này đều từ tài liệu chính thức của Google Cloud, truy xuất 2026-10-27:

- Mẫu GPU và dòng máy, Spot VM, host maintenance, GPU quota, phạm vi SLA: [About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)
- Tạo máy ảo GPU, lựa chọn OS, giới hạn G2/DLVM, quota: [Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)
- Nội dung DLVM (framework, gói Python, CUDA/cuDNN/NCCL): [Introduction to Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/introduction)
- Các họ image DLVM hiện tại: [Choose an image](https://cloud.google.com/deep-learning-vm/docs/images)
- Cờ tạo instance DLVM: [Create a Deep Learning VM instance from the command line](https://cloud.google.com/deep-learning-vm/docs/cli)
- Tạo instance G2/G4: [Create a G2 or G4 instance](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series)
- Giá GPU (chỉ liên kết, không trích giá): [VM instance pricing](https://cloud.google.com/compute/gpus-pricing)
- Vertex AI custom training và endpoint: [Train and deploy your model](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/train-and-deploy-model), [Prediction classes](https://cloud.google.com/vertex-ai/docs/python-sdk/prediction-classes)
- DLVM với Vertex AI: [Use Deep Learning VM Images and Deep Learning Containers with Vertex AI](https://cloud.google.com/vertex-ai/docs/general/deep-learning)
