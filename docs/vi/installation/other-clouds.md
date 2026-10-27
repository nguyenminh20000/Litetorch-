# Cài đặt trên các cloud khác

Trang này bao quát việc chạy LiteTorch trên các cloud GPU ngoài Docker và Kaggle: Lambda Labs (Lambda Cloud), Oracle Cloud Infrastructure (OCI), và một bảng ngắn về các nhà cung cấp khác. Bản thân việc cài đặt LiteTorch giống nhau trên mọi cloud: cài qua pip ([Hướng dẫn cài đặt](../installation.md) §2.2) hoặc biên dịch từ mã nguồn ([Hướng dẫn cài đặt](../installation.md) §2.4). Trang này chỉ bao quát việc lấy máy GPU và các bước kiểm tra đặc thù của LiteTorch sau đó. Nó không lặp lại các bước cài đặt.

> Ghi chú trạng thái: mọi bước provisioning cloud và mọi lần chạy GPU trên cloud trong trang này đều là `status=unverified`. Tác giả không có tài khoản và không có credential trên bất kỳ nền tảng nào trong số này, nên không có gì phía cloud được chạy. Các sự thật về cloud dưới đây chỉ lấy từ tài liệu chính thức của từng nhà cung cấp, trích dẫn theo từng khẳng định. Không có giá nào được nêu: hãy kiểm tra trang giá của nhà cung cấp tại thời điểm chạy.

---

## 1. Lambda Labs (Lambda Cloud)

Public cloud của Lambda cho phép bạn khởi chạy từng máy ảo riêng lẻ hoặc các cụm máy và tắt chúng theo lịch của bạn ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)).

- **On Demand Cloud**: khởi chạy ngay một máy ảo Linux có GPU, dùng nhiều cấu hình chip NVIDIA khác nhau ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)).
- **1-Click Clusters**: các cụm instance GPU gồm 16 đến 512 GPU NVIDIA H100 SXM Tensor Core ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)). Các node của 1-Click Cluster được cài sẵn Ubuntu 22.04 LTS và Lambda Stack, bao gồm NCCL, Open MPI, PyTorch với hỗ trợ DDP và FSDP, và TensorFlow ([Lambda Docs: 1-Click Clusters](https://docs.lambda.ai/public-cloud/1-click-clusters/)).

Cách hoạt động, ở mức phác thảo:

1. Tạo tài khoản Lambda và thêm một SSH key.
2. Khởi chạy một instance On Demand (hoặc một 1-Click Cluster cho công việc đa node) từ console, CLI, hoặc API.
3. Kết nối qua SSH.
4. Cài đặt LiteTorch bằng pip ([Hướng dẫn cài đặt](../installation.md) §2.2.2) hoặc biên dịch từ mã nguồn ([Hướng dẫn cài đặt](../installation.md) §2.4). Kiểm tra CUDA toolkit trước — xem §4 dưới đây.
5. Hủy instance khi xong việc. Instance On Demand vẫn tính phí trong khi nó tồn tại; hãy xác nhận những gì vẫn tính phí sau khi bạn dừng nó (storage, IP) trên trang giá và thanh toán của nhà cung cấp trước khi rời đi.

Kết nối và kiểm tra toolkit:

<!-- ex: id=other-1 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
ssh ubuntu@<INSTANCE-IP>
nvcc --version
nvidia-smi
```

Các mẫu GPU có sẵn theo nhu cầu thay đổi theo thời gian; hãy kiểm tra danh sách instance và trang giá của nhà cung cấp tại thời điểm chạy thay vì tin vào một ảnh chụp ở đây.

---

## 2. Oracle Cloud Infrastructure (OCI)

OCI cho thuê tài nguyên tính toán GPU dưới dạng các shape có tên. Tên shape mã hóa loại hình (`VM.` máy ảo, `BM.` bare metal), dòng GPU, và số lượng GPU. Các shape bare metal cung cấp phần cứng chuyên dụng không có chi phí ảo hóa và hỗ trợ song song tensor đa GPU; các shape máy ảo tiết kiệm chi phí hơn cho các mô hình nhỏ ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).

| Shape | GPU | GPU | Tổng bộ nhớ GPU |
| :--- | :---: | :--- | :---: |
| `VM.GPU.A10.1` | 1 | NVIDIA A10 | 24 GB |
| `VM.GPU.A10.2` | 2 | NVIDIA A10 | 48 GB |
| `BM.GPU4.8` | 8 | NVIDIA A100 40 GB | 320 GB |
| `BM.GPU.A100-v2.8` | 8 | NVIDIA A100 80 GB | 640 GB |
| `BM.GPU.H100.8` | 8 | NVIDIA H100 | 640 GB |

Nguồn bảng: [Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/). `BM.GPU.H100.8` được liệt kê ở đó với hỗ trợ RDMA. `BM.GPU.GB200.4` còn được Oracle liệt kê là một shape hỗ trợ compute cluster ([Oracle Docs: Compute Clusters](https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/compute-clusters.htm)); trang này không nêu số lượng GPU hay bộ nhớ cho nó vì trang được trích dẫn không cung cấp chúng. OCI bổ sung shape theo thời gian — hãy xác nhận danh sách shape hiện tại trong OCI Console hoặc CLI trước khi lập kế hoạch.

Cách hoạt động, ở mức phác thảo:

1. Tạo tài khoản OCI và một compartment có quyền tạo compute instance.
2. Yêu cầu quota tính toán GPU cho shape bạn muốn (ví dụ `VM.GPU.A10.1` hoặc `BM.GPU.A100-v2.8`) qua support ticket nếu bạn chưa có ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).
3. Khởi chạy instance trong một region và availability domain nơi shape đó có sẵn. Dung lượng GPU thay đổi theo region và availability domain; hãy chuẩn bị thử một availability domain khác nếu bạn gặp lỗi hết dung lượng ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).
4. Kết nối qua SSH.
5. Cài đặt LiteTorch bằng pip ([Hướng dẫn cài đặt](../installation.md) §2.2.2) hoặc biên dịch từ mã nguồn ([Hướng dẫn cài đặt](../installation.md) §2.4). Kiểm tra CUDA toolkit trước — xem §4 dưới đây.
6. Hủy instance (và mọi volume đính kèm) khi xong việc. Đây là tài nguyên GPU có trả phí; hãy luôn chạy các bước dọn dẹp để tránh phí phát sinh ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).

Kiểm tra GPU và toolkit trên instance:

<!-- ex: id=other-2 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
```

Không có giá nào được nêu ở đây: hãy kiểm tra trang giá của nhà cung cấp tại thời điểm chạy.

---

## 3. Các nhà cung cấp khác

| Nhà cung cấp | Cung cấp gì | Tài liệu chính thức | Trạng thái |
| :--- | :--- | :--- | :--- |
| RunPod | GPU Pod (instance container với loại GPU và số lượng GPU được chọn) cùng với endpoint Serverless; triển khai Pod từ console, CLI `runpodctl`, hoặc REST API | [RunPod Docs: Manage Pods](https://docs.runpod.io/pods/manage-pods) | `status=unverified` |
| Paperspace | Máy ảo có GPU (Machines) và dịch vụ container dựa trên Kubernetes, cùng với Notebooks, Deployments, và Workflows, để phát triển, huấn luyện, và triển khai ứng dụng AI | [DigitalOcean Docs: Paperspace](https://docs.digitalocean.com/products/paperspace/) | `status=unverified` |
| CoreWeave | Nền tảng cloud được xây dựng riêng cho AI và tính toán hiệu năng cao; CoreWeave Kubernetes Service (CKS) chạy các cụm bare metal trên đội GPU và CPU của hãng, và SUNK chạy Slurm trên Kubernetes cho huấn luyện phân tán | [CoreWeave Docs: Get started](https://docs.coreweave.com/get-started) | `status=unverified` |

Các hàng trên chỉ nêu những gì trang tài liệu chính thức được liên kết xác lập. Dòng GPU, region, quota, và giá của các nhà cung cấp này không được kiểm chứng cho trang này; hãy kiểm tra tài liệu và trang giá của chính nhà cung cấp tại thời điểm chạy.

---

## 4. Ghi chú đặc thù của LiteTorch (mọi cloud)

Các ghi chú này áp dụng trên mọi cloud ở trên, đúng như trên Kaggle và Docker.

- **Kiểm tra CUDA toolkit trước mọi lần chạy GPU.** Chạy `nvcc --version` trước và cài đặt CUDA Toolkit nếu thiếu. Thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ được biên dịch, và framework sau đó lặng lẽ fallback về CPU.
- **Cổng kiểm tra tiền điều kiện GPU.** Trước khi tin bất kỳ kết quả GPU nào, hãy xác nhận cả hai điều sau:
  1. Dòng khởi động `Native GPU (CUDA/HIP)` xuất hiện (KHÔNG phải OpenCL).
  2. `liblitetorch_gpu.so` nằm cạnh extension trong thư mục `python/` (bản sao trong `python/` mới là bản `dlopen` tìm thấy). Trên một số image, `build_ext --inplace` ghi nó vào `build/` thay vì đó; hãy sao chép nó sang `python/` trước khi import.
- **Chuỗi thiết bị là `gpu:N`, không phải `cuda:N`.** `str(t.device)` trả về `gpu:0`, `gpu:1`, v.v. (D-50). Bộ phân tích cũng chấp nhận `cuda` làm đầu vào, nhưng chuỗi chuẩn mà một thiết bị in ra là `gpu:N`. Đừng assert `cuda:0` trong test hay script.

Chạy bước kiểm tra tiền điều kiện này sau khi cài đặt (thêm `<tree>/python` vào `sys.path` trước khi import cây được biên dịch từ mã nguồn):

<!-- ex: id=other-3 device=gpu backend=cuda runtime=~1s torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

print(lt.get_backend_name())
print(lt.auto_device())
print(lt.is_gpu_available())
```

Trên máy GPU, các dòng in ra tên backend CUDA, `gpu:0`, và `True`. Nếu thiết bị in ra `cpu:0` trong khi bạn mong đợi GPU, backend GPU đã không tải — hãy kiểm tra `nvcc --version`, biên dịch lại, và xác nhận dòng khởi động `Native GPU (CUDA/HIP)` xuất hiện. Đừng cho rằng GPU đang được dùng: `.to(device)` lặng lẽ fallback về CPU khi accelerator không khả dụng.

---

## 5. Xử lý sự cố

Xem [Xử lý sự cố và FAQ](../troubleshooting.md) về lỗi biên dịch, hết bộ nhớ, fallback CPU lặng lẽ, và phát hiện backend. Bổ sung riêng cho cloud:

- Backend GPU không tải (`cpu:0` trên máy GPU) — kiểm tra lại `nvcc --version` trên instance, biên dịch lại, và xác nhận dòng khởi động `Native GPU (CUDA/HIP)` cùng vị trí `liblitetorch_gpu.so` trong §4.
- `ModuleNotFoundError: No module named 'litetorch'` — `sys.path` không bao gồm `<tree>/python` (biên dịch từ mã nguồn) hoặc quá trình cài pip chưa hoàn tất; xem [Hướng dẫn cài đặt](../installation.md) §2.4.
- Hết dung lượng host / shape không khả dụng trong region hoặc availability domain đã chọn — thử một availability domain khác hoặc một shape khác; xem mục OCI ở trên.

---

## Nguồn

Tài liệu chính thức đã tra cứu ngày 2026-10-27 (mọi bước cloud đều `status=unverified`; không tài khoản, không credential, không lần chạy cloud nào):

- Lambda Docs, Public Cloud overview: https://docs.lambda.ai/public-cloud/
- Lambda Docs, 1-Click Clusters: https://docs.lambda.ai/public-cloud/1-click-clusters/
- Oracle Docs, Deploy OpenAI vLLM Production Stack on Oracle Kubernetes Engine (OKE): https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/
- Oracle Docs, Compute Clusters: https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/compute-clusters.htm
- RunPod Docs, Manage Pods: https://docs.runpod.io/pods/manage-pods
- RunPod Docs, Deploy your first Pod: https://docs.runpod.io/get-started
- DigitalOcean Docs, Paperspace: https://docs.digitalocean.com/products/paperspace/
- CoreWeave Docs, Get started with CoreWeave: https://docs.coreweave.com/get-started
