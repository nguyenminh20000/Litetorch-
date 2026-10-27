# Cài đặt trên AWS

AWS cung cấp tài nguyên GPU cho LiteTorch mà không cần phần cứng tại chỗ, qua bốn hướng: một EC2 GPU instance thuần, một training job hoặc endpoint SageMaker được quản lý, hoặc một container trên ECS hay EKS. Trang này đối chiếu các hướng, liệt kê các họ EC2 GPU hiện tại, và đưa ra các bước LiteTorch sau khi instance đã chạy. Trang này không lặp lại các bước cài qua pip và biên dịch từ mã nguồn: các bước đó nằm trong [Cài đặt](installation.md) §2.2 (pip) và §2.4 (biên dịch từ mã nguồn).

> Ghi chú trạng thái: mọi bước cấp phát cloud và mọi lần chạy GPU trên cloud trong trang này đều có `status=unverified`. Tác giả không có tài khoản AWS và không có thông tin xác thực, nên không có instance nào được khởi chạy, không có AMI nào được boot, và không có lần chạy GPU nào được thực thi trên AWS. Các sự kiện về cloud dưới đây lấy từ tài liệu chính thức của AWS, được trích dẫn theo từng khẳng định; các bước kiểm tra LiteTorch là những bước đã được kiểm chứng cục bộ trên CPU và trên GPU Kaggle trong runbook của dự án, áp dụng ở đây cho một đích chưa từng khởi chạy. Đừng coi bất kỳ bước nào trên trang này là đã chạy kiểm chứng cho đến khi một hạng mục kiểm tra AWS của Batch B5 đóng lại. Docker không được cài trên máy kiểm tra tài liệu, nên Docker runtime trên trang này cũng chưa được kiểm chứng.

---

## 1. Chọn hướng nào

| Hướng | AWS chạy gì cho bạn | Chọn khi |
| :--- | :--- | :--- |
| EC2 GPU instance (§2, §3) | Một máy chủ ảo có GPU NVIDIA; bạn tự quản lý hệ điều hành, driver và bản cài LiteTorch | Bạn muốn toàn quyền kiểm soát, phát triển tương tác, hoặc một bản build tùy chỉnh |
| SageMaker training job (§4) | Tài nguyên huấn luyện được quản lý; bạn cung cấp script hoặc container | Bạn muốn huấn luyện mà không quản lý máy chủ |
| SageMaker endpoint (§4) | Một endpoint suy luận HTTPS được quản lý | Bạn muốn phục vụ một mô hình đã huấn luyện |
| ECS / EKS (§5) | Điều phối container trên các instance GPU | Bạn đã đóng gói LiteTorch thành container và muốn mở rộng quy mô |

Trang này không nêu giá: giá thay đổi nhanh. Hãy kiểm tra [trang giá EC2 On-Demand](https://aws.amazon.com/ec2/pricing/on-demand/) cho loại instance và region bạn định dùng trước khi khởi chạy bất cứ thứ gì. Trang này cũng không nêu hạn mức (quota), tính khả dụng theo region, hay SLA; hãy xác nhận các thông tin đó trong console AWS cho tài khoản và region của bạn tại thời điểm khởi chạy.

---

## 2. Các EC2 GPU instance

AWS chia các instance GPU thành hai dòng: **P** cho huấn luyện ML và HPC, **G** cho đồ họa nặng và suy luận ML. Bảng đối chiếu họ instance với GPU dưới đây lấy từ trang GPU được khuyến nghị của AWS Deep Learning AMI (DLAMI), đối chiếu chéo với các trang EC2 theo từng họ:

| Họ | GPU | GPU trên mỗi instance (tối đa) | Nguồn |
| :--- | :--- | :--- | :--- |
| G4dn | NVIDIA T4 | 4 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G5 | NVIDIA A10G | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G6 | NVIDIA L4 Tensor Core, 24 GB trên mỗi GPU | 8 | [EC2 G6 instances](http://aws.amazon.com/ec2/instance-types/g6/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G6e | NVIDIA L40S Tensor Core | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P3 | NVIDIA V100 | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P4d | NVIDIA A100 Tensor Core | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) (họ P4), [ECS GPU instances](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html) (p4d: A100) |
| P5 | NVIDIA H100 Tensor Core, 80 GB HBM3 trên mỗi GPU | 8 | [EC2 P5 instances](http://aws.amazon.com/ec2/instance-types/p5/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P5e / P5en | NVIDIA H200 Tensor Core | 8 | [EC2 P5 instances](http://aws.amazon.com/ec2/instance-types/p5/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P6-B200 | NVIDIA Blackwell B200 | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |

Ghi chú về bảng:

- G6 là họ suy luận đa dụng hiện tại: trang EC2 G6 ghi các instance này "powered by NVIDIA L4 Tensor Core GPUs" cho "deep learning inference and graphics-intensive applications", tối đa 8 GPU L4 với 24 GB mỗi GPU, cộng thêm các cỡ GPU phân mảnh (chỉ 1/8 GPU L4, 3 GB) cho các workload không dùng hết một GPU.
- P5 là họ huấn luyện lớn hiện tại: trang EC2 P5 ghi P5 cung cấp tối đa 8 GPU NVIDIA H100 (tổng 640 GB HBM3 trên mỗi instance) và P5e/P5en tối đa 8 GPU NVIDIA H200 (tổng 1128 GB HBM3e), với mạng EFA lên tới 3.200 Gbps và kết nối NVSwitch giữa các GPU.
- P3 (V100) là thế hệ trước, vẫn được liệt kê trong hướng dẫn DLAMI; hãy ưu tiên G6 hoặc P5 cho công việc mới trừ khi bạn có lý do cụ thể để dùng V100.
- P6-B200 (Blackwell) là họ mới nhất được hướng dẫn DLAMI liệt kê. Thư viện GPU native của LiteTorch chưa từng được build hay chạy trên Blackwell trong dự án này, nên một lần chạy P6-B200 chưa được kiểm chứng theo cả hai nghĩa: không có lần chạy AWS và không có bản build Blackwell.
- G5g (Arm64, Graviton2) được hướng dẫn DLAMI liệt kê như một họ Arm64 riêng với DLAMI riêng; các bước build LiteTorch trong [Cài đặt](installation.md) §2.4 được viết cho x86_64, nên G5g nằm ngoài phạm vi trang này.

Hướng dẫn LiteTorch: bắt đầu trên một G6 đơn GPU (suy luận, huấn luyện nhỏ, phát triển) hoặc G5 (mô hình lớn hơn, huấn luyện nhanh hơn); chuyển sang P4d hoặc P5 khi bạn cần huấn luyện đa GPU hoặc bộ nhớ GPU lớn nhất. Chính hướng dẫn DLAMI cũng khuyên kích thước mô hình nên quyết định lựa chọn: nếu mô hình vượt quá bộ nhớ khả dụng của instance, hãy chọn một loại instance khác.

---

## 3. Khởi chạy một EC2 instance: Deep Learning AMI so với Ubuntu thuần

### 3.1. Deep Learning AMI (DLAMI)

DLAMI là một image máy do AWS cung cấp với driver NVIDIA và phần mềm deep learning đã cài sẵn: các GPU AMI đi kèm driver NVIDIA, CUDA Toolkit, NCCL, cuDNN, Docker và NVIDIA Container Toolkit đã cài sẵn, cộng thêm một môi trường framework (các PyTorch DLAMI đi kèm một môi trường PyTorch). Phiên bản driver, CUDA, cuDNN và PyTorch chính xác khác nhau theo từng bản phát hành DLAMI: hãy đọc chúng từ release notes của DLAMI tại thời điểm khởi chạy; trang này không ghim phiên bản nào.

Các bước khởi chạy, theo hướng dẫn dành cho nhà phát triển DLAMI:

1. Mở EC2 console.
2. Chọn **Launch Instance**.
3. Nhập tên và chọn DLAMI (tìm theo DLAMI ID, hoặc chọn từ Quick Start / My AMIs).
4. Chọn loại instance. Release notes của DLAMI liệt kê các họ instance được khuyến nghị cho từng DLAMI.
5. Chọn **Launch Instance**, rồi đợi instance sẵn sàng (thường chỉ vài phút) và kết nối tới nó.

Chọn DLAMI khi bạn muốn khởi đầu nhanh nhất: driver và bộ CUDA đã được cài sẵn và khớp phiên bản với nhau, nên lỗi thường gặp (driver/CUDA lệch phiên bản) được loại bỏ. Chọn PyTorch DLAMI, không phải Base DLAMI, nếu bạn muốn một môi trường PyTorch sẵn sàng bên cạnh LiteTorch; chọn Base DLAMI nếu bạn chỉ muốn bộ NVIDIA và định tự cài mọi thứ còn lại.

### 3.2. Ubuntu thuần + tự cài CUDA

Khởi chạy một AMI Ubuntu thuần (Ubuntu 22.04 là phiên bản mà các image Docker của LiteTorch và các image PyTorch của DLAMI dùng) rồi tự cài NVIDIA driver, CUDA Toolkit và NVIDIA Container Toolkit, theo hướng dẫn cài đặt trên re:Post của AWS cho các instance EC2 Ubuntu.

Chọn Ubuntu thuần khi bạn cần toàn quyền kiểm soát: một phiên bản CUDA cụ thể, một image tối giản không có framework cài sẵn, hoặc một bản cài khớp chính xác với quy trình biên dịch từ mã nguồn trong [Cài đặt](installation.md) §2.4. Cái giá phải trả là bạn tự chịu trách nhiệm ghép cặp driver/CUDA; một sự không khớp sẽ lỗi giống như khi lỗi cục bộ (thư viện GPU native không bao giờ build, rơi về CPU một cách lặng lẽ — xem §6).

### 3.3. Sau khi instance đã chạy: cài LiteTorch

Khi bạn đã SSH được vào instance, các bước LiteTorch giống như trên bất kỳ máy Ubuntu nào. Đừng làm theo một quy trình cài đặt riêng chỉ dành cho AWS; hãy làm theo [Cài đặt](installation.md) §2.2 cho pip (`pip install --upgrade litetorch`) hoặc §2.4 cho bản build từ mã nguồn (`git clone`, `pip install pybind11`, `python3 setup.py build_ext --inplace`). Trước cả hai hướng, hãy chạy các bước kiểm tra trong §6 — bước kiểm tra CUDA toolkit và cổng kiểm tra pre-flight — vì đó là các bước bắt lỗi một thiết lập GPU hỏng trước khi tốn bất kỳ thời gian huấn luyện nào.

<!-- ex: id=aws-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
ls /opt/litetorch/python/
```

`nvidia-smi` chứng minh driver nhìn thấy GPU; `nvcc --version` chứng minh CUDA toolkit đã được cài (quy tắc đứng: kiểm tra CUDA toolkit trước mọi lần chạy GPU). Lệnh `ls` chứng minh extension đã build nằm dưới `<tree>/python/` — `setup.py` đặt `package_dir={"": "python"}`, nên extension có thể import nằm dưới `<tree>/python/`, không phải thư mục gốc của cây mã nguồn. Nếu không tìm thấy `nvcc`, hãy cài CUDA Toolkit rồi chạy lại cho đến khi nó in ra một phiên bản; chỉ sau đó mới build hoặc chạy LiteTorch trên GPU. Nếu `nvidia-smi` lỗi, vấn đề nằm ở driver, không phải LiteTorch: hãy sửa driver trước (người dùng DLAMI: khởi chạy lại từ DLAMI thay vì gỡ lỗi driver bằng tay).

---

## 4. SageMaker training job và endpoint (ngắn gọn)

Amazon SageMaker AI là dịch vụ ML được quản lý: nó cấp phát tài nguyên tính toán, chạy huấn luyện của bạn và phục vụ mô hình, mà bạn không cần quản lý máy chủ. SageMaker chấp nhận mọi mô hình, mọi framework và mọi container.

- **Training job.** Một training job chạy code huấn luyện của bạn trên tài nguyên tính toán được quản lý (CPU, GPU hoặc cụm phân tán). Có hai cách đưa code LiteTorch vào:
  - **Script mode:** đưa script huấn luyện của bạn vào; SageMaker lấy container do AWS quản lý phù hợp, tải script của bạn xuống và chạy nó. Đây là hướng ít công sức nhất, nhưng không có container LiteTorch do AWS quản lý, nên script mode chỉ phù hợp khi container được quản lý đã chứa mọi thứ LiteTorch cần — mà với một LiteTorch build từ mã nguồn thì không.
  - **Bring your own container (BYOC):** đóng gói thuật toán và các dependency của bạn trong một image Docker, đẩy lên Amazon ECR, và SageMaker lấy image đó về rồi chạy huấn luyện. Đây là hướng đi của LiteTorch: build image từ Dockerfile CUDA của LiteTorch trong [Cài đặt trên Docker](installation/docker.md), đẩy lên ECR và trỏ training job tới nó. Chỉ ở mức dàn ý — định nghĩa training job chính xác và hợp đồng container chưa được kiểm chứng (không có lần chạy AWS) và không được nêu ở đây; hãy làm theo hướng dẫn dành cho nhà phát triển SageMaker về các yêu cầu đối với container.
- **Endpoint.** Một mô hình đã huấn luyện được triển khai thành một endpoint suy luận thời gian thực (một endpoint HTTPS được quản lý) hoặc thành một batch transform job để dự đoán trên một tập dữ liệu lớn mà không cần endpoint thường trực. Đối với một container suy luận tự mang theo, SageMaker yêu cầu container phục vụ một web server trên cổng `8080` và chấp nhận các request `POST` tới các endpoint `/invocations` và `/ping`. LiteTorch không có inference server trong trang này, nên một endpoint cần một lớp wrapper phục vụ bao quanh suy luận LiteTorch; lớp wrapper đó nằm ngoài phạm vi và chưa được kiểm chứng.

LiteTorch phù hợp như thế nào, gói trong một câu: huấn luyện SageMaker qua BYOC với image CUDA của LiteTorch; script mode của SageMaker không áp dụng cho một LiteTorch build từ mã nguồn; các endpoint cần một lớp wrapper phục vụ chưa tồn tại. Trang này không đưa ra lệnh gọi API SageMaker nào — mọi bước SageMaker đều chưa được kiểm chứng.

---

## 5. ECS và EKS với Dockerfile của LiteTorch (ngắn gọn)

### 5.1. ECS

Amazon Elastic Container Service chạy các container trên tài nguyên EC2 được quản lý. Hỗ trợ GPU trên ECS Managed Instances bao gồm các họ `g4dn` (NVIDIA T4), `g5` (NVIDIA A10G), `p3` (NVIDIA V100), `p4d` (NVIDIA A100) và `g6f` (NVIDIA L4 phân mảnh), và driver NVIDIA cùng CUDA toolkit đã được cài sẵn trên các instance có GPU. Dàn ý: đẩy image CUDA của LiteTorch (build từ [Cài đặt trên Docker](installation/docker.md)) lên ECR, viết một task definition yêu cầu tài nguyên GPU, và chạy nó dưới dạng một ECS service hoặc task trên một GPU capacity provider. Chỉ ở mức dàn ý — không có task definition nào được đưa ra, và không có lần chạy ECS nào được thực thi.

### 5.2. EKS

Amazon Elastic Kubernetes Service chạy Kubernetes với các node group GPU. Tài liệu EKS cho thấy các GPU NodePool chọn các họ `g6` và `g6e`, các pod yêu cầu GPU qua resource `nvidia.com/gpu`, và NVIDIA device plugin phơi bày GPU của node cho bộ lập lịch. Một pod yêu cầu một GPU trông như sau (rút gọn từ hướng dẫn workload tăng tốc của EKS):

<!-- ex: id=aws-2 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: litetorch-gpu
spec:
  restartPolicy: OnFailure
  containers:
    - name: litetorch
      image: <your-ecr-registry>/litetorch:cuda
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
```

Thay `<your-ecr-registry>` bằng URI registry ECR của bạn; image là image CUDA của LiteTorch từ [Cài đặt trên Docker](installation/docker.md). Yêu cầu `nvidia.com/gpu: 1` là thứ đưa pod lên một node GPU; thiếu nó, pod sẽ rơi vào tài nguyên CPU và LiteTorch lặng lẽ rơi về CPU (§6).

> Docker runtime chưa được kiểm chứng: không có binary Docker trên máy kiểm tra tài liệu, nên image CUDA của LiteTorch chưa được build hay chạy tại đây, và không có lần chạy ECS hay EKS nào được thực thi. Nội dung Dockerfile là deliverable đã được xem xét; bản thân quá trình build và các lần chạy ECS/EKS là các hạng mục kiểm tra của Batch B5.

---

## 6. Ghi chú LiteTorch trên mọi hướng

Các ghi chú này áp dụng giống hệt nhau trên EC2, SageMaker BYOC, ECS và EKS. Chúng là các bước kiểm tra phân biệt "GPU đang được dùng" với "lần chạy lặng lẽ dùng CPU".

1. **Kiểm tra CUDA toolkit trước mọi lần chạy GPU.** Chạy `nvcc --version` trước, trên instance hoặc bên trong container. Thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ build, và framework lặng lẽ rơi về CPU.
2. **Chạy cổng kiểm tra pre-flight.** Xác nhận dòng khởi tạo `Native GPU (CUDA/HIP)` xuất hiện khi import, và xác nhận extension đã build cùng `liblitetorch_gpu.so` nằm cạnh nó dưới `<tree>/python/`. Cổng kiểm tra là: dòng init xuất hiện VÀ vị trí `.so` đúng. Đừng bao giờ chỉ tin dòng `BUILD OK`; `setup.py` in nó ngay cả khi các lần biên dịch từng file đã lỗi.
3. **Chuỗi device là `gpu:0`, không phải `cuda:0`.** Chuỗi device của LiteTorch dùng tiền tố `gpu:`: `str(t.device)` là `gpu:0` đối với GPU đầu tiên. Code viết `"cuda:0"` dưới dạng chuỗi device LiteTorch là sai; bộ phân tích chấp nhận `cuda` làm đầu vào, nhưng chuỗi chuẩn mà một device in ra là `gpu:N`.
4. **Rơi về CPU một cách lặng lẽ là lỗi mặc định.** Nếu backend GPU không tải được, `.to(device)` lặng lẽ rơi về CPU. Triệu chứng là một lần chạy hoàn thành với tốc độ CPU và kết quả CPU; cách chẩn đoán là dòng init bị thiếu trong bước 2, không phải đầu ra huấn luyện.

<!-- ex: id=aws-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "/opt/litetorch/python")
import litetorch as lt

print(lt.is_gpu_available())
print(lt.get_backend_name())
print(lt.auto_device())
```

Kết quả mong đợi trên một đường GPU hoạt động: `True`, tên một backend CUDA, và `gpu:0`. Nếu dòng thứ ba in `cpu:0` trên một máy mà bạn mong đợi có GPU, backend GPU đã không tải — kiểm tra `nvcc --version`, build lại, và xác nhận dòng khởi tạo `Native GPU (CUDA/HIP)`. Dòng `sys.path` chỉ cần thiết đối với một cây build từ mã nguồn; bản cài qua pip không cần nó. Hãy chỉnh đường dẫn tới nơi cây mã nguồn nằm trên instance của bạn hoặc trong image của bạn.

5. **Dừng những gì bạn không dùng.** Các instance GPU vẫn tính phí khi đang chạy. Hãy stop (không chỉ ngắt kết nối khỏi) một EC2 instance khi công việc tạm dừng, và xóa các endpoint cùng tài nguyên GPU ECS/EKS nhàn rỗi khi không còn cần chúng:

<!-- ex: id=aws-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
aws ec2 stop-instances --instance-ids i-0123456789abcdef0
```

Thay instance ID bằng ID của bạn. Stop giữ lại volume EBS; terminate xóa instance. Dù cách nào, hãy xác nhận trong console rằng instance thực sự đã dừng — một instance GPU đang chạy bị bỏ quên qua đêm là bất ngờ về chi phí AWS thường gặp nhất, và trang này không nêu giá nào để cảnh báo bạn.

---

## 7. Xử lý sự cố

- Lần chạy GPU lặng lẽ dùng CPU — xem §6, bước 1–4: `nvcc`, dòng init, vị trí `.so`, và chuỗi device `gpu:0`.
- `ModuleNotFoundError: No module named 'litetorch'` — `<tree>/python` không nằm trên `sys.path` (build từ mã nguồn) hoặc `PYTHONPATH` (container). Xem [Cài đặt](installation.md) §2.4 và [Cài đặt trên Docker](installation/docker.md).
- `nvidia-smi` lỗi trên EC2 — lỗi driver, không phải lỗi LiteTorch. Trên DLAMI, hãy khởi chạy lại từ DLAMI; trên Ubuntu thuần, hãy cài lại driver theo hướng dẫn cài đặt của AWS.
- Build in `BUILD OK` nhưng import lỗi — các lần biên dịch từng file đã lỗi một cách lặng lẽ; cổng kiểm tra thực sự là file extension tồn tại dưới `<tree>/python/` và `import litetorch` thành công.
- Các vấn đề khác: [Xử lý sự cố](troubleshooting.md).

---

## Nguồn

Tài liệu chính thức của AWS đã tham khảo cho trang này (truy cập 2026-10-27). Mọi khẳng định về họ instance, mẫu GPU, AMI và dịch vụ ở trên đều truy về một trong các trang này; trang này không đưa ra khẳng định nào về giá, hạn mức, tính khả dụng theo region hay SLA.

- EC2 G6 instances (L4, tối đa 8 GPU, 24 GB trên mỗi GPU, các cỡ phân mảnh): <http://aws.amazon.com/ec2/instance-types/g6/>
- EC2 P5 instances (H100/H200, tối đa 8 GPU, 640 GB HBM3 / 1128 GB HBM3e): <http://aws.amazon.com/ec2/instance-types/p5/>
- DLAMI recommended GPU instances (P6-B200/B300, P6e-GB200, P5e, P5, P4, P3, G3, G4, G5, G6, G6e): <https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html>
- DLAMI launching an instance (các bước console/CLI): <https://docs.aws.amazon.com/dlami/latest/devguide/launch.html>
- Nội dung GPU AMI của DLAMI (NVIDIA driver, CUDA Toolkit, NCCL, cuDNN, Docker, NVIDIA Container Toolkit): AWS re:Post, "Deep Learning graphical desktop on Ubuntu Linux with AWS Deep Learning AMI (DLAMI)": <https://repost.aws/articles/AR6RrDeUL1Tq6R8TgDs59iEA/deep-learning-graphical-desktop-on-ubuntu-linux-with-aws-deep-learning-ami-dlami>
- Cài đặt NVIDIA driver, CUDA Toolkit và NVIDIA Container Toolkit trên các instance EC2 Ubuntu: AWS re:Post: <https://repost.aws/articles/ARWGxLArMBQ4y1MKoSHTq3gQ/install-nvidia-gpu-driver-cuda-toolkit-nvidia-container-toolkit-on-amazon-ec2-instances-running-ubuntu-linux>
- Tổng quan SageMaker AI (mang theo mọi mô hình, framework, container): <https://aws.amazon.com/sagemaker/ai/>
- SageMaker script mode so với bring-your-own-container: <https://aws.amazon.com/blogs/machine-learning/use-your-own-training-scripts-and-automatically-select-the-best-model-using-hyperparameter-optimization-in-amazon-sagemaker/> và <https://aws.amazon.com/ko/blogs/machine-learning/introducing-amazon-sagemaker-components-for-kubeflow-pipelines/>
- SageMaker bring-your-own inference container (cổng 8080, `/invocations`, `/ping`): <https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html>
- ECS GPU instances (g4dn T4, g5 A10G, p3 V100, p4d A100, g6f L4 phân mảnh; driver và CUDA cài sẵn): <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html>
- EKS accelerated workloads (GPU NodePool, `nvidia.com/gpu`): <https://docs.aws.amazon.com/eks/latest/userguide/auto-accelerated.html>
- EKS NVIDIA device plugin: <https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia-dra-device-plugin.html>
- Giá EC2 On-Demand (được liên kết, không trích giá): <https://aws.amazon.com/ec2/pricing/on-demand/>
