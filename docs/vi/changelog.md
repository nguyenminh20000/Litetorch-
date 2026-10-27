# Nhật Ký Thay Đổi — Viết Lại Tài Liệu

Đây là nhật ký thay đổi cho chính quá trình viết lại tài liệu LiteTorch (2026-10-26 → nay): mỗi đợt đã đưa vào những gì, xác minh ra sao, tìm thấy khuyết điểm nào, và còn gì mở. Thay đổi sản phẩm được theo dõi trong repo framework; trang này chỉ theo dõi tài liệu.

- **Cơ sở:** quá trình viết lại bắt đầu từ bộ khung trong `integration/docs-en/` + `docs-vie/` (mỗi bên 12 mục, ~1.600 dòng EN / ~1.800 dòng VI) cộng `integration/docs/mobile-arm.md` + `tokenizers.md`. Bản viết lại nằm trong `workspace/framework/docs/` (`en/` + `vi/`), với patch trong `workspace/framework/patches/docs/` để QA duyệt. Không viết trực tiếp vào `integration/`.
- **Chuẩn xác minh:** mọi ví dụ mã đều chạy trên cây hiện tại — tối thiểu CPU, GPU xác minh trên Kaggle qua QA. Ví dụ không chạy được là khuyết điểm. Parity với PyTorch (torch 2.14.1+cpu) ở FP32 1e-4, cả giá trị VÀ gradient. Phương pháp đầy đủ: `plan/verification.md`.
- **Cơ sở extension:** mọi đợt tới nay đều xác minh trên extension D3 của Core `build_commit 88a4ba1` (`workspace/framework/core/litetorch-d3/python`). HEAD của cây đã di chuyển kể từ đó (nay là `b326f29`); B5 xác minh lại tại head.

## Đợt B0 — Harness, glossary, sửa khuyết điểm (2026-10-26)

- **Đã đưa vào:** harness xác minh (`docs/tools/`: `extract_examples.py`, `run_examples.py`, `parity_check.py`, `requirements-check.txt`); glossary tiếng Việt (`docs/glossary/vi-terms.md`, 50 thuật ngữ, 11 điểm không nhất quán đã ghi); phân tích khuyết điểm D1–D8 + cách giải quyết D9.
- **Xác minh:** trích xuất bộ khung chạy 122 ví dụ (61 EN + 61 VI): PASS 68 / FAIL 54. Phân loại lỗi trong `plan/skeleton-failures.md`: Nhóm A DEPENDENT 19 ID, Nhóm B không khớp API thật 7 ID, Nhóm C treo 1 ID (8.2-1, rendezvous không có mp spawn). Mọi lỗi đối xứng EN/VI — 27 ID duy nhất.
- **Phát hiện:** D9 (mục §8.3 FSDP dùng `lt.distributed.FullyShardedDataParallel`, không tồn tại ở đâu trong cây) — giải quyết qua Distributed dist-w16: viết lại quanh `fsdp_fully_shard`. Đo thời gian build: build đầy đủ cục bộ không hoàn thành trong một session (bị kill ở 44.6 phút, 99/167 nguồn) — build xác minh chuyển sang Kaggle CPU.
- **Khuyết điểm đã gửi:** yêu cầu sửa D-06 + D-07 cho Core (comms `20261026-212600-docs-to-core-d6f1.md`); câu hỏi D9 cho Distributed (comms `20261026-212800-docs-to-distributed-d9f2.md`); phát hiện kèm D9 cho QA (comms `20261026-213400-docs-to-qa-d9side.md`).

## Đợt B1 — Phần mở đầu + tutorial (2026-10-26)

- **Đã đưa vào:** phần mở đầu, 10 trang, 1.828 dòng (`en/index.md`, `en/installation.md`, `en/installation/docker.md`, `en/installation/kaggle.md`, `en/quickstart.md` + bản mirror VI). Tutorial T1–T4, 8 tập tin, 2.178 dòng, 50 block có hàng rào.
- **Xác minh:** phần mở đầu 16/16 PASS (8 block × EN/VI) với ext D3 88a4ba1; mọi đầu ra kỳ vọng đã ghi đều khớp từng byte. Tutorial 48/48 PASS, 0 FAIL, 2 SKIPPED (biến thể GPU T2.4, REQUIRES-GPU/Kaggle-B5). Parity FP32 1e-4: loss CE+gradient giống hệt, giá trị flash/paged attention giống hệt torch, w8a8 chính xác so với công thức int8, JIT chính xác, GELU fused chính xác so với `F.gelu(approximate='tanh')` (D-09).
- **Phát hiện:** `DataLoader(shuffle=True)` KHÔNG tái lập được dù đã `lt.manual_seed(42)` (D-46) — các ví dụ chuyển sang `shuffle=False`. Không có binary Docker cục bộ (Dockerfile chưa chạy); kiểm tra Kaggle GPU chưa chạy (B5); parity torch của phần mở đầu chưa chạy (torch nằm trong `~/torch-venv`, không phải pip hệ thống) — mục B5.

## Đợt B2 — Hướng dẫn how-to + tham chiếu lõi (2026-10-26)

- **Đã đưa vào:** how-to H1–H8, 16 trang, 2.854 dòng. Tham chiếu R1–R4, 8 tập tin, 3.008 dòng (1.504 EN / 1.504 VI), 127 mục.
- **Xác minh:** how-to PASS 50 / FAIL 0 / SKIPPED 6 (56 block; H1.3 GPU bỏ qua, 2 fragment skip-run). Parity: H3.3 4.8e-07, H4.1 8.9e-08, H5.4 7.5e-09 (fwd+grad), H8.1 gradcheck True, H8.2 0.0/0.0. Tham chiếu R1–R4: 112/112 PASS, parity 70/70 PASS FP32 1e-4 kể cả gradient autograd.
- **Phát hiện:** lỗi bộ khung đã sửa (treo Nhóm C qua mp spawn; Nhóm B paged_attention/scheduler/prefix-cache/w8a8; viết lại D9 §8.3; D3 `lt.DataType.INT64`; D4 H7.1 độc lập; D6). Khuyết điểm mới H7.2 gửi cho Core (comms `20261026-233500-docs-to-core-h7d2.md`): `Dataset` backed bởi Python abort khi tắt (SIGABRT) trừ khi `del loader/dataset` tường minh; trampoline điều phối tới `size`/`get`, không phải `__len__`/`__getitem__` (D-37).
- **Bản sửa đã vào trong đợt:** D-07 hợp nhất agent-v1 @7b8981e; D-06 hợp nhất agent-v1 @89afd67 (qa-r62). HEAD của cây di chuyển 114f2e1 → 89afd67 trong đợt.

## Đợt B3 — Tham chiếu API đầy đủ R5–R16 (2026-10-26/27)

- **Đã đưa vào:** 24 tập tin, tổng 15.720 dòng.
  - R5–R6 (ref-a): 4 tập tin, 1.680 dòng, 50 mục. Ví dụ 40/40 PASS, parity 39/39 PASS (tệ nhất 1.9e-06).
  - R7–R8 (ref-b): 4 tập tin, 6.234 dòng, 109 lớp nn + `nn.generate` + 10 functional + 11 init + 5 utils. Ví dụ 136/136 PASS, parity 100 kiểm tra PASS.
  - R9–R11 (ref-c): 6 tập tin, 3.928 dòng. Ví dụ 196/196 PASS, parity 86/88.
  - R12–R16 (ref-d): 10 tập tin, 3.878 dòng, ~129 mục. Ví dụ 68/74 PASS (6 fragment bỏ qua), parity 7/9.
  - **Tổng đợt:** 440 ví dụ PASS, 0 FAIL; parity 232 kiểm tra PASS.
- **Khuyết điểm đã gửi:**
  - Core, comms `20261026-234500-docs-to-core-ceint64.md`: CE/NLL/fused-LCE từ chối target INT64 (D-23).
  - Distributed, comms `20261026-234600-docs-to-distributed-ring.md`: `ring_attention` không thể gọi từ Python (D-38).
  - Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md`: lỗi NAdam (D-30), lệch OneCycleLR (D-31), hoán đổi NegativeBinomial (D-32), treo AsyncCheckpointSaver (D-36, ghi chú kèm).
  - Core, comms `20261027-000800-docs-to-core-ddp-grad.md`: nhân đôi gradient DDP (D-33).
  - Core, comms `20261027-001600-docs-to-core-r7-divergences.md`: D-22–D-29 (`__call__` đa đối số, target loss INT32, đăng ký Module, tạo Parameter, HingeEmbeddingLoss, MultiMarginLoss, CTCLoss, TransformerDecoderLayer) + forward VocabParallelEmbedding hỏng (D-39).
- **Sự cố môi trường:** VM bị thay giữa đợt; torch trong `~/torch-venv` bị xóa (còn 27 gói, dist-info sympy hỏng). Torch 2.14.1+cpu cài lại với `--ignore-installed`; mọi xác minh sau khi cài lại đều dùng venv đã khôi phục. HEAD của cây di chuyển 88a4ba1 → b326f29 (đồng bộ tới remote agent-v1 @cdbc716c, hợp nhất sigmoid). Drift binding 88a4ba1→b326f29 trên mọi tập tin trích dẫn: đúng hai thay đổi — bản sửa D-06 (`bind_tpu.cpp:155`) và một `NoGradGuard` trong `load_parameters` (không ảnh hưởng API).

## Đợt B4 — Giải thích + phần mở đầu (2026-10-27, đang tiến hành)

- **Đã đưa vào (worker này):** `migration.md` (bảng ánh xạ PyTorch → LiteTorch + ví dụ chuyển đổi, parity 5.96e-08), `behavioral-differences.md` (chuẩn D-01–D-46 kèm trạng thái sửa + minh họa trực tiếp), `troubleshooting.md` (triệu chứng → nguyên nhân → cách sửa, chỉ từ lỗi đã xác minh), `changelog.md` (trang này). EN + VI parity đầy đủ.
- **Đã đưa vào (worker song song, docs-b4-expl):** giải thích E1–E8 (tensor, autograd, giải phóng bộ nhớ, backend, phân tán, fusion JIT, +2).
- **Mới trong đợt này:** D-30–D-46 được đánh số và đưa vào trang Khác Biệt Hành Vi; D-01 được đính chính (data_ptr chỉ có trong C++ — không bind trong Python, đã xác minh trực tiếp); dòng khắc phục tạm thời `.cast(INT64)` lỗi thời của D-06 và dòng "ép kiểu kết quả về lại" của D-07 bị loại khỏi tài liệu.
- **Xác minh:** xem `results/b4-front2-<timestamp>/summary.md` (log từng ví dụ kèm theo).

## Khoảng trống đã biết (B5 — xác minh GPU/TPU qua QA)

- Parity torch của phần mở đầu (8 ví dụ của B1-front, `torch-parity=n/a`) — torch nằm trong `~/torch-venv`.
- Dockerfile (không có binary docker cục bộ — chưa chạy, đã ghi trên trang).
- Kiểm tra Kaggle GPU: biến thể GPU T2.4, H1.3 (huấn luyện 2-GPU), `flash_attention_qkv`, chú thích `UNVERIFIED-ROCm`, gói TPU (push `kaggle-run-inputs` riêng, chỉ sau khi GPU xanh).
- Xác minh lại tại head: mọi đợt đều xác minh trên ext 88a4ba1; HEAD của cây là b326f29. B5 chạy lại harness trên extension head.
- Kiểm tra hồi quy: L-02 (parity linalg D-07), L-05 (`.long()` INT64 của D-06 làm chỉ mục), chạy lại ví dụ `r1-cast-long` (đầu ra in của nó thay đổi theo bản sửa D-06; header đặt lại thành `unverified`).
- Kiểm tra trực tiếp L-01–L-04 (Kaggle GPU/TPU): D-02 rơi về với GPU thật nhưng bận, hướng chuyển D-10, độ lớn sai số FP32 int64 D-21, nâng kiểu lượng tử D-19 so với torch, `flash_attention_half` D-20 trên TPU.
- Khuyết điểm mở đang chờ team sở hữu phân loại: D-22–D-29, D-30–D-33, D-37–D-39 (đã gửi, xem trang Khác Biệt Hành Vi để có thread comms); vấn đề mở D-14 (`.numpy()` INT64), D-01, D-02, D-09, D-20, D-35, D-40–D-45; OOM sau huấn luyện (9 lần) vẫn mở. OOM MNIST [BLOCKER] ĐÃ ĐÓNG 2026-10-27 vì không tái hiện được (Path A PASS trên b476a5e, đúng cấu hình; không có nguyên nhân được đặt tên, không tìm thấy ứng viên sửa).
