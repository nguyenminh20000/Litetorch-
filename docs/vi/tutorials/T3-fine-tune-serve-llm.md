# Hướng dẫn T3 — Fine-tune và phục vụ một LLM nhỏ

Fine-tune một mô hình ngôn ngữ nhỏ trên CPU, rồi phục vụ nó: flash
attention, KV cache dạng paged, continuous batching, prefix caching,
speculative decoding, Medusa heads, và guided decoding. Mọi lời gọi phục
vụ dưới đây đều chạy trên CPU với LiteTorch 88a4ba1 — các shape và chữ
ký là thật, đã sửa ở những chỗ tài liệu khung cũ viết sai.

Bạn sẽ học được:

- Cách fine-tune một mô hình ngôn ngữ embedding + linear nhỏ.
- Cách attention hoạt động trong LiteTorch (`flash_attention`,
  `paged_attention`).
- Cách ngăn xếp phục vụ khớp với nhau: scheduler, prefix cache,
  speculative decoding, Medusa, guided decoding.
- Những chữ ký phục vụ nào tài liệu cũ viết sai, và dạng đúng là gì.

Điều kiện tiên quyết: Hướng dẫn T1–T2. Mô hình ở đây cố tình nhỏ
(vocab 64): chạy trong vài giây trên CPU và mọi cơ chế đều nhìn thấy
được. Các con số throughput thuộc về GPU và được kiểm chứng trên Kaggle
ở B5, không khẳng định ở đây.

> Các điểm khác biệt với PyTorch trong hướng dẫn này: D-20 (không có
> `F.scaled_dot_product_attention` — dùng `Ops.flash_attention(q, k, v)`;
> không có tham số `scale` hay `dropout_p`, tự nhân Q với hệ số scale;
> head dim phải thỏa 1 <= D <= 256), D-03 (`.to()` nhận một `Device`),
> D-02 (rơi lặng lẽ về CPU — kiểm tra thiết bị sau `.to()`).

## T3.1 — Flash attention

`lt.Ops.flash_attention(q, k, v)` tính softmax(QK^T / sqrt(d))V. Không
có tham số `scale`: tự nhân Q với hệ số scale trước lời gọi.
`causal=True` che các vị trí tương lai.

<!-- ex: id=T3.1-1 device=cpu backend=cpu runtime=~8.3s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import math
import litetorch as lt
import torch

device = lt.auto_device()

q = lt.Tensor.from_vector([0.1] * 64, [1, 2, 4, 8], device, False)
k = lt.Tensor.from_vector([0.2] * 64, [1, 2, 4, 8], device, False)
v = lt.Tensor.from_vector([0.3] * 64, [1, 2, 4, 8], device, False)

out = lt.Ops.flash_attention(q, k, v)
print("shape của flash attention:", out.shape)
print("các giá trị đầu:", out.to_vector()[:4])

qt = torch.full((1, 2, 4, 8), 0.1)
kt = torch.full((1, 2, 4, 8), 0.2)
vt = torch.full((1, 2, 4, 8), 0.3)
ref = torch.softmax(qt @ kt.transpose(-2, -1) / math.sqrt(8), dim=-1) @ vt
print("các giá trị đầu của tham chiếu torch:", ref.flatten().tolist()[:4])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(flash attention khớp với `softmax(QK^T/sqrt(d))V` tính tay).

## T3.2 — Paged attention và KV cache

Paged attention chia KV cache thành các block kích thước cố định để
cache không bao giờ bị phân mảnh. Chữ ký có thứ tự và nghiêm ngặt: `q`
có rank 3, `k_cache`/`v_cache` có rank 4 `[num_blocks, num_kv_heads,
block_size, head_dim]`, `block_tables` và `context_lens` có rank 2 /
rank 1 và phải là INT32, còn `block_size` là một tham số int thường —
không phải keyword. (Khung cũ §9.1 truyền bảng FP32, cache rank 3, và
`block_size=16` dạng keyword; cả ba đều sai.)

<!-- ex: id=T3.2-1 device=cpu backend=cpu runtime=~3.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import math
import litetorch as lt
import torch

device = lt.auto_device()

q = lt.Tensor.from_vector([0.1] * 16, [1, 1, 16], device, False)
k_cache = lt.Tensor.from_vector([0.2] * (2 * 1 * 4 * 16), [2, 1, 4, 16], device, False)
v_cache = lt.Tensor.from_vector([0.3] * (2 * 1 * 4 * 16), [2, 1, 4, 16], device, False)
block_tables = lt.Tensor.from_vector([0.0, 1.0], [1, 2], device, False, lt.DataType.INT32)
context_lens = lt.Tensor.from_vector([4.0], [1], device, False, lt.DataType.INT32)

out = lt.Ops.paged_attention(q, k_cache, v_cache, block_tables, context_lens, 4)
print("shape của paged attention:", out.shape)
print("các giá trị đầu:", out.to_vector()[:4])

qt = torch.full((1, 1, 16), 0.1)
kt = torch.full((4, 16), 0.2)
vt = torch.full((4, 16), 0.3)
ref = torch.softmax(qt.squeeze(0) @ kt.T / math.sqrt(16), dim=-1) @ vt
print("các giá trị đầu của tham chiếu torch:", ref.flatten().tolist()[:4])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(paged attention trên 4 vị trí đã cache khớp với attention tính tay).

## T3.3 — Continuous batching với serving scheduler

Scheduler nhận request, cấp phát block KV, và bước mô hình một lần mỗi
tick: prefill cho prompt mới, một token decode cho mỗi request đang
chạy. `step(model)` nhận mô hình và trả về các request hoàn thành ở
bước này.

<!-- ex: id=T3.3-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(5)
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Embedding(64, 16),
    lt.nn.Linear(16, 64, True)
])
model.to(device)

scheduler = lt.nn.ServingScheduler(64, 4, 4, 64)
scheduler.add_request(lt.nn.Request(1, [1, 5, 9, 2], 4))
scheduler.add_request(lt.nn.Request(2, [3, 7], 4))

completed = []
for _ in range(10):
    completed.extend(scheduler.step(model))

for req in completed:
    print("request", req.id, "đã sinh:", req.generated_tokens)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: cả hai request hoàn thành
với 4 token được sinh mỗi request. Mô hình ở đây là LM embedding +
linear nhỏ trong T3.8 — scheduler đưa cho nó tensor id `[batch,
tokens]` và đọc logits `[batch, tokens, vocab]`, nên mọi mô hình có
cùng hợp đồng đó đều hoạt động.

## T3.4 — Prefix caching

Các system prompt lặp lại tái dùng trạng thái KV của chúng thay vì tính
lại. `put` nhận danh sách token và một danh sách các tensor KV — một
lời gọi, hai tham số, không phải ba.

<!-- ex: id=T3.4-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

cache = lt.nn.PrefixCache.get()
cache.clear()

system_tokens = [1, 105, 200, 350]
k = lt.Tensor.from_vector([0.1] * 512, [1, 4, 128], device, False)
v = lt.Tensor.from_vector([0.2] * 512, [1, 4, 128], device, False)

cache.put(system_tokens, [k, v])
cached = cache.get_cache(system_tokens)
print("số tensor đã cache:", len(cached))
print("shape đã cache:", [t.shape for t in cached])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: cùng danh sách token trả về
2 tensor KV đã lưu với shape gốc của chúng.

## T3.5 — Speculative decoding

Một mô hình draft nhỏ đề xuất `lookahead` token; mô hình target kiểm
chứng chúng song song và chấp nhận tiền tố khớp nhau. Kết quả là prompt
cộng các token mới.

<!-- ex: id=T3.5-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(5)
device = lt.auto_device()

draft = lt.nn.Sequential([
    lt.nn.Embedding(64, 16),
    lt.nn.Linear(16, 64, True)
])
draft.to(device)

target = lt.nn.Sequential([
    lt.nn.Embedding(64, 16),
    lt.nn.Linear(16, 64, True)
])
target.to(device)

engine = lt.nn.SpeculativeEngine(draft, target)
tokens = engine.generate(prompt=[1, 5, 9], max_new_tokens=4, lookahead=2)
print("prompt + token đã sinh:", tokens)
print("số token mới:", len(tokens) - 3)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: 4 token mới được thêm vào
prompt 3 token. Cả hai mô hình đều phải ánh xạ id `[batch, tokens]`
thành logits `[batch, tokens, vocab]` — cùng hợp đồng như scheduler.

## T3.6 — Medusa multi-head decoding

Các Medusa head dự đoán thêm token mỗi forward pass mà không cần mô
hình draft riêng. Mỗi head ánh xạ logits trở lại logits, nên một head
là `MedusaHead(vocab, vocab)` — chiều rộng đầu vào của nó là kích thước
vocab, không phải kích thước hidden.

<!-- ex: id=T3.6-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(5)
device = lt.auto_device()

base = lt.nn.Sequential([
    lt.nn.Embedding(64, 16),
    lt.nn.Linear(16, 64, True)
])
base.to(device)

head = lt.nn.MedusaHead(64, 64)
head.to(device)

medusa = lt.nn.MedusaEngine(base, [head])
tokens = medusa.generate(prompt=[10, 20, 30], max_new_tokens=4, lookahead=2)
print("prompt + token đã sinh:", tokens)
print("số token mới:", len(tokens) - 3)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: 4 token mới được thêm vào
prompt 3 token.

## T3.7 — Guided decoding (đầu ra có cấu trúc)

`GuidedDecoder` che các logits để quá trình sinh chỉ có thể tiếp tục
một mẫu cho trước. `apply_mask` hoạt động tại chỗ và không trả về gì —
đọc logits sau lời gọi. Mẫu phải bắt đầu bằng `prefix:` hoặc `regex:`;
một mẫu trần khớp với mọi thứ và không che gì cả.

<!-- ex: id=T3.7-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

vocab = ["{", "}", '"name"', ":", '"John"', '"age"', "25", ","]
decoder = lt.nn.GuidedDecoder(vocab)

logits = lt.Tensor.from_vector([1.0] * len(vocab), [1, len(vocab)], device, False)
decoder.apply_mask(logits, prefix="{", target_pattern='prefix:{"name": "J')
print("vocab:", vocab)
print("logits sau khi che:", logits.to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: chỉ `'"name"'` giữ nguyên
logit (1.0); mọi token khác bị che thành -1e9, vì chỉ `{"name"` tiếp tục
được tiền tố bắt buộc `{"name": "J`.

## T3.8 — Fine-tune LM nhỏ, rồi sinh văn bản

Huấn luyện mô hình embedding + linear trên các chuỗi token ngẫu nhiên
với dự đoán token tiếp theo, rồi sinh văn bản từ nó. Tám epoch trên 32
chuỗi dài 8: loss trung bình giảm từ 4.2863 xuống 2.4816.

<!-- ex: id=T3.8-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import random
import litetorch as lt

lt.manual_seed(11)
random.seed(1)
device = lt.auto_device()

vocab, d, T = 64, 16, 8
seqs = [[random.randrange(vocab) for _ in range(T + 1)] for _ in range(32)]

model = lt.nn.Sequential([
    lt.nn.Embedding(vocab, d),
    lt.nn.Linear(d, vocab, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
loss_fn = lt.nn.CrossEntropyLoss()

for epoch in range(8):
    total_loss = 0.0
    for s in seqs:
        inp = lt.Tensor.from_vector([float(t) for t in s[:-1]], [1, T], device, False)
        tgt = lt.Tensor.from_vector([float(t) for t in s[1:]], [1, T], device, False)
        optimizer.zero_grad()
        logits = model.forward(inp)
        loss = loss_fn.forward(logits.view([T, vocab]), tgt.view([T]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch %d, Loss: %.4f" % (epoch + 1, total_loss / len(seqs)))

config = lt.nn.GenerateConfig()
config.max_new_tokens = 8
tokens = lt.nn.generate(model, [1, 5, 9], config)
print("đã sinh:", tokens)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: loss 4.2863 → 2.4816, quá
trình sinh trả về 8 token mới. Không có khẳng định parity từng điểm với
torch: dữ liệu huấn luyện là ngẫu nhiên và khởi tạo khác nhau giữa các
framework. Bản thân hàm loss là `CrossEntropyLoss` đã kiểm chứng ở
T2.3.
