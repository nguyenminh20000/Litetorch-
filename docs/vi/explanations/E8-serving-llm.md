# Giải thích E8 — Phục Vụ / LLM

Phục vụ một mô hình ngôn ngữ là bài toán khác với huấn luyện nó.
Huấn luyện bị giới hạn bởi throughput: đóng batch lớn nhất vừa bộ
nhớ và giữ accelerator bận. Phục vụ bị giới hạn bởi độ trễ và bộ nhớ:
request đến từng cái một, mỗi request muốn token đầu tiên nhanh, và
mỗi request đang chạy giữ một KV cache tăng lên theo từng token nó
sinh ra. Ngăn xếp phục vụ tồn tại để quản lý đúng sự căng thẳng đó —
nhận request, chia sẻ những gì chia sẻ được, sinh nhiều token mỗi
forward pass khi có thể, và giải phóng bộ nhớ ngay khi một request
xong.

Trang này giải thích các cơ chế LiteTorch phục vụ: paged attention,
prefix caching, serving scheduler, speculative decoding, Medusa, và
phục vụ GPT/GPT2B — cùng vòng đời KV cache nối chúng lại. Mọi ví dụ
chạy trên CPU với LiteTorch 88a4ba1, PyTorch 2.14.1+cpu, trên một mô
hình nhỏ vocab 64 nơi mọi cơ chế đều nhìn thấy được. Số throughput
thuộc về GPU và được đánh dấu REQUIRES-GPU, không khẳng định ở đây.

Điều kiện tiên quyết: Hướng dẫn T1–T3 (đạo hàm tự động, attention,
hướng dẫn phục vụ). Các mục scheduler, speculative decoding, và
Medusa mở rộng T3.3–T3.6 với phần lý do; các lời gọi chạy được là
cùng hợp đồng.

> Các khác biệt với PyTorch được chạm tới trong trang này: D-20
> (không có `F.scaled_dot_product_attention` — dùng
> `Ops.flash_attention(q, k, v)`; không có đối số `scale` hay
> `dropout_p`, tự scale Q; head dim phải thỏa 1 <= D <= 256), D-03
> (`.to()` nhận một `Device`), D-02 (fallback CPU lặng lẽ — kiểm tra
> thiết bị sau `.to()`).

## E8.1 — Vòng đời KV cache

Sinh tự hồi quy không tính lại gì nếu tránh được. Với mỗi token được
sinh, mô hình cần key và value của mọi token trước đó; tính lại chúng
là bậc hai theo độ dài chuỗi, nên chúng được cache. Vòng đời cache
của một request là:

```
cấp phát block            prefill                 decode (mỗi token)        giải phóng
     |                        |                          |                     |
     v                        v                          v                     v
 [free list]  --pop-->  [token prompt: K,V]  --append-->  [K,V + token mới]  --push--> [free list]
     ^                        |                          |                     |
     |                   một forward trên           một forward cho        |
     |                   toàn bộ prompt           mỗi token được sinh    |
     |                   (song song)              (tuần tự)              |
```

Hai pha, hai chi phí. **Prefill** xử lý toàn bộ prompt trong một
forward pass song song — bị giới hạn bởi tính toán, nhanh trên mỗi
token. **Decode** sinh một token mỗi forward pass, đọc toàn bộ KV
cache từ bộ nhớ mỗi lần — bị giới hạn bởi băng thông bộ nhớ, chậm
trên mỗi token, và là lý do throughput decode được đo bằng tokens/s
chứ không phải FLOPs. Mọi thứ trong trang này là tối ưu một trong hai
pha này: paged attention và prefix caching làm cache rẻ hơn để giữ,
scheduler đóng gói các bước decode lại với nhau, speculative decoding
và Medusa làm mỗi bước decode sinh ra nhiều hơn một token.

Cache cũng là nút thắt bộ nhớ. Một request với context 4k token trong
mô hình 28 lớp giữ 28 lớp × 2 (K và V) × 4k vị trí vector; nhân với
batch size và cache — chứ không phải trọng số — quyết định bao nhiêu
request chạy cùng lúc được. Khi cache hết, scheduler phải xếp hàng,
preempt, hoặc swap — nó không thể chỉ cấp phát thêm.

## E8.2 — Paged attention: block thay cho một buffer lớn

Cache ngây thơ cấp phát một buffer liên tục `[max_len, ...]` cho mỗi
request. Các request có độ dài khác nhau, nên phần lớn buffer trống,
và một request vượt quá buffer của nó phải được sao chép sang buffer
lớn hơn. Paged attention (ý tưởng vLLM) chia cache thành các block
kích thước cố định — ví dụ 16 token mỗi block — và một block table
ánh xạ vị trí logic của mỗi request tới block vật lý:

```
request A, 40 token, block_size 16:
  logic:    [0..15] [16..31] [32..39]
  vật lý:   block 7, block 3, block 12 (8 slot đầu được dùng)
request B, 20 token:
  logic:    [0..15] [16..19]
  vật lý:   block 9, block 5
block trống: [0, 1, 2, 4, 6, 8, 10, 11, ...]   <- pool chung
```

Phân mảnh biến mất: một request được giải phóng trả block của nó về
pool, và request tiếp theo tái dùng chúng bất kể độ dài. Hai request
chia sẻ prefix prompt thậm chí có thể chia sẻ block vật lý cho prefix
đó — đó là prefix cache (E8.3) xây trên cùng ý tưởng.

Lời gọi LiteTorch là `Ops.paged_attention(q, k_cache, v_cache,
block_tables, context_lens, block_size)` — sáu đối số positional,
không kwargs. `q` có rank 3 `[num_seqs, num_heads, head_dim]`;
`k_cache` / `v_cache` có rank 4 `[num_blocks, num_kv_heads,
block_size, head_dim]`; `block_tables` và `context_lens` phải là
INT32; `block_size` là một int thuần. Head dim tối đa 128. Không có
bản tương đương torch; mirror torch dưới đây gom các vị trí đã cache
bằng tay và chạy attention thủ công trên chúng.

<!-- ex: id=E8-1 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import math
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
q = lt.Tensor.from_vector([0.1] * 8, [1, 1, 8], cpu, False)
k_cache = lt.Tensor.from_vector([0.2] * (2 * 1 * 4 * 8), [2, 1, 4, 8], cpu, False)
v_cache = lt.Tensor.from_vector([0.3] * (2 * 1 * 4 * 8), [2, 1, 4, 8], cpu, False)
block_tables = lt.Tensor.from_vector([0.0, 1.0], [1, 2], cpu, False, lt.DataType.INT32)
context_lens = lt.Tensor.from_vector([6.0], [1], cpu, False, lt.DataType.INT32)
out = lt.Ops.paged_attention(q, k_cache, v_cache, block_tables, context_lens, 4)
print("shape của paged attention:", out.shape)
print("các giá trị đầu:", out.to_vector()[:4])
qt = torch.full((1, 1, 8), 0.1)
kt = torch.full((6, 8), 0.2)
vt = torch.full((6, 8), 0.3)
ref = torch.softmax(qt.squeeze(0) @ kt.T / math.sqrt(8), dim=-1) @ vt
print("các giá trị đầu của tham chiếu torch:", ref.flatten().tolist()[:4])
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("Sai lệch lớn nhất so với attention tính tay torch:", diff)
assert diff <= 1e-4
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity
1e-4: attention trên 6 vị trí đã cache (4 trong block 0, 2 trong
block 1) khớp attention tính tay; với key đều nhau đầu ra là giá trị
0.3 ở mọi nơi.

## E8.3 — Prefix caching: không bao giờ tính lại prompt chung

Phần lớn lưu lượng phục vụ lặp lại cùng prefix: một system prompt,
một template few-shot, một tài liệu được hỏi đi hỏi lại. Prefill cho
prefix đó là lãng phí thuần túy từ lần thứ hai. Prefix cache lưu các
tensor KV cho một danh sách token và trả chúng về khi khớp đúng
prefix:

```
request 1: [SYS, "summarize:", DOC]  -> prefill, cache K,V theo danh sách token
request 2: [SYS, "summarize:", DOC]  -> trúng cache, bỏ qua prefill
request 3: [SYS, "translate:", DOC]  -> trượt trên toàn danh sách (prefix khác)
```

`PrefixCache.get()` trả về cache singleton toàn process.
`put(tokens, [k, v])` nhận danh sách token và một **danh sách** các
tensor KV — một lời gọi, hai đối số, không phải ba.
`get_cache(tokens)` trả về danh sách đã lưu, hoặc một danh sách rỗng
khi trượt. `clear()` làm rỗng cache. Khớp dựa trên đúng danh sách
token; một prompt khác một token là trượt, đó là lý do hệ thống phục
vụ trong production băm prefix theo từng block (E8.2) thay vì theo
toàn bộ prompt.

<!-- ex: id=E8-2 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
cache = lt.nn.PrefixCache.get()
cache.clear()
system_tokens = [2, 77, 300]
print("trượt, số tensor đã cache:", len(cache.get_cache(system_tokens)))
k = lt.Tensor.from_vector([0.1] * 192, [1, 3, 64], cpu, False)
v = lt.Tensor.from_vector([0.2] * 192, [1, 3, 64], cpu, False)
cache.put(system_tokens, [k, v])
cached = cache.get_cache(system_tokens)
print("trúng, số tensor đã cache:", len(cached))
print("shape đã cache:", [t.shape for t in cached])
cache.clear()
print("sau clear, số tensor đã cache:", len(cache.get_cache(system_tokens)))
assert len(cached) == 2
assert list(cached[0].shape) == [1, 3, 64]
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: trượt trả về 0 tensor,
trúng trả về 2 tensor KV đã lưu với shape gốc của chúng, và `clear()`
làm rỗng cache. Cache là singleton — `get()` trả về cùng cache mỗi
lần gọi, nên một test quên `clear()` sẽ thấy các entry của test
trước.

## E8.4 — Serving scheduler: continuous batching

Không có scheduler, phục vụ là từng request một: prefill, decode tới
khi xong, request tiếp theo — và GPU nhàn rỗi trong mỗi bước decode
một token. Continuous batching đóng gói các bước decode của mọi
request đang chạy vào một forward pass mỗi tick, và nhận request mới
ngay khi một slot trống, thay vì chờ toàn bộ batch xong (static
batching).

`ServingScheduler(num_blocks, block_size, max_num_seqs,
max_num_batched_tokens)` sở hữu pool block từ E8.2.
`add_request(Request(id, prompt_tokens, max_new_tokens))` xếp một
request vào hàng. `step(model)` chạy một tick — prefill cho request
mới được nhận, một token decode cho mỗi request đang chạy — và trả về
các request xong trong tick này. Hợp đồng của mô hình là tensor id
`[batch, tokens]` vào, logits `[batch, tokens, vocab]` ra.

```
tick 1: nhận A, B       -> prefill A, B
tick 2: decode A1, B1   -> A xong? chưa
tick 3: decode A2, B2; nhận C (slot trống) -> prefill C
...
```

<!-- ex: id=E8-3 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(5)
model = lt.nn.Sequential([lt.nn.Embedding(64, 16), lt.nn.Linear(16, 64, True)])
model.to(cpu)
scheduler = lt.nn.ServingScheduler(64, 4, 4, 64)
scheduler.add_request(lt.nn.Request(1, [2, 4, 6], 3))
scheduler.add_request(lt.nn.Request(2, [8], 2))
completed = []
for _ in range(10):
    completed.extend(scheduler.step(model))
for req in completed:
    print("request", req.id, "đã sinh:", req.generated_tokens)
by_id = {req.id: req for req in completed}
assert len(completed) == 2
assert len(by_id[1].generated_tokens) == 3
assert len(by_id[2].generated_tokens) == 2
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: cả hai request xong với
đúng số token yêu cầu (3 và 2). Mọi mô hình có hợp đồng `[batch,
tokens]` → `[batch, tokens, vocab]` đều dùng được — scheduler không
bao giờ nhìn vào trong mô hình.

## E8.5 — Speculative decoding: draft nhanh, kiểm chứng song song

Decode là tuần tự: một token mỗi forward pass, và mỗi pass bị giới
hạn bởi bộ nhớ. Speculative decoding phá vỡ tính tuần tự bằng một mô
hình draft nhỏ. Mô hình draft đề xuất `lookahead` token với chi phí
rẻ; mô hình target kiểm chứng tất cả chúng trong **một** forward pass
song song (đường prefill là song song, nên kiểm chứng 4 token tốn
gần bằng kiểm chứng 1 token); prefix khớp được chấp nhận và điểm
không khớp đầu tiên được target sửa:

```
draft đề xuất:  [t1, t2, t3, t4]
target kiểm chứng: [t1, t2, X,  .]   -> chấp nhận t1, t2; target phát t3'
ròng mỗi pass target: 3 token thay vì 1
```

`SpeculativeEngine(draft, target).generate(prompt, max_new_tokens,
lookahead)` trả về prompt cộng các token mới. Cả hai mô hình phải
thỏa cùng hợp đồng `[batch, tokens]` → `[batch, tokens, vocab]` như
scheduler. Tốc độ tăng phụ thuộc vào tỷ lệ chấp nhận của draft: một
draft đồng ý với target 80% thời gian gần như nhân đôi throughput;
một draft không bao giờ đồng ý tốn một pass draft mỗi token và làm
mọi thứ chậm lại. Chất lượng draft, không phải kích thước draft, là
núm chỉnh.

<!-- ex: id=E8-4 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(5)
draft = lt.nn.Sequential([lt.nn.Embedding(64, 16), lt.nn.Linear(16, 64, True)])
draft.to(cpu)
target = lt.nn.Sequential([lt.nn.Embedding(64, 16), lt.nn.Linear(16, 64, True)])
target.to(cpu)
engine = lt.nn.SpeculativeEngine(draft, target)
tokens = engine.generate(prompt=[1, 5, 9], max_new_tokens=4, lookahead=2)
print("prompt + token được sinh:", tokens)
print("số token mới:", len(tokens) - 3)
assert len(tokens) - 3 == 4
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: 4 token mới được thêm vào
prompt 3 token. Mô hình draft và target ở đây cố tình cùng shape —
ví dụ cho thấy hợp đồng, không phải tốc độ tăng; một triển khai thật
dùng draft nhỏ hơn nhiều.

## E8.6 — Medusa: head phụ thay cho mô hình draft

Medusa loại bỏ hoàn toàn mô hình draft. Các head dự đoán phụ ngồi
trên hidden state cuối (hoặc logits) của mô hình gốc và mỗi head dự
đoán một token xa hơn về phía trước: head 1 dự đoán token `t+1`,
head 2 dự đoán `t+2`, và cứ thế. Một forward pass cho ra nhiều token
ứng viên, rồi chúng được kiểm chứng với mô hình gốc giống cách
speculative decoding kiểm chứng token draft — thường với một cây ứng
viên thay vì một chuỗi duy nhất, giữ lại các tổ hợp có khả năng nhất.

Đánh đổi: không có mô hình thứ hai để load, lập lịch, hay giữ đồng
bộ — nhưng các head phải được huấn luyện (chúng dự đoán xa hơn từ
cùng representation, khó hơn hẳn dự đoán token tiếp theo), và tỷ lệ
chấp nhận của chúng thường thấp hơn một mô hình draft tốt. Trong
LiteTorch một head là `MedusaHead(vocab, vocab)` — nó ánh xạ logits
trở lại logits, nên chiều rộng đầu vào của nó là kích thước vocab,
không phải hidden size.

<!-- ex: id=E8-5 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(5)
base = lt.nn.Sequential([lt.nn.Embedding(64, 16), lt.nn.Linear(16, 64, True)])
base.to(cpu)
head = lt.nn.MedusaHead(64, 64)
head.to(cpu)
medusa = lt.nn.MedusaEngine(base, [head])
tokens = medusa.generate(prompt=[10, 20, 30], max_new_tokens=4, lookahead=2)
print("prompt + token được sinh:", tokens)
print("số token mới:", len(tokens) - 3)
assert len(tokens) - 3 == 4
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: 4 token mới được thêm vào
prompt 3 token. `MedusaEngine(base, [head, ...])` nhận một danh sách
head — thêm một head cho mỗi vị trí phụ bạn muốn dự đoán.

## E8.7 — Attention decode và phục vụ GPT/GPT2B

Hai mảnh nữa hoàn thiện bức tranh phục vụ. **Flash decoding**
(`Ops.flash_decoding(q, k, v, num_splits)`) là bản song sinh pha
decode của flash attention: cùng công thức softmax(QK^T/sqrt(d))V,
nhưng chiều KV được chia `num_splits` phần rồi rút gọn, nên một
context dài decode song song thay vì rút gọn tuần tự. `num_splits`
là núm chỉnh hiệu năng, không phải thay đổi toán học — kết quả khớp
`F.scaled_dot_product_attention` (D-20: LiteTorch không có SDPA;
mirror torch dưới đây là dạng tính tay).

**GPT2B** là mô hình phục vụ kiểu GPT của LiteTorch:
`GPT2B(vocab_size, d_model, n_layers, n_q_heads, n_kv_heads, d_ff,
rope_theta)` — grouped-query attention (`n_kv_heads < n_q_heads`
thu nhỏ KV cache mỗi lớp, đúng nút thắt E8.1), vị trí RoPE, và
`stage_forward(input, stage)` cho phục vụ song song pipeline qua
nhiều thiết bị. `forward` nhận id `[batch, tokens]` và trả logits
`[batch, tokens, vocab]` — cùng hợp đồng scheduler và
`lt.nn.generate` kỳ vọng. Mặc định (`vocab_size=50257, d_model=2560,
n_layers=28`) là mô hình 2B thật; ví dụ dưới đây xây một bản nhỏ để
chạy trên CPU.

<!-- ex: id=E8-6 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import math
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
q = lt.Tensor.from_vector([0.1 * (i % 7 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
k = lt.Tensor.from_vector([0.05 * (i % 5 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
v = lt.Tensor.from_vector([0.3 * (i % 6 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
out = lt.Ops.flash_decoding(q, k, v, 2)
print("shape của flash decoding:", out.shape)
print("các giá trị đầu:", out.to_vector()[:4])
qt = torch.tensor([0.1 * (i % 7 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
kt = torch.tensor([0.05 * (i % 5 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
vt = torch.tensor([0.3 * (i % 6 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
ref = torch.softmax(qt @ kt.transpose(-2, -1) / math.sqrt(8), dim=-1) @ vt
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("Sai lệch lớn nhất so với attention tính tay torch:", diff)
assert diff <= 1e-4
lt.manual_seed(7)
gpt = lt.nn.GPT2B(64, 32, 2, 4, 2, 64, 10000.0)
gpt.to(cpu)
ids = lt.Tensor.from_vector([1.0, 5.0, 9.0, 2.0], [1, 4], cpu, False)
logits = gpt.forward(ids)
print("shape logits GPT2B:", logits.shape)
config = lt.nn.GenerateConfig()
config.max_new_tokens = 4
print("Token GPT2B đã sinh:", lt.nn.generate(gpt, [1, 5, 9], config))
assert list(logits.shape) == [1, 4, 64]
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: flash
decoding khớp attention tính tay ở parity 1e-4 (`num_splits=2`
không thay đổi gì về số học), và GPT2B nhỏ trả về logits `[1, 4,
64]` và sinh 4 token mới. `generate` chỉ trả về token mới, không trả
prompt.

## E8.8 — Những gì KHÔNG được kiểm chứng ở đây

> REQUIRES-GPU: tokens/s và time-to-first-token cho scheduler,
> speculative decoding, và Medusa; paged attention ở quy mô lớn (hàng
> nghìn block, context dài); tỷ lệ trúng prefix cache dưới lưu lượng
> thật; phục vụ 2-GPU. Tất cả đều là phép đo GPU Kaggle (B5). Không
> khẳng định số throughput nào trên trang này, và không có timing CPU
> nào ở đây dự đoán hiệu năng phục vụ GPU — decode bị giới hạn bởi
> băng thông bộ nhớ, nên timing CPU sẽ gây hiểu lầm theo cả hai hướng.
