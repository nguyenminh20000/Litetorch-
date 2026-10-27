# How-To H4 — Phục vụ với Paged Attention và Continuous Batching

Stack phục vụ, từ dưới lên: KV cache dạng paged, một scheduler
continuous-batching, prefix caching, rồi speculative decoding. Mọi ví
dụ dưới đây chạy trên CPU với LiteTorch 88a4ba1. Các mô hình ở đây là
các stack Embedding + Linear nhỏ — engine phục vụ đưa token ID vào mô
hình, nên mô hình phải bắt đầu bằng một `Embedding`, không phải một
`Linear` trần.

Điều kiện tiên quyết: LiteTorch đã được cài và import được.

> Các khác biệt với PyTorch liên quan ở đây: D-20 (không có
> `F.scaled_dot_product_attention` — dùng `Ops.flash_attention(q, k, v)`),
> D-03 (`.to()` nhận một `Device`, không có dạng chuỗi).

## H4.1 — Paged attention trên block KV cache

`paged_attention(q, k_cache, v_cache, block_tables, context_lens,
block_size)` nhận sáu đối số positional — không có kwargs. Shape rất
nghiêm ngặt: `q` có rank 3 `[num_seqs, num_heads, head_dim]`, `k_cache`
và `v_cache` có rank 4
`[num_blocks, num_kv_heads, block_size, head_dim]`, `block_tables` có
rank 2 `[num_seqs, max_num_blocks_per_seq]` và `context_lens` có rank
1 `[num_seqs]` — cả hai đều **INT32**. `head_dim` tối đa 128.

<!-- ex: id=H4.1-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

q = lt.Tensor.from_vector([0.1] * 32, [1, 1, 32], device, False)
k_cache = lt.Tensor.from_vector([0.2] * (2 * 1 * 16 * 32), [2, 1, 16, 32], device, False)
v_cache = lt.Tensor.from_vector([0.3] * (2 * 1 * 16 * 32), [2, 1, 16, 32], device, False)
block_tables = lt.Tensor.from_vector([0.0, 1.0], [1, 2], device, False, lt.DataType.INT32)
context_lens = lt.Tensor.from_vector([32.0], [1], device, False, lt.DataType.INT32)

out = lt.Ops.paged_attention(q, k_cache, v_cache, block_tables, context_lens, 16)
print("PagedAttention output shape:", list(out.shape))
print("PagedAttention output values:", out.to_vector()[:4])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, parity với torch 1e-4 theo
attention thủ công (`softmax(q @ k.T / sqrt(head_dim)) @ v` trên 32 vị
trí đã cache được gom qua block table). Kết quả mong đợi: shape
`[1, 1, 32]`, mọi giá trị đều `0.3` (in ra `0.3000001…` — làm tròn
FP32 của 0.3). Hai lỗi kinh điển, cả hai đều loud: `k_cache`/`v_cache`
rank 3 raise `paged_attention: bad input ranks`, và bảng FP32 raise
`block_tables and context_lens must be INT32`.

## H4.2 — Continuous batching với serving scheduler

`ServingScheduler(num_blocks, block_size, max_num_seqs,
max_num_batched_tokens)` nhận request, cấp phát KV block, và chạy
prefill + decode. `step(model)` nhận mô hình và trả về danh sách các
request đã xong ở bước này — gọi nó trong vòng lặp cho tới khi hàng
đợi running và pending đều rỗng. Mô hình phải ánh xạ token ID sang
logits: `Embedding(vocab, d)` rồi `Linear(d, vocab)`.

<!-- ex: id=H4.2-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

lt.manual_seed(0)
model = lt.nn.Sequential([
    lt.nn.Embedding(1000, 32),
    lt.nn.Linear(32, 1000, True),
])
model.to(device)

scheduler = lt.nn.ServingScheduler(
    num_blocks=16,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=128
)

scheduler.add_request(lt.nn.Request(1, [101, 205, 200, 102], 8))
scheduler.add_request(lt.nn.Request(2, [101, 759, 102], 8))

done = []
block_table_shape = None
context_lens = None
for step in range(20):
    finished = scheduler.step(model)
    done.extend(finished)
    if step == 0:
        block_table_shape = list(scheduler.get_block_table_tensor().shape)
        context_lens = scheduler.get_context_lens_tensor().to_vector()
    if not scheduler.running_queue and not scheduler.pending_queue:
        break

print("steps:", step + 1, "finished requests:", len(done))
for req in done:
    print("request", req.id, "generated tokens:", req.generated_tokens)
print("block table shape:", block_table_shape)
print("context lens:", context_lens)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: cả hai
request đều xong (mỗi cái 8 token đã sinh) trong 20 bước; block table
có một hàng cho mỗi request đã nhận. Hai request sinh ra cùng dãy token
ở đây — token tiếp theo của mô hình đồ chơi chỉ phụ thuộc token cuối
của prompt, và token đó là 102 cho cả hai. Một mô hình `Linear` trần
không dùng được ở đây: scheduler đưa cho mô hình một tensor `[1, chunk]`
chứa token ID và đọc một output `[1, seq, vocab]`, nên lớp đầu tiên
phải là một `Embedding`.

## H4.3 — Prefix caching cho system prompt lặp lại

`PrefixCache.get()` trả về singleton toàn process. `put(prefix,
[k, v])` lưu cặp KV dưới một prefix token — chú ý đối số thứ hai là
một **list** hai tensor, không phải hai đối số riêng. `get_cache(prefix)`
trả về cặp đã lưu, hoặc một list rỗng khi miss.

<!-- ex: id=H4.3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

cache = lt.nn.PrefixCache.get()
cache.clear()
system_tokens = [1, 105, 200, 350]

k_data = lt.Tensor.from_vector([0.1] * 512, [1, 4, 128], device, False)
v_data = lt.Tensor.from_vector([0.2] * 512, [1, 4, 128], device, False)

cache.put(system_tokens, [k_data, v_data])
cached_pair = cache.get_cache(system_tokens)
print("cache hit tensors:", len(cached_pair))
print("cached K shape:", list(cached_pair[0].shape))
print("cache miss:", cache.get_cache([9, 9, 9]))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: một hit
trả về 2 tensor (`[1, 4, 128]` mỗi cái); một miss trả về một list rỗng.
Cache system prompt một lần cho mỗi process và tái dùng cho mọi request
có chung nó — đó là toàn bộ ý nghĩa; một cache cho mỗi request chẳng
mua được gì.

## H4.4 — Speculative decoding với một draft model

Một draft model nhỏ đề xuất `lookahead` token; target model kiểm chứng
chúng song song và chấp nhận prefix khớp. Cả hai mô hình đều phải ánh
xạ token ID sang logits, nên cả hai đều bắt đầu bằng một `Embedding`.

<!-- ex: id=H4.4-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

lt.manual_seed(0)
draft_model = lt.nn.Sequential([
    lt.nn.Embedding(64, 16),
    lt.nn.Linear(16, 64, False),
])
draft_model.to(device)

target_model = lt.nn.Sequential([
    lt.nn.Embedding(64, 32),
    lt.nn.Linear(32, 64, False),
])
target_model.to(device)

engine = lt.nn.SpeculativeEngine(draft_model, target_model)
generated = engine.generate(prompt=[1, 5, 9], max_new_tokens=5, lookahead=2)
print("prompt + generated:", generated)
print("new tokens:", len(generated) - 3)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: prompt
cộng 5 token mới (tổng 8 token ID). Một draft hoặc target model
`Linear` trần sẽ fail với `Matrix inner dimensions must match`: engine
đưa cho mô hình một tensor `[1, seq]` chứa token ID, và chỉ một lớp
`Embedding` đầu tiên mới biến nó thành hidden state mà `Linear` có
thể tiêu thụ.

## H4.5 — Medusa: head phụ thay cho draft model

`MedusaEngine` gắn các head phụ vào base model, để một mô hình dự đoán
vài token tương lai trong mỗi forward pass. Mỗi head là
`MedusaHead(in_features, vocab_size)` trong đó `in_features` phải bằng
kích thước vocab — head nhân với logits của base model, nên chiều đầu
vào của nó là vocab, không phải hidden size.

<!-- ex: id=H4.5-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

lt.manual_seed(0)
base_model = lt.nn.Sequential([
    lt.nn.Embedding(64, 32),
    lt.nn.Linear(32, 64, False),
])
base_model.to(device)

head1 = lt.nn.MedusaHead(64, 64)
head1.to(device)

medusa = lt.nn.MedusaEngine(base_model, [head1])
tokens = medusa.generate(prompt=[10, 20, 30], max_new_tokens=5, lookahead=2)
print("prompt + generated:", tokens)
print("new tokens:", len(tokens) - 3)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: prompt
cộng 5 token mới (tổng 8 token ID). `MedusaHead(512, 32000)` — shape
trong tài liệu cũ — là sai vì cùng lý do như H4.4: head tiêu thụ logits
có chiều rộng `vocab`, nên `in_features` phải là `vocab` (64 ở đây).
