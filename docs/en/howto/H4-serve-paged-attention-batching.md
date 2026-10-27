# How-To H4 — Serve with Paged Attention and Continuous Batching

The serving stack, bottom-up: paged KV cache, a continuous-batching
scheduler, prefix caching, then speculative decoding. All examples below
run on CPU against LiteTorch 88a4ba1. The models here are tiny
Embedding + Linear stacks — serving engines feed token IDs to the model,
so the model must start with an `Embedding`, not a bare `Linear`.

Prerequisites: LiteTorch installed and importable.

> PyTorch differences touched here: D-20 (no
> `F.scaled_dot_product_attention` — use `Ops.flash_attention(q, k, v)`),
> D-03 (`.to()` takes a `Device`, no string form).

## H4.1 — Paged attention over a block KV cache

`paged_attention(q, k_cache, v_cache, block_tables, context_lens,
block_size)` takes six positional arguments — no kwargs. Shapes are
strict: `q` is rank 3 `[num_seqs, num_heads, head_dim]`, `k_cache` and
`v_cache` are rank 4 `[num_blocks, num_kv_heads, block_size, head_dim]`,
`block_tables` is rank 2 `[num_seqs, max_num_blocks_per_seq]` and
`context_lens` is rank 1 `[num_seqs]` — both **INT32**. `head_dim`
must be at most 128.

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

Verified on CPU, LiteTorch 88a4ba1, torch parity 1e-4 against manual
attention (`softmax(q @ k.T / sqrt(head_dim)) @ v` over the 32 cached
positions gathered through the block table). Expected output: shape
`[1, 1, 32]`, all values `0.3` (printed as `0.3000001…` — FP32
rounding of 0.3).
Two classic mistakes, both loud: rank-3 `k_cache`/`v_cache` raises
`paged_attention: bad input ranks`, and FP32 tables raise
`block_tables and context_lens must be INT32`.

## H4.2 — Continuous batching with the serving scheduler

`ServingScheduler(num_blocks, block_size, max_num_seqs,
max_num_batched_tokens)` admits requests, allocates KV blocks, and runs
prefill + decode. `step(model)` takes the model and returns the list of
requests that finished on this step — call it in a loop until the
running and pending queues are empty. The model must map token IDs to
logits: `Embedding(vocab, d)` followed by `Linear(d, vocab)`.

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

Verified on CPU, LiteTorch 88a4ba1. Expected output: both requests
finish (8 generated tokens each) within 20 steps; the block table has
one row per admitted request. The two requests generate identical token
sequences here — the toy model's next token depends only on the last
prompt token, which is 102 for both. A bare `Linear` model does not work here:
the scheduler feeds the model a `[1, chunk]` tensor of token IDs and
reads a `[1, seq, vocab]` output, so the first layer must be an
`Embedding`.

## H4.3 — Prefix caching for repeated system prompts

`PrefixCache.get()` returns the process-wide singleton. `put(prefix,
[k, v])` stores the KV pair under a token prefix — note the second
argument is a **list** of two tensors, not two separate arguments.
`get_cache(prefix)` returns the stored pair, or an empty list on a miss.

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

Verified on CPU, LiteTorch 88a4ba1. Expected output: a hit returns 2
tensors (`[1, 4, 128]` each); a miss returns an empty list. Cache the
system prompt once per process and reuse it for every request that
shares it — that is the whole point; a per-request cache buys nothing.

## H4.4 — Speculative decoding with a draft model

A small draft model proposes `lookahead` tokens; the target model
verifies them in parallel and accepts the matching prefix. Both models
must map token IDs to logits, so both start with an `Embedding`.

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

Verified on CPU, LiteTorch 88a4ba1. Expected output: the prompt plus 5
new tokens (8 token IDs total). A bare-`Linear` draft or target model
fails with `Matrix inner dimensions must match`: the engine feeds the
model a `[1, seq]` tensor of token IDs, and only an `Embedding` first
layer turns that into a hidden state the `Linear` can consume.

## H4.5 — Medusa: extra heads instead of a draft model

`MedusaEngine` appends auxiliary heads to the base model, so one model
predicts several future tokens per forward pass. Each head is
`MedusaHead(in_features, vocab_size)` where `in_features` must equal
the vocabulary size — the head multiplies the base model's logits, so
its input dimension is the vocab, not the hidden size.

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

Verified on CPU, LiteTorch 88a4ba1. Expected output: the prompt plus 5
new tokens (8 token IDs total). `MedusaHead(512, 32000)` — the shape in
older material — is wrong for the same reason as H4.4: the head consumes
logits of width `vocab`, so `in_features` must be `vocab` (64 here).
