# Explanation E8 — Serving / LLM

Serving a language model is a different problem from training one.
Training is throughput-bound: pack the biggest batch that fits and
keep the accelerator busy. Serving is latency-bound and memory-bound:
requests arrive one by one, each one wants its first token fast, and
every running request holds a KV cache that grows with every token it
generates. The serving stack exists to manage exactly that tension —
admit requests, share what can be shared, generate several tokens per
forward pass where possible, and free memory the moment a request
finishes.

This page explains the mechanisms LiteTorch serves with: paged
attention, prefix caching, the serving scheduler, speculative
decoding, Medusa, and GPT/GPT2B serving — and the KV-cache lifecycle
that ties them together. All examples run on CPU against LiteTorch
88a4ba1, PyTorch 2.14.1+cpu, on a tiny vocab-64 model where every
mechanism is visible. Throughput numbers belong on GPU and are marked
REQUIRES-GPU, not claimed here.

Prerequisites: Tutorials T1–T3 (autograd, attention, the serving
tutorial). The scheduler, speculative decoding, and Medusa sections
extend T3.3–T3.6 with the why; the runnable calls are the same
contracts.

> PyTorch differences touched in this page: D-20 (no
> `F.scaled_dot_product_attention` — use `Ops.flash_attention(q, k,
> v)`; no `scale` or `dropout_p` argument, scale Q yourself; head dim
> must satisfy 1 <= D <= 256), D-03 (`.to()` takes a `Device`),
> D-02 (silent CPU fallback — check the device after `.to()`).

## E8.1 — The KV-cache lifecycle

Autoregressive generation recomputes nothing if it can help it. For
each generated token the model needs the keys and values of all
previous tokens; recomputing them is quadratic in sequence length,
so they are cached. The lifecycle of one request's cache is:

```
allocate blocks            prefill                 decode (per token)        free
     |                        |                          |                     |
     v                        v                          v                     v
 [free list]  --pop-->  [prompt tokens: K,V]  --append-->  [K,V + new token]  --push--> [free list]
     ^                        |                          |                     |
     |                   one forward over           one forward per        |
     |                   the whole prompt           generated token        |
     |                   (parallel)                 (sequential)           |
```

Two phases, two costs. **Prefill** processes the whole prompt in one
parallel forward pass — compute-bound, fast per token. **Decode**
generates one token per forward pass, reading the entire KV cache
from memory each time — memory-bandwidth-bound, slow per token, and
the reason decode throughput is measured in tokens/s rather than
FLOPs. Everything in this page is an optimization of one of these
two phases: paged attention and prefix caching make the cache cheaper
to hold, the scheduler packs decodes together, speculative decoding
and Medusa make each decode step produce more than one token.

The cache is also the memory bottleneck. A request with a 4k-token
context in a 28-layer model holds 28 layers × 2 (K and V) × 4k
positions of vectors; multiply by batch size and the cache, not the
weights, decides how many requests can run at once. When the cache
runs out, the scheduler must queue, preempt, or swap — it cannot
simply allocate more.

## E8.2 — Paged attention: blocks instead of one big buffer

A naive cache allocates one contiguous `[max_len, ...]` buffer per
request. Requests have different lengths, so most buffers are mostly
empty, and a request that outgrows its buffer must be copied to a
bigger one. Paged attention (the vLLM idea) splits the cache into
fixed-size blocks — say 16 tokens each — and a block table maps each
request's logical positions to physical blocks:

```
request A, 40 tokens, block_size 16:
  logical:  [0..15] [16..31] [32..39]
  physical: block 7, block 3, block 12 (first 8 slots used)
request B, 20 tokens:
  logical:  [0..15] [16..19]
  physical: block 9, block 5
free blocks: [0, 1, 2, 4, 6, 8, 10, 11, ...]   <- shared pool
```

Fragmentation disappears: a freed request returns its blocks to the
pool, and the next request reuses them regardless of length. Two
requests sharing a prompt prefix can even share the physical blocks
for that prefix — that is the prefix cache (E8.3) built on the same
idea.

The LiteTorch call is `Ops.paged_attention(q, k_cache, v_cache,
block_tables, context_lens, block_size)` — six positional arguments,
no kwargs. `q` is rank 3 `[num_seqs, num_heads, head_dim]`;
`k_cache` / `v_cache` are rank 4 `[num_blocks, num_kv_heads,
block_size, head_dim]`; `block_tables` and `context_lens` must be
INT32; `block_size` is a plain int. Head dim must be at most 128.
There is no torch equivalent; the torch mirror below gathers the
cached positions by hand and runs manual attention over them.

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
print("paged attention shape:", out.shape)
print("first values:", out.to_vector()[:4])
qt = torch.full((1, 1, 8), 0.1)
kt = torch.full((6, 8), 0.2)
vt = torch.full((6, 8), 0.3)
ref = torch.softmax(qt.squeeze(0) @ kt.T / math.sqrt(8), dim=-1) @ vt
print("torch reference first values:", ref.flatten().tolist()[:4])
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("max diff vs torch manual attention:", diff)
assert diff <= 1e-4
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4:
attention over the 6 cached positions (4 in block 0, 2 in block 1)
matches manual attention; with uniform keys the output is the value
0.3 everywhere.

## E8.3 — Prefix caching: never recompute a shared prompt

Most serving traffic repeats the same prefix: a system prompt, a
few-shot template, a document being asked about repeatedly. The
prefill for that prefix is pure waste the second time. The prefix
cache stores the KV tensors for a token list and returns them on an
exact-prefix hit:

```
request 1: [SYS, "summarize:", DOC]  -> prefill, cache K,V under the token list
request 2: [SYS, "summarize:", DOC]  -> cache hit, prefill skipped
request 3: [SYS, "translate:", DOC]  -> miss on the full list (prefix differs)
```

`PrefixCache.get()` returns the process-wide singleton cache.
`put(tokens, [k, v])` takes the token list and a **list** of KV
tensors — one call, two arguments, not three. `get_cache(tokens)`
returns the stored list, or an empty list on a miss. `clear()` empties
the cache. Matching is on the exact token list; a prompt that differs
in one token is a miss, which is why serving systems hash prefixes
per block (E8.2) rather than per whole prompt in production.

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
print("miss, cached tensors:", len(cache.get_cache(system_tokens)))
k = lt.Tensor.from_vector([0.1] * 192, [1, 3, 64], cpu, False)
v = lt.Tensor.from_vector([0.2] * 192, [1, 3, 64], cpu, False)
cache.put(system_tokens, [k, v])
cached = cache.get_cache(system_tokens)
print("hit, cached tensors:", len(cached))
print("cached shapes:", [t.shape for t in cached])
cache.clear()
print("after clear, cached tensors:", len(cache.get_cache(system_tokens)))
assert len(cached) == 2
assert list(cached[0].shape) == [1, 3, 64]
```

Verified on CPU, LiteTorch 88a4ba1: miss returns 0 tensors, hit
returns the 2 stored KV tensors with their original shapes, and
`clear()` empties the cache. The cache is a singleton — `get()`
returns the same cache every call, so a test that forgets `clear()`
sees the previous test's entries.

## E8.4 — The serving scheduler: continuous batching

Without a scheduler, serving is one request at a time: prefill,
decode to completion, next request — and the GPU idles during every
single-token decode. Continuous batching packs the decode steps of
all running requests into one forward pass per tick, and admits new
requests the moment a slot frees, instead of waiting for the whole
batch to finish (static batching).

`ServingScheduler(num_blocks, block_size, max_num_seqs,
max_num_batched_tokens)` owns the block pool from E8.2.
`add_request(Request(id, prompt_tokens, max_new_tokens))` queues a
request. `step(model)` runs one tick — prefill for newly admitted
requests, one decode token per running request — and returns the
requests that finished on this step. The model contract is
`[batch, tokens]` id tensors in, `[batch, tokens, vocab]` logits out.

```
tick 1: admit A, B      -> prefill A, B
tick 2: decode A1, B1   -> A finishes? no
tick 3: decode A2, B2; admit C (slot freed) -> prefill C
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
    print("request", req.id, "generated:", req.generated_tokens)
by_id = {req.id: req for req in completed}
assert len(completed) == 2
assert len(by_id[1].generated_tokens) == 3
assert len(by_id[2].generated_tokens) == 2
```

Verified on CPU, LiteTorch 88a4ba1: both requests complete with
their requested token counts (3 and 2). Any model with the
`[batch, tokens]` → `[batch, tokens, vocab]` contract works — the
scheduler never looks inside the model.

## E8.5 — Speculative decoding: draft fast, verify in parallel

Decode is sequential: one token per forward pass, and each pass is
memory-bound. Speculative decoding breaks the serialization with a
small draft model. The draft proposes `lookahead` tokens cheaply;
the target model verifies all of them in **one** parallel forward
pass (the prefill path is parallel, so verifying 4 tokens costs
about the same as verifying 1); the matching prefix is accepted and
the first mismatch is corrected by the target:

```
draft proposes:  [t1, t2, t3, t4]
target verifies: [t1, t2, X,  .]   -> accept t1, t2; target emits t3'
net per target pass: 3 tokens instead of 1
```

`SpeculativeEngine(draft, target).generate(prompt, max_new_tokens,
lookahead)` returns the prompt plus the new tokens. Both models must
satisfy the same `[batch, tokens]` → `[batch, tokens, vocab]`
contract as the scheduler. The speedup depends on the draft's
acceptance rate: a draft that agrees with the target 80% of the time
nearly doubles throughput; a draft that never agrees costs a draft
pass per token and slows things down. Draft quality, not draft size,
is the tuning knob.

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
print("prompt + generated:", tokens)
print("new tokens:", len(tokens) - 3)
assert len(tokens) - 3 == 4
```

Verified on CPU, LiteTorch 88a4ba1: 4 new tokens appended to the
3-token prompt. The draft and target here are the same shape on
purpose — the example shows the contract, not a speedup; a real
deployment uses a much smaller draft.

## E8.6 — Medusa: extra heads instead of a draft model

Medusa removes the draft model entirely. Extra prediction heads sit
on top of the base model's final hidden state (or logits) and each
predicts one token further ahead: head 1 predicts token `t+1`, head 2
predicts `t+2`, and so on. One forward pass yields several candidate
tokens, which are then verified against the base model the same way
speculative decoding verifies draft tokens — typically with a tree of
candidates rather than a single chain, keeping the most likely
combinations.

The trade: no second model to load, schedule, or keep in sync — but
the heads must be trained (they predict further ahead from the same
representation, which is strictly harder than next-token
prediction), and their acceptance rate is usually lower than a good
draft model's. In LiteTorch a head is `MedusaHead(vocab, vocab)` —
it maps logits back to logits, so its input width is the vocab size,
not the hidden size.

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
print("prompt + generated:", tokens)
print("new tokens:", len(tokens) - 3)
assert len(tokens) - 3 == 4
```

Verified on CPU, LiteTorch 88a4ba1: 4 new tokens appended to the
3-token prompt. `MedusaEngine(base, [head, ...])` takes a list of
heads — add one head per extra position you want to predict.

## E8.7 — Decode attention and GPT/GPT2B serving

Two more pieces complete the serving picture. **Flash decoding**
(`Ops.flash_decoding(q, k, v, num_splits)`) is the decode-phase twin
of flash attention: same softmax(QK^T/sqrt(d))V math, but the KV
dimension is split `num_splits` ways and reduced, so a long context
decodes in parallel instead of in one serial reduction. `num_splits`
is a performance knob, not a math change — the result matches
`F.scaled_dot_product_attention` (D-20: LiteTorch has no SDPA; the
torch mirror below is the manual form).

**GPT2B** is LiteTorch's GPT-style serving model:
`GPT2B(vocab_size, d_model, n_layers, n_q_heads, n_kv_heads, d_ff,
rope_theta)` — grouped-query attention (`n_kv_heads < n_q_heads`
shrinks the KV cache per layer, which is exactly the E8.1
bottleneck), RoPE positions, and `stage_forward(input, stage)` for
pipeline-parallel serving across devices. `forward` takes `[batch,
tokens]` ids and returns `[batch, tokens, vocab]` logits — the same
contract the scheduler and `lt.nn.generate` expect. The defaults
(`vocab_size=50257, d_model=2560, n_layers=28`) are the real 2B
model; the example below builds a tiny one so it runs on CPU.

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
print("flash decoding shape:", out.shape)
print("first values:", out.to_vector()[:4])
qt = torch.tensor([0.1 * (i % 7 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
kt = torch.tensor([0.05 * (i % 5 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
vt = torch.tensor([0.3 * (i % 6 + 1) for i in range(64)]).reshape(1, 2, 4, 8)
ref = torch.softmax(qt @ kt.transpose(-2, -1) / math.sqrt(8), dim=-1) @ vt
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("max diff vs torch manual attention:", diff)
assert diff <= 1e-4
lt.manual_seed(7)
gpt = lt.nn.GPT2B(64, 32, 2, 4, 2, 64, 10000.0)
gpt.to(cpu)
ids = lt.Tensor.from_vector([1.0, 5.0, 9.0, 2.0], [1, 4], cpu, False)
logits = gpt.forward(ids)
print("GPT2B logits shape:", logits.shape)
config = lt.nn.GenerateConfig()
config.max_new_tokens = 4
print("GPT2B generated:", lt.nn.generate(gpt, [1, 5, 9], config))
assert list(logits.shape) == [1, 4, 64]
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: flash decoding
matches manual attention at parity 1e-4 (`num_splits=2` changes
nothing numerically), and the tiny GPT2B returns `[1, 4, 64]`
logits and generates 4 new tokens. `generate` returns only the new
tokens, not the prompt.

## E8.8 — What is NOT verified here

> REQUIRES-GPU: tokens/s and time-to-first-token for the scheduler,
> speculative decoding, and Medusa; paged attention at scale (thousands
> of blocks, long contexts); prefix-cache hit rates under real traffic;
> 2-GPU serving. All of these are Kaggle GPU measurements (B5). No
> throughput number is claimed on this page, and no CPU timing here
> predicts GPU serving performance — decode is memory-bandwidth-bound,
> so CPU timings would mislead in both directions.
