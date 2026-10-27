# Tutorial T3 — Fine-Tune and Serve a Small LLM

Fine-tune a tiny language model on CPU, then serve it: flash attention,
paged KV cache, continuous batching, prefix caching, speculative
decoding, Medusa heads, and guided decoding. Every serving call below
runs on CPU against LiteTorch 88a4ba1 — the shapes and signatures are
the real ones, corrected where the old skeleton docs were wrong.

What you will learn:

- How to fine-tune a small embedding + linear language model.
- How attention works in LiteTorch (`flash_attention`, `paged_attention`).
- How the serving stack fits together: scheduler, prefix cache,
  speculative decoding, Medusa, guided decoding.
- Which serving signatures the old docs got wrong, and what is correct.

Prerequisites: Tutorials T1–T2. The model here is tiny (vocab 64) on
purpose: it runs in seconds on CPU and every mechanism is visible.
Throughput numbers belong on GPU and are verified on Kaggle in B5, not
claimed here.

> PyTorch differences touched in this tutorial: D-20 (no
> `F.scaled_dot_product_attention` — use `Ops.flash_attention(q, k, v)`;
> no `scale` or `dropout_p` argument, scale Q yourself; head dim must
> satisfy 1 <= D <= 256), D-03 (`.to()` takes a `Device`), D-02 (silent
> CPU fallback — check the device after `.to()`).

## T3.1 — Flash attention

`lt.Ops.flash_attention(q, k, v)` computes softmax(QK^T / sqrt(d))V.
There is no `scale` argument: scale Q yourself before the call.
`causal=True` masks future positions.

<!-- ex: id=T3.1-1 device=cpu backend=cpu runtime=~6.5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import math
import litetorch as lt
import torch

device = lt.auto_device()

q = lt.Tensor.from_vector([0.1] * 64, [1, 2, 4, 8], device, False)
k = lt.Tensor.from_vector([0.2] * 64, [1, 2, 4, 8], device, False)
v = lt.Tensor.from_vector([0.3] * 64, [1, 2, 4, 8], device, False)

out = lt.Ops.flash_attention(q, k, v)
print("flash attention shape:", out.shape)
print("first values:", out.to_vector()[:4])

qt = torch.full((1, 2, 4, 8), 0.1)
kt = torch.full((1, 2, 4, 8), 0.2)
vt = torch.full((1, 2, 4, 8), 0.3)
ref = torch.softmax(qt @ kt.transpose(-2, -1) / math.sqrt(8), dim=-1) @ vt
print("torch reference first values:", ref.flatten().tolist()[:4])
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(flash attention matches manual `softmax(QK^T/sqrt(d))V`).

## T3.2 — Paged attention and the KV cache

Paged attention splits the KV cache into fixed-size blocks so the cache
never fragments. The signature is positional and strict: `q` is rank 3,
`k_cache`/`v_cache` are rank 4 `[num_blocks, num_kv_heads, block_size,
head_dim]`, `block_tables` and `context_lens` are rank 2 / rank 1 and
must be INT32, and `block_size` is a plain int argument — not a keyword.
(The old skeleton §9.1 passed FP32 tables, rank-3 caches, and
`block_size=16` as a keyword; all three are wrong.)

<!-- ex: id=T3.2-1 device=cpu backend=cpu runtime=~6.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
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
print("paged attention shape:", out.shape)
print("first values:", out.to_vector()[:4])

qt = torch.full((1, 1, 16), 0.1)
kt = torch.full((4, 16), 0.2)
vt = torch.full((4, 16), 0.3)
ref = torch.softmax(qt.squeeze(0) @ kt.T / math.sqrt(16), dim=-1) @ vt
print("torch reference first values:", ref.flatten().tolist()[:4])
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(paged attention over the 4 cached positions matches manual attention).

## T3.3 — Continuous batching with the serving scheduler

The scheduler admits requests, assigns KV blocks, and steps the model
once per tick: prefill for new prompts, one decode token per running
request. `step(model)` takes the model and returns the requests that
finished on this step.

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
    print("request", req.id, "generated:", req.generated_tokens)
```

Verified on CPU, LiteTorch 88a4ba1: both requests complete with 4
generated tokens each. The model here is the tiny embedding + linear
LM from T3.8 — the scheduler feeds it `[batch, tokens]` id tensors
and reads `[batch, tokens, vocab]` logits, so any model with that
contract works.

## T3.4 — Prefix caching

Recurring system prompts reuse their KV states instead of recomputing
them. `put` takes the token list and a list of KV tensors — one call,
two arguments, not three.

<!-- ex: id=T3.4-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
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
print("cached tensors:", len(cached))
print("cached shapes:", [t.shape for t in cached])
```

Verified on CPU, LiteTorch 88a4ba1: the same token list returns the 2
stored KV tensors with their original shapes.

## T3.5 — Speculative decoding

A small draft model proposes `lookahead` tokens; the target model
verifies them in parallel and accepts the matching prefix. The result
is the prompt plus the new tokens.

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
print("prompt + generated:", tokens)
print("new tokens:", len(tokens) - 3)
```

Verified on CPU, LiteTorch 88a4ba1: 4 new tokens appended to the
3-token prompt. Both models must map `[batch, tokens]` ids to
`[batch, tokens, vocab]` logits — the same contract as the scheduler.

## T3.6 — Medusa multi-head decoding

Medusa heads predict extra tokens per forward pass without a separate
draft model. Each head maps logits back to logits, so a head is
`MedusaHead(vocab, vocab)` — its input width is the vocab size, not
the hidden size.

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
print("prompt + generated:", tokens)
print("new tokens:", len(tokens) - 3)
```

Verified on CPU, LiteTorch 88a4ba1: 4 new tokens appended to the
3-token prompt.

## T3.7 — Guided decoding (structured outputs)

`GuidedDecoder` masks logits so generation can only continue a given
pattern. `apply_mask` works in place and returns nothing — read the
logits after the call. The pattern must start with `prefix:` or
`regex:`; a bare pattern matches everything and masks nothing.

<!-- ex: id=T3.7-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

vocab = ["{", "}", '"name"', ":", '"John"', '"age"', "25", ","]
decoder = lt.nn.GuidedDecoder(vocab)

logits = lt.Tensor.from_vector([1.0] * len(vocab), [1, len(vocab)], device, False)
decoder.apply_mask(logits, prefix="{", target_pattern='prefix:{"name": "J')
print("vocab:", vocab)
print("masked logits:", logits.to_vector())
```

Verified on CPU, LiteTorch 88a4ba1: only `'"name"'` keeps its logit
(1.0); every other token is masked to -1e9, because only `{"name"`
continues the required prefix `{"name": "J`.

## T3.8 — Fine-tune the tiny LM, then generate

Train the embedding + linear model on random token sequences with
next-token prediction, then generate from it. Eight epochs on 32
sequences of length 8: average loss falls from 4.2863 to 2.4816.

<!-- ex: id=T3.8-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
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
print("generated:", tokens)
```

Verified on CPU, LiteTorch 88a4ba1: loss 4.2863 → 2.4816, generation
returns 8 new tokens. No pointwise torch parity claim: the training
data is random and the initialization differs across frameworks. The
loss itself is the T2.3-verified `CrossEntropyLoss`.
