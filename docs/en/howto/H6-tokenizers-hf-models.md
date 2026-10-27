# How-To H6 — Tokenizers and HF Models

LiteTorch does not reimplement tokenizers. Use the HuggingFace
`tokenizers` Python package (Rust-backed, fast) and feed integer token
IDs into LiteTorch models. LiteTorch stays tokenizer-agnostic by design.

Prerequisites: `pip install tokenizers`. The first example downloads
the GPT-2 tokenizer from the HuggingFace Hub — it needs network access.
All examples below run on CPU against LiteTorch 88a4ba1.

## H6.1 — Tokenize text and feed the IDs to a model

Tokenize with `Tokenizer.from_pretrained`, then build the input tensor
with `lt.tensor([ids], dtype=lt.DataType.INT64)`. The dtype spelling
matters: `lt.int64` does not exist (the enum is exported as
`lt.DataType.INT64`, also available as top-level `lt.INT64`). The
resulting `[1, seq]` tensor feeds an `Embedding` directly.

<!-- ex: id=H6.1-1 device=cpu backend=cpu runtime=~10s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
from tokenizers import Tokenizer

import litetorch as lt

tok = Tokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello world").ids
print("token ids:", ids)

input_ids = lt.tensor([ids], dtype=lt.DataType.INT64)
print("input_ids shape:", list(input_ids.shape), "dtype:", input_ids.dtype)

device = lt.auto_device()
emb = lt.nn.Embedding(50257, 8)
emb.to(device)
out = emb.forward(input_ids)
print("embedding output shape:", list(out.shape))
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: token IDs
`[15496, 995]`, `input_ids` shape `[1, 2]` with dtype `INT64`, embedding
output shape `[1, 2, 8]`.

Precision caveat: `lt.tensor` materializes values through
`std::vector<float>` first and casts to the requested dtype after, so
integer token IDs above 2^24 lose precision in the round-trip. Real
vocabularies are far below that (GPT-2: 50,257), but do not use
`lt.tensor` for arbitrary int64 data — use `Tensor.from_vector` with
an explicit dtype instead (see also D-18).

## H6.2 — Load pretrained weights with lt.hf.from_pretrained

`lt.hf.from_pretrained(repo_id)` loads model weights (safetensors) and
`config.json` for a HuggingFace model — but not the tokenizer. Load the
tokenizer separately as in H6.1. For chat templates and special tokens,
use the `transformers` package's `AutoTokenizer` if you need them.

```python skip-run
import litetorch as lt

bundle = lt.hf.from_pretrained("gpt2")
print("config:", bundle.config)
print("state dict keys:", list(bundle.state_dict.keys())[:5])
```

This block is a fragment, not a runnable example: it downloads model
weights (hundreds of megabytes) and needs network access plus disk
space. The API itself is verified against the binding
(`bind_hf.cpp:33`, `from_pretrained(repo_id, revision="main")`). To map
the downloaded weights onto a LiteTorch model, use
`lt.hf.remap_state_dict(state_dict, model_type)` and
`lt.hf.weight_name_map(model_type)` — both take the model type string
(e.g. `"gpt2"`) and translate HuggingFace weight names to LiteTorch
parameter names.

## H6.3 — Build a tiny tokenizer offline (no download)

When there is no network, build a tokenizer from your own text with the
same `tokenizers` package — train a small BPE on a few sentences, then
use it exactly like the pretrained one in H6.1.

<!-- ex: id=H6.3-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

import litetorch as lt

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tok.train_from_iterator(
    ["Hello world", "LiteTorch runs on CPU", "Hello LiteTorch", "world of tensors"],
    trainer=trainer,
)
ids = tok.encode("Hello world").ids
print("token ids:", ids)

input_ids = lt.tensor([ids], dtype=lt.DataType.INT64)
device = lt.auto_device()
emb = lt.nn.Embedding(200, 8)
emb.to(device)
out = emb.forward(input_ids)
print("embedding output shape:", list(out.shape))
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: a short list of
token IDs (exact values depend on the trained vocabulary — the example
asserts the shapes, not the IDs), `input_ids` dtype `INT64`, embedding
output shape `[1, seq, 8]`. Note the embedding size (200) matches the
trained vocabulary size: keep the two in sync or the lookup raises.
