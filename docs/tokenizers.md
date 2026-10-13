# Tokenizers in LiteTorch

LiteTorch does not reimplement tokenizers. Use the HuggingFace `tokenizers`
Python package (Rust-backed, fast) and feed integer token IDs into LiteTorch
models.

## Install

```bash
pip install tokenizers
```

## Usage

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello world").ids

import litetorch as lt
input_ids = lt.tensor([ids], dtype=lt.int64)  # [1, seq]
# model embedding consumes input_ids
```

## Notes

- `lt.hf.from_pretrained(repo_id)` loads model weights (safetensors) and
  `config.json`, but not the tokenizer — load it separately as above.
- For chat templates / special tokens, use the `transformers` package's
  `AutoTokenizer` if you need them; LiteTorch stays tokenizer-agnostic by
  design.

# Tokenizer trong LiteTorch

LiteTorch không tự viết lại tokenizer. Hãy dùng package Python
`tokenizers` của HuggingFace (viết bằng Rust, nhanh) rồi đưa token ID
dạng số nguyên vào model LiteTorch.

## Cài đặt

```bash
pip install tokenizers
```

## Cách dùng

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("gpt2")
ids = tok.encode("Xin chào").ids

import litetorch as lt
input_ids = lt.tensor([ids], dtype=lt.int64)  # [1, seq]
# embedding của model nhận input_ids
```

## Lưu ý

- `lt.hf.from_pretrained(repo_id)` chỉ tải trọng số model (safetensors) và
  `config.json`, không tải tokenizer — tải riêng như trên.
- Nếu cần chat template / special token, dùng `AutoTokenizer` của package
  `transformers`; LiteTorch cố ý không phụ thuộc tokenizer nào.
