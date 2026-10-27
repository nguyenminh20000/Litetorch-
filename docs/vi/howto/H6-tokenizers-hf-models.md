# How-To H6 — Tokenizer và mô hình HF

LiteTorch không tự implement tokenizer. Hãy dùng gói Python
`tokenizers` của HuggingFace (chạy bằng Rust, nhanh) và đưa các token
ID nguyên vào mô hình LiteTorch. LiteTorch cố tình không phụ thuộc
tokenizer.

Điều kiện tiên quyết: `pip install tokenizers`. Ví dụ đầu tiên tải
tokenizer GPT-2 từ HuggingFace Hub — nó cần truy cập mạng. Mọi ví dụ
dưới đây chạy trên CPU với LiteTorch 88a4ba1.

## H6.1 — Tokenize text và đưa ID vào mô hình

Tokenize bằng `Tokenizer.from_pretrained`, rồi tạo tensor đầu vào bằng
`lt.tensor([ids], dtype=lt.DataType.INT64)`. Cách viết dtype rất quan
trọng: `lt.int64` không tồn tại (enum được export dưới dạng
`lt.DataType.INT64`, cũng có ở cấp cao nhất là `lt.INT64`). Tensor
`[1, seq]` thu được đưa thẳng vào một `Embedding`.

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

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: token ID
`[15496, 995]`, `input_ids` shape `[1, 2]` với dtype `INT64`, output
embedding shape `[1, 2, 8]`.

Caveat về độ chính xác: `lt.tensor` materialize giá trị qua
`std::vector<float>` trước rồi mới cast sang dtype yêu cầu, nên các
token ID nguyên trên 2^24 mất độ chính xác trong round-trip. Vocabulary
thực tế nhỏ hơn nhiều (GPT-2: 50.257), nhưng đừng dùng `lt.tensor`
cho dữ liệu int64 tùy ý — hãy dùng `Tensor.from_vector` với dtype
tường minh thay thế (xem thêm D-18).

## H6.2 — Load trọng số pretrained bằng lt.hf.from_pretrained

`lt.hf.from_pretrained(repo_id)` load trọng số mô hình (safetensors)
và `config.json` cho một mô hình HuggingFace — nhưng không load
tokenizer. Load tokenizer riêng như ở H6.1. Đối với chat template và
special token, dùng `AutoTokenizer` của gói `transformers` nếu bạn cần.

```python skip-run
import litetorch as lt

bundle = lt.hf.from_pretrained("gpt2")
print("config:", bundle.config)
print("state dict keys:", list(bundle.state_dict.keys())[:5])
```

Block này là fragment, không phải ví dụ chạy được: nó tải trọng số mô
hình (hàng trăm megabyte) và cần truy cập mạng cộng dung lượng đĩa.
Bản thân API đã được kiểm chứng với binding
(`bind_hf.cpp:33`, `from_pretrained(repo_id, revision="main")`). Để map
trọng số đã tải lên một mô hình LiteTorch, dùng
`lt.hf.remap_state_dict(state_dict, model_type)` và
`lt.hf.weight_name_map(model_type)` — cả hai nhận chuỗi model type
(ví dụ `"gpt2"`) và dịch tên trọng số HuggingFace sang tên tham số
LiteTorch.

## H6.3 — Build một tokenizer nhỏ offline (không cần tải)

Khi không có mạng, hãy build tokenizer từ text của chính bạn bằng cùng
gói `tokenizers` — huấn luyện một BPE nhỏ trên vài câu, rồi dùng nó
y hệt tokenizer pretrained ở H6.1.

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

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: một danh
sách ngắn các token ID (giá trị chính xác phụ thuộc vocabulary đã huấn
luyện — ví dụ khẳng định shape, không khẳng định ID), `input_ids`
dtype `INT64`, output embedding shape `[1, seq, 8]`. Chú ý kích thước
embedding (200) khớp với kích thước vocabulary đã huấn luyện: giữ hai
cái đồng bộ, không lookup sẽ raise.
