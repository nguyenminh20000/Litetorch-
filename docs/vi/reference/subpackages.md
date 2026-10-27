# Subpackage Python — Tài liệu tham khảo API

LiteTorch kèm bảy subpackage tùy chọn dưới `python/` (`ltvision`, `lttext`, `ltaudio`, `ltmobile`, `ltgeometric`, `ltrec`, `ltrrl`). Chúng là các lớp Python thuần trên extension cốt lõi — không cần bước cài đặt riêng, không cần build C++.

- **Nguồn chân lý:** `python/ltvision/`, `python/lttext/`, `python/ltaudio/`, `python/ltmobile/`, `python/ltgeometric/`, `python/ltrec/`, `python/ltrrl/`.
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu là extension 88a4ba1; B5 xác minh lại tại head).
- **Kiểm tra import (đã xác minh 2026-10-26 với extension 88a4ba1):** cả bảy package đều import sạch — `ltvision` 39 tên, `lttext` 13, `ltaudio` 16, `ltmobile` 9, `ltgeometric` 8, `ltrec` 6, `ltrrl` 11. Không có gì trong trang này bị đánh dấu thiếu.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Ngang bằng torch là FP32 1e-4 trừ khi tiêu đề ghi khác.
- **Ánh xạ PyTorch:** các subpackage phản chiếu `torchvision` / `torchtext` / `torchaudio` / `torchvision` (mobile) / `torch_geometric` / `torchrec` / `gym` + agent RL. Ánh xạ theo vai trò, không theo API chính xác — chữ ký khác nhau ở chỗ được ghi chú.

## ltvision — `python/ltvision/`

39 tên: mô hình (`ResNet`, `BasicBlock`, `Bottleneck`, `resnet18/34/50/101`, `VGG`, `vgg11/13/16/19`), tập dữ liệu (`CIFAR10`, `CIFAR100`, `MNIST`, `ImageFolder`, `FakeData`), transform (`Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, `Grayscale`, `Pad`, `ToTensor`, `ToPILImage`, `Normalize`), tiện ích (`make_grid`, `save_image`), cộng các re-export `DataLoader`, `Dataset`, `Sequential` và các submodule `datasets` / `models` / `transforms` / `utils`.

### `ltvision.resnet18` / `resnet34` / `resnet50` / `resnet101`

- **Chữ ký:** `resnet18(num_classes=1000) -> ResNet`; tương tự cho `resnet34`, `resnet50`, `resnet101`
- **Nguồn:** `python/ltvision/models.py:162-177`
- **Tham số:** `num_classes` — int, mặc định 1000.
- **Trả về:** một module `ResNet`. `ResNet(block, layers, num_classes=1000)` là class đứng sau chúng (`models.py:117-118`).
- **Trường hợp biên:** đầu vào mặc định là ảnh 3 kênh; convolution đầu tiên kỳ vọng 3 kênh đầu vào. `num_classes` chỉ thay đổi lớp linear cuối.
- **PyTorch:** `torchvision.models.resnet18(weights=None)` — cùng kiến trúc; factory của torch nhận trọng số, của LiteTorch chỉ nhận `num_classes` (không có trọng số tiền huấn luyện).

### `ltvision.VGG` / `vgg11` / `vgg13` / `vgg16` / `vgg19`

- **Chữ ký:** `vgg11(num_classes=1000) -> VGG` (tương tự cho `vgg13`, `vgg16`, `vgg19`); `VGG(features, num_classes=1000)` (`models.py:179`)
- **Trả về:** một module `VGG`.
- **PyTorch:** `torchvision.models.vgg11(weights=None)` — cùng kiến trúc, cùng lưu ý về trọng số.

### Tập dữ liệu `ltvision` — `CIFAR10`, `CIFAR100`, `MNIST`, `ImageFolder`, `FakeData`

- **Chữ ký:** `CIFAR10(root, train=True, transform=None, download=False)` (cùng hình thức cho `CIFAR100`, `MNIST`); `ImageFolder(root, transform=None)`; `FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=None)`
- **Nguồn:** `python/ltvision/datasets.py`
- **Trả về:** một `Dataset` (có thể index, `len()` chạy được). `download=True` tải tập dữ liệu từ web — cần mạng.
- **Trường hợp biên:** `download=False` (mặc định) với `root` thiếu sẽ raise tại lần truy cập đầu tiên, không phải tại lúc khởi tạo.
- **PyTorch:** `torchvision.datasets.CIFAR10` / `MNIST` / `ImageFolder` / `FakeData` — cùng chữ ký.

### Transform `ltvision` — `Compose`, `Resize`, `Normalize`, `ToTensor`, ...

- **Chữ ký:** `Compose(transforms)` (`transforms.py:34`) · `Resize(size, interpolation="bilinear")` (`transforms.py:131`) · `CenterCrop(size)` (`transforms.py:152`) · `RandomCrop(size, padding=0, pad_if_needed=False)` (`transforms.py:171`) · `RandomHorizontalFlip(p=0.5)` (`transforms.py:203`) · `Normalize(mean, std, inplace=False)` (`transforms.py:99`) · `ToTensor()` · `ToPILImage()` · `Grayscale()` · `Pad()` · `ColorJitter()` · `RandomRotation()` · `RandomVerticalFlip()` · `RandomResizedCrop()`
- **Trả về:** mỗi transform đều callable trên ảnh/tensor; `Compose` nối chúng theo thứ tự.
- **PyTorch:** `torchvision.transforms` — cùng tên và cùng thứ tự gọi.

### `ltvision.make_grid` / `ltvision.save_image`

- **Chữ ký:** `make_grid(tensor, nrow=8, padding=2) -> Tensor`; `save_image(tensor, fp, format=None) -> None`
- **Nguồn:** `python/ltvision/utils.py`
- **Trả về:** `make_grid` xếp một batch ảnh thành một ảnh lưới; `save_image` ghi nó ra file.
- **PyTorch:** `torchvision.utils.make_grid` / `save_image` — cùng vai trò.

<!-- ex: id=r16-vision device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import ltvision

cpu = lt.Device("cpu")
model = ltvision.resnet18(num_classes=10)
model.to(cpu)
x = lt.Tensor.from_vector([0.1] * (1 * 3 * 32 * 32), [1, 3, 32, 32], cpu)
print("resnet18 out shape:", model(x).shape)
norm = ltvision.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
print("normalized shape:", norm(x).shape)
print("grid shape:", ltvision.make_grid(x).shape)
```

## lttext — `python/lttext/`

13 tên: `Vocab`, `Field`, `Batch`, `build_vocab_from_iterator`, `get_tokenizer`, `basic_english`, `whitespace_tokenizer`, `ngrams_iterator`, `batch_iterator`, `bucket_iterator`, cộng các submodule `data` / `tokenizer` / `vocab`.

### `lttext.get_tokenizer` / `lttext.basic_english` / `lttext.whitespace_tokenizer`

- **Chữ ký:** `get_tokenizer(name) -> callable`; `basic_english(text) -> list[str]`; `whitespace_tokenizer(text) -> list[str]`
- **Nguồn:** `python/lttext/tokenizer.py`
- **Tham số:** `get_tokenizer("basic_english")` trả về tokenizer tiếng Anh cơ bản (viết thường, tách theo dấu câu); `get_tokenizer("whitespace")` trả về bộ tách theo khoảng trắng.
- **Trả về:** một danh sách token. `basic_english("Hello World")` trả về `['hello', 'world']` (đã xác minh).
- **PyTorch:** `torchtext.data.utils.get_tokenizer` — cùng tên và hành vi.

### `lttext.build_vocab_from_iterator` / `lttext.Vocab`

- **Chữ ký:** `build_vocab_from_iterator(iterator, specials=None, max_tokens=None, min_freq=1) -> Vocab`; `Vocab(counter, specials=None)`
- **Nguồn:** `python/lttext/vocab.py`
- **Tham số:** `iterator` — một iterator các danh sách token. `specials` — ví dụ `["<unk>"]`. `min_freq` — số lần xuất hiện tối thiểu để đưa một token vào.
- **Trả về:** một `Vocab` ánh xạ token tới chỉ số (`vocab["a"]`, `vocab.get_itos()`, `len(vocab)`).
- **PyTorch:** `torchtext.vocab.build_vocab_from_iterator` — cùng chữ ký.

### `lttext.Field` / `lttext.Batch` / `lttext.batch_iterator` / `lttext.bucket_iterator` / `lttext.ngrams_iterator`

- **Chữ ký:** `Field(tokenize=None, lower=False, ...)` · `Batch(data)` · `batch_iterator(data, batch_size)` · `bucket_iterator(data, batch_size)` · `ngrams_iterator(tokens, ngrams)`
- **Trả về:** `Field` tiền xử lý một cột của tập dữ liệu; `batch_iterator` yield các batch cỡ cố định; `bucket_iterator` yield các batch đã sắp theo độ dài (ít padding hơn); `ngrams_iterator` yield n-gram của một danh sách token.
- **PyTorch:** `torchtext.legacy.data.Field` / `BucketIterator` — cùng vai trò.

<!-- ex: id=r16-text device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import lttext

tok = lttext.get_tokenizer("basic_english")
print("tokens:", tok("Hello World"))
vocab = lttext.build_vocab_from_iterator([["a", "b", "a"]], specials=["<unk>"])
print("vocab size:", len(vocab), "index of a:", vocab["a"])
print("bigrams:", list(lttext.ngrams_iterator(["a", "b", "c"], 2)))
```

## ltaudio — `python/ltaudio/`

16 tên: transform (`Spectrogram`, `MelSpectrogram`, `MFCC`, `AmplitudeToDB`, `InverseSpectrogram`, `Resample`, `STFT`), I/O (`load`, `save`, `info`), tập dữ liệu (`WaveformDataset`, `SpeechCommands`), cộng `Dataset` và các submodule `datasets` / `io` / `transforms`.

### `ltaudio.Spectrogram` / `ltaudio.MelSpectrogram` / `ltaudio.MFCC` / `ltaudio.STFT`

- **Chữ ký:** `Spectrogram(n_fft=400, win_length=None, hop_length=None, window="hann", ...)` (`transforms.py:76`) · `MelSpectrogram(sample_rate=16000, n_fft=400, ...)` (`transforms.py:102`) · `MFCC(sample_rate=16000, n_mfcc=40, n_mels=128, n_fft=400, ...)` (`transforms.py:257`) · `STFT(n_fft=400, ...)` (`transforms.py:128`)
- **Tham số:** `n_fft` — kích thước FFT, mặc định 400. `hop_length` — bước nhảy giữa các khung (mặc định `n_fft // 4`). `sample_rate` — mặc định 16000.
- **Trả về:** một transform callable; gọi nó trên waveform tensor shape `[channels, time]` trả về spectrogram `[channels, n_fft // 2 + 1, frames]`. Đã xác minh: waveform mono 1600 mẫu cho shape `[1, 201, 13]` với cài đặt mặc định.
- **Trường hợp biên:** waveform ngắn hơn `n_fft` raise `ValueError: need at least one array to stack` (đã xác minh trên waveform 16 mẫu) — đầu vào phải chứa ít nhất một khung đầy đủ.
- **PyTorch:** `torchaudio.transforms.Spectrogram` / `MelSpectrogram` / `MFCC` — cùng tên và mặc định.

### `ltaudio.Resample` / `ltaudio.AmplitudeToDB` / `ltaudio.InverseSpectrogram`

- **Chữ ký:** `Resample(orig_freq, new_freq)` (`transforms.py:36`) · `AmplitudeToDB(stype="power", top_db=80.0, amin=1e-10, ref_value=1.0)` (`transforms.py:233`) · `InverseSpectrogram(...)`
- **Trả về:** `Resample` đổi tốc độ lấy mẫu; `AmplitudeToDB` chuyển spectrogram công suất/biên độ sang decibel; `InverseSpectrogram` đảo spectrogram về waveform.
- **PyTorch:** `torchaudio.transforms.Resample` / `AmplitudeToDB` / `InverseSpectrogram` — cùng tên.

### `ltaudio.load` / `ltaudio.save` / `ltaudio.info`

- **Chữ ký:** `load(filepath) -> (Tensor, sample_rate)`; `save(filepath, waveform, sample_rate) -> None`; `info(filepath) -> dict`
- **Trả về:** `load` đọc file `.wav`; `save` ghi một file; `info` trả về metadata mà không load mẫu.
- **PyTorch:** `torchaudio.load` / `save` / `info` — cùng chữ ký.

<!-- ex: id=r16-audio device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import ltaudio

cpu = lt.Device("cpu")
wav = lt.Tensor.from_vector([0.1] * 1600, [1, 1600], cpu)
spec = ltaudio.Spectrogram()
print("spectrogram shape:", spec(wav).shape)
db = ltaudio.AmplitudeToDB()
print("db shape:", db(spec(wav)).shape)
```

## ltmobile — `python/ltmobile/`

9 tên: `LiteModule`, `LiteSequential`, `QuantizedLinear`, `quantize_linear`, `quantize_module`, `quantization_error`, `export`, `export_litemodel`, cộng các submodule `quant` / `export`.

### `ltmobile.quantize_linear` / `ltmobile.quantize_module` / `ltmobile.quantization_error`

- **Chữ ký:** `quantize_linear(linear) -> QuantizedLinear` (`quant.py:80`) · `quantize_module(seq) -> LiteSequential` (`quant.py:86`) · `quantization_error(linear, qlinear, x) -> float` (`quant.py:98`)
- **Tham số:** `quantize_linear` nhận một `nn.Linear`; `quantize_module` nhận một `Sequential`; `quantization_error` so sánh đầu ra gốc và đầu ra đã lượng tử hóa trên đầu vào `x`.
- **Trả về:** `QuantizedLinear` là một module wrap trọng số INT8; `quantization_error` trả về độ chênh lệch tuyệt đối lớn nhất giữa hai đầu ra.
- **PyTorch:** `torch.ao.quantization.quantize_dynamic` — cùng vai trò; bộ lượng tử hóa mobile của LiteTorch là một triển khai riêng, nhỏ hơn.

### `ltmobile.LiteModule` / `ltmobile.LiteSequential` / `ltmobile.QuantizedLinear`

- **Chữ ký:** class. `LiteModule` là module cơ sở cho mô hình trên thiết bị; `LiteSequential` là `Sequential` các module lite; `QuantizedLinear` là lớp linear đã lượng tử hóa do `quantize_linear` trả về.
- **Trả về:** thay thế drop-in cho `nn.Module` / `nn.Sequential` / `nn.Linear` với footprint runtime nhỏ hơn.
- **PyTorch:** không có tương đương trực tiếp; gần nhất là mobile của `torch.jit` (`torch.utils.mobile_optimizer`).

### `ltmobile.export` / `ltmobile.export_litemodel`

- **Chữ ký:** `export(model, path) -> None`; `export_litemodel(model, path) -> None`
- **Tham số:** `model` — một `LiteModule` / `LiteSequential`. `path` — đường dẫn file đầu ra.
- **Trả về:** `None`. Ghi một file mô hình mobile tự chứa (trọng số + đồ thị) mà runtime mobile có thể load mà không cần toàn bộ framework.
- **PyTorch:** `torch.jit.save` + `torch.utils.mobile_optimizer.optimize_for_mobile` — cùng vai trò.

<!-- ex: id=r16-mobile device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import ltmobile

cpu = lt.Device("cpu")
lin = lt.nn.Linear(4, 4, True)
lin.to(cpu)
q = ltmobile.quantize_linear(lin)
print("quantized:", type(q).__name__)
x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
print("quantization error:", ltmobile.quantization_error(lin, q, x))
print("quantized forward shape:", q.forward(x).shape)
```

## ltgeometric — `python/ltgeometric/`

8 tên: `GCNConv`, `GATConv`, `SAGEConv`, `GraphConv`, `Data`, `Batch`, cộng các submodule `data` / `nn`.

### `ltgeometric.GCNConv` / `ltgeometric.SAGEConv` / `ltgeometric.GraphConv`

- **Chữ ký:** `GCNConv(in_channels, out_channels, bias=True)` (`nn.py:42`) · `SAGEConv(in_channels, out_channels, bias=True)` (`nn.py:59`) · `GraphConv(in_channels, out_channels, bias=True)` (`nn.py:79`); `forward(x, edge_index) -> Tensor`
- **Tham số:** `in_channels` / `out_channels` — chiều đặc trưng. `forward(x, edge_index)` — `x` là ma trận đặc trưng node `[num_nodes, in_channels]`; `edge_index` là danh sách cạnh.
- **Trả về:** đặc trưng node đã cập nhật `[num_nodes, out_channels]`. Đã xác minh: `GCNConv(4, 4).forward(x, adj)` với `x` shape `[1, 4]` và ma trận kề `[4, 4]` trả về shape `[1, 4]`.
- **PyTorch:** `torch_geometric.nn.GCNConv` / `SAGEConv` — cùng tên và cùng hình thức `forward(x, edge_index)`.

### `ltgeometric.GATConv`

- **Chữ ký:** `GATConv(in_channels, out_channels, heads=1, concat=True, ...)` (`nn.py:99`); `forward(x, edge_index) -> Tensor`
- **Tham số:** `heads` — số head attention, mặc định 1. `concat` — nối đầu ra các head (mặc định True) hoặc trung bình chúng.
- **Trả về:** đặc trưng node `[num_nodes, out_channels * heads]` khi `concat=True`, nếu không là `[num_nodes, out_channels]`.
- **PyTorch:** `torch_geometric.nn.GATConv` — cùng chữ ký.

### `ltgeometric.Data` / `ltgeometric.Batch`

- **Chữ ký:** `Data(x=None, edge_index=None, y=None, ...)`; `Batch.from_data_list(list_of_data) -> Batch`
- **Trả về:** `Data` giữ một đồ thị (đặc trưng node, cạnh, nhãn); `Batch` gộp một danh sách đồ thị thành một đồ thị batch (với vector `batch` gán node cho đồ thị).
- **PyTorch:** `torch_geometric.data.Data` / `Batch` — cùng vai trò.

<!-- ex: id=r16-geo device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import ltgeometric

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
adj = lt.Tensor.from_vector([1.0] * 16, [4, 4], cpu)
conv = ltgeometric.GCNConv(4, 4)
print("gcn out shape:", conv.forward(x, adj).shape)
print("sage out shape:", ltgeometric.SAGEConv(4, 4).forward(x, adj).shape)
```

## ltrec — `python/ltrec/`

6 tên: `DLRM`, `EmbeddingBag`, `FactorizationMachine`, `MLP`, `negative_sampling`, cộng submodule `rec`.

### `ltrec.DLRM`

- **Chữ ký:** `DLRM(num_embeddings_list, embedding_dim, dense_dim, bottom_dims=(), top_dims=())` (`rec.py:135`); `forward(dense_x, sparse_inputs) -> Tensor`
- **Nguồn:** `python/ltrec/rec.py:135-155`
- **Tham số:** `num_embeddings_list` — một **danh sách** kích thước bảng embedding, một cho mỗi đặc trưng thưa. `embedding_dim` — int. `dense_dim` — int, chiều đặc trưng dày. `bottom_dims` / `top_dims` — tuple kích thước ẩn của MLP.
- **Trả về:** một dự đoán click-through-rate `[batch, 1]`.
- **Trường hợp biên:** đối số đầu tiên là một danh sách — `DLRM([10, 8], 4, 8)` là đúng; `DLRM(4, 4, [4])` raise `TypeError: 'int' object is not iterable` bên trong `nn.Embedding`. `forward` nhận đặc trưng dày và một danh sách tensor chỉ số thưa.
- **PyTorch:** `torchrec.models.dlrm.DLRM` — cùng kiến trúc; của torch nhận config embedding, của LiteTorch nhận kích thước thuần.

### `ltrec.FactorizationMachine`

- **Chữ ký:** `FactorizationMachine(input_dim, factor_dim)` (`rec.py:85`); `forward(x) -> Tensor`
- **Tham số:** `input_dim` — số đặc trưng. `factor_dim` — kích thước latent-factor.
- **Trả về:** `[batch, 1]` — dự đoán FM (hạng tử tuyến tính + tương tác từng cặp).
- **PyTorch:** không có tương đương `torch`; đây là FM cổ điển của Rendle.

### `ltrec.EmbeddingBag`

- **Chữ ký:** `EmbeddingBag(num_embeddings, embedding_dim, mode="sum")` (`rec.py:37`); `forward(input, offsets) -> Tensor`
- **Tham số:** `mode` — `"sum"` (mặc định), `"mean"`, hoặc `"max"`.
- **Trả về:** embedding đã pool `[batch, embedding_dim]`.
- **PyTorch:** `torch.nn.EmbeddingBag` — cùng chữ ký và mode.

### `ltrec.MLP` / `ltrec.negative_sampling`

- **Chữ ký:** `MLP(dims)` (`rec.py:109`); `forward(x) -> Tensor`; `negative_sampling(edge_index, num_nodes, num_neg_samples=None, force_undirected=False) -> Tensor`
- **Tham số:** `MLP` nhận một danh sách kích thước lớp, ví dụ `MLP([8, 16, 1])` (ReLU giữa các lớp, không activation ở lớp cuối).
- **Trả về:** `MLP` trả về tensor đã biến đổi; `negative_sampling` nhận một tensor edge-index shape `[2, num_edges]` (hàng đầu là nguồn, hàng thứ hai là đích) và trả về một tensor các cạnh âm đã lấy mẫu, để huấn luyện recommender.
- **PyTorch:** không có tương đương trực tiếp; `MLP` là một helper thuần.

<!-- ex: id=r16-rec device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import ltrec

cpu = lt.Device("cpu")
fm = ltrec.FactorizationMachine(4, 8)
x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
print("fm out shape:", fm.forward(x).shape)
dlrm = ltrec.DLRM([10, 8], 4, 8)
print("dlrm:", type(dlrm).__name__)
ei = lt.Tensor.from_vector([0, 1, 1, 2], [2, 2], cpu)
print("negatives:", ltrec.negative_sampling(ei, 4, 2).to_vector())
```

## ltrrl — `python/ltrrl/`

11 tên: `Env`, `GridWorldEnv`, `BanditEnv`, `ReplayBuffer`, `DQN`, `PPO`, `REINFORCE`, `compute_returns`, cộng các submodule `env` / `buffer` / `agents`.

### `ltrrl.Env` / `ltrrl.GridWorldEnv` / `ltrrl.BanditEnv`

- **Chữ ký:** `Env()` (cơ sở — `reset()` / `step(action)` raise `NotImplementedError`) · `GridWorldEnv()` · `BanditEnv(probs)`
- **Nguồn:** `python/ltrrl/env.py`
- **Trả về:** `reset()` trả về quan sát ban đầu (một danh sách float — vị trí lưới one-hot cho `GridWorldEnv`, `[0.0]` cho `BanditEnv`). `step(action)` trả về `(observation, reward, done, info)`. Đã xác minh: `GridWorldEnv().reset()` trả về danh sách one-hot 16 phần tử; `step(0)` trả về `(obs, -0.01, False, {})`; `BanditEnv(probs).reset()` trả về `[0.0]` và `step` trả về `([0.0], reward, True, {})`.
- **PyTorch:** `gym.Env` — cùng hợp đồng `reset` / `step` (observation, reward, done, info).

### `ltrrl.ReplayBuffer`

- **Chữ ký:** `ReplayBuffer(capacity)` (`buffer.py:8`); `push(obs, action, reward, next_obs, done)` · `sample(batch_size) -> list` · `__len__()`
- **Tham số:** `capacity` — int, số transition tối đa được lưu (cũ nhất bị loại trước).
- **Trả về:** `push` lưu một transition; `sample` trả về một batch ngẫu nhiên gồm `batch_size` transition.
- **PyTorch:** không có tương đương `torch`; đây là replay buffer DQN chuẩn.

### `ltrrl.DQN` / `ltrrl.PPO` / `ltrrl.REINFORCE`

- **Chữ ký:** `DQN(obs_dim, n_actions, hidden=128, lr=1e-3, gamma=0.99)` (`agents.py:69`) · `PPO(obs_dim, n_actions, hidden=128, lr=3e-4, clip_eps=0.2)` (`agents.py:109`) · `REINFORCE(obs_dim, n_actions, hidden=128, lr=1e-3)` (`agents.py:159`)
- **Tham số:** chiều quan sát/hành động cộng siêu tham số. (`agents.py:30` định nghĩa một helper `MLP` mà các agent dùng, không phải agent.)
- **Trả về:** các agent với các phương thức kiểu `act(obs)` / `update(batch)` (xem `agents.py` để biết danh sách phương thức chính xác).
- **PyTorch:** không có tương đương `torch`; đây là các agent RL viết từ đầu trên module LiteTorch.

### `ltrrl.compute_returns`

- **Chữ ký:** `compute_returns(rewards, gamma=0.99) -> list[float]`
- **Tham số:** `rewards` — danh sách reward từng bước. `gamma` — hệ số chiết khấu, mặc định 0.99.
- **Trả về:** return chiết khấu từ mỗi bước.
- **PyTorch:** không có tương đương trực tiếp.

<!-- ex: id=r16-rl device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import ltrrl

env = ltrrl.GridWorldEnv()
obs = env.reset()
print("obs len:", len(obs))
obs2, reward, done, info = env.step(0)
print("step reward:", reward, "done:", done)
buf = ltrrl.ReplayBuffer(10)
buf.push(obs, 0, reward, obs2, done)
print("buffer len:", len(buf), "sample:", len(buf.sample(1)))
print("returns:", ltrrl.compute_returns([1.0, 1.0, 1.0], 0.9))
```
