# Python Subpackages — API Reference

LiteTorch ships seven optional subpackages under `python/` (`ltvision`, `lttext`, `ltaudio`, `ltmobile`, `ltgeometric`, `ltrec`, `ltrrl`). They are pure-Python layers on top of the core extension — no separate install step, no C++ to build.

- **Source of truth:** `python/ltvision/`, `python/lttext/`, `python/ltaudio/`, `python/ltmobile/`, `python/ltgeometric/`, `python/ltrec/`, `python/ltrrl/`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline is the 88a4ba1 extension; B5 re-verifies at head).
- **Import check (verified 2026-10-26 against the 88a4ba1 extension):** all seven packages import cleanly — `ltvision` 39 names, `lttext` 13, `ltaudio` 16, `ltmobile` 9, `ltgeometric` 8, `ltrec` 6, `ltrrl` 11. Nothing in this page is marked missing.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **PyTorch mapping:** the subpackages mirror `torchvision` / `torchtext` / `torchaudio` / `torchvision` (mobile) / `torch_geometric` / `torchrec` / `gym` + RL agents. The mapping is by role, not by exact API — signatures differ where noted.

## ltvision — `python/ltvision/`

39 names: models (`ResNet`, `BasicBlock`, `Bottleneck`, `resnet18/34/50/101`, `VGG`, `vgg11/13/16/19`), datasets (`CIFAR10`, `CIFAR100`, `MNIST`, `ImageFolder`, `FakeData`), transforms (`Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, `Grayscale`, `Pad`, `ToTensor`, `ToPILImage`, `Normalize`), utils (`make_grid`, `save_image`), plus `DataLoader`, `Dataset`, `Sequential` re-exports and the `datasets` / `models` / `transforms` / `utils` submodules.

### `ltvision.resnet18` / `resnet34` / `resnet50` / `resnet101`

- **Signature:** `resnet18(num_classes=1000) -> ResNet`; same for `resnet34`, `resnet50`, `resnet101`
- **Source:** `python/ltvision/models.py:162-177`
- **Parameters:** `num_classes` — int, default 1000.
- **Returns:** a `ResNet` module. `ResNet(block, layers, num_classes=1000)` is the class behind them (`models.py:117-118`).
- **Edge cases:** the default input is a 3-channel image; the first convolution expects 3 input channels. `num_classes` changes only the final linear layer.
- **PyTorch:** `torchvision.models.resnet18(weights=None)` — same architecture; torch's factory takes weights, LiteTorch's takes only `num_classes` (no pretrained weights).

### `ltvision.VGG` / `vgg11` / `vgg13` / `vgg16` / `vgg19`

- **Signature:** `vgg11(num_classes=1000) -> VGG` (same for `vgg13`, `vgg16`, `vgg19`); `VGG(features, num_classes=1000)` (`models.py:179`)
- **Returns:** a `VGG` module.
- **PyTorch:** `torchvision.models.vgg11(weights=None)` — same architecture, same weights caveat.

### `ltvision` datasets — `CIFAR10`, `CIFAR100`, `MNIST`, `ImageFolder`, `FakeData`

- **Signature:** `CIFAR10(root, train=True, transform=None, download=False)` (same shape for `CIFAR100`, `MNIST`); `ImageFolder(root, transform=None)`; `FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=None)`
- **Source:** `python/ltvision/datasets.py`
- **Returns:** a `Dataset` (indexable, `len()` works). `download=True` fetches the dataset from the web — needs network.
- **Edge cases:** `download=False` (the default) with a missing `root` raises at first access, not at construction.
- **PyTorch:** `torchvision.datasets.CIFAR10` / `MNIST` / `ImageFolder` / `FakeData` — same signatures.

### `ltvision` transforms — `Compose`, `Resize`, `Normalize`, `ToTensor`, ...

- **Signature:** `Compose(transforms)` (`transforms.py:34`) · `Resize(size, interpolation="bilinear")` (`transforms.py:131`) · `CenterCrop(size)` (`transforms.py:152`) · `RandomCrop(size, padding=0, pad_if_needed=False)` (`transforms.py:171`) · `RandomHorizontalFlip(p=0.5)` (`transforms.py:203`) · `Normalize(mean, std, inplace=False)` (`transforms.py:99`) · `ToTensor()` · `ToPILImage()` · `Grayscale()` · `Pad()` · `ColorJitter()` · `RandomRotation()` · `RandomVerticalFlip()` · `RandomResizedCrop()`
- **Returns:** each transform is callable on an image/tensor; `Compose` chains them in order.
- **PyTorch:** `torchvision.transforms` — same names and the same call order.

### `ltvision.make_grid` / `ltvision.save_image`

- **Signature:** `make_grid(tensor, nrow=8, padding=2) -> Tensor`; `save_image(tensor, fp, format=None) -> None`
- **Source:** `python/ltvision/utils.py`
- **Returns:** `make_grid` tiles a batch of images into one grid image; `save_image` writes it to a file path.
- **PyTorch:** `torchvision.utils.make_grid` / `save_image` — same role.

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

13 names: `Vocab`, `Field`, `Batch`, `build_vocab_from_iterator`, `get_tokenizer`, `basic_english`, `whitespace_tokenizer`, `ngrams_iterator`, `batch_iterator`, `bucket_iterator`, plus the `data` / `tokenizer` / `vocab` submodules.

### `lttext.get_tokenizer` / `lttext.basic_english` / `lttext.whitespace_tokenizer`

- **Signature:** `get_tokenizer(name) -> callable`; `basic_english(text) -> list[str]`; `whitespace_tokenizer(text) -> list[str]`
- **Source:** `python/lttext/tokenizer.py`
- **Parameters:** `get_tokenizer("basic_english")` returns the basic-English tokenizer (lowercases, splits on punctuation); `get_tokenizer("whitespace")` returns the whitespace splitter.
- **Returns:** a list of tokens. `basic_english("Hello World")` returns `['hello', 'world']` (verified).
- **PyTorch:** `torchtext.data.utils.get_tokenizer` — same names and behavior.

### `lttext.build_vocab_from_iterator` / `lttext.Vocab`

- **Signature:** `build_vocab_from_iterator(iterator, specials=None, max_tokens=None, min_freq=1) -> Vocab`; `Vocab(counter, specials=None)`
- **Source:** `python/lttext/vocab.py`
- **Parameters:** `iterator` — an iterator of token lists. `specials` — e.g. `["<unk>"]`. `min_freq` — minimum count to include a token.
- **Returns:** a `Vocab` mapping tokens to indices (`vocab["a"]`, `vocab.get_itos()`, `len(vocab)`).
- **PyTorch:** `torchtext.vocab.build_vocab_from_iterator` — same signature.

### `lttext.Field` / `lttext.Batch` / `lttext.batch_iterator` / `lttext.bucket_iterator` / `lttext.ngrams_iterator`

- **Signature:** `Field(tokenize=None, lower=False, ...)` · `Batch(data)` · `batch_iterator(data, batch_size)` · `bucket_iterator(data, batch_size)` · `ngrams_iterator(tokens, ngrams)`
- **Returns:** `Field` preprocesses one column of a dataset; `batch_iterator` yields fixed-size batches; `bucket_iterator` yields batches sorted by length (less padding); `ngrams_iterator` yields n-grams of a token list.
- **PyTorch:** `torchtext.legacy.data.Field` / `BucketIterator` — same roles.

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

16 names: transforms (`Spectrogram`, `MelSpectrogram`, `MFCC`, `AmplitudeToDB`, `InverseSpectrogram`, `Resample`, `STFT`), I/O (`load`, `save`, `info`), datasets (`WaveformDataset`, `SpeechCommands`), plus `Dataset` and the `datasets` / `io` / `transforms` submodules.

### `ltaudio.Spectrogram` / `ltaudio.MelSpectrogram` / `ltaudio.MFCC` / `ltaudio.STFT`

- **Signature:** `Spectrogram(n_fft=400, win_length=None, hop_length=None, window="hann", ...)` (`transforms.py:76`) · `MelSpectrogram(sample_rate=16000, n_fft=400, ...)` (`transforms.py:102`) · `MFCC(sample_rate=16000, n_mfcc=40, n_mels=128, n_fft=400, ...)` (`transforms.py:257`) · `STFT(n_fft=400, ...)` (`transforms.py:128`)
- **Parameters:** `n_fft` — FFT size, default 400. `hop_length` — hop between frames (default `n_fft // 4`). `sample_rate` — default 16000.
- **Returns:** a callable transform; calling it on a waveform tensor of shape `[channels, time]` returns a spectrogram `[channels, n_fft // 2 + 1, frames]`. Verified: a 1600-sample mono waveform gives shape `[1, 201, 13]` with default settings.
- **Edge cases:** a waveform shorter than `n_fft` raises `ValueError: need at least one array to stack` (verified on a 16-sample waveform) — the input must contain at least one full frame.
- **PyTorch:** `torchaudio.transforms.Spectrogram` / `MelSpectrogram` / `MFCC` — same names and defaults.

### `ltaudio.Resample` / `ltaudio.AmplitudeToDB` / `ltaudio.InverseSpectrogram`

- **Signature:** `Resample(orig_freq, new_freq)` (`transforms.py:36`) · `AmplitudeToDB(stype="power", top_db=80.0, amin=1e-10, ref_value=1.0)` (`transforms.py:233`) · `InverseSpectrogram(...)`
- **Returns:** `Resample` changes the sample rate; `AmplitudeToDB` converts a power/amplitude spectrogram to decibels; `InverseSpectrogram` inverts a spectrogram back to a waveform.
- **PyTorch:** `torchaudio.transforms.Resample` / `AmplitudeToDB` / `InverseSpectrogram` — same names.

### `ltaudio.load` / `ltaudio.save` / `ltaudio.info`

- **Signature:** `load(filepath) -> (Tensor, sample_rate)`; `save(filepath, waveform, sample_rate) -> None`; `info(filepath) -> dict`
- **Returns:** `load` reads a `.wav` file; `save` writes one; `info` returns metadata without loading the samples.
- **PyTorch:** `torchaudio.load` / `save` / `info` — same signatures.

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

9 names: `LiteModule`, `LiteSequential`, `QuantizedLinear`, `quantize_linear`, `quantize_module`, `quantization_error`, `export`, `export_litemodel`, plus the `quant` / `export` submodules.

### `ltmobile.quantize_linear` / `ltmobile.quantize_module` / `ltmobile.quantization_error`

- **Signature:** `quantize_linear(linear) -> QuantizedLinear` (`quant.py:80`) · `quantize_module(seq) -> LiteSequential` (`quant.py:86`) · `quantization_error(linear, qlinear, x) -> float` (`quant.py:98`)
- **Parameters:** `quantize_linear` takes an `nn.Linear`; `quantize_module` takes a `Sequential`; `quantization_error` compares the original and quantized outputs on input `x`.
- **Returns:** `QuantizedLinear` is a module wrapping the INT8 weight; `quantization_error` returns the max absolute difference between the two outputs.
- **PyTorch:** `torch.ao.quantization.quantize_dynamic` — same role; LiteTorch's mobile quantizer is a separate, smaller implementation.

### `ltmobile.LiteModule` / `ltmobile.LiteSequential` / `ltmobile.QuantizedLinear`

- **Signature:** classes. `LiteModule` is the base module for on-device models; `LiteSequential` is a `Sequential` of lite modules; `QuantizedLinear` is the quantized linear layer returned by `quantize_linear`.
- **Returns:** drop-in replacements for `nn.Module` / `nn.Sequential` / `nn.Linear` with a smaller runtime footprint.
- **PyTorch:** no direct equivalent; closest is `torch.jit` mobile (`torch.utils.mobile_optimizer`).

### `ltmobile.export` / `ltmobile.export_litemodel`

- **Signature:** `export(model, path) -> None`; `export_litemodel(model, path) -> None`
- **Parameters:** `model` — a `LiteModule` / `LiteSequential`. `path` — output file path.
- **Returns:** `None`. Writes a self-contained mobile model file (weights + graph) that a mobile runtime can load without the full framework.
- **PyTorch:** `torch.jit.save` + `torch.utils.mobile_optimizer.optimize_for_mobile` — same role.

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

8 names: `GCNConv`, `GATConv`, `SAGEConv`, `GraphConv`, `Data`, `Batch`, plus the `data` / `nn` submodules.

### `ltgeometric.GCNConv` / `ltgeometric.SAGEConv` / `ltgeometric.GraphConv`

- **Signature:** `GCNConv(in_channels, out_channels, bias=True)` (`nn.py:42`) · `SAGEConv(in_channels, out_channels, bias=True)` (`nn.py:59`) · `GraphConv(in_channels, out_channels, bias=True)` (`nn.py:79`); `forward(x, edge_index) -> Tensor`
- **Parameters:** `in_channels` / `out_channels` — feature dimensions. `forward(x, edge_index)` — `x` is the node-feature matrix `[num_nodes, in_channels]`; `edge_index` is the edge list.
- **Returns:** the updated node features `[num_nodes, out_channels]`. Verified: `GCNConv(4, 4).forward(x, adj)` with `x` of shape `[1, 4]` and a `[4, 4]` adjacency returns shape `[1, 4]`.
- **PyTorch:** `torch_geometric.nn.GCNConv` / `SAGEConv` — same names and the same `forward(x, edge_index)` shape.

### `ltgeometric.GATConv`

- **Signature:** `GATConv(in_channels, out_channels, heads=1, concat=True, ...)` (`nn.py:99`); `forward(x, edge_index) -> Tensor`
- **Parameters:** `heads` — number of attention heads, default 1. `concat` — concatenate head outputs (default True) or average them.
- **Returns:** node features `[num_nodes, out_channels * heads]` when `concat=True`, else `[num_nodes, out_channels]`.
- **PyTorch:** `torch_geometric.nn.GATConv` — same signature.

### `ltgeometric.Data` / `ltgeometric.Batch`

- **Signature:** `Data(x=None, edge_index=None, y=None, ...)`; `Batch.from_data_list(list_of_data) -> Batch`
- **Returns:** `Data` holds one graph (node features, edges, labels); `Batch` collates a list of graphs into one batched graph (with a `batch` vector assigning nodes to graphs).
- **PyTorch:** `torch_geometric.data.Data` / `Batch` — same roles.

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

6 names: `DLRM`, `EmbeddingBag`, `FactorizationMachine`, `MLP`, `negative_sampling`, plus the `rec` submodule.

### `ltrec.DLRM`

- **Signature:** `DLRM(num_embeddings_list, embedding_dim, dense_dim, bottom_dims=(), top_dims=())` (`rec.py:135`); `forward(dense_x, sparse_inputs) -> Tensor`
- **Source:** `python/ltrec/rec.py:135-155`
- **Parameters:** `num_embeddings_list` — a **list** of embedding-table sizes, one per sparse feature. `embedding_dim` — int. `dense_dim` — int, the dense-feature dimension. `bottom_dims` / `top_dims` — tuples of MLP hidden sizes.
- **Returns:** a click-through-rate prediction `[batch, 1]`.
- **Edge cases:** the first argument is a list — `DLRM([10, 8], 4, 8)` is correct; `DLRM(4, 4, [4])` raises `TypeError: 'int' object is not iterable` inside `nn.Embedding`. `forward` takes the dense features and a list of sparse index tensors.
- **PyTorch:** `torchrec.models.dlrm.DLRM` — same architecture; torch's takes embedding configs, LiteTorch's takes plain sizes.

### `ltrec.FactorizationMachine`

- **Signature:** `FactorizationMachine(input_dim, factor_dim)` (`rec.py:85`); `forward(x) -> Tensor`
- **Parameters:** `input_dim` — feature count. `factor_dim` — latent-factor size.
- **Returns:** `[batch, 1]` — the FM prediction (linear term + pairwise interactions).
- **PyTorch:** no `torch` equivalent; this is the classic Rendle FM.

### `ltrec.EmbeddingBag`

- **Signature:** `EmbeddingBag(num_embeddings, embedding_dim, mode="sum")` (`rec.py:37`); `forward(input, offsets) -> Tensor`
- **Parameters:** `mode` — `"sum"` (default), `"mean"`, or `"max"`.
- **Returns:** the pooled embedding `[batch, embedding_dim]`.
- **PyTorch:** `torch.nn.EmbeddingBag` — same signature and modes.

### `ltrec.MLP` / `ltrec.negative_sampling`

- **Signature:** `MLP(dims)` (`rec.py:109`); `forward(x) -> Tensor`; `negative_sampling(edge_index, num_nodes, num_neg_samples=None, force_undirected=False) -> Tensor`
- **Parameters:** `MLP` takes a list of layer sizes, e.g. `MLP([8, 16, 1])` (ReLU between layers, no activation on the last).
- **Returns:** `MLP` returns the transformed tensor; `negative_sampling` takes an edge-index tensor of shape `[2, num_edges]` (first row sources, second row destinations) and returns a tensor of sampled negative edges, for training a recommender.
- **PyTorch:** no direct equivalent; `MLP` is a plain helper.

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

11 names: `Env`, `GridWorldEnv`, `BanditEnv`, `ReplayBuffer`, `DQN`, `PPO`, `REINFORCE`, `compute_returns`, plus the `env` / `buffer` / `agents` submodules.

### `ltrrl.Env` / `ltrrl.GridWorldEnv` / `ltrrl.BanditEnv`

- **Signature:** `Env()` (base — `reset()` / `step(action)` raise `NotImplementedError`) · `GridWorldEnv()` · `BanditEnv(probs)`
- **Source:** `python/ltrrl/env.py`
- **Returns:** `reset()` returns the initial observation (a list of floats — the one-hot grid position for `GridWorldEnv`, `[0.0]` for `BanditEnv`). `step(action)` returns `(observation, reward, done, info)`. Verified: `GridWorldEnv().reset()` returns a 16-element one-hot list; `step(0)` returns `(obs, -0.01, False, {})`; `BanditEnv(probs).reset()` returns `[0.0]` and `step` returns `([0.0], reward, True, {})`.
- **PyTorch:** `gym.Env` — same `reset` / `step` contract (observation, reward, done, info).

### `ltrrl.ReplayBuffer`

- **Signature:** `ReplayBuffer(capacity)` (`buffer.py:8`); `push(obs, action, reward, next_obs, done)` · `sample(batch_size) -> list` · `__len__()`
- **Parameters:** `capacity` — int, maximum stored transitions (oldest dropped first).
- **Returns:** `push` stores one transition; `sample` returns a random batch of `batch_size` transitions.
- **PyTorch:** no `torch` equivalent; this is the standard DQN replay buffer.

### `ltrrl.DQN` / `ltrrl.PPO` / `ltrrl.REINFORCE`

- **Signature:** `DQN(obs_dim, n_actions, hidden=128, lr=1e-3, gamma=0.99)` (`agents.py:69`) · `PPO(obs_dim, n_actions, hidden=128, lr=3e-4, clip_eps=0.2)` (`agents.py:109`) · `REINFORCE(obs_dim, n_actions, hidden=128, lr=1e-3)` (`agents.py:159`)
- **Parameters:** observation/action dimensions plus hyperparameters. (`agents.py:30` defines a helper `MLP` used by the agents, not an agent itself.)
- **Returns:** agents with `act(obs)` / `update(batch)`-style methods (see `agents.py` for the exact method list).
- **PyTorch:** no `torch` equivalent; these are from-scratch RL agents on LiteTorch modules.

### `ltrrl.compute_returns`

- **Signature:** `compute_returns(rewards, gamma=0.99) -> list[float]`
- **Parameters:** `rewards` — list of per-step rewards. `gamma` — discount, default 0.99.
- **Returns:** the discounted return from each step.
- **PyTorch:** no direct equivalent.

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
