import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

DATA_DIR = "/content/data/shakespeare"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

SEQ_LEN = 32
BATCH = 32
EMB_DIM = 64
N_HEAD = 4
N_LAYER = 2
EPOCHS = 2
LR = 1e-3


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print("downloading tinyshakespeare...", flush=True)
        urllib.request.urlretrieve(URL, path)
    return open(path).read()


def build_vocab(text):
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, chars


def get_batches(ids, batch, seq_len):
    n = len(ids) - seq_len - 1
    idx = np.arange(n)
    np.random.shuffle(idx)
    for i in range(0, n, batch):
        bi = idx[i:i+batch]
        xb = np.stack([ids[j:j+seq_len] for j in bi])
        yb = np.stack([ids[j+1:j+seq_len+1] for j in bi])
        yield xb, yb


class TorchTransformer(tnn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = tnn.Embedding(vocab, EMB_DIM)
        self.pos = tnn.Embedding(SEQ_LEN, EMB_DIM)
        self.layers = tnn.ModuleList([
            tnn.TransformerDecoderLayer(EMB_DIM, N_HEAD, EMB_DIM*4, batch_first=True)
            for _ in range(N_LAYER)
        ])
        self.fc = tnn.Linear(EMB_DIM, vocab)

    def forward(self, x):
        b, s = x.shape
        p = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(p)
        for l in self.layers:
            h = l(h)
        return self.fc(h)


def build_lt(vocab):
    dev = lt.Device("gpu:0")
    emb = lt.nn.Embedding(vocab, EMB_DIM)
    pos = lt.nn.Embedding(SEQ_LEN, EMB_DIM)
    layers = [lt.nn.TransformerDecoderLayer(EMB_DIM, N_HEAD, EMB_DIM*4) for _ in range(N_LAYER)]
    fc = lt.nn.Linear(EMB_DIM, vocab)
    for m in [emb, pos, fc] + layers:
        m.to(dev)
    return emb, pos, layers, fc, dev


def lt_forward(emb, pos, layers, fc, x, dev):
    b, s = x.shape
    pv = list(range(s))
    p = lt.Tensor.from_vector([float(v) for v in pv], [s], dev)
    p = lt.Ops.unsqueeze(p, 0)
    h = lt.Ops.add(emb.forward(x), pos.forward(p))
    for l in layers:
        h = l.forward(h)
    return fc.forward(h)


def main():
    text = download()
    stoi, itos = build_vocab(text)
    vocab = len(stoi)
    print(f"vocab={vocab} chars={len(text)}", flush=True)
    ids = np.array([stoi[c] for c in text], dtype=np.int64)

    dev = lt.Device("gpu:0")
    emb, pos, layers, fc, _ = build_lt(vocab)
    params = emb.parameters() + pos.parameters() + fc.parameters()
    for l in layers:
        params += l.parameters()
    opt = lt.optim.Adam(params, lr=LR)

    n_tok = 0
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        te = time.perf_counter()
        for xb, yb in get_batches(ids, BATCH, SEQ_LEN):
            b = xb.shape[0]
            xt = lt.Tensor.from_vector([float(v) for v in xb.reshape(-1)], [b, SEQ_LEN], dev)
            yt = lt.Tensor.from_vector([float(v) for v in yb.reshape(-1)], [b*SEQ_LEN], dev)
            opt.zero_grad()
            logits = lt_forward(emb, pos, layers, fc, xt, dev)
            logits2d = lt.Ops.reshape(logits, [b*SEQ_LEN, vocab])
            loss = lt.Ops.cross_entropy_loss(logits2d, yt)
            loss.backward()
            opt.step()
            n_tok += b * SEQ_LEN
        dt = time.perf_counter() - te
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s", flush=True)
    total = time.perf_counter() - t0
    print(f"TRANSFORMER RESULT: {total/EPOCHS:.2f}s/epoch, {n_tok} tokens", flush=True)
    print("TRANSFORMER DONE", flush=True)


if __name__ == "__main__":
    main()
