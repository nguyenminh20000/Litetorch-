import os
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

DATA_DIR = "/content/data/imdb"
WEIGHT_DIR = "/content/weights"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

BATCH = 64
EPOCHS = 2
LR = 1e-3
VOCAB_SIZE = 20000
SEQ_LEN = 256
EMB_DIM = 128
N_FILTERS = 100


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz = os.path.join(DATA_DIR, "aclImdb_v1.tar.gz")
    out = os.path.join(DATA_DIR, "aclImdb")
    if not os.path.exists(out):
        print("downloading imdb...", flush=True)
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def load_split(split):
    texts, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        d = os.path.join(DATA_DIR, "aclImdb", split, name)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
    return texts, np.array(labels, dtype=np.int64)


def build_vocab(texts):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(VOCAB_SIZE - 2):
        vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab):
    arr = np.zeros((len(texts), SEQ_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)[:SEQ_LEN]]
        arr[i, : len(ids)] = ids
    return arr


def get_data():
    download()
    train_texts, y_train = load_split("train")
    test_texts, y_test = load_split("test")
    print(f"imdb: train {len(train_texts)}, test {len(test_texts)}", flush=True)
    vocab = build_vocab(train_texts)
    print(f"vocab size: {len(vocab)}", flush=True)
    x_train = encode(train_texts, vocab)
    x_test = encode(test_texts, vocab)
    return (x_train, y_train), (x_test, y_test), len(vocab)


def batches(x, y, batch, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), batch):
        j = idx[i : i + batch]
        yield x[j], y[j]


class TorchTextCNN(tnn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = tnn.Embedding(vocab, EMB_DIM, padding_idx=0)
        self.conv = tnn.Conv2d(1, N_FILTERS, 3, padding=1)
        self.pool = tnn.AdaptiveAvgPool2d((1, 1))
        self.fc = tnn.Linear(N_FILTERS, 2)

    def forward(self, x):
        h = self.emb(x).unsqueeze(1)
        h = torch.relu(self.conv(h))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def build_lt(vocab):
    dev = lt.Device("gpu:0")
    emb = lt.nn.Embedding(vocab, EMB_DIM)
    emb.to(dev)
    cnn = lt.nn.Sequential(
        [
            lt.nn.Conv2d(1, N_FILTERS, 3, padding=1),
            lt.nn.ReLU(),
            lt.nn.AdaptiveAvgPool2d(1, 1),
            lt.nn.Flatten(),
            lt.nn.Linear(N_FILTERS, 2),
        ]
    )
    cnn.to(dev)
    return emb, cnn


def lt_forward(emb, cnn, x):
    return cnn.forward(lt.Ops.unsqueeze(emb.forward(x), 1))


def sync():
    torch.cuda.synchronize()


def train_torch(data, vocab):
    (x_train, y_train), (x_test, y_test) = data
    torch.manual_seed(0)
    model = TorchTextCNN(vocab).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = tnn.CrossEntropyLoss()
    n = len(x_train)
    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xb, yb in batches(x_train, y_train, BATCH):
            xt = torch.from_numpy(np.ascontiguousarray(xb)).cuda()
            yt = torch.from_numpy(yb).cuda()
            opt.zero_grad()
            loss = ce(model(xt), yt)
            loss.backward()
            opt.step()
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[torch] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s)", flush=True)
    acc = eval_torch(model, x_test, y_test)
    print(f"[torch] test acc: {acc:.4f}", flush=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "text_torch.pt"))
    return times, acc


@torch.no_grad()
def eval_torch(model, x_test, y_test):
    model.eval()
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        xt = torch.from_numpy(np.ascontiguousarray(xb)).cuda()
        pred = model(xt).argmax(1).cpu().numpy()
        correct += (pred == yb).sum()
        total += len(yb)
    model.train()
    return correct / total


def train_lt(data, vocab):
    (x_train, y_train), (x_test, y_test) = data
    dev = lt.Device("gpu:0")
    emb, cnn = build_lt(vocab)
    opt = lt.optim.Adam(emb.parameters() + cnn.parameters(), lr=LR)
    n = len(x_train)
    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xb, yb in batches(x_train, y_train, BATCH):
            b = len(xb)
            xt = lt.Tensor.from_vector([float(v) for v in xb.reshape(-1)], [b, SEQ_LEN], dev)
            yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
            opt.zero_grad()
            loss = lt.Ops.cross_entropy_loss(lt_forward(emb, cnn, xt), yt)
            loss.backward()
            opt.step()
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s)", flush=True)
    acc = eval_lt(emb, cnn, x_test, y_test, dev)
    print(f"[lt] test acc: {acc:.4f}", flush=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    lt.save_parameters(emb.parameters() + cnn.parameters(), os.path.join(WEIGHT_DIR, "text_lt.bin"))
    return times, acc


def eval_lt(emb, cnn, x_test, y_test, dev):
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        b = len(xb)
        xt = lt.Tensor.from_vector([float(v) for v in xb.reshape(-1)], [b, SEQ_LEN], dev)
        out = lt_forward(emb, cnn, xt)
        pred = np.array(out.to(lt.Device("cpu")).to_vector()).reshape(b, 2).argmax(1)
        correct += (pred == yb).sum()
        total += len(yb)
    sync()
    return correct / total


def main():
    (x_train, y_train), (x_test, y_test), vocab = get_data()
    data = ((x_train, y_train), (x_test, y_test))
    print("== text: pytorch ==", flush=True)
    t_times, t_acc = train_torch(data, vocab)
    print("== text: litetorch ==", flush=True)
    l_times, l_acc = train_lt(data, vocab)
    tt, ll = sum(t_times) / len(t_times), sum(l_times) / len(l_times)
    print(f"TEXT RESULT: torch {tt:.2f}s/epoch acc={t_acc:.4f} | lt {ll:.2f}s/epoch acc={l_acc:.4f} | speedup x{tt/ll:.2f}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
