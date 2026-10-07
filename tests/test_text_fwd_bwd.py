import litetorch as lt
import numpy as np

np.random.seed(0)
dev = lt.Device("gpu:0")
cpu = lt.Device("cpu")

VOCAB, EMB_DIM, SEQ_LEN, BATCH = 20000, 128, 256, 64
N_FILTERS = 100

def to_lt(arr, req=True):
    v = np.ascontiguousarray(arr, dtype=np.float32).ravel().tolist()
    return lt.Tensor.from_vector(v, list(arr.shape), dev, req)

emb = lt.nn.Embedding(VOCAB, EMB_DIM)
emb.to(dev)
cnn = lt.nn.Sequential([
    lt.nn.Conv2d(1, N_FILTERS, 3, padding=1),
    lt.nn.ReLU(),
    lt.nn.AdaptiveAvgPool2d(1, 1),
    lt.nn.Flatten(),
    lt.nn.Linear(N_FILTERS, 2),
])
cnn.to(dev)

x_idx = np.random.randint(0, VOCAB, (BATCH, SEQ_LEN)).astype(np.int64)
xl = lt.Tensor.from_vector([float(v) for v in x_idx.reshape(-1)], [BATCH, SEQ_LEN], dev)
print("forward embedding...", flush=True)
e = emb.forward(xl)
print(f"emb out shape: {e.shape}", flush=True)
u = lt.Ops.unsqueeze(e, 1)
print(f"unsqueeze shape: {u.shape}", flush=True)
print("forward cnn...", flush=True)
out = cnn.forward(u)
print(f"cnn out shape: {out.shape}", flush=True)
print("backward...", flush=True)
yt = to_lt(np.random.randint(0, 2, BATCH).astype(np.float32), False)
loss = lt.Ops.cross_entropy_loss(out, yt)
loss.backward()
print("backward done", flush=True)
lv = float(np.array(loss.to(cpu).to_vector())[0])
print(f"loss={lv:.4f}")
print("TEXT FWD/BWD PASS")
