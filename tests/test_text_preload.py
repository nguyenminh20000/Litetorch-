import litetorch as lt
import numpy as np

np.random.seed(0)
dev = lt.Device("gpu:0")
cpu = lt.Device("cpu")

VOCAB, EMB_DIM, SEQ_LEN, BATCH = 20000, 128, 256, 64
N_FILTERS = 100
N_BATCHES, STEPS = 20, 100

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
opt = lt.optim.Adam(emb.parameters() + cnn.parameters(), lr=1e-3)

print("preloading...", flush=True)
batches = []
for _ in range(N_BATCHES):
    x_idx = np.random.randint(0, VOCAB, (BATCH, SEQ_LEN)).astype(np.int64)
    xl = lt.Tensor.from_vector([float(v) for v in x_idx.reshape(-1)], [BATCH, SEQ_LEN], dev)
    yl = to_lt(np.random.randint(0, 2, BATCH).astype(np.float32), False)
    batches.append((xl, yl))
print(f"preloaded {len(batches)}", flush=True)

for step in range(STEPS):
    xl, yl = batches[step % N_BATCHES]
    e = emb.forward(xl)
    u = lt.Ops.unsqueeze(e, 1)
    out = cnn.forward(u)
    opt.zero_grad()
    loss = lt.Ops.cross_entropy_loss(out, yl)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        lv = float(np.array(loss.to(cpu).to_vector())[0])
        print(f"step {step}: loss={lv:.4f}", flush=True)

print("TEXT PRELOAD 100 STEPS PASS")
