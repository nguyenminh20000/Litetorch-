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
opt = lt.optim.Adam(emb.parameters() + cnn.parameters(), lr=1e-3)

for step in range(10):
    x_idx = np.random.randint(0, VOCAB, (BATCH, SEQ_LEN)).astype(np.int64)
    xl = lt.Tensor.from_vector([float(v) for v in x_idx.reshape(-1)], [BATCH, SEQ_LEN], dev)
    e = emb.forward(xl)
    u = lt.Ops.unsqueeze(e, 1)
    out = cnn.forward(u)
    yt = to_lt(np.random.randint(0, 2, BATCH).astype(np.float32), False)
    opt.zero_grad()
    loss = lt.Ops.cross_entropy_loss(out, yt)
    loss.backward()
    opt.step()
    lv = float(np.array(loss.to(cpu).to_vector())[0])
    print(f"step {step}: loss={lv:.4f}", flush=True)

print("TEXT ADAM 10 STEPS PASS")
