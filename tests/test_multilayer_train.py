import litetorch as lt
import numpy as np

np.random.seed(0)
dev = lt.Device("gpu:0")
cpu = lt.Device("cpu")

def to_lt(arr, req=True):
    v = np.ascontiguousarray(arr, dtype=np.float32).ravel().tolist()
    return lt.Tensor.from_vector(v, list(arr.shape), dev, req)

def to_np(t):
    return np.array(t.to(cpu).to_vector(), dtype=np.float32).reshape(t.shape)

N, C, H, W = 16, 3, 8, 8
X = to_lt(np.random.randn(N, C, H, W).astype(np.float32), False)
Y = np.random.randint(0, 2, N)

w1 = to_lt(np.random.randn(8, 3, 3, 3).astype(np.float32) * 0.1)
b1 = to_lt(np.zeros(8, dtype=np.float32))
w2 = to_lt(np.random.randn(2, 8, 4, 4).astype(np.float32).reshape(2, 8, 4, 4) * 0.1)
b2 = to_lt(np.zeros(2, dtype=np.float32))
params = [w1, b1, w2, b2]
lr = 0.01

for epoch in range(5):
    c1 = lt.Ops.conv2d(X, w1, b1, 1, 1)
    r1 = lt.Ops.relu(c1)
    c2 = lt.Ops.conv2d(r1, w2, b2, 1, 0)
    logits = to_np(c2).reshape(N, 2)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(N), Y] + 1e-9).mean()
    pred = p.argmax(1)
    acc = (pred == Y).mean()
    g = (p.copy())
    g[np.arange(N), Y] -= 1
    g /= N
    gl = to_lt(g.reshape(N, 2, 1, 1), False)
    c2.backward(gl)
    for prm in params:
        gp = to_np(prm.grad)
        nv = to_np(prm) - lr * gp
        prm.assign(to_lt(nv, False))
        prm.zero_grad()
    print(f"epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")

print("TRAINING DONE")
