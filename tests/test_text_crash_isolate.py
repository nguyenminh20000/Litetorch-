import litetorch as lt
import torch
import time

def sync():
    torch.cuda.synchronize()

dev = lt.Device("gpu:0")
print("Step 1: Create Embedding", flush=True)
emb = lt.nn.Embedding(20000, 128)
print("Step 2: emb.to(dev)", flush=True)
emb.to(dev)
print("Step 3: emb.weight = emb.weight.to(dev)", flush=True)
emb.weight = emb.weight.to(dev)
print(f"weight device: {emb.weight.device}", flush=True)
sync()
print("Step 4: sync OK after embedding", flush=True)

print("Step 5: Create 10 tensors via from_vector", flush=True)
tensors = []
for i in range(10):
    xt = lt.Tensor.from_vector([float(j % 20000) for j in range(64*256)], [64, 256], dev)
    tensors.append(xt)
    if i % 5 == 0:
        print(f"  created {i+1}/10", flush=True)
sync()
print("Step 6: sync OK after 10 from_vector", flush=True)

print("Step 7: Embedding forward", flush=True)
y = emb.forward(tensors[0])
sync()
print(f"Step 8: forward OK, y shape {y.shape}", flush=True)

print("ISOLATE TEST PASS", flush=True)
