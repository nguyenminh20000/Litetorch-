import litetorch as lt
from litetorch import nn
import os
import tempfile

print("=== SafeTensors save/load test ===")

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "test.safetensors")

print("\n--- Test 1: FP32 tensors ---")
lt.manual_seed(42)
a = lt.randn([4, 8], lt.Device("cpu"), False)
b = lt.randn([8], lt.Device("cpu"), False)
lt.save_safetensors({"weight": a, "bias": b}, path)
print(f"Saved to {path}, size={os.path.getsize(path)} bytes")

loaded = lt.load_safetensors(path)
assert set(loaded.keys()) == {"weight", "bias"}, f"Keys mismatch: {loaded.keys()}"
assert list(loaded["weight"].shape) == [4, 8]
assert list(loaded["bias"].shape) == [8]
av = a.to_vector()
lv = loaded["weight"].to_vector()
assert av == lv, "FP32 weight values mismatch"
bv = b.to_vector()
blv = loaded["bias"].to_vector()
assert bv == blv, "FP32 bias values mismatch"
print("FP32 save/load OK")

print("\n--- Test 2: FP16 tensors ---")
path16 = os.path.join(tmpdir, "test16.safetensors")
c = lt.randn([16, 32], lt.Device("cpu"), False, lt.DataType.FP16)
lt.save_safetensors({"fp16_weight": c}, path16)
loaded16 = lt.load_safetensors(path16)
assert loaded16["fp16_weight"].dtype == lt.DataType.FP16, "dtype mismatch"
cv = c.to_vector()
clv = loaded16["fp16_weight"].to_vector()
assert len(cv) == len(clv) == 512
maxdiff = max(abs(x - y) for x, y in zip(cv, clv))
print(f"FP16 max diff: {maxdiff}")
assert maxdiff < 1e-3, f"FP16 values mismatch: {maxdiff}"
print("FP16 save/load OK")

print("\n--- Test 3: HF safetensors library compat ---")
try:
    from safetensors.numpy import load_file
    import numpy as np
    data = load_file(path)
    assert "weight" in data and "bias" in data
    assert data["weight"].shape == (4, 8)
    assert data["bias"].shape == (8,)
    np.testing.assert_allclose(data["weight"], np.array(av, dtype=np.float32).reshape(4, 8))
    print("HF safetensors library reads our file OK")
except ImportError:
    print("safetensors pip package not installed, skipping compat check")

print("\n--- Test 4: Module save_pretrained / load_pretrained ---")
model = nn.Linear(16, 8, True)
lt.manual_seed(123)
for p in model.parameters():
    w = lt.randn(list(p.shape), lt.Device("cpu"), False)
    p.to_vector()
orig_vals = [p.to_vector() for p in model.parameters()]

mpath = os.path.join(tmpdir, "model.safetensors")
model.save_pretrained(mpath)
print(f"Saved model to {mpath}")

model2 = nn.Linear(16, 8, True)
model2.load_pretrained(mpath)
new_vals = [p.to_vector() for p in model2.parameters()]
for ov, nv in zip(orig_vals, new_vals):
    assert ov == nv, "Module param mismatch after load"
print("Module save/load_pretrained OK")

print("\n--- Test 5: Sharded save_pretrained ---")
shard_dir = os.path.join(tmpdir, "shards")
model.save_pretrained(shard_dir, num_shards=2)
files = sorted(os.listdir(shard_dir))
print(f"Shard files: {files}")
assert any("model-00001-of-00002.safetensors" in f for f in files)
assert any("model-00002-of-00002.safetensors" in f for f in files)

model3 = nn.Linear(16, 8, True)
model3.load_pretrained(shard_dir)
shard_vals = [p.to_vector() for p in model3.parameters()]
for ov, sv in zip(orig_vals, shard_vals):
    assert ov == sv, "Sharded param mismatch after load"
print("Sharded save/load_pretrained OK")

print("\n=== ALL SAFETENSORS TESTS PASSED ===")
