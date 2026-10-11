import litetorch as lt
from litetorch import Ops as ops

print("=== Ops.split correctness (torch.split semantics) ===")

lt.manual_seed(0)
x = lt.randn([2, 8, 12], lt.Device("cpu"), True)

parts = ops.split(x, [3, 5, 4], -1)
assert len(parts) == 3
assert list(parts[0].shape) == [2, 8, 3]
assert list(parts[1].shape) == [2, 8, 5]
assert list(parts[2].shape) == [2, 8, 4]
print("Forward shapes OK")

xv = x.to_vector()
p0v = parts[0].to_vector()
assert abs(p0v[0] - xv[0]) < 1e-6
assert abs(p0v[2] - xv[2]) < 1e-6
p1v = parts[1].to_vector()
assert abs(p1v[0] - xv[3]) < 1e-6
p2v = parts[2].to_vector()
assert abs(p2v[0] - xv[8]) < 1e-6
print("Forward values OK (zero-copy views)")

loss = ops.add(ops.add(ops.sum(parts[0]), ops.sum(parts[1])), ops.sum(parts[2]))
loss.backward()
g = x.grad.to_vector()
assert abs(g[0] - 1.0) < 1e-5, f"g[0]={g[0]} expected 1.0"
assert abs(g[3] - 1.0) < 1e-5, f"g[3]={g[3]} expected 1.0"
assert abs(g[8] - 1.0) < 1e-5, f"g[8]={g[8]} expected 1.0"
print("Backward OK (grads routed to correct regions)")

print("\n=== Ops.chunk correctness (torch.chunk semantics) ===")

lt.manual_seed(1)
y = lt.randn([2, 8, 12], lt.Device("cpu"), True)

c = ops.chunk(y, 5, -1)
assert len(c) == 4, f"expected 4 chunks (chunk_size=ceil(12/5)=3), got {len(c)}"
assert [list(p.shape) for p in c] == [[2, 8, 3]] * 4, [list(p.shape) for p in c]
print("Uneven chunk shapes OK (fewer than requested, like torch)")

c2 = ops.chunk(y, 6, 1)
assert len(c2) == 4, f"expected 4 chunks (chunk_size=ceil(8/6)=2), got {len(c2)}"
assert [list(p.shape) for p in c2] == [[2, 2, 12]] * 4
yv = y.to_vector()
cv = c2[2].to_vector()
assert abs(cv[0] - yv[4 * 12]) < 1e-6
print("Dim-1 chunk values OK")

c3 = ops.chunk(y, 20, 0)
assert len(c3) == 2, f"expected 2 chunks (fewer than 20), got {len(c3)}"
assert [list(p.shape) for p in c3] == [[1, 8, 12]] * 2
print("chunks > dim_size OK (fewer chunks returned)")

loss = ops.sum(c[0])
for p in c[1:]:
    loss = ops.add(loss, ops.sum(p))
loss.backward()
gy = y.grad.to_vector()
assert all(abs(v - 1.0) < 1e-5 for v in gy), "chunk backward grads wrong"
print("Chunk backward OK")

print("\n=== ALL CHUNK/SPLIT TESTS PASSED ===")
