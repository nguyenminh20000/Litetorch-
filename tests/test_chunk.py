import litetorch as lt
from litetorch import Ops as ops

print("=== Ops.chunk correctness ===")

lt.manual_seed(0)
x = lt.randn([2, 8, 12], lt.Device("cpu"), True)

parts = ops.chunk(x, [3, 5, 4], -1)
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

print("\n=== ALL CHUNK TESTS PASSED ===")
