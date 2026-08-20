#!/usr/bin/env python3
import litetorch as lt
import sys

def main():
    print("==================================================")
    print("      LITETORCH PYTHON BINDINGS TEST SUITE")
    print("==================================================")

    device = lt.auto_device()
    print(f"[1] Device initialized: {device}")

    x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
    y = lt.Tensor.from_vector([2.0, 0.5, 1.0, 2.0], [2, 2], device, True)

    z = lt.Ops.add(x, y)
    print(f"[2] Tensor addition result vector: {z.to_vector()}")

    loss = lt.Ops.sum(z)
    loss.backward()

    print(f"[3] Loss sum item: {loss.item()}")
    print(f"[4] Gradient of x: {x.grad.to_vector()}")
    assert x.grad is not None, "Gradient of x should not be None"
    assert x.grad.to_vector() == [1.0, 1.0, 1.0, 1.0], "Gradient values mismatch"

    linear = lt.nn.Linear(2, 1, True)
    optimizer = lt.optim.AdamW(linear.parameters(), lr=0.01)

    inp = lt.Tensor.from_vector([1.0, 2.0], [1, 2], device, False)
    target = lt.Tensor.from_vector([5.0], [1, 1], device, False)

    out = linear.forward(inp)
    mse = lt.Ops.mse_loss(out, target)
    
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

    print(f"[5] Linear Forward & AdamW Step Success. Loss: {mse.item():.6f}")

    def custom_fn(input_tensor):
        h1 = lt.Ops.mul(input_tensor, input_tensor)
        return lt.Ops.add(h1, input_tensor)

    x_cp = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], device, True)
    cp_out = lt.checkpoint(custom_fn, x_cp)
    cp_loss = lt.Ops.sum(cp_out)
    cp_loss.backward()

    print(f"[6] Python Checkpoint Grad vector: {x_cp.grad.to_vector()}")
    assert x_cp.grad is not None, "Checkpointed gradient should not be None"

    print("==================================================")
    print("   ALL PYTHON TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
