import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

raw_linear = lt.nn.Linear(512, 256, True)
raw_linear.to(device)

qat_layer = lt.quantization.QATLinear(
    raw_linear,
    w_scale=0.005,
    w_zp=0.0,
    a_scale=0.01,
    a_zp=0.0,
    b=8
)

x = lt.Tensor.from_vector([0.5] * (4 * 512), [4, 512], device, True)
out = qat_layer.forward(x)
loss = lt.Ops.sum(out)
loss.backward()

print("QAT Output Shape:", out.shape)
print("Weight grad:", qat_layer.weight.grad.to_vector()[:5])
