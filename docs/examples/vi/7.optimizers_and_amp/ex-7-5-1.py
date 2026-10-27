import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(128, 256, True),
    lt.nn.GELU(),
    lt.nn.Linear(256, 10, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=1e-3)
scaler = lt.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
loss_fn = lt.nn.CrossEntropyLoss()

x = lt.Tensor.from_vector([0.5] * (32 * 128), [32, 128], device, False)
y = lt.Tensor.from_vector([1.0] * (32 * 10), [32, 10], device, False)

for step in range(5):
    optimizer.zero_grad()
    
    with lt.amp.AutocastGuard(True, lt.DataType.FP16):
        preds = model.forward(x)
        loss = loss_fn.forward(preds, y)
        scaled_loss = scaler.scale(loss)
    
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Step {step + 1}, Loss: {loss.item():.4f}, Current Scale: {scaler.get_scale()}")
