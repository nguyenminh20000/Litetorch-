import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x_train = lt.Tensor.from_vector([
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    5.0, 6.0, 7.0, 8.0,
    8.0, 7.0, 6.0, 5.0
], [4, 4], device, False)

y_train = lt.Tensor.from_vector([
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0
], [4, 2], device, False)

dataset = lt.data.TensorDataset(x_train, y_train)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=True, device=device)

for epoch in range(10):
    loader.reset()
    total_loss = 0.0
    steps = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        
        optimizer.zero_grad()
        preds = model.forward(bx)
        loss = loss_fn.forward(preds, by)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
    
    avg_loss = total_loss / max(steps, 1)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
