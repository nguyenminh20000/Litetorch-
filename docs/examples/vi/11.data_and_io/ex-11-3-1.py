import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

dataset = lt.data.TensorDataset(x, y)
loader = lt.data.DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    device=device,
    prefetch_limit=2
)

loader.reset()
while True:
    batch = loader.next()
    if batch is None:
        break
    bx, by = batch
    print("Batch x shape:", bx.shape, "Batch y shape:", by.shape)
