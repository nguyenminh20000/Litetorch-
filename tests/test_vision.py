import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import litetorch as lt
from litetorch import nn
from PIL import Image
import numpy as np

import ltvision
from ltvision import transforms as T
from ltvision.datasets import ImageFolder, FakeData, CIFAR10, MNIST, DataLoader
from ltvision.models import resnet18, resnet50, vgg16
from ltvision.utils import make_grid, save_image

print("=== vision transforms ===")
img = Image.new("RGB", (64, 48), color=(255, 0, 0))
t = T.ToTensor()(img)
assert list(t.shape) == [3, 48, 64], t.shape
back = T.ToPILImage()(t)
assert back.size == (64, 48)
print("ToTensor/ToPILImage OK", list(t.shape))

n = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(t)
v = np.array(n.to_vector()).reshape(3, 48, 64)
assert abs(v[0].mean() - 1.0) < 1e-5, v[0].mean()
print("Normalize OK")

r = T.Resize((32, 32))(img)
assert r.size == (32, 32)
c = T.CenterCrop(16)(img)
assert c.size == (16, 16)
rc = T.RandomCrop(16)(img)
assert rc.size == (16, 16)
print("Resize/Crop OK")

fh = T.RandomHorizontalFlip(p=1.0)(img)
assert fh.size == img.size
cj = T.ColorJitter(brightness=0.2, contrast=0.2)(img)
assert cj.size == img.size
rr = T.RandomRotation(15)(img)
assert rr.size == img.size
g = T.Grayscale()(img)
assert g.mode == "L"
print("Flip/Jitter/Rotation/Grayscale OK")

comp = T.Compose([T.Resize(32), T.CenterCrop(16), T.ToTensor()])
out = comp(img)
assert list(out.shape) == [3, 16, 16], out.shape
print("Compose OK")

print("\n=== ImageFolder ===")
tmp = tempfile.mkdtemp()
for cls in ["cat", "dog"]:
    d = os.path.join(tmp, cls)
    os.makedirs(d)
    for i in range(2):
        Image.new("RGB", (16, 16), color=(i * 100, 0, 0)).save(os.path.join(d, "%d.png" % i))
ds = ImageFolder(tmp, transform=T.Compose([T.ToTensor()]))
assert len(ds) == 4, len(ds)
x, y = ds[0]
assert list(x.shape) == [3, 16, 16] and y in (0, 1)
assert ds.classes == ["cat", "dog"]
print("ImageFolder OK", len(ds), ds.classes)

print("\n=== CIFAR10 parser (synthetic) ===")
croot = os.path.join(tempfile.mkdtemp(), "cifar")
os.makedirs(os.path.join(croot, "cifar-10-batches-py"))
import pickle
fake = {b"data": np.random.randint(0, 256, (10, 3072), dtype=np.uint8),
        b"labels": [i % 10 for i in range(10)]}
with open(os.path.join(croot, "cifar-10-batches-py", "test_batch"), "wb") as f:
    pickle.dump(fake, f)
c10 = CIFAR10(croot, train=False, transform=T.ToTensor())
assert len(c10) == 10
x, y = c10[3]
assert list(x.shape) == [3, 32, 32] and y == 3
print("CIFAR10 OK")

print("\n=== MNIST parser (synthetic) ===")
mroot = tempfile.mkdtemp()
rawdir = os.path.join(mroot, "MNIST", "raw")
os.makedirs(rawdir)
import gzip, struct
imgs = np.random.randint(0, 256, (5, 28, 28), dtype=np.uint8)
lbls = np.arange(5, dtype=np.uint8)
with gzip.open(os.path.join(rawdir, "t10k-images-idx3-ubyte.gz"), "wb") as f:
    f.write(struct.pack(">IIII", 2051, 5, 28, 28) + imgs.tobytes())
with gzip.open(os.path.join(rawdir, "t10k-labels-idx1-ubyte.gz"), "wb") as f:
    f.write(struct.pack(">II", 2049, 5) + lbls.tobytes())
mn = MNIST(mroot, train=False, transform=T.ToTensor())
assert len(mn) == 5
x, y = mn[2]
assert list(x.shape) == [1, 28, 28] and y == 2, (x.shape, y)
print("MNIST OK")

print("\n=== DataLoader ===")
fds = FakeData(size=8, image_size=(3, 16, 16), num_classes=4, transform=T.ToTensor())
dl = DataLoader(fds, batch_size=4, shuffle=True)
nb = 0
for bx, by in dl:
    assert list(bx.shape) == [4, 3, 16, 16], bx.shape
    assert list(by.shape) == [4]
    nb += 1
assert nb == 2, nb
print("DataLoader OK")

print("\n=== ResNet18 forward ===")
lt.manual_seed(0)
model = resnet18(num_classes=10)
model.eval()
dev = lt.Device("cpu")
model.to(dev)
x = lt.randn([1, 3, 64, 64], dev, False)
with lt.no_grad():
    y = model(x)
assert list(y.shape) == [1, 10], y.shape
yv = y.to_vector()
assert all(abs(v) < 1e6 for v in yv), "logits exploded"
nparams = sum(p.numel() for p in model.parameters())
print("resnet18 OK out", list(y.shape), "params", nparams)
assert nparams > 10_000_000, nparams

print("\n=== ResNet50 forward ===")
m50 = resnet50(num_classes=10)
m50.eval()
m50.to(dev)
with lt.no_grad():
    y50 = m50(x)
assert list(y50.shape) == [1, 10], y50.shape
print("resnet50 OK")

print("\n=== VGG16 forward ===")
vgg = vgg16(num_classes=10)
vgg.eval()
vgg.to(dev)
with lt.no_grad():
    yvgg = vgg(x)
assert list(yvgg.shape) == [1, 10], yvgg.shape
print("vgg16 OK")

print("\n=== make_grid / save_image ===")
batch = lt.randn([4, 3, 8, 8], dev, False)
grid = make_grid(batch, nrow=2, normalize=True)
assert list(grid.shape) == [3, 22, 22], grid.shape
p = os.path.join(tempfile.mkdtemp(), "grid.png")
save_image(batch, p, nrow=2, normalize=True)
assert os.path.getsize(p) > 0
print("make_grid/save_image OK", list(grid.shape))

print("\n=== ALL VISION TESTS PASSED ===")
