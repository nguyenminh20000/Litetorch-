import os
import struct
import urllib.request
import tarfile
import gzip
import pickle

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

import litetorch as lt


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, loader=None):
        if not _PIL:
            raise ImportError("PIL is required for ImageFolder")
        self.root = root
        self.transform = transform
        self.loader = loader or (lambda p: Image.open(p).convert("RGB"))
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for c in classes:
            d = os.path.join(root, c)
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    self.samples.append((p, self.class_to_idx[c]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


class FakeData(Dataset):
    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10, transform=None):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

    def __getitem__(self, index):
        if not _NP:
            raise ImportError("numpy is required for FakeData")
        c, h, w = self.image_size
        arr = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB") if _PIL else arr
        target = index % self.num_classes
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.size


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


class CIFAR10(Dataset):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"

    def __init__(self, root, train=True, transform=None, download=False):
        if not _NP:
            raise ImportError("numpy is required for CIFAR10")
        self.root = root
        self.train = train
        self.transform = transform
        archive = os.path.join(root, self.filename)
        if download:
            _download(self.url, archive)
        data_dir = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(data_dir):
            if os.path.exists(archive):
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(root)
            else:
                raise RuntimeError("CIFAR10 data not found; pass download=True")
        batches = ["data_batch_%d" % i for i in range(1, 6)] if train else ["test_batch"]
        data, labels = [], []
        for b in batches:
            with open(os.path.join(data_dir, b), "rb") as f:
                d = pickle.load(f, encoding="bytes")
            data.append(d[b"data"])
            labels += d[b"labels"]
        self.data = np.concatenate(data, axis=0).reshape(-1, 3, 32, 32)
        self.targets = labels

    def __getitem__(self, index):
        arr = self.data[index].transpose(1, 2, 0)
        img = Image.fromarray(arr, mode="RGB") if _PIL else arr
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)


class CIFAR100(CIFAR10):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"

    def __init__(self, root, train=True, transform=None, download=False):
        if not _NP:
            raise ImportError("numpy is required for CIFAR100")
        Dataset.__init__(self)
        self.root = root
        self.train = train
        self.transform = transform
        archive = os.path.join(root, self.filename)
        if download:
            _download(self.url, archive)
        data_dir = os.path.join(root, "cifar-100-python")
        if not os.path.isdir(data_dir):
            if os.path.exists(archive):
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(root)
            else:
                raise RuntimeError("CIFAR100 data not found; pass download=True")
        name = "train" if train else "test"
        with open(os.path.join(data_dir, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        self.data = d[b"data"].reshape(-1, 3, 32, 32)
        self.targets = d[b"fine_labels"]


class MNIST(Dataset):
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, root, train=True, transform=None, download=False):
        if not _NP:
            raise ImportError("numpy is required for MNIST")
        self.root = root
        self.train = train
        self.transform = transform
        raw = os.path.join(root, "MNIST", "raw")
        if download:
            for f in self.files.values():
                _download(self.base_url + f, os.path.join(raw, f))
        kind = "train" if train else "test"
        images = self._read_images(os.path.join(raw, self.files[kind + "_images"]))
        labels = self._read_labels(os.path.join(raw, self.files[kind + "_labels"]))
        self.data = images
        self.targets = labels

    @staticmethod
    def _read_images(path):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, "bad MNIST image magic %d" % magic
            buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)

    @staticmethod
    def _read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            assert magic == 2049, "bad MNIST label magic %d" % magic
            buf = f.read(n)
        return np.frombuffer(buf, dtype=np.uint8)

    def __getitem__(self, index):
        arr = self.data[index]
        img = Image.fromarray(arr, mode="L") if _PIL else arr
        target = int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)
        idx = list(range(n))
        if self.shuffle:
            import random
            random.shuffle(idx)
        batch = []
        for i in idx:
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:
                yield _collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield _collate(batch)

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size and not self.drop_last:
            n += 1
        return n


def _collate(batch):
    imgs, targets = zip(*batch)
    if isinstance(imgs[0], lt.Tensor):
        if not _NP:
            raise ImportError("numpy is required for batch collate")
        shape = list(imgs[0].shape)
        vecs = [t.to_vector() for t in imgs]
        flat = [v for vec in vecs for v in vec]
        stacked = lt.Tensor.from_vector(flat, [len(batch)] + shape, lt.Device("cpu"))
        t_targets = lt.Tensor.from_vector([float(t) for t in targets], [len(batch)], lt.Device("cpu"))
        return stacked, t_targets
    return list(imgs), list(targets)
