import math
import os

import litetorch as lt

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


def make_grid(tensors, nrow=8, padding=2, normalize=False, value_range=None, pad_value=0):
    if not _NP:
        raise ImportError("numpy is required for make_grid")
    if isinstance(tensors, lt.Tensor):
        shape = list(tensors.shape)
        if len(shape) == 4:
            n = shape[0]
            vecs = np.array(tensors.to_vector(), dtype=np.float32).reshape(shape)
            tensors = [lt.Tensor.from_vector(vecs[i].reshape(-1).tolist(), shape[1:], lt.Device("cpu"))
                       for i in range(n)]
        else:
            tensors = [tensors]
    imgs = []
    for t in tensors:
        s = list(t.shape)
        if len(s) == 2:
            arr = np.array(t.to_vector(), dtype=np.float32).reshape(1, s[0], s[1])
        elif len(s) == 3:
            arr = np.array(t.to_vector(), dtype=np.float32).reshape(s)
        else:
            raise ValueError("make_grid expects 2D/3D tensors or 4D batch")
        imgs.append(arr)
    c = imgs[0].shape[0]
    h, w = imgs[0].shape[1], imgs[0].shape[2]
    n = len(imgs)
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    gh = nrows * h + (nrows + 1) * padding
    gw = ncol * w + (ncol + 1) * padding
    grid = np.full((c, gh, gw), pad_value, dtype=np.float32)
    for i, img in enumerate(imgs):
        r, col = divmod(i, ncol)
        y = r * (h + padding) + padding
        x = col * (w + padding) + padding
        grid[:, y:y + h, x:x + w] = img
    if normalize:
        if value_range is not None:
            lo, hi = value_range
        else:
            lo, hi = float(grid.min()), float(grid.max())
        if hi > lo:
            grid = (grid - lo) / (hi - lo)
    return lt.Tensor.from_vector(grid.reshape(-1).tolist(), [c, gh, gw], lt.Device("cpu"))


def save_image(tensor, fp, nrow=8, padding=2, normalize=False, value_range=None, pad_value=0):
    if not _PIL:
        raise ImportError("PIL is required for save_image")
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize,
                     value_range=value_range, pad_value=pad_value)
    shape = list(grid.shape)
    c, h, w = shape
    arr = np.array(grid.to_vector(), dtype=np.float32).reshape(c, h, w)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if c == 1:
        img = Image.fromarray(arr[0], mode="L")
    elif c == 3:
        img = Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
    elif c == 4:
        img = Image.fromarray(arr.transpose(1, 2, 0), mode="RGBA")
    else:
        raise ValueError("save_image supports 1/3/4 channels")
    if isinstance(fp, str):
        os.makedirs(os.path.dirname(os.path.abspath(fp)), exist_ok=True)
    img.save(fp)
    return img
