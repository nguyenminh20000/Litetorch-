import math
import random

import litetorch as lt

try:
    from PIL import Image, ImageEnhance, ImageOps
    _PIL = True
except ImportError:
    _PIL = False

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def _to_pil(img):
    if _PIL and isinstance(img, Image.Image):
        return img
    if isinstance(img, lt.Tensor):
        return ToPILImage()(img)
    raise TypeError("expected PIL Image or litetorch Tensor")


def _restore_type(out, img):
    if isinstance(img, lt.Tensor):
        return ToTensor()(out)
    return out


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        return "Compose([%s])" % ", ".join(repr(t) for t in self.transforms)


class ToTensor:
    def __call__(self, img):
        if isinstance(img, lt.Tensor):
            return img
        if not _PIL:
            raise ImportError("PIL is required for ToTensor from image files")
        if not _NP:
            raise ImportError("numpy is required for ToTensor")
        if not isinstance(img, Image.Image):
            raise TypeError("ToTensor expects PIL Image, got %s" % type(img))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
        c, h, w = arr.shape
        return lt.Tensor.from_vector(arr.reshape(-1).tolist(), [c, h, w], lt.Device("cpu"))

    def __repr__(self):
        return "ToTensor()"


class ToPILImage:
    def __call__(self, tensor):
        if _PIL and isinstance(tensor, Image.Image):
            return tensor
        if not _NP:
            raise ImportError("numpy is required for ToPILImage")
        if not isinstance(tensor, lt.Tensor):
            raise TypeError("ToPILImage expects litetorch Tensor, got %s" % type(tensor))
        if not _PIL:
            raise ImportError("PIL is required for ToPILImage")
        shape = list(tensor.shape)
        if len(shape) == 2:
            c, h, w = 1, shape[0], shape[1]
        elif len(shape) == 3:
            c, h, w = shape
        else:
            raise ValueError("ToPILImage expects 2D or 3D tensor")
        arr = np.array(tensor.to_vector(), dtype=np.float32).reshape(c, h, w)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        if c == 1:
            return Image.fromarray(arr[0], mode="L")
        if c == 3:
            return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
        if c == 4:
            return Image.fromarray(arr.transpose(1, 2, 0), mode="RGBA")
        raise ValueError("ToPILImage supports 1/3/4 channels, got %d" % c)

    def __repr__(self):
        return "ToPILImage()"


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def __call__(self, tensor):
        if not isinstance(tensor, lt.Tensor):
            raise TypeError("Normalize expects litetorch Tensor, got %s" % type(tensor))
        if not _NP:
            raise ImportError("numpy is required for Normalize")
        shape = list(tensor.shape)
        c = shape[0] if len(shape) == 3 else shape[-3]
        if len(self.mean) != c or len(self.std) != c:
            raise ValueError("mean/std length must match channels")
        arr = np.array(tensor.to_vector(), dtype=np.float32).reshape(shape)
        if len(shape) == 3:
            for i in range(c):
                arr[i] = (arr[i] - self.mean[i]) / self.std[i]
        else:
            for i in range(c):
                arr[:, i] = (arr[:, i] - self.mean[i]) / self.std[i]
        out = lt.Tensor.from_vector(arr.reshape(-1).tolist(), shape, lt.Device("cpu"))
        if self.inplace:
            tensor.copy_(out)
            return tensor
        return out

    def __repr__(self):
        return "Normalize(mean=%s, std=%s)" % (self.mean, self.std)


class Resize:
    def __init__(self, size, interpolation="bilinear"):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img):
        pil = _to_pil(img)
        if not _PIL:
            raise ImportError("PIL is required for Resize")
        resample = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST,
                    "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}.get(self.interpolation, Image.BILINEAR)
        out = pil.resize((self.size[1], self.size[0]), resample)
        return _restore_type(out, img)

    def __repr__(self):
        return "Resize(size=%s)" % (self.size,)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, img):
        pil = _to_pil(img)
        th, tw = self.size
        w, h = pil.size
        top = max(0, (h - th) // 2)
        left = max(0, (w - tw) // 2)
        return _restore_type(pil.crop((left, top, left + tw, top + th)), img)

    def __repr__(self):
        return "CenterCrop(size=%s)" % (self.size,)


class RandomCrop:
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def __call__(self, img):
        pil = _to_pil(img)
        if self.padding > 0:
            pil = ImageOps.expand(pil, border=self.padding)
        th, tw = self.size
        w, h = pil.size
        if self.pad_if_needed:
            if w < tw:
                pil = ImageOps.expand(pil, border=(tw - w, 0))
                w = tw
            if h < th:
                pil = ImageOps.expand(pil, border=(0, th - h))
                h = th
        if w < tw or h < th:
            raise ValueError("image smaller than crop size")
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        return _restore_type(pil.crop((left, top, left + tw, top + th)), img)

    def __repr__(self):
        return "RandomCrop(size=%s)" % (self.size,)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        pil = _to_pil(img)
        if random.random() < self.p:
            pil = ImageOps.mirror(pil)
        return _restore_type(pil, img)

    def __repr__(self):
        return "RandomHorizontalFlip(p=%s)" % self.p


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        pil = _to_pil(img)
        if random.random() < self.p:
            pil = ImageOps.flip(pil)
        return _restore_type(pil, img)

    def __repr__(self):
        return "RandomVerticalFlip(p=%s)" % self.p


class Grayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        pil = _to_pil(img).convert("L")
        if self.num_output_channels == 3:
            pil = pil.convert("RGB")
        return _restore_type(pil, img)

    def __repr__(self):
        return "Grayscale(num_output_channels=%d)" % self.num_output_channels


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        pil = _to_pil(img)
        if self.brightness > 0:
            f = 1.0 + random.uniform(-self.brightness, self.brightness)
            pil = ImageEnhance.Brightness(pil).enhance(f)
        if self.contrast > 0:
            f = 1.0 + random.uniform(-self.contrast, self.contrast)
            pil = ImageEnhance.Contrast(pil).enhance(f)
        if self.saturation > 0:
            f = 1.0 + random.uniform(-self.saturation, self.saturation)
            pil = ImageEnhance.Color(pil).enhance(f)
        if self.hue > 0:
            shift = random.uniform(-self.hue, self.hue)
            pil = _shift_hue(pil, shift)
        return _restore_type(pil, img)

    def __repr__(self):
        return "ColorJitter(brightness=%s, contrast=%s, saturation=%s, hue=%s)" % (
            self.brightness, self.contrast, self.saturation, self.hue)


def _shift_hue(pil, shift):
    if not _NP:
        return pil
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    hsv = _rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
    rgb = _hsv_to_rgb(hsv)
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), mode="RGB")


def _rgb_to_hsv(arr):
    cmax = arr.max(-1)
    cmin = arr.min(-1)
    d = cmax - cmin
    h = np.zeros_like(cmax)
    m = d != 0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    idx = m & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = m & (cmax == g)
    h[idx] = (b[idx] - r[idx]) / d[idx] + 2
    idx = m & (cmax == b)
    h[idx] = (r[idx] - g[idx]) / d[idx] + 4
    h = h / 6.0
    s = np.zeros_like(cmax)
    s[cmax != 0] = d[cmax != 0] / cmax[cmax != 0]
    return np.stack([h, s, cmax], axis=-1)


def _hsv_to_rgb(arr):
    h, s, v = arr[..., 0] * 6.0, arr[..., 1], arr[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    out = np.zeros_like(arr)
    for k, (a, b, c) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m, 0], out[m, 1], out[m, 2] = a[m], b[m], c[m]
    return out


class RandomRotation:
    def __init__(self, degrees, expand=False):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = tuple(degrees)
        self.expand = expand

    def __call__(self, img):
        pil = _to_pil(img)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return _restore_type(pil.rotate(angle, expand=self.expand), img)

    def __repr__(self):
        return "RandomRotation(degrees=%s)" % (self.degrees,)


class Pad:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        pil = _to_pil(img)
        return _restore_type(ImageOps.expand(pil, border=self.padding, fill=self.fill), img)

    def __repr__(self):
        return "Pad(padding=%s)" % (self.padding,)
