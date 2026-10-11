import os

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

from . import io as _io


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SpeechCommands(Dataset):
    LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off",
              "stop", "go", "zero", "one", "two", "three", "four", "five",
              "six", "seven", "eight", "nine", "backward", "bed", "bird",
              "cat", "dog", "follow", "forward", "happy", "house", "learn",
              "marvin", "sheila", "tree", "visual", "wow"]

    def __init__(self, root, subset="training", transform=None, download=False):
        self.root = root
        self.subset = subset
        self.transform = transform
        self.label_to_idx = {l: i for i, l in enumerate(self.LABELS)}
        self.samples = []
        if not os.path.isdir(root):
            if download:
                raise RuntimeError("SpeechCommands download not implemented; place wav files under root/<label>/*.wav")
            return
        for label in self.LABELS:
            d = os.path.join(root, label)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".wav"):
                    self.samples.append((os.path.join(d, f), self.label_to_idx[label]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        waveform, sr = _io.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, sr, target

    def __len__(self):
        return len(self.samples)


class WaveformDataset(Dataset):
    def __init__(self, root, transform=None, extensions=(".wav",)):
        self.root = root
        self.transform = transform
        self.files = []
        for dirpath, _, filenames in os.walk(root):
            for f in sorted(filenames):
                if f.lower().endswith(extensions):
                    self.files.append(os.path.join(dirpath, f))

    def __getitem__(self, index):
        path = self.files[index]
        waveform, sr = _io.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, sr

    def __len__(self):
        return len(self.files)
