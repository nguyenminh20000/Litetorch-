import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import litetorch as lt

import ltaudio
from ltaudio import io as aio
from ltaudio.transforms import (Resample, Spectrogram, STFT, InverseSpectrogram,
                                MelSpectrogram, AmplitudeToDB, MFCC)
from ltaudio.datasets import SpeechCommands, WaveformDataset

print("=== wav io ===")
sr = 16000
t = np.arange(sr, dtype=np.float32) / sr
wav = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)]).astype(np.float32)
p = os.path.join(tempfile.mkdtemp(), "tone.wav")
aio.save(p, wav, sr)
data, sr2 = aio.load(p)
assert sr2 == sr, sr2
assert data.shape == (2, sr), data.shape
assert abs(data[0, 100] - wav[0, 100]) < 0.01, (data[0, 100], wav[0, 100])
inf = aio.info(p)
assert inf["sample_rate"] == sr and inf["num_channels"] == 2
print("wav save/load OK", data.shape, sr2)

print("\n=== Resample ===")
r = Resample(sr, 8000)(wav)
assert list(r.shape) == [2, 8000], r.shape
print("Resample OK", list(r.shape))

print("\n=== Spectrogram ===")
spec = Spectrogram(n_fft=256, hop_length=64)(wav)
s = list(spec.shape)
assert s[0] == 2 and s[1] == 129, s
print("Spectrogram OK (C,F,T)", s)

print("\n=== MelSpectrogram ===")
mel = MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=40)(wav)
m = list(mel.shape)
assert m == [2, 40, s[2]], m
print("MelSpectrogram OK (C,F,T)", m)

print("\n=== AmplitudeToDB ===")
db = AmplitudeToDB()(mel)
d = list(db.shape)
assert d == m, d
dv = db.to_vector()
assert all(v == v and abs(v) < 1e9 for v in dv), "non-finite dB values"
print("AmplitudeToDB OK")

print("\n=== MFCC ===")
mfcc = MFCC(sample_rate=sr, n_mfcc=20, n_mels=40, n_fft=256, hop_length=64)(wav)
mc = list(mfcc.shape)
assert mc == [2, 20, s[2]], mc
print("MFCC OK (C,F,T)", mc)

print("\n=== STFT / ISTFT ===")
stft = STFT(n_fft=256, hop_length=64)(wav)
ss = list(stft.shape)
assert ss == [2, 129, ss[2], 2], ss
rec = InverseSpectrogram(n_fft=256, hop_length=64)(stft)
rs = list(rec.shape)
assert rs[0] == 2, rs
rv = np.array(rec.to_vector()).reshape(rs)
ov = wav[:, :rv.shape[1]]
err = np.abs(rv - ov).mean()
print("STFT/ISTFT shapes OK, recon MAE=%.4f" % err)
assert err < 0.05, err

print("\n=== datasets ===")
droot = tempfile.mkdtemp()
os.makedirs(os.path.join(droot, "yes"))
os.makedirs(os.path.join(droot, "no"))
for i, lab in enumerate(["yes", "no"]):
    aio.save(os.path.join(droot, lab, "%d.wav" % i), wav[:1, :sr // 4], sr)
sc = SpeechCommands(droot)
assert len(sc) == 2, len(sc)
w, srr, tgt = sc[0]
assert srr == sr and tgt in (0, 1)
print("SpeechCommands OK", len(sc))
wd = WaveformDataset(droot)
assert len(wd) == 2
print("WaveformDataset OK")

print("\n=== ALL AUDIO TESTS PASSED ===")
