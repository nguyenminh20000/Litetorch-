import os
import struct
import wave

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    import soundfile as _sf
    _SF = True
except ImportError:
    _SF = False


def _require_np():
    if not _NP:
        raise ImportError("numpy is required for audio I/O")


def load(path, normalize=True):
    _require_np()
    if _SF:
        data, sr = _sf.read(path, always_2d=True, dtype="float32")
        return data.T.copy(), sr
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported sample width %d" % sampwidth)
    data = data.reshape(-1, n_channels).T.copy()
    return data, sr


def save(path, waveform, sample_rate):
    _require_np()
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[None, :]
    if _SF:
        _sf.write(path, wav.T, sample_rate)
        return
    n_channels = wav.shape[0]
    clipped = np.clip(wav, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    interleaved = pcm.T.reshape(-1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(interleaved.tobytes())


def info(path):
    _require_np()
    if _SF:
        i = _sf.info(path)
        return {"sample_rate": i.samplerate, "num_channels": i.channels,
                "num_frames": i.frames, "format": i.format}
    with wave.open(path, "rb") as w:
        return {"sample_rate": w.getframerate(), "num_channels": w.getnchannels(),
                "num_frames": w.getnframes(), "format": "WAV"}
