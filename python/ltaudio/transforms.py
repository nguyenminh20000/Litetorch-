import math

import litetorch as lt

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def _require_np():
    if not _NP:
        raise ImportError("numpy is required for audio transforms")


def _as_waveform(waveform):
    _require_np()
    if isinstance(waveform, lt.Tensor):
        shape = list(waveform.shape)
        arr = np.array(waveform.to_vector(), dtype=np.float32).reshape(shape)
    else:
        arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _to_tensor(arr):
    shape = list(arr.shape)
    return lt.Tensor.from_vector(np.asarray(arr, dtype=np.float32).reshape(-1).tolist(),
                                 shape, lt.Device("cpu"))


class Resample:
    def __init__(self, orig_freq, new_freq):
        _require_np()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform):
        wav = _as_waveform(waveform)
        if self.orig_freq == self.new_freq:
            return _to_tensor(wav)
        ratio = self.new_freq / self.orig_freq
        n_out = int(round(wav.shape[1] * ratio))
        old_idx = np.arange(wav.shape[1])
        new_idx = np.linspace(0, wav.shape[1] - 1, n_out)
        out = np.stack([np.interp(new_idx, old_idx, wav[c]) for c in range(wav.shape[0])])
        return _to_tensor(out.astype(np.float32))

    def __repr__(self):
        return "Resample(orig_freq=%d, new_freq=%d)" % (self.orig_freq, self.new_freq)


def _stft_matrix(x, n_fft, hop_length, win_length, window):
    _require_np()
    if window is None:
        win = np.ones(win_length, dtype=np.float32)
    elif window == "hann":
        win = np.hanning(win_length + 1)[:-1].astype(np.float32)
    elif window == "hamming":
        win = np.hamming(win_length).astype(np.float32)
    else:
        raise ValueError("unknown window %s" % window)
    pad = (n_fft - win_length) // 2
    win_padded = np.pad(win, (pad, n_fft - win_length - pad))
    n_frames = 1 + (len(x) - n_fft) // hop_length
    frames = np.stack([x[i * hop_length:i * hop_length + n_fft] * win_padded
                       for i in range(n_frames)])
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec


class Spectrogram:
    def __init__(self, n_fft=400, win_length=None, hop_length=None, window="hann",
                 power=2.0, normalized=False):
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.power = power
        self.normalized = normalized

    def __call__(self, waveform):
        wav = _as_waveform(waveform)
        specs = []
        for c in range(wav.shape[0]):
            spec = _stft_matrix(wav[c], self.n_fft, self.hop_length, self.win_length, self.window)
            mag = np.abs(spec)
            if self.power is not None:
                mag = mag ** self.power
            specs.append(mag.astype(np.float32))
        out = np.stack(specs).transpose(0, 2, 1)
        return _to_tensor(out)

    def __repr__(self):
        return "Spectrogram(n_fft=%d, hop_length=%d)" % (self.n_fft, self.hop_length)


class STFT:
    def __init__(self, n_fft=400, win_length=None, hop_length=None, window="hann",
                 return_complex=True):
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.return_complex = return_complex

    def __call__(self, waveform):
        wav = _as_waveform(waveform)
        specs = []
        for c in range(wav.shape[0]):
            spec = _stft_matrix(wav[c], self.n_fft, self.hop_length, self.win_length, self.window)
            if self.return_complex:
                stacked = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
            else:
                stacked = np.abs(spec).astype(np.float32)
            specs.append(stacked)
        out = np.stack(specs).transpose(0, 2, 1, 3)
        return _to_tensor(out)

    def __repr__(self):
        return "STFT(n_fft=%d, hop_length=%d)" % (self.n_fft, self.hop_length)


class InverseSpectrogram:
    def __init__(self, n_fft=400, win_length=None, hop_length=None, window="hann"):
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window

    def __call__(self, spec):
        _require_np()
        if isinstance(spec, lt.Tensor):
            shape = list(spec.shape)
            arr = np.array(spec.to_vector(), dtype=np.float32).reshape(shape)
        else:
            arr = np.asarray(spec, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :]
        if arr.shape[-1] == 2:
            complex_spec = (arr[..., 0] + 1j * arr[..., 1]).transpose(0, 2, 1)
        else:
            complex_spec = arr.astype(np.complex64).transpose(0, 2, 1)
        waves = []
        for c in range(complex_spec.shape[0]):
            frames = np.fft.irfft(complex_spec[c], n=self.n_fft, axis=1)
            n_frames = frames.shape[0]
            length = self.n_fft + (n_frames - 1) * self.hop_length
            y = np.zeros(length, dtype=np.float64)
            wsum = np.zeros(length, dtype=np.float64)
            if self.window == "hann":
                win = np.hanning(self.win_length + 1)[:-1]
            else:
                win = np.ones(self.win_length)
            pad = (self.n_fft - self.win_length) // 2
            win_p = np.pad(win, (pad, self.n_fft - self.win_length - pad))
            for i in range(n_frames):
                s = i * self.hop_length
                y[s:s + self.n_fft] += frames[i] * win_p
                wsum[s:s + self.n_fft] += win_p ** 2
            wsum[wsum < 1e-10] = 1.0
            waves.append((y / wsum).astype(np.float32))
        return _to_tensor(np.stack(waves))

    def __repr__(self):
        return "InverseSpectrogram(n_fft=%d, hop_length=%d)" % (self.n_fft, self.hop_length)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, f_min=0.0, f_max=None):
    _require_np()
    f_max = f_max or sr / 2.0
    m_min, m_max = _hz_to_mel(f_min), _hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m1, f_m, f_m2 = bins[m - 1], bins[m], bins[m + 1]
        if f_m > f_m1:
            fb[m - 1, f_m1:f_m] = (np.arange(f_m1, f_m) - f_m1) / max(1, f_m - f_m1)
        if f_m2 > f_m:
            fb[m - 1, f_m:f_m2] = (f_m2 - np.arange(f_m, f_m2)) / max(1, f_m2 - f_m)
    return fb


class MelSpectrogram:
    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None,
                 window="hann", n_mels=128, f_min=0.0, f_max=None, power=2.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2.0
        self.power = power
        self._fb = None

    def _filterbank(self):
        if self._fb is None:
            self._fb = _mel_filterbank(self.sample_rate, self.n_fft, self.n_mels,
                                       self.f_min, self.f_max)
        return self._fb

    def __call__(self, waveform):
        wav = _as_waveform(waveform)
        fb = self._filterbank()
        mels = []
        for c in range(wav.shape[0]):
            spec = _stft_matrix(wav[c], self.n_fft, self.hop_length, self.win_length, self.window)
            mag = np.abs(spec) ** self.power
            mel = (fb @ mag.T).T.astype(np.float32)
            mels.append(mel.T)
        return _to_tensor(np.stack(mels))

    def __repr__(self):
        return "MelSpectrogram(sample_rate=%d, n_mels=%d)" % (self.sample_rate, self.n_mels)


class AmplitudeToDB:
    def __init__(self, stype="power", top_db=80.0, amin=1e-10, ref_value=1.0):
        self.stype = stype
        self.top_db = top_db
        self.amin = amin
        self.ref_value = ref_value

    def __call__(self, spec):
        _require_np()
        if isinstance(spec, lt.Tensor):
            shape = list(spec.shape)
            arr = np.array(spec.to_vector(), dtype=np.float32).reshape(shape)
        else:
            arr = np.asarray(spec, dtype=np.float32)
        mult = 10.0 if self.stype == "power" else 20.0
        db = mult * np.log10(np.maximum(self.amin, arr) / max(self.amin, self.ref_value))
        if self.top_db is not None:
            db = np.maximum(db, db.max() - self.top_db)
        return _to_tensor(db.astype(np.float32))

    def __repr__(self):
        return "AmplitudeToDB(stype=%s)" % self.stype


class MFCC:
    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=128, n_fft=400,
                 hop_length=None, f_min=0.0, f_max=None):
        self.n_mfcc = n_mfcc
        self.mel = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                  n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0)
        self.to_db = AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, waveform):
        _require_np()
        mel = self.mel(waveform)
        shape = list(mel.shape)
        log_mel = np.array(self.to_db(mel).to_vector(), dtype=np.float32).reshape(shape)
        dct = _dct(log_mel, axis=1, norm="ortho")
        return _to_tensor(dct[:, :self.n_mfcc, :].astype(np.float32))

    def __repr__(self):
        return "MFCC(n_mfcc=%d)" % self.n_mfcc


def _dct(x, axis=-1, norm=None):
    _require_np()
    ax = axis % x.ndim
    n = x.shape[ax]
    k = np.arange(n)
    nn_ = np.arange(n)
    basis = np.cos(np.pi / n * np.outer(nn_ + 0.5, k))
    if norm == "ortho":
        basis[:, 0] *= 1 / math.sqrt(n)
        basis[:, 1:] *= math.sqrt(2 / n)
    moved = np.moveaxis(x, ax, -1)
    out = moved @ basis
    return np.moveaxis(out, -1, ax)
