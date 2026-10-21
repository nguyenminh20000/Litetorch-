import numpy as np

import litetorch as lt
from litetorch import nn


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.qweight = np.zeros((out_features, in_features), dtype=np.int8)
        self.scale = 1.0
        self.weight_f = lt.Tensor.zeros([out_features, in_features])
        self.bias_t = lt.Tensor.zeros([out_features]) if bias else None

    @staticmethod
    def from_linear(linear, in_features, out_features):
        params = linear.parameters()
        q = QuantizedLinear(in_features, out_features, len(params) > 1)
        wv = np.array(params[0].to_vector(), dtype=np.float32)
        amax = float(np.abs(wv).max()) if wv.size else 0.0
        scale = amax / 127.0 if amax > 0 else 1.0
        qi = np.clip(np.round(wv / scale), -128, 127).astype(np.int8)
        q.qweight = qi.reshape(out_features, in_features)
        q.scale = scale
        q.weight_f = lt.Tensor.from_vector((qi.astype(np.float32) * scale).tolist(),
                                           [out_features, in_features])
        if q.has_bias:
            q.bias_t = params[1].clone()
        return q

    def forward(self, x):
        y = lt.Ops.matmul(x, self.weight_f.transpose(0, 1))
        if self.has_bias:
            y = lt.Ops.add(y, self.bias_t)
        return y

    def parameters(self):
        return [self.bias_t] if self.has_bias else []

    def to(self, device):
        self.weight_f = self.weight_f.to(device)
        if self.has_bias:
            self.bias_t = self.bias_t.to(device)

    def storage_bytes(self):
        n = int(self.qweight.nbytes)
        if self.has_bias:
            n += self.out_features * 4
        return n + 8


class LiteSequential(nn.Module):
    def __init__(self, layers=()):
        super().__init__()
        self.layers = list(layers)

    def add(self, module):
        self.layers.append(module)
        return self

    def forward(self, x):
        for m in self.layers:
            x = m.forward(x)
        return x

    def parameters(self):
        ps = []
        for m in self.layers:
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in self.layers:
            m.to(device)


def quantize_linear(linear):
    w = linear.parameters()[0]
    out_features, in_features = w.shape[0], w.shape[1]
    return QuantizedLinear.from_linear(linear, in_features, out_features)


def quantize_module(seq):
    out = LiteSequential()
    for m in seq.layers:
        if isinstance(m, nn.Linear):
            out.add(quantize_linear(m))
        elif isinstance(m, QuantizedLinear):
            out.add(m)
        else:
            out.add(m)
    return out


def quantization_error(linear, qlinear, x):
    a = linear.forward(x).to_vector()
    b = qlinear.forward(x).to_vector()
    return max(abs(i - j) for i, j in zip(a, b))
