import json
import os
import time

import numpy as np

import litetorch as lt
from litetorch import nn

from .quant import QuantizedLinear, LiteSequential


def _layer_spec(m):
    if isinstance(m, QuantizedLinear):
        return {"type": "QuantizedLinear", "in": m.in_features, "out": m.out_features,
                "bias": m.has_bias, "scale": m.scale,
                "qweight": m.qweight.astype(np.int16).reshape(-1).tolist(),
                "bias_v": m.bias_t.to_vector() if m.has_bias else []}
    if isinstance(m, nn.Linear):
        w = m.parameters()[0]
        spec = {"type": "Linear", "in": w.shape[1], "out": w.shape[0],
                "bias": len(m.parameters()) > 1,
                "weight": w.to_vector()}
        if len(m.parameters()) > 1:
            spec["bias_v"] = m.parameters()[1].to_vector()
        return spec
    if isinstance(m, nn.ReLU):
        return {"type": "ReLU"}
    if isinstance(m, nn.Sigmoid):
        return {"type": "Sigmoid"}
    if isinstance(m, nn.Tanh):
        return {"type": "Tanh"}
    if isinstance(m, nn.GELU):
        return {"type": "GELU"}
    raise TypeError("export_litemodel: unsupported layer %r" % type(m).__name__)


def _build_layer(spec):
    t = spec["type"]
    if t == "QuantizedLinear":
        m = QuantizedLinear(spec["in"], spec["out"], spec["bias"])
        m.scale = spec["scale"]
        m.qweight = np.array(spec["qweight"], dtype=np.int16).astype(np.int8).reshape(spec["out"], spec["in"])
        m.weight_f = lt.Tensor.from_vector(
            (m.qweight.astype(np.float32) * m.scale).reshape(-1).tolist(),
            [spec["out"], spec["in"]])
        if spec["bias"]:
            m.bias_t = lt.Tensor.from_vector(spec["bias_v"], [spec["out"]])
        return m
    if t == "Linear":
        m = nn.Linear(spec["in"], spec["out"], spec["bias"])
        m.parameters()[0].copy_(lt.Tensor.from_vector(spec["weight"], [spec["out"], spec["in"]]))
        if spec["bias"]:
            m.parameters()[1].copy_(lt.Tensor.from_vector(spec["bias_v"], [spec["out"]]))
        return m
    if t == "ReLU":
        return nn.ReLU()
    if t == "Sigmoid":
        return nn.Sigmoid()
    if t == "Tanh":
        return nn.Tanh()
    if t == "GELU":
        return nn.GELU()
    raise TypeError("LiteModule: unknown layer type %r" % t)


def export_litemodel(seq, path):
    spec = [_layer_spec(m) for m in seq.layers]
    meta = {"format": "litetorch-lite-v1", "layers": spec}
    with open(path + ".json", "w") as f:
        json.dump(meta, f)
    lt.save_parameters(seq.parameters(), path + ".params")
    return path + ".json"


class LiteModule:
    def __init__(self, path):
        with open(path + ".json") as f:
            meta = json.load(f)
        assert meta.get("format") == "litetorch-lite-v1", meta.get("format")
        self.seq = LiteSequential([_build_layer(s) for s in meta["layers"]])
        lt.load_parameters(self.seq.parameters(), path + ".params")
        self.meta = meta

    def forward(self, x):
        with lt.no_grad():
            return self.seq.forward(x)

    def benchmark(self, x, iters=100, warmup=10):
        sync = None
        if x.device.type == lt.DeviceType.GPU:
            sync = lt.cuda.synchronize
        for _ in range(warmup):
            self.forward(x)
        if sync is not None:
            sync()
        t0 = time.time()
        for _ in range(iters):
            self.forward(x)
        if sync is not None:
            sync()
        dt = time.time() - t0
        return {"ms_per_iter": dt / iters * 1000.0, "iters": iters}

    def storage_bytes(self):
        total = 0
        for s, m in zip(self.meta["layers"], self.seq.layers):
            if s["type"] == "QuantizedLinear":
                total += m.storage_bytes()
            elif s["type"] == "Linear":
                total += s["in"] * s["out"] * 4 + (s["out"] * 4 if s["bias"] else 0)
        return total

    def num_layers(self):
        return len(self.seq.layers)
