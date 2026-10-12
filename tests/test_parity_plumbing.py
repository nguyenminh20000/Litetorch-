import os
import tempfile

import numpy as np

import litetorch as lt


def make_model():
    return lt.nn.Sequential([
        lt.nn.Linear(8, 16),
        lt.nn.BatchNorm2d(16),
    ])


def t2np(t):
    return np.array(t.to_vector(), dtype=np.float64).reshape(t.shape)


def assert_tensors_equal(a, b):
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"dtype mismatch {a.dtype} vs {b.dtype}"
    assert np.allclose(t2np(a), t2np(b)), "values differ"


def test_state_dict_keys():
    m = make_model()
    sd = m.state_dict()
    keys = list(sd.keys())
    assert keys == ["0.weight", "0.bias", "1.weight", "1.bias",
                    "1.running_mean", "1.running_var"], keys
    assert sd["0.weight"].shape == [16, 8]
    assert sd["1.running_mean"].shape == [16]


def test_named_parameters_exact():
    m = make_model()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["0.weight", "0.bias", "1.weight", "1.bias"], names


def test_named_buffers():
    m = make_model()
    nb = m.named_buffers()
    assert [n for n, _ in nb] == ["1.running_mean", "1.running_var"]
    bufs = m.buffers()
    assert len(bufs) == 2
    assert all(b.shape == [16] for b in bufs)


def test_named_children():
    m = make_model()
    nc = m.named_children()
    assert [n for n, _ in nc] == ["0", "1"]
    assert isinstance(nc[0][1], lt.nn.Linear)
    assert isinstance(nc[1][1], lt.nn.BatchNorm2d)


def test_named_modules_nested():
    inner = lt.nn.Sequential([lt.nn.Linear(4, 4)])
    m = lt.nn.Sequential([inner, lt.nn.Linear(4, 2)])
    names = [n for n, _ in m.named_modules()]
    assert names == ["", "0", "0.0", "1"], names


def test_state_dict_roundtrip_via_save_load():
    m = make_model()
    sd = m.state_dict()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.bin")
        lt.save_parameters(list(sd.values()), path)
        m2 = make_model()
        before = [t2np(t) for t in m2.state_dict().values()]
        lt.load_parameters(list(m2.state_dict().values()), path)
        for (k, t), b in zip(m2.state_dict().items(), before):
            assert_tensors_equal(t, sd[k])
            assert not np.allclose(t2np(t), b) or np.allclose(b, t2np(sd[k]))


def test_load_state_dict_direct():
    m = make_model()
    sd = m.state_dict()
    m2 = make_model()
    missing, unexpected = m2.load_state_dict(sd)
    assert missing == [] and unexpected == [], (missing, unexpected)
    for k, t in m2.state_dict().items():
        assert_tensors_equal(t, sd[k])


def test_load_state_dict_strict_missing():
    m = make_model()
    sd = m.state_dict()
    partial = {k: v for k, v in sd.items() if k != "0.bias"}
    m2 = make_model()
    try:
        m2.load_state_dict(partial, strict=True)
        raise AssertionError("strict load should have raised on missing key")
    except RuntimeError as e:
        assert "0.bias" in str(e), str(e)
    missing, unexpected = m2.load_state_dict(partial, strict=False)
    assert missing == ["0.bias"], missing
    assert unexpected == []


def test_load_state_dict_unexpected():
    m = make_model()
    sd = dict(m.state_dict())
    sd["bogus.key"] = lt.Tensor.zeros([2], lt.Device("cpu:0"))
    m2 = make_model()
    try:
        m2.load_state_dict(sd, strict=True)
        raise AssertionError("strict load should have raised on unexpected key")
    except RuntimeError as e:
        assert "bogus.key" in str(e), str(e)
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    assert unexpected == ["bogus.key"], unexpected
    assert missing == []


def test_load_state_dict_shape_mismatch():
    m = make_model()
    sd = dict(m.state_dict())
    sd["0.weight"] = lt.Tensor.zeros([4, 8], lt.Device("cpu:0"))
    m2 = make_model()
    try:
        m2.load_state_dict(sd)
        raise AssertionError("shape mismatch should raise")
    except RuntimeError as e:
        assert "size mismatch" in str(e), str(e)


def test_parameter_requires_grad_default():
    data = lt.Tensor.randn([3, 4], lt.Device("cpu:0"))
    p = lt.nn.Parameter(data)
    assert p.requires_grad is True
    assert p.shape == [3, 4]
    assert np.allclose(t2np(p), t2np(data))
    p2 = lt.nn.Parameter(data, requires_grad=False)
    assert p2.requires_grad is False


def test_register_parameter_and_buffer():
    m = lt.nn.Module()
    w = lt.Tensor.randn([4, 4], lt.Device("cpu:0"))
    assert w.requires_grad is False
    m.register_parameter("weight", w)
    assert w.requires_grad is True
    assert [n for n, _ in m.named_parameters()] == ["weight"]
    assert len(m.parameters()) == 1

    b = lt.Tensor.zeros([4], lt.Device("cpu:0"))
    m.register_buffer("running", b)
    assert len(m.parameters()) == 1
    assert [n for n, _ in m.named_buffers()] == ["running"]
    assert list(m.state_dict().keys()) == ["weight", "running"]
    assert len(m.buffers()) == 1

    m.register_parameter("weight", lt.nn.Parameter(w))
    assert len(m.parameters()) == 1


def test_add_module_naming():
    m = lt.nn.Module()
    m.add_module("fc", lt.nn.Linear(4, 4))
    m.add_module("bn", lt.nn.BatchNorm2d(4))
    assert [n for n, _ in m.named_children()] == ["fc", "bn"]
    assert [n for n, _ in m.named_parameters()] == [
        "fc.weight", "fc.bias", "bn.weight", "bn.bias"]
    assert [n for n, _ in m.named_modules()] == [
        "", "fc", "bn"]


def test_modulelist():
    ml = lt.nn.ModuleList([lt.nn.Linear(4, 4), lt.nn.Linear(4, 2)])
    assert len(ml) == 2
    assert isinstance(ml[0], lt.nn.Linear)
    assert isinstance(ml[-1], lt.nn.Linear)
    assert isinstance(ml[1], lt.nn.Linear)
    assert [type(x).__name__ for x in ml] == ["Linear", "Linear"]
    ml.append(lt.nn.Linear(2, 2))
    assert len(ml) == 3
    ml.extend([lt.nn.Linear(2, 1)])
    assert len(ml) == 4
    ml[0] = lt.nn.Linear(4, 8)
    assert ml[0].weight.shape == [8, 4]
    try:
        _ = ml[10]
        raise AssertionError("index out of range should raise")
    except IndexError:
        pass
    params = ml.parameters()
    assert len(params) == 8
    names = [n for n, _ in ml.named_parameters()]
    assert names[0] == "0.weight" and names[-1] == "3.bias", names
    x = lt.Tensor.randn([2, 4], lt.Device("cpu:0"))
    try:
        ml(x)
        raise AssertionError("ModuleList forward should raise")
    except RuntimeError:
        pass
    mods = [n for n, _ in ml.named_modules()]
    assert mods == ["", "0", "1", "2", "3"], mods


def test_moduledict():
    md = lt.nn.ModuleDict()
    md["fc"] = lt.nn.Linear(4, 4)
    md.put("bn", lt.nn.BatchNorm2d(4))
    assert len(md) == 2
    assert set(md.keys()) == {"fc", "bn"}
    assert "fc" in md
    assert "nope" not in md
    assert isinstance(md["fc"], lt.nn.Linear)
    assert md.get("missing") is None
    names = [n for n, _ in md.named_parameters()]
    assert names == ["fc.weight", "fc.bias", "bn.weight", "bn.bias"], names
    sd = md.state_dict()
    assert list(sd.keys()) == ["fc.weight", "fc.bias", "bn.weight", "bn.bias",
                               "bn.running_mean", "bn.running_var"], list(sd.keys())
    del md["bn"]
    assert len(md) == 1 and "bn" not in md
    x = lt.Tensor.randn([2, 4], lt.Device("cpu:0"))
    out = md["fc"](x)
    assert list(out.shape) == [2, 4]
    mods = [n for n, _ in md.named_modules()]
    assert mods == ["", "fc"], mods


def test_zero_grad():
    m = make_model()
    for p in m.parameters():
        p.grad = lt.Tensor.ones(p.shape, lt.Device("cpu:0"))
    m.zero_grad()
    for p in m.parameters():
        assert p.grad is not None
        assert np.allclose(t2np(p.grad), 0.0)


def test_apply():
    m = make_model()
    seen = []
    m.apply(lambda mod: seen.append(type(mod).__name__))
    assert seen == ["Sequential", "Linear", "BatchNorm2d"], seen
    m.apply(lambda mod: mod.eval())
    assert all(not mod.is_training() for _, mod in m.named_modules())


def test_train_eval_recurse():
    m = make_model()
    m.eval()
    assert all(not mod.is_training() for _, mod in m.named_modules())
    m.train()
    assert all(mod.is_training() for _, mod in m.named_modules())


def test_hf_style_safetensors_roundtrip():
    m = make_model()
    sd = m.state_dict()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        lt.save_safetensors(dict(sd), path)
        loaded = lt.load_safetensors(path)
        m2 = make_model()
        missing, unexpected = m2.load_state_dict(loaded, strict=True)
        assert missing == [] and unexpected == []
        for k, t in m2.state_dict().items():
            assert_tensors_equal(t, sd[k])


def test_sequential_forward_still_works():
    m = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.ReLU()])
    x = lt.Tensor.randn([2, 4], lt.Device("cpu:0"))
    out = m(x)
    assert list(out.shape) == [2, 4]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
