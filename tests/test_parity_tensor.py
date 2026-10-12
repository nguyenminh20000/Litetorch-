import numpy as np
import pytest

import litetorch as lt


def cpu():
    return lt.Device(lt.DeviceType.CPU, 0)


def test_cuda_device_count():
    n = lt.cuda.device_count()
    assert isinstance(n, int) and n >= 0


def test_cuda_current_device():
    d = lt.cuda.current_device()
    assert isinstance(d, int) and d >= 0


def test_cuda_set_device_no_throw():
    lt.cuda.set_device(0)


def test_cuda_memory_reserved():
    r = lt.cuda.memory_reserved()
    assert isinstance(r, int) and r >= 0


def test_cuda_manual_seed_all_reproducible():
    lt.cuda.manual_seed_all(1234)
    a = lt.randn([8, 8], cpu())
    lt.cuda.manual_seed_all(1234)
    b = lt.randn([8, 8], cpu())
    assert lt.equal(a, b)
    lt.cuda.manual_seed_all(999)
    c = lt.randn([8, 8], cpu())
    assert not lt.equal(a, c)


def test_numpy_zero_copy_shares_memory():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu())
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2) and arr.dtype == np.float32
    assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]])
    arr[0, 0] = 42.0
    assert t.tolist()[0][0] == pytest.approx(42.0)


def test_numpy_strided_zero_copy():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu())
    tr = t.transpose(0, 1)
    arr = tr.numpy()
    assert arr.shape == (2, 2)
    assert np.allclose(arr, np.array([[1.0, 2.0], [3.0, 4.0]]).T)


def test_numpy_scalar_zero_dim():
    t = lt.Tensor.from_vector([3.5], [1], cpu())
    s = t.reshape([])
    arr = s.numpy()
    assert arr.shape == ()
    assert float(arr) == pytest.approx(3.5)


def test_numpy_copy_dtypes():
    vals = [1.0, -2.5, 3.25]
    for dt, npdt in [(lt.DataType.FP64, np.float64), (lt.DataType.FP16, np.float16),
                     (lt.DataType.INT32, np.int32), (lt.DataType.INT16, np.int16),
                     (lt.DataType.INT8, np.int8)]:
        t = lt.Tensor.from_vector(vals, [3], cpu(), False, dt)
        arr = t.numpy()
        assert arr.dtype == npdt, (dt, arr.dtype)
        assert np.allclose(arr.astype(np.float64), vals, atol=1e-3)


def test_numpy_bf16_converts_to_fp32():
    t = lt.Tensor.from_vector([1.5, 2.5], [2], cpu(), False, lt.DataType.BF16)
    arr = t.numpy()
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.5, 2.5], atol=1e-2)


def test_tolist_nested():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu())
    assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tolist_scalar_and_1d():
    t = lt.Tensor.from_vector([7.5], [1], cpu()).reshape([])
    assert t.tolist() == pytest.approx(7.5)
    t1 = lt.Tensor.from_vector([1.0, 2.0], [2], cpu())
    assert t1.tolist() == [pytest.approx(1.0), pytest.approx(2.0)]


def test_tolist_int_dtype_gives_python_ints():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu(), False, lt.DataType.INT32)
    out = t.tolist()
    assert out == [1, 2, 3]
    assert all(isinstance(x, int) for x in out)


def test_tolist_non_contiguous():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu())
    tr = t.transpose(0, 1)
    assert tr.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_cpu_roundtrip_values():
    t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu())
    c = t.cpu()
    assert c.device.type == lt.DeviceType.CPU
    assert c.tolist() == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]


def test_cuda_method_no_crash_values_preserved():
    t = lt.Tensor.from_vector([5.0, 6.0], [2], cpu())
    g = t.cuda()
    assert g.device.type in (lt.DeviceType.GPU, lt.DeviceType.CPU)
    back = g.cpu()
    assert back.tolist() == [pytest.approx(5.0), pytest.approx(6.0)]


def test_dim_and_ndim():
    t = lt.zeros([2, 3, 4], cpu())
    assert t.dim() == 3
    assert t.ndim == 3
    s = lt.zeros([5], cpu()).reshape([])
    assert s.dim() == 0 and s.ndim == 0


def test_dtype_shorthands():
    t = lt.Tensor.from_vector([1.5, 2.5], [2], cpu())
    assert t.half().dtype == lt.DataType.FP16
    assert t.float().dtype == lt.DataType.FP32
    assert t.double().dtype == lt.DataType.FP64
    assert t.long().dtype == lt.DataType.INT32
    assert t.int().dtype == lt.DataType.INT32
    h = t.half()
    assert np.allclose(h.numpy().astype(np.float64), [1.5, 2.5], atol=1e-3)
    d = t.double()
    assert np.allclose(d.numpy(), [1.5, 2.5])
    li = lt.Tensor.from_vector([1.0, 2.0], [2], cpu()).long()
    assert li.numpy().tolist() == [1, 2]


def test_type_as():
    a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu())
    b = lt.Tensor.from_vector([0.0], [1], cpu(), False, lt.DataType.FP16)
    c = a.type_as(b)
    assert c.dtype == lt.DataType.FP16
    assert np.allclose(c.numpy().astype(np.float64), [1.0, 2.0], atol=1e-3)


def test_equal_true_cases():
    a = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu())
    b = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu())
    assert lt.equal(a, b) is True
    ai = lt.Tensor.from_vector([1.0, 2.0], [2], cpu(), False, lt.DataType.INT32)
    af = lt.Tensor.from_vector([1.0, 2.0], [2], cpu())
    assert lt.equal(ai, af) is True


def test_equal_false_cases():
    a = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu())
    b = lt.Tensor.from_vector([1.0, 2.0, 4.0], [3], cpu())
    assert lt.equal(a, b) is False
    c = lt.Tensor.from_vector([1.0, 2.0], [2], cpu())
    assert lt.equal(a, c) is False


def test_equal_nan_is_false():
    a = lt.Tensor.from_vector([float("nan")], [1], cpu())
    b = lt.Tensor.from_vector([float("nan")], [1], cpu())
    assert lt.equal(a, b) is False


def test_equal_non_contiguous():
    a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu()).transpose(0, 1)
    b = lt.Tensor.from_vector([1.0, 3.0, 2.0, 4.0], [2, 2], cpu())
    assert lt.equal(a, b) is True
