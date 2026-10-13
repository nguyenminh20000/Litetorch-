import numpy as np
import litetorch as lt

cpu = lt.Device("cpu")

def to_np(t):
    return np.array(t.to_vector(), dtype=np.float64).reshape(t.shape)

def to_cplx(t):
    a = to_np(t)
    return a[..., 0] + 1j * a[..., 1]

def from_real(a):
    return lt.Tensor.from_vector(a.astype(np.float64).ravel().tolist(), list(a.shape), cpu, False)

def check(name, got, ref, tol=1e-9):
    err = np.max(np.abs(got - ref))
    print(f"{name}: max err {err:.2e}")
    assert err < tol, f"{name} mismatch: {err}"

print("=== fft vs numpy ===")
np.random.seed(0)
for n in [1, 2, 3, 4, 5, 7, 8, 16, 100, 256]:
    x = np.random.randn(n)
    got = to_cplx(lt.fft.fft(from_real(x)))
    check(f"fft n={n}", got, np.fft.fft(x))

print("\n=== ifft roundtrip ===")
for n in [3, 8, 100]:
    x = np.random.randn(n)
    xc = lt.fft.fft(from_real(x))
    back = to_np(lt.fft.ifft(xc))
    check(f"ifft(fft) n={n}", back, x)

print("\n=== rfft / irfft ===")
for n in [8, 100]:
    x = np.random.randn(n)
    got = to_cplx(lt.fft.rfft(from_real(x)))
    check(f"rfft n={n}", got, np.fft.rfft(x))
    back = to_np(lt.fft.irfft(lt.fft.rfft(from_real(x))))
    check(f"irfft(rfft) n={n}", back, x)

print("\n=== rfft odd n ===")
x = np.random.randn(7)
got = to_cplx(lt.fft.rfft(from_real(x)))
check("rfft n=7", got, np.fft.rfft(x))
back = to_np(lt.fft.irfft(lt.fft.rfft(from_real(x))))
check("irfft(rfft) n=7", back, x)

print("\n=== fft dim / batched ===")
x = np.random.randn(4, 8, 16)
got = to_cplx(lt.fft.fft(from_real(x), dim=1))
check("fft dim=1", got, np.fft.fft(x, axis=1))
got = to_cplx(lt.fft.fft(from_real(x), dim=-1))
check("fft dim=-1", got, np.fft.fft(x, axis=-1))

print("\n=== fft n pad/truncate ===")
x = np.random.randn(8)
got = to_cplx(lt.fft.fft(from_real(x), n=16))
check("fft n=16 pad", got, np.fft.fft(x, n=16))
got = to_cplx(lt.fft.fft(from_real(x), n=5))
check("fft n=5 trunc", got, np.fft.fft(x, n=5))

print("\n=== norms ===")
x = np.random.randn(32)
for norm in ["backward", "forward", "ortho"]:
    got = to_cplx(lt.fft.fft(from_real(x), norm=norm))
    check(f"fft norm={norm}", got, np.fft.fft(x, norm=norm))
    xc = lt.fft.fft(from_real(x), norm=norm)
    back = to_np(lt.fft.ifft(xc, norm=norm))
    check(f"ifft norm={norm}", back, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm))

print("\n=== fftn / ifftn ===")
x = np.random.randn(4, 6, 8)
got = to_cplx(lt.fft.fftn(from_real(x)))
check("fftn all", got, np.fft.fftn(x))
back = to_np(lt.fft.ifftn(lt.fft.fftn(from_real(x))))
check("ifftn(fftn)", back, x)
got = to_cplx(lt.fft.fftn(from_real(x), dims=[0, 2]))
check("fftn dims=[0,2]", got, np.fft.fftn(x, axes=[0, 2]))
got = to_cplx(lt.fft.fftn(from_real(x), s=[8, 8], dims=[1, 2]))
check("fftn s=[8,8]", got, np.fft.fftn(x, s=[8, 8], axes=[1, 2]))

print("\n=== fftshift / ifftshift ===")
for n in [8, 7]:
    x = np.random.randn(n)
    got = to_np(lt.fft.fftshift(from_real(x)))
    check(f"fftshift n={n}", got, np.fft.fftshift(x))
    got = to_np(lt.fft.ifftshift(from_real(x)))
    check(f"ifftshift n={n}", got, np.fft.ifftshift(x))
x = np.random.randn(4, 7)
got = to_np(lt.fft.fftshift(from_real(x)))
check("fftshift 2d", got, np.fft.fftshift(x))
got = to_np(lt.fft.ifftshift(from_real(x), dims=[1]))
check("ifftshift dims=[1]", got, np.fft.ifftshift(x, axes=[1]))
back = to_np(lt.fft.ifftshift(lt.fft.fftshift(from_real(x))))
check("ifftshift(fftshift)", back, x)

print("\n=== fftfreq ===")
for n in [8, 7, 10]:
    for d in [1.0, 0.5]:
        got = to_np(lt.fft.fftfreq(n, d))
        check(f"fftfreq n={n} d={d}", got, np.fft.fftfreq(n, d))

print("\n=== error cases ===")
for fn, args in [
    (lambda: lt.fft.fft(from_real(np.random.randn(4)), norm="bad"), "bad norm"),
    (lambda: lt.fft.ifft(from_real(np.random.randn(4))), "ifft real input"),
    (lambda: lt.fft.fft(from_real(np.random.randn(4)), dim=5), "dim oob"),
    (lambda: lt.fft.fftfreq(0), "fftfreq n=0"),
]:
    try:
        fn()
        raise AssertionError(f"should have raised: {args}")
    except RuntimeError:
        print(f"raises OK: {args}")

print("\nALL FFT TESTS PASSED")
