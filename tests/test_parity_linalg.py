import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import litetorch as lt
from litetorch import linalg as LA


DEV = lt.Device("cpu")
RESULTS = []


def tv(data, shape):
    return lt.Tensor.from_vector([float(v) for v in data], list(shape), DEV)


def run(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:
        RESULTS.append((name, "FAIL", type(e).__name__ + ": " + str(e)))


def close(a, b, tol=1e-4, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}"
    d = np.max(np.abs(a - b))
    assert d <= tol, f"{msg}: max diff {d} > {tol}"


def mat(n, vals):
    return np.array(vals, dtype=np.float64).reshape(n, n)


def test_inv():
    A = mat(3, [4, 7, 2, 3, 5, 1, 2, 3, 6])
    close(LA.inv(tv(A.flatten(), [3, 3])).to_vector(), np.linalg.inv(A).flatten(),
          msg="inv")


def test_inv_batched():
    A = np.stack([mat(2, [4, 7, 2, 6]), mat(2, [1, 2, 3, 4])])
    t = lt.Tensor.from_vector([float(v) for v in A.flatten()], [2, 2, 2], DEV)
    close(LA.inv(t).to_vector(), np.linalg.inv(A).flatten(), msg="inv batched")


def test_inv_singular_raises():
    A = tv([1.0, 2.0, 2.0, 4.0], [2, 2])
    try:
        LA.inv(A)
        raise AssertionError("expected singular error")
    except RuntimeError:
        pass


def test_solve():
    A = mat(3, [3, 1, 2, 1, 5, 3, 2, 3, 8])
    B = np.array([[1.0], [2.0], [3.0]])
    x = LA.solve(tv(A.flatten(), [3, 3]), tv(B.flatten(), [3, 1]))
    close(x.to_vector(), np.linalg.solve(A, B).flatten(), msg="solve")


def test_solve_vec():
    A = mat(2, [2, 1, 1, 3])
    b = np.array([5.0, 6.0])
    x = LA.solve(tv(A.flatten(), [2, 2]), tv(b, [2]))
    close(x.to_vector(), np.linalg.solve(A, b), msg="solve vec")


def test_cholesky():
    A = mat(3, [4, 2, 1, 2, 5, 3, 1, 3, 6])
    close(LA.cholesky(tv(A.flatten(), [3, 3])).to_vector(),
          np.linalg.cholesky(A).flatten(), msg="cholesky")


def test_cholesky_upper():
    A = mat(2, [4, 2, 2, 5])
    L = np.linalg.cholesky(A)
    close(LA.cholesky(tv(A.flatten(), [2, 2]), upper=True).to_vector(),
          L.T.flatten(), msg="cholesky upper")


def test_cholesky_not_pd_raises():
    A = tv([1.0, 2.0, 2.0, 1.0], [2, 2])
    try:
        LA.cholesky(A)
        raise AssertionError("expected PD error")
    except RuntimeError:
        pass


def test_cholesky_inverse():
    A = mat(3, [4, 2, 1, 2, 5, 3, 1, 3, 6])
    close(LA.cholesky_inverse(tv(A.flatten(), [3, 3])).to_vector(),
          np.linalg.inv(A).flatten(), tol=1e-3, msg="cholesky_inverse")


def test_qr():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 3)
    Q, R = LA.qr(tv(A.flatten(), [4, 3]))
    Qn, Rn = np.linalg.qr(A, mode="reduced")
    close(np.abs(Q.to_vector()), np.abs(Qn.flatten()), tol=1e-3, msg="qr Q")
    close(np.abs(R.to_vector()), np.abs(Rn.flatten()), tol=1e-3, msg="qr R")
    Qm = np.array(Q.to_vector()).reshape(4, 3)
    Rm = np.array(R.to_vector()).reshape(3, 3)
    close((Qm @ Rm).flatten(), A.flatten(), tol=1e-3, msg="qr reconstruct")


def test_qr_complete():
    rng = np.random.RandomState(1)
    A = rng.randn(3, 3)
    Q, R = LA.qr(tv(A.flatten(), [3, 3]), mode="complete")
    assert Q.shape == [3, 3], f"Q shape {Q.shape}"
    close((np.array(Q.to_vector()).reshape(3, 3) @ np.array(R.to_vector()).reshape(3, 3)).flatten(),
          A.flatten(), tol=1e-3, msg="qr complete reconstruct")


def test_svd():
    rng = np.random.RandomState(2)
    A = rng.randn(4, 3)
    U, S, Vh = LA.svd(tv(A.flatten(), [4, 3]), full_matrices=False)
    Un, Sn, Vhn = np.linalg.svd(A, full_matrices=False)
    close(S.to_vector(), Sn, tol=1e-3, msg="svd S")
    rec = np.array(U.to_vector()).reshape(4, 3) @ np.diag(S.to_vector()) @ \
        np.array(Vh.to_vector()).reshape(3, 3)
    close(rec.flatten(), A.flatten(), tol=1e-3, msg="svd reconstruct")


def test_svd_full():
    rng = np.random.RandomState(3)
    A = rng.randn(3, 4)
    U, S, Vh = LA.svd(tv(A.flatten(), [3, 4]), full_matrices=True)
    assert U.shape == [3, 3] and Vh.shape == [4, 4], \
        f"shapes {U.shape} {Vh.shape}"
    close(S.to_vector(), np.linalg.svd(A, compute_uv=False), tol=1e-3, msg="svd full S")


def test_eigh():
    A = mat(3, [4, 2, 1, 2, 5, 3, 1, 3, 6])
    w, V = LA.eigh(tv(A.flatten(), [3, 3]))
    wn, Vn = np.linalg.eigh(A)
    close(w.to_vector(), wn, tol=1e-3, msg="eigh w")
    close(np.abs(V.to_vector()), np.abs(Vn.flatten()), tol=1e-3, msg="eigh V")


def test_eigvalsh():
    A = mat(2, [2, 1, 1, 3])
    close(LA.eigvalsh(tv(A.flatten(), [2, 2])).to_vector(),
          np.linalg.eigvalsh(A), tol=1e-4, msg="eigvalsh")


def test_det():
    A = mat(3, [6, 1, 1, 4, -2, 5, 2, 8, 7])
    close([LA.det(tv(A.flatten(), [3, 3])).to_vector()[0]],
          [np.linalg.det(A)], tol=1e-3, msg="det")


def test_slogdet():
    A = mat(3, [6, 1, 1, 4, -2, 5, 2, 8, 7])
    s, l = LA.slogdet(tv(A.flatten(), [3, 3]))
    sn, ln = np.linalg.slogdet(A)
    close([s.to_vector()[0]], [sn], msg="slogdet sign")
    close([l.to_vector()[0]], [ln], tol=1e-4, msg="slogdet log")


def test_vector_norm():
    x = tv([3.0, 4.0], [2])
    close([LA.vector_norm(x).to_vector()[0]], [5.0], msg="vector_norm")
    close([LA.vector_norm(x, 1.0).to_vector()[0]], [7.0], msg="vector_norm ord=1")
    close([LA.norm(x).to_vector()[0]], [5.0], msg="norm default")


def test_matrix_norm():
    A = mat(2, [1, 2, 3, 4])
    close([LA.matrix_norm(tv(A.flatten(), [2, 2])).to_vector()[0]],
          [np.linalg.norm(A, "fro")], tol=1e-4, msg="matrix_norm fro")
    close([LA.matrix_norm(tv(A.flatten(), [2, 2]), "nuc").to_vector()[0]],
          [np.linalg.norm(A, "nuc")], tol=1e-3, msg="matrix_norm nuc")
    close([LA.matrix_norm(tv(A.flatten(), [2, 2]), "inf").to_vector()[0]],
          [np.linalg.norm(A, np.inf)], tol=1e-4, msg="matrix_norm inf")


def test_matrix_rank():
    A = mat(3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    close([LA.matrix_rank(tv(A.flatten(), [3, 3])).to_vector()[0]],
          [float(np.linalg.matrix_rank(A))], msg="matrix_rank")
    I = np.eye(3)
    close([LA.matrix_rank(tv(I.flatten(), [3, 3])).to_vector()[0]], [3.0],
          msg="matrix_rank eye")


def test_pinv():
    rng = np.random.RandomState(4)
    A = rng.randn(4, 3)
    close(LA.pinv(tv(A.flatten(), [4, 3])).to_vector(),
          np.linalg.pinv(A).flatten(), tol=1e-3, msg="pinv")


def test_lstsq():
    rng = np.random.RandomState(5)
    A = rng.randn(5, 3)
    b = rng.randn(5)
    sol, res, rank, sv = LA.lstsq(tv(A.flatten(), [5, 3]), tv(b, [5]))
    soln, resn, rankn, svn = np.linalg.lstsq(A, b, rcond=None)
    close(sol.to_vector(), soln, tol=1e-3, msg="lstsq sol")
    close([rank.to_vector()[0]], [float(rankn)], msg="lstsq rank")
    close(sv.to_vector(), svn, tol=1e-3, msg="lstsq sv")


def test_lu():
    A = mat(3, [2, 1, 1, 4, 3, 3, 8, 7, 9])
    P, L, U = LA.lu(tv(A.flatten(), [3, 3]))
    Pn = np.array(P.to_vector()).reshape(3, 3)
    Ln = np.array(L.to_vector()).reshape(3, 3)
    Un = np.array(U.to_vector()).reshape(3, 3)
    close((Pn @ Ln @ Un).flatten(), A.flatten(), tol=1e-3, msg="lu reconstruct")


def test_cross():
    a = tv([1.0, 0.0, 0.0], [3])
    b = tv([0.0, 1.0, 0.0], [3])
    close(LA.cross(a, b).to_vector(), [0.0, 0.0, 1.0], msg="cross")


def test_vecdot():
    x = tv([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = tv([5.0, 6.0, 7.0, 8.0], [2, 2])
    close(LA.vecdot(x, y).to_vector(), [1*5+2*6, 3*7+4*8], msg="vecdot")


def test_multi_dot():
    rng = np.random.RandomState(6)
    A, B, C = rng.randn(2, 3), rng.randn(3, 4), rng.randn(4, 2)
    R = LA.multi_dot([tv(A.flatten(), [2, 3]), tv(B.flatten(), [3, 4]),
                      tv(C.flatten(), [4, 2])])
    close(R.to_vector(), (A @ B @ C).flatten(), tol=1e-3, msg="multi_dot")


def test_diagonal():
    A = mat(3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    close(LA.diagonal(tv(A.flatten(), [3, 3])).to_vector(), [1, 5, 9], msg="diagonal")
    close(LA.diagonal(tv(A.flatten(), [3, 3]), offset=1).to_vector(), [2, 6],
          msg="diagonal offset")


def test_trace():
    A = mat(3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    close([LA.trace(tv(A.flatten(), [3, 3])).to_vector()[0]], [15.0], msg="trace")


def test_matrix_power():
    A = mat(2, [2, 0, 0, 3])
    close(LA.matrix_power(tv(A.flatten(), [2, 2]), 3).to_vector(),
          np.linalg.matrix_power(A, 3).flatten(), msg="matrix_power")
    close(LA.matrix_power(tv(A.flatten(), [2, 2]), -1).to_vector(),
          np.linalg.matrix_power(A, -1).flatten(), tol=1e-4, msg="matrix_power neg")


def test_cond():
    A = mat(2, [1, 2, 3, 4])
    close([LA.cond(tv(A.flatten(), [2, 2])).to_vector()[0]],
          [np.linalg.cond(A)], tol=1e-3, msg="cond")
    close([LA.cond(tv(A.flatten(), [2, 2]), "fro").to_vector()[0]],
          [np.linalg.cond(A, "fro")], tol=1e-3, msg="cond fro")


def test_outer():
    a = tv([1.0, 2.0], [2])
    b = tv([3.0, 4.0, 5.0], [3])
    close(LA.outer(a, b).to_vector(), np.outer([1, 2], [3, 4, 5]).flatten(),
          msg="outer")


def test_inner():
    a = tv([1.0, 2.0, 3.0], [3])
    b = tv([4.0, 5.0, 6.0], [3])
    close([LA.inner(a, b).to_vector()[0]], [32.0], msg="inner")


def test_kron():
    A = mat(2, [1, 2, 3, 4])
    B = mat(2, [0, 5, 6, 7])
    close(LA.kron(tv(A.flatten(), [2, 2]), tv(B.flatten(), [2, 2])).to_vector(),
          np.kron(A, B).flatten(), msg="kron")


def test_vdot():
    a = tv([1.0, 2.0], [2])
    b = tv([3.0, 4.0], [2])
    close([LA.vdot(a, b).to_vector()[0]], [11.0], msg="vdot")


def test_norm_dim():
    x = tv([3.0, 4.0, 5.0, 12.0], [2, 2])
    close(LA.norm(x, 2.0, 1).to_vector(), [5.0, 13.0], msg="norm dim")


TESTS = [
    test_inv, test_inv_batched, test_inv_singular_raises, test_solve, test_solve_vec,
    test_cholesky, test_cholesky_upper, test_cholesky_not_pd_raises, test_cholesky_inverse,
    test_qr, test_qr_complete, test_svd, test_svd_full, test_eigh, test_eigvalsh,
    test_det, test_slogdet, test_vector_norm, test_matrix_norm, test_matrix_rank,
    test_pinv, test_lstsq, test_lu, test_cross, test_vecdot, test_multi_dot,
    test_diagonal, test_trace, test_matrix_power, test_cond, test_outer, test_inner,
    test_kron, test_vdot, test_norm_dim,
]

for t in TESTS:
    run(t.__name__, t)

npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"\nPASS {npass}/{len(RESULTS)}")
for name, s, e in RESULTS:
    if s != "PASS":
        print(f"  {s} {name}: {e}")
