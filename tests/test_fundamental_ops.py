import litetorch as lt
import sys

def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond

ok = True

lt.manual_seed(42)
a = lt.randn([3, 4])
lt.manual_seed(42)
b = lt.randn([3, 4])
ok &= check("manual_seed reproducibility", a.to_vector() == b.to_vector())

o = lt.ones([2, 3])
ok &= check("ones", o.to_vector() == [1.0] * 6)

r = lt.arange(0, 10, 2)
ok &= check("arange", r.to_vector() == [0.0, 2.0, 4.0, 6.0, 8.0])

t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.Device("cpu"), False)
mask = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], lt.Device("cpu"), False)
mf = t.masked_fill(mask, -999.0)
ok &= check("masked_fill", mf.to_vector() == [1.0, 2.0, 3.0, -999.0] or mf.to_vector() == [-999.0, 2.0, 3.0, -999.0])

t2 = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], lt.Device("cpu"), False)
row = t2[1]
ok &= check("t[i] indexing", row.to_vector() == [3.0, 4.0])

sl = t2[0:2]
ok &= check("t[a:b] slicing", sl.to_vector() == [1.0, 2.0, 3.0, 4.0])

idx = lt.Tensor.from_vector([2.0, 0.0], [2], lt.Device("cpu"), False)
sel = t2.index_select(0, idx)
ok &= check("index_select", sel.to_vector() == [5.0, 6.0, 1.0, 2.0])

if lt.is_gpu_available():
    g = lt.randn([2, 2], lt.Device("cuda:0"))
    ok &= check("randn GPU", len(g.to_vector()) == 4)
    gm = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], lt.Device("cuda:0"), False)
    gt = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.Device("cuda:0"), False)
    gmf = gt.masked_fill(gm, -1.0)
    v = gmf.to_vector()
    ok &= check("masked_fill GPU", v[0] == -1.0 and v[3] == -1.0 and v[1] == 2.0)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
