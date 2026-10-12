import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import math
import litetorch as lt
from litetorch import distributions as D

passed = 0
failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {name} {detail}")

def t(data, shape=None):
    if shape is None:
        shape = [len(data)] if isinstance(data, list) else [1]
        if not isinstance(data, list):
            data = [data]
    return lt.Tensor.from_vector(data, shape)

def mean_of(x):
    d = x.data_ptr() if hasattr(x, 'data_ptr') else None
    import numpy as np
    arr = np.array(x.numpy() if hasattr(x, 'numpy') else list(x))
    return float(arr.mean())

def to_list(x):
    try:
        return list(x.numpy().flatten())
    except Exception:
        pass
    try:
        import numpy as np
        return list(np.array(x.tolist()).flatten())
    except Exception:
        return None

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

N = 20000

def sample_mean_var(dist, n=N):
    s = dist.sample([n])
    vals = to_list(s)
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, v

def test_normal():
    d = D.Normal(t(0.0), t(1.0))
    m, v = sample_mean_var(d)
    check("normal_mean", approx(m, 0.0, 0.05), f"m={m}")
    check("normal_var", approx(v, 1.0, 0.05), f"v={v}")
    lp = to_list(d.log_prob(t(0.0)))[0]
    check("normal_logprob", approx(lp, -0.9189385, 1e-4), f"lp={lp}")
    e = to_list(d.entropy())[0]
    check("normal_entropy", approx(e, 1.4189385, 1e-4), f"e={e}")
    check("normal_batch", d.batch_shape == [1])

def test_uniform():
    d = D.Uniform(t(2.0), t(5.0))
    m, v = sample_mean_var(d)
    check("uniform_mean", approx(m, 3.5, 0.05), f"m={m}")
    check("uniform_var", approx(v, 0.75, 0.05), f"v={v}")
    lp = to_list(d.log_prob(t(3.0)))[0]
    check("uniform_logprob", approx(lp, -math.log(3.0), 1e-4), f"lp={lp}")
    e = to_list(d.entropy())[0]
    check("uniform_entropy", approx(e, math.log(3.0), 1e-4), f"e={e}")

def test_exponential():
    d = D.Exponential(t(2.0))
    m, v = sample_mean_var(d)
    check("exp_mean", approx(m, 0.5, 0.03), f"m={m}")
    check("exp_var", approx(v, 0.25, 0.03), f"v={v}")
    lp = to_list(d.log_prob(t(1.0)))[0]
    check("exp_logprob", approx(lp, math.log(2.0) - 2.0, 1e-4), f"lp={lp}")

def test_laplace():
    d = D.Laplace(t(1.0), t(2.0))
    m, v = sample_mean_var(d)
    check("laplace_mean", approx(m, 1.0, 0.08), f"m={m}")
    check("laplace_var", approx(v, 8.0, 0.4), f"v={v}")

def test_cauchy():
    d = D.Cauchy(t(0.0), t(1.0))
    s = to_list(d.sample([1000]))
    med = sorted(s)[500]
    check("cauchy_median", approx(med, 0.0, 0.15), f"med={med}")
    lp = to_list(d.log_prob(t(0.0)))[0]
    check("cauchy_logprob", approx(lp, -math.log(math.pi), 1e-4), f"lp={lp}")

def test_gamma():
    d = D.Gamma(t(2.0), t(3.0))
    m, v = sample_mean_var(d)
    check("gamma_mean", approx(m, 2.0/3.0, 0.03), f"m={m}")
    check("gamma_var", approx(v, 2.0/9.0, 0.03), f"v={v}")

def test_beta():
    d = D.Beta(t(2.0), t(5.0))
    m, v = sample_mean_var(d)
    check("beta_mean", approx(m, 2.0/7.0, 0.02), f"m={m}")
    s = to_list(d.sample([1000]))
    check("beta_range", all(0 < x < 1 for x in s))

def test_dirichlet():
    d = D.Dirichlet(t([1.0, 2.0, 3.0], [3]))
    s = d.sample([5000])
    vals = to_list(s)
    n = len(vals) // 3
    m0 = sum(vals[i*3] for i in range(n)) / n
    m1 = sum(vals[i*3+1] for i in range(n)) / n
    m2 = sum(vals[i*3+2] for i in range(n)) / n
    check("dirichlet_mean", approx(m0, 1/6, 0.02) and approx(m1, 2/6, 0.02) and approx(m2, 3/6, 0.02),
          f"m=({m0},{m1},{m2})")
    sums = [vals[i*3] + vals[i*3+1] + vals[i*3+2] for i in range(n)]
    check("dirichlet_sum1", all(approx(x, 1.0, 1e-4) for x in sums[:100]))

def test_bernoulli():
    d = D.Bernoulli(t(0.7))
    m, v = sample_mean_var(d)
    check("bernoulli_mean", approx(m, 0.7, 0.02), f"m={m}")
    check("bernoulli_var", approx(v, 0.21, 0.02), f"v={v}")
    s = to_list(d.sample([1000]))
    check("bernoulli_binary", all(x in (0.0, 1.0) for x in s))

def test_categorical():
    d = D.Categorical(t([0.2, 0.5, 0.3], [3]))
    s = to_list(d.sample([10000]))
    c0 = sum(1 for x in s if x == 0.0) / len(s)
    c1 = sum(1 for x in s if x == 1.0) / len(s)
    c2 = sum(1 for x in s if x == 2.0) / len(s)
    check("categorical_probs", approx(c0, 0.2, 0.02) and approx(c1, 0.5, 0.02) and approx(c2, 0.3, 0.02),
          f"p=({c0},{c1},{c2})")
    lp = to_list(d.log_prob(t(1.0)))[0]
    check("categorical_logprob", approx(lp, math.log(0.5), 1e-4), f"lp={lp}")

def test_poisson():
    d = D.Poisson(t(4.0))
    m, v = sample_mean_var(d)
    check("poisson_mean", approx(m, 4.0, 0.1), f"m={m}")
    check("poisson_var", approx(v, 4.0, 0.2), f"v={v}")
    e = to_list(d.entropy())[0]
    check("poisson_entropy", approx(e, 2.08, 0.05), f"e={e}")

def test_geometric():
    d = D.Geometric(t(0.5))
    m, v = sample_mean_var(d)
    check("geometric_mean", approx(m, 1.0, 0.05), f"m={m}")

def test_binomial():
    d = D.Binomial(t(10.0), t(0.4))
    m, v = sample_mean_var(d)
    check("binomial_mean", approx(m, 4.0, 0.1), f"m={m}")
    check("binomial_var", approx(v, 2.4, 0.15), f"v={v}")

def test_negbinomial():
    d = D.NegativeBinomial(t(5.0), t(0.5))
    m, v = sample_mean_var(d)
    check("negbin_mean", approx(m, 5.0, 0.2), f"m={m}")

def test_studentt():
    d = D.StudentT(t(10.0))
    m, v = sample_mean_var(d, n=30000)
    check("studentt_mean", approx(m, 0.0, 0.05), f"m={m}")
    check("studentt_var", approx(v, 10.0/8.0, 0.1), f"v={v}")

def test_chi2():
    d = D.Chi2(t(4.0))
    m, v = sample_mean_var(d)
    check("chi2_mean", approx(m, 4.0, 0.1), f"m={m}")
    check("chi2_var", approx(v, 8.0, 0.4), f"v={v}")

def test_lognormal():
    d = D.LogNormal(t(0.0), t(1.0))
    m, v = sample_mean_var(d, n=30000)
    check("lognormal_mean", approx(m, math.exp(0.5), 0.05), f"m={m}")
    s = to_list(d.sample([1000]))
    check("lognormal_pos", all(x > 0 for x in s))
    e = to_list(d.entropy())[0]
    check("lognormal_entropy", approx(e, 1.4189385, 1e-4), f"e={e}")

def test_gumbel():
    d = D.Gumbel(t(0.0), t(1.0))
    m, v = sample_mean_var(d, n=30000)
    check("gumbel_mean", approx(m, 0.5772, 0.03), f"m={m}")
    check("gumbel_var", approx(v, math.pi**2/6, 0.1), f"v={v}")

def test_weibull():
    d = D.Weibull(t(1.0), t(2.0))
    m, v = sample_mean_var(d)
    check("weibull_mean", approx(m, math.gamma(1.5), 0.03), f"m={m}")
    s = to_list(d.sample([1000]))
    check("weibull_pos", all(x > 0 for x in s))

def test_mvn():
    d = D.MultivariateNormal(t([0.0, 0.0], [2]), t([1.0, 0.5, 0.5, 1.0], [2, 2]))
    s = d.sample([10000])
    vals = to_list(s)
    n = len(vals) // 2
    m0 = sum(vals[i*2] for i in range(n)) / n
    m1 = sum(vals[i*2+1] for i in range(n)) / n
    check("mvn_mean", approx(m0, 0.0, 0.05) and approx(m1, 0.0, 0.05), f"m=({m0},{m1})")
    v0 = sum((vals[i*2] - m0)**2 for i in range(n)) / n
    v1 = sum((vals[i*2+1] - m1)**2 for i in range(n)) / n
    check("mvn_var", approx(v0, 1.0, 0.08) and approx(v1, 1.0, 0.08), f"v=({v0},{v1})")
    lp = to_list(d.log_prob(t([0.0, 0.0], [2])))[0]
    check("mvn_logprob_finite", math.isfinite(lp), f"lp={lp}")

def test_kl():
    p = D.Normal(t(0.0), t(1.0))
    q = D.Normal(t(1.0), t(2.0))
    kl = to_list(D.kl_divergence(p, q))[0]
    expected = math.log(2.0) + (1.0 + 1.0) / 8.0 - 0.5
    check("kl_normal", approx(kl, expected, 1e-4), f"kl={kl} exp={expected}")
    p2 = D.Exponential(t(1.0))
    q2 = D.Exponential(t(2.0))
    kl2 = to_list(D.kl_divergence(p2, q2))[0]
    check("kl_exp", approx(kl2, math.log(0.5) + 2.0 - 1.0, 1e-4), f"kl2={kl2}")

def test_validation():
    try:
        D.Normal(t(0.0), t(-1.0))
        check("normal_neg_scale", False, "should raise")
    except Exception:
        check("normal_neg_scale", True)
    try:
        D.Bernoulli(t(1.5))
        check("bernoulli_bad_p", False, "should raise")
    except Exception:
        check("bernoulli_bad_p", True)

for fn in [test_normal, test_uniform, test_exponential, test_laplace, test_cauchy,
           test_gamma, test_beta, test_dirichlet, test_bernoulli, test_categorical,
           test_poisson, test_geometric, test_binomial, test_negbinomial, test_studentt,
           test_chi2, test_lognormal, test_gumbel, test_weibull, test_mvn, test_kl,
           test_validation]:
    try:
        fn()
    except Exception as e:
        failed += 1
        print(f"ERROR in {fn.__name__}: {e}")

print(f"\nPASS {passed} FAIL {failed}")
sys.exit(1 if failed else 0)
