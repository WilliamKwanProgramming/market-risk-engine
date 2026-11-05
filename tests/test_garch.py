"""GARCH(1,1)-t: filter, likelihood, estimation (cross-checked against the arch package)."""

import numpy as np
import pytest
from scipy import stats

from mre.garch import (
    GarchParams,
    backcast,
    fit_garch_t,
    garch_filter,
    garch_loglik,
    simulate_garch_t,
)

P = GarchParams(mu=0.0002, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)


def test_backcast_by_hand():
    e = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 0.94, 0.94**2])
    assert backcast(e) == pytest.approx((w * e**2).sum() / w.sum())
    long = np.arange(1.0, 201.0)
    w75 = 0.94 ** np.arange(75)
    assert backcast(long) == pytest.approx((w75 * long[:75] ** 2).sum() / w75.sum())


def test_filter_by_hand():
    p = GarchParams(mu=0.5, omega=0.1, alpha=0.2, beta=0.7, nu=5.0)
    r = np.array([1.5, -0.5, 2.5])
    eps = r - 0.5  # [1, -1, 2]
    b = backcast(eps)
    s1 = 0.1 + 0.2 * b + 0.7 * b
    s2 = 0.1 + 0.2 * 1.0 + 0.7 * s1
    s3 = 0.1 + 0.2 * 1.0 + 0.7 * s2
    s4 = 0.1 + 0.2 * 4.0 + 0.7 * s3
    var, nxt = garch_filter(r, p)
    np.testing.assert_allclose(var, [s1, s2, s3], rtol=1e-14)
    assert nxt == pytest.approx(s4, rel=1e-14)


def test_filter_matches_explicit_loop_and_is_causal():
    r, _ = simulate_garch_t(500, P, np.random.default_rng(0))
    var, nxt = garch_filter(r, P)
    eps = r - P.mu
    b = backcast(eps)
    s = np.empty(501)
    prev_s, prev_e2 = b, b
    for t in range(501):
        s[t] = P.omega + P.alpha * prev_e2 + P.beta * prev_s
        if t < 500:
            prev_s, prev_e2 = s[t], eps[t] ** 2
    np.testing.assert_allclose(var, s[:500], rtol=1e-12)
    assert nxt == pytest.approx(s[500], rel=1e-12)
    # sigma_t^2 must not depend on r_t: change r from day 300 on (the backcast uses the first
    # 75 values only, which are untouched)
    r2 = r.copy()
    r2[300:] *= 3.0
    var2, _ = garch_filter(r2, P)
    np.testing.assert_array_equal(var2[:301], var[:301])
    assert var2[301] != var[301]


def test_loglik_matches_scipy_t_density():
    r, _ = simulate_garch_t(300, P, np.random.default_rng(1))
    var, _ = garch_filter(r, P)
    scale = np.sqrt(var * (P.nu - 2) / P.nu)
    ll = np.sum(stats.t.logpdf((r - P.mu) / scale, P.nu) - np.log(scale))
    assert garch_loglik(r, P) == pytest.approx(ll, rel=1e-12)


def test_simulated_variances_follow_the_recursion():
    r, s2 = simulate_garch_t(400, P, np.random.default_rng(2))
    eps = r - P.mu
    np.testing.assert_allclose(s2[1:], P.omega + P.alpha * eps[:-1] ** 2 + P.beta * s2[:-1],
                               rtol=1e-12)
    z = eps / np.sqrt(s2)
    assert abs(np.mean(z)) < 0.15 and abs(np.std(z) - 1.0) < 0.15


def test_estimates_recover_true_parameters_on_average():
    """Mean of 12 estimates on 4000-day paths. The tolerances are about 3 Monte Carlo standard
    errors of the mean (sd of one estimate at n = 4000: alpha ~0.014, beta ~0.016, nu ~0.6,
    measured in a separate 30-path run at n = 8000 and scaled by sqrt(2))."""
    est = []
    for seed in range(12):
        fit = fit_garch_t(simulate_garch_t(4000, P, np.random.default_rng(100 + seed))[0])
        assert fit.converged and fit.params.persistence < 1.0
        est.append([fit.params.alpha, fit.params.beta, fit.params.nu])
    alpha, beta, nu = np.mean(est, axis=0)
    assert alpha == pytest.approx(P.alpha, abs=0.012)
    assert beta == pytest.approx(P.beta, abs=0.014)
    assert nu == pytest.approx(P.nu, abs=0.6)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_estimates_match_arch_package(seed):
    """Differential test against arch (same likelihood, same backcast).

    Tolerances: our maximised log-likelihood is not below arch's by more than 1e-4 and not above
    it by more than 0.01; alpha and beta within 0.003; nu within 2%; omega within 3%. Both are
    numerical optimisers with their own stopping rules, so exact agreement is not expected.
    """
    arch = pytest.importorskip("arch")
    r, _ = simulate_garch_t(1000, P, np.random.default_rng(seed))
    ours = fit_garch_t(r)
    res = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t",
                          rescale=False).fit(disp="off")
    ll_arch = res.loglikelihood + r.size * np.log(100.0)  # undo the scaling by 100
    assert ll_arch - 1e-4 <= ours.loglik <= ll_arch + 0.01
    mu, omega, alpha, beta, nu = res.params.to_numpy()
    assert ours.params.alpha == pytest.approx(alpha, abs=3e-3)
    assert ours.params.beta == pytest.approx(beta, abs=3e-3)
    assert ours.params.nu == pytest.approx(nu, rel=0.02)
    assert ours.params.omega == pytest.approx(omega / 1e4, rel=0.03)
    assert ours.params.mu == pytest.approx(mu / 100, abs=2e-4)


def test_fit_reports_and_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_garch_t(np.zeros(100))
    with pytest.raises(ValueError):
        fit_garch_t(np.ones(10))
    fit = fit_garch_t(simulate_garch_t(300, P, np.random.default_rng(4))[0], max_iter=2)
    assert not fit.converged  # two iterations are not enough; the flag says so
    assert "Iteration limit" in fit.message
