"""Lower bounds for sensitivity (or specificity) transported to a new density mix."""
import numpy as np
from scipy import stats

DELTA = 0.05


def cp_lower(k, n, alpha):
    """One-sided Clopper-Pearson lower bound with error alpha."""
    if n <= 0 or k <= 0:
        return 0.0
    return float(stats.beta.ppf(alpha, k, n - k + 1))


def cp_upper(k, n, alpha):
    if n <= 0 or k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def cp_interval(k, n, level=0.95):
    a = (1 - level) / 2
    return cp_lower(k, n, a), cp_upper(k, n, a)


def sidak(delta, G):
    return 1 - (1 - delta) ** (1 / G)


def _clean(k, n, q):
    k, n, q = (np.asarray(x, float) for x in (k, n, q))
    keep = n > 0
    k, n, q = k[keep], n[keep], q[keep]
    return k, n, q / q.sum()


def point(k, n, q):
    k, n, q = _clean(k, n, q)
    return float(q @ (k / n))


def exact(k, n, q, delta=DELTA):
    """Eq. (3): sum_d q_d l_d with Sidak-level Clopper-Pearson bounds."""
    k, n, q = _clean(k, n, q)
    a = sidak(delta, len(k))
    return float(q @ np.array([cp_lower(ki, ni, a) for ki, ni in zip(k, n)]))


def any_mix(k, n, delta=DELTA):
    k, n, _ = _clean(k, n, np.ones(len(k)))
    a = sidak(delta, len(k))
    return min(cp_lower(ki, ni, a) for ki, ni in zip(k, n))


def jeffreys(k, n, q, delta=DELTA, draws=4_000_000, seed=20261019, rng=None):
    """delta-quantile of sum_d q_d S_d, S_d ~ Beta(k_d + 1/2, n_d - k_d + 1/2)."""
    k, n, q = _clean(k, n, q)
    rng = rng if rng is not None else np.random.default_rng(seed)
    s = np.zeros(draws)
    for ki, ni, qi in zip(k, n, q):
        s += qi * rng.beta(ki + 0.5, ni - ki + 0.5, draws)
    return float(np.quantile(s, delta))


def pooled(k, n, delta=DELTA):
    """Common practice: pooled estimate and its one-sided CP bound."""
    K, N = int(np.sum(k)), int(np.sum(n))
    return K / N, cp_lower(K, N, delta)


def wald(k, n, q, delta=DELTA):
    """Direct standardisation with a delta-method (Wald) bound."""
    k, n, q = _clean(k, n, q)
    s = k / n
    se = np.sqrt(np.sum(q ** 2 * s * (1 - s) / n))
    return float(q @ s - stats.norm.ppf(1 - delta) * se)


def all_bounds(k, n, q, delta=DELTA, **jeff):
    p, plb = pooled(k, n, delta)
    return dict(pooled=p, pooled_lb=plb, point=point(k, n, q), jeffreys=jeffreys(k, n, q, delta, **jeff),
                exact=exact(k, n, q, delta), any_mix=any_mix(k, n, delta))


def pooled_coverage(N, sens_s, sens_t, delta=DELTA):
    """Eq. (2): exact P(L_N <= Sens_T) for K ~ Bin(N, Sens_S), and its normal approximation."""
    ks = np.arange(N + 1)
    lb = np.where(ks == 0, 0.0, stats.beta.ppf(delta, np.maximum(ks, 1), N - ks + 1))
    k_star = ks[lb <= sens_t].max()
    sigma = np.sqrt(sens_s * (1 - sens_s))
    approx = stats.norm.cdf(stats.norm.ppf(1 - delta) - (sens_s - sens_t) * np.sqrt(N) / sigma)
    return float(stats.binom.cdf(k_star, N, sens_s)), float(approx)
