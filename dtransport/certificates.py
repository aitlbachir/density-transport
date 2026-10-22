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
    """All G strata are kept, as in Eq. (3) and Proposition 2: a stratum without source cases has
    l_d = 0 (exact) and S_d ~ Beta(1/2, 1/2) (Jeffreys); q is only normalised to sum to one."""
    k, n, q = (np.asarray(x, float) for x in (k, n, q))
    return k, n, q / q.sum()


def point(k, n, q):
    """Point value sum_d q_d k_d / n_d; undefined when a stratum with q_d > 0 has no source cases,
    so such strata are dropped and q renormalised (point value only, never a certificate)."""
    k, n, q = _clean(k, n, q)
    keep = n > 0
    return float(q[keep] @ (k[keep] / n[keep]) / q[keep].sum())


def exact(k, n, q, delta=DELTA):
    """Eq. (3): sum_d q_d l_d with Sidak-level Clopper-Pearson bounds over all G strata."""
    k, n, q = _clean(k, n, q)
    a = sidak(delta, len(k))
    return float(q @ np.array([cp_lower(ki, ni, a) for ki, ni in zip(k, n)]))


def any_mix(k, n, delta=DELTA):
    """Eq. (3): min_d l_d over all G strata."""
    k, n, _ = _clean(k, n, np.ones(len(k)))
    a = sidak(delta, len(k))
    return min(cp_lower(ki, ni, a) for ki, ni in zip(k, n))


def jeffreys(k, n, q, delta=DELTA, draws=4_000_000, seed=20261019, rng=None):
    """delta-quantile of sum_d q_d S_d, S_d ~ Beta(k_d + 1/2, n_d - k_d + 1/2), all G strata."""
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
    """Direct standardisation with a delta-method (Wald) bound (comparator; strata without
    source cases are dropped and q renormalised, as standardisation requires)."""
    k, n, q = _clean(k, n, q)
    keep = n > 0
    k, n, q = k[keep], n[keep], q[keep] / q[keep].sum()
    s = k / n
    se = np.sqrt(np.sum(q ** 2 * s * (1 - s) / n))
    return float(q @ s - stats.norm.ppf(1 - delta) * se)


def all_bounds(k, n, q, delta=DELTA, **jeff):
    p, plb = pooled(k, n, delta)
    return dict(pooled=p, pooled_lb=plb, point=point(k, n, q), jeffreys=jeffreys(k, n, q, delta, **jeff),
                exact=exact(k, n, q, delta), any_mix=any_mix(k, n, delta))


def local_cancers_needed(sens, eps, delta=DELTA, n_max=100_000):
    """Smallest N for which the one-sided Clopper-Pearson bound on N local cancers, with the
    expected sens * N of them detected (continuous in k), lies within eps of sens."""
    def gap(N):
        k = sens * N
        return sens - float(stats.beta.ppf(delta, k, N - k + 1))
    N = next(N for N in range(2, n_max) if gap(N) <= eps)
    assert all(gap(M) <= eps for M in range(N, min(4 * N, n_max))), "gap not monotone after N"
    return N


def pooled_coverage(N, sens_s, sens_t, delta=DELTA):
    """Eq. (2): exact P(L_N <= Sens_T) for K ~ Bin(N, Sens_S), and its normal approximation."""
    ks = np.arange(N + 1)
    lb = np.where(ks == 0, 0.0, stats.beta.ppf(delta, np.maximum(ks, 1), N - ks + 1))
    k_star = ks[lb <= sens_t].max()
    sigma = np.sqrt(sens_s * (1 - sens_s))
    approx = stats.norm.cdf(stats.norm.ppf(1 - delta) - (sens_s - sens_t) * np.sqrt(N) / sigma)
    return float(stats.binom.cdf(k_star, N, sens_s)), float(approx)
