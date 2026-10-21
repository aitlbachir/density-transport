"""E1: coverage of one-sided 95% lower bounds for the target sensitivity.

Setting 1: stratum sensitivities s = (0.95, 0.90, 0.80, 0.65), source mix p = (0.15, 0.45, 0.32, 0.08),
target = Korean national cancer mix (Park et al. 2020).
Setting 2: s and p taken from the RSNA model (site 1, women 40+, strata A+B, C, D), same target mix
merged to 3 strata.
Each replicate draws N cancers (multinomial over strata), then detections per stratum.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dtransport as dt

R = 10_000
NS = [143, 300, 1000, 2000, 5000]
JEFF_DRAWS = 4000
CHUNK = 500
Q_KOR4 = np.array([0.148, 0.235, 0.378, 0.239])
Q_KOR3 = np.array([0.148 + 0.235, 0.378, 0.239])


def lower_cp(k, n, a):
    k, n = np.asarray(k, float), np.asarray(n, float)
    out = np.zeros(np.broadcast(k, n).shape)
    ok = (k > 0) & (n > 0)
    out[ok] = stats.beta.ppf(a, k[ok], (n - k + 1)[ok])
    return out


def bounds(K, n, q, rng, delta=dt.DELTA):
    """K, n: (R, G) arrays. Returns dict of (R,) lower bounds for the target sensitivity."""
    G = n.shape[1]
    has = n > 0
    qq = np.where(has, q, 0.0)
    qq = qq / qq.sum(1, keepdims=True)
    s = np.where(has, K / np.maximum(n, 1), 0.0)
    out = {"Pooled": lower_cp(K.sum(1), n.sum(1), delta)}
    se = np.sqrt(np.sum(qq ** 2 * s * (1 - s) / np.maximum(n, 1), 1))
    out["Wald"] = (qq * s).sum(1) - stats.norm.ppf(1 - delta) * se
    a = dt.sidak(delta, G)
    out["Exact"] = (qq * lower_cp(K, n, a)).sum(1)
    jef = np.empty(len(K))
    for i in range(0, len(K), CHUNK):
        k, m, w = K[i:i + CHUNK], n[i:i + CHUNK], qq[i:i + CHUNK]
        S = rng.beta(k + 0.5, m - k + 0.5, size=(JEFF_DRAWS,) + k.shape)
        jef[i:i + CHUNK] = np.quantile((S * w).sum(-1), delta, axis=0)
    out["Jeffreys"] = jef
    return out


def simulate(s, p, q, seed):
    rng = np.random.default_rng(seed)
    truth = float(q @ s)
    rows = []
    for N in NS:
        n = rng.multinomial(N, p, size=R)
        K = rng.binomial(n, s)
        for m, lb in bounds(K, n, q, rng).items():
            rows.append(dict(N=N, method=m, coverage=float(np.mean(lb <= truth)), mean_diff=float(np.mean(lb - truth))))
    return dict(s=s.tolist(), p=p.tolist(), q=q.tolist(), sens_source=float(p @ s), sens_target=truth, rows=rows)


def eq2(sens_s, sens_t):
    grid = np.unique(np.round(np.geomspace(120, 6500, 60)).astype(int))
    curve = [(int(N), dt.pooled_coverage(int(N), sens_s, sens_t)[0]) for N in grid]
    table = {N: dt.pooled_coverage(N, sens_s, sens_t) for N in NS}
    cov = np.array([dt.pooled_coverage(N, sens_s, sens_t)[0] for N in range(1, 6001)])
    below_half = int(next(N for N in range(1, 6001) if (cov[N - 1:] < 0.5).all()))
    below_1e4 = int(next(N for N in range(1, 6001) if (cov[N - 1:] < 1e-4).all()))
    return dict(table={str(N): dict(exact=v[0], normal=v[1]) for N, v in table.items()}, curve=curve,
                below_half_from=below_half, below_1e4_from=below_1e4)


def rsna_profile(counts_csv):
    t = pd.read_csv(counts_csv)
    t = t[(t.cancer == 1) & (t.age_band != "<40")]
    groups = [["A", "B"], ["C"], ["D"]]
    k = np.array([t[t.density.isin(g)]["sum"].sum() for g in groups], float)
    n = np.array([t[t.density.isin(g)]["size"].sum() for g in groups], float)
    return k / n, n / n.sum()


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    s1, p1 = np.array([0.95, 0.90, 0.80, 0.65]), np.array([0.15, 0.45, 0.32, 0.08])
    sim1 = simulate(s1, p1, Q_KOR4, seed=2027)
    s2, p2 = rsna_profile(os.path.join(root, "data", "rsna_counts_flag10.csv"))
    sim2 = simulate(s2, p2, Q_KOR3, seed=2028)
    res = dict(replicates=R, jeffreys_draws=JEFF_DRAWS, synthetic=sim1, rsna_profile=sim2,
               eq2=eq2(sim1["sens_source"], sim1["sens_target"]))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(root, "results", "e1.json"), "w"), indent=1)
    for name in ("synthetic", "rsna_profile"):
        print(name, "Sens_S %.3f Sens_T %.3f" % (res[name]["sens_source"], res[name]["sens_target"]))
        for r in res[name]["rows"]:
            print("  N=%5d %-8s coverage %.4f  mean(bound - truth) %+.4f" % (r["N"], r["method"], r["coverage"], r["mean_diff"]))
    print("Eq. (2):", {k: round(v["exact"], 6) for k, v in res["eq2"]["table"].items()},
          "| below 0.5 from N =", res["eq2"]["below_half_from"], "| below 1e-4 from N =", res["eq2"]["below_1e4_from"])
