"""E2, pre-specified RSNA analysis from aggregate counts (site 1, breast level).

Input: data/rsna_counts_flag10.csv and _flag5.csv (age band x density x cancer: breasts, flagged),
printed by the training notebook. Source women >= 50, target 40-49, and the reverse.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dtransport as dt

ROOT = os.path.join(os.path.dirname(__file__), "..")
STRATA = [["A", "B"], ["C"], ["D"]]
OLD, YOUNG = ["50-59", "60-69", "70+"], ["40-49"]


def strat(t, cancer, ages):
    s = t[(t.cancer == cancer) & t.age_band.isin(ages)]
    n = np.array([s[s.density.isin(g)]["size"].sum() for g in STRATA])
    f = np.array([s[s.density.isin(g)]["sum"].sum() for g in STRATA])
    return f, n


def transport(t, src, tgt):
    out = {}
    for cancer, name in ((1, "sens"), (0, "spec")):
        f, n = strat(t, cancer, src)
        ft, nt = strat(t, cancer, tgt)
        k, kt = (f, ft) if cancer else (n - f, nt - ft)
        r = dt.all_bounds(k, n, nt / nt.sum())
        r.update(observed=kt.sum() / nt.sum(), observed_ci=dt.cp_interval(int(kt.sum()), int(nt.sum())),
                 n_target=int(nt.sum()), source_counts=[k.tolist(), n.tolist()], target_counts=[kt.tolist(), nt.tolist()])
        if cancer:
            r["p_vs_exact"] = float(stats.binom.cdf(kt.sum(), nt.sum(), r["exact"]))
        else:
            r["flag_rate_source"] = (f / n).tolist()
            r["flag_rate_target"] = (ft / nt).tolist()
        out[name] = r
    return out


if __name__ == "__main__":
    res = {}
    for flag in ("flag10", "flag5"):
        t = pd.read_csv(os.path.join(ROOT, "data", f"rsna_counts_{flag}.csv"))
        res[flag] = {"old_to_young": transport(t, OLD, YOUNG), "young_to_old": transport(t, YOUNG, OLD)}
    json.dump(res, open(os.path.join(ROOT, "results", "rsna_prespecified.json"), "w"), indent=1)
    for flag, d in res.items():
        for key, r in d.items():
            for m in ("sens", "spec"):
                x = r[m]
                print(f"{flag} {key} {m}: pooled {x['pooled']:.3f} bound {x['pooled_lb']:.3f} point {x['point']:.3f} "
                      f"Jeffreys {x['jeffreys']:.3f} exact {x['exact']:.3f} observed {x['observed']:.3f} "
                      f"[{x['observed_ci'][0]:.3f}, {x['observed_ci'][1]:.3f}]"
                      + (f" p vs exact {x['p_vs_exact']:.3f}" if m == "sens" else ""))
