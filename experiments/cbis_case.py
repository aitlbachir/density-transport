"""E2, CBIS-DDSM diagnostic case: validation subset (source) to official test split (target).

Input: data/cbis_predictions.csv, malignancy probabilities of five EfficientNet-B0 models (seeds 0-4)
for the validation subset and the test split. Operating point: Youden threshold of the 5-model mean on
the validation subset. Strata: BI-RADS density 1-4.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dtransport as dt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def strata(d, flag):
    out = {}
    for name, y, hit in (("sens", 1, 1), ("spec", 0, 0)):
        k = [int(((d.malignant == y) & (d.breast_density == g) & (flag == hit)).sum()) for g in (1, 2, 3, 4)]
        n = [int(((d.malignant == y) & (d.breast_density == g)).sum()) for g in (1, 2, 3, 4)]
        out[name] = (np.array(k), np.array(n))
    return out


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(ROOT, "data", "cbis_predictions.csv"))
    d["p"] = d[[f"p_seed{s}" for s in range(5)]].mean(axis=1)
    va, te = d[d.split == "val"], d[d.split == "test"]
    fpr, tpr, thr = roc_curve(va.malignant, va.p)
    t = float(thr[np.argmax(tpr - fpr)])
    sv = strata(va, (va.p >= t).astype(int))
    st = strata(te, (te.p >= t).astype(int))
    res = dict(threshold=t, n_val=len(va), n_test=len(te),
               auc_val=roc_auc_score(va.malignant, va.p), auc_test=roc_auc_score(te.malignant, te.p))
    for m in ("sens", "spec"):
        k, n = sv[m]
        kt, nt = st[m]
        r = dt.all_bounds(k, n, nt / nt.sum())
        r.update(observed=kt.sum() / nt.sum(), observed_ci=dt.cp_interval(int(kt.sum()), int(nt.sum())),
                 val_strata=[k.tolist(), n.tolist()], test_strata=[kt.tolist(), nt.tolist()])
        res[m] = r
    k3, n3 = res["sens"]["val_strata"][0][2], res["sens"]["val_strata"][1][2]
    kt3, nt3 = res["sens"]["test_strata"][0][2], res["sens"]["test_strata"][1][2]
    res["fisher_density3_sens_p"] = float(stats.fisher_exact([[k3, n3 - k3], [kt3, nt3 - kt3]])[1])
    json.dump(res, open(os.path.join(ROOT, "results", "cbis.json"), "w"), indent=1, default=float)
    print(f"n val {len(va)}, n test {len(te)}, threshold {t:.4f}, AUC val {res['auc_val']:.4f}, test {res['auc_test']:.4f}")
    for m in ("sens", "spec"):
        r = res[m]
        print(f"{m}: val {r['val_strata']} test {r['test_strata']}")
        print(f"   pooled {r['pooled']:.3f} bound {r['pooled_lb']:.3f} point {r['point']:.3f} Jeffreys {r['jeffreys']:.3f} "
              f"exact {r['exact']:.3f} observed {r['observed']:.3f} [{r['observed_ci'][0]:.3f}, {r['observed_ci'][1]:.3f}]")
    print(f"density 3 sensitivity {k3}/{n3} vs {kt3}/{nt3}, Fisher p = {res['fisher_density3_sens_p']:.5f}")
