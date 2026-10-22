"""E3 and E4: published density-stratified counts transported to the Korean cancer mix,
and the number of local cancers needed to certify the transported sensitivity directly."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dtransport as dt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def e4(s, q, eps_list=(0.10, 0.05, 0.03), delta=dt.DELTA):
    """Local validation: smallest number N of local cancers (arriving at the target mix q, so the
    target sensitivity is P = sum_d q_d s_d) for which a one-sided 95% Clopper-Pearson bound on them
    lies within eps of P, with the expected P N detections (continuous in k)."""
    P = float(q @ s)
    return P, {str(eps): dt.local_cancers_needed(P, eps, delta) for eps in eps_list}


if __name__ == "__main__":
    pub = json.load(open(os.path.join(ROOT, "data", "published_counts.json")))
    q4 = np.array(pub["korean_cancer_mix_4"]["q"])
    q3 = np.array([q4[0] + q4[1], q4[2], q4[3]])
    out = {}
    for key, q in (("kwon2024", q3), ("badawy2023", q4)):
        k, n = np.array(pub[key]["k"]), np.array(pub[key]["n"])
        out[key] = dt.all_bounds(k, n, q)
        out[key]["source_mix"] = (n / n.sum()).tolist()
        out[key]["target_mix"] = q.tolist()
    kw = pub["kwon2024"]
    s = np.array(kw["k"]) / np.array(kw["n"])
    rate = kw["cancers"] / kw["screens"]
    P, need = e4(s, q3)
    out["e4"] = dict(target_sens=P, cancers=need, cancers_per_1000_screens=1000 * rate,
                     screens_millions={e: N / rate / 1e6 for e, N in need.items()})
    json.dump(out, open(os.path.join(ROOT, "results", "e3_e4.json"), "w"), indent=1)
    for key in ("kwon2024", "badawy2023"):
        r = out[key]
        print(key, " ".join(f"{m} {100 * r[m]:.2f}%" for m in ("pooled", "pooled_lb", "point", "jeffreys", "exact", "any_mix")))
    print("E4 cancers", need, "| screens (millions)", {e: round(v, 3) for e, v in out["e4"]["screens_millions"].items()},
          "| rate per 1,000 %.3f" % out["e4"]["cancers_per_1000_screens"])
