"""Collect the CBIS-DDSM validation and test predictions of the five training runs (seeds 0-4)
into one CSV. Run on Kaggle with the training notebook output attached as input.

Usage: python kaggle/export_cbis_predictions.py INPUT_DIR OUT_CSV
"""
import glob
import os
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
parts = []
for split in ("val", "test"):
    base = None
    for s in range(5):
        hits = glob.glob(os.path.join(src, "**", f"mt_s{s}", f"pred_{split}_clean0.csv"), recursive=True)
        assert len(hits) == 1, (split, s, hits)
        p = pd.read_csv(hits[0])[["idx", "patient_id", "y", "p"]].rename(columns={"p": f"p_seed{s}"})
        if base is None:
            base = p
        else:
            assert (base.idx.to_numpy() == p.idx.to_numpy()).all() and (base.y.to_numpy() == p.y.to_numpy()).all()
            base[f"p_seed{s}"] = p[f"p_seed{s}"].to_numpy()
    parts.append(base.assign(split=split))
d = pd.concat(parts, ignore_index=True)
d.to_csv(out, index=False, float_format="%.10g")
print(len(d), "rows written to", out)
print(open(out).read())
