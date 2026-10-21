"""Add CBIS-DDSM metadata (breast density, view, abnormality id) to the exported predictions.

The row order follows the training script: official mass CSVs, rows with a cropped ROI, margin known
(circumscribed, obscured, microlobulated, ill-defined, spiculated), validation = 15% of training
patients (GroupShuffleSplit, random_state 0).

Usage: python tools/make_cbis_predictions.py RAW_CSV TRAIN_CSV TEST_CSV CROP_DIR OUT_CSV
"""
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MARGINS = ["CIRCUMSCRIBED", "OBSCURED", "MICROLOBULATED", "ILL_DEFINED", "SPICULATED"]
raw_csv, train_csv, test_csv, crop_dir, out_csv = sys.argv[1:6]

pat = re.compile(r"(Mass-(?:Training|Test)_P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO)_\d+)_")
have = {m.group(1) for f in os.listdir(crop_dir) if (m := pat.match(f))}
parts = []
for split, f in (("train", train_csv), ("test", test_csv)):
    d = pd.read_csv(f)
    parts.append(d[d["cropped image file path"].str.split("/").str[0].isin(have)].assign(_split=split))
df = pd.concat(parts, ignore_index=True)
df["malignant"] = df.pathology.str.upper().str.startswith("MALIGNANT").astype(int)
df = df[df["mass margins"].astype(str).str.upper().str.split("-").str[0].isin(MARGINS)].copy()
trv = df[df._split == "train"]
_, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(trv, groups=trv.patient_id))
ref = {"val": trv.iloc[va_i].reset_index(drop=True), "test": df[df._split == "test"].reset_index(drop=True)}

raw = pd.read_csv(raw_csv)
rows = []
for split, r in ref.items():
    p = raw[raw.split == split].sort_values("idx")
    m = r.loc[p.idx.to_numpy()]
    assert (m.patient_id.to_numpy() == p.patient_id.to_numpy()).all() and (m.malignant.to_numpy() == p.y.to_numpy()).all()
    keep = m[["patient_id", "left or right breast", "image view", "abnormality id", "breast_density", "malignant"]].copy()
    for s in range(5):
        keep[f"p_seed{s}"] = p[f"p_seed{s}"].to_numpy()
    rows.append(keep.assign(split=split))
out = pd.concat(rows, ignore_index=True).rename(columns={"left or right breast": "side", "image view": "view",
                                                          "abnormality id": "abnormality_id"})
out = out[["split", "patient_id", "side", "view", "abnormality_id", "breast_density", "malignant"] + [f"p_seed{s}" for s in range(5)]]
out.to_csv(out_csv, index=False, float_format="%.10g")
print(len(out), "rows written to", out_csv, out.groupby("split").size().to_dict())
