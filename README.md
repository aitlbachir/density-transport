# Density transport of mammography AI performance

Code and aggregate data for

> I. Ait Lbachir and I. Daoudi, "Certifying mammography AI performance for a new population: breast-density
> transport with a cancer-free check," submitted to IEEE ISBI 2027.

A pooled sensitivity describes the breast-density mix of the cancers that produced it. Given
density-stratified counts from a published evaluation and the density mix of a target population, the code
computes finite-sample lower bounds (certificates) for the sensitivity or specificity in the target, and a
check of the transport assumption that uses only local non-cancer examinations.

## Use on your own counts

```python
import numpy as np
import dtransport as dt

k = np.array([14, 42, 40])          # detected cancers per density stratum (source study)
n = np.array([18, 66, 59])          # cancers per stratum (source study)
q = np.array([0.383, 0.378, 0.239]) # density mix of cancers in the target population

print(dt.all_bounds(k, n, q))       # pooled, pooled_lb, point, jeffreys, exact, any_mix
```

`exact` is Eq. (3) of the paper (Sidak-level Clopper-Pearson bounds per stratum, weighted by `q`),
`any_mix` holds for every target mix, and `jeffreys` is the approximately calibrated Monte Carlo version.
The check of Eq. (4) is `z_check` in `experiments/rsna_posthoc.py`.

## Reproduce the paper

```bash
pip install -r requirements.txt
bash run_all.sh            # CPU, about 3 minutes
```

| Paper item | Script | Input | Output |
|---|---|---|---|
| Eq. (2), Fig. 1, E1 (both simulation settings) | `experiments/e1_simulation.py`, `experiments/make_figure.py` | none; `data/rsna_counts_flag10.csv` for the RSNA-profile setting | `results/e1.json`, `figures/fig_coverage.tex` |
| E3, E4 (published evaluations, local validation) | `experiments/e3_published.py` | `data/published_counts.json` | `results/e3_e4.json` |
| E2, RSNA age shift (pre-specified) | `experiments/rsna_prespecified.py` | `data/rsna_counts_flag10.csv`, `data/rsna_counts_flag5.csv` | `results/rsna_prespecified.json` |
| E2, RSNA post hoc (woman-level odds ratios, check, re-setting, unit shift, 1024-px rerun) | `experiments/rsna_posthoc.py` | out-of-fold scores of the training notebook (see below) | `results/rsna_posthoc_512.json`, `results/rsna_posthoc_1024.json` |
| E2, CBIS-DDSM | `experiments/cbis_case.py` | `data/cbis_predictions.csv` | `results/cbis.json` |
| Every number in the paper | `audit.py` | `results/*.json` | printed list; `--tex PAPER.tex` checks the manuscript |

The Monte Carlo steps use fixed seeds, so repeated runs give the same values.

## RSNA data

The RSNA Screening Mammography data may not be redistributed. `data/rsna_counts_*.csv` hold only the
aggregate counts (site 1, breast level, by age band, density and cancer status) printed by the training
notebook. The post hoc analyses need breast-level out-of-fold scores: run `training/rsna_notebook.ipynb`
on Kaggle with the competition data and Theo Viel's 512 x 512 PNGs attached (GPU T4 x2, about 75 minutes),
then

```bash
python experiments/rsna_posthoc.py RUN_DIR 512      # RUN_DIR holds oof_breast.csv and folds.csv
```

The 1024 x 1024 rerun used the same notebook with the 1024 x 1024 PNGs, 12 epochs, batch size 16 and
augmentation on the GPU. `results/rsna_posthoc_*.json` are the outputs of these two runs.

## CBIS-DDSM predictions

`data/cbis_predictions.csv` gives, for each mass in the validation subset (15% of training patients) and
the official test split, the malignancy probability of five EfficientNet-B0 models (ImageNet weights,
512 x 512 ROI crops, 30 epochs, AdamW, learning rate 3e-4, batch size 16, auxiliary BI-RADS shape and
margin heads with weight 0.5, checkpoint with the best validation AUC; seeds 0 to 4). Masses with an
unknown margin are excluded. CBIS-DDSM is distributed by The Cancer Imaging Archive under CC BY 3.0
(R. S. Lee et al., Sci. Data 4, 170177, 2017).

## Layout

```
dtransport/       certificates and Eq. (2)
experiments/      one script per experiment
data/             aggregate counts and CBIS-DDSM predictions
results/          JSON outputs used in the paper
figures/          Fig. 1 (pgfplots)
training/         RSNA training notebook and fold script
kaggle/, tools/   export of the CBIS-DDSM predictions
docs/             RSNA analysis plan
```

## License

MIT (see `LICENSE`).
