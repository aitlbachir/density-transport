"""E2, post hoc RSNA analyses from out-of-fold breast scores.

Input: a training run folder with oof_breast.csv and folds.csv (written by training/rsna_notebook.ipynb).
Site 1, women aged 40+, BI-RADS density present, breast level. Strata 0 = A+B, 1 = C, 2 = D.
Flags: per fold, the threshold flagging 10% (or 5%) of that fold's non-cancer breasts.

Usage: python experiments/rsna_posthoc.py RUN_DIR LABEL
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dtransport as dt

SEED = 20261019
STRAT = {"A": 0, "B": 0, "C": 1, "D": 2}


def scenario(src, tgt):
    """Certificates from source counts, compared with the observed target values."""
    sens = dt.all_bounds(src["kC"], src["nC"], tgt["nC"])
    spec = dt.all_bounds(np.array(src["nN"]) - np.array(src["kN"]), src["nN"], tgt["nN"])
    kT, nT = sum(tgt["kC"]), sum(tgt["nC"])
    sens.update(observed=kT / nT, obs_ci=dt.cp_interval(kT, nT),
                p_vs_exact=float(stats.binom.cdf(kT, nT, sens["exact"])),
                p_vs_jeffreys=float(stats.binom.cdf(kT, nT, sens["jeffreys"])))
    tn, tf = sum(tgt["nN"]), sum(tgt["kN"])
    spec.update(observed=1 - tf / tn, obs_ci=dt.cp_interval(tn - tf, tn))
    return dict(sens=sens, spec=spec)


def load(run_dir):
    B = pd.read_csv(f"{run_dir}/oof_breast.csv")
    F = pd.read_csv(f"{run_dir}/folds.csv")
    for fpr in (0.10, 0.05):
        thr = {k: np.quantile(B[(B.fold == k) & (B.cancer == 0)].score, 1 - fpr) for k in range(5)}
        B[f"f{int(fpr * 100)}"] = (B.score > B.fold.map(thr)).astype(int)
    if "machine_id" in F.columns:
        m = F.groupby(["patient_id", "laterality"]).machine_id.agg(lambda x: x.mode().iloc[0]).rename("machine")
        B = B.merge(m.reset_index(), on=["patient_id", "laterality"], how="left")
    S = B[(B.site_id == 1) & B.density.notna() & (B.age >= 40)].copy()
    S["s"] = S.density.map(STRAT).astype(int)
    S["older"] = (S.age >= 50).astype(int)
    auc = dict(all=roc_auc_score(B.cancer, B.score), site1=roc_auc_score(B[B.site_id == 1].cancer, B[B.site_id == 1].score))
    return B, S, auc


def counts(D, flag="f10"):
    c = lambda q, col: [int(q[q.s == s][col].sum()) for s in range(3)]
    n = lambda q: [int((q.s == s).sum()) for s in range(3)]
    C, N = D[D.cancer == 1], D[D.cancer == 0]
    return dict(kC=c(C, flag), nC=n(C), kN=c(N, flag), nN=n(N))


def woman_tables(D, flag):
    ids = np.array(sorted(D.patient_id.unique()))
    idx = {p: i for i, p in enumerate(ids)}
    Fw = np.zeros((len(ids), 3))
    Nw = np.zeros((len(ids), 3))
    for p, s, f in zip(D.patient_id, D.s, D[flag]):
        Nw[idx[p], s] += 1
        Fw[idx[p], s] += f
    g = D.groupby("patient_id").older.first().reindex(ids).to_numpy().astype(float)
    return Fw, Nw, g


def mh_or(Fw, Nw, G):
    """Mantel-Haenszel odds ratio (older vs younger) over strata; G: (R, W) group indicators."""
    a = G @ Fw
    b = G @ (Nw - Fw)
    c = (1 - G) @ Fw
    d = (1 - G) @ (Nw - Fw)
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.nansum(np.where(n > 0, a * d / n, 0), axis=-1)
        den = np.nansum(np.where(n > 0, b * c / n, 0), axis=-1)
        return num / den


def woman_age_effect(D, flag, rng, B_boot=2000, P_perm=10000):
    """MH odds ratio with a bootstrap over women and a permutation of age group between women."""
    Fw, Nw, g = woman_tables(D, flag)
    W = len(g)
    orv = float(mh_or(Fw, Nw, g[None])[0])
    boots = []
    for _ in range(B_boot // 500):
        ix = rng.integers(0, W, (500, W))
        Fb = np.stack([Fw[i] for i in ix])
        Nb = np.stack([Nw[i] for i in ix])
        gb = g[ix]
        a = np.einsum("rw,rws->rs", gb, Fb)
        b = np.einsum("rw,rws->rs", gb, Nb - Fb)
        c = np.einsum("rw,rws->rs", 1 - gb, Fb)
        d = np.einsum("rw,rws->rs", 1 - gb, Nb - Fb)
        n = a + b + c + d
        with np.errstate(divide="ignore", invalid="ignore"):
            boots.append(np.nansum(np.where(n > 0, a * d / n, 0), 1) / np.nansum(np.where(n > 0, b * c / n, 0), 1))
    boots = np.concatenate(boots)
    lb = np.log(boots[np.isfinite(boots) & (boots > 0)])
    perms = []
    for _ in range(P_perm // 1000):
        Gp = np.stack([rng.permutation(g) for _ in range(1000)])
        perms.append(mh_or(Fw, Nw, Gp))
    lp = np.log(np.clip(np.concatenate(perms), 1e-12, 1e12))
    lo = np.log(orv)
    return dict(women=int(W), OR=orv, boot_ci=[float(np.exp(np.quantile(lb, .025))), float(np.exp(np.quantile(lb, .975)))],
                perm_p=float((np.abs(lp) >= abs(lo) - 1e-12).mean()), perm_resolution=1 / len(lp))


def z_check(L, fS, NS):
    """Eq. (4). L: local non-cancer breasts (patient_id, s, flag); fS, NS: source flag rates and breast counts."""
    E = np.asarray(fS)[L.s.to_numpy()]
    dev = pd.Series(L.flag.to_numpy() - E).groupby(L.patient_id.to_numpy()).sum().to_numpy()
    m = np.bincount(L.s.to_numpy(), minlength=3)
    var = (dev ** 2).sum() + (m ** 2 * np.asarray(fS) * (1 - np.asarray(fS)) / np.asarray(NS)).sum()
    return float(dev.sum() / np.sqrt(var))


def src_rates(Dsrc, flag="f10"):
    N = Dsrc[Dsrc.cancer == 0]
    NS = np.array([(N.s == s).sum() for s in range(3)])
    fS = np.array([N[N.s == s][flag].sum() for s in range(3)]) / NS
    return fS, NS


def check_power(Dsrc, Dtgt, ns, rng, R=2000, flag="f10"):
    """Share of R samples (without replacement) of n target non-cancer women with |Z| > 1.96."""
    fS, NS = src_rates(Dsrc, flag)
    L = Dtgt[Dtgt.cancer == 0]
    w = L.patient_id.to_numpy()
    s = L.s.to_numpy()
    dev = L[flag].to_numpy() - fS[s]
    ids, inv = np.unique(w, return_inverse=True)
    d = np.bincount(inv, weights=dev, minlength=len(ids))
    m = np.zeros((len(ids), 3))
    np.add.at(m, (inv, s), 1)
    c = fS * (1 - fS) / NS
    out = {}
    for n in ns:
        if n > len(ids):
            continue
        pick = np.argsort(rng.random((R, len(ids))), axis=1)[:, :n]
        z = d[pick].sum(1) / np.sqrt((d[pick] ** 2).sum(1) + (m[pick].sum(1) ** 2 * c).sum(1))
        out[n] = float((np.abs(z) > 1.959964).mean())
    return out


def check_size_halves(D, ns, rng, R=2000, flag="f10"):
    """Size of the check: split the non-cancer women of one group into random halves,
    estimate source rates on one half and draw n local women from the other."""
    L = D[D.cancer == 0]
    ids, inv = np.unique(L.patient_id.to_numpy(), return_inverse=True)
    s = L.s.to_numpy()
    F = np.zeros((len(ids), 3))
    M = np.zeros((len(ids), 3))
    np.add.at(F, (inv, s), L[flag].to_numpy())
    np.add.at(M, (inv, s), 1)
    W = len(ids)
    out = {}
    for n in ns:
        rej = 0
        for _ in range(R):
            perm = rng.permutation(W)
            a, b = perm[: W // 2], perm[W // 2:]
            NS = M[a].sum(0)
            fS = F[a].sum(0) / NS
            pick = rng.choice(b, n, replace=False)
            u = (F[pick] - M[pick] * fS).sum(1)
            mm = M[pick].sum(0)
            z = u.sum() / np.sqrt((u ** 2).sum() + (mm ** 2 * fS * (1 - fS) / NS).sum())
            rej += abs(z) > 1.959964
        out[n] = rej / R
    return out


def recal_full(Dsrc, Dtgt, flag="f10"):
    """Re-set the threshold so that target non-cancer breasts are flagged at the transported rate rho."""
    fS, _ = src_rates(Dsrc, flag)
    TN = Dtgt[Dtgt.cancer == 0]
    rT = np.array([(TN.s == s).sum() for s in range(3)], float)
    rT /= rT.sum()
    rho = float((rT * fS).sum())
    t0 = np.quantile(TN.score, 1 - rho)
    t = {k: t0 for k in range(5)}
    c = counts(Dtgt.assign(rf=(Dtgt.score > t0).astype(int)), "rf")
    c.update(rho=rho, fS=fS.tolist(), rT=rT.tolist(),
             flag_rate_after=[k / n if n else None for k, n in zip(c["kN"], c["nN"])])
    return c, t, rho


def recal_local(Dtgt, rho, ns, rng, R=2000):
    """Threshold from n sampled local non-cancer women; target sensitivity and specificity
    as [mean, 2.5th, 97.5th percentile] over R samples."""
    TN = Dtgt[Dtgt.cancer == 0]
    TC = Dtgt[Dtgt.cancer == 1]
    women = TN.patient_id.unique()
    by = {p: g.score.to_numpy() for p, g in TN.groupby("patient_id")}
    sc, sn = np.sort(TC.score.to_numpy()), np.sort(TN.score.to_numpy())
    out = {}
    for n in ns:
        if n > len(women):
            continue
        sens, spec = [], []
        for _ in range(R):
            t = np.quantile(np.concatenate([by[p] for p in rng.choice(women, n, replace=False)]), 1 - rho)
            sens.append(1 - np.searchsorted(sc, t, side="right") / len(sc))
            spec.append(np.searchsorted(sn, t, side="right") / len(sn))
        q = lambda v: [float(np.mean(v)), float(np.quantile(v, .025)), float(np.quantile(v, .975))]
        out[f"n{n}"] = dict(sens=q(sens), spec=q(spec))
    return out


def analyse(run_dir, label):
    rng = np.random.default_rng(SEED)
    B, S, auc = load(run_dir)
    Y, O = S[S.older == 0], S[S.older == 1]
    res = dict(label=label, auc=auc, women=dict(Y=int(Y.patient_id.nunique()), O=int(O.patient_id.nunique())),
               counts=dict(Y=counts(Y), O=counts(O)))
    N, C = S[S.cancer == 0], S[S.cancer == 1]
    res["woman_level_age"] = dict(noncancer_f10=woman_age_effect(N, "f10", rng), noncancer_f5=woman_age_effect(N, "f5", rng),
                                  cancer_f10=woman_age_effect(C, "f10", rng))
    fO, nO = src_rates(O)
    fY, nY = src_rates(Y)
    LY = Y[Y.cancer == 0][["patient_id", "s", "f10"]].rename(columns={"f10": "flag"})
    LO = O[O.cancer == 0][["patient_id", "s", "f10"]].rename(columns={"f10": "flag"})
    res["check_age"] = dict(z_full=dict(old_to_young=z_check(LY, fO, nO), young_to_old=z_check(LO, fY, nY)),
                            power_old_to_young=check_power(O, Y, (100, 200, 300, 500), rng),
                            power_young_to_old=check_power(Y, O, (100, 200, 300, 500), rng),
                            same_population_old=check_power(O, O, (100, 200, 300, 500), rng),
                            same_population_young=check_power(Y, Y, (100, 200, 300, 500), rng))
    res["recal_age"], res["recal_local_age"], res["certs_age"] = {}, {}, {}
    for key, src, tgt in (("old_to_young", O, Y), ("young_to_old", Y, O)):
        c, t, rho = recal_full(src, tgt)
        res["recal_age"][key] = dict(src=counts(src), orig=counts(tgt), recal=c)
        res["recal_local_age"][key] = recal_local(tgt, rho, (200, 300, 500), rng)
        res["certs_age"][key] = dict(orig=scenario(counts(src), counts(tgt)), recal=scenario(counts(src), c))
    if "machine" in S.columns and (S.machine != 49).any():
        M49, MO = S[S.machine == 49], S[S.machine != 49]
        c, t, rho = recal_full(M49, MO)
        dv = dict(women=dict(source=int(M49.patient_id.nunique()), target_noncancer=int(MO[MO.cancer == 0].patient_id.nunique()),
                             target_cancers=int(MO.cancer.sum()), source_cancers=int(M49.cancer.sum())),
                  source=dict(**counts(M49), auc=roc_auc_score(M49.cancer, M49.score)),
                  target_orig=dict(**counts(MO), auc=roc_auc_score(MO.cancer, MO.score)), recal=c)
        fM, nM = src_rates(M49)
        per = {}
        for m, g in MO.groupby("machine"):
            L = g[g.cancer == 0][["patient_id", "s", "f10"]].rename(columns={"f10": "flag"})
            rT = np.array([(L.s == s).sum() for s in range(3)], float)
            per[str(int(m))] = dict(noncancer_women=int(L.patient_id.nunique()), cancers=int(g.cancer.sum()),
                                    z=z_check(L, fM, nM) if len(L) else None,
                                    spec_observed=float(1 - L.flag.mean()) if len(L) else None,
                                    spec_transported=float((rT / rT.sum() * (1 - fM)).sum()) if len(L) else None)
        dv["per_unit"] = per
        dv["check_power"] = check_power(M49, MO, (25, 50, 100), rng)
        dv["recal_local"] = recal_local(MO, rho, (100, 200, 300), rng)
        dv["certs"] = dict(orig=scenario(counts(M49), counts(MO)), recal=scenario(counts(M49), c))
        res["unit_shift"] = dv
    rng2 = np.random.default_rng(SEED + 1)
    res["check_size_halves"] = dict(old=check_size_halves(O, (100, 200, 300), rng2),
                                    young=check_size_halves(Y, (100, 200, 300), rng2))
    return res


if __name__ == "__main__":
    run_dir, label = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "results")
    r = analyse(run_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rsna_posthoc_{label}.json")
    json.dump(r, open(path, "w"), indent=1, default=float)
    print("written", path)
    print(json.dumps(r, default=float))
