"""Every number of the paper, rebuilt from results/*.json.

python audit.py               print each text fragment and run the consistency checks
python audit.py --tex FILE    also require each fragment to appear verbatim in FILE

Odds ratios are reported for women aged 40-49 against women aged 50 and over (the scripts
compute the reverse, so the paper gives 1/OR).
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


def r(x, d):
    """Round half up on the printed decimal value."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x, d=1):
    return r(100 * x, d)


def ci(c, d=2):
    return f"[{r(c[0], d)}, {r(c[1], d)}]"


def pw(x):
    s = r(x, 2)
    return r(x, 3) if s == "1.00" else s


def big(n):
    return f"{n:,}"


def p_text(p):
    if p < 1e-4:
        return "p<10^{-4}"
    if p < 0.001:
        return "p<0.001"
    return f"p={r(p, 3) if p < 0.1 else r(p, 2)}"


frags, checks = [], []


def frag(where, text):
    frags.append((where, text))


def check(where, ok, what):
    checks.append((where, bool(ok), what))


e1 = load("e1.json")
e34 = load("e3_e4.json")
pre = load("rsna_prespecified.json")
ph = load("rsna_posthoc_512.json")
ph2 = load("rsna_posthoc_1024.json")
cb = load("cbis.json")
pub = json.load(open(os.path.join(ROOT, "data", "published_counts.json")))

# E1
eq2 = e1["eq2"]["table"]
syn, rp = e1["synthetic"], e1["rsna_profile"]


def cov(setting, method):
    return [x for x in setting["rows"] if x["method"] == method]


frag("abstract", f"{r(eq2['143']['exact'], 2)} with 143 validation cancers, below $10^{{-4}}$ from 2,000")
check("abstract", e1["eq2"]["below_1e4_from"] <= 2000, "Eq. (2) below 1e-4 from N = 2,000")
mant, ex = f"{eq2['2000']['exact']:.0e}".split("e")
frag("results E1", f"{r(eq2['143']['exact'], 2)}, {r(eq2['300']['exact'], 2)}, {r(eq2['1000']['exact'], 3)} "
     f"and ${mant}\\times10^{{{int(ex)}}}$ for 143, 300, 1,000 and 2,000 cancers")
frag("results E1", f"below one half for every $N\\ge{e1['eq2']['below_half_from']}$")
for row in cov(syn, "Pooled"):
    eq = eq2[str(row["N"])]["exact"]
    check("results E1", abs(row["coverage"] - eq) < 3 * (eq * (1 - eq) / e1["replicates"]) ** 0.5 + 1e-3,
          f"simulated pooled coverage {row['coverage']} agrees with Eq. (2) {eq:.4f} at N = {row['N']}")
frag("results E1", f"with {big(5000)} cancers the bound exceeds $\\Sens_T$ by {pct(cov(syn, 'Pooled')[-1]['mean_diff'])} points on average")
wald = cov(syn, "Wald")
under = [x["coverage"] for x in wald if x["coverage"] < 0.95]
check("results E1", wald[-1]["N"] == 5000 and wald[-1]["coverage"] >= 0.95 and len(under) == len(wald) - 1,
      "Wald under-covers below 5,000 cancers only")
frag("results E1", f"here under-covers below 5,000 cancers ({r(min(under), 3)} to {r(max(under), 3)})")
check("results E1", min(x["coverage"] for x in cov(rp, "Wald")) >= 0.95, "Wald under-coverage is specific to the synthetic setting")
js, jr = [x["coverage"] for x in cov(syn, "Jeffreys")], [x["coverage"] for x in cov(rp, "Jeffreys")]
frag("results E1", f"covers in {r(min(js), 3)} to {r(max(js), 3)} of replicates ({r(min(jr), 3)} to {r(max(jr), 3)} in the RSNA setting)")
emin = min(x["coverage"] for x in cov(syn, "Exact") + cov(rp, "Exact"))
emin_t = str(Decimal(repr(emin)).quantize(Decimal("0.001"), rounding="ROUND_DOWN"))
ex = cov(syn, "Exact")
frag("results E1", f"at least {emin_t} of replicates in both settings but lies {pct(-ex[0]['mean_diff'])} points below the truth "
     f"for 143 cancers and {pct(-ex[-1]['mean_diff'])} for 5,000")
frag("E1 setup", f"$\\Sens_S={r(syn['sens_source'], 3)}$ and $\\Sens_T={r(syn['sens_target'], 3)}$")
frag("E1 setup", "$q=(" + ",".join(f"{v:.3f}" for v in syn["q"][:2]) + ",$ $" + ",".join(f"{v:.3f}" for v in syn["q"][2:]) + ")$")
frag("Fig. 1 caption", f"E1, synthetic setting: coverage of $\\Sens_T$ by 95\\% lower bounds (markers: {big(e1['replicates'])} replicates per size")
frag("E1 setup", "$s=(" + ",".join(f"{v:.2f}" for v in syn["s"][:2]) + ",$ $" + ",".join(f"{v:.2f}" for v in syn["s"][2:]) + ")$")
frag("E1 setup", "$p=(" + ",".join(f"{v:.2f}" for v in syn["p"]) + ")$")
s2, p2 = e1["rsna_profile"]["s"], e1["rsna_profile"]["p"]
cnt = pre["flag10"]
check("E1 setup", len(s2) == 3 and abs(sum(p2) - 1) < 1e-12, "RSNA setting: three strata, a cancer mix")
check("E1 setup", abs(e1["rsna_profile"]["q"][0] - (syn["q"][0] + syn["q"][1])) < 1e-12 and e1["rsna_profile"]["q"][1:] == syn["q"][2:],
      "RSNA setting: Korean mix merged to A+B, C, D")

# E3, E4
kw, bd = e34["kwon2024"], e34["badawy2023"]
frag("E3 setup", f"{big(pub['kwon2024']['screens'])} Korean women (first screening, mean age {pub['kwon2024']['mean_age']}) with {pub['kwon2024']['cancers']} cancers")
frag("E3 setup", "{}/{} (non-dense), {}/{} (heterogeneously dense) and {}/{} (extremely dense)".format(
    *[v for kn in zip(pub["kwon2024"]["k"], pub["kwon2024"]["n"]) for v in kn]))
frag("E3 setup", "A {}/{}, B {}/{}, C {}/{}, D {}/{}".format(
    *[v for kn in zip(pub["badawy2023"]["k"], pub["badawy2023"]["n"]) for v in kn]))
frag("E3 setup", f"on {sum(pub['badawy2023']['n'])} cancers in Egypt")
frag("results E3", f"non-dense {pct(kw['source_mix'][0])}\\% vs {pct(kw['target_mix'][0])}\\%")
frag("results E3", f"from {pct(kw['pooled'])}\\% to {pct(kw['point'])}\\%")
frag("results E3", f"Jeffreys certificate ({pct(kw['jeffreys'])}\\%) exceeds the pooled bound ({pct(kw['pooled_lb'])}\\%)")
check("results E3", kw["jeffreys"] > kw["pooled_lb"], "Kwon: Jeffreys above the pooled bound")
frag("results E3", f"({pct(bd['source_mix'][3])}\\% vs {pct(bd['target_mix'][3])}\\%)")
frag("results E3", f"falls from {pct(bd['pooled'])}\\% to {pct(bd['point'])}\\%")
frag("results E3", f"Jeffreys certificate ({pct(bd['jeffreys'])}\\%) lies {pct(bd['pooled_lb'] - bd['jeffreys'])} points "
     f"below the pooled bound ({pct(bd['pooled_lb'])}\\%)")
frag("results E3", f"Exact certificates are looser ({pct(kw['exact'])}\\%, {pct(bd['exact'])}\\%)")
e4 = e34["e4"]
c4, s4 = e4["cancers"], e4["screens_millions"]
eps = sorted((float(k) for k in e34["e4"]["cancers"]), reverse=True)
check("E4", eps == [0.10, 0.05, 0.03], "E4 tolerances 10, 5 and 3 points")
check("E4", abs(e4["target_sens"] - kw["point"]) < 1e-12, "E4 target sensitivity is the transported Kwon point value")
frag("results E4", f"detected at the transported rate ({pct(e4['target_sens'])}\\%)")
thousands = lambda m: big(int(Decimal(repr(m * 1000)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) * 1000)
frag("results E4", f"within 10, 5 and 3 points of it with {big(c4['0.1'])}, {big(c4['0.05'])} and {big(c4['0.03'])} cancers, about "
     f"{thousands(s4['0.1'])}, {thousands(s4['0.05'])} and {thousands(s4['0.03'])} screens at "
     f"{r(e4['cancers_per_1000_screens'], 2)} cancers per 1,000")

# E2, RSNA counts: the post hoc run must reproduce the pre-specified aggregate counts
for key, a, b in (("old_to_young", "O", "Y"), ("young_to_old", "Y", "O")):
    for m, kk, nn in (("sens", "kC", "nC"), ("spec", "kN", "nN")):
        c_src, c_tgt = ph["counts"][a], ph["counts"][b]
        k_src = c_src[kk] if m == "sens" else [n - k for k, n in zip(c_src[kk], c_src[nn])]
        check("E2 RSNA", pre["flag10"][key][m]["source_counts"] == [k_src, c_src[nn]],
              f"{key} {m}: post hoc counts equal the aggregate counts")


def row(c, recal=None):
    s = " & ".join(r(c[k], 3) for k in ("pooled", "pooled_lb", "point", "jeffreys", "exact"))
    obs_ci = c.get("observed_ci", c.get("obs_ci"))
    s += f" & {r(c['observed'], 3)} {ci(obs_ci)}"
    if recal is None:
        return s + " & --"
    return s + f" & {r(recal['observed'], 3)} {ci(recal['obs_ci'])}"


ca, us = ph["certs_age"], ph["unit_shift"]
for key, label in (("old_to_young", "age $\\ge$50 $\\to$ 40--49"), ("young_to_old", "age 40--49 $\\to$ $\\ge$50")):
    p = pre["flag10"][key]
    for m in ("sens", "spec"):
        for k in ("pooled", "pooled_lb", "point", "jeffreys", "exact", "observed"):
            check("Table 1", abs(p[m][k] - ca[key]["orig"][m][k]) < 2e-3, f"{key} {m} {k}: aggregate and post hoc agree")
    frag("Table 1", f"RSNA, {label} ({p['sens']['n_target']}) & Sens. & {row(p['sens'], ca[key]['recal']['sens'])}")
    frag("Table 1", f" & Spec. & {row(p['spec'], ca[key]['recal']['spec'])}")
frag("Table 1", f"RSNA, unit 49 $\\to$ other units ({us['women']['target_cancers']}) & Sens. & "
     f"{row(us['certs']['orig']['sens'], us['certs']['recal']['sens'])}")
frag("Table 1", f" & Spec. & {row(us['certs']['orig']['spec'], us['certs']['recal']['spec'])}")
frag("Table 1", f"CBIS-DDSM, val.\\ $\\to$ test ({sum(cb['sens']['test_strata'][1])}) & Sens. & {row(cb['sens'])}")
frag("Table 1", f" & Spec. & {row(cb['spec'])}")

# E2 setup
check("E2 setup", sum(us["target_orig"]["nC"]) == us["women"]["target_cancers"] and sum(us["source"]["nC"]) == us["women"]["source_cancers"],
      "unit shift: cancer counts are cancer breasts")
check("E2 setup", sum(v["noncancer_women"] for v in us["per_unit"].values()) == us["women"]["target_noncancer"],
      "unit shift: per-unit women add up to the target women with non-cancer breasts")
frag("E2 setup", f"({big(us['women']['source'])} women: {big(sum(us['source']['nN']))} non-cancer breasts, "
     f"{us['women']['source_cancers']} cancers)")
frag("E2 setup", f"({big(sum(us['target_orig']['nN']))} non-cancer breasts of {big(us['women']['target_noncancer'])} women, "
     f"{us['women']['target_cancers']} cancers)")

# Results, pre-specified age shift
oy = pre["flag10"]["old_to_young"]
frag("results E2", f"the {oy['sens']['n_target']} target cancers gave {r(oy['sens']['observed'], 3)}")
check("results E2", oy["sens"]["observed"] < min(oy["sens"][k] for k in ("pooled_lb", "point", "jeffreys", "exact")),
      "young target sensitivity below all bounds")
frag("results E2", f"too few to reject any at the 5\\% level (${p_text(oy['sens']['p_vs_exact'])}$ against the exact certificate)")
kt_, nt_ = sum(oy["sens"]["target_counts"][0]), sum(oy["sens"]["target_counts"][1])
p_bounds = {k: stats.binom.cdf(kt_, nt_, oy["sens"][k]) for k in ("pooled_lb", "jeffreys", "exact")}
check("results E2", min(p_bounds.values()) > 0.05, f"no bound rejected at 5%: one-sided p {({k: round(v, 3) for k, v in p_bounds.items()})}")
fs, ft = oy["spec"]["flag_rate_source"], oy["spec"]["flag_rate_target"]
check("results E2", all(a < b for a, b in zip(ft, fs)), "fewer flags in every stratum for women aged 40-49")
frag("results E2", "A+B {}\\% vs {}\\%, C {}\\% vs {}\\%, D {}\\% vs {}\\%".format(
    *[pct(v) for ab in zip(ft, fs) for v in ab]))
wl = ph["woman_level_age"]
nc, cc = wl["noncancer_f10"], wl["cancer_f10"]
frag("results E2", f"Mantel--Haenszel odds ratio {r(1 / nc['OR'], 2)}, woman-level bootstrap 95\\% interval "
     f"{r(1 / nc['boot_ci'][1], 2)}--{r(1 / nc['boot_ci'][0], 2)}, permutation ${p_text(nc['perm_p'])}$")
check("results E2", nc["perm_p"] == 0 and nc["perm_resolution"] <= 1e-4, "permutation p below 1e-4 (no permutation as extreme)")
frag("results E2", f"(odds ratio {r(1 / cc['OR'], 2)}, permutation ${p_text(cc['perm_p'])}$)")
yo5 = pre["flag5"]["young_to_old"]["spec"]
yo10 = pre["flag10"]["young_to_old"]["spec"]
for d, lab in ((yo10, "10%"), (yo5, "5%")):
    check("results E2", all(d[k] > d["observed"] for k in ("pooled_lb", "point", "jeffreys", "exact", "any_mix")),
          f"reverse direction at {lab}: every certificate exceeds the observed specificity")
frag("results E2", f"(5\\% operating point: exact certificate {r(yo5['exact'], 3)}, observed {r(yo5['observed'], 3)})")
check("results E2", yo5["exact"] == min(yo5[k] for k in ("pooled_lb", "jeffreys", "exact")), "the exact certificate is the lowest at 5%")
uo = us["certs"]["orig"]
frag("results E2", f"flagged {pct(1 - uo['spec']['observed'])}\\% of non-cancer breasts instead of a transported "
     f"{pct(1 - uo['spec']['point'])}\\%, and detected {round(uo['sens']['observed'] * us['women']['target_cancers'])} "
     f"of {us['women']['target_cancers']} cancers")

# CBIS-DDSM
frag("results E2", f"AUC fell from {r(cb['auc_val'], 3)} to {r(cb['auc_test'], 3)}")
v3, t3 = cb["sens"]["val_strata"], cb["sens"]["test_strata"]
frag("results E2", f"density-3 sensitivity from {v3[0][2]}/{v3[1][2]} to {t3[0][2]}/{t3[1][2]} (Fisher ${p_text(cb['fisher_density3_sens_p'])}$)")
for m in ("sens", "spec"):
    c = cb[m]
    check("results E2", c["exact"] <= c["observed"] < c["jeffreys"], f"CBIS {m}: only the exact certificate covers")

# Check (Eq. 4)
sz = [v for g in ("old", "young") for v in ph["check_size_halves"][g].values()]
ns_sz = sorted(int(n) for n in ph["check_size_halves"]["old"])
frag("results check", f"on $n={ns_sz[0]}$ to {ns_sz[-1]} local women rejected in {pct(min(sz))}\\% to {pct(max(sz))}\\% of resamples (nominal 5\\%)")
ca_ = ph["check_age"]
po, py = ca_["power_old_to_young"], ca_["power_young_to_old"]
frag("results check", f"power {pw(po['100'])}, {pw(po['200'])} and {pw(po['300'])} for $n=100$, 200 and 300 local women "
     f"(reverse: {pw(py['100'])}, {pw(py['200'])}, {pw(py['300'])})")
cp = us["check_power"]
frag("results check", f"unit shift with power {pw(cp['25'])} and {pw(cp['50'])} for $n=25$ and 50")
frag("abstract", f"power {pw(po['200'])} from 200 and {pw(cp['25'])} from 25 local examinations")
pu = us["per_unit"]
big_units = {u: v for u, v in pu.items() if v["noncancer_women"] > 100}
fail = sorted(int(u) for u, v in big_units.items() if abs(v["z"]) > 1.959964)
ok = [u for u, v in big_units.items() if abs(v["z"]) <= 1.959964]
check("results check", fail == [93, 210, 216] and ok == ["170"], f"units with over 100 women failing {fail}, passing {ok}")
check("results check", min(abs(pu[str(u)]["z"]) for u in fail) > 13, "|Z| > 13 for the failing units")
frag("results check", "among units with over 100 women, it failed for 93, 210 and 216 ($|Z|>13$) and passed for 170")
u170 = pu["170"]
frag("results check", f"passed for 170 ($Z={r(u170['z'], 2)}$; specificity {r(u170['spec_observed'], 3)}, "
     f"transported {r(u170['spec_transported'], 3)})")

# Re-setting
ra = ph["recal_age"]


def rates(key):
    rc = ra[key]["recal"]
    return ", ".join(f"{pct(v)}\\%" for v in rc["flag_rate_after"]), ", ".join(f"{pct(v)}\\%" for v in rc["fS"])


a1, b1 = rates("old_to_young")
a2, b2 = rates("young_to_old")
frag("results re-set", f"(age: {a1} vs {b1}; reverse: {a2} vs {b2})")
rl = ph["recal_local_age"]
frag("results re-set", f"specificity stayed within {r(rl['old_to_young']['n200']['spec'][1], 2)}--{r(rl['old_to_young']['n200']['spec'][2], 2)} "
     f"and {r(rl['young_to_old']['n200']['spec'][1], 2)}--{r(rl['young_to_old']['n200']['spec'][2], 2)} in 95\\% of resamples "
     f"(transported {r(ca['old_to_young']['orig']['spec']['point'], 3)}, {r(ca['young_to_old']['orig']['spec']['point'], 3)})")
yr = ca["young_to_old"]["recal"]["sens"]
frag("results re-set", f"sensitivity ({r(yr['observed'], 3)} {ci(yr['obs_ci'])}) covered the transported point ({r(yr['point'], 3)}) "
     f"and exceeded every certificate")
check("results re-set", yr["obs_ci"][0] <= yr["point"] <= yr["obs_ci"][1], "reverse: interval contains the transported point")
check("results re-set", all(yr["observed"] > yr[k] for k in ("pooled_lb", "jeffreys", "exact", "any_mix")),
      "reverse: re-set sensitivity exceeds every certificate")
ur = us["certs"]["recal"]["sens"]
nu = us["women"]["target_cancers"]
frag("results re-set", f"(AUC {r(us['target_orig']['auc'], 3)} vs {r(us['source']['auc'], 3)} on unit 49), "
     f"{round(ur['observed'] * nu)} of {nu} cancers were detected ({r(ur['observed'], 3)}), above the Jeffreys "
     f"({r(ur['jeffreys'], 3)}) and exact ({r(ur['exact'], 3)}) certificates")
check("results re-set", ur["observed"] > ur["jeffreys"] > ur["exact"], "unit shift: re-set sensitivity above both certificates")
frag("abstract", f"({round(ur['observed'] * nu)} of {nu} cancers detected; certified lower bound {r(ur['exact'], 2)})")
oyr = ca["old_to_young"]["recal"]["sens"]
nt = oy["sens"]["n_target"]
frag("results re-set", f"{round(oyr['observed'] * nt)} of {nt} cancers were detected ({r(oyr['observed'], 3)}), with an interval "
     f"containing the transported point, below the exact certificate ({r(oyr['exact'], 3)}; ${p_text(oyr['p_vs_exact'])}$)")
check("results re-set", oyr["obs_ci"][0] <= oyr["point"] <= oyr["obs_ci"][1] and oyr["observed"] < oyr["exact"],
      "age: interval contains the point, observed below the exact certificate")

# The post hoc certificates (computed on Kaggle) must equal those of the current dtransport code
sys.path.insert(0, ROOT)
import dtransport as dt  # noqa: E402


def recheck(src, tgt, stored, where):
    for m, k_src, n_src, q in (("sens", src["kC"], src["nC"], tgt["nC"]),
                               ("spec", [n - k for k, n in zip(src["kN"], src["nN"])], src["nN"], tgt["nN"])):
        new = dt.all_bounds(k_src, n_src, q)
        ok = all(abs(new[x] - stored[m][x]) < 1e-9 for x in ("pooled", "pooled_lb", "point", "exact", "any_mix"))
        ok = ok and abs(new["jeffreys"] - stored[m]["jeffreys"]) < 1e-3
        check("code", ok, f"{where} {m}: stored certificates equal the current dtransport code")


for lab, P in (("512", ph), ("1024", ph2)):
    for key in ("old_to_young", "young_to_old"):
        recheck(P["recal_age"][key]["src"], P["recal_age"][key]["orig"], P["certs_age"][key]["orig"], f"{lab} {key}")
    recheck(P["unit_shift"]["source"], P["unit_shift"]["target_orig"], P["unit_shift"]["certs"]["orig"], f"{lab} unit shift")

# 1024 x 1024 rerun
u2 = ph2["unit_shift"]
r2 = u2["certs"]["recal"]["sens"]
frag("limitations", f"(site-1 AUC {r(ph2['auc']['site1'], 2)} vs {r(ph['auc']['site1'], 2)})")
frag("limitations", f"({round(r2['observed'] * u2['women']['target_cancers'])} of {u2['women']['target_cancers']}; "
     f"exact {r(r2['exact'], 3)}, Jeffreys {r(r2['jeffreys'], 3)})")
check("limitations", r2["exact"] <= r2["observed"] < r2["jeffreys"], "1024: exact but not Jeffreys coverage under the unit shift")
z2 = ph2["check_age"]["z_full"]
check("limitations", min(abs(v) for v in z2.values()) > 1.959964, "1024: the check rejects the age shift in both directions")
fail2 = sorted(int(u) for u, v in u2["per_unit"].items() if v["noncancer_women"] > 100 and abs(v["z"]) > 1.959964)
check("limitations", fail2 == fail, f"1024: same units fail the check ({fail2})")
for key in ("old_to_young", "young_to_old"):
    c2 = ph2["certs_age"][key]
    check("limitations", not (c2["orig"]["spec"]["obs_ci"][0] <= c2["orig"]["spec"]["point"] <= c2["orig"]["spec"]["obs_ci"][1])
          and c2["recal"]["spec"]["obs_ci"][0] <= c2["recal"]["spec"]["point"] <= c2["recal"]["spec"]["obs_ci"][1],
          f"1024 {key}: specificity shifted, restored after re-setting")
cc2 = ph2["woman_level_age"]["cancer_f10"]
frag("limitations", f"(odds ratio {r(1 / cc2['OR'], 2)}, ${p_text(cc2['perm_p'])}$)")
check("limitations", cc2["perm_p"] > 0.05, "1024: lower detection in younger women not reproduced")


if __name__ == "__main__":
    tex = None
    if "--tex" in sys.argv:
        tex = open(sys.argv[sys.argv.index("--tex") + 1]).read()
        tex = " ".join(tex.split())
    bad = 0
    for where, text in frags:
        found = "" if tex is None else ("ok  " if " ".join(text.split()) in tex else "MISS")
        bad += found == "MISS"
        print(f"{found}{where:16s} {text}")
    print()
    for where, ok, what in checks:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}{where:16s} {what}")
    print(f"\n{len(frags)} fragments, {len(checks)} checks, {bad} problems")
    sys.exit(1 if bad else 0)
