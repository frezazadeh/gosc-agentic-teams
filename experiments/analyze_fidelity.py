"""Cluster-robust analysis of the counterfactual value-fidelity samples.

    python -m experiments.analyze_fidelity
"""
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def rk(x):
    return np.argsort(np.argsort(x))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def main():
    rows = json.load(open(os.path.join(RES, "rev_value_fidelity.json")))
    missions = sorted({r["seed"] for r in rows})
    rng = np.random.default_rng(0)
    out = {"missions": len(missions), "samples": len(rows)}
    ests = ("tom", "tv", "bits", "aoi")
    # --- cluster (mission) bootstrap of rank correlations
    corr = {}
    for kind in ("E", "I", "EI"):
        R = [r for r in rows if r["kind"] == kind]
        for e in ests:
            if e not in R[0]["est"]:
                continue
            for tgt in ("gt_gain", "dT"):
                point = spearman([r["est"][e] for r in R], [r[tgt] for r in R])
                boots = []
                by = {m: [r for r in R if r["seed"] == m] for m in missions}
                for _ in range(2000):
                    pick = rng.choice(missions, len(missions))
                    S = [r for m in pick for r in by[m]]
                    v = spearman([r["est"][e] for r in S], [r[tgt] for r in S])
                    if not np.isnan(v):
                        boots.append(v)
                lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
                corr[f"{kind}/{e}/{tgt}"] = [point, float(lo), float(hi)]
    out["corr_cluster"] = corr
    # --- distribution of measured mission effects
    dist = {}
    for kind in ("E", "I", "EI"):
        d = np.array([r["dT"] for r in rows if r["kind"] == kind])
        se = np.array([r["dT_se"] for r in rows if r["kind"] == kind])
        dist[kind] = {"n": int(d.size), "mean": float(d.mean()),
                      "q": [float(v) for v in np.percentile(d, [5, 25, 50, 75, 95])],
                      "frac_pos": float(np.mean(d > 0)), "frac_neg": float(np.mean(d < 0)),
                      "frac_sig_pos": float(np.mean(d > 2 * se)), "frac_sig_neg": float(np.mean(d < -2 * se)),
                      "mean_se": float(se.mean())}
    out["effects"] = dist
    # --- action-selection regret among {nothing, E, I, EI} at decision points with >=2 options
    pts = {}
    for r in rows:
        pts.setdefault((r["seed"], r["t"], r["agent"]), []).append(r)
    reg = {e: [] for e in ests}
    reg["random"], reg["always_send_all"] = [], []
    for grp in pts.values():
        opts = [("none", 0.0, {e: 0.0 for e in ests})] + [(r["kind"], r["dT"], r["est"]) for r in grp]
        if len(opts) < 3:
            continue
        best = max(o[1] for o in opts)
        for e in ests:
            if all(e in o[2] for o in opts):
                choice = max(opts, key=lambda o: o[2][e])
                reg[e].append(best - choice[1])
        reg["random"].append(best - np.mean([o[1] for o in opts]))
        full = [o for o in opts if o[0] == "EI"] or [opts[-1]]
        reg["always_send_all"].append(best - full[0][1])
    out["regret"] = {k: [float(np.mean(v)), len(v)] for k, v in reg.items() if v}
    json.dump(out, open(os.path.join(RES, "rev_fidelity_analysis.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "corr_cluster"}, indent=1))
    for k, v in corr.items():
        if k.split("/")[0] in ("E", "EI"):
            print(k, [round(x, 2) for x in v])


if __name__ == "__main__":
    main()
