"""Fresh-seed validation (seeds 3000-3099) of the realistic-resource experiments.

Selection rule (the main evaluation's, fixed in advance): on the evaluation seeds 0-99, take
for each family the operating point with the fewest mean uplink channel uses whose
mean outcome meets the target (1.25 x the ideal-communication completion time in the
search task, 0.90 x its saved fraction in the rescue task); freeze it and re-run it on
the fresh seeds, with common random numbers across methods.

  overhead  H = 16/32/64 bits: GOSC and the periodic family re-selected per H; H = 0
            re-runs the main-evaluation frozen points with the complete cost account;
            search default: conventional schemes too (cost table)
  noise     each value-estimation arm re-selected per setting (the periodic
            comparator is the main-evaluation frozen periodic point, same fresh seeds)
  shared    shared network-wide budgets

    python -m experiments.validate_realistic --only overhead noise shared
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from experiments.run_realistic import ARMS, HEADLINE, OUT, SETTINGS, SHARED
from gosc import run_realistic

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FRESH = range(3000, 3100)


def load_jsonl(name):
    rows = []
    with open(os.path.join(OUT, f"{name}.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main_rows(name):
    with open(os.path.join(RES, f"rev_{name}.json")) as f:
        return json.load(f)["rows"]


def reference(setting):
    """Target of a setting from the main-evaluation ideal-communication (genie) runs."""
    task, st = setting.split("-", 1)
    if task == "search":
        key = {"W100": "W100", "K10": "K10", "clustered": "clustered"}.get(st, "W100")
        g = [r for r in main_rows("genie_ref") if r["param"] == key]
        return "completion_time", 1.25 * float(np.mean([r["completion_time"] for r in g]))
    key = {"W100": "W100", "W50": "W50", "snr-5": "snr-5"}.get(st, "W100")
    g = [r for r in main_rows("rescue") if r["scheme"] == "genie" and r["param"] == f"{key}/ref"]
    return "saved_fraction", 0.90 * float(np.mean([r["saved_fraction"] for r in g]))


def meets(metric, m, target):
    return m <= target if metric == "completion_time" else m >= target


def select(rows, setting, family, metric, target, **match):
    """Operating point (scheme, point, cfg-kw) with the fewest channel uses meeting
    the target on the evaluation seeds."""
    pts = {}
    for r in rows:
        if r["setting"] != setting or r["family"] != family:
            continue
        if any(r.get(k) != v for k, v in match.items()):
            continue
        pts.setdefault((r["scheme"], r["point"]), []).append(r)
    best = None
    for (sch, p), rs in pts.items():
        if len(rs) < 100:
            continue
        m = np.mean([r[metric] for r in rs])
        y = np.mean([r["symbols"] for r in rs])
        if meets(metric, m, target) and (best is None or y < best[0]):
            best = (y, sch, p)
    return best


def point_kw(p):
    kw = {}
    for part in p.split("/"):
        if part.startswith("eta"):
            kw["energy_price"] = float(part[3:])
        elif part.startswith("k"):
            kw["sem_interval"] = int(part[1:])
        elif part.startswith("th"):
            kw["filter_eps"] = float(part[2:])
    return kw


def _run(a):
    cfg, scheme, seed = a
    o = run_realistic(cfg, scheme, seed)
    keep = ("completion_time", "saved_fraction", "symbols", "bits", "ul_tx", "ul_overhead_bits",
            "ul_ctrl_symbols", "dl_bcast_symbols", "dl_ctrl_symbols", "total_symbols", "completed")
    return {k: o[k] for k in keep if k in o}


def fresh(pool, cfg, scheme):
    outs = pool.map(_run, [(cfg, scheme, s) for s in FRESH])
    return {k: [o.get(k) for o in outs] for k in outs[0]}


def main_frozen():
    with open(os.path.join(RES, "rev_validation.json")) as f:
        return json.load(f)


def overhead(pool):
    rows = load_jsonl("overhead")
    jv = main_frozen()
    res = {}
    for st in HEADLINE:
        metric, target = reference(st)
        for H in (0, 16, 32, 64):
            base = SETTINGS[st].with_(overhead_bits=H)
            entry = {"target": target, "metric": metric}
            for fam in ("gosc", "periodic"):
                if H == 0:
                    sch, p = jv[st][fam]["scheme"], jv[st][fam]["point"].split("/", 1)[1]
                else:
                    b = select([r for r in rows if not (st.startswith("rescue") and r["family"] == "gosc"
                                                        and r["point"] in ("eta0.0003", "eta0.001"))],
                               st, fam, metric, target, H=H)
                    if b is None:
                        entry[fam] = None
                        continue
                    _, sch, p = b
                entry[fam] = {"scheme": sch, "point": p,
                              **fresh(pool, base.with_(**point_kw(p)), sch)}
            res[f"{st}/H{H}"] = entry
            print("overhead", st, H, {f: (entry[f]["point"], np.mean(entry[f][metric]),
                                          np.mean(entry[f]["symbols"])) if entry.get(f) else None
                                      for f in ("gosc", "periodic")}, flush=True)
    for H in (0, 32):
        for s in ("report", "raw", "nl", "sem"):
            res[f"conv/{s}/H{H}"] = fresh(pool, SETTINGS["search-W100"].with_(overhead_bits=H), s)
    return res


def noise(pool):
    rows = load_jsonl("noise")
    res = {}
    for st in HEADLINE:
        metric, target = reference(st)
        for arm, (scheme, kw) in ARMS.items():
            if arm == "exact" and not st.startswith("rescue"):
                continue
            b = select(rows, st, "gosc", metric, target, arm=arm)
            if b is None:
                res[f"{st}/{arm}"] = None
                continue
            _, sch, p = b
            res[f"{st}/{arm}"] = {"target": target, "metric": metric, "scheme": sch, "point": p,
                                  **fresh(pool, SETTINGS[st].with_(**kw, **point_kw(p)), sch)}
            print("noise", st, arm, p, np.mean(res[f"{st}/{arm}"][metric]),
                  np.mean(res[f"{st}/{arm}"]["symbols"]), flush=True)
    return res


def shared(pool):
    rows = load_jsonl("shared")
    res = {}
    for st, (cfg, bud) in SHARED.items():
        metric, target = reference(st)
        c = cfg.with_(shared_budget=bud, symbols_per_slot=bud)
        entry = {"target": target, "metric": metric}
        for fam in ("gosc", "periodic"):
            b = select(rows, st, fam, metric, target)
            if b is None:
                entry[fam] = None
                continue
            _, sch, p = b
            entry[fam] = {"scheme": sch, "point": p, **fresh(pool, c.with_(**point_kw(p)), sch)}
        res[st] = entry
        print("shared", st, {f: (entry[f]["point"], np.mean(entry[f][metric]), np.mean(entry[f]["symbols"]))
                             if entry.get(f) else None for f in ("gosc", "periodic")}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["noise", "shared", "overhead"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        for name in args.only:
            out = globals()[name](pool)
            with open(os.path.join(OUT, f"validate_{name}.json"), "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    main()
