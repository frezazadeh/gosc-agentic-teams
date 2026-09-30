"""Direct validation of the headline efficiency claims at realizable operating points.

For every headline setting, the operating points are selected on the evaluation seeds
0-99: for GOSC, the price with the fewest channel uses whose mean outcome meets the
target; for the periodic family (rate-aware, rate-aware + filter, suppression, all
intervals/thresholds), the configuration with the fewest channel uses meeting the
target.  The selected points are frozen and re-run on fresh seeds 3000-3099; the
achieved outcome and channel uses are reported side by side, without interpolation.

    python -m experiments.validate_headline
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from gosc import Config, run_episode
from experiments.plot_revision import table, load, arr

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "rev_validation.json")
B = Config()
RB = B.with_(task="rescue", deadline_min=60, deadline_max=180)
SETTINGS = {  # name: (base config, task metric, target factor, source files for GOSC)
    "search-W100": (B, "completion_time", 1.25),
    "search-K10": (B.with_(n_agents=10), "completion_time", 1.25),
    "search-clustered": (B.with_(victim_layout="clustered"), "completion_time", 1.25),
    "rescue-W100": (RB, "saved_fraction", 0.90),
    "rescue-W50": (RB.with_(symbols_per_slot=50), "saved_fraction", 0.90),
    "rescue-snr-5": (RB.with_(snr_ref_db=-5), "saved_fraction", 0.90),
}


def candidates(name):
    """(label, scheme, config) of every GOSC and periodic operating point, with its
    evaluation-seed rows."""
    base, metric, _ = SETTINGS[name]
    task, st = name.split("-", 1)
    out = []
    if task == "search":
        src = table(load("frontier")) if st in ("W100", "W50") else table(load("where_tom"))
        for (s, p), rs in src.items():
            if s == "gosc" and p.startswith(st + "/eta"):
                e = float(p.split("eta")[1])
                out.append(("gosc", p, "gosc", base.with_(energy_price=e), rs))
        ra = table(load("rate_aware"))
        for (s, p), rs in ra.items():
            if p.startswith(st + "/k"):
                out.append(("periodic", p, "sem_ra", base.with_(sem_interval=int(p.split("/k")[1])), rs))
        pre = st
    else:
        rs_tab = table(load("rescue"))
        for (s, p), rs in rs_tab.items():
            if s == "gosc" and p.startswith(st + "/eta"):
                out.append(("gosc", p, "gosc", base.with_(energy_price=float(p.split("eta")[1])), rs))
            if s == "sem_ra" and p.startswith(st + "/k"):
                out.append(("periodic", p, "sem_ra", base.with_(sem_interval=int(p.split("/k")[1])), rs))
        pre = "rescue-" + st
    for (s, p), rs in table(load("filtered_periodic")).items():
        if p.startswith(pre + "/k"):
            out.append(("periodic", p, "sem_raf", base.with_(sem_interval=int(p.split("/k")[1])), rs))
    for (s, p), rs in table(load("suppress_periodic")).items():
        if p.startswith(pre + "/k"):
            k, th = p.split("/k")[1].split("/th")
            out.append(("periodic", p, "sem_ras", base.with_(sem_interval=int(k), filter_eps=float(th)), rs))
    return out


def reference(name):
    base, metric, factor = SETTINGS[name]
    if name.startswith("search"):
        st = name.split("-", 1)[1]
        g = table(load("genie_ref"))[("genie", "W100" if st in ("W100", "W50") else st)]
        return factor * float(arr(g, "completion_time").mean())
    st = name.split("-", 1)[1]
    g = table(load("rescue"))[("genie", st + "/ref")]
    return factor * float(arr(g, "saved_fraction").mean())


def select(name):
    base, metric, factor = SETTINGS[name]
    target = reference(name)
    ok = (lambda m: m <= target) if metric == "completion_time" else (lambda m: m >= target)
    chosen = {}
    for fam in ("gosc", "periodic"):
        pts = [(arr(rs, "symbols").mean(), lab, sch, cfg) for f, lab, sch, cfg, rs in candidates(name)
               if f == fam and ok(arr(rs, metric).mean())]
        chosen[fam] = min(pts, key=lambda x: x[0]) if pts else None
    return target, chosen


def _run(a):
    sch, cfg, seed = a
    o = run_episode(cfg, sch, seed)
    return o


def main():
    res = {}
    for name in SETTINGS:
        target, chosen = select(name)
        metric = SETTINGS[name][1]
        row = {"target": target, "metric": metric}
        for fam, c in chosen.items():
            if c is None:
                row[fam] = None
                continue
            sym_eval, lab, sch, cfg = c
            with Pool(8) as pool:
                outs = pool.map(_run, [(sch, cfg, s) for s in range(3000, 3100)])
            m = np.array([o[metric] for o in outs], float)
            y = np.array([o["symbols"] for o in outs], float)
            row[fam] = {"point": lab, "scheme": sch, "eval_ksym": float(sym_eval / 1e3),
                        "fresh_metric": float(m.mean()),
                        "fresh_metric_ci": float(1.96 * m.std(ddof=1) / np.sqrt(len(m))),
                        "fresh_ksym": float(y.mean() / 1e3),
                        "fresh_ksym_ci": float(1.96 * y.std(ddof=1) / np.sqrt(len(y)) / 1e3),
                        "meets_target": bool(m.mean() <= target if metric == "completion_time"
                                             else m.mean() >= target),
                        "metric_per_seed": m.tolist(), "ksym_per_seed": (y / 1e3).tolist()}
        if row.get("gosc") and row.get("periodic"):
            a = np.array(row["periodic"]["ksym_per_seed"])
            g = np.array(row["gosc"]["ksym_per_seed"])
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(a), (4000, len(a)))
            rat = a[idx].mean(1) / g[idx].mean(1)
            row["ratio"] = [float(a.mean() / g.mean()), *[float(v) for v in np.percentile(rat, [2.5, 97.5])]]
        res[name] = row
        print(name, {k: (v["point"], round(v["fresh_metric"], 3), round(v["fresh_ksym"], 1), v["meets_target"])
                     if isinstance(v, dict) else v for k, v in row.items() if k in ("gosc", "periodic")},
              "ratio", row.get("ratio"), flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
