"""Belief-divergence value (gosc_tv) in the K=10 and clustered search settings, which
the "where does ToM matter" experiment of the main evaluation did not include.  Same
price grid as the frontier experiment for this value, evaluation seeds 0-99; results go to
results/belief_divergence.jsonl.  The matched-budget comparison then reuses the
main-evaluation functions (experiments.plot_revision.matched).

    python -m experiments.run_belief_divergence            # simulate
    python -m experiments.run_belief_divergence --table    # fill the n/e cells of table_values.tex
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from gosc import Config, run_episode

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "belief_divergence.jsonl")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
ETAS_TV = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)   # frontier grid for gosc_tv


def jobs():
    b = Config()
    out = []
    for e in ETAS_TV:
        out.append(("gosc_tv", f"K10/eta{e}", b.with_(n_agents=10, energy_price=e)))
        out.append(("gosc_tv", f"clustered/eta{e}", b.with_(victim_layout="clustered", energy_price=e)))
    return out


def _job(a):
    scheme, param, cfg, seed = a
    o = run_episode(cfg, scheme, seed)
    o.pop("curve")
    return {"scheme": scheme, "param": param, "seed": seed, **o}


def simulate(seeds, workers):
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["param"], r["seed"]))
    todo = [(s, p, c, seed) for seed in range(seeds) for s, p, c in jobs() if (p, seed) not in done]
    with Pool(workers) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_job, todo, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)


def fill_table():
    from experiments.plot_revision import matched, table
    rows = json.load(open(os.path.join(RES, "rev_where_tom.json")))["rows"]
    rows += [json.loads(line) for line in open(OUT)]
    tab = table(rows)
    curves = {}
    for (s, p), rs in tab.items():
        W, knob = p.split("/")
        curves.setdefault((W, s), []).append((knob, rs))
    cells = {}
    for W in ("K10", "clustered"):
        st = matched(curves, W)
        for B, row in st.items():
            v = row.get("gosc_tv")
            cells[(W, B)] = "--" if not v else f"{v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]"
    path = os.path.join(FIG, "table_values.tex")
    lines = open(path).read().splitlines()
    names = {"$K=10$": "K10", "Clustered": "clustered"}
    out = []
    for ln in lines:
        c = [x.strip() for x in ln.split("&")]
        if len(c) > 3 and c[0] in names and c[3] == "n/e":
            B = int(c[1].rstrip("k")) * 1000
            c[3] = cells[(names[c[0]], B)]
            ln = " & ".join(c)
        out.append(ln)
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    return cells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        fill_table()
    else:
        simulate(args.seeds, args.workers)
