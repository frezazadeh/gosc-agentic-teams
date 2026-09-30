"""Simulation campaigns with realistic radio resources and imperfect values (evaluation
seeds 0-99).

    python -m experiments.run_realistic --only noise shared overhead [--seeds 100 --workers 6]

  noise     imperfect semantic values: log-normal errors, order-of-magnitude values,
            content-independent (permuted) values, with and without the relevance
            filter; GOSC's price is swept in every setting
  shared    a network-wide uplink budget B shared by all agents (K=6): price-
            coordinated GOSC vs the periodic family with max-min fair grants
  overhead  a fixed per-packet overhead of 16/32/64 bits on every uplink packet, for
            GOSC and the full periodic family in the six headline settings

Every mission is appended to results/<name>.jsonl (the run can be resumed).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

from gosc import Config
from gosc import run_realistic

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

B = Config()
RB = B.with_(task="rescue", deadline_min=60, deadline_max=180)
SETTINGS = {
    "search-W100": B,
    "search-W50": B.with_(symbols_per_slot=50),
    "search-K10": B.with_(n_agents=10),
    "search-clustered": B.with_(victim_layout="clustered"),
    "rescue-W100": RB,
    "rescue-W50": RB.with_(symbols_per_slot=50),
    "rescue-snr-5": RB.with_(snr_ref_db=-5),
}
HEADLINE = ("search-W100", "search-K10", "search-clustered", "rescue-W100", "rescue-W50", "rescue-snr-5")
ETA_SEARCH = (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
ETA_RESCUE = (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)  # extended beyond the main-evaluation grid
THETAS = (0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0)
PERIODIC = ([("sem_ra", k, None) for k in (1, 2, 4, 8, 16)]
            + [("sem_raf", k, None) for k in (1, 2, 4, 8, 16)]
            + [("sem_ras", k, th) for k in (1, 2, 4, 8) for th in THETAS])
ARMS = {  # value-estimation arms of the noise experiment
    "exact": ("gosc", {}),   # rescue only (search: identical main-evaluation runs)
    "ln0.5": ("gosc", dict(value_mode="lognormal", value_sigma=0.5)),
    "ln1": ("gosc", dict(value_mode="lognormal", value_sigma=1.0)),
    "ln2": ("gosc", dict(value_mode="lognormal", value_sigma=2.0)),
    "quant": ("gosc", dict(value_mode="quantized")),
    "perm": ("gosc", dict(value_mode="permuted")),
    "perm_nofilter": ("gosc_nofilter", dict(value_mode="permuted")),
}
SHARED = {"search-B600": (B, 600), "search-B300": (B, 300), "search-B150": (B, 150),
          "rescue-B300": (RB, 300)}
OVERHEADS = (16, 32, 64)


def etas(setting):
    return ETA_RESCUE if setting.startswith("rescue") else ETA_SEARCH


def noise_etas(setting):
    """Noise arms: the main-evaluation grid without 3e-6 (between 0 and 1e-5, never the most
    frugal point meeting a target in the main evaluation)."""
    if setting.startswith("rescue"):
        # extended upwards: inaccurate values send more at a given price, and the
        # main-evaluation rescue grid already met the targets at its highest price
        return (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    return tuple(e for e in etas(setting) if e != 3e-6)


def periodic_rows(setting, cfg, extra):
    rows = []
    for s, k, th in PERIODIC:
        kw = dict(sem_interval=k)
        lab = f"k{k}"
        if th is not None:
            kw["filter_eps"] = th
            lab += f"/th{th}"
        rows.append(dict(setting=setting, family="periodic", scheme=s, point=lab, **extra,
                         cfg=cfg.with_(**kw)))
    return rows


def experiments():
    ex = {}
    rows = []
    for st in HEADLINE:
        cfg = SETTINGS[st]
        for arm, (scheme, kw) in ARMS.items():
            if arm == "exact" and not st.startswith("rescue"):
                continue
            for e in noise_etas(st):
                rows.append(dict(setting=st, family="gosc", arm=arm, scheme=scheme, point=f"eta{e}",
                                 cfg=cfg.with_(energy_price=e, **kw)))
    ex["noise"] = rows
    rows = []
    for st, (cfg, bud) in SHARED.items():
        c = cfg.with_(shared_budget=bud, symbols_per_slot=bud)
        for e in etas(st):
            rows.append(dict(setting=st, family="gosc", scheme="gosc", point=f"eta{e}",
                             cfg=c.with_(energy_price=e)))
        rows += periodic_rows(st, c, {})
        rows.append(dict(setting=st, family="conventional", scheme="sem", point="k1", cfg=c))
    ex["shared"] = rows
    rows = []
    for H in OVERHEADS:
        for st in HEADLINE:
            c = SETTINGS[st].with_(overhead_bits=H)
            for e in etas(st):
                rows.append(dict(setting=st, family="gosc", scheme="gosc", point=f"eta{e}", H=H,
                                 cfg=c.with_(energy_price=e)))
            rows += periodic_rows(st, c, dict(H=H))
    for H in (0,) + OVERHEADS:
        for s in ("report", "raw", "nl", "sem"):
            rows.append(dict(setting="search-W100", family="conventional", scheme=s, point="ref", H=H,
                             cfg=B.with_(overhead_bits=H)))
    ex["overhead"] = rows
    return ex


def _job(args):
    row, seed = args
    out = run_realistic(row["cfg"], row["scheme"], seed)
    meta = {k: v for k, v in row.items() if k != "cfg"}
    return {**meta, "seed": seed, **out}


def key(r):
    return (r["setting"], r.get("arm"), r.get("H"), r["scheme"], r["point"], r["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["noise", "shared", "overhead"])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ex = experiments()
    for name in args.only:
        path = os.path.join(OUT, f"{name}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        done.add(key(json.loads(line)))
                    except json.JSONDecodeError:
                        pass
        jobs = [(row, s) for s in range(args.seeds) for row in ex[name]
                if key({**row, "seed": s}) not in done]
        t0 = time.time()
        with Pool(args.workers) as pool, open(path, "a") as f:
            for i, out in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
                f.write(json.dumps(out) + "\n")
                if i % 500 == 0:
                    f.flush()
                    print(f"{name}: {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        print(f"{name}: {len(jobs)} missions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
