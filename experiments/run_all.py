"""Run every experiment reported in the paper and store raw per-episode results.

Evaluation uses seeds 0..N-1.  Hyper-parameters of the proposed scheme were
calibrated on disjoint seeds (1000+), see README.

    python -m experiments.run_all            # full campaign
    python -m experiments.run_all --seeds 8  # smoke test
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

from gosc import Config, run_episode

ALL = ["report", "raw", "nl", "sem", "gosc_nouep", "gosc_novoi", "gosc_tv", "gosc", "genie"]
MAIN = ["report", "raw", "nl", "sem", "gosc", "genie"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def experiments(n_seeds):
    base = Config()
    exps = {
        "default": [(s, "default", base, n_seeds) for s in ALL],
        "snr": [(s, v, base.with_(snr_ref_db=v), n_seeds)
                for v in (-10, -5, 0, 5, 10, 15, 20) for s in MAIN],
        "budget": [(s, v, base.with_(symbols_per_slot=v), n_seeds)
                   for v in (25, 50, 100, 200, 400) for s in MAIN],
        "agents": [(s, v, base.with_(n_agents=v), n_seeds)
                   for v in (2, 4, 6, 8, 10) for s in MAIN],
        "pareto": [("gosc", v, base.with_(energy_price=v), n_seeds)
                   for v in (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)],
        "pareto_tv": [("gosc_tv", v, base.with_(energy_price=v), n_seeds)
                      for v in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    }
    return exps


def _job(args):
    exp, scheme, param, cfg, seed = args
    out = run_episode(cfg, scheme, seed)
    if exp != "default":
        out.pop("curve")
    return exp, {"scheme": scheme, "param": param, "seed": seed, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--only", nargs="*", help="subset of experiments")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    exps = experiments(args.seeds)
    names = args.only or list(exps)
    for name in names:
        jobs = [(name, s, p, cfg, seed) for s, p, cfg, n in exps[name] for seed in range(n)]
        t0 = time.time()
        with Pool(args.workers) as pool:
            rows = [r for _, r in pool.imap_unordered(_job, jobs, chunksize=4)]
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump({"config": Config().__dict__, "rows": rows}, f)
        print(f"{name}: {len(rows)} episodes in {time.time() - t0:.0f}s", flush=True)
    # one recorded episode for the trajectory illustration
    traj = {s: run_episode(Config(), s, 0, record_traj=True) for s in ("sem", "gosc")}
    with open(os.path.join(OUT, "trajectories.json"), "w") as f:
        json.dump({s: {k: v for k, v in o.items() if k in ("trajectories", "victims", "completion_time")}
                   for s, o in traj.items()}, f)


if __name__ == "__main__":
    main()
