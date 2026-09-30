"""Decisive test of set-valued (joint) scheduling vs independent valuation.

    python -m experiments.run_joint --seeds 100
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

from gosc import Config, run_episode

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "rev_joint.json")


def _job(a):
    s, p, cfg, seed = a
    o = run_episode(cfg, s, seed)
    o.pop("curve")
    return {"scheme": s, "param": p, "seed": seed, **o}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    b = Config()
    rows = [(s, f"W{W}/eta{e}", b.with_(symbols_per_slot=W, energy_price=e))
            for W in (100, 50) for e in (0.0, 1e-5, 3e-5)
            for s in ("gosc", "gosc_joint", "gosc_joint_nb")]
    jobs = [(s, p, cfg, seed) for s, p, cfg in rows for seed in range(args.seeds)]
    t0 = time.time()
    with Pool(8) as pool:
        out = list(pool.imap_unordered(_job, jobs, chunksize=2))
    with open(OUT, "w") as f:
        json.dump({"rows": out}, f)
    print(f"joint: {len(out)} episodes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
