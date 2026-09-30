"""Scheduler optimality gap and VoI additivity on logged GOSC decisions.

For every scheduling instance of GOSC missions, compares the Lagrangian + greedy
solution of the multiple-choice knapsack (Eq. 11) with the exact optimum found by
enumeration, and, whenever an intent and an evidence message compete, compares the
joint VoI of sending both with the sum of their individual VoIs.

    python -m experiments.analyze_scheduler --seeds 30
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

from gosc import Config, Simulation
from gosc.schemes import GOSC
from gosc.voi import TeamModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "rev_scheduler.json")


def exact(groups, budget, price):
    best = 0.0
    for combo in itertools.product(*[[None] + list(g.options) for g in groups]):
        sym = sum(o.symbols for o in combo if o is not None)
        if sym <= budget:
            best = max(best, sum(o.value - price * o.symbols for o in combo if o is not None))
    return best


def run(args):
    W, seed = args
    cfg = Config().with_(symbols_per_slot=W)
    sim = Simulation(cfg, "gosc", seed)
    sched = sim.scheme
    sched.log = []
    additivity = []
    orig = GOSC.build_groups

    def build(agent, snr_mean, view):
        groups = orig(sched, agent, snr_mean, view)
        kinds = {g.kind for g in groups}
        if {"INTENT", "EVID"} <= kinds:
            n = max(o.n_records for g in groups if g.kind == "EVID" for o in g.options)
            recs = agent.pending_records[:n]
            delta = np.zeros_like(view.Lc)
            for r in recs:
                view.sensor.apply(delta, r)
            team = TeamModel(view, agent, cfg.tom_temperature)
            ve = team.evidence_voi([delta], [recs[-1].pos])[0]
            vi = team.intent_voi(agent.target)
            vj = team.joint_voi(delta, recs[-1].pos, agent.target)
            additivity.append((ve, vi, vj))
        return groups

    sched.build_groups = build
    sim.run()
    gaps, binding = [], 0
    for groups, chosen, price, budget in sched.log:
        if not groups:
            continue
        opt = exact(groups, budget, price)
        got = sum(o.value - price * o.symbols for _, o in chosen)
        gaps.append(0.0 if opt <= 1e-15 else max(opt - got, 0.0) / opt)
        full = sum(max((o.symbols for o in g.options), default=0) for g in groups)
        binding += int(full > budget)
    return W, gaps, binding, additivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    with Pool(4) as pool:
        res = pool.map(run, [(W, s) for W in (100, 50) for s in range(args.seeds)])
    summary = {}
    for W in (100, 50):
        gaps = np.concatenate([np.array(g) for w, g, _, _ in res if w == W])
        add = np.array([a for w, _, _, al in res if w == W for a in al])
        ve, vi, vj = add[:, 0], add[:, 1], add[:, 2]
        tot = ve + vi
        rel = np.abs(vj - tot) / np.maximum(np.abs(vj), 1e-12)
        bounded = np.abs(vj - tot) / np.maximum(np.maximum(np.abs(vj), np.abs(tot)), 1e-12)
        rank = lambda x: np.argsort(np.argsort(x))
        spearman = float(np.corrcoef(rank(vj), rank(tot))[0, 1])
        np.save(os.path.join(os.path.dirname(OUT), f"rev_additivity_W{W}.npy"), add)
        summary[f"W{W}"] = {
            "instances": int(gaps.size),
            "optimal_fraction": float(np.mean(gaps < 1e-9)),
            "mean_gap": float(gaps.mean()),
            "p99_gap": float(np.percentile(gaps, 99)),
            "max_gap": float(gaps.max()),
            "additivity_pairs": int(len(add)),
            "additivity_median_rel_error": float(np.median(rel)),
            "additivity_p90_rel_error": float(np.percentile(rel, 90)),
            "superadditive_fraction": float(np.mean(vj > tot)),
            "bounded_rel_error_median": float(np.median(bounded)),
            "bounded_rel_error_p90": float(np.percentile(bounded, 90)),
            "spearman_joint_vs_sum": spearman,
        }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
