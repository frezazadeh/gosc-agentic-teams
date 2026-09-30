"""Empirical companion of the scheduler theory (Section IV of the revised paper).

On logged GOSC scheduling instances (search task, default price, seeds 0-29, W=100
and W=50, as in the journal's optimality-gap analysis) it measures
  * how often the price alone makes the budget slack (mu* = 0), in which case the
    decoupled Lagrangian solution is exactly optimal (Theorem 2(i));
  * for the remaining instances, the exact optimum by enumeration, the actual gap
    and the a-posteriori duality certificate (lambda - eta) (W - S(x_lambda));
  * the multiplicative distance |ln theta_g| of every candidate message from its
    send threshold (Theorem 1): a common value error of factor e^{+-delta} can flip
    a decision only if |ln theta_g| < delta (Proposition 3);
  * the selected rates relative to the cost-efficient rate argmin s/P_s and to the
    region P_s >= 1/e.

    python -m gosc_ext.analyze_theory --seeds 30
"""
import argparse
import itertools
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from gosc import Config, Simulation

OUT = os.path.join(os.path.dirname(__file__), "..", "results_r2", "theory.json")


def pick(groups, lam):
    chosen, used = [], 0
    for g in groups:
        best, best_u = None, 0.0
        for o in g.options:
            u = o.value - lam * o.symbols
            if u > best_u:
                best, best_u = o, u
        if best is not None:
            chosen.append((g, best))
            used += best.symbols
    return chosen, used


def lagrangian(groups, budget, price):
    """journal lagrangian_select, returning also the multiplier and x_lambda"""
    chosen, used = pick(groups, price)
    if used <= budget:
        return chosen, price, chosen, used, False
    lo, hi = 0.0, max((o.value / o.symbols for g in groups for o in g.options), default=1.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pick(groups, price + mid)[1] <= budget:
            hi = mid
        else:
            lo = mid
    xl, used = pick(groups, price + hi)
    chosen = list(xl)
    taken = {id(g) for g, _ in chosen}
    for g in sorted((g for g in groups if id(g) not in taken),
                    key=lambda g: -max(o.value / o.symbols for o in g.options)):
        fits = [o for o in g.options
                if o.symbols <= budget - used and o.value - price * o.symbols > 0]
        if fits:
            o = max(fits, key=lambda o: o.value - price * o.symbols)
            chosen.append((g, o))
            used += o.symbols
    return chosen, price + hi, xl, sum(o.symbols for _, o in xl), True


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
    rates_ps = {}
    orig = sched.build_groups

    def build(agent, snr_mean, view):
        gs = orig(agent, snr_mean, view)
        for g in gs:
            for o in g.options:
                rates_ps[id(o)] = float(sched.channel.success_prob(o.rate, snr_mean, n=o.symbols))
        return gs
    sched.build_groups = build
    sim.run()
    inst = []
    for groups, chosen, price, budget in sched.log:
        if not groups:
            continue
        mine, lam, xl, sl, binding = lagrangian(groups, budget, price)
        assert [(id(g), id(o)) for g, o in mine] == [(id(g), id(o)) for g, o in chosen]
        got = sum(o.value - price * o.symbols for _, o in chosen)
        rec = {"binding": binding}
        if binding:
            opt = exact(groups, budget, price)
            jl = sum(o.value - price * o.symbols for _, o in xl)
            cert = jl + (lam - price) * (budget - sl)
            rec.update(opt=opt, got=got, cert=cert)
        # distance of each non-confirmation message from its send threshold at lam
        th = []
        for g in groups:
            if g.kind == "CONF":
                continue
            r = [lam * o.symbols / o.value for o in g.options if o.value > 0]
            th.append((g.kind, math.log(min(r)) if r else None))
        rec["theta"] = th
        # selected rates
        rs = []
        for g, o in chosen:
            eff = min(g.options, key=lambda q: q.symbols / max(rates_ps[id(q)], 1e-300))
            rs.append((g.kind, rates_ps[id(o)] >= math.exp(-1), o.rate <= eff.rate,
                       o.rate == eff.rate, o.rate == min(q.rate for q in g.options)))
        rec["rates"] = rs
        inst.append(rec)
    return W, inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        res = pool.map(run, [(W, s) for W in (100, 50) for s in range(args.seeds)])
    summary = {}
    for W in (100, 50):
        inst = [r for w, rs in res if w == W for r in rs]
        b = [r for r in inst if r["binding"]]
        gaps = np.array([max(r["opt"] - r["got"], 0.0) / r["opt"] if r["opt"] > 1e-15 else 0.0 for r in b])
        cert = np.array([max(r["cert"] - r["got"], 0.0) / r["opt"] if r["opt"] > 1e-15 else 0.0 for r in b])
        th = [(k, v) for r in inst for k, v in r["theta"]]
        finite = np.array([abs(v) for _, v in th if v is not None])
        rates = [x for r in inst for x in r["rates"]]
        ev = np.array([x[1:] for x in rates if x[0] == "EVID"], dtype=bool)
        fx = {k: np.array([x[1:] for x in rates if x[0] == k], dtype=bool) for k in ("CONF", "INTENT")}
        summary[f"W{W}"] = {
            "instances": len(inst),
            "slack_fraction": 1 - len(b) / len(inst),
            "binding": len(b),
            "binding_exact_fraction": float(np.mean(gaps < 1e-9)) if len(b) else None,
            "overall_exact_fraction": (len(inst) - len(b) + float(np.sum(gaps < 1e-9))) / len(inst),
            "binding_mean_gap": float(gaps.mean()) if len(b) else None,
            "overall_mean_gap": float(gaps.sum() / len(inst)),
            "cert_zero_fraction_binding": float(np.mean(cert < 1e-9)) if len(b) else None,
            "cert_mean_binding": float(cert.mean()) if len(b) else None,
            "messages": len(th),
            "zero_value_fraction": float(np.mean([v is None for _, v in th])),
            "within": {d: float(np.mean(finite < d) * len(finite) / len(th)) for d in (0.5, 1.0, 2.0)},
            "median_abs_log_theta": float(np.median(finite)),
            "evid_ps_ge_1e": float(ev[:, 0].mean()),
            **{f"{k.lower()}_{name}": float(a[:, i].mean()) for k, a in fx.items() if len(a)
               for i, name in enumerate(("ps_ge_1e", "rate_le_eff", "rate_eq_eff", "rate_min"))},
            "sent_counts": {k: int(len(a)) for k, a in fx.items()} | {"EVID": int(len(ev))},
        }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
