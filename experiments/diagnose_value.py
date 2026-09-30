"""How well do value estimates predict the real effect of a message?

At sampled decision points of GOSC missions (after sensing and planning, before the
uplink), each candidate message m of one agent (evidence E, intention I, or both EI)
is delivered instantly in a copy of the simulation, and compared with delivering
nothing (the sender stays silent in that slot in every branch).

* Decision-level effect: change in the ground-truth utility of the teammates' next
  targets, U = sum over true, undeclared survivors of max_j 1[s in F(g_j)] / (1 + beta d_j),
  i.e., distance-discounted coverage of real survivors, each counted once.
* Mission-level effect: difference in completion time between the branches, from
  R continuation rollouts with common random numbers across branches.

Estimates recorded: theory-of-mind VoI (individual, joint, additive), belief
divergence (total variation), payload bits and age of information.

    python -m experiments.diagnose_value --missions 20 --every 8 --reps 12
"""
import argparse
import copy
import json
import os
from multiprocessing import Pool

import numpy as np

from gosc import Config, Simulation
from gosc.belief import sigmoid
from gosc.messages import Packet, evidence_sizes, intent_bits
from gosc.voi import TeamModel

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "rev_value_fidelity.json")


def has_intent(a):
    return (a.target is not None and a.target != a.shared_intent
            and a.target != getattr(a, "acked_intent", a.shared_intent))


def estimates(sim, a):
    c, t = sim.cfg, sim.t
    recs = list(a.pending_records)
    est = {}
    team = TeamModel(sim, a, c.tom_temperature)
    if recs:
        delta = np.zeros_like(sim.Lc)
        for r in recs:
            sim.sensor.apply(delta, r)
        live = ~sim.declared.reshape(sim.Lc.shape)
        est["E"] = {"tom": team.evidence_voi([delta], [recs[-1].pos])[0],
                    "tv": float(np.abs(sigmoid(sim.Lc + delta) - sigmoid(sim.Lc))[live].sum()),
                    "bits": float(evidence_sizes(recs, c)[-1]),
                    "aoi": float(sum(t - r.t + 1 for r in recs))}
    if has_intent(a):
        est["I"] = {"tom": team.intent_voi(a.target), "tv": 0.0, "bits": float(intent_bits(c)),
                    "aoi": float(t - getattr(a, "_intent_t", t) + 1)}
    if "E" in est and "I" in est:
        est["EI"] = {"tom": team.joint_voi(delta, recs[-1].pos, a.target),
                     "tom_add": est["E"]["tom"] + est["I"]["tom"],
                     "tv": est["E"]["tv"], "bits": est["E"]["bits"] + est["I"]["bits"],
                     "aoi": est["E"]["aoi"] + est["I"]["aoi"]}
    return est


def force(sim, k, kind):
    a, t = sim.agents[k], sim.t
    if "E" in kind and a.pending_records:
        recs = list(a.pending_records)
        a.pending_records.clear()
        for r in recs:
            sim.sensor.apply(a.queued_llr, r, sign=-1.0)
        sim.deliver(a, Packet("EVID", k, evidence_sizes(recs, sim.cfg)[-1], t, records=recs), t)
    if "I" in kind:
        a.acked_intent = a.target
        sim.deliver(a, Packet("INTENT", k, intent_bits(sim.cfg), t, intent=a.target), t)
    sim.skip_tx = {k}


def gt_utility(sim, k):
    """Ground-truth coverage utility of all agents' next targets after this slot's
    information: teammates re-plan on their updated beliefs (on copies)."""
    c, n = sim.cfg, sim.cfg.grid
    rng = copy.deepcopy(sim.plan_rng)
    targets, pos = {}, {}
    for a in sim.agents:
        if a.idx == k:
            targets[a.idx], pos[a.idx] = a.target, a.pos
            continue
        b = copy.deepcopy(a)
        p = sigmoid(sim.Lc + b.pending_llr).reshape(-1)
        p[sim.declared] = 0.0
        if b.confirmed_pending:
            p[list(b.confirmed_pending)] = 0.0
        b.plan(p.reshape(n, n), sim.teammates(b.idx), rng)
        targets[a.idx], pos[a.idx] = b.target, b.pos
    r, u = c.sense_radius, 0.0
    for s in sim.world.victims:
        if sim.declared[s]:
            continue
        sx, sy = divmod(s, n)
        best = 0.0
        for j, g in targets.items():
            if g is None:
                continue
            gx, gy = divmod(g, n)
            if max(abs(gx - sx), abs(gy - sy)) <= r:
                d = abs(pos[j][0] - gx) + abs(pos[j][1] - gy)
                best = max(best, 1.0 / (1.0 + c.distance_discount * d))
        u += best
    return u


def rollout(sim, reseed):
    s = copy.deepcopy(sim)
    s.sense_rng = np.random.default_rng(reseed + [1])
    s.chan_rng = np.random.default_rng(reseed + [2])
    s.plan_rng = np.random.default_rng(reseed + [3])
    s.channel.rng = np.random.default_rng(reseed + [4])
    t = s.t
    if not s.phase_communicate(t):
        for t in range(t + 1, s.cfg.max_steps):
            s.phase_decide(t)
            if s.phase_communicate(t):
                break
    return s.done_at


def point(args):
    seed, t0, reps = args
    cfg = Config()
    sim = Simulation(cfg, "gosc", seed)
    sim._start()
    for t in range(t0):
        sim.phase_decide(t)
        if sim.phase_communicate(t):
            return []
    sim.phase_decide(t0)
    out = []
    rng = np.random.default_rng([seed, t0, 99])
    cand = [a.idx for a in sim.agents if a.pending_records or has_intent(a)]
    if not cand:
        return []
    for k in rng.permutation(cand)[:2]:
        k = int(k)
        est = estimates(sim, sim.agents[k])
        branches = {"none": copy.deepcopy(sim)}
        branches["none"].skip_tx = {k}
        for kind in est:
            b = copy.deepcopy(sim)
            force(b, k, kind)
            branches[kind] = b
        gt = {kind: gt_utility(b, k) for kind, b in branches.items()}
        T = {kind: [rollout(b, [seed, t0, k, rr]) for rr in range(reps)] for kind, b in branches.items()}
        for kind in est:
            out.append({"seed": seed, "t": t0, "agent": k, "kind": kind, "est": est[kind],
                        "gt_gain": gt[kind] - gt["none"],
                        "dT": float(np.mean(T["none"]) - np.mean(T[kind])),
                        "dT_se": float(np.std(np.array(T["none"]) - np.array(T[kind]), ddof=1) / np.sqrt(reps)),
                        "n_teammate_changes": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", type=int, default=20)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--first_seed", type=int, default=2000)
    args = ap.parse_args()
    jobs = [(s, t0, args.reps) for s in range(args.first_seed, args.first_seed + args.missions)
            for t0 in range(4, 120, args.every)]
    with Pool(8) as pool:
        rows = [r for res in pool.imap_unordered(point, jobs) for r in res]
    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"{len(rows)} (decision, message) samples")


if __name__ == "__main__":
    main()
