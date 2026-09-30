"""Episode simulator for multi-agent search and rescue over a fading uplink.

Slot structure: sense -> self-confirm -> plan -> uplink -> edge fusion and
broadcast -> move.  The edge coordinator fuses every decoded packet into the
shared belief `Lc` and broadcasts it.  With an ideal downlink (default) every
agent's local belief is `Lc + pending_llr` (its own undelivered evidence).  With
broadcast loss/delay (Config.dl_loss, Config.dl_delay) each agent instead acts on
its own, possibly stale, snapshot of the common knowledge (a `View`).
"""
import argparse
import json

import numpy as np

from .agent import Agent
from .belief import Obs, SensorModel, logit, sigmoid
from .channel import Channel
from .config import Config
from .environment import World
from .messages import cell_bits
from .schemes import SCHEMES


def teammate_points(cfg, team_pos, team_intent, k):
    """Points agent k should stay away from: intents of higher-priority agents
    (j < k, sequential deconfliction), otherwise last known positions."""
    n = cfg.grid
    pts = []
    for j in range(cfg.n_agents):
        if j == k:
            continue
        if j < k and j in team_intent:
            pts.append(divmod(team_intent[j], n))
        elif j in team_pos:
            pts.append(team_pos[j])
    return pts


class View:
    """One agent's (possibly stale) copy of the common knowledge."""

    def __init__(self, sim, snap):
        self.cfg, self.sensor, self.t, self.agents = sim.cfg, sim.sensor, sim.t, sim.agents
        self.Lc, self.declared, self.team_pos, self.team_intent = snap

    def teammates(self, k):
        return teammate_points(self.cfg, self.team_pos, self.team_intent, k)


class Simulation:
    def __init__(self, cfg: Config, scheme: str, seed: int, record_traj: bool = False):
        self.cfg = cfg
        n = cfg.grid
        world_rng = np.random.default_rng([seed, 0])
        self.sense_rng = np.random.default_rng([seed, 1])
        self.chan_rng = np.random.default_rng([seed, 2])
        self.plan_rng = np.random.default_rng([seed, 3])
        self.dl_rng = np.random.default_rng([seed, 5])
        self.world = World(cfg, world_rng)
        self.sensor = SensorModel(cfg.p_detect, cfg.p_false_alarm, cfg.sense_radius, n)
        self.channel = Channel(cfg, self.world.bs, rng=np.random.default_rng([seed, 4]))
        self.scheme = SCHEMES[scheme](cfg, self.channel)
        self.prior = logit(cfg.n_victims / (n * n))
        self.Lc = np.full((n, n), self.prior)          # edge / shared belief
        self.declared = np.zeros(n * n, dtype=bool)
        bx, by = self.world.bs
        starts = [(int(np.clip(bx + dx, 0, n - 1)), int(np.clip(by + dy, 0, n - 1)))
                  for dx, dy in world_rng.integers(-1, 2, size=(cfg.n_agents, 2))]
        self.t = 0
        if cfg.task == "rescue":
            dls = world_rng.integers(cfg.deadline_min, cfg.deadline_max + 1, size=len(self.world.victims))
            self.world.deadline = {v: int(d) for v, d in zip(sorted(self.world.victims), dls)}
        self.rescue_progress = {}   # task cell -> slots of service so far (common knowledge)
        self.rescue_done = {}       # task cell -> slot when finished
        self.lost = set()           # survivors whose deadline passed before rescue
        self.agents = []
        for k, p in enumerate(starts):
            if k in cfg.llm_agents:
                from .llm_agent import LLMAgent, get_client
                self.agents.append(LLMAgent(k, p, cfg, self, get_client(cfg.llm_model)))
            else:
                self.agents.append(Agent(k, p, cfg))
        # agent -> last position known to the team (launch points are common knowledge)
        self.team_pos = {k: p for k, p in enumerate(starts)}
        self.team_intent = {}   # agent -> last intent known to the team
        self.record_traj = record_traj
        self.located_time = {}
        self.first_confirm = {}
        self.false_declarations = 0
        self.bits = 0
        self.symbols = 0
        self.delivered_bits = 0
        self.thr = logit(cfg.confirm_threshold)
        # downlink broadcast of the common knowledge
        self.ideal_dl = cfg.dl_loss == 0 and cfg.dl_delay == 0
        self.bcast_bits = {}     # slot -> bits fused (and rebroadcast) in that slot
        self.dl_bits = 0
        self.private_tv = []
        if not self.ideal_dl:
            self.snaps = {-1: self._snapshot()}
            self.view_t = [-1] * cfg.n_agents
            self.views = [View(self, self.snaps[-1]) for _ in self.agents]
            for a in self.agents:
                a.inflight = []

    # ------------------------------------------------------------------
    def _snapshot(self):
        return (self.Lc.copy(), self.declared.copy(), dict(self.team_pos), dict(self.team_intent))

    def view(self, agent):
        """The common knowledge as agent `agent` currently sees it."""
        return self if self.ideal_dl else self.views[agent.idx]

    def local_llr(self, agent):
        return self.view(agent).Lc + agent.pending_llr

    def declare(self, cell: int, t: int) -> None:
        if self.declared[cell]:
            return
        self.declared[cell] = True
        self.bcast_bits[t] = self.bcast_bits.get(t, 0) + cell_bits(self.cfg)
        if cell in self.world.victims:
            self.located_time[cell] = t
        else:
            self.false_declarations += 1
        if self.ideal_dl:
            for a in self.agents:
                a.confirmed_pending.discard(cell)

    def deliver(self, agent, pkt, t: int) -> None:
        self.delivered_bits += pkt.bits
        self.bcast_bits[t] = self.bcast_bits.get(t, 0) + pkt.bits
        for rec in pkt.records:
            self.sensor.apply(self.Lc, rec)
            if self.ideal_dl:
                self.sensor.apply(agent.pending_llr, rec, sign=-1.0)
        if pkt.records:
            self.team_pos[agent.idx] = pkt.records[-1].pos
        if pkt.intent is not None:
            self.team_intent[agent.idx] = pkt.intent
            if self.ideal_dl:
                agent.shared_intent = pkt.intent
        if not self.ideal_dl and (pkt.records or pkt.intent is not None):
            agent.inflight.append((t, pkt.records, pkt.intent))
        if pkt.confirm is not None:
            self.declare(pkt.confirm, t)

    def teammates(self, k: int):
        return teammate_points(self.cfg, self.team_pos, self.team_intent, k)

    def open_tasks(self):
        """Located (declared) survivors still to be rescued: common knowledge."""
        dl = self.world.deadline
        return {int(c): dl.get(int(c), self.t + self.cfg.rescue_service + 1000)
                for c in np.flatnonzero(self.declared)
                if int(c) not in self.rescue_done and int(c) not in self.lost}

    def known_tasks(self, agent):
        """Tasks agent knows: open common tasks plus its own confirmed survivors."""
        tasks = self.open_tasks()
        dl = self.world.deadline
        for c in agent.confirmed_pending:
            if c not in self.rescue_done and c not in self.lost:
                tasks[c] = dl.get(c, self.t + self.cfg.rescue_service + 1000)
        return tasks

    def _downlink(self, t: int) -> None:
        """Broadcast of slot t's fused updates; lossy/delayed if configured."""
        cfg = self.cfg
        self.dl_bits += self.bcast_bits.get(t, 0)
        if self.ideal_dl:
            return
        self.snaps[t] = self._snapshot()
        target = t - cfg.dl_delay
        for a in self.agents:
            k = a.idx
            if target <= self.view_t[k] or self.dl_rng.random() < cfg.dl_loss:
                continue
            old = self.view_t[k]
            # catch-up: updates of the slots this agent missed are re-sent
            self.dl_bits += sum(self.bcast_bits.get(s, 0) for s in range(old + 1, target) if s >= 0
                                ) if target - old > 1 else 0
            self.view_t[k] = target
            keep = []
            for (s, recs, intent) in a.inflight:
                if s <= target:
                    for rec in recs:
                        self.sensor.apply(a.pending_llr, rec, sign=-1.0)
                    if intent is not None:
                        a.shared_intent = intent
                else:
                    keep.append((s, recs, intent))
            a.inflight = keep
            a.confirmed_pending -= set(np.flatnonzero(self.snaps[target][1]).tolist())
        oldest = min(self.view_t)
        for s in [s for s in self.snaps if s < oldest]:
            del self.snaps[s]

    # ------------------------------------------------------------------
    def run(self):
        self._start()
        for t in range(self.cfg.max_steps):
            self.phase_decide(t)
            if self.phase_communicate(t):
                break
        return self.result()

    def _start(self):
        V = self.cfg.n_victims
        self.curve = np.full(self.cfg.max_steps, V, dtype=np.int16)
        self.done_at = self.cfg.max_steps
        self.skip_tx = set()   # agents that do not transmit in the current slot (analysis)

    def phase_decide(self, t):
        """Sensing, self-confirmation and intention selection of slot t."""
        cfg, n = self.cfg, self.cfg.grid
        self.t = t
        if not self.ideal_dl:
            self.views = [View(self, self.snaps[self.view_t[a.idx]]) for a in self.agents]
        # 1) sensing and local Bayesian update
        recs = []
        for a in self.agents:
            det = self.world.sense(a.pos, self.sensor, self.sense_rng)
            rec = Obs(t, a.idx, a.pos, det)
            self.sensor.apply(a.pending_llr, rec)
            recs.append(rec)
        if cfg.log_private:
            live = ~self.declared.reshape(n, n)
            p0 = sigmoid(self.Lc)
            self.private_tv.append(np.mean([
                float(np.abs(sigmoid(self.Lc + a.pending_llr) - p0)[live].sum())
                for a in self.agents]))
        # 2) self-confirmation of survivors
        for a in self.agents:
            vw = self.view(a)
            L = self.local_llr(a).reshape(-1)
            for cell in np.flatnonzero((L >= self.thr) & ~vw.declared):
                cell = int(cell)
                if cell not in a.confirmed_pending:
                    a.confirmed_pending.add(cell)
                    self.first_confirm.setdefault(cell, t)
                    self.scheme.on_confirm(a, cell, t)
        # 3) intention selection
        for a, rec in zip(self.agents, recs):
            vw = self.view(a)
            p = sigmoid(self.local_llr(a)).reshape(-1)
            p[vw.declared] = 0.0
            if a.confirmed_pending:
                p[list(a.confirmed_pending)] = 0.0
            a.plan(p.reshape(n, n), vw.teammates(a.idx), self.plan_rng)
            if cfg.task == "rescue":
                from .rescue import rescue_choice
                tasks = self.known_tasks(a)
                s = rescue_choice(cfg, a.idx, a.pos, t, tasks, vw.team_intent, a.target,
                                  self.rescue_progress, vw.team_pos)
                if s is not None:
                    a.target = s
            self.scheme.on_step(a, rec, t, vw)

    def phase_communicate(self, t):
        """Uplink, edge fusion, broadcast and motion of slot t; True when done."""
        cfg = self.cfg
        # 4) uplink over block-fading channel (orthogonal resources)
        results = []
        for a in self.agents:
            snr_mean = self.channel.mean_snr(a.pos)
            snr_inst = snr_mean * self.chan_rng.exponential()
            if a.idx in self.skip_tx:
                continue
            known = snr_inst if cfg.csit else snr_mean
            res = self.scheme.transmit(a, t, known, snr_inst, self.view(a))
            self.bits += res.bits
            self.symbols += res.symbols
            results.append((a, res))
        self.skip_tx = set()
        # 5) edge fusion + broadcast
        for a, res in results:
            for pkt in res.delivered:
                self.deliver(a, pkt, t)
        for cell in np.flatnonzero((self.Lc.reshape(-1) >= self.thr) & ~self.declared):
            self.declare(int(cell), t)
        self._downlink(t)
        found = len(self.located_time)
        self.curve[t] = found
        if found == cfg.n_victims and cfg.task != "rescue":
            self.done_at = t + 1
            return True
        # 6) motion
        for a in self.agents:
            a.step()
        if cfg.task == "rescue" and self._rescue_update(t):
            self.done_at = t + 1
            return True
        return False

    def _rescue_update(self, t):
        """Rescue service after motion; returns True when every survivor is rescued
        or lost."""
        cfg, n = self.cfg, self.cfg.grid
        tasks = set(self.open_tasks())
        for a in self.agents:
            tasks |= {c for c in a.confirmed_pending}
        for c in tasks:
            if c in self.rescue_done or c in self.lost:
                continue
            if any(a.pos == divmod(c, n) and a.target == c for a in self.agents):
                self.rescue_progress[c] = self.rescue_progress.get(c, 0) + 1
                if self.rescue_progress[c] >= cfg.rescue_service:
                    self.rescue_done[c] = t
                    if c not in self.declared:
                        self.declare(c, t)
        for v, d in self.world.deadline.items():
            if v not in self.rescue_done and v not in self.lost and t >= d:
                self.lost.add(v)
        return all(v in self.rescue_done or v in self.lost for v in self.world.victims)

    def result(self):
        cfg, V = self.cfg, self.cfg.n_victims
        latency = [self.located_time[c] - self.first_confirm[c]
                   for c in self.located_time if c in self.first_confirm]
        out = {
            "completion_time": self.done_at,
            "completed": len(self.located_time) == V,
            "found": len(self.located_time),
            "curve": self.curve.tolist(),
            "bits": int(self.bits),
            "symbols": int(self.symbols),
            "delivered_bits": int(self.delivered_bits),
            "dl_symbols": int(np.ceil(self.dl_bits / cfg.dl_rate)),
            "false_declarations": self.false_declarations,
            "confirm_latency": float(np.mean(latency)) if latency else None,
        }
        if cfg.log_private:
            out["private_tv"] = float(np.mean(self.private_tv))
        if cfg.task == "rescue":
            saved = [v for v in self.world.victims
                     if v in self.rescue_done and self.rescue_done[v] < self.world.deadline[v]]
            out["saved"] = len(saved)
            out["lost"] = len(self.lost)
            out["saved_fraction"] = len(saved) / cfg.n_victims
        llm = [a for a in self.agents if getattr(a, "kind", "") == "llm"]
        if llm:
            out["llm_decisions"] = sum(a.decisions for a in llm)
            out["llm_fallbacks"] = sum(a.fallbacks for a in llm)
            out["llm_agree"] = sum(a.agree_with_planner for a in llm)
        if self.record_traj:
            out["trajectories"] = [a.trajectory for a in self.agents]
            out["victims"] = sorted(self.world.victims)
        return out


def run_episode(cfg: Config, scheme: str, seed: int, record_traj: bool = False) -> dict:
    return Simulation(cfg, scheme, seed, record_traj).run()


def main():
    ap = argparse.ArgumentParser(description="Run one search-and-rescue episode.")
    ap.add_argument("--scheme", default="gosc", choices=sorted(SCHEMES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr", type=float, default=Config.snr_ref_db)
    ap.add_argument("--symbols", type=int, default=Config.symbols_per_slot)
    ap.add_argument("--agents", type=int, default=Config.n_agents)
    args = ap.parse_args()
    cfg = Config().with_(snr_ref_db=args.snr, symbols_per_slot=args.symbols, n_agents=args.agents)
    out = run_episode(cfg, args.scheme, args.seed)
    out.pop("curve")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
