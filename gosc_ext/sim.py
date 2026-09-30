"""Simulation with complete cost accounting and an optional shared uplink budget.

Orthogonal mode (shared_budget = 0, the journal model): every agent owns
`symbols_per_slot` channel uses per slot; the slot loop is the journal's.

Shared mode (shared_budget = B > 0): all agents compete for B channel uses per slot.
  * GOSC: each agent reports the options of its candidate messages that are
    Lagrangian-optimal for some multiplier (a (value, size) pair per option); the
    base station solves one network-wide multiple-choice knapsack with the same
    Lagrangian scheduler and grants the selected options.
  * Periodic/FIFO baselines: each agent with a backlog sends a buffer status report;
    the base station divides B by max-min fair water-filling over the requests.
Requests are counted as uplink control bits (1 bit per channel use) and grants as
downlink control bits.
"""
import math
from collections import defaultdict

import numpy as np

from gosc.schemes import SCHEMES, TxResult, lagrangian_select
from gosc.simulator import Simulation

from .schemes import EXT_SCHEMES, ExtGOSC, FifoExt


def hull_size(options, eta):
    """Number of options that maximize value - lambda * symbols (and are positive)
    for some lambda >= eta, i.e. what a GOSC agent reports to the base station."""
    lams = {eta}
    for i, o in enumerate(options):
        if o.symbols > 0:
            lams.add(o.value / o.symbols)
        for p in options[i + 1:]:
            if o.symbols != p.symbols:
                lams.add((o.value - p.value) / (o.symbols - p.symbols))
    chosen = set()
    for lam in lams:
        if lam < eta:
            continue
        for x in (lam, lam * (1 + 1e-9) + 1e-15):
            best, bu = None, 0.0
            for j, o in enumerate(options):
                u = o.value - x * o.symbols
                if u > bu:
                    best, bu = j, u
            if best is not None:
                chosen.add(best)
    return len(chosen)


def waterfill(needs, total, min_useful, offset=0):
    """Max-min fair division of `total` channel uses over integer requests; grants
    smaller than an agent's minimum useful size are withdrawn and redistributed."""
    n = len(needs)
    grants = [0] * n
    active = [i for i in range(n) if needs[i] > 0]
    for _ in range(2):
        grants = [0] * n
        rem = total
        act = list(active)
        while act and rem > 0:
            share = rem // len(act)
            if share == 0:
                order = sorted(act, key=lambda i: (i - offset) % n)
                for i in order[:rem]:
                    grants[i] += 1
                break
            done = []
            for i in act:
                give = min(share, needs[i] - grants[i])
                grants[i] += give
                rem -= give
                if grants[i] >= needs[i]:
                    done.append(i)
            act = [i for i in act if i not in done]
            if not done and share * len(act) >= rem:
                continue
        small = [i for i in active if 0 < grants[i] < min_useful[i]]
        if not small:
            break
        active = [i for i in active if i not in small]
    return grants


class ExtSimulation(Simulation):
    def __init__(self, cfg, scheme, seed, record_traj=False):
        super().__init__(cfg, scheme if scheme in SCHEMES else "report", seed, record_traj)
        self.scheme = EXT_SCHEMES[scheme](cfg, self.channel)
        if hasattr(self.scheme, "rng"):
            self.scheme.rng = np.random.default_rng([seed, 7])
        self.req_bits = 0
        self.grant_bits = 0

    # ------------------------------------------------------------------
    def phase_communicate(self, t):
        cfg = self.cfg
        if not cfg.shared_budget or not isinstance(self.scheme, (ExtGOSC, FifoExt)):
            return super().phase_communicate(t)
        links = []
        for a in self.agents:
            snr_mean = self.channel.mean_snr(a.pos)
            snr_inst = snr_mean * self.chan_rng.exponential()
            links.append((a, snr_inst if cfg.csit else snr_mean, snr_inst))
        if isinstance(self.scheme, ExtGOSC):
            results = (self._uplink_gosc(t, links) if cfg.shared_mode == "central"
                       else self._uplink_gosc_price(t, links))
        else:
            results = self._uplink_fifo(t, links)
        self.skip_tx = set()
        # edge fusion, broadcast, motion: as in the journal simulator
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
        for a in self.agents:
            a.step()
        if cfg.task == "rescue" and self._rescue_update(t):
            self.done_at = t + 1
            return True
        return False

    def _uplink_gosc(self, t, links):
        cfg, sch = self.cfg, self.scheme
        groups = []
        for a, known, _ in links:
            gs = sch.build_groups(a, known, self.view(a))
            n_opt = sum(hull_size(g.options, sch.price) for g in gs)
            if n_opt:
                self.req_bits += cfg.req_header_bits + cfg.req_option_bits * n_opt
            for g in gs:
                g.agent = a.idx
            groups += gs
        chosen = lagrangian_select(groups, cfg.shared_budget, sch.price)
        by_agent = defaultdict(list)
        for g, o in chosen:
            by_agent[g.agent].append((g, o))
        results = []
        for a, known, inst in links:
            ch = by_agent.get(a.idx, [])
            if ch:
                self.grant_bits += cfg.grant_bits
            res = sch.apply(a, ch, t, inst, self.view(a))
            self.bits += res.bits
            self.symbols += res.symbols
            results.append((a, res))
        return results

    def _uplink_gosc_price(self, t, links):
        """Price-coordinated GOSC: the base station broadcasts a network price mu
        (the multiplier of the shared budget); each agent selects its messages and
        rates with the journal rule at price eta + mu and requests the total channel
        uses with the value density of its selection (16 bits); the base station
        grants requests in decreasing value density while they fit, and updates mu
        multiplicatively towards demand = budget."""
        cfg, sch = self.cfg, self.scheme
        B = cfg.shared_budget
        mu = getattr(self, "mu", 0.0)
        lam = sch.price + mu
        reqs = []
        for a, known, inst in links:
            gs = sch.build_groups(a, known, self.view(a))
            ch = []
            for g in gs:
                best, bu = None, 0.0
                for o in g.options:
                    u = o.value - lam * o.symbols
                    if u > bu:
                        best, bu = o, u
                if best is not None:
                    ch.append((g, best))
            if ch:
                s = sum(o.symbols for _, o in ch)
                v = sum(o.value for _, o in ch)
                reqs.append((v / s, s, a.idx, ch))
                self.req_bits += cfg.req_header_bits + 8
        demand = sum(r[1] for r in reqs)
        granted, used = {}, 0
        for dens, s, k, ch in sorted(reqs, key=lambda r: -r[0]):
            if used + s <= B:
                granted[k] = ch
                used += s
        results = []
        for a, known, inst in links:
            ch = granted.get(a.idx, [])
            if ch:
                self.grant_bits += cfg.grant_bits
            res = sch.apply(a, ch, t, inst, self.view(a))
            self.bits += res.bits
            self.symbols += res.symbols
            results.append((a, res))
        floor = 1e-7
        if demand > B:
            mu = max(mu, floor) * math.exp(cfg.price_gain * min(demand / B - 1.0, 3.0))
        else:
            mu = mu * math.exp(cfg.price_gain * (demand / B - 1.0))
            if mu < floor:
                mu = 0.0
        self.mu = mu
        self.grant_bits += cfg.price_bits
        return results

    def _uplink_fifo(self, t, links):
        cfg, sch = self.cfg, self.scheme
        needs, min_useful = [], []
        for a, known, _ in links:
            need = sch.request(a, t, known, self.view(a))
            rate = sch.rate_for(known, cfg.symbols_per_slot) if need else None
            needs.append(need)
            min_useful.append(math.ceil((cfg.overhead_bits + 1) / rate) if rate else 1)
            if need:
                self.req_bits += cfg.bsr_bits
        grants = waterfill(needs, cfg.shared_budget, min_useful, offset=t)
        results = []
        for (a, known, inst), g in zip(links, grants):
            if g > 0:
                self.grant_bits += cfg.grant_bits
            res = sch.transmit(a, t, known, inst, self.view(a), W=g)
            self.bits += res.bits
            self.symbols += res.symbols
            results.append((a, res))
        return results

    # ------------------------------------------------------------------
    def result(self):
        out = super().result()
        cfg = self.cfg
        H = cfg.overhead_bits
        ntx = int(getattr(self.scheme, "n_tx", 0))
        bslots = sum(1 for v in self.bcast_bits.values() if v > 0)
        dl_ctrl = cfg.ack_bits * ntx + self.grant_bits
        dl_oh = H * bslots if cfg.bcast_overhead else 0
        out.update(
            ul_tx=ntx,
            ul_overhead_bits=H * ntx,
            ul_ctrl_symbols=int(self.req_bits),          # requests at 1 bit per channel use
            dl_bcast_symbols=int(math.ceil((self.dl_bits + dl_oh) / cfg.dl_rate)),
            dl_ctrl_symbols=int(math.ceil(dl_ctrl / cfg.dl_rate)),
        )
        out["total_symbols"] = (out["symbols"] + out["ul_ctrl_symbols"]
                                + out["dl_bcast_symbols"] + out["dl_ctrl_symbols"])
        pairs = getattr(self.scheme, "value_pairs", None)
        if pairs:
            rng = np.random.default_rng(1)
            idx = rng.choice(len(pairs), size=min(60, len(pairs)), replace=False)
            out["value_pairs"] = [pairs[i] for i in sorted(idx)]
        return out


def run_ext(cfg, scheme, seed):
    out = ExtSimulation(cfg, scheme, seed).run()
    out.pop("curve", None)
    return out
