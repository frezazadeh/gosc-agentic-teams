"""Realistic radio resources and imperfect message values.

Extends the simulator by subclassing, so that the default Config reproduces the base
model exactly (tests/test_realistic.py):
  * imperfect semantic values (log-normal errors, order-of-magnitude values,
    content-independent permuted values);
  * a fixed per-packet overhead (header, CRC) on every uplink packet and a complete
    uplink + downlink cost account (broadcast, HARQ feedback, grants);
  * a shared, network-wide uplink budget for which the agents compete.

Shared mode (Config.shared_budget = B > 0): GOSC agents apply the decoupled rule at a
broadcast network price and request their selection; periodic/FIFO baselines send a
buffer status report and receive max-min fair grants.  Requests are counted as uplink
control bits (1 bit per channel use) and grants as downlink control bits.
"""
import math
from collections import defaultdict

import numpy as np

from .messages import Packet, confirm_bits, evidence_sizes, intent_bits
from .schemes import (SCHEMES, GOSC, PRIORITY, Genie, Group, NaturalLanguage, Option, RawShare,
                          ReportOnly, SemanticPeriodic, SemanticRateAware,
                          SemanticRateAwareFiltered, SemanticRateAwareSuppress, TxResult,
                          lagrangian_select, sim_tasks)
from .voi import TeamModel
from .simulator import Simulation


# --------------------------------------------------------------------------
# GOSC with packet overhead and (optionally) imperfect message values
# --------------------------------------------------------------------------
class RealisticGOSC(GOSC):
    def __init__(self, cfg, channel, **kw):
        super().__init__(cfg, channel, **kw)
        self.H = cfg.overhead_bits
        self.rng = np.random.default_rng(0)   # re-seeded per mission by RealisticSimulation
        self.reservoir = {"INTENT": [], "EVID": []}
        self.value_pairs = []                  # (true, estimated) group values
        self.n_tx = 0

    # -- imperfect valuation -------------------------------------------------
    def _estimate(self, kind, vals, sizes=None):
        """Map the true values of one candidate message (a dict prefix -> value,
        or {0: value} for an intent) to the estimates the scheduler uses."""
        mode = self.cfg.value_mode
        if mode == "exact":
            return vals
        true = max(vals.values())
        if mode == "lognormal":
            xi = math.exp(self.cfg.value_sigma * self.rng.standard_normal())
            est = {n: v * xi for n, v in vals.items()}
        elif mode == "quantized":
            est = {n: (10.0 ** round(math.log10(v)) if v > 0 else 0.0) for n, v in vals.items()}
        elif mode == "permuted":
            res = self.reservoir[kind]
            u = res[int(self.rng.integers(len(res)))] if res else true
            res.append(true)
            if len(res) > 1000:
                del res[0]
            if sizes is None:
                est = {n: u for n in vals}
            else:
                top = sizes[max(vals) - 1]
                est = {n: u * sizes[n - 1] / top for n in vals}
        else:
            raise ValueError(mode)
        if len(self.value_pairs) < 400:
            self.value_pairs.append((true, max(est.values())))
        return est

    # -- options with overhead ---------------------------------------------
    def _options(self, bits, value, rates, snr_mean, n_records=0):
        W, H = self.cfg.symbols_per_slot, self.H
        opts = []
        for r in rates:
            sym = math.ceil((bits + H) / r)
            if sym <= W:
                opts.append(Option(r, bits + H, sym,
                                   value * float(self.channel.success_prob(r, snr_mean, n=sym)),
                                   n_records))
        return opts

    def build_groups(self, agent, snr_mean, sim):
        """GOSC.build_groups of gosc.schemes with (i) the overhead in the evidence-prefix fit
        and (ii) value estimation errors on intents and evidence."""
        c, H = self.cfg, self.H
        rates = c.rates if self.use_uep else (self.channel.fixed_rate(snr_mean),)
        acked_confirms = getattr(agent, "acked_confirms", set())
        groups = []
        mode = self.voi_mode
        team = TeamModel(sim, agent, c.tom_temperature) if mode == "tom" else None
        for cell in sorted(agent.confirmed_pending - acked_confirms):
            b = confirm_bits(c)
            if not self.use_uep:
                rates = self._fixed_rates(b, snr_mean)
            if not self.use_voi:
                v = b
            elif mode in ("tom", "tv"):
                v = c.confirm_value
            else:
                v = PRIORITY
            groups.append(Group("CONF", self._options(b, v, rates, snr_mean), cell=cell))
        acked_intent = getattr(agent, "acked_intent", agent.shared_intent)
        if agent.target is not None and agent.target != agent.shared_intent and agent.target != acked_intent:
            b = intent_bits(c)
            if not self.use_uep:
                rates = self._fixed_rates(b, snr_mean)
            if team is not None and c.task == "rescue" and agent.target in sim_tasks(sim, agent):
                from .rescue import claim_value
                v = claim_value(sim, agent, {**sim.open_tasks(),
                                             agent.target: sim_tasks(sim, agent)[agent.target]})
            elif team is not None:
                v = max(team.intent_voi(agent.target), 0.0)
            elif mode == "aoi":
                v = float(sim.t - getattr(agent, "_intent_t", sim.t) + 1)
            elif mode == "bits":
                v = b
            elif self.use_voi:
                if agent.shared_intent is None:
                    v = c.intent_value
                else:
                    (x1, y1), (x2, y2) = divmod(agent.target, c.grid), divmod(agent.shared_intent, c.grid)
                    v = c.intent_value * min(1.0, (abs(x1 - x2) + abs(y1 - y2)) / (2 * c.sense_radius + 1))
            else:
                v = b
            v = self._estimate("INTENT", {0: v})[0]
            groups.append(Group("INTENT", self._options(b, v, rates, snr_mean)))
        recs = agent.pending_records
        if recs:
            sizes = evidence_sizes(recs, c)
            if not self.use_uep:
                rates = self._fixed_rates(sizes[0], snr_mean)
            fits = {}
            for r in rates:
                cap = int(c.symbols_per_slot * r) - H
                n = int(np.searchsorted(sizes, cap, side="right"))
                if n > 0:
                    fits[r] = n
            if fits:
                vals = (self.evidence_voi(sim, recs, fits.values(), team)
                        if mode in ("tom", "tv", "aoi")
                        else {n: float(sizes[n - 1]) for n in fits.values()})
                vals = self._estimate("EVID", vals, sizes)
                opts = []
                for r, n in fits.items():
                    b = sizes[n - 1]
                    opts += self._options(b, vals[n], (r,), snr_mean, n_records=n)
                groups.append(Group("EVID", opts))
        return [g for g in groups if g.options]

    # -- transmission --------------------------------------------------------
    def apply(self, agent, chosen, t, snr_inst, sim):
        """Transmit the chosen options (GOSC.transmit of gosc.schemes after the selection)."""
        res = TxResult([])
        for g, o in chosen:
            res.symbols += o.symbols
            res.bits += o.bits
            self.n_tx += 1
            if not self.channel.decodes(o.rate, snr_inst, n=o.symbols):
                continue
            if g.kind == "CONF":
                agent.acked_confirms = getattr(agent, "acked_confirms", set()) | {g.cell}
                res.delivered.append(Packet("CONF", agent.idx, o.bits - self.H, t, confirm=g.cell))
            elif g.kind == "INTENT":
                agent.acked_intent = agent.target
                res.delivered.append(Packet("INTENT", agent.idx, o.bits - self.H, t, intent=agent.target))
            else:
                recs = agent.pending_records[:o.n_records]
                del agent.pending_records[:o.n_records]
                for r in recs:
                    sim.sensor.apply(agent.queued_llr, r, sign=-1.0)
                res.delivered.append(Packet("EVID", agent.idx, o.bits - self.H, t, records=recs))
        return res

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        groups = self.build_groups(agent, snr_mean, sim)
        chosen = lagrangian_select(groups, self.cfg.symbols_per_slot, self.price)
        if self.log is not None:
            self.log.append((groups, chosen, self.price, self.cfg.symbols_per_slot))
        return self.apply(agent, chosen, t, snr_inst, sim)


# --------------------------------------------------------------------------
# Periodic / FIFO baselines with overhead and an explicit per-slot grant
# --------------------------------------------------------------------------
class FifoRealistic:
    """Mixin for the FIFO baselines.  One transport block per slot and agent carries
    the multiplexed queue content plus `overhead_bits`; the grant W defaults to the
    orthogonal per-agent budget."""
    rate_aware = False

    def rate_for(self, snr_mean, W):
        H = self.cfg.overhead_bits
        rates = self.channel.rates
        if self.rate_aware:
            n = self.cfg.symbols_per_slot
            good = np.maximum(W * rates - H, 0.0) * np.asarray(self.channel.success_prob(rates, snr_mean, n=n))
            return float(rates[int(np.argmax(good))]) if good.max() > 0 else None
        r0 = self.channel.fixed_rate(snr_mean)
        if int(W * r0) - H > 0:
            return r0
        ok = [float(r) for r in sorted(rates) if int(W * r) - H > 0]
        return ok[0] if ok else None

    def backlog(self, agent, t, sim):
        c = self.cfg
        agent.prio = type(agent.prio)(p for p in agent.prio if not sim.declared[p.confirm])
        while agent.queue and t - agent.queue[0].created > c.max_packet_age:
            agent.queue.popleft()
        return sum(p.bits - p.sent for p in agent.prio) + sum(p.bits - p.sent for p in agent.queue)

    def request(self, agent, t, snr_mean, sim):
        """Channel uses the agent needs this slot (for a shared-budget grant)."""
        b = self.backlog(agent, t, sim)
        if b == 0:
            return 0
        W = self.cfg.symbols_per_slot
        rate = self.rate_for(snr_mean, W)
        if rate is None:
            return 0
        use = min(int(W * rate) - self.cfg.overhead_bits, b)
        return math.ceil((use + self.cfg.overhead_bits) / rate) if use > 0 else 0

    def transmit(self, agent, t, snr_mean, snr_inst, sim, W=None):
        c, H = self.cfg, self.cfg.overhead_bits
        full = c.symbols_per_slot
        W = full if W is None else W
        backlog = self.backlog(agent, t, sim)
        if backlog == 0 or W <= 0:
            return TxResult([])
        rate = self.rate_for(snr_mean, full)
        if rate is None:
            return TxResult([])
        cap = int(W * rate) - H
        if cap <= 0:
            return TxResult([])
        use = min(cap, backlog)
        res = TxResult([], symbols=math.ceil((use + H) / rate), bits=use + H)
        self.n_tx += 1
        if not self.channel.decodes(rate, snr_inst, n=res.symbols):
            return res
        budget = use
        for q in (agent.prio, agent.queue):
            while q and budget > 0:
                p = q[0]
                take = min(p.bits - p.sent, budget)
                p.sent += take
                budget -= take
                if p.sent == p.bits:
                    res.delivered.append(q.popleft())
        if isinstance(self, SemanticRateAwareFiltered):
            for p in res.delivered:
                for r in p.records:
                    sim.sensor.apply(agent.queued_llr, r, sign=-1.0)
        return res


def _fifo(base, aware=False):
    cls = type("X" + base.__name__, (FifoRealistic, base), {"rate_aware": aware})

    def init(self, cfg, channel):
        base.__init__(self, cfg, channel)
        self.n_tx = 0
    cls.__init__ = init
    return cls


XReport = _fifo(ReportOnly)
XRaw = _fifo(RawShare)
XNL = _fifo(NaturalLanguage)
XSem = _fifo(SemanticPeriodic)
XSemRA = _fifo(SemanticRateAware, aware=True)
XSemRAF = _fifo(SemanticRateAwareFiltered, aware=True)
XSemRAS = _fifo(SemanticRateAwareSuppress, aware=True)


class XGenie(Genie):
    def __init__(self, cfg, channel):
        super().__init__(cfg, channel)
        self.n_tx = 0


REALISTIC_SCHEMES = {
    "report": XReport,
    "raw": XRaw,
    "nl": XNL,
    "sem": XSem,
    "sem_ra": XSemRA,
    "sem_raf": XSemRAF,
    "sem_ras": XSemRAS,
    "gosc": lambda cfg, ch: RealisticGOSC(cfg, ch),
    "gosc_bits": lambda cfg, ch: RealisticGOSC(cfg, ch, voi_mode="bits", name="gosc_bits"),
    "gosc_aoi": lambda cfg, ch: RealisticGOSC(cfg, ch, voi_mode="aoi", name="gosc_aoi"),
    "gosc_nofilter": lambda cfg, ch: RealisticGOSC(cfg, ch, use_filter=False, name="gosc_nofilter"),
    "genie": XGenie,
}


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


class RealisticSimulation(Simulation):
    def __init__(self, cfg, scheme, seed, record_traj=False):
        super().__init__(cfg, scheme if scheme in SCHEMES else "report", seed, record_traj)
        self.scheme = REALISTIC_SCHEMES[scheme](cfg, self.channel)
        if hasattr(self.scheme, "rng"):
            self.scheme.rng = np.random.default_rng([seed, 7])
        self.req_bits = 0
        self.grant_bits = 0

    # ------------------------------------------------------------------
    def phase_communicate(self, t):
        cfg = self.cfg
        if not cfg.shared_budget or not isinstance(self.scheme, (RealisticGOSC, FifoRealistic)):
            return super().phase_communicate(t)
        links = []
        for a in self.agents:
            snr_mean = self.channel.mean_snr(a.pos)
            snr_inst = snr_mean * self.chan_rng.exponential()
            links.append((a, snr_inst if cfg.csit else snr_mean, snr_inst))
        if isinstance(self.scheme, RealisticGOSC):
            results = (self._uplink_gosc(t, links) if cfg.shared_mode == "central"
                       else self._uplink_gosc_price(t, links))
        else:
            results = self._uplink_fifo(t, links)
        self.skip_tx = set()
        # edge fusion, broadcast, motion: as in gosc.simulator
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
        rates with the decoupled rule at price eta + mu and requests the total channel
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


def run_realistic(cfg, scheme, seed):
    out = RealisticSimulation(cfg, scheme, seed).run()
    out.pop("curve", None)
    return out
