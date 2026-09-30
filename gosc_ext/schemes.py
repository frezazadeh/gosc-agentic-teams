"""Scheme extensions: imperfect values, per-packet overhead, grant-limited uplink.

Every class subclasses the journal implementation; with the default ExtConfig the
behaviour (and every random draw) is identical to the journal simulator.
"""
import math

import numpy as np

from gosc.messages import Packet, confirm_bits, evidence_sizes, intent_bits
from gosc.schemes import (GOSC, PRIORITY, Genie, Group, NaturalLanguage, Option, RawShare,
                          ReportOnly, SemanticPeriodic, SemanticRateAware,
                          SemanticRateAwareFiltered, SemanticRateAwareSuppress, TxResult,
                          lagrangian_select, sim_tasks)
from gosc.voi import TeamModel


# --------------------------------------------------------------------------
# GOSC with packet overhead and (optionally) imperfect message values
# --------------------------------------------------------------------------
class ExtGOSC(GOSC):
    def __init__(self, cfg, channel, **kw):
        super().__init__(cfg, channel, **kw)
        self.H = cfg.overhead_bits
        self.rng = np.random.default_rng(0)   # re-seeded per mission by ExtSimulation
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
        """Journal GOSC.build_groups with (i) the overhead in the evidence-prefix fit
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
                from gosc.rescue import claim_value
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
        """Transmit the chosen options (journal GOSC.transmit after the selection)."""
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
class FifoExt:
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
    cls = type("X" + base.__name__, (FifoExt, base), {"rate_aware": aware})

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


EXT_SCHEMES = {
    "report": XReport,
    "raw": XRaw,
    "nl": XNL,
    "sem": XSem,
    "sem_ra": XSemRA,
    "sem_raf": XSemRAF,
    "sem_ras": XSemRAS,
    "gosc": lambda cfg, ch: ExtGOSC(cfg, ch),
    "gosc_bits": lambda cfg, ch: ExtGOSC(cfg, ch, voi_mode="bits", name="gosc_bits"),
    "gosc_aoi": lambda cfg, ch: ExtGOSC(cfg, ch, voi_mode="aoi", name="gosc_aoi"),
    "gosc_nofilter": lambda cfg, ch: ExtGOSC(cfg, ch, use_filter=False, name="gosc_nofilter"),
    "genie": XGenie,
}
