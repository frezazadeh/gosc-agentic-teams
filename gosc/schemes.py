"""Communication schemes: the proposed GOSC framework and the baselines.

Every scheme decides, for one agent and one slot, which packets are put on the
uplink, at which rate, and returns the packets that were decoded by the edge.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .belief import sigmoid
from .channel import Channel
from .config import Config
from .voi import TeamModel
from .messages import (Packet, cell_bits, confirm_bits, evidence_bits, evidence_sizes, intent_bits,
                       nl_confirm_text, nl_step_text, raw_bits, sem_step_bits, text_bits)


@dataclass
class TxResult:
    delivered: List[Packet]
    symbols: int = 0
    bits: int = 0


class Scheme:
    name = "base"
    shares_evidence = True

    def __init__(self, cfg: Config, channel: Channel):
        self.cfg = cfg
        self.channel = channel

    def on_step(self, agent, rec, t: int, sim) -> None:
        """Called once per slot after sensing and planning."""

    def on_confirm(self, agent, cell: int, t: int) -> None:
        """Called when the agent's own posterior crosses the confirmation threshold."""

    def transmit(self, agent, t: int, snr_mean: float, snr_inst: float, sim) -> TxResult:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Conventional (task-agnostic) baselines: FIFO queue, fixed link adaptation,
# confirmations served with strict priority, stale data discarded.
# --------------------------------------------------------------------------
class FifoScheme(Scheme):
    def _rate(self, snr_mean: float) -> float:
        """Conventional link adaptation; if no rate meets the BLER target, the lowest
        rate is used (packets are fragmented over slots when needed)."""
        return self.channel.fixed_rate(snr_mean)

    def _data_packet(self, agent, rec, t) -> Optional[Packet]:
        return None

    def _confirm_packet(self, agent, cell, t) -> Packet:
        return Packet("CONF", agent.idx, confirm_bits(self.cfg), t, confirm=cell)

    def on_step(self, agent, rec, t, sim):
        pkt = self._data_packet(agent, rec, t)
        if pkt is not None:
            agent.queue.append(pkt)

    def on_confirm(self, agent, cell, t):
        agent.prio.append(self._confirm_packet(agent, cell, t))

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        c = self.cfg
        agent.prio = type(agent.prio)(p for p in agent.prio if not sim.declared[p.confirm])
        while agent.queue and t - agent.queue[0].created > c.max_packet_age:
            agent.queue.popleft()
        backlog = sum(p.bits - p.sent for p in agent.prio) + sum(p.bits - p.sent for p in agent.queue)
        if backlog == 0:
            return TxResult([])
        rate = self._rate(snr_mean)
        use = min(int(c.symbols_per_slot * rate), backlog)
        res = TxResult([], symbols=math.ceil(use / rate), bits=use)
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
        return res


class ReportOnly(FifoScheme):
    """Agents only report confirmed survivors; no coordination traffic."""
    name = "report"
    shares_evidence = False


class RawShare(FifoScheme):
    """Bit-level sharing of raw sensor frames (conventional Shannon-layer design)."""
    name = "raw"

    def _data_packet(self, agent, rec, t):
        return Packet("RAW", agent.idx, raw_bits(self.cfg), t, records=[rec])


class NaturalLanguage(FifoScheme):
    """LLM-style verbose natural-language status messages."""
    name = "nl"

    def _data_packet(self, agent, rec, t):
        s = nl_step_text(agent.idx, t, rec.pos, rec.det, agent.target, self.cfg)
        return Packet("NL", agent.idx, text_bits(s), t, records=[rec], intent=agent.target)

    def _confirm_packet(self, agent, cell, t):
        return Packet("CONF", agent.idx, text_bits(nl_confirm_text(agent.idx, cell, self.cfg)),
                      t, confirm=cell)


class SemanticPeriodic(FifoScheme):
    """Compact structured semantic updates every slot, but task-agnostic transport."""
    name = "sem"

    def _data_packet(self, agent, rec, t):
        return Packet("SEM", agent.idx, sem_step_bits(len(rec.det), self.cfg), t,
                      records=[rec], intent=agent.target)


class SemanticAggregated(FifoScheme):
    """Strong periodic baseline: every `sem_interval` slots, one packet carries the
    records of the interval with the same lossless aggregated encoding as GOSC,
    plus the current intention; FIFO queue and conventional link adaptation."""
    name = "sem_agg"

    def on_step(self, agent, rec, t, sim):
        batch = getattr(agent, "batch", None)
        if batch is None:
            batch = agent.batch = []
        batch.append(rec)
        if len(batch) >= self.cfg.sem_interval:
            bits = evidence_sizes(batch, self.cfg)[-1] + cell_bits(self.cfg)
            agent.queue.append(Packet("SEMA", agent.idx, bits, t, records=list(batch),
                                      intent=agent.target))
            batch.clear()


class SemanticRateAwareFiltered(SemanticAggregated):
    """Rate-aware periodic scheduler with GOSC's semantic source filter: records that
    would move the team belief (given what is already queued) by less than
    filter_eps are not queued.  Isolates the effect of value-priced scheduling."""
    name = "sem_raf"

    def _rate(self, snr_mean):
        return SemanticRateAware._rate(self, snr_mean)

    def on_step(self, agent, rec, t, sim):
        if not hasattr(agent, "queued_llr"):
            agent.queued_llr = np.zeros_like(sim.Lc)
        sx, sy = sim.sensor.window(rec.pos)
        base = (sim.Lc + agent.queued_llr)[sx, sy]
        new = base.copy()
        new += sim.sensor.llr_neg
        n = self.cfg.grid
        for c in rec.det:
            new[c // n - sx.start, c % n - sy.start] += sim.sensor.llr_pos - sim.sensor.llr_neg
        live = ~sim.declared.reshape(sim.Lc.shape)[sx, sy]
        keep = float(np.abs(sigmoid(new) - sigmoid(base))[live].sum()) >= self.cfg.filter_eps
        batch = getattr(agent, "batch", None)
        if batch is None:
            batch = agent.batch = []
        if keep:
            batch.append(rec)
            sim.sensor.apply(agent.queued_llr, rec)
        agent._slots = getattr(agent, "_slots", 0) + 1
        if agent._slots >= self.cfg.sem_interval:
            agent._slots = 0
            if batch:
                bits = evidence_sizes(batch, self.cfg)[-1] + cell_bits(self.cfg)
                agent.queue.append(Packet("SEMA", agent.idx, bits, t, records=list(batch),
                                          intent=agent.target))
                batch.clear()

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        res = super().transmit(agent, t, snr_mean, snr_inst, sim)
        for p in res.delivered:
            for r in p.records:
                sim.sensor.apply(agent.queued_llr, r, sign=-1.0)
        return res


class SemanticRateAwareSuppress(SemanticRateAwareFiltered):
    """Information-suppressing periodic baseline (send-on-delta with periodic timing):
    a record is retained only if it would move the team belief (given what is already
    queued) by at least the suppression threshold `filter_eps`, which is swept like
    GOSC's price.  Every `sem_interval` slots it sends the retained records plus the
    intention; with no retained record it sends a short intent-only packet if the
    intention changed, and nothing otherwise.  Goodput-optimal rate, fragmentation."""
    name = "sem_ras"

    def on_step(self, agent, rec, t, sim):
        c = self.cfg
        if not hasattr(agent, "queued_llr"):
            agent.queued_llr = np.zeros_like(sim.Lc)
        sx, sy = sim.sensor.window(rec.pos)
        base = (sim.Lc + agent.queued_llr)[sx, sy]
        new = base.copy()
        new += sim.sensor.llr_neg
        n = c.grid
        for cell in rec.det:
            new[cell // n - sx.start, cell % n - sy.start] += sim.sensor.llr_pos - sim.sensor.llr_neg
        live = ~sim.declared.reshape(sim.Lc.shape)[sx, sy]
        batch = getattr(agent, "batch", None)
        if batch is None:
            batch = agent.batch = []
        if float(np.abs(sigmoid(new) - sigmoid(base))[live].sum()) >= c.filter_eps:
            batch.append(rec)
            sim.sensor.apply(agent.queued_llr, rec)
        agent._slots = getattr(agent, "_slots", 0) + 1
        if agent._slots < c.sem_interval:
            return
        agent._slots = 0
        if batch:
            bits = evidence_sizes(batch, c)[-1] + cell_bits(c)
            agent.queue.append(Packet("SEMA", agent.idx, bits, t, records=list(batch),
                                      intent=agent.target))
            batch.clear()
            agent._sent_intent = agent.target
        elif agent.target is not None and agent.target != getattr(agent, "_sent_intent", None):
            agent.queue.append(Packet("INTENT", agent.idx, intent_bits(c), t, intent=agent.target))
            agent._sent_intent = agent.target


class SemanticRateAware(SemanticAggregated):
    """Rate-aware periodic semantic scheduler (strong external baseline): aggregated
    periodic updates as `sem_agg`, but each slot uses the rate that maximizes the
    expected goodput R * P_s(R) for a full-slot packet, with fragmentation."""
    name = "sem_ra"

    def _rate(self, snr_mean):
        W = self.cfg.symbols_per_slot
        rates = self.channel.rates
        good = rates * np.asarray(self.channel.success_prob(rates, snr_mean, n=W))
        return float(rates[int(np.argmax(good))])


# --------------------------------------------------------------------------
# Proposed: goal-oriented semantic communication (GOSC)
# --------------------------------------------------------------------------
@dataclass
class Option:
    rate: float
    bits: int
    symbols: int
    value: float          # expected goal value = VoI * P(success)
    n_records: int = 0


@dataclass
class Group:
    kind: str
    options: List[Option]
    cell: Optional[int] = None


def lagrangian_select(groups: List[Group], budget: int, price: float):
    """max sum(value) - price*symbols  s.t.  sum(symbols) <= budget, one option per group.

    Solved by dualising the budget constraint (bisection on the multiplier) followed by
    a greedy fill of the residual budget.
    """
    def pick(lam):
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

    chosen, used = pick(price)
    if used <= budget:
        return chosen
    lo, hi = 0.0, max((o.value / o.symbols for g in groups for o in g.options), default=1.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pick(price + mid)[1] <= budget:
            hi = mid
        else:
            lo = mid
    chosen, used = pick(price + hi)
    taken = {id(g) for g, _ in chosen}
    for g in sorted((g for g in groups if id(g) not in taken),
                    key=lambda g: -max(o.value / o.symbols for o in g.options)):
        fits = [o for o in g.options
                if o.symbols <= budget - used and o.value - price * o.symbols > 0]
        if fits:
            o = max(fits, key=lambda o: o.value - price * o.symbols)
            chosen.append((g, o))
            used += o.symbols
    return chosen


PRIORITY = 1e6  # confirmation value under the non-goal valuation rules (strict priority)


class GOSC(Scheme):
    """Value-of-information scheduling with importance-aware rate selection.

    voi_mode selects the valuation rule, all on the same aggregation/transport:
      "tom"  decision-theoretic theory-of-mind VoI (proposed),
      "tv"   total-variation shift of the common belief,
      "bits" payload size (throughput objective),
      "aoi"  age reduction of the delivered records (freshness objective).
    use_voi=False is the legacy throughput ablation (bits, no filter, zero price).
    use_filter toggles the semantic source filter independently.
    use_uep=False restricts every packet to the conventional link-adaptation rate.
    """
    name = "gosc"

    def __init__(self, cfg, channel, use_voi=True, use_uep=True, voi_mode="tom", name=None,
                 use_filter=None):
        super().__init__(cfg, channel)
        if not use_voi:
            voi_mode = "bits"
        self.use_voi = use_voi
        self.voi_mode = voi_mode
        self.use_uep = use_uep
        self.price = cfg.energy_price if use_voi else 0.0
        if use_filter is None:
            use_filter = use_voi
        self.filter_eps = cfg.filter_eps if use_filter else -1.0
        self.log = None  # set to a list to record every scheduling instance
        if name:
            self.name = name

    def on_step(self, agent, rec, t, sim):
        """Semantic source filter: queue the record only if it would move the team
        belief (given everything already queued) by more than filter_eps."""
        if not hasattr(agent, "queued_llr"):
            agent.queued_llr = np.zeros_like(sim.Lc)
        if agent.target != getattr(agent, "_last_target", None):
            agent._last_target, agent._intent_t = agent.target, t
        if self.filter_eps >= 0:
            sx, sy = sim.sensor.window(rec.pos)
            base = (sim.Lc + agent.queued_llr)[sx, sy]
            new = base.copy()
            new += sim.sensor.llr_neg
            n = self.cfg.grid
            for c in rec.det:
                new[c // n - sx.start, c % n - sy.start] += sim.sensor.llr_pos - sim.sensor.llr_neg
            live = ~sim.declared.reshape(sim.Lc.shape)[sx, sy]
            if float(np.abs(sigmoid(new) - sigmoid(base))[live].sum()) < self.filter_eps:
                return
        agent.pending_records.append(rec)
        sim.sensor.apply(agent.queued_llr, rec)

    # -- value of information ---------------------------------------------
    def evidence_voi(self, sim, records, prefixes, team=None):
        """VoI of delivering the first n pending records, for each n.

        voi_mode="tom": decision-theoretic VoI under the theory-of-mind team model.
        voi_mode="tv":  total-variation shift of the team's survivor-occupancy belief.
        """
        delta = np.zeros_like(sim.Lc)
        ns, deltas, i = sorted(set(prefixes)), [], 0
        for n in ns:
            while i < n:
                sim.sensor.apply(delta, records[i])
                i += 1
            deltas.append(delta.copy())
        if self.voi_mode == "tom":
            vals = team.evidence_voi(deltas, [records[n - 1].pos for n in ns])
        elif self.voi_mode == "aoi":
            vals = [float(sum(sim.t - r.t + 1 for r in records[:n])) for n in ns]
        else:
            p0 = sigmoid(sim.Lc)
            live = ~sim.declared.reshape(sim.Lc.shape)
            vals = [float(np.abs(sigmoid(sim.Lc + d) - p0)[live].sum()) for d in deltas]
        return {n: max(v, 0.0) for n, v in zip(ns, vals)}

    def _options(self, bits, value, rates, snr_mean, n_records=0):
        W = self.cfg.symbols_per_slot
        opts = []
        for r in rates:
            sym = math.ceil(bits / r)
            if sym <= W:
                opts.append(Option(r, bits, sym,
                                   value * float(self.channel.success_prob(r, snr_mean, n=sym)),
                                   n_records))
        return opts

    def _fixed_rates(self, bits, snr_mean):
        """Fixed-rate ablation: the conventional rate, or, if the packet does not fit
        into one slot at that rate, the lowest higher rate at which it fits."""
        c = self.cfg
        r0 = self.channel.fixed_rate(snr_mean)
        for r in sorted(c.rates):
            if r >= r0 and math.ceil(bits / r) <= c.symbols_per_slot:
                return (r,)
        return (r0,)

    def build_groups(self, agent, snr_mean, sim):
        c = self.cfg
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
            groups.append(Group("INTENT", self._options(b, v, rates, snr_mean)))
        recs = agent.pending_records
        if recs:
            sizes = evidence_sizes(recs, c)
            if not self.use_uep:
                rates = self._fixed_rates(sizes[0], snr_mean)
            fits = {}
            for r in rates:
                cap = int(c.symbols_per_slot * r)
                n = int(np.searchsorted(sizes, cap, side="right"))
                if n > 0:
                    fits[r] = n
            if fits:
                vals = (self.evidence_voi(sim, recs, fits.values(), team)
                        if mode in ("tom", "tv", "aoi")
                        else {n: float(sizes[n - 1]) for n in fits.values()})
                opts = []
                for r, n in fits.items():
                    b = sizes[n - 1]
                    opts += self._options(b, vals[n], (r,), snr_mean, n_records=n)
                groups.append(Group("EVID", opts))
        return [g for g in groups if g.options]

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        groups = self.build_groups(agent, snr_mean, sim)
        chosen = lagrangian_select(groups, self.cfg.symbols_per_slot, self.price)
        if self.log is not None:
            self.log.append((groups, chosen, self.price, self.cfg.symbols_per_slot))
        res = TxResult([])
        for g, o in chosen:
            res.symbols += o.symbols
            res.bits += o.bits
            if not self.channel.decodes(o.rate, snr_inst, n=o.symbols):
                continue
            if g.kind == "CONF":
                # reliable HARQ acknowledgment: never re-send a delivered confirmation
                agent.acked_confirms = getattr(agent, "acked_confirms", set()) | {g.cell}
                res.delivered.append(Packet("CONF", agent.idx, o.bits, t, confirm=g.cell))
            elif g.kind == "INTENT":
                agent.acked_intent = agent.target
                res.delivered.append(Packet("INTENT", agent.idx, o.bits, t, intent=agent.target))
            else:
                recs = agent.pending_records[:o.n_records]
                del agent.pending_records[:o.n_records]
                for r in recs:
                    sim.sensor.apply(agent.queued_llr, r, sign=-1.0)
                res.delivered.append(Packet("EVID", agent.idx, o.bits, t, records=recs))
        return res


class Genie(GOSC):
    """Ideal reference: every update reaches the team instantly and error-free."""
    name = "genie"

    def __init__(self, cfg, channel):
        super().__init__(cfg, channel, use_voi=False)

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        c = self.cfg
        out = TxResult([])
        for cell in sorted(agent.confirmed_pending):
            out.delivered.append(Packet("CONF", agent.idx, confirm_bits(c), t, confirm=cell))
        if agent.target != agent.shared_intent:
            out.delivered.append(Packet("INTENT", agent.idx, intent_bits(c), t, intent=agent.target))
        if agent.pending_records:
            recs = agent.pending_records
            out.delivered.append(Packet("EVID", agent.idx, evidence_sizes(recs, c)[-1], t,
                                        records=list(recs)))
            agent.pending_records = []
            agent.queued_llr[:] = 0.0
        out.bits = sum(p.bits for p in out.delivered)
        return out


def sim_tasks(sim, agent):
    return sim.known_tasks(agent) if hasattr(sim, "known_tasks") else {}


def _joint():
    from .joint import GOSCJoint
    return GOSCJoint


SCHEMES = {
    "report": lambda cfg, ch: ReportOnly(cfg, ch),
    "raw": lambda cfg, ch: RawShare(cfg, ch),
    "nl": lambda cfg, ch: NaturalLanguage(cfg, ch),
    "sem": lambda cfg, ch: SemanticPeriodic(cfg, ch),
    "gosc": lambda cfg, ch: GOSC(cfg, ch),
    "gosc_nouep": lambda cfg, ch: GOSC(cfg, ch, use_uep=False, name="gosc_nouep"),
    "gosc_novoi": lambda cfg, ch: GOSC(cfg, ch, use_voi=False, name="gosc_novoi"),
    "gosc_tv": lambda cfg, ch: GOSC(cfg, ch, voi_mode="tv", name="gosc_tv"),
    "gosc_nofilter": lambda cfg, ch: GOSC(cfg, ch, use_filter=False, name="gosc_nofilter"),
    "gosc_bits": lambda cfg, ch: GOSC(cfg, ch, voi_mode="bits", name="gosc_bits"),
    "gosc_aoi": lambda cfg, ch: GOSC(cfg, ch, voi_mode="aoi", name="gosc_aoi"),
    "sem_agg": lambda cfg, ch: SemanticAggregated(cfg, ch),
    "sem_ra": lambda cfg, ch: SemanticRateAware(cfg, ch),
    "sem_raf": lambda cfg, ch: SemanticRateAwareFiltered(cfg, ch),
    "sem_ras": lambda cfg, ch: SemanticRateAwareSuppress(cfg, ch),
    "gosc_joint": lambda cfg, ch: _joint()(cfg, ch),
    "gosc_joint_nb": lambda cfg, ch: _joint()(cfg, ch, bundle=False, name="gosc_joint_nb"),
    "genie": lambda cfg, ch: Genie(cfg, ch),
}
