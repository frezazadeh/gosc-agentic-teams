"""Exact joint message, grouping and rate selection (set-valued scheduling).

Independent valuation (GOSC) scores each message by its own value.  Here the
scheduler scores the *set of messages the receiver obtains*: for an intention I
and an evidence prefix E it uses V({E}), V({I}) and the joint value V({E, I})
from the same theory-of-mind team model, so interactions between the two are
taken into account.  Reception is modelled exactly for block fading with
statistical CSI: packets of one agent in one slot see the same fading, so at
rates R1 <= R2 the packet at R2 is decoded only if the packet at R1 is, i.e.
P(both) = P_s(R2), P(only R1) = P_s(R1) - P_s(R2), P(only R2) = 0.  Intention and
evidence may also be grouped into a single packet (one header, both-or-none).
Confirmations are valued additively.  The per-slot problem is solved exactly by
enumeration.
"""
import itertools
import math

import numpy as np

from .messages import HEADER_BITS, Packet, confirm_bits, evidence_sizes, intent_bits
from .schemes import GOSC, TxResult
from .voi import TeamModel


class GOSCJoint(GOSC):
    name = "gosc_joint"

    def __init__(self, cfg, channel, bundle=True, name=None):
        super().__init__(cfg, channel, name=name)
        self.bundle = bundle
        self.stats = {"slots": 0, "separate": 0, "bundle": 0, "e_only": 0, "i_only": 0}

    def _ps(self, rate, snr_mean, n):
        return float(self.channel.success_prob(rate, snr_mean, n=n))

    def transmit(self, agent, t, snr_mean, snr_inst, sim):
        c = self.cfg
        W, price = c.symbols_per_slot, self.price
        rates = sorted(c.rates)
        team = TeamModel(sim, agent, c.tom_temperature)

        # -- confirmations: additive value, one group each ---------------------
        acked_c = getattr(agent, "acked_confirms", set())
        conf_groups = []
        for cell in sorted(agent.confirmed_pending - acked_c):
            b = confirm_bits(c)
            opts = [None]
            for r in rates:
                s = math.ceil(b / r)
                if s <= W:
                    opts.append(((("C", r, b, s, cell),), c.confirm_value * self._ps(r, snr_mean, s), s))
            conf_groups.append(opts)

        # -- intention / evidence: set-valued options -------------------------
        acked_i = getattr(agent, "acked_intent", agent.shared_intent)
        has_i = (agent.target is not None and agent.target != agent.shared_intent
                 and agent.target != acked_i)
        recs = agent.pending_records
        sizes = evidence_sizes(recs, c) if recs else []

        def fit(cap):
            return int(np.searchsorted(sizes, cap, side="right")) if sizes else 0

        sep_e = [(r, fit(int(W * r))) for r in rates if fit(int(W * r)) > 0]
        i_extra = intent_bits(c) - HEADER_BITS
        bund = ([(r, fit(int(W * r) - i_extra)) for r in rates if fit(int(W * r) - i_extra) > 0]
                if has_i and self.bundle else [])
        need = sorted({n for _, n in sep_e} | {n for _, n in bund})
        v_e, v_ei = {}, {}
        if need:
            delta, i, deltas = np.zeros_like(sim.Lc), 0, []
            for n in need:
                while i < n:
                    sim.sensor.apply(delta, recs[i])
                    i += 1
                deltas.append(delta.copy())
            for n, v, d in zip(need, team.evidence_voi(deltas, [recs[n - 1].pos for n in need]), deltas):
                v_e[n] = v
                if has_i:
                    v_ei[n] = team.joint_voi(d, recs[n - 1].pos, agent.target)
        v_i = team.intent_voi(agent.target) if has_i else 0.0

        ei = [None]
        e_opts = []
        for r, n in sep_e:
            b = sizes[n - 1]
            s = math.ceil(b / r)
            e_opts.append(("E", r, b, s, n))
            ei.append(((e_opts[-1],), v_e[n] * self._ps(r, snr_mean, s), s))
        i_opts = []
        if has_i:
            b = intent_bits(c)
            for r in rates:
                s = math.ceil(b / r)
                if s <= W:
                    i_opts.append(("I", r, b, s, 0))
                    ei.append(((i_opts[-1],), v_i * self._ps(r, snr_mean, s), s))
            for eo, io in itertools.product(e_opts, i_opts):
                if eo[3] + io[3] > W:
                    continue
                pe, pi = self._ps(eo[1], snr_mean, eo[3]), self._ps(io[1], snr_mean, io[3])
                both = min(pe, pi)  # nested decoding under a common fading block
                val = both * v_ei[eo[4]] + (pe - both) * v_e[eo[4]] + (pi - both) * v_i
                ei.append(((eo, io), val, eo[3] + io[3]))
            for r, n in bund:
                b = sizes[n - 1] + i_extra
                s = math.ceil(b / r)
                if s <= W:
                    ei.append(((("B", r, b, s, n),), v_ei[n] * self._ps(r, snr_mean, s), s))

        # -- exact selection ---------------------------------------------------
        best, best_u = (), 0.0
        for combo in itertools.product(ei, *conf_groups):
            sym = sum(o[2] for o in combo if o is not None)
            if sym > W:
                continue
            u = sum(o[1] - price * o[2] for o in combo if o is not None)
            if u > best_u + 1e-15:
                best, best_u = combo, u

        res = TxResult([])
        self.stats["slots"] += 1
        for opt in best:
            if opt is None:
                continue
            kinds = [p[0] for p in opt[0]]
            if kinds == ["B"]:
                self.stats["bundle"] += 1
            elif set(kinds) == {"E", "I"}:
                self.stats["separate"] += 1
            elif kinds == ["E"]:
                self.stats["e_only"] += 1
            elif kinds == ["I"]:
                self.stats["i_only"] += 1
            for kind, r, b, s, x in opt[0]:
                res.symbols += s
                res.bits += b
                if not self.channel.decodes(r, snr_inst, n=s):
                    continue
                if kind == "C":
                    agent.acked_confirms = getattr(agent, "acked_confirms", set()) | {x}
                    res.delivered.append(Packet("CONF", agent.idx, b, t, confirm=x))
                if kind in ("I", "B"):
                    agent.acked_intent = agent.target
                    res.delivered.append(Packet("INTENT", agent.idx, intent_bits(c), t,
                                                intent=agent.target))
                if kind in ("E", "B"):
                    n = x
                    rs = agent.pending_records[:n]
                    del agent.pending_records[:n]
                    for rec in rs:
                        sim.sensor.apply(agent.queued_llr, rec, sign=-1.0)
                    res.delivered.append(Packet("EVID", agent.idx, sizes[n - 1], t, records=rs))
        return res
