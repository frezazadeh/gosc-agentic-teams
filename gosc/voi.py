"""Decision-theoretic value of information via a theory-of-mind team model.

The sender models every teammate as a Boltzmann-rational planner that acts on the
common (edge-broadcast) belief.  This is an approximation: each teammate also holds
private, not yet delivered evidence that the sender cannot observe.  With
Config.tom_oracle (analysis only) the model instead uses the teammates' true
private evidence and positions, which quantifies the cost of the approximation.  A message is worth the increase in the teammates'
expected search utility, evaluated under the sender's own, better-informed belief:

    VoI(m) = sum_j E_{c ~ pi_j(. | K + m)}[U_j(c)] - E_{c ~ pi_j(. | K)}[U_j(c)]

where K is the common knowledge, pi_j the softmax policy of teammate j and U_j its
utility (survivor mass in the footprint, discounted by travel distance and by
proximity to higher-priority teammates) computed with the sender's local belief.
"""
import numpy as np

from .belief import box_sum, sigmoid


def repulsion(points, X, Y, sigma):
    f = np.ones(X.shape)
    two_s2 = 2.0 * sigma ** 2
    for ox, oy in points:
        f = f * (1.0 - np.exp(-((X - ox) ** 2 + (Y - oy) ** 2) / two_s2))
    return f


class TeamModel:
    def __init__(self, sim, sender, temperature: float):
        cfg = sim.cfg
        self.cfg, self.sim, self.sender, self.temp = cfg, sim, sender, temperature
        self.X, self.Y = sender._X, sender._Y
        n = cfg.grid
        self.live = ~sim.declared.reshape(sim.Lc.shape)
        p_true = sigmoid(sim.Lc + sender.pending_llr) * self.live
        if sender.confirmed_pending:
            p_true.reshape(-1)[list(sender.confirmed_pending)] = 0.0
        self.mass_true = box_sum(p_true, cfg.sense_radius)
        self.mass_shared = self.mass_of(sim.Lc)
        k = sender.idx
        self.true_k = divmod(sender.target, n) if sender.target is not None else sender.pos
        self.mates = [j for j in range(cfg.n_agents) if j != k and j in sim.team_pos]
        self.oracle = cfg.tom_oracle
        self.dist, self.rep_known, self.rep_true, self.others = {}, {}, {}, {}
        self.priv, self.mass_mate = {}, {}
        for j in self.mates:
            x, y = sim.agents[j].pos if self.oracle else sim.team_pos[j]
            if self.oracle:
                mate = sim.agents[j]
                self.priv[j] = mate.pending_llr
                self.mass_mate[j] = self.mass_of(sim.Lc + mate.pending_llr, mate.confirmed_pending)
            self.dist[j] = 1.0 / (1.0 + cfg.distance_discount * (np.abs(self.X - x) + np.abs(self.Y - y)))
            pts = self._points(j)
            self.others[j] = repulsion([p for i, p in pts if i != k], self.X, self.Y, cfg.repulsion_sigma)
            self.rep_known[j] = self.others[j] * self._rep_point(dict(pts).get(k))
            self.rep_true[j] = self.others[j] * self._rep_point(self.true_k)
        self.u_true = {j: self.mass_true * self.dist[j] * self.rep_true[j] for j in self.mates}

    def _rep_point(self, pt):
        if pt is None:
            return 1.0
        return repulsion([pt], self.X, self.Y, self.cfg.repulsion_sigma)

    def _points(self, j):
        """Points teammate j deconflicts against (mirrors Simulation.teammates)."""
        sim, n = self.sim, self.cfg.grid
        pts = []
        for i in range(self.cfg.n_agents):
            if i == j:
                continue
            if i < j and i in sim.team_intent:
                pts.append((i, divmod(sim.team_intent[i], n)))
            elif i in sim.team_pos:
                pts.append((i, sim.team_pos[i]))
        return pts

    def _uses_intent_of_sender(self, j):
        k = self.sender.idx
        return k < j and k in self.sim.team_intent

    def mass_of(self, L, exclude=()):
        p = sigmoid(L) * self.live
        if exclude:
            p.reshape(-1)[list(exclude)] = 0.0
        return box_sum(p, self.cfg.sense_radius)

    def _belief_mass(self, j, delta=None):
        """Survivor mass teammate j acts on, optionally after receiving `delta`."""
        if not self.oracle:
            return self.mass_shared if delta is None else self.mass_of(self.sim.Lc + delta)
        if delta is None:
            return self.mass_mate[j]
        mate = self.sim.agents[j]
        return self.mass_of(self.sim.Lc + self.priv[j] + delta, mate.confirmed_pending)

    def _expected_utility(self, mass_belief, j, rep_belief):
        s = mass_belief * self.dist[j] * rep_belief
        m = s.max()
        if m <= 0:
            return 0.0
        w = np.exp((s - m) / (self.temp * m))
        return float((w * self.u_true[j]).sum() / w.sum())

    def evidence_voi(self, delta_llrs, new_positions):
        """VoI of candidate evidence updates.  Delivering evidence also reveals the
        sender's latest position to teammates that deconflict against it."""
        base = {j: self._expected_utility(self._belief_mass(j), j, self.rep_known[j])
                for j in self.mates}
        out = []
        for d, pos in zip(delta_llrs, new_positions):
            mass = None if self.oracle else self.mass_of(self.sim.Lc + d)
            v = 0.0
            for j in self.mates:
                rep = (self.rep_known[j] if self._uses_intent_of_sender(j)
                       else self.others[j] * self._rep_point(pos))
                m = self._belief_mass(j, d) if self.oracle else mass
                v += self._expected_utility(m, j, rep) - base[j]
            out.append(v)
        return out

    def intent_voi(self, target: int):
        """VoI of announcing a new intention to the lower-priority teammates."""
        k = self.sender.idx
        tgt = divmod(target, self.cfg.grid)
        v = 0.0
        for j in self.mates:
            if j < k:
                continue  # higher-priority agents do not deconflict against k's intent
            rep_new = self.others[j] * self._rep_point(tgt)
            m = self._belief_mass(j)
            v += (self._expected_utility(m, j, rep_new)
                  - self._expected_utility(m, j, self.rep_known[j]))
        return v

    def joint_voi(self, delta, pos, target: int):
        """VoI of delivering an evidence update and a new intention together (used to
        measure how far the per-message values are from additive)."""
        k = self.sender.idx
        tgt = divmod(target, self.cfg.grid)
        v = 0.0
        for j in self.mates:
            base = self._expected_utility(self._belief_mass(j), j, self.rep_known[j])
            rep = self.others[j] * self._rep_point(tgt if j > k else pos)
            v += self._expected_utility(self._belief_mass(j, delta), j, rep) - base
        return v
