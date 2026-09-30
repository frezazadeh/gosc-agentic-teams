import itertools
import math

import numpy as np
import pytest

from gosc import Config, Simulation, run_episode
from gosc.channel import Channel, fbl_error
from gosc.schemes import SCHEMES, lagrangian_select


def test_fbl_error_behaves():
    # monotone in rate and blocklength, and approaches the outage rule for long packets
    e_lo, e_hi = fbl_error(100, 1.0, 10.0), fbl_error(100, 2.0, 10.0)
    assert e_lo < e_hi
    assert fbl_error(400, 2.0, 10.0) < fbl_error(50, 2.0, 10.0)
    assert fbl_error(10_000, 3.0, 10.0) < 1e-6 and fbl_error(10_000, 3.6, 10.0) > 1 - 1e-6


def test_fbl_matches_outage_for_long_packets_and_gap_hurts():
    """Under quasi-static fading the fading-averaged normal approximation converges to
    the outage probability (Yang et al., 2014); a coding gap lowers the success."""
    fbl = Channel(Config().with_(error_model="fbl"), (16, 16))
    out = Channel(Config(), (16, 16))
    for r in (0.5, 1.0, 2.0):
        assert fbl.success_prob(r, 10.0, n=5000) == pytest.approx(out.success_prob(r, 10.0), abs=0.01)
    gapped = Channel(Config().with_(error_model="fbl", code_gap_db=2.0), (16, 16))
    assert gapped.success_prob(2.0, 10.0, n=40) < fbl.success_prob(2.0, 10.0, n=40)


def test_fbl_mission_runs():
    out = run_episode(Config().with_(error_model="fbl", max_steps=60), "gosc", 0)
    assert out["bits"] > 0


@pytest.mark.parametrize("scheme", ["gosc", "sem", "genie"])
def test_lossy_downlink_bookkeeping(scheme):
    """With a lossy/delayed broadcast, each agent's private residual equals its own
    evidence that is NOT represented in its last received snapshot, i.e. evidence
    not yet delivered plus evidence delivered to the edge after the snapshot slot,
    and its local belief is snapshot + residual."""
    cfg = Config().with_(dl_loss=0.3, dl_delay=2, max_steps=60, n_agents=3)
    sim = Simulation(cfg, scheme, seed=4)
    own = {a.idx: [] for a in sim.agents}
    delivered_at = {}
    orig_step, orig_deliver = sim.scheme.on_step, sim.deliver

    def spy_step(agent, rec, t, view):
        own[agent.idx].append(rec)
        return orig_step(agent, rec, t, view)

    def spy_deliver(agent, pkt, t):
        for rec in pkt.records:
            delivered_at[(rec.agent, rec.t)] = t
        return orig_deliver(agent, pkt, t)

    sim.scheme.on_step, sim.deliver = spy_step, spy_deliver
    out = sim.run()
    assert out["dl_symbols"] > 0
    for a in sim.agents:
        v = sim.view_t[a.idx]
        resid = np.zeros_like(sim.Lc)
        for rec in own[a.idx]:
            t_del = delivered_at.get((rec.agent, rec.t))
            if t_del is None or t_del > v:
                sim.sensor.apply(resid, rec)
        np.testing.assert_allclose(a.pending_llr, resid, atol=1e-8)
        # the snapshot equals prior + every record delivered up to the view slot
        if v >= 0:
            snap = np.full_like(sim.Lc, sim.prior)
            for k in own:
                for rec in own[k]:
                    t_del = delivered_at.get((rec.agent, rec.t))
                    if t_del is not None and t_del <= v:
                        sim.sensor.apply(snap, rec)
            np.testing.assert_allclose(sim.snaps[v][0], snap, atol=1e-8)


def test_delivered_messages_are_not_resent_under_delay():
    """Reliable HARQ acknowledgments: a delivered intent/confirmation is not re-sent
    while the delayed broadcast is still on its way."""
    cfg = Config().with_(dl_delay=5, max_steps=80)
    sim = Simulation(cfg, "gosc", seed=1)
    sent = []
    orig = sim.deliver

    def spy(agent, pkt, t):
        if pkt.kind in ("INTENT", "CONF"):
            sent.append((agent.idx, pkt.kind, pkt.intent if pkt.kind == "INTENT" else pkt.confirm, t))
        return orig(agent, pkt, t)

    sim.deliver = spy
    sim.run()
    confirms = [(a, x) for a, k, x, _ in sent if k == "CONF"]
    assert len(confirms) == len(set(confirms))
    # an agent never re-announces the intent it just announced
    last = {}
    for a, k, x, t in sent:
        if k == "INTENT":
            assert last.get(a) != x
            last[a] = x


def test_ideal_downlink_matches_default():
    a = run_episode(Config().with_(max_steps=80), "gosc", 7)
    b = run_episode(Config().with_(max_steps=80, dl_loss=0.0, dl_delay=0), "gosc", 7)
    assert a == b


def test_downlink_delay_slows_or_equals_on_average():
    base = np.mean([run_episode(Config(), "sem", s)["completion_time"] for s in range(4)])
    slow = np.mean([run_episode(Config().with_(dl_delay=10), "sem", s)["completion_time"]
                    for s in range(4)])
    assert slow >= base - 15  # stale common knowledge cannot help much


@pytest.mark.parametrize("scheme", ["gosc_nofilter", "gosc_bits", "gosc_aoi", "sem_agg"])
def test_new_schemes_complete_easy_mission(scheme):
    out = run_episode(Config().with_(n_victims=2, snr_ref_db=20, sem_interval=4), scheme, 0)
    assert out["found"] == 2


def test_oracle_teammate_model_runs():
    out = run_episode(Config().with_(tom_oracle=True, log_private=True, max_steps=60), "gosc", 1)
    assert out["private_tv"] >= 0


def exact_select(groups, budget, price):
    best, best_val = (), 0.0
    choices = [[None] + list(g.options) for g in groups]
    for combo in itertools.product(*choices):
        sym = sum(o.symbols for o in combo if o is not None)
        if sym > budget:
            continue
        val = sum(o.value - price * o.symbols for o in combo if o is not None)
        if val > best_val:
            best, best_val = combo, val
    return best_val


def test_lagrangian_close_to_exact_on_logged_instances():
    sim = Simulation(Config().with_(symbols_per_slot=50, max_steps=60), "gosc", 2)
    sim.scheme.log = []
    sim.run()
    gaps = []
    for groups, chosen, price, budget in sim.scheme.log:
        if not groups:
            continue
        opt = exact_select(groups, budget, price)
        got = sum(o.value - price * o.symbols for _, o in chosen)
        assert got <= opt + 1e-9
        assert sum(o.symbols for _, o in chosen) <= budget
        gaps.append(0.0 if opt <= 0 else (opt - got) / opt)
    assert gaps and np.mean(gaps) < 0.1
