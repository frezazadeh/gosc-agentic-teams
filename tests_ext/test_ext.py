"""The revision-2 extensions must reproduce the journal simulator exactly with the
default ExtConfig, and the new options must behave as specified."""
import numpy as np
import pytest

from gosc import Config, run_episode
from gosc_ext import ExtConfig, run_ext
from gosc_ext.sim import hull_size, waterfill
from gosc.schemes import Option

KEYS = ("completion_time", "symbols", "bits", "delivered_bits", "dl_symbols", "found")
SCHEMES = ("gosc", "gosc_bits", "sem", "sem_ra", "sem_raf", "sem_ras", "raw", "nl", "report", "genie")


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reproduces_journal(scheme, seed):
    kw = dict(sem_interval=4, filter_eps=0.1) if scheme in ("sem_ra", "sem_raf", "sem_ras") else {}
    a = run_episode(Config().with_(**kw), scheme, seed)
    b = run_ext(ExtConfig().with_(**kw), scheme, seed)
    assert {k: a[k] for k in KEYS} == {k: b[k] for k in KEYS}


@pytest.mark.parametrize("scheme", ("gosc", "sem_ra"))
def test_reproduces_journal_rescue_w50(scheme):
    kw = dict(task="rescue", deadline_min=60, deadline_max=180, symbols_per_slot=50)
    a = run_episode(Config().with_(**kw), scheme, 3)
    b = run_ext(ExtConfig().with_(**kw), scheme, 3)
    assert a["saved_fraction"] == b["saved_fraction"] and a["symbols"] == b["symbols"]


def test_journal_regression_episode():
    o = run_ext(ExtConfig(), "gosc", 0)
    assert (o["completion_time"], o["symbols"], o["bits"]) == (122, 13304, 17186)


def test_overhead_is_charged():
    base = run_ext(ExtConfig(), "sem_ra", 0)
    oh = run_ext(ExtConfig(overhead_bits=32), "sem_ra", 0)
    assert oh["ul_overhead_bits"] == 32 * oh["ul_tx"]
    g = run_ext(ExtConfig(overhead_bits=32), "gosc", 0)
    assert g["bits"] >= 32 * g["ul_tx"] and g["ul_tx"] > 0
    assert base["ul_overhead_bits"] == 0


def test_zero_noise_equals_exact():
    a = run_ext(ExtConfig(), "gosc", 1)
    b = run_ext(ExtConfig(value_mode="lognormal", value_sigma=0.0), "gosc", 1)
    assert a["completion_time"] == b["completion_time"] and a["symbols"] == b["symbols"]


def test_noise_changes_decisions_and_is_logged():
    o = run_ext(ExtConfig(value_mode="lognormal", value_sigma=2.0), "gosc", 1)
    assert "value_pairs" in o and len(o["value_pairs"]) > 10
    p = run_ext(ExtConfig(value_mode="permuted"), "gosc", 1)
    assert p["completed"]


@pytest.mark.parametrize("scheme", ("gosc", "sem_ra"))
def test_shared_budget_respected(scheme):
    from gosc_ext.sim import ExtSimulation
    cfg = ExtConfig(shared_budget=150, symbols_per_slot=150)
    sim = ExtSimulation(cfg, scheme, 0)
    sim._start()
    last = 0
    for t in range(cfg.max_steps):
        sim.phase_decide(t)
        done = sim.phase_communicate(t)
        assert sim.symbols - last <= 150
        last = sim.symbols
        if done:
            break
    r = sim.result()
    assert r["ul_ctrl_symbols"] > 0 and r["dl_ctrl_symbols"] > 0


def test_waterfill():
    assert waterfill([10, 0, 50, 50], 60, [1, 1, 1, 1]) == [10, 0, 25, 25]
    assert sum(waterfill([100, 100, 100], 100, [1, 1, 1])) == 100
    g = waterfill([100, 100], 100, [60, 1])
    assert g == [0, 100]


def test_hull_size():
    opts = [Option(1, 10, 10, 1.0), Option(2, 10, 5, 0.9), Option(4, 10, 3, 0.1)]
    assert hull_size(opts, 0.0) == 2
    assert hull_size(opts, 1.0) == 0
