"""Experiments added in the revision (valuation rules at matched resources,
separated ablations, teammate-model approximation, downlink impairments,
finite-blocklength errors).  Results go to results/rev_<name>.json.

    python -m experiments.run_revision                 # everything
    python -m experiments.run_revision --only ablation --seeds 10
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

from gosc import Config, run_episode

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

ETAS = {
    "gosc": (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    "gosc_tv": (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "gosc_bits": (0.0, 0.1, 0.3, 1.0, 2.0, 3.0, 4.0),
    "gosc_aoi": (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
}


def experiments():
    b = Config()
    ex = {}
    rows = []
    for W in (100, 50):
        for scheme, etas in ETAS.items():
            for e in etas:
                rows.append((scheme, f"W{W}/eta{e}", b.with_(symbols_per_slot=W, energy_price=e)))
        for k in (1, 2, 4, 8, 16):
            rows.append(("sem_agg", f"W{W}/k{k}", b.with_(symbols_per_slot=W, sem_interval=k)))
        rows.append(("sem", f"W{W}/k1", b.with_(symbols_per_slot=W)))
    ex["frontier"] = rows
    ex["ablation"] = [(s, f"W{W}", b.with_(symbols_per_slot=W))
                      for W in (100, 50)
                      for s in ("gosc", "gosc_nofilter", "gosc_bits", "gosc_novoi", "gosc_nouep")]
    ex["private"] = [("gosc", f"{name}/{'oracle' if o else 'shared'}",
                      cfg.with_(tom_oracle=o, log_private=True))
                     for name, cfg in (("default", b), ("W50", b.with_(symbols_per_slot=50)),
                                       ("snr-5", b.with_(snr_ref_db=-5)))
                     for o in (False, True)]
    dl = []
    for s in ("sem", "gosc", "genie"):
        for p in (0.0, 0.1, 0.3, 0.5):
            dl.append((s, f"loss{p}", b.with_(dl_loss=p)))
        for d in (1, 3, 5, 10):
            dl.append((s, f"delay{d}", b.with_(dl_delay=d)))
    ex["downlink"] = dl
    ex["fbl"] = [(s, f"gap{g}/snr{v}", b.with_(error_model="fbl", code_gap_db=g, snr_ref_db=v))
                 for g in (0.0, 2.0) for v in (-10, -5, 0, 5, 10, 15, 20)
                 for s in ("raw", "nl", "sem", "gosc")]
    where = []
    for scheme in ("gosc", "gosc_bits", "gosc_aoi"):
        for e in ETAS[scheme]:
            where.append((scheme, f"K10/eta{e}", b.with_(n_agents=10, energy_price=e)))
            where.append((scheme, f"clustered/eta{e}", b.with_(victim_layout="clustered", energy_price=e)))
    for k in (1, 2, 4, 8):
        where.append(("sem_agg", f"K10/k{k}", b.with_(n_agents=10, sem_interval=k)))
        where.append(("sem_agg", f"clustered/k{k}", b.with_(victim_layout="clustered", sem_interval=k)))
    ex["where_tom"] = where
    rb = b.with_(task="rescue", deadline_min=60, deadline_max=180)  # frozen after calibration
    resc = []
    for name, cfg in (("W100", rb), ("W50", rb.with_(symbols_per_slot=50)),
                      ("snr-5", rb.with_(snr_ref_db=-5))):
        for k in (1, 2, 4, 8, 16):
            resc.append(("sem_ra", f"{name}/k{k}", cfg.with_(sem_interval=k)))
        for e in (0.0, 3e-6, 1e-5, 3e-5, 1e-4):
            resc.append(("gosc", f"{name}/eta{e}", cfg.with_(energy_price=e)))
        for e in (0.0, 0.1, 0.3, 1.0, 2.0):
            resc.append(("gosc_bits", f"{name}/eta{e}", cfg.with_(energy_price=e)))
        for e in (0.0, 0.01, 0.03, 0.1, 0.3):
            resc.append(("gosc_aoi", f"{name}/eta{e}", cfg.with_(energy_price=e)))
        for sch in ("sem", "raw", "nl", "report", "genie"):
            resc.append((sch, f"{name}/ref", cfg))
    ex["rescue"] = resc
    rf = []
    for name, cfg in (("W100", b), ("W50", b.with_(symbols_per_slot=50)), ("K10", b.with_(n_agents=10)),
                      ("clustered", b.with_(victim_layout="clustered")),
                      ("rescue-W100", rb), ("rescue-W50", rb.with_(symbols_per_slot=50)),
                      ("rescue-snr-5", rb.with_(snr_ref_db=-5))):
        for k in (1, 2, 4, 8, 16):
            rf.append(("sem_raf", f"{name}/k{k}", cfg.with_(sem_interval=k)))
    ex["filtered_periodic"] = rf
    sup = []
    for name, cfg in (("W100", b), ("W50", b.with_(symbols_per_slot=50)), ("K10", b.with_(n_agents=10)),
                      ("clustered", b.with_(victim_layout="clustered")),
                      ("rescue-W100", rb), ("rescue-W50", rb.with_(symbols_per_slot=50)),
                      ("rescue-snr-5", rb.with_(snr_ref_db=-5))):
        for k in (1, 2, 4, 8):
            for th in (0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0):
                sup.append(("sem_ras", f"{name}/k{k}/th{th}", cfg.with_(sem_interval=k, filter_eps=th)))
    ex["suppress_periodic"] = sup
    ex["genie_ref"] = [("genie", "K10", b.with_(n_agents=10)),
                       ("genie", "clustered", b.with_(victim_layout="clustered")),
                       ("genie", "W100", b)]
    ex["rate_aware"] = [("sem_ra", f"{name}/k{k}", cfg.with_(sem_interval=k))
                        for name, cfg in (("W100", b), ("W50", b.with_(symbols_per_slot=50)),
                                          ("K10", b.with_(n_agents=10)),
                                          ("clustered", b.with_(victim_layout="clustered")))
                        for k in (1, 2, 4, 8)]
    return ex


def _job(args):
    exp, scheme, param, cfg, seed = args
    out = run_episode(cfg, scheme, seed)
    out.pop("curve")
    return {"scheme": scheme, "param": param, "seed": seed, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ex = experiments()
    for name in args.only or list(ex):
        jobs = [(name, s, p, cfg, seed) for s, p, cfg in ex[name] for seed in range(args.seeds)]
        t0 = time.time()
        with Pool(args.workers) as pool:
            rows = list(pool.imap_unordered(_job, jobs, chunksize=2))
        with open(os.path.join(OUT, f"rev_{name}.json"), "w") as f:
            json.dump({"rows": rows}, f)
        print(f"{name}: {len(rows)} episodes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
