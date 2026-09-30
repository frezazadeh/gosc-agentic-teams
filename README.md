# GOSC: Resource-Efficient Semantic Communication for Heterogeneous Agentic Teams

Simulator, configurations, per-mission results and cached LLM responses for the paper

> F. Rezazadeh, H. Chergui, L. Liu, and M. Debbah, "Resource-Efficient Semantic
> Communication for Heterogeneous Agentic Teams," submitted for publication.

Teams of autonomous agents, including large language model (LLM) agents, must
coordinate over scarce and unreliable wireless links. **GOSC** (goal-oriented semantic
communication) is a closed-loop co-design that jointly decides *what* each agent sends,
*when* it sends it, and *how reliably* it is transmitted, based on each message's value to
the team task; an edge broadcast of the team's common knowledge closes the loop. The use
case is multi-UAV search and rescue over a block-fading uplink.

Every number, table and figure in the paper is generated from the per-mission results in
this repository, and every mission can be re-simulated from its seed.

## Contents

| Path | What it contains |
|---|---|
| `gosc/` | The simulator: world and sensing, beliefs, block-fading channel (outage and finite-blocklength models), message formats, planner and LLM agents, value functions, GOSC scheduler and all baselines |
| `gosc_ext/` | Extensions used in the revision (imperfect values, per-packet overhead with uplink + downlink cost accounting, shared network-wide uplink, scheduler-structure analysis). It subclasses `gosc/` and reproduces it exactly with default settings |
| `experiments/` | Campaigns, statistics, figures and tables of the main evaluation |
| `tests/`, `tests_ext/` | Unit and regression tests |
| `results/` | Per-mission results of the main evaluation (JSON) and cached LLM responses (`results/llm_cache/`) |
| `results_r2/` | Per-mission results of the revision experiments (large files gzip-compressed) and cached LLM responses (`results_r2/llm_cache/`) |
| `scripts/unpack_results.sh` | Decompresses `results_r2/*.jsonl.gz` |

## Installation

Python 3.9 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/unpack_results.sh                          # decompress the revision results
.venv/bin/python -m pytest -q tests tests_ext         # 85 tests
```

Run one mission:

```bash
.venv/bin/python -m gosc.simulator --scheme gosc --seed 0
```

## Reproducing the paper

The analysis scripts read the stored results, so every table and figure can be rebuilt in
seconds without re-simulating; they are written to `figures/`.

```bash
.venv/bin/python -m experiments.plot                  # main evaluation
.venv/bin/python -m experiments.plot_revision         # value functions, rescue task, validation
PYTHONPATH=. .venv/bin/python -m gosc_ext.report      # revision experiments (Tables VII, IX-XII)
```

To re-simulate from scratch (times on an 8-core Apple M1):

```bash
# main evaluation
.venv/bin/python -m experiments.run_all --seeds 100                 # ~30 min
.venv/bin/python -m experiments.run_revision --seeds 100            # ~90 min
.venv/bin/python -m experiments.validate_headline                   # fresh seeds 3000-3099
.venv/bin/python -m experiments.analyze_scheduler --seeds 30
.venv/bin/python -m experiments.run_joint --seeds 100
.venv/bin/python -m experiments.diagnose_value && .venv/bin/python -m experiments.analyze_fidelity

# revision experiments
PYTHONPATH=. .venv/bin/python -m gosc_ext.analyze_theory --seeds 30                  # Theorems 1-2, Proposition 2
PYTHONPATH=. .venv/bin/python -m gosc_ext.run --only noise shared overhead           # ~6.5 h on 7 workers
PYTHONPATH=. .venv/bin/python -m gosc_ext.validate --only noise shared overhead      # fresh seeds
PYTHONPATH=. .venv/bin/python -m gosc_ext.run_tv_where                               # belief divergence, K=10 / clustered
```

| Paper item | Produced by | Results |
|---|---|---|
| Main result, fresh-seed validation (Table IV) and frozen points (Table XIII) | `experiments/validate_headline.py`, `experiments/plot_revision.py` | `results/rev_validation.json` |
| Representations (Table V), SNR sweep (Figure 4) | `experiments/run_all.py`, `experiments/plot.py` | `results/default.json`, `results/snr.json` |
| Rescue frontiers (Figure 2), value functions (Table VI) | `experiments/run_revision.py`, `gosc_ext/run_tv_where.py` | `results/rev_*.json`, `results_r2/where_tom_tv.jsonl` |
| Scheduler structure (Section IV) | `gosc_ext/analyze_theory.py` | `results_r2/theory.json` |
| Value accuracy (Table VII, Figure 3) | `gosc_ext/run.py`, `gosc_ext/validate.py` | `results_r2/noise.jsonl.gz`, `results_r2/validate_noise.json` |
| Packet overhead and total cost (Tables IX, X) | same | `results_r2/overhead.jsonl.gz`, `results_r2/validate_overhead.json` |
| Shared uplink (Table XI, Figure 5) | same | `results_r2/shared.jsonl.gz`, `results_r2/validate_shared.json` |
| LLM teams (Table XII) | `experiments/run_llm.py`, `gosc_ext/run_llm.py` | `results/llm_mixed.jsonl`, `results_r2/llm_r2.jsonl` |

## LLM agents

Agents 0, 2 and 4 can be driven by an open-source LLM served locally by
[Ollama](https://ollama.com) (Qwen2.5-3B, Llama-3.2-3B and Qwen2.5-7B in the paper). Every
LLM decision is cached, keyed by model and prompt, so the LLM missions replay exactly
without Ollama. To query the models again:

```bash
ollama serve &
ollama pull qwen2.5:3b && ollama pull llama3.2:3b && ollama pull qwen2.5:7b
.venv/bin/python -m experiments.run_llm --seeds 50 --teams planner mixed
PYTHONPATH=. .venv/bin/python -m gosc_ext.run_llm --seeds 50 --teams mixed_llama
PYTHONPATH=. .venv/bin/python -m gosc_ext.run_llm --seeds 30 --teams mixed_qwen7b
```

The revision runner aborts if any LLM call fails, so an unreachable server can never
silently turn LLM agents into planners.

## Reproducibility notes

- Hyper-parameters were fixed on calibration seeds 1000-1079; all reported results use the
  disjoint evaluation seeds 0-99 and, for frozen operating points, fresh seeds 3000-3099.
- For a given seed, every scheme sees the same survivors and launch positions (common
  random numbers).
- With default settings, `gosc_ext` reproduces the `gosc` simulator exactly; this is
  checked by `tests_ext/` (for example, seed 0 with GOSC: 122 slots, 13,304 channel uses).

## License

MIT; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper (the reference will be updated on
publication):

```bibtex
@article{rezazadeh2026gosc,
  author  = {F. Rezazadeh and H. Chergui and L. Liu and M. Debbah},
  title   = {Resource-Efficient Semantic Communication for Heterogeneous Agentic Teams},
  journal = {submitted for publication},
  year    = {2026}
}
```
