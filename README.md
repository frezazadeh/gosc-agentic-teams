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
| `gosc/` | The simulator: world and sensing, beliefs, block-fading channel (outage and finite-blocklength models), message formats, planner and LLM agents, value functions, GOSC scheduler and all baselines; `gosc/realistic.py` adds imperfect values, per-packet overhead with uplink + downlink cost accounting, and a shared network-wide uplink |
| `experiments/` | Campaigns, statistics, figures and tables of the main evaluation |
| `tests/` | Unit and regression tests |
| `results/` | Per-mission results of every experiment (large files gzip-compressed) and cached LLM responses (`results/llm_cache/`) |
| `scripts/unpack_results.sh` | Decompresses `results/*.jsonl.gz` |

## Installation

Python 3.9 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/unpack_results.sh                          # decompress the large result files
.venv/bin/python -m pytest -q tests                   # 85 tests
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
.venv/bin/python -m experiments.plot_realistic        # Tables VII, IX-XII, Figures 3 and 5
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

# imperfect values, packet overhead, shared uplink, scheduler structure
.venv/bin/python -m experiments.analyze_theory --seeds 30                    # Theorems 1-2, Proposition 2
.venv/bin/python -m experiments.run_realistic --only noise shared overhead     # ~6.5 h on 7 workers
.venv/bin/python -m experiments.validate_realistic --only noise shared overhead # fresh seeds
.venv/bin/python -m experiments.run_belief_divergence                         # belief divergence, K=10 / clustered
```

| Paper item | Produced by | Results |
|---|---|---|
| Main result, fresh-seed validation (Table IV) and frozen points (Table XIII) | `experiments/validate_headline.py`, `experiments/plot_revision.py` | `results/rev_validation.json` |
| Representations (Table V), SNR sweep (Figure 4) | `experiments/run_all.py`, `experiments/plot.py` | `results/default.json`, `results/snr.json` |
| Rescue frontiers (Figure 2), value functions (Table VI) | `experiments/run_revision.py`, `experiments/run_belief_divergence.py` | `results/rev_*.json`, `results/belief_divergence.jsonl` |
| Scheduler structure (Section IV) | `experiments/analyze_theory.py` | `results/theory.json` |
| Value accuracy (Table VII, Figure 3) | `experiments/run_realistic.py`, `experiments/validate_realistic.py` | `results/noise.jsonl.gz`, `results/validate_noise.json` |
| Packet overhead and total cost (Tables IX, X) | same | `results/overhead.jsonl.gz`, `results/validate_overhead.json` |
| Shared uplink (Table XI, Figure 5) | same | `results/shared.jsonl.gz`, `results/validate_shared.json` |
| LLM teams (Table XII) | `experiments/run_llm.py`, `experiments/run_llm_extended.py` | `results/llm_mixed.jsonl`, `results/llm_extended.jsonl` |

## LLM agents

Agents 0, 2 and 4 can be driven by an open-source LLM served locally by
[Ollama](https://ollama.com) (Qwen2.5-3B, Llama-3.2-3B and Qwen2.5-7B in the paper). Every
LLM decision is cached, keyed by model and prompt, so the LLM missions replay exactly
without Ollama. To query the models again:

```bash
ollama serve &
ollama pull qwen2.5:3b && ollama pull llama3.2:3b && ollama pull qwen2.5:7b
.venv/bin/python -m experiments.run_llm --seeds 50 --teams planner mixed
.venv/bin/python -m experiments.run_llm_extended --seeds 50 --teams mixed_llama
.venv/bin/python -m experiments.run_llm_extended --seeds 30 --teams mixed_qwen7b
```

The extended LLM runner aborts if any LLM call fails, so an unreachable server can never
silently turn LLM agents into planners.

## Reproducibility notes

- Hyper-parameters were fixed on calibration seeds 1000-1079; all reported results use the
  disjoint evaluation seeds 0-99 and, for frozen operating points, fresh seeds 3000-3099.
- For a given seed, every scheme sees the same survivors and launch positions (common
  random numbers).
- With default settings, `gosc/realistic.py` reproduces the base simulator exactly; this is
  checked by `tests/test_realistic.py` (for example, seed 0 with GOSC: 122 slots, 13,304
  channel uses).

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
