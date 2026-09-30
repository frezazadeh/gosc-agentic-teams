"""Heterogeneous-team experiment: LLM agents and classical planner agents cooperate
through the same communication scheme.

Results are appended per mission to results/llm_mixed.jsonl, so the run can be
interrupted and resumed. Requires a running Ollama server with the model pulled
(unless every decision is already in results/llm_cache/).

    python -m experiments.run_llm --seeds 20
"""
import argparse
import json
import os
import time

from gosc import Config, run_episode

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "llm_mixed.jsonl")
# scheme-specific settings: the throughput-value rule at a price that gives about the
# same channel use as GOSC's default price in the planner team
SCHEME_CFG = {"gosc_bits": {"energy_price": 0.3}}
TEAMS = {"mixed": ((0, 2, 4), "qwen2.5:3b"),
         "mixed_llama": ((0, 2, 4), "llama3.2:3b"),
         "planner": ((), "qwen2.5:3b")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--schemes", nargs="*", default=["sem", "gosc", "genie"])
    ap.add_argument("--teams", nargs="*", default=["planner", "mixed"])
    args = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["team"], r["scheme"], r["seed"]))
    for seed in range(args.seeds):  # seed-major: partial runs stay balanced
        for team in args.teams:
            for scheme in args.schemes:
                if (team, scheme, seed) in done:
                    continue
                agents, model = TEAMS[team]
                cfg = Config().with_(llm_agents=agents, llm_model=model, **SCHEME_CFG.get(scheme, {}))
                t0 = time.time()
                out = run_episode(cfg, scheme, seed)
                out.pop("curve")
                out.update(team=team, scheme=scheme, seed=seed, model=model if agents else None,
                           wall_s=round(time.time() - t0, 1))
                with open(OUT, "a") as f:
                    f.write(json.dumps(out) + "\n")
                print(f"seed {seed} {team:7s} {scheme:5s} T={out['completion_time']:3d} "
                      f"wall={out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
