"""Agents whose intention selection is performed by an open-source LLM.

The LLM receives a structured, text-rendered summary of the agent's belief
(candidate targets with survivor mass and distance) and of the common knowledge
about teammates, and returns a typed decision constrained by a JSON schema.
Its communication is handled by the same scheme as every other agent, so LLM and
classical planner agents interoperate through one semantic protocol.

Responses are cached on disk (keyed by model and prompt), which makes every run
deterministic and lets the experiments be re-run without the model.
"""
import hashlib
import json
import os
import string
import time
import urllib.request

import numpy as np

from .agent import Agent
from .belief import box_sum

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "llm_cache")


class LLMClient:
    """Minimal Ollama client with a persistent append-only response cache."""

    def __init__(self, model: str, cache_dir: str = CACHE_DIR, transport=None):
        self.model = model
        self.transport = transport or self._http
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"{model.replace(':', '_')}.jsonl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a partially written last line
        self.calls = 0
        self.hits = 0
        self.seconds = 0.0

    def _http(self, body: dict) -> dict:
        req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())

    def choose(self, prompt: str, labels) -> str:
        schema = {"type": "object",
                  "properties": {"choice": {"type": "string", "enum": list(labels)}},
                  "required": ["choice"]}
        key = hashlib.sha256(f"{self.model}\n{prompt}\n{','.join(labels)}".encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        body = {"model": self.model, "prompt": prompt, "format": schema, "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_predict": 12, "num_ctx": 1024},
                "keep_alive": "60m"}
        t0 = time.time()
        for attempt in range(3):
            try:
                resp = json.loads(self.transport(body)["response"])["choice"]
                break
            except Exception:  # server hiccup or malformed output
                if attempt == 2:
                    resp = None
                time.sleep(1.0)
        self.seconds += time.time() - t0
        self.calls += 1
        if resp in labels:
            self.cache[key] = resp
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "response": resp}) + "\n")
        return resp


_CLIENTS = {}


def get_client(model: str) -> LLMClient:
    if model not in _CLIENTS:
        _CLIENTS[model] = LLMClient(model)
    return _CLIENTS[model]


PROMPT = """You control one autonomous UAV in an earthquake search-and-rescue team. The team shares knowledge through a 6G edge server. You will see candidate search targets on a grid. "mass" is the expected number of survivors in the 5x5 area around a target according to your belief; "distance" is your travel time in steps; "nearest teammate" is the distance from the target to the closest teammate target or position. Good choices have high mass, are not too far, and do not overlap with areas teammates are already covering. Answer with the letter of ONE target.

You are Agent {k} of {K} on a {n}x{n} grid. Position: ({x}, {y}). Survivors located: {found}.
Candidates:
{cands}
Teammates (announced target, or last known position):
{mates}"""


class LLMAgent(Agent):
    kind = "llm"

    def __init__(self, idx, pos, cfg, sim, client: LLMClient):
        super().__init__(idx, pos, cfg)
        self.sim = sim
        self.client = client
        self.last_call = -10 ** 9
        self.target_mass = 0.0
        self.decisions = 0
        self.fallbacks = 0
        self.agree_with_planner = 0

    def _candidates(self, p, rng):
        c, n = self.cfg, self.cfg.grid
        mass = box_sum(p, c.sense_radius)
        dist = np.abs(self._X - self.pos[0]) + np.abs(self._Y - self.pos[1])
        score = (mass / (1.0 + c.distance_discount * dist)).reshape(-1).copy()
        picks, sep = [], 2 * c.sense_radius + 1
        for cell in np.argsort(-score):
            if len(picks) == c.llm_candidates or score[cell] <= 0:
                break
            x, y = divmod(int(cell), n)
            if all(max(abs(x - a), abs(y - b)) >= sep for a, b in (divmod(q, n) for q in picks)):
                picks.append(int(cell))
        order = rng.permutation(len(picks))  # remove position bias of the ranking
        return [picks[i] for i in order], mass, dist

    def _mates(self):
        sim, n = self.sim.view(self), self.cfg.grid
        out = []
        for j in range(self.cfg.n_agents):
            if j == self.idx:
                continue
            if j in sim.team_intent:
                out.append((j, "target", divmod(sim.team_intent[j], n)))
            elif j in sim.team_pos:
                out.append((j, "position", sim.team_pos[j]))
        return out

    def plan(self, p, others, rng):
        c, n = self.cfg, self.cfg.grid
        t = self.sim.t
        here = self.pos[0] * n + self.pos[1]
        if self.target is not None and self.target != here and t - self.last_call < c.llm_replan_every:
            m = box_sum(p, c.sense_radius).reshape(-1)[self.target]
            if m >= 0.5 * self.target_mass:
                return  # commitment to the current intention
        cands, mass, dist = self._candidates(p, rng)
        if not cands:
            return super().plan(p, others, rng)
        mates = self._mates()
        pts = [q for _, _, q in mates]
        labels = list(string.ascii_uppercase[:len(cands)])
        lines = []
        for lab, cell in zip(labels, cands):
            x, y = divmod(cell, n)
            near = min((abs(x - a) + abs(y - b) for a, b in pts), default=None)
            lines.append(f"{lab}: ({x}, {y}) mass {mass[x, y]:.2f}, distance {int(dist[x, y])}, "
                         f"nearest teammate {near if near is not None else 'unknown'}")
        mates_txt = "\n".join(f"Agent {j} {kind} ({a}, {b})" for j, kind, (a, b) in mates) or "none known"
        prompt = PROMPT.format(k=self.idx, K=c.n_agents, n=n, nm=n - 1, x=self.pos[0], y=self.pos[1],
                               found=int(self.sim.view(self).declared.sum()), cands="\n".join(lines),
                               mates=mates_txt)
        choice = self.client.choose(prompt, labels)
        # what the classical planner would pick (no hysteresis), for the fallback and
        # for the agreement statistic: the candidate closest to the planner's target
        self.target = None
        Agent.plan(self, p, others, rng)
        px, py = divmod(self.target, n)
        closest = min(cands, key=lambda q: abs(q // n - px) + abs(q % n - py))
        if choice is None:
            self.fallbacks += 1
            chosen = self.target
        else:
            chosen = cands[labels.index(choice)]
        self.agree_with_planner += int(chosen == closest)
        self.decisions += 1
        self.target = chosen
        self.target_mass = float(mass.reshape(-1)[chosen])
        self.last_call = t
