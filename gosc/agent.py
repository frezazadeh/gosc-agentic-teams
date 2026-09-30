"""Cognitive search agent: belief state, intention selection and motion."""
from collections import deque
from typing import Iterable, List, Optional

import numpy as np

from .belief import box_sum
from .config import Config


class Agent:
    def __init__(self, idx: int, pos, cfg: Config):
        n = cfg.grid
        self.idx = idx
        self.pos = tuple(pos)
        self.cfg = cfg
        self.target: Optional[int] = None       # intention (flat cell)
        self.shared_intent: Optional[int] = None  # intention the team currently knows
        self.pending_llr = np.zeros((n, n))     # own evidence not yet delivered
        self.pending_records: List = []         # GOSC: undelivered records, oldest first
        self.confirmed_pending = set()          # self-confirmed survivors awaiting declaration
        self.queue = deque()                    # FIFO baselines: data packets
        self.prio = deque()                     # FIFO baselines: confirmation packets
        self.trajectory = [self.pos]
        X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self._X, self._Y = X, Y

    def plan(self, p: np.ndarray, others: Iterable, rng: np.random.Generator) -> None:
        """Pick the cell whose footprint holds the most survivor mass, discounted by
        travel distance and by proximity to teammates' known intents/positions."""
        c, n = self.cfg, self.cfg.grid
        X, Y = self._X, self._Y
        score = box_sum(p, c.sense_radius)
        score = score / (1.0 + c.distance_discount * (np.abs(X - self.pos[0]) + np.abs(Y - self.pos[1])))
        two_s2 = 2.0 * c.repulsion_sigma ** 2
        for ox, oy in others:
            score = score * (1.0 - np.exp(-((X - ox) ** 2 + (Y - oy) ** 2) / two_s2))
        score = score.reshape(-1) + rng.random(n * n) * 1e-12
        best = int(np.argmax(score))
        here = self.pos[0] * n + self.pos[1]
        if (self.target is None or self.target == here
                or score[best] > c.hysteresis * score[self.target]):
            self.target = best

    def step(self) -> None:
        n = self.cfg.grid
        tx, ty = divmod(self.target, n)
        x, y = self.pos
        dx, dy = tx - x, ty - y
        if abs(dx) >= abs(dy) and dx != 0:
            x += int(np.sign(dx))
        elif dy != 0:
            y += int(np.sign(dy))
        self.pos = (x, y)
        self.trajectory.append(self.pos)
