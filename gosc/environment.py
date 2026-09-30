"""Disaster area with hidden survivors and a noisy on-board detector."""
import numpy as np

from .belief import SensorModel
from .config import Config


class World:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        n = cfg.grid
        if cfg.victim_layout == "clustered":
            cells = self._clustered(cfg, rng)
        else:
            cells = rng.choice(n * n, size=cfg.n_victims, replace=False)
        self.victims = frozenset(int(c) for c in cells)
        self.mask = np.zeros(n * n, dtype=bool)
        self.mask[cells] = True
        self.mask = self.mask.reshape(n, n)
        self.bs = (n // 2, n // 2)  # base station / edge coordinator
        self.deadline = {}
        self.p_detect = cfg.p_detect
        self.p_false_alarm = cfg.p_false_alarm
        self.n = n

    @staticmethod
    def _clustered(cfg, rng):
        """Survivors grouped around `victim_clusters` random centres (collapsed
        buildings): each survivor lies within 3 cells of its cluster centre."""
        n, out = cfg.grid, []
        centres = rng.integers(4, n - 4, size=(cfg.victim_clusters, 2))
        while len(out) < cfg.n_victims:
            cx, cy = centres[len(out) % cfg.victim_clusters]
            x, y = cx + rng.integers(-3, 4), cy + rng.integers(-3, 4)
            c = int(x * n + y)
            if c not in out:
                out.append(c)
        return np.array(out)

    def sense(self, pos, sensor: SensorModel, rng: np.random.Generator):
        sx, sy = sensor.window(pos)
        truth = self.mask[sx, sy]
        prob = np.where(truth, self.p_detect, self.p_false_alarm)
        xs, ys = np.nonzero(rng.random(truth.shape) < prob)
        return tuple(int((x + sx.start) * self.n + y + sy.start) for x, y in zip(xs, ys))
