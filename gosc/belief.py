"""Log-odds occupancy beliefs, the sensor model and evidence records."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def logit(p):
    return float(np.log(p) - np.log1p(-p))


@dataclass(frozen=True)
class Obs:
    """One sensing event: agent position and the cells that triggered a detection."""
    t: int
    agent: int
    pos: Tuple[int, int]
    det: Tuple[int, ...]  # flat cell indices with a positive detection


class SensorModel:
    def __init__(self, p_detect: float, p_false_alarm: float, radius: int, grid: int):
        self.llr_pos = float(np.log(p_detect / p_false_alarm))
        self.llr_neg = float(np.log((1 - p_detect) / (1 - p_false_alarm)))
        self.radius = radius
        self.grid = grid

    def window(self, pos):
        x, y = pos
        r, n = self.radius, self.grid
        return slice(max(x - r, 0), min(x + r + 1, n)), slice(max(y - r, 0), min(y + r + 1, n))

    def apply(self, L: np.ndarray, rec: Obs, sign: float = 1.0) -> None:
        """Add (sign=+1) or remove (sign=-1) the log-likelihood ratio of a record."""
        sx, sy = self.window(rec.pos)
        L[sx, sy] += sign * self.llr_neg
        if rec.det:
            flat = L.reshape(-1)
            flat[np.asarray(rec.det, dtype=np.int64)] += sign * (self.llr_pos - self.llr_neg)


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of `a` over the (2r+1)x(2r+1) window centred on every cell (clipped)."""
    n = a.shape[0]
    c = np.zeros((n + 1, n + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    i = np.arange(n)
    lo = np.clip(i - r, 0, n)
    hi = np.clip(i + r + 1, 0, n)
    return c[np.ix_(hi, hi)] - c[np.ix_(lo, hi)] - c[np.ix_(hi, lo)] + c[np.ix_(lo, lo)]
