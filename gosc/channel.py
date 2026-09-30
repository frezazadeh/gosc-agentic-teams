"""Block-fading uplink: path loss, Rayleigh fading and packet decoding.

Two packet-error models are available (Config.error_model):
  "outage": a packet at rate R is decoded iff log2(1 + snr |h|^2) >= R
            (infinite-blocklength idealization; the default);
  "fbl":    finite-blocklength normal approximation of the block error
            probability for a packet of n channel uses (Polyanskiy et al.),
            averaged over the Rayleigh fading when only the mean SNR is known;
            Config.code_gap_db additionally models the SNR gap of practical
            short codes to the normal approximation.
"""
import math
from functools import lru_cache

import numpy as np

from .config import Config

# E_x[f(x)] for x ~ Exp(1) by the midpoint rule in probability space (x = -ln(1-u)),
# which resolves the sharp decoding transition at small fading gains.
_U = (np.arange(4000) + 0.5) / 4000
_EXP_Q = -np.log1p(-_U)
_LOG2E = 1.0 / math.log(2.0)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def fbl_error(n: int, rate: float, snr):
    """Normal approximation of the block error probability of an n-symbol packet at
    rate R (bit/channel use) over an AWGN channel with the given SNR."""
    snr = np.asarray(snr, dtype=float)
    cap = np.log2(1.0 + snr)
    disp = (1.0 - (1.0 + snr) ** -2) * _LOG2E ** 2
    z = (n * (cap - rate) + 0.5 * math.log2(n)) / np.sqrt(np.maximum(n * disp, 1e-300))
    return np.asarray(0.5 * _erfc(z / math.sqrt(2.0)), dtype=float)


@lru_cache(maxsize=500_000)
def _fbl_success(n: int, rate: float, snr_mean: float) -> float:
    return float(np.mean(1.0 - fbl_error(n, rate, snr_mean * _EXP_Q)))


class Channel:
    def __init__(self, cfg: Config, bs, rng: np.random.Generator = None):
        self.cfg = cfg
        self.bs = bs
        self.rates = np.asarray(cfg.rates, dtype=float)
        self.snr_ref = 10 ** (cfg.snr_ref_db / 10)
        self.fbl = cfg.error_model == "fbl"
        self.csit = cfg.csit
        self.gap = 10 ** (-cfg.code_gap_db / 10)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def mean_snr(self, pos) -> float:
        """Large-scale SNR known at the transmitter (statistical CSI)."""
        c = self.cfg
        d = math.hypot(pos[0] - self.bs[0], pos[1] - self.bs[1]) * c.cell_size_m
        d = max(d, c.min_distance_m)
        return self.snr_ref * (c.ref_distance_m / d) ** c.pathloss_exp

    def success_prob(self, rate, snr_mean, n=None):
        """Probability that a packet of n channel uses at `rate` is decoded, given
        only the mean SNR (Rayleigh fading averaged out).  With CSIT, `snr_mean` is
        the known instantaneous SNR and the probability is conditional on it."""
        if self.csit:
            if not self.fbl:
                return (np.log2(1.0 + snr_mean) >= np.asarray(rate)).astype(float)
            n = self.cfg.symbols_per_slot if n is None else max(int(n), 1)
            snr = snr_mean * self.gap
            if np.ndim(rate):
                return np.array([1.0 - float(fbl_error(n, float(r), snr)) for r in rate])
            return 1.0 - float(fbl_error(n, float(rate), snr))
        if not self.fbl:
            return np.exp(-(2.0 ** np.asarray(rate) - 1.0) / snr_mean)
        n = self.cfg.symbols_per_slot if n is None else max(int(n), 1)
        if np.ndim(rate):
            return np.array([_fbl_success(n, float(r), float(snr_mean * self.gap)) for r in rate])
        return _fbl_success(n, float(rate), float(snr_mean * self.gap))

    def fixed_rate(self, snr_mean: float) -> float:
        """Conventional link adaptation: highest rate meeting the BLER target
        (for the FBL model, evaluated for a packet filling the whole slot)."""
        ok = self.rates[self.success_prob(self.rates, snr_mean) >= 1 - self.cfg.target_bler]
        return float(ok.max()) if ok.size else float(self.rates[0])

    def decodes(self, rate: float, snr_inst: float, n=None) -> bool:
        if not self.fbl:
            return math.log2(1.0 + snr_inst) >= rate
        n = self.cfg.symbols_per_slot if n is None else max(int(n), 1)
        return bool(self.rng.random() >= float(fbl_error(n, rate, snr_inst * self.gap)))
