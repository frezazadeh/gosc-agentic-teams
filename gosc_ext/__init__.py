"""Extensions of the GOSC simulator for the second revision of the journal paper.

The journal package `gosc` is imported read-only and extended by subclassing, so
every published number stays reproducible:

  * imperfect semantic values (multiplicative log-normal error, order-of-magnitude
    quantization, content-independent permuted values),
  * a fixed per-packet overhead (headers, CRC) on every uplink transmission, and a
    complete uplink + downlink cost account (broadcast, HARQ feedback, grants),
  * a shared, network-wide uplink budget for which the agents compete.

With the default `ExtConfig` every scheme reproduces the journal simulator exactly
(see tests/test_ext.py).
"""
from .config import ExtConfig
from .sim import ExtSimulation, run_ext

__all__ = ["ExtConfig", "ExtSimulation", "run_ext"]
