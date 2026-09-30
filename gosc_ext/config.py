"""Configuration of the revision-2 extensions (defaults = journal model)."""
from dataclasses import dataclass

from gosc import Config


@dataclass(frozen=True)
class ExtConfig(Config):
    # --- imperfect semantic valuation ---------------------------------------
    # "exact": the scheme's own value; "lognormal": value * exp(sigma * z), one draw
    # per candidate message; "quantized": value rounded to the nearest power of ten;
    # "permuted": a content-independent value whose magnitude is drawn from the
    # values of earlier messages of the same type in the mission (shape across
    # evidence prefixes proportional to the payload size).
    value_mode: str = "exact"
    value_sigma: float = 0.0

    # --- packet overhead and cost accounting ---------------------------------
    overhead_bits: int = 0          # fixed bits (header + CRC) added to every uplink packet
    ack_bits: int = 1               # HARQ feedback per uplink transmission (downlink)
    bcast_overhead: bool = True     # each non-empty broadcast carries overhead_bits too

    # --- shared, network-wide uplink budget ----------------------------------
    shared_budget: int = 0          # total channel uses per slot for all agents (0: orthogonal)
    bsr_bits: int = 8               # buffer status report of a periodic agent (per slot)
    req_header_bits: int = 8        # GOSC scheduling request header (per slot)
    req_option_bits: int = 16       # GOSC: (value, size) pair per reported option
    grant_bits: int = 32            # downlink grant per scheduled agent and slot
    shared_mode: str = "price"      # GOSC: "price" (broadcast network price) or "central"
    price_bits: int = 8             # broadcast network price (per slot)
    price_gain: float = 1.0         # multiplicative price update gain
