"""Packet formats and their exact payload sizes in bits."""
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config

HEADER_BITS = 8      # 3-bit message type + 5-bit agent id
COUNT_BITS = 8
MOVE_BITS = 3        # {stay, N, S, E, W}
RAW_SAMPLE_BITS = 8  # quantised sensor score per cell


def cell_bits(cfg: Config) -> int:
    return math.ceil(math.log2(cfg.grid * cfg.grid))


def footprint_bits(cfg: Config) -> int:
    return math.ceil(math.log2((2 * cfg.sense_radius + 1) ** 2))


JUMP_GAP_BITS = 8    # slot gap after a JUMP move code


def evidence_bits(n_steps: int, n_pos: int, cfg: Config, n_jumps: int = 0) -> int:
    """Aggregated evidence: anchor cell + move chain + (step, offset) of each detection.

    Records that are not consecutive in time are joined by a JUMP move code
    followed by an absolute cell and the slot gap.
    """
    if n_steps == 0:
        return 0
    step_idx = math.ceil(math.log2(n_steps)) if n_steps > 1 else 0
    return (HEADER_BITS + cell_bits(cfg) + 2 * COUNT_BITS
            + MOVE_BITS * (n_steps - 1) + n_jumps * (cell_bits(cfg) + JUMP_GAP_BITS)
            + n_pos * (step_idx + footprint_bits(cfg)))


def evidence_sizes(records, cfg: Config):
    """Size of the evidence packet carrying the first k+1 records, for every k."""
    sizes, npos, jumps = [], 0, 0
    for k, r in enumerate(records):
        npos += len(r.det)
        if k > 0 and r.t != records[k - 1].t + 1:
            jumps += 1
        sizes.append(evidence_bits(k + 1, npos, cfg, jumps))
    return sizes


def sem_step_bits(n_pos: int, cfg: Config) -> int:
    """Per-slot structured semantic update: position, detections, intent."""
    return HEADER_BITS + cell_bits(cfg) + 5 + n_pos * footprint_bits(cfg) + cell_bits(cfg)


def raw_bits(cfg: Config) -> int:
    """Raw sensor frame: position + quantised score of every footprint cell."""
    return HEADER_BITS + cell_bits(cfg) + RAW_SAMPLE_BITS * (2 * cfg.sense_radius + 1) ** 2


def confirm_bits(cfg: Config) -> int:
    return HEADER_BITS + cell_bits(cfg)


def intent_bits(cfg: Config) -> int:
    return HEADER_BITS + cell_bits(cfg)


def _xy(cell: int, n: int):
    return cell // n, cell % n


def nl_step_text(agent: int, t: int, pos, det, target: Optional[int], cfg: Config) -> str:
    """Natural-language status message, as exchanged by LLM-based agents."""
    n, side = cfg.grid, 2 * cfg.sense_radius + 1
    s = (f"[t={t}] Agent {agent} here. My position is ({pos[0]}, {pos[1]}). "
         f"I scanned the {side}x{side} area around me. ")
    if det:
        s += ("Possible survivor signatures detected at "
              + ", ".join("({}, {})".format(*_xy(c, n)) for c in det) + ". ")
    else:
        s += "No survivor signatures detected in this area. "
    if target is not None:
        s += "Next, I plan to move toward ({}, {}) to continue the search.".format(*_xy(target, n))
    return s


def nl_confirm_text(agent: int, cell: int, cfg: Config) -> str:
    return ("URGENT from Agent {}: survivor CONFIRMED at ({}, {}). "
            "Please dispatch the rescue team.").format(agent, *_xy(cell, cfg.grid))


def text_bits(s: str) -> int:
    return 8 * len(s.encode("utf-8"))


@dataclass
class Packet:
    kind: str                     # EVID | RAW | SEM | NL | CONF | INTENT
    sender: int
    bits: int
    created: int
    records: List = field(default_factory=list)
    confirm: Optional[int] = None
    intent: Optional[int] = None
    sent: int = 0                 # bits already delivered (FIFO fragmentation)
