"""Simulation parameters for the multi-agent search-and-rescue scenario."""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class Config:
    # --- mission / world -------------------------------------------------
    grid: int = 32                  # N x N search area (cells)
    n_agents: int = 6               # cooperating agents (UAVs)
    n_victims: int = 8              # hidden survivors
    victim_layout: str = "uniform"  # "uniform" or "clustered"
    task: str = "search"            # "search" or "rescue" (deadlines + rescue service)
    deadline_min: int = 100         # rescue task: survivor deadlines ~ U[min, max] slots
    deadline_max: int = 300
    rescue_service: int = 8         # rescue task: slots an agent must spend at a survivor
    victim_clusters: int = 2        # clusters for the clustered layout
    sense_radius: int = 2           # Chebyshev radius of the sensing footprint
    p_detect: float = 0.85          # P(detection | survivor in cell)
    p_false_alarm: float = 0.05     # P(detection | empty cell)
    confirm_threshold: float = 0.99  # posterior needed to declare a survivor
    max_steps: int = 400            # mission horizon (slots)

    # --- 6G uplink ---------------------------------------------------------
    cell_size_m: float = 10.0
    ref_distance_m: float = 100.0   # distance at which mean SNR equals snr_ref_db
    min_distance_m: float = 10.0
    pathloss_exp: float = 3.0
    snr_ref_db: float = 5.0
    symbols_per_slot: int = 100     # channel uses granted to each agent per slot
    rates: Tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    target_bler: float = 0.1        # conventional link adaptation target
    max_packet_age: int = 20        # FIFO baselines discard data older than this
    error_model: str = "outage"     # "outage" or "fbl" (finite blocklength)
    code_gap_db: float = 0.0        # FBL only: SNR gap of practical short codes
    csit: bool = False              # transmitter knows the instantaneous SNR (e.g., TDD reciprocity)

    # --- downlink broadcast of the common knowledge -----------------------------
    dl_loss: float = 0.0            # P(an agent misses a slot's broadcast)
    dl_delay: int = 0               # broadcast latency (slots)
    dl_rate: float = 2.0            # downlink spectral efficiency, for overhead accounting

    # --- agent planner -----------------------------------------------------
    distance_discount: float = 0.1
    hysteresis: float = 1.2
    repulsion_sigma: float = 5.0

    # --- goal-oriented semantic communication (GOSC) ------------------------
    energy_price: float = 1e-5      # eta: value units per channel use
    confirm_value: float = 5.0      # value of a survivor confirmation
    intent_value: float = 0.5       # value of a fully changed intent
    tom_temperature: float = 0.1    # softmax temperature of the teammate model
    filter_eps: float = 0.02        # semantic source filter threshold (expected survivors)
    tom_oracle: bool = False        # (analysis) teammate model sees teammates' private evidence
    log_private: bool = False       # (analysis) log the size of agents' private evidence
    sem_interval: int = 1           # reporting interval of the aggregated periodic baseline

    # --- LLM agents (heterogeneous teams) ------------------------------------
    llm_agents: Tuple[int, ...] = ()  # indices of agents driven by the LLM
    llm_model: str = "qwen2.5:3b"
    llm_replan_every: int = 5         # slots between LLM deliberations
    llm_candidates: int = 5           # candidate targets shown to the LLM

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
