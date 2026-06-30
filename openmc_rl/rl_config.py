"""
rl_config.py — RLConfig dataclass + chromosome/geometry helpers  (NxN)
--------------------------------------------------------------------------

[At the moment, 'make_sym_index, chromosome_to_enr_grid, enr_grid_to_chromosome' below are copied from environments.mapping.py and 
openmc_ga/ga_config.py rather than imported, so this package can be dropped in or removed without touching anything else in the platform later]

The chromosome FORMAT ––> identical to GA's: 
    - a binary vector of length n_sym_rods = n_rods_side*(n_rods_side+1)/2 under 1/2-diagonal symmetry, bit 0 = ENR_LOW, bit 1 = ENR_HIGH. 
    - Implying that the chromosome built by the RL can be handed straight to environments.env.evaluate_for_ga(chrom, cfg) — the same physics oracle GA 
    already uses — with zero translation.

(Note:- evaluate_for_ga() itself decodes chromosomes using environments.mapping's own ENR_LOW/ ENR_HIGH (not cfg.enr_low/enr_high) — so if
cfg.enr_low/enr_high is changed here, update mapping.py, or RLConfig)


"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np



# Defaults

RL_ENR_LOW_DEFAULT     = 1.87   # wt% U-235
RL_ENR_HIGH_DEFAULT    = 2.53   # wt% U-235
RL_N_RODS_SIDE_DEFAULT = 6



# Geometry helpers

def make_sym_index(n_rods_side: int) -> List[Tuple[int, int]]:
    
    #Builds the list of unique (row, col) positions under 1/2-diagonal symmetry 

    return [
        (r, c)
        for r in range(n_rods_side)
        for c in range(r + 1)
    ]


def chromosome_to_enr_grid(
    chrom:       np.ndarray,
    sym_index:   List[Tuple[int, int]],
    n_rods_side: int,
    enr_low:     float = RL_ENR_LOW_DEFAULT,
    enr_high:    float = RL_ENR_HIGH_DEFAULT,
) -> np.ndarray:
    #Decode a binary chromosome to a full nxn grid
    chrom = np.asarray(chrom, dtype=int).ravel()
    n_sym = len(sym_index)
    if len(chrom) != n_sym:
        raise ValueError(f"Expected {n_sym} chromosome bits, got {len(chrom)}")

    grid = np.zeros((n_rods_side, n_rods_side), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        grid[r, c] = chrom[idx]
        grid[c, r] = chrom[idx]   # mirror — no-op on diagonal (r == c)
    return np.where(grid == 1, enr_high, enr_low)


def enr_grid_to_chromosome(
    enr_grid:  np.ndarray,
    sym_index: List[Tuple[int, int]],
    enr_high:  float = RL_ENR_HIGH_DEFAULT,
) -> np.ndarray:
    #Encode an nxn grid back to a binary chromosome.
    chrom = np.zeros(len(sym_index), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        chrom[idx] = 1 if round(enr_grid[r, c], 4) == round(enr_high, 4) else 0
    return chrom


def high_units_reachable(n_high_target: int, n_diag_avail: int, n_offdiag_avail: int) -> bool:
    """
   Checks if a target number of high-enrichment pins is physically possible using the available symmetry slots (O(1) check).

    Formulas used:
      target_pins = d + 2*o  (d = diagonal slots used, o = off-diagonal slots used)
      max_pins = n_diag_avail + 2*n_offdiag_avail

    The logic shortcut:
      - Off-diagonal slots (o) mirror twice, so they always add in pairs (even numbers).
      - Diagonal slots (d) sit on the symmetry fold line, adding exactly 1.
      - If we have at least 1 diagonal slot left, we can use it as a "+1" to hit odd numbers. 
      - If we have 0 diagonal slots left, we are forced to build with pairs, meaning odd targets are impossible.
    
    """
    if n_high_target < 0:
        return False
    max_high = n_diag_avail + 2 * n_offdiag_avail
    if n_high_target > max_high:
        return False
    if n_diag_avail == 0:
        return n_high_target % 2 == 0
    return True



#********************** RLConfig dataclass ***************************####################


@dataclass
class RLConfig:
    """
    All parameters for the RL (PPO + action masking) enrichment-layout agent.

     RL and GA share the same core geometry inputs, so the RL agent's output is directly compatible with evaluate_for_ga().

     PPO/network hyperparameters are consumed by rl_policy.build_agent(); they have no counterpart in GAConfig/RunConfig since GA/BO don't train a model.


    reward = -fitness = -{alpha_k*|k_target − k∞| + alpha_ppf*max(0, PPF − ppf_target)^2  #########–––––––––> fitness samee as GA
                        + adj_penalty_weight*adj_high_rods}
   
    [==> reward = -Fitness:

        - GA minimizes penalties (wants the lowest score possible).
        - RL maximizes rewards (wants the highest score possible).

      Flipping the sign, GA penalty of 100 becomes an RL reward of -100. 
      The RL agent will naturally try to climb up toward 0, which does exactly what we want: it minimizes the physical penalties.]
   - Reactivity Penalty: Penalizes deviation from target k-infinity.
   - PPF Penalty: Uses squared hinge loss; only punishes PPF if it exceeds 
     the target threshold.
   - Adjacency Penalty: A flat cost per pair of adjacent high-enrichment rods.

    (note:- Corner masking (use_corner_masking) is a HARD constraint here too, same as GA, corner positions are removed from the HIGH action's legal set
    entirely inside BWREnrichmentEnv.action_masks(), not penalised after the fact.)
    """

    # Assembly size
    n_rods_side: int = RL_N_RODS_SIDE_DEFAULT

    # Enrichment inventory (see module docstring re: mapping.py agreement)
    enr_low:     float = RL_ENR_LOW_DEFAULT
    enr_high:    float = RL_ENR_HIGH_DEFAULT
    n_high_rods: int   = 23   # high-enrichment rods in the FULL N x N grid

    # Targets (identical formula to GAConfig.fitness)
    k_target:   float           = 1.25
    ppf_target: Optional[float] = 1.35
    alpha_k:    float           = 10.0
    alpha_ppf:  float           = 5.0

    # Physics-informed constraints
    use_corner_masking:    bool  = False   # hard coded 
    use_adjacency_penalty: bool  = False   # soft penalty
    adj_penalty_weight:    float = 10.0
    shaping_scale:         float = 0.05    # keeps per-step shaping < terminal reward

    # PPO hyperparameters –––––––––––––––––––> (consumed by rl_policy.build_agent)
    policy_hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate:       float = 3e-4
    n_steps:              int  = 128     # rollout length collected per PPO update
    batch_size:            int  = 64
    n_epochs:              int  = 10
    gamma:                float = 0.99
    gae_lambda:           float = 0.95
    clip_range:           float = 0.2
    ent_coef:             float = 0.01
    vf_coef:              float = 0.5
    total_timesteps:       int  = 50000
    random_seed:            int  = 7

    # Episode-logging cadence — similar to GA's "generation"
    log_every_n_episodes: int = 20

    # OpenMC settings (production)
    n_particles: int = 5000
    n_inactive:  int = 25
    n_active:    int = 50

    # OpenMC settings (smoke-test)
    n_particles_smoke: int = 500
    n_inactive_smoke:  int = 10
    n_active_smoke:    int = 20

    model_name:  str = "bwr_nxn_rl"
    results_dir: str = "results_rl"

    # Derived (do not set directly)
    n_rods_total: int                  = field(init=False, default=0)
    n_sym_rods:   int                  = field(init=False, default=0)
    n_vars:       int                  = field(init=False, default=0)
    sym_index:    List[Tuple[int, int]] = field(init=False, default_factory=list)
    diag_idx:     List[int]            = field(init=False, default_factory=list)
    offdiag_idx:  List[int]            = field(init=False, default_factory=list)
    n_diag:       int                  = field(init=False, default=0)
    n_offdiag:    int                  = field(init=False, default=0)

    def __post_init__(self):
        # Geometry
        self.n_rods_total = self.n_rods_side ** 2
        self.sym_index    = make_sym_index(self.n_rods_side)
        self.n_sym_rods   = len(self.sym_index)
        self.n_vars       = self.n_sym_rods   # one binary decision per symmetry position

        self.diag_idx    = [i for i, (r, c) in enumerate(self.sym_index) if r == c]
        self.offdiag_idx = [i for i, (r, c) in enumerate(self.sym_index) if r != c]
        self.n_diag       = len(self.diag_idx)
        self.n_offdiag    = len(self.offdiag_idx)

        # Validation — unmasked feasibility
        assert 0 < self.n_high_rods < self.n_rods_total, (
            f"n_high_rods={self.n_high_rods} must be in "
            f"(0, {self.n_rods_total}) for a {self.n_rods_side}x{self.n_rods_side} assembly"
        )
        if not high_units_reachable(self.n_high_rods, self.n_diag, self.n_offdiag):
            raise ValueError(
                f"n_high_rods={self.n_high_rods} is not reachable with "
                f"n_diag={self.n_diag}, n_offdiag={self.n_offdiag} "
                f"(max={self.n_diag + 2 * self.n_offdiag}, "
                f"parity required if n_diag==0)."
            )

        # Validation — masked feasibility (corner positions excluded from HIGH)
        if self.use_corner_masking:
            corner_idx     = self.get_corner_indices()
            usable_diag    = sum(1 for i in self.diag_idx    if i not in corner_idx)
            usable_offdiag = sum(1 for i in self.offdiag_idx if i not in corner_idx)
            if not high_units_reachable(self.n_high_rods, usable_diag, usable_offdiag):
                raise ValueError(
                    f"n_high_rods={self.n_high_rods} is not reachable once corner "
                    f"masking excludes corner positions from HIGH candidacy "
                    f"(usable_diag={usable_diag}, usable_offdiag={usable_offdiag}, "
                    f"max={usable_diag + 2 * usable_offdiag}). Lower n_high_rods, "
                    f"disable use_corner_masking, or increase n_rods_side."
                )

    
#***************************** Helpers ***********************################
    

    def decode(self, chrom: np.ndarray) -> np.ndarray:
        return chromosome_to_enr_grid(
            chrom, self.sym_index, self.n_rods_side, self.enr_low, self.enr_high,
        )

    def encode(self, enr_grid: np.ndarray) -> np.ndarray:
        return enr_grid_to_chromosome(enr_grid, self.sym_index, self.enr_high)

    def fitness(self, keff: float, ppf: float = 0.0) -> float:
        """
        Scalar fitness (lower the better) — identical formula to GAConfig.fitness(). 
        Does not include the adjacency penalty term; that's added where the heuristic count is available (see rl_env.BWREnrichmentEnv.step()'s terminal branch), 
        the same division of responsibility ga_fitness.fitness_single() uses.
        """
        k_term   = self.alpha_k * abs(self.k_target - keff)
        ppf_term = 0.0
        if self.ppf_target is not None:
            excess   = max(0.0, ppf - self.ppf_target)
            ppf_term = self.alpha_ppf * excess ** 2
        return float(k_term + ppf_term)

    def average_enrichment(self, enr_grid: np.ndarray) -> float:
        return float(enr_grid.mean())

    def get_corner_indices(self) -> set:
        """
        Returns the set of 1D chromosome (sym_index) positions that map to the physical N x N corners (like in mirrors GAConfig.get_corner_indices().)
        Null set if use_corner_masking is False.
        """
        if not self.use_corner_masking:
            return set()
        N = self.n_rods_side
        physical_corners = {(0, 0), (N - 1, N - 1), (N - 1, 0)}
        return {i for i, (r, c) in enumerate(self.sym_index) if (r, c) in physical_corners}

    def summary(self) -> str:
        avg_enr = (
            self.n_high_rods * self.enr_high
            + (self.n_rods_total - self.n_high_rods) * self.enr_low
        ) / self.n_rods_total
        lines = [
            f"RLConfig — {self.model_name}",
            f"  Assembly           : {self.n_rods_side}x{self.n_rods_side} = {self.n_rods_total} pins",
            f"  Episode length     : {self.n_sym_rods} decisions/episode (1/2-diagonal symmetry)",
            f"  ENR_LOW            : {self.enr_low} wt%",
            f"  ENR_HIGH           : {self.enr_high} wt%",
            f"  n_high_rods        : {self.n_high_rods} / {self.n_rods_total}  (avg {avg_enr:.4f} wt%)",
            f"  k-inf target       : {self.k_target}  (alpha_k = {self.alpha_k})",
            f"  PPF target         : {self.ppf_target}  (alpha_ppf = {self.alpha_ppf})",
            f"  Corner masking     : {self.use_corner_masking}  (hard mask, applied in action_masks())",
            f"  Adjacency penalty  : {self.use_adjacency_penalty}  (weight={self.adj_penalty_weight})",
            f"  Policy net_arch    : {self.policy_hidden_sizes}",
            f"  Learning rate      : {self.learning_rate}",
            f"  Rollout / batch    : n_steps={self.n_steps}  batch_size={self.batch_size}  n_epochs={self.n_epochs}",
            f"  Gamma / GAE-lambda : {self.gamma} / {self.gae_lambda}",
            f"  Total timesteps    : {self.total_timesteps:,}  "
            f"(~{self.total_timesteps // max(self.n_sym_rods, 1):,} OpenMC evaluations)",
            f"  Random seed        : {self.random_seed}",
            f"  OpenMC particles   : {self.n_particles}  ({self.n_inactive} inactive + {self.n_active} active)",
            f"  Smoke particles    : {self.n_particles_smoke}  "
            f"({self.n_inactive_smoke} inactive + {self.n_active_smoke} active)",
            f"  Results dir        : {self.results_dir}",
        ]
        return "\n".join(lines)
