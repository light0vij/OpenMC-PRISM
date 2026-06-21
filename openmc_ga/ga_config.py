"""
ga_config.py — GAConfig dataclass + chromosome helpers  (NxN)
--------------------------------------------------------------------------

An NxN BWR assembly has N^2 pins and in a 1/2-diagonal symmetry there are N*(N+1)/2 unique positions. 
The pins are binary enriched, i.e., 2 types of enrichment. 

  ENR_LOW  = 1.87 wt%   (chromosome bit 0)
  ENR_HIGH = 2.53 wt%   (chromosome bit 1)

The value or the number of high enriched rods (n_high_rods) is given by the user when constructing GAConfig

The only number that must be changed in the code is BWR_N, whcih is passed as
n_rods_side to GAConfig. Everything else including the chromosome length, symmetry
mapping, rod counts are computed automatically.

"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple



# Defaults 
GA_ENR_LOW_DEFAULT  = 1.87    # wt% U-235
GA_ENR_HIGH_DEFAULT = 2.53    # wt% U-235
GA_N_RODS_SIDE_DEFAULT = 6



# Geometry helpers  

def make_sym_index(n_rods_side: int) -> List[Tuple[int, int]]:
    '''
    Build the list of unique (row, col) positions under 1/2-diagonal symmetry
    for an n×n assembly.

    Length = n x (n + 1) / 2

      For BWR6x6, n=6  ==> 6 x (6+1)/2 = 21 positions
    
    '''
    return [
        (r, c)
        for r in range(n_rods_side)
        for c in range(r + 1)
    ]


def chromosome_to_enr_grid(
    chrom: np.ndarray,
    sym_index: List[Tuple[int, int]],
    n_rods_side: int,
    enr_low:  float = GA_ENR_LOW_DEFAULT,
    enr_high: float = GA_ENR_HIGH_DEFAULT,
) -> np.ndarray:
    
    #Decodes a binary chromosome to a full N×N enrichment grid.

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
    enr_grid: np.ndarray,
    sym_index: List[Tuple[int, int]],
    enr_high: float = GA_ENR_HIGH_DEFAULT,
) -> np.ndarray:
    '''
    Encodes an N×N enrichment grid back to a binary chromosome.
    Reads only the lower-triangle + diagonal positions.

    '''
    chrom = np.zeros(len(sym_index), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        chrom[idx] = 1 if round(enr_grid[r, c], 4) == round(enr_high, 4) else 0
    return chrom



#************ GAConfig dataclass *******************############


@dataclass
class GAConfig:
    '''
    All parameters that can be changed in the GA enrichment-layout optimisation are given below.

   
    n_rods_side=N ––> defines the assembly size. 

    n_high_rods must be achievable with the chosen symmetry. It is enforced by:
      1. Population initialisation 
      2. Swap-mutation —> swaps between 1 and 2
      3. Crossover repair 

    fitness  (lower the better) =  = alpha_k*|k_target − k∞| + alpha_ppf*max(0, PPF − ppf_target)^2 
                                    + corner_penalty_weight*corner_violations  --------------------------------------> (if toggles are on)  
                                    + adj_penalty_weight*adj_high_rods   -------------------------------------------->(if toggles are on) 
    
    (corner penalty weight ––––> removed)
    
    '''

    # Assembly size (Changes between different assemblies) T
    n_rods_side: int = GA_N_RODS_SIDE_DEFAULT   # N  (side length)

    # Enrichment inventory 
    enr_low:     float = GA_ENR_LOW_DEFAULT
    enr_high:    float = GA_ENR_HIGH_DEFAULT
    n_high_rods: int   = 23    # high-enrichment rods in the FULL N×N grid
                               # MUST be supplied correctly for the chosen N

    # Targets 
    k_target:   float           = 1.25
    ppf_target: Optional[float] = 1.41
    alpha_ppf:  float           = 5.0
    alpha_k:    float           = 10.0

    # Physics-informed penalty & toggles
    use_corner_masking:    bool  = False
    use_adjacency_penalty: bool  = False
    #corner_penalty_weight: float = 100.0 #–––––––––––> Hardcoded, so removed
    adj_penalty_weight:    float = 10.0

    # GA hyper-parameters 
    population_size:  int   = 60
    n_generations:    int   = 80
    crossover_rate:   float = 0.85
    mutation_rate:    float = 0.15
    n_swaps_per_mut:  int   = 2
    tournament_size:  int   = 3
    elitism_count:    int   = 2
    random_seed:      int   = 99

    # OpenMC settings 
    n_particles: int = 5000
    n_inactive:  int = 25
    n_active:    int = 50

    # Smoke-test settings 
    n_particles_smoke: int = 500
    n_inactive_smoke:  int = 10
    n_active_smoke:    int = 20

    
    model_name:  str = "bwr_nxn_ga"
    results_dir: str = "results_ga"

    
    n_rods_total:  int              = field(init=False, default=0)
    n_sym_rods:    int              = field(init=False, default=0)
    n_vars:        int              = field(init=False, default=0)
    sym_index:     List[Tuple[int,int]] = field(init=False, default_factory=list)
    _n_low_rods:   int              = field(init=False, default=0)

    def __post_init__(self):
        # Geometry 
        self.n_rods_total = self.n_rods_side ** 2
        self.sym_index    = make_sym_index(self.n_rods_side)
        self.n_sym_rods   = len(self.sym_index)
        self.n_vars       = self.n_sym_rods
        self._n_low_rods  = self.n_rods_total - self.n_high_rods

        # Validation 
        assert 0 < self.crossover_rate <= 1.0, "crossover_rate must be in (0, 1]"
        assert 0 < self.mutation_rate  <= 1.0, "mutation_rate must be in (0, 1]"
        assert self.elitism_count < self.population_size, (
            "elitism_count must be smaller than population_size"
        )
        assert 0 < self.n_high_rods < self.n_rods_total, (
            f"n_high_rods={self.n_high_rods} must be in "
            f"(0, {self.n_rods_total}) for a {self.n_rods_side}×{self.n_rods_side} assembly"
        )

#**************************Helpers*************######################### 

    def decode(self, chrom: np.ndarray) -> np.ndarray:
        ###Decode chromosome –––> N×N enrichment grid
        return chromosome_to_enr_grid(
            chrom, self.sym_index, self.n_rods_side,
            self.enr_low, self.enr_high,
        )

    def encode(self, enr_grid: np.ndarray) -> np.ndarray:
        #Encode N×N enrichment grid ––> chromosome
        return enr_grid_to_chromosome(enr_grid, self.sym_index, self.enr_high)

    def fitness(self, keff: float, ppf: float = 0.0) -> float:
        #Fitness function (lower the better). (does not apply heuristic penalties)
        k_term   = self.alpha_k * abs(self.k_target - keff)
        ppf_term = 0.0
        if self.ppf_target is not None:
            excess   = max(0.0, ppf - self.ppf_target)
            ppf_term = self.alpha_ppf * excess ** 2
        return float(k_term + ppf_term)

    def average_enrichment(self, enr_grid: np.ndarray) -> float:
        #Returns the mean enrichment of the N×N enrichment grid.
        return float(enr_grid.mean())

    def count_full_high(self, chrom: np.ndarray) -> int:
        
        #Counts the high-enrichment rods in the N×N grid from a chromosome.
        chrom = np.asarray(chrom, dtype=int)
        total = 0
        for idx, (r, c) in enumerate(self.sym_index):
            multiplicity = 1 if r == c else 2
            total += int(chrom[idx]) * multiplicity
        return total
    
    #Corner Helper Function ––––––––––––––––––––> to mask the corners; heuristic constraint hard coded so that no high enriched rods occupy the corners. 
    def get_corner_indices(self) -> set:
        #Returns a set of 1D chromosome indices that map to the physical NxN corners.
        
        # If the toggle is off, return an empty set (no corners are masked)
        if not self.use_corner_masking:
            return set()
        
        N = self.n_rods_side
        physical_corners = {(0, 0), (N-1, N-1), (N-1, 0)} # In 1/2 diagonal symmetry, (0, N-1) mirrors to (N-1, 0)
        
        corner_idx = set()
        for idx, (r, c) in enumerate(self.sym_index):
            if (r, c) in physical_corners:
                corner_idx.add(idx)
                
        return corner_idx


 ############# Summary #############    
    def summary(self) -> str:
        avg_enr = (
            self.n_high_rods * self.enr_high
            + self._n_low_rods * self.enr_low
        ) / self.n_rods_total
        lines = [
            f"GAConfig — {self.model_name}",
            f"  Assembly           : {self.n_rods_side}×{self.n_rods_side} = {self.n_rods_total} pins",
            f"  Chromosome length  : {self.n_sym_rods}  "
            f"(1/2-diagonal symmetry)",
            f"  ENR_LOW            : {self.enr_low} wt%",
            f"  ENR_HIGH           : {self.enr_high} wt%",
            f"  n_high_rods        : {self.n_high_rods} / {self.n_rods_total}", 
            f"  k-inf target       : {self.k_target}  (alpha_k = {self.alpha_k})",
            f"  PPF target         : {self.ppf_target}  (alpha_ppf = {self.alpha_ppf})",
            f"  Corner masking     : {self.use_corner_masking} (hard mask — no weight)", 
            #f"(weight={self.corner_penalty_weight})",              #––––––––––––––––––––––-> hardcoded, so removed
            f"  Adjacency penalty  : {self.use_adjacency_penalty}  "
            f"(weight={self.adj_penalty_weight})",
            f"  Population size    : {self.population_size}",
            f"  Generations        : {self.n_generations}",
            f"  Crossover rate     : {self.crossover_rate}",
            f"  Mutation rate      : {self.mutation_rate}  ({self.n_swaps_per_mut} swap(s)/event)",
            f"  Tournament size    : {self.tournament_size}",
            f"  Elitism count      : {self.elitism_count}",
            f"  Random seed        : {self.random_seed}",
            f"  OpenMC particles   : {self.n_particles}  "
            f"({self.n_inactive} inactive + {self.n_active} active)",
            f"  Smoke particles    : {self.n_particles_smoke}  "
            f"({self.n_inactive_smoke} inactive + {self.n_active_smoke} active)",
            f"  Results dir        : {self.results_dir}",
        ]
        return "\n".join(lines)


