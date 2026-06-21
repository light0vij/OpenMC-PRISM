"""
ga_population.py — Physics-valid population initialisation for NxN
-------------------------------------------------------------------

All geometry (chromosome length, diagonal/off-diagonal split, rod counts) is read from cfg at runtime. 


All (d, o) pairs satisfying the following conditions are listed. 
    d + 2·o = n_high_rods   (0 ≤ d ≤ n_diag,  0 ≤ o ≤ n_offdiag)

where:
    n_diag    = n_rods_side         ––––––––––––> diagonal positions
    n_offdiag = n_sym_rods - n_diag  ––––––––––> off-diagonal positions

For each of the valid pair, a chromosome is constructed by randomly selecting a d diagonal and an o off-diagonal positions to be 1. 
iterate through all valid pairs to ensure population diversity.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .ga_config import GAConfig


#*************** Helpers ****************####


def _diag_offdiag_indices(cfg):
    #Corner Masking Code
    last    = cfg.n_rods_side - 1
    corners = {(0, 0), (last, last), (last, 0)}  # lower-triangle corners only
    diag    = [
        i for i, (r, c) in enumerate(cfg.sym_index)
        if r == c and (not cfg.use_corner_masking or (r, c) not in corners)
    ]
    offdiag = [
        i for i, (r, c) in enumerate(cfg.sym_index)
        if r != c and (not cfg.use_corner_masking or (r, c) not in corners)
    ]
    return diag, offdiag



def _valid_do_pairs(
    n_high: int,
    n_diag: int,
    n_offdiag: int,
) -> List[Tuple[int, int]]:
    
    #Returns all (d, o) pairs satisfying  d + 2·o = n_high.
    pairs = []
    for d in range(n_diag + 1):
        rem = n_high - d
        if rem < 0 or rem % 2 != 0:
            continue
        o = rem // 2
        if o <= n_offdiag:
            pairs.append((d, o))
    return pairs


def _make_chromosome(        #––––––––––––> Create one random chromosome with exactly d diagonal bits and o off-diagonal bits set to 1. Everything else 0.
    d: int,
    o: int,
    diag_idx:    List[int],
    offdiag_idx: List[int],
    n_sym_rods:  int,
    rng: np.random.Generator,
) -> np.ndarray:
    chrom = np.zeros(n_sym_rods, dtype=int)
    chrom[rng.choice(diag_idx,    size=d, replace=False)] = 1
    chrom[rng.choice(offdiag_idx, size=o, replace=False)] = 1
    return chrom


def count_full_high(chrom: np.ndarray, cfg: GAConfig) -> int:
    #Count HIGH-enrichment rods in the full N×N grid from a chromosome.
    chrom = np.asarray(chrom, dtype=int)
    total = 0
    for idx, (r, c) in enumerate(cfg.sym_index):
        total += int(chrom[idx]) * (1 if r == c else 2)
    return total



#***************** Public  - Everything below will be called by other modules in the openmc_ga *****************************#


def init_population(cfg: GAConfig, verbose: bool = True) -> np.ndarray:
    '''
    Generate an initial population of valid chromosomes. where each individual satisfies, 'count_full_high(chrom, cfg) == cfg.n_high_rods'  
    Returns ––> population int array
    '''
    rng              = np.random.default_rng(cfg.random_seed)
    diag, offdiag    = _diag_offdiag_indices(cfg)
    n_diag           = len(diag)
    n_offdiag        = len(offdiag)
    pairs            = _valid_do_pairs(cfg.n_high_rods, n_diag, n_offdiag)

    if not pairs:
        raise ValueError(                  #–––––––––––––––––––––––––––––––––––> Checks whether the inventory can be achieved with the symmetry.
            f"No valid (d, o) pairs found for n_high_rods={cfg.n_high_rods} "
            f"in a {cfg.n_rods_side}×{cfg.n_rods_side} assembly "
            f"(n_diag={n_diag}, n_offdiag={n_offdiag}). "
        )

    population = np.zeros((cfg.population_size, cfg.n_sym_rods), dtype=int)
    for i in range(cfg.population_size):
        d, o = pairs[i % len(pairs)]
        population[i] = _make_chromosome(d, o, diag, offdiag, cfg.n_sym_rods, rng)

    
    counts = [count_full_high(c, cfg) for c in population]
    assert all(c == cfg.n_high_rods for c in counts), (
        f"Inventory violation in init_population! Unique counts: {set(counts)}"
    )

    if verbose:
        avg_enr = (
            cfg.n_high_rods * cfg.enr_high
            + (cfg.n_rods_total - cfg.n_high_rods) * cfg.enr_low
        ) / cfg.n_rods_total
        print(f"Population initialised  [{cfg.n_rods_side}×{cfg.n_rods_side} assembly]")
        print(f"  {cfg.population_size} individuals  ×  {cfg.n_sym_rods} bits/chromosome")
        print(f"  Inventory : {cfg.n_high_rods} × {cfg.enr_high}%  +  "
              f"{cfg.n_rods_total - cfg.n_high_rods} × {cfg.enr_low}%  "
              f"→ avg {avg_enr:.4f} wt%")
        print(f"  Valid (d,o) pairs : {pairs}")
        print(f"  Constraints OK    : OK")

    return population




def make_random_individual(
    cfg: GAConfig, 
    rng: np.random.Generator
) -> np.ndarray:
    '''
    Creates a single random valid chromosome with exactly `n_high_rods`.
    
    Enforces the physics-informed corner mask so that Generation 0 
    never spawns with illegal high-enrichment corners.
    '''
    chrom = np.zeros(cfg.n_sym_rods, dtype=int)
    
    
    all_idx = set(range(cfg.n_sym_rods))
    corner_idx = cfg.get_corner_indices()
    valid_idx = list(all_idx - corner_idx)
    
   
    while True:
        current = cfg.count_full_high(chrom)
        
        if current == cfg.n_high_rods:
            break  
            
        
        available_zeros = [z for z in valid_idx if chrom[z] == 0]
        
        if not available_zeros:
            break # Failsafe
            
        
        choice = rng.choice(available_zeros)
        chrom[choice] = 1
        
        
        if cfg.count_full_high(chrom) > cfg.n_high_rods:
            chrom[choice] = 0  
            
    return chrom

