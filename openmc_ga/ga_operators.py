"""
ga_operators.py — Crossover and swap-mutation operators for NxN 
-----------------------------------------------------------------

All operators are inventory-safe: the number of HIGH-enrichment rods in the full N×N grid is NEVER changed by any operator.

The chromosome length (n_sym_rods) is read from cfg at runtime 

Mutation:-
  swap_mutate — select n_swaps_per_mut random (1→0, 0→1) pairs and swap them. 
              Inventory ––> ALWAYS conserved.
              Standard bit-flip mutation ––> deliberately excluded.

"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from .ga_config     import GAConfig
from .ga_population import count_full_high



#************ Crossover ********************###


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    cfg:     GAConfig,
    rng:     np.random.Generator,
    method:  str = "two_point",
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Produce two children from two parents.
    The crossover fires only with probability cfg.crossover_rate.
    If it does not fire, children are clones of the parents.
    After crossover, repair_chromosome() restores the inventory if needed.
    '''
    
    if rng.random() > cfg.crossover_rate:
        return parent1.copy(), parent2.copy()

    n = cfg.n_sym_rods

    if method == "single_point":
        child1, child2 = _single_point(parent1, parent2, n, rng)
    elif method == "two_point":
        child1, child2 = _two_point(parent1, parent2, n, rng)
    elif method == "uniform":
        child1, child2 = _uniform(parent1, parent2, n, rng)
    else:
        raise ValueError(f"Unknown crossover method: '{method}'")

    child1 = repair_chromosome(child1, cfg, rng)
    child2 = repair_chromosome(child2, cfg, rng)
    return child1, child2

#Single-point crossover
def _single_point(
    p1: np.ndarray,
    p2: np.ndarray,
    n:  int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pt = rng.integers(1, n)
    c1 = np.concatenate([p1[:pt], p2[pt:]])
    c2 = np.concatenate([p2[:pt], p1[pt:]])
    return c1, c2

#Two-point crossover
def _two_point(
    p1: np.ndarray,
    p2: np.ndarray,
    n:  int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = sorted(rng.choice(n - 1, size=2, replace=False) + 1)
    a, b = pts
    c1 = np.concatenate([p1[:a], p2[a:b], p1[b:]])
    c2 = np.concatenate([p2[:a], p1[a:b], p2[b:]])
    return c1, c2

#Uniform crossover
def _uniform(
    p1: np.ndarray,
    p2: np.ndarray,
    n:  int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.integers(0, 2, size=n).astype(bool)
    c1   = np.where(mask, p1, p2)
    c2   = np.where(mask, p2, p1)
    return c1, c2



#************************ Repair **********************


def repair_chromosome(
    chrom:         np.ndarray,
    cfg:           GAConfig,
    rng:           np.random.Generator,
) -> np.ndarray:
    '''
    Restores the exact required inventory of high-enrichment rods (1-bits) after the crossover operator potentially alters the total count.

    Algorithm:- 
    Here the function calculates the current number of high-enrichment rods in the 
    full N x N assembly using the symmetry map.
            -  Phase 1 (Too Many): If the count exceeds the target, it randomly selects existing 1-bits and flips them to 0 until the target is reached.
            -  Phase 2 (Too Few): If the count is below the target, vice versa until the target is reached. 

    Physics-Informed Constraint (Hard Masking):
    In a BWR assembly, placing high-enrichment fuel in the extreme corners can lead to thermal power peaking due to 
    the adjacent water gaps. 
    
    So, rather than depending on a "soft" penalty to teach the GA to avoid corners, a "hard" heuristic constraint is applied through this funtion.
    
    (Heuristic Constraint used here - When the algorithm needs to flip 0s to 1s (Phase 2), it dynamically queries the configuration for the 
    physical corner indices. It mathematically filters these corners out of the 'valid_zeros' pool. Therefore, it is impossible for the repair 
    mechanism to accidentally spawn a high- enrichment pin in a restricted corner zone.)

    '''
    chrom   = chrom.copy()
    current = cfg.count_full_high(chrom)
    target  = cfg.n_high_rods

    corner_idx = cfg.get_corner_indices()

    # PHASE 1: Too many highs. 
    # (No need to filter corners here, turning a corner to 0 is good!)
    while current > target:
        ones = np.where(chrom == 1)[0]
        if len(ones) == 0:
            break
        flip_idx = rng.choice(ones)
        chrom[flip_idx] = 0
        current = cfg.count_full_high(chrom)

    # PHASE 2: Too few highs. 
    # (MUST filter corners so that they wont be filled accidentally)
    while current < target:
        zeros = np.where(chrom == 0)[0]
        valid_zeros = np.array([z for z in zeros if z not in corner_idx])

        # Safe break if we somehow run out of valid spots
        if len(valid_zeros) == 0:
            break
            
        flip_idx = rng.choice(valid_zeros)
        chrom[flip_idx] = 1
        current = cfg.count_full_high(chrom)

    return chrom

    

#**************** Swap mutation *********************####


def swap_mutate(
    chrom: np.ndarray,
    cfg:   GAConfig,
    rng:   np.random.Generator,
) -> np.ndarray:
    '''
    Swap-mutation: randomly exchange one HIGH bit and one low bit.

    The ONLY mutation operator used.
   (Standard bit-flip is deliberately excluded because it violates the fixed-count constraint.)

    Fires with probability cfg.mutation_rate. 
    When it fires, performs cfg.n_swaps_per_mut independent 1<->0 swaps.

    Enforces corner masking by excluding corner indices from the 'zeros' pool ––––––––––––––––––––––> Physics informed Heuristic constraint
    '''
    if rng.random() > cfg.mutation_rate:
        return chrom.copy()

    chrom = chrom.copy()

    corner_idx = cfg.get_corner_indices()
    
    for _ in range(cfg.n_swaps_per_mut):
        ones  = np.where(chrom == 1)[0]
        zeros = np.where(chrom == 0)[0]

        valid_zeros = np.array([z for z in zeros if z not in corner_idx])
        

        if len(ones) == 0 or len(valid_zeros) == 0:
            break
        
        i = rng.choice(ones)
        j = rng.choice(valid_zeros)
        chrom[i] = 0
        chrom[j] = 1

    return chrom


# Convenience–––>apply crossover + mutation in one call

def breed(
    parent1:          np.ndarray,
    parent2:          np.ndarray,
    cfg:              GAConfig,
    rng:              np.random.Generator,
    crossover_method: str = "two_point",
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Full breeding step: crossover then mutation for both children.

    '''
    c1, c2 = crossover(parent1, parent2, cfg, rng, method=crossover_method)
    c1     = swap_mutate(c1, cfg, rng)
    c2     = swap_mutate(c2, cfg, rng)
    return c1, c2
