"""
mapping.py — Chromosome ↔ enrichment grid symmetry mapping  (NxN generalised)
------------------------------------------------------------------------------

All geometry is derived from BWR_N (the side length) at runtime.


Functions:- 
        make_sym_index(n_rods_side) - Build the list of unique (row, col) positions under 1/2-diagonal symmetry.

        apply_symmetry_to_grid(chrom, n_rods_side, enr_low, enr_high) - Decode a binary chromosome of length N*(N+1)/2 to a full N×N grid.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np



# Default constants 

ENR_LOW  = 1.87   # wt% U-235
ENR_HIGH = 2.53   # wt% U-235



# Symmetric index builder


def make_sym_index(n_rods_side: int) -> List[Tuple[int, int]]:
    """
    Builds the list of unique (row, col) positions under 1/2-diagonal symmetry
    for an n_rods_side × n_rods_side assembly.
    """
    return [
        (r, c)
        for r in range(n_rods_side)
        for c in range(r + 1)
    ]



# Applying symmetry to grid


def apply_symmetry_to_grid(
    chrom:       np.ndarray,
    n_rods_side: int,
    enr_low:     float = ENR_LOW,
    enr_high:    float = ENR_HIGH,
) -> np.ndarray:
    """
    Decode a binary chromosome to a full N×N enrichment grid.

    """
    sym_index = make_sym_index(n_rods_side)
    n_sym     = len(sym_index)

    chrom = np.asarray(chrom, dtype=int).ravel()
    if len(chrom) != n_sym:
        raise ValueError(
            f"Expected {n_sym} chromosome bits for a "
            f"{n_rods_side}×{n_rods_side} assembly, got {len(chrom)}"
        )

    grid = np.zeros((n_rods_side, n_rods_side), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        grid[r, c] = chrom[idx]
        grid[c, r] = chrom[idx]   # mirror — no-op on diagonal (r == c)

    return np.where(grid == 1, enr_high, enr_low)

'''

#**********************Centralisation of Geometry*********************************



**********************************************************************************
Architectural change


The fucntions 'chromosome_to_enr_grid and  'enr_grid_to_chromosome' are copied in both GA and RL packges rather than the openmc_bo/ga/Rl packages importing from here as a centralised version for the time being.  
This to prevent confusion, Import Cascades (prevent accidental `openmc` import triggers during config initialization), Standalone Portability (completely independent openmc_bo/ga/rl packages), Safe Iteration.

So now the openmc_bo/ga/rl packages interact only via 'evaluate_for_ga/bo' from the environment. 

Problem with this DRY Violation: Future changes to the core geometry definition (e.g., altering symmetry from 
1/2-diagonal to 8-fold, or changing tie-breaking rules) has be manually updated across the GA, RL, and BO packages. 


********************************************************************************************




# Chromosome -> grid (explicit sym_index — used by Config.decode())
def chromosome_to_enr_grid(
    chrom:       np.ndarray,
    sym_index:   List[Tuple[int, int]],
    n_rods_side: int,
    enr_low:     float = ENR_LOW,
    enr_high:    float = ENR_HIGH,
) -> np.ndarray:
    """
    Decode a binary chromosome to a full N×N enrichment grid.

    (Common for GAConfig, RLConfig, RunConfig)
    """
    chrom = np.asarray(chrom, dtype=int).ravel()
    n_sym = len(sym_index)
    if len(chrom) != n_sym:
        raise ValueError(
            f"Expected {n_sym} chromosome bits for a "
            f"{n_rods_side}×{n_rods_side} assembly, got {len(chrom)}"
        )

    grid = np.zeros((n_rods_side, n_rods_side), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        grid[r, c] = chrom[idx]
        grid[c, r] = chrom[idx]   # mirror — no-op on diagonal (r == c)

    return np.where(grid == 1, enr_high, enr_low)



# Grid -> chromosome (inverse of chromosome_to_enr_grid)
def enr_grid_to_chromosome(
    enr_grid:  np.ndarray,
    sym_index: List[Tuple[int, int]],
    enr_high:  float = ENR_HIGH,
) -> np.ndarray:
    """
    Encode an N×N enrichment grid back to a binary chromosome.
    (Common for GAConfig, RLConfig, RunConfig)
    """
    chrom = np.zeros(len(sym_index), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        chrom[idx] = 1 if round(float(enr_grid[r, c]), 4) == round(enr_high, 4) else 0
    return chrom
'''
