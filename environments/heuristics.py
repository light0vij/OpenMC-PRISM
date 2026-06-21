"""
heuristics.py — Physics-informed geometry counting for the BWR nxn assembly
-----------------------------------------------------------------------------

This module evaluates physical constraints to bridge black-box AI algorithms with classical nuclear engineering. Following the methodology of Radaideh 
et al. (2021), it utilizes Reward Shaping to restrict the AI's search space to physically viable designs, accelerating convergence.

Heuristics Implemented:
1. Corner Masking: High-enrichment rods in outer corners face wide water gaps, causing massive thermal flux and PPF spikes. Masking prevents this.
2. Adjacency Penalties: Clustering highly reactive pins creates dangerous power hot-spots. Penalizing contiguous pairs enforces a flatter, safer flux profile.

Architecture & Comparative Studies:
To maintain a strict "plug-and-play" architecture, this module ONLY counts violations (returning integers). Penalty weighting is handled entirely by the 
Agent (GAConfig/RunConfig). Toggling these weights in the notebook enables direct Comparative Studies between a baseline "Unconstrained AI" and a 
"Physics-Informed AI".

Reference:
Radaideh, M. I., et al. (2021). "Physics-informed reinforcement learning optimization of nuclear assembly design." Nuclear Engineering and Design, 
372, 110966. DOI: 10.1016/j.nucengdes.2020.110966

"""

from __future__ import annotations

import numpy as np

from .mapping import ENR_HIGH



# check_corner_violations

def check_corner_violations(enr_grid: np.ndarray) -> int:
    """
    Count how many of the four extreme assembly corners contain high-
    enrichment fuel.

    Corners are surrounded by three water reflectors (two assembly gaps + one corner gap). 
    The higher moderation greatly increases the local thermal flux, making a high-enrichment corner rod overpowered and thermally unsafe.

    Returns
    -------
    n_violations : int in {0, 1, 2, 3, 4}
         ↳ Number of corner positions occupied by high-enrichment rods.
           0 = perfect. 
    """
    n_side    = enr_grid.shape[0]
    last      = n_side - 1
    corners   = [(0, 0), (0, last), (last, 0), (last, last)]
    n_violations = sum(
        1 for (r, c) in corners
        if round(float(enr_grid[r, c]), 4) == round(ENR_HIGH, 4)
    )
    return int(n_violations)



# count_adjacent_high_rods


def count_adjacent_high_rods(enr_grid: np.ndarray) -> int:
    """
    Count the number of directly-adjacent (horizontal or vertical) pairs of high-enrichment rods.

    Adjacent high-enrichment rods create a localised power cluster. The thermal-hydraulic safety margin (DNB ratio) is lowest in dense high-
    power zones, so a good layout spreads the high-enrichment rods in a near-checkerboard pattern.

    Algorithm: Fast numpy array slicing: compare each cell to its right/bottom neighbour using vectorised equality tests. Avoids an explicit double loop.
    
    """
    hi = (np.round(enr_grid, 4) == round(ENR_HIGH, 4))

    
    h_pairs = int(np.sum(hi[:, :-1] & hi[:, 1:]))     # Horizontal touches: cell [r, c] and [r, c+1] are both HIGH

    
    v_pairs = int(np.sum(hi[:-1, :] & hi[1:, :]))     # Vertical touches: cell [r, c] and [r+1, c] are both HIGH

    return h_pairs + v_pairs
