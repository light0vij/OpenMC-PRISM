"""
ga_fitness.py — Fitness evaluation for the GA for NxN 
------------------------------------------------------

Fitness convention ––> LOWER the BETTER.

    fitness = alpha_k · |k_target − k∞|
            + alpha_ppf · max(0, PPF − ppf_target)²
            + corner_penalty_weight · corner_violations   #–––––––––––––––––––>(if toggled on)
            + adj_penalty_weight    · adj_high_rods       #–––––––––––––––––>(if toggled)


The follwoing code is just pure maths. It includes:
  • NO imports from any environment package.
  • NO OpenMC calls, geometry logic, or heuristic counting.
  • NO symmetry mapping arrays.
  • NO hardcoded assembly dimensions.

"""

from __future__ import annotations

import numpy as np
from tqdm.notebook import tqdm
from typing import Callable, Optional

from .ga_config import GAConfig



#******************** Individual fitness ******************#

# Compute scalar fitness for one individual from a pre-evaluated result dict.
def fitness_single(cfg: GAConfig, res: dict) -> float:
   

    keff = float(res["keff"])
    ppf  = float(res.get("ppf", 0.0))

    # Distance from k-infinity target
    fit = cfg.alpha_k * abs(cfg.k_target - keff)

    # PPF quadratic penalty
    if cfg.ppf_target is not None and ppf > cfg.ppf_target:
        fit += cfg.alpha_ppf * (ppf - cfg.ppf_target) ** 2

    '''
    Corner masking is a hard mask, so no penalty weight needed. 
    The GA operates directly on binary chromosomes, so corners are enforced by never placing a 1-bit at those positions during initialisation and repair.
    However, in BO, which optimises over continuous [0,1]^n space and can only do corner masking via a penalty score. 
    Since GA guarantees the constraint at the chromosome level, a violation cannot occur.
    *********************************************************************************************************
    # Corner penalty — count injected by Environment, weight applied here
    if getattr(cfg, "use_corner_masking", False):
        fit += cfg.corner_penalty_weight * float(res.get("corner_violations", 0))
    '''
    
    # Adjacency penalty — count injected by Environment, weight applied here
    if getattr(cfg, "use_adjacency_penalty", False):
        fit += cfg.adj_penalty_weight * float(res.get("adj_high_rods", 0))

    return float(fit)



#***************** Population evaluation ********************###

#Determine every individual in the population.
def evaluate_population(
    cfg:         GAConfig,
    population:  np.ndarray,
    evaluate_fn: Callable[[np.ndarray], dict],
    verbose:     bool = True,
    generation:  Optional[int] = None,
) -> dict:
    pop_size     = len(population)
    keff_arr     = np.zeros(pop_size)
    keff_std_arr = np.zeros(pop_size)
    ppf_arr      = np.zeros(pop_size)
    fitness_arr  = np.zeros(pop_size)

    gen_str = f"Gen {generation:3d}" if generation is not None else "Eval"

    for i, chrom in enumerate(
        tqdm(population, desc=gen_str, disable=not verbose, leave=False)
    ):
        res  = evaluate_fn(chrom)
        k    = float(res["keff"])
        kstd = float(res.get("keff_std", 0.0))
        ppf  = float(res.get("ppf", 0.0))
        fit  = fitness_single(cfg, res)

        keff_arr[i]     = k
        keff_std_arr[i] = kstd
        ppf_arr[i]      = ppf
        fitness_arr[i]  = fit

    feasible = (
        (ppf_arr <= cfg.ppf_target)
        if cfg.ppf_target is not None
        else np.ones(pop_size, bool)
    )

    return dict(
        keff     = keff_arr,
        keff_std = keff_std_arr,
        ppf      = ppf_arr,
        fitness  = fitness_arr,
        feasible = feasible,
    )

#Determine a single chromosome. Useful for post-hoc validation runs.
def evaluate_single(
    cfg:         GAConfig,
    chrom:       np.ndarray,
    evaluate_fn: Callable[[np.ndarray], dict],
) -> dict:
    
    res  = evaluate_fn(chrom)
    k    = float(res["keff"])
    kstd = float(res.get("keff_std", 0.0))
    ppf  = float(res.get("ppf", 0.0))
    fit  = fitness_single(cfg, res)
    feas = (ppf <= cfg.ppf_target) if cfg.ppf_target is not None else True
    return dict(keff=k, keff_std=kstd, ppf=ppf, fitness=fit, feasible=feas)



#************* Helper *************************#

#Mean pairwise Hamming distance across the population, normalised to [0, 1]. (Works for any chromosome lengt)

def hamming_diversity(population: np.ndarray) -> float:

    pop_size, n_bits = population.shape
    if pop_size < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            total += np.sum(population[i] != population[j])
            count += 1
    return (total / count) / n_bits if count > 0 else 0.0
