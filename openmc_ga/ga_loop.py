"""
ga_loop.py — Main Genetic Algorithm loop  (NxN generalised)
------------------------------------------------------------
All chromosome dimensions are read from cfg at runtime.
  
Pure algorithm. 

The evaluate_fn (from environments.env.py) called at runtime via functools.partial. 

"""

from __future__ import annotations
import sys

import numpy as np
from tqdm import tqdm
from typing import Callable, Optional

from .ga_config     import GAConfig
from .ga_population import init_population, make_random_individual
from .ga_fitness    import evaluate_population, evaluate_single, fitness_single, hamming_diversity
from .ga_operators  import breed



#********************* Selection *********************#####

#k-way tournament selection (minimisation )
def tournament_select(
    fitness: np.ndarray,
    k:       int,
    rng:     np.random.Generator,
) -> int:
    contestants = rng.choice(len(fitness), size=k, replace=False)
    return int(contestants[np.argmin(fitness[contestants])])

#Fitness-proportionate roulette selection (minimisation via inversion)
def roulette_select(
    fitness: np.ndarray,
    rng:     np.random.Generator,
) -> int:
    inv   = 1.0 / (fitness + 1e-12)
    probs = inv / inv.sum()
    return int(rng.choice(len(fitness), p=probs))



###############*****************  GA loop *******************#####################

# Runs the full Genetic Algorithm for any NxN BWR assembly
def run_ga(
    cfg:                   GAConfig,
    evaluate_fn:           Callable[[np.ndarray], dict],
    verbose:               bool  = True,
    selection:             str   = "tournament",
    crossover_method:      str   = "two_point",
    convergence_tol:       float = 1e-6,
    convergence_patience:  int   = 15,
    inject_diversity_every: Optional[int] = None,
    inject_fraction:       float = 0.10,
) -> dict:
    rng = np.random.default_rng(cfg.random_seed)

    # Population initialisation
    population = init_population(cfg, verbose=verbose)

    if verbose:
        print(f"\n{cfg.summary()}\n")
        print("=" * 70)
        print(f"  GENETIC ALGORITHM — {cfg.model_name}")
        print(f"  Assembly   : {cfg.n_rods_side}×{cfg.n_rods_side}  "
              f"({cfg.n_sym_rods} sym positions)")
        print(f"  Selection  : {selection}   Crossover : {crossover_method}")
        print("═" * 70)

    
    all_chromosomes  = []
    all_keff         = []
    all_keff_std     = []
    all_ppf          = []
    all_fitness      = []

    gen_best_fitness = []
    gen_mean_fitness = []
    gen_diversity    = []

    hof_chromosome   = None
    hof_keff         = np.inf
    hof_keff_std     = 0.0
    hof_ppf          = np.inf
    hof_fitness      = np.inf

    stagnant_gens     = 0
    converged         = False
    n_generations_run = 0
    n_evaluations     = 0

    # Generational loop 
    gen_iter = tqdm(range(cfg.n_generations),
                    desc="Generations", disable=not verbose, file=sys.stdout)

    for gen in gen_iter:
        n_generations_run = gen + 1

        gen_results  = evaluate_population(
            cfg, population, evaluate_fn,
            verbose=verbose, generation=gen + 1,
        )
        keff_gen     = gen_results["keff"]
        keff_std_gen = gen_results["keff_std"]
        ppf_gen      = gen_results["ppf"]
        fit_gen      = gen_results["fitness"]

        n_evaluations += cfg.population_size

        # Archive
        for i in range(cfg.population_size):
            all_chromosomes.append(population[i].copy())
            all_keff.append(keff_gen[i])
            all_keff_std.append(keff_std_gen[i])
            all_ppf.append(ppf_gen[i])
            all_fitness.append(fit_gen[i])

        # Hall-of-Fame update
        best_idx_gen = int(np.argmin(fit_gen))
        if fit_gen[best_idx_gen] < hof_fitness:
            hof_fitness    = float(fit_gen[best_idx_gen])
            hof_keff       = float(keff_gen[best_idx_gen])
            hof_keff_std   = float(keff_std_gen[best_idx_gen])
            hof_ppf        = float(ppf_gen[best_idx_gen])
            hof_chromosome = population[best_idx_gen].copy()

        gen_best_fitness.append(float(np.min(fit_gen)))
        gen_mean_fitness.append(float(np.mean(fit_gen)))
        gen_diversity.append(hamming_diversity(population))

        if verbose:
            feas_count = gen_results["feasible"].sum()
            std_str    = (f" ±{keff_std_gen[best_idx_gen]:.5f}"
                          if keff_std_gen[best_idx_gen] > 0 else "")
            ppf_str    = (f"  PPF={ppf_gen[best_idx_gen]:.4f}"
                          if cfg.ppf_target is not None else "")
            gen_iter.write(
                f"  Gen {gen+1:3d}/{cfg.n_generations}:  "
                f"best_fit={fit_gen[best_idx_gen]:.5f}  "
                f"k∞={keff_gen[best_idx_gen]:.5f}{std_str}"
                f"{ppf_str}  "
                f"feas={feas_count}/{cfg.population_size}  "
                f"div={gen_diversity[-1]:.3f}  "
                f"[HOF fit={hof_fitness:.5f}]"
            )

        # Convergence check 
        if gen >= 1:
            delta = abs(gen_best_fitness[-2] - gen_best_fitness[-1])
            stagnant_gens = stagnant_gens + 1 if delta < convergence_tol else 0

        if stagnant_gens >= convergence_patience:
            converged = True
            if verbose:
                print(f"\n  Early stopping — no improvement for "
                      f"{convergence_patience} generations.")
            break

        # Diversity injection 
        if (inject_diversity_every is not None
                and (gen + 1) % inject_diversity_every == 0):
            n_inject = max(1, int(inject_fraction * cfg.population_size))
            if verbose:
                print(f"  [Gen {gen+1}] Injecting {n_inject} random individuals.")
            sort_idx = np.argsort(fit_gen)[::-1]
            for k in range(n_inject):
                population[sort_idx[k]] = make_random_individual(cfg, rng)

        # Next generation building
        population = _next_generation(
            population, fit_gen, cfg, rng,
            selection, crossover_method, hof_chromosome,
        )

    # Final population evaluation
    if verbose:
        print(f"\n  Final population evaluation-")
    final_results = evaluate_population(
        cfg, population, evaluate_fn,
        verbose=verbose, generation=n_generations_run,
    )
    n_evaluations += cfg.population_size

    for i in range(cfg.population_size):
        all_chromosomes.append(population[i].copy())
        all_keff.append(final_results["keff"][i])
        all_keff_std.append(final_results["keff_std"][i])
        all_ppf.append(final_results["ppf"][i])
        all_fitness.append(final_results["fitness"][i])

    best_idx_final = int(np.argmin(final_results["fitness"]))
    if final_results["fitness"][best_idx_final] < hof_fitness:
        hof_fitness    = float(final_results["fitness"][best_idx_final])
        hof_keff       = float(final_results["keff"][best_idx_final])
        hof_keff_std   = float(final_results["keff_std"][best_idx_final])
        hof_ppf        = float(final_results["ppf"][best_idx_final])
        hof_chromosome = population[best_idx_final].copy()

    if verbose:
        _print_ga_summary(cfg, hof_keff, hof_keff_std, hof_ppf,
                          hof_fitness, hof_chromosome, n_evaluations)

    return dict(
        population        = population,
        all_chromosomes   = np.array(all_chromosomes),
        all_keff          = np.array(all_keff),
        all_keff_std      = np.array(all_keff_std),
        all_ppf           = np.array(all_ppf),
        all_fitness       = np.array(all_fitness),
        gen_best_fitness  = np.array(gen_best_fitness),
        gen_mean_fitness  = np.array(gen_mean_fitness),
        gen_diversity     = np.array(gen_diversity),
        hof_chromosome    = hof_chromosome,
        hof_keff          = hof_keff,
        hof_keff_std      = hof_keff_std,
        hof_ppf           = hof_ppf,
        hof_fitness       = hof_fitness,
        n_evaluations     = n_evaluations,
        converged         = converged,
        n_generations_run = n_generations_run,
    )



#*************** Smoke test ********************###################

#Quick low-particle verification run
def run_ga_smoke_test(
    cfg:         GAConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    smoke_chrom: Optional[np.ndarray] = None,
    smoke_label: Optional[str]        = None,
) -> dict:
    rng = np.random.default_rng(cfg.random_seed)
    if smoke_chrom is None:
        smoke_chrom = make_random_individual(cfg, rng)

    _prod = (cfg.n_particles, cfg.n_inactive, cfg.n_active)
    cfg.n_particles = cfg.n_particles_smoke
    cfg.n_inactive  = cfg.n_inactive_smoke
    cfg.n_active    = cfg.n_active_smoke

    label    = smoke_label or cfg.model_name
    enr_grid = cfg.decode(smoke_chrom)
    avg_enr  = enr_grid.mean()
    n_high   = int((np.round(enr_grid, 4) == round(cfg.enr_high, 4)).sum())

    print("═" * 70)
    print(f"  GA SMOKE TEST — {label}")
    print(f"  Assembly : {cfg.n_rods_side}×{cfg.n_rods_side}  "
          f"({cfg.n_sym_rods} sym positions)")
    print("═" * 70)
    print(f"  Chromosome (first 10): {smoke_chrom[:10]} …")
    print(f"  Decoded grid : {n_high} × {cfg.enr_high}%  +  "
          f"{cfg.n_rods_total - n_high} × {cfg.enr_low}%  "
          f"(avg {avg_enr:.4f} wt%)")
    print(f"  Particles    : {cfg.n_particles}  "
          f"({cfg.n_inactive} inactive + {cfg.n_active} active)")
    print()

    smoke_res = evaluate_fn(smoke_chrom)

    k    = float(smoke_res["keff"])
    kstd = float(smoke_res.get("keff_std", 0.0))
    ppf  = float(smoke_res.get("ppf", 0.0))
    fit  = fitness_single(cfg, smoke_res)

    std_str = f" ± {kstd:.5f}" if kstd > 0 else ""
    print(f"  k∞       = {k:.5f}{std_str}   (target = {cfg.k_target})")
    print(f"  |Δk∞|    = {abs(cfg.k_target - k):.5f}")
    if cfg.ppf_target is not None:
        feas = " feasible" if ppf <= cfg.ppf_target else " NOt feasible"
        print(f"  PPF      = {ppf:.4f}   (limit ≤ {cfg.ppf_target})  {feas}")
    print(f"  Fitness  = {fit:.5f}  (lower is better; 0 = perfect)")

    cfg.n_particles, cfg.n_inactive, cfg.n_active = _prod
    smoke_res["fitness"] = fit
    print("═" * 64)
    return smoke_res



#*****************88 Helpers ************************####

# Elitism + selection + crossover + mutation ––––> new population
def _next_generation(
    population:      np.ndarray,
    fitness:         np.ndarray,
    cfg:             GAConfig,
    rng:             np.random.Generator,
    selection:       str,
    crossover_method: str,
    hof_chromosome:  Optional[np.ndarray],
) -> np.ndarray:
    pop_size   = cfg.population_size
    new_pop    = np.zeros_like(population)
    sorted_idx = np.argsort(fitness)

    if hof_chromosome is not None:
        new_pop[0]     = hof_chromosome.copy()
        n_elite_copy   = min(cfg.elitism_count, pop_size - 1)
        for k in range(n_elite_copy):
            if 1 + k < pop_size:
                new_pop[1 + k] = population[sorted_idx[k]].copy()
        filled = 1 + n_elite_copy
    else:
        n_elite_copy = min(cfg.elitism_count, pop_size)
        for k in range(n_elite_copy):
            new_pop[k] = population[sorted_idx[k]].copy()
        filled = n_elite_copy

    slot = filled
    while slot < pop_size:
        if selection == "tournament":
            p1 = tournament_select(fitness, cfg.tournament_size, rng)
            p2 = tournament_select(fitness, cfg.tournament_size, rng)
        elif selection == "roulette":
            p1 = roulette_select(fitness, rng)
            p2 = roulette_select(fitness, rng)
        else:
            raise ValueError(f"Unknown selection method: '{selection}'")

        c1, c2 = breed(population[p1], population[p2], cfg, rng, crossover_method)
        new_pop[slot] = c1;  slot += 1
        if slot < pop_size:
            new_pop[slot] = c2;  slot += 1

    return new_pop

#--------------------------------------------------------------------------------------------------------------
def _print_ga_summary(cfg, keff, keff_std, ppf, fitness, chrom, n_evals):
    std_str = f" ± {keff_std:.5f}" if keff_std > 0 else ""
    print("\n" + "═" * 64)
    print(f"  GA SUMMARY — {cfg.model_name}  "
          f"[{cfg.n_rods_side}×{cfg.n_rods_side}]")
    print("═" * 70)
    print(f"  Best k∞            : {keff:.5f}{std_str}")
    print(f"  k∞ target          : {cfg.k_target}")
    print(f"  |Δk∞|              : {abs(cfg.k_target - keff):.5f}  "
          f"({abs(cfg.k_target - keff)*1e5:.1f} pcm)")
    if cfg.ppf_target is not None:
        sat = "SATISFIED" if ppf <= cfg.ppf_target else "NOT MET"
        print(f"  PPF                : {ppf:.4f}  "
              f"(limit ≤ {cfg.ppf_target})  {sat}")
    print(f"  Best fitness       : {fitness:.6f}")
    print(f"  Total OpenMC calls : {n_evals}")
    print(f"  Best chromosome    : {chrom}")
    print("═" * 70)
