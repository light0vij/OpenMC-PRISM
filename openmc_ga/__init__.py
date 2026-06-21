"""
openmc_ga — Genetic Algorithm framework for OpenMC models for BWR NxN Reactor 
------------------------------------------------------------------------------

"""

from .ga_config import (
    GAConfig,
    make_sym_index,
    chromosome_to_enr_grid,
    enr_grid_to_chromosome,
    GA_ENR_LOW_DEFAULT,
    GA_ENR_HIGH_DEFAULT,
    GA_N_RODS_SIDE_DEFAULT,
)
from .ga_population import init_population, make_random_individual, count_full_high
from .ga_fitness    import (
    evaluate_population,
    evaluate_single,
    fitness_single,
    hamming_diversity,
)
from .ga_operators  import crossover, swap_mutate, repair_chromosome, breed
from .ga_loop       import run_ga, run_ga_smoke_test
from .ga_results    import summarise_ga_results, plot_ga_convergence

__all__ = [
    # Config & helpers
    "GAConfig",
    "make_sym_index",
    "chromosome_to_enr_grid",
    "enr_grid_to_chromosome",
    "GA_ENR_LOW_DEFAULT",
    "GA_ENR_HIGH_DEFAULT",
    "GA_N_RODS_SIDE_DEFAULT",
    # Population
    "init_population",
    "make_random_individual",
    "count_full_high",
    # Fitness
    "evaluate_population",
    "evaluate_single",
    "fitness_single",
    "hamming_diversity",
    # Operators
    "crossover",
    "swap_mutate",
    "repair_chromosome",
    "breed",
    # Loop
    "run_ga",
    "run_ga_smoke_test",
    # Results & visualisation
    "summarise_ga_results",
    "plot_ga_convergence",
]



