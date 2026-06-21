"""
openmc_bo — Bayesian Optimisation framework for OpenMC models
-------------------------------------------------------------


Optimises the spatial arrangement of binary fuel enrichments (Here, ENR_LOW = 1.87 wt%  and  ENR_HIGH = 2.53 wt%) 
across the N (21 if BWR_N = 6) unique positions of the BWR assembly (1/2-diagonal symmetry).


The evaluate_fn callable is injected by the caller at runtime via functools.partial.

Contents:
  config       : RunConfig dataclass + decode_enrichment_vector
  lhs_sampler  : Latin Hypercube Sampling (reads heuristic counts from env dict)
  gp_surrogate : Gaussian Process surrogate 
  bo_loop      : Bayesian Optimisation loop (reads heuristic counts from env dict)
  results      : Result storage, display, convergence plots

Note:- No imports from Environment package, OpenMC calls, geomeetry, heuristic constraints, or other inventories. 
The openmc_bo package contains only the pure mathematical algorithm required for the BO. 

  
"""

from .config import (
    RunConfig,
    decode_enrichment_vector,
    ENR_LOW_DEFAULT,
    ENR_HIGH_DEFAULT,
    N_RODS_SIDE_DEFAULT,
)
from .lhs_sampler  import run_lhs
from .gp_surrogate import train_gp, expected_improvement
from .bo_loop      import run_bo, run_smoke_test
from .results      import (
    summarise_results,
    plot_convergence,
)

__all__ = [
    # Config & decode
    "RunConfig",
    "decode_enrichment_vector",
    "ENR_LOW_DEFAULT",
    "ENR_HIGH_DEFAULT",
    "N_RODS_SIDE_DEFAULT",
    
    # Sampling & GP
    "run_lhs",
    "train_gp",
    "expected_improvement",
    
    # Optimisation
    "run_bo",
    "run_smoke_test",
    
    # Results
    "summarise_results",
    "plot_convergence",
]