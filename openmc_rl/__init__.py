"""
openmc_rl — Reinforcement Learning (PPO + action masking) agent for OpenMC BWR NxN Reactor.
------------------------------------------------------------------------------------------


The chromosome produced by the trained policy uses the identical binary, 1/2-diagonal-symmetry format GA uses, so it can be passed straight into
environments.env.evaluate_for_ga(chrom, cfg).

"""

from .rl_config import (
    RLConfig,
    make_sym_index,
    chromosome_to_enr_grid,
    enr_grid_to_chromosome,
    high_units_reachable,
    RL_ENR_LOW_DEFAULT,
    RL_ENR_HIGH_DEFAULT,
    RL_N_RODS_SIDE_DEFAULT,
)
from .rl_utils import make_random_individual
from .rl_env import BWREnrichmentEnv
from .rl_policy import build_agent, policy_summary
from .rl_loop import run_rl, run_rl_smoke_test, make_env
from .rl_results import summarise_rl_results, plot_rl_convergence

__all__ = [
    # Config & helpers
    "RLConfig",
    "make_sym_index",
    "chromosome_to_enr_grid",
    "enr_grid_to_chromosome",
    "high_units_reachable",
    "RL_ENR_LOW_DEFAULT",
    "RL_ENR_HIGH_DEFAULT",
    "RL_N_RODS_SIDE_DEFAULT",
    # Random individual
    "make_random_individual",
    # Environment
    "BWREnrichmentEnv",
    # Policy
    "build_agent",
    "policy_summary",
    # Loop
    "run_rl",
    "run_rl_smoke_test",
    "make_env",
    # Results & visualisation
    "summarise_rl_results",
    "plot_rl_convergence",
]
