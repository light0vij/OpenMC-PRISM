"""
bo_loop.py — Bayesian Optimisation loop 
----------------------------------------

This module is 100% pure algorithm (zero environment imports).

1. Setup (Caller/Notebook)
    environments.env  ––> Import evaluate_for_bo
    functools.partial ––> Bind config: evaluate_fn = partial(evaluate_for_bo, cfg=cfg_bo)

2. Injection
    evaluate_fn ––> Pass to module: run_bo(..., evaluate_fn)  

3. Execution 
    Input: x_cont (np.ndarray) ––> Call: evaluate_fn(x_cont)–––––––––––––––––––––––––––––––––––––> (from environments.env.py)
        ↓
    Output: ––> Returns dict
    ↳ Required Keys: {'keff', 'keff_std', 'ppf', 'corner_violations', 'adj_high_rods'}

4. Objective (maximisation of the composite score)

    score = −alpha_k·|k_target − k∞| − alpha_ppf·max(0, PPF − ppf_target)² − adj_penalty_weight·adj_high_rods    (if toggled on)

Note:- Here the adjacency count is provided the environment codes. This script reads it from the result dict and applies the weight from cfg.

"""

from __future__ import annotations

import numpy as np
from tqdm.notebook import tqdm
from typing import Callable, Optional

from .config       import RunConfig
from .gp_surrogate import train_gp, expected_improvement, gp_summary

import warnings
from sklearn.exceptions import ConvergenceWarning

# Removes GP convergence warnings caused by argsort plateaus
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_bo(
    cfg: RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    lhs_data: dict,
    verbose: bool = True,
) -> dict:
    """
    Run the Bayesian Optimisation loop.

    """
    # Unpack LHS seed data
    X        = lhs_data["X"].copy()
    keff     = lhs_data["keff"].copy()
    keff_std = lhs_data["keff_std"].copy()
    ppf      = lhs_data["ppf"].copy()
    score    = lhs_data["score"].copy()
    n_lhs    = len(X)

    # Initial GP fit
    gp, scaler = train_gp(X, score, cfg)
    if verbose:
        print(gp_summary(gp, cfg))

    
    # The GP models the continuous score landscape and never sees the binary grid directly.
    
    rng    = np.random.default_rng(cfg.random_seed + 1)
    X_cand = rng.uniform(cfg.lower, cfg.upper,
                         size=(cfg.n_candidates, cfg.n_vars))

    bo_log = []

    if verbose:
        print(f"\nRunning {cfg.n_bo_iterations} BO iterations …")
        print(f"Objective : − {cfg.alpha_k}·|{cfg.k_target} − k∞|"
              f" − {cfg.alpha_ppf}·max(0, PPF − {cfg.ppf_target})²\n")

    for i in tqdm(range(cfg.n_bo_iterations),
                  desc="BO iterations", disable=not verbose):

        # Selects next candidate via Expected Improvement
        ei     = expected_improvement(X_cand, gp, scaler,
                                      y_best=score.max(), xi=cfg.ei_xi)
        x_next = X_cand[np.argmax(ei)].copy()

        
        # runs OpenMC, counts heuristic violations, and packs everything into res.
        res  = evaluate_fn(x_next)
        k    = float(res["keff"])
        kstd = float(res.get("keff_std", 0.0))
        p    = float(res.get("ppf", 0.0))

       
        s = cfg.composite_score(k, p) #––––––––––––––––––––––––>  Base composite score

    ##**************** Adjacency penalty (weight applied here) *********************#########
        if getattr(cfg, "use_adjacency_penalty", False):
            adj_high_rods = float(res.get("adj_high_rods", 0))
            s -= cfg.adj_penalty_weight * adj_high_rods

        
        X        = np.vstack([X, x_next])
        keff     = np.append(keff,     k)
        keff_std = np.append(keff_std, kstd)
        ppf      = np.append(ppf,      p)
        score    = np.append(score,    s)

        # Retrain GP
        gp, scaler = train_gp(X, score, cfg)

        feas  = (p <= cfg.ppf_target) if cfg.ppf_target is not None else True
        entry = dict(
            iter       = i + 1,
            x          = x_next,
            keff       = k,
            keff_std   = kstd,
            ppf        = p,
            score      = s,
            feasible   = feas,
            best_score = score.max(),
            delta_k    = abs(cfg.k_target - k),
        )
        bo_log.append(entry)

        if verbose:
            _print_bo_iter(cfg, entry, n_lhs + i + 1)

    if verbose:
        _print_bo_summary(cfg, keff[n_lhs:], ppf[n_lhs:], score[n_lhs:])

    return dict(
        X        = X,
        keff     = keff,
        keff_std = keff_std,
        ppf      = ppf,
        score    = score,
        bo_log   = bo_log,
        gp       = gp,
        scaler   = scaler,
        n_lhs    = n_lhs,
    )

###**************8 SMOKE TEST BELOW ********************###############

def run_smoke_test(
    cfg: RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    smoke_x: np.ndarray,
    smoke_label: Optional[str] = None,
) -> dict:
    
    # Store original values to restore after test
    _prod = (cfg.n_particles, cfg.n_inactive, cfg.n_active)
    cfg.n_particles = cfg.n_particles_smoke
    cfg.n_inactive  = cfg.n_inactive_smoke
    cfg.n_active    = cfg.n_active_smoke

    label    = smoke_label or cfg.model_name
    enr_grid = cfg.decode(smoke_x)
    avg_enr  = enr_grid.mean()
    n_high   = int((enr_grid == cfg.enr_high).sum())

    print(f"  SMOKE TEST — {label}")
    print("═" * 70)
    print(f"  Continuous vector (first 6): {np.round(smoke_x[:6], 3)}")
    
    # FIXED: Replaced hardcoded '36' with cfg.n_rods_total
    print(f"  Decoded grid       : {n_high} × {cfg.enr_high}%  +  "
          f"{cfg.n_rods_total - n_high} × {cfg.enr_low}%  "
          f"(avg {avg_enr:.4f} wt%)")
    
    print(f"  Particles          : {cfg.n_particles}  "
          f"({cfg.n_inactive} inactive + {cfg.n_active} active)")
    print()
    
    smoke_res = evaluate_fn(smoke_x)

    k    = float(smoke_res["keff"])
    kstd = float(smoke_res.get("keff_std", 0.0))
    ppf  = float(smoke_res.get("ppf", 0.0))

    std_str = f" ± {kstd:.5f}" if kstd > 0 else ""
    print(f"  k∞    = {k:.5f}{std_str}   (target = {cfg.k_target})")
    print(f"  |Δk∞| = {abs(cfg.k_target - k):.5f}")
    if cfg.ppf_target is not None:
        feas_str = " feasible" if ppf <= cfg.ppf_target else " infeasible"
        print(f"  PPF   = {ppf:.4f}   (limit ≤ {cfg.ppf_target})  {feas_str}")

    # Restore original settings
    cfg.n_particles, cfg.n_inactive, cfg.n_active = _prod
    print("=" * 70)
    return smoke_res


#**************** Helpers ******************


def _print_bo_iter(cfg, entry, total_idx):
    ppf_str = (f"  PPF={entry['ppf']:.3f}"
               if cfg.ppf_target is not None else "")
    std_str = (f" ±{entry['keff_std']:.5f}"
               if entry["keff_std"] > 0 else "")
    print(f"  BO {entry['iter']:2d} [total={total_idx}]:  "
          f"k∞={entry['keff']:.5f}{std_str}  |Δk|={entry['delta_k']:.5f}"
          f"{ppf_str}  "
          f"score={entry['score']:.5f}  best={entry['best_score']:.5f}")


def _print_bo_summary(cfg, keff_bo, ppf_bo, score_bo):
    best_i = int(np.argmax(score_bo))
    print(f"\n── BO summary {'─'*50}")
    print(f"BO iterations    : {len(keff_bo)}")
    print(f"k∞ range (BO)    : [{keff_bo.min():.5f}, {keff_bo.max():.5f}]")
    print(f"k∞ target        : {cfg.k_target}")
    print(f"Best |Δk∞|       : {abs(cfg.k_target - keff_bo[best_i]):.5f}")
    if cfg.ppf_target is not None:
        feas = (ppf_bo <= cfg.ppf_target).sum()
        print(f"Feasible (BO)    : {feas} / {len(keff_bo)}")
    print(f"Best score (BO)  : {score_bo.max():.5f}")
