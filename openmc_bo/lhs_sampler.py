"""
lhs_sampler.py — Latin Hypercube Sampling  for initial GP training for NxN system
----------------------------------------------------------------------------------

The GP lives in [0, 1]^n_vars where n_vars = n_rods_side*(n_rods_side+1)/2.

The sample dimensionality is read from cfg.n_vars at runtime. 

Each continuous sample is evaluated by the injected evaluate_fn, which

internally decodes it to a binary enrichment grid and runs OpenMC.

"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc
from tqdm.notebook import tqdm
from typing import Callable, Optional

from .config import RunConfig


def generate_lhs_samples(cfg: RunConfig, seed: Optional[int] = None) -> np.ndarray:
    
    rng_seed = seed if seed is not None else cfg.random_seed
    sampler  = qmc.LatinHypercube(d=cfg.n_vars, seed=rng_seed)
    X_unit   = sampler.random(n=cfg.n_initial_samples)
    return qmc.scale(X_unit, cfg.lower, cfg.upper)


def run_lhs(
    cfg:         RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    verbose:     bool = True,
) -> dict:
    """
    Run Latin Hypercube Sampling and collect OpenMC results as GP seed data.

    """
    X = generate_lhs_samples(cfg)

    keff_list     = []
    keff_std_list = []
    ppf_list      = []
    score_list    = []

    if verbose:
        avg_enr = (
            cfg.n_high_rods * cfg.enr_high
            + (cfg.n_rods_total - cfg.n_high_rods) * cfg.enr_low
        ) / cfg.n_rods_total
        print(f"Running {cfg.n_initial_samples} LHS samples  "
              f"[{cfg.n_rods_side}×{cfg.n_rods_side}, "
              f"{cfg.n_vars}-D continuous relaxation]")
        print(f"  ENR inventory : {cfg.n_high_rods} × {cfg.enr_high}%  +  "
              f"{cfg.n_rods_total - cfg.n_high_rods} × {cfg.enr_low}%  "
              f"=> avg {avg_enr:.4f} wt%")
        print(f"  k∞ target     : {cfg.k_target}   "
              f"PPF limit ≤ {cfg.ppf_target}")

    for i, x in enumerate(tqdm(X, desc="LHS samples", disable=not verbose)):
        res  = evaluate_fn(x)
        k    = float(res["keff"])
        kstd = float(res.get("keff_std", 0.0))
        ppf  = float(res.get("ppf", 0.0))
        s    = cfg.composite_score(k, ppf)

     #*************** Applies adjacency penalty using the count injected by the Environment************#########
        if getattr(cfg, "use_adjacency_penalty", False):
            s -= cfg.adj_penalty_weight * float(res.get("adj_high_rods", 0))

    #********************************************##############

        score_list.append(s)
        keff_list.append(k)
        keff_std_list.append(kstd)
        ppf_list.append(ppf)

        if verbose:
            _print_sample(cfg, i, k, kstd, ppf, s)

    keff  = np.array(keff_list)
    kstd  = np.array(keff_std_list)
    ppf   = np.array(ppf_list)
    score = np.array(score_list)

    if verbose:
        _print_lhs_summary(cfg, keff, ppf, score)

    return dict(X=X, keff=keff, keff_std=kstd, ppf=ppf, score=score)



#************* Helpers *****************############


def _print_sample(cfg, i, keff, keff_std, ppf, score):
    idx_str = f"[{i+1:2d}/{cfg.n_initial_samples}]"
    ppf_str = f"  PPF={ppf:.3f}" if cfg.ppf_target is not None else ""
    std_str = f" ±{keff_std:.5f}" if keff_std > 0 else ""
    print(f"  {idx_str}  k∞={keff:.5f}{std_str}{ppf_str}  score={score:.5f}")


def _print_lhs_summary(cfg, keff, ppf, score):
    n = len(keff)
    print(f"\n── LHS summary {'─'*45}")
    print(f"Samples          : {n}")
    print(f"k∞ range         : [{keff.min():.5f}, {keff.max():.5f}]")
    print(f"k∞ target        : {cfg.k_target}")
    if cfg.ppf_target is not None:
        feasible = (ppf <= cfg.ppf_target).sum()
        print(f"PPF range        : [{ppf.min():.3f}, {ppf.max():.3f}]")
        print(f"Feasible         : {feasible} / {n}")
    best_i = int(np.argmax(score))
    print(f"Best score       : {score.max():.5f}  "
          f"(k∞={keff[best_i]:.5f}, "
          f"|Δk|={abs(cfg.k_target - keff[best_i]):.5f})")
