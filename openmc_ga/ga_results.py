"""
ga_results.py — Result analysis and CSV export for the GA 
-----------------------------------------------------------


"""

from __future__ import annotations

import os
import csv
import numpy as np
from typing import Optional


import matplotlib.pyplot as plt

from .ga_config import GAConfig


def summarise_ga_results(
    cfg:        GAConfig,
    ga_data:    dict,
    extra_info: Optional[dict] = None,
    save:       bool = True,
    verbose:    bool = True,
) -> dict:
    '''
    Prints a summary of the GA run and the best design.

    '''
    hof      = ga_data["hof_chromosome"]
    keff     = ga_data["hof_keff"]
    ppf      = ga_data["hof_ppf"]
    fit      = ga_data["hof_fitness"]

    enr_grid = cfg.decode(hof)
    delta_k  = abs(cfg.k_target - keff)
    n_high   = int((np.round(enr_grid, 4) == round(cfg.enr_high, 4)).sum())
    avg_enr  = enr_grid.mean()

    all_ppf      = ga_data["all_ppf"]
    feasible_mask = (
        (all_ppf <= cfg.ppf_target)
        if cfg.ppf_target is not None
        else np.ones(len(all_ppf), bool)
    )

    best = dict(
        chromosome = hof,
        keff       = keff,
        keff_std   = ga_data["hof_keff_std"],
        ppf        = ppf,
        fitness    = fit,
        enr_grid   = enr_grid,
        delta_k    = delta_k,
    )

    if verbose:
        kstd_str = (f" ± {ga_data['hof_keff_std']:.5f}"
                    if ga_data["hof_keff_std"] > 0 else "")
        print("═" * 70)
        print(f"  OPTIMAL LAYOUT (GA) — {cfg.model_name}  "
              f"[{cfg.n_rods_side}×{cfg.n_rods_side}]")
        print("═" * 70)
        print(f"  High-enrichment rods ({cfg.enr_high}%) : "
              f"{n_high} / {cfg.n_rods_total}")
        print(f"  Low-enrichment rods  ({cfg.enr_low}%) : "
              f"{cfg.n_rods_total - n_high} / {cfg.n_rods_total}")
        print(f"  Average enrichment                 : {avg_enr:.4f} wt%")
        print()
        print(f"  k∞                                 : {keff:.5f}{kstd_str}")
        print(f"  k∞ target                          : {cfg.k_target}")
        print(f"  |Δk∞|                              : {delta_k:.5f}  "
              f"({delta_k*1e5:.1f} pcm)")
        if cfg.ppf_target is not None:
            sat = "SATISFIED " if ppf <= cfg.ppf_target else "NOT SATISFIED  "
            print(f"  PPF                                : {ppf:.4f}  "
                  f"(target ≤ {cfg.ppf_target})")
            print(f"  PPF constraint                     : {sat}")
        print(f"  Best fitness (HOF)                 : {fit:.6f}")
        print(f"  Total OpenMC evaluations           : {ga_data['n_evaluations']}")
        print(f"  Generations run                    : {ga_data['n_generations_run']}")
        print(f"  Converged (early stop)             : {ga_data['converged']}")
        print(f"  Feasible designs found             : "
              f"{feasible_mask.sum()} / {len(all_ppf)}")
        if extra_info:
            print()
            for k_, v in extra_info.items():
                print(f"  {k_:<37s}: {v}")
        print("═" * 70)

    if save:
        _save_ga_results_csv(cfg, ga_data)

    return best


def _save_ga_results_csv(cfg: GAConfig, ga_data: dict) -> str:
    #Save all evaluated chromosomes + physics results to a CSV file
    
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{cfg.model_name}_all_evaluations.csv")

    chroms  = ga_data["all_chromosomes"]
    keffs   = ga_data["all_keff"]
    ppfs    = ga_data["all_ppf"]
    fits    = ga_data["all_fitness"]
    pop_s   = cfg.population_size
    n_total = len(keffs)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        bit_names = [f"bit_{i:02d}" for i in range(cfg.n_sym_rods)]
        w.writerow(["index", "generation", "individual"]
                   + bit_names + ["keff", "ppf", "fitness"])
        for i in range(n_total):
            gen = i // pop_s + 1
            ind = i  % pop_s
            w.writerow([i, gen, ind]
                       + list(chroms[i])
                       + [keffs[i], ppfs[i], fits[i]])

    print(f"GA results saved → {path}")
    return path



# GA convergence plot
#**********************#  #––––––––––––––––––––––––> moved from environments.bwr_vis.py

def plot_ga_convergence(
    cfg,
    ga_data: dict,
    save: bool = True,
    dark: bool = True,
) -> plt.Figure:

    gen_best = ga_data["gen_best_fitness"]
    gen_mean = ga_data["gen_mean_fitness"]
    gen_div  = ga_data["gen_diversity"]
    n_gen    = len(gen_best)
    gens     = np.arange(1, n_gen + 1)

    pop_s    = cfg.population_size
    all_k    = ga_data["all_keff"]
    all_fit  = ga_data["all_fitness"]

    gen_best_k  = []
    gen_best_dk = []
    for g in range(n_gen):
        start = g * pop_s
        end   = min(start + pop_s, len(all_k))
        if start >= len(all_k):
            break
        best_i = int(np.argmin(all_fit[start:end])) + start
        gen_best_k.append(all_k[best_i])
        gen_best_dk.append(abs(cfg.k_target - all_k[best_i]))
    gen_best_k  = np.array(gen_best_k)
    gen_best_dk = np.array(gen_best_dk)

    bg_fig = "black"         if dark else "white"
    bg_ax  = "midnightblue"  if dark else "whitesmoke"
    tc     = "silver"        if dark else "dimgray"
    spine  = "darkslateblue" if dark else "lightgray"
    wc     = "white"         if dark else "black"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(bg_fig)
    fig.suptitle(f"Genetic Algorithm Convergence — {cfg.model_name}",
                 color=wc, fontsize=14, y=1.01)
    for ax in axes.flat:
        ax.set_facecolor(bg_ax)
        ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(tc)
        ax.yaxis.label.set_color(tc)
        ax.title.set_color(wc)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine)

    # Best fitness
    ax = axes[0, 0]
    ax.plot(gens, gen_best, "o-", color="orange", lw=2, ms=4, label="Best fitness")
    ax.axhline(0, color="mediumseagreen", ls="--", lw=1.5, label="Perfect = 0")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (lower = better)")
    ax.set_title("Best fitness per generation")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    #  Mean vs best
    ax = axes[0, 1]
    ax.plot(gens, gen_mean, "s-", color="steelblue", lw=2, ms=4, label="Mean")
    ax.plot(gens, gen_best, "o--", color="orange",   lw=1.5, ms=3, alpha=0.6, label="Best")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("Mean vs best fitness per generation")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # k∞ + |Δk∞|
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    k_gens = np.arange(1, len(gen_best_k) + 1)
    ax.plot(k_gens, gen_best_k, "o-", color="coral",  lw=2, ms=4, label="Best k∞")
    ax.axhline(cfg.k_target, color="mediumseagreen",   ls="--", lw=1.5,
               label=f"Target = {cfg.k_target}")
    ax2.plot(k_gens, gen_best_dk, "^--", color="plum", lw=1.5, ms=4,
             alpha=0.7, label="|Δk∞|")
    ax2.set_ylabel("|Δk∞|", color="plum")
    ax2.tick_params(axis="y", colors="plum")
    ax.set_xlabel("Generation"); ax.set_ylabel("k∞", color="coral")
    ax.tick_params(axis="y", colors="coral")
    ax.set_title("Best k∞ and |Δk∞| per generation")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # Diversity
    ax = axes[1, 1]
    ax.plot(gens, gen_div, "D-", color="mediumaquamarine", lw=2, ms=4,
            label="Hamming diversity")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation"); ax.set_ylabel("Normalised Hamming distance")
    ax.set_title("Population diversity over generations")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save:
        os.makedirs(cfg.results_dir, exist_ok=True)
        path = os.path.join(cfg.results_dir, f"{cfg.model_name}_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        print(f"Figure saved → {path}")

    return fig

