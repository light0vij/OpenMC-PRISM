"""
rl_results.py — Result analysis, CSV export and convergence plots for the RL agent
------------------------------------------------------------------------------------

"""

from __future__ import annotations

import os
import csv
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .rl_config import RLConfig


def summarise_rl_results(
    cfg: RLConfig,
    rl_data: dict,
    extra_info: Optional[dict] = None,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Decode the HOF chromosome, prints summary block, saves the full evaluation history to CSV, and return the
    "best" dict (chromosome, keff, ppf, fitness, enr_grid, delta_k) 
    """
    hof  = rl_data["hof_chromosome"]
    keff = rl_data["hof_keff"]
    ppf  = rl_data["hof_ppf"]
    fit  = rl_data["hof_fitness"]

    enr_grid = cfg.decode(hof)
    delta_k  = abs(cfg.k_target - keff)
    n_high   = int((np.round(enr_grid, 4) == round(cfg.enr_high, 4)).sum())
    avg_enr  = float(enr_grid.mean())

    best = dict(
        chromosome = hof,
        keff       = keff,
        keff_std   = rl_data["hof_keff_std"],
        ppf        = ppf,
        fitness    = fit,
        enr_grid   = enr_grid,
        delta_k    = delta_k,
    )

    if verbose:
        kstd_str = f" +/- {rl_data['hof_keff_std']:.5f}" if rl_data["hof_keff_std"] > 0 else ""
        print("=" * 70)
        print(f"  OPTIMAL LAYOUT (RL) — {cfg.model_name}  [{cfg.n_rods_side}x{cfg.n_rods_side}]")
        print("=" * 70)
        print(f"  High-enrichment rods ({cfg.enr_high}%) : {n_high} / {cfg.n_rods_total}")
        print(f"  Low-enrichment rods  ({cfg.enr_low}%) : {cfg.n_rods_total - n_high} / {cfg.n_rods_total}")
        print(f"  Average enrichment                 : {avg_enr:.4f} wt%")
        print()
        print(f"  k∞                              : {keff:.5f}{kstd_str}")
        print(f"  k∞ target                       : {cfg.k_target}")
        print(f"  |∆k∞|                           : {delta_k:.5f}  ({delta_k * 1e5:.1f} pcm)")
        if cfg.ppf_target is not None:
            sat = "SATISFIED" if ppf <= cfg.ppf_target else "NOT SATISFIED"
            print(f"  PPF                                 : {ppf:.4f}  (target <= {cfg.ppf_target})")
            print(f"  PPF constraint                      : {sat}")
        print(f"  Best fitness (HOF)                  : {fit:.6f}")
        print(f"  Total OpenMC evaluations             : {rl_data['n_evaluations']}")
        print(f"  Total training timesteps             : {rl_data['total_timesteps']}")
        if extra_info:
            print()
            for k_, v in extra_info.items():
                print(f"  {k_:<37s}: {v}")
        print("=" * 70)

    if save:
        _save_rl_results_csv(cfg, rl_data)

    return best


def _save_rl_results_csv(cfg: RLConfig, rl_data: dict) -> str:
    """
    One row per completed episode (== one OpenMC evaluation).
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{cfg.model_name}_all_evaluations.csv")

    chroms  = rl_data["all_chromosomes"]
    keffs   = rl_data["all_keff"]
    ppfs    = rl_data["all_ppf"]
    fits    = rl_data["all_fitness"]
    n_total = len(keffs)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        bit_names = [f"bit_{i:02d}" for i in range(cfg.n_sym_rods)]
        w.writerow(["index", "episode"] + bit_names + ["keff", "ppf", "fitness"])
        for i in range(n_total):
            w.writerow([i, i + 1] + list(chroms[i]) + [keffs[i], ppfs[i], fits[i]])

    print(f"RL results saved -> {path}")
    return path


#Convergence PLot
def plot_rl_convergence(cfg: RLConfig, rl_data: dict, save: bool = True, dark: bool = True) -> plt.Figure:
    all_fit  = rl_data["all_fitness"]
    all_k    = rl_data["all_keff"]
    win_best = rl_data["window_best_fitness"]
    win_mean = rl_data["window_mean_fitness"]
    n_win    = len(win_best)
    episodes_per_window = cfg.log_every_n_episodes
    win_x = np.arange(1, n_win + 1) * episodes_per_window
    ep_x  = np.arange(1, len(all_fit) + 1)

    running_best = np.minimum.accumulate(all_fit) if len(all_fit) else np.array([])
    delta_k      = np.abs(cfg.k_target - all_k)

    bg_fig = "black" if dark else "white"
    bg_ax  = "midnightblue" if dark else "whitesmoke"
    tc     = "silver" if dark else "dimgray"
    spine  = "darkslateblue" if dark else "lightgray"
    wc     = "white" if dark else "black"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(bg_fig)
    fig.suptitle(f"Reinforcement Learning Convergence — {cfg.model_name}", color=wc, fontsize=14, y=1.01)
    for ax in axes.flat:
        ax.set_facecolor(bg_ax)
        ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(tc)
        ax.yaxis.label.set_color(tc)
        ax.title.set_color(wc)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine)

    # Best fitness per logging window
    ax = axes[0, 0]
    if n_win:
        ax.plot(win_x, win_best, "o-", color="orange", lw=2, ms=4, label="Best fitness / window")
    ax.axhline(0, color="mediumseagreen", ls="--", lw=1.5, label="Perfect = 0")
    ax.set_xlabel("Episode"); ax.set_ylabel("Fitness (lower = better)")
    ax.set_title(f"Best fitness per {episodes_per_window}-episode window")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # Mean vs best per window
    ax = axes[0, 1]
    if n_win:
        ax.plot(win_x, win_mean, "s-", color="steelblue", lw=2, ms=4, label="Mean")
        ax.plot(win_x, win_best, "o--", color="orange", lw=1.5, ms=3, alpha=0.6, label="Best")
    ax.set_xlabel("Episode"); ax.set_ylabel("Fitness")
    ax.set_title("Mean vs best fitness per window")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    #  k∞ + |∆k∞| over all episodes
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(ep_x, all_k, ".", color="coral", ms=3, alpha=0.5, label="k∞ (all episodes)")
    ax.axhline(cfg.k_target, color="mediumseagreen", ls="--", lw=1.5, label=f"Target = {cfg.k_target}")
    ax2.plot(ep_x, delta_k, ".", color="plum", ms=3, alpha=0.4, label="|∆k∞|")
    ax2.set_ylabel("|∆k∞|", color="plum"); ax2.tick_params(axis="y", colors="plum")
    ax.set_xlabel("Episode"); ax.set_ylabel("k∞", color="coral"); ax.tick_params(axis="y", colors="coral")
    ax.set_title("k∞ and |∆k∞| across training")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    #  All-episode fitness scatter + running best (exploration -> exploitation)
    ax = axes[1, 1]
    ax.plot(ep_x, all_fit, ".", color="mediumaquamarine", ms=3, alpha=0.35, label="Episode fitness")
    if len(running_best):
        ax.plot(ep_x, running_best, "-", color=("white" if dark else "black"), lw=1.8, label="Running best")
    ax.set_xlabel("Episode"); ax.set_ylabel("Fitness")
    ax.set_title("Per-episode fitness vs running best")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save:
        os.makedirs(cfg.results_dir, exist_ok=True)
        path = os.path.join(cfg.results_dir, f"{cfg.model_name}_rl_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        print(f"Figure saved -> {path}")

    return fig
