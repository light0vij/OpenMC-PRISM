"""
bwr_vis.py — Visualisation tools for the reactor enrichment-layout platform
-----------------------------------------------------------------------------

Functions in this file:-

plot_enr_grid(enr_grid, title, ax)--------------> Enrichment plotting
    
plot_ga_convergence(cfg, ga_data, save, dark) ------------------------> GA convergence plot
    
plot_bo_convergence(cfg, bo_data, save, dark)-----------------------> BO convergence plot
    
validate_and_plot_hifi(cfg, hifi_ref, best_lofi, hifi_opt, ...)--------------> for both GA and BO
    
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional

from .mapping import ENR_LOW, ENR_HIGH



# Shared enrichment grid plot


def plot_enr_grid(
    enr_grid: np.ndarray,
    title: str = "Enrichment Layout",
    ax: Optional[plt.Axes] = None,
    enr_low:  float = ENR_LOW,
    enr_high: float = ENR_HIGH,
    dark: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Render enrichment grid as a colour-coded cell map.

    Low-enrichment cells -> blue; high-enrichment cells -> red.
    Enrichment values are printed inside each cell.

    
    Visualizes the AI-generated enrichment layout.
    
    This is method is used to plot the geometry instead of OpenMC's native geometry plot, because:- 
    1. Conceptual Clarity: Focuses on the AI's binary decision grid (High vs. Low enrichment) 
        rather than cluttered physical geometry (cladding, gaps, etc.).
    2. High-Speed Reporting: Lightweight Matplotlib rendering avoids the heavy computational overhead 
        and kernel instability of OpenMC's geometry engine.
    3. AI Data Overlays: Designed to dynamically display AI-specific metrics like k-infinity and PPF 
        directly on the layout for rapid optimization feedback.
    
     """


    

    
    bg_ax  = "midnightblue" if dark else "whitesmoke"
    bg_fig = "black" if dark else "white"

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(bg_fig)

    ax.set_facecolor(bg_ax)
    n    = enr_grid.shape[0]
    cmap = {enr_low: "steelblue", enr_high: "crimson"}

    for r in range(n):
        for c in range(n):
            val = enr_grid[r, c]
            col = cmap.get(round(val, 4), "gray")
            rect = Rectangle([c, n - 1 - r], 1, 1,
                              facecolor=col, edgecolor="black", lw=1.5)
            ax.add_patch(rect)
            ax.text(c + 0.5, n - 0.5 - r, f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color="white" if dark else "black")
    ax.tick_params(colors="silver" if dark else "dimgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("darkslateblue" if dark else "lightgray")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap[enr_high]),
        plt.Rectangle((0, 0), 1, 1, color=cmap[enr_low]),
    ]
    ax.legend(handles,
              [f"{enr_high}% (High)", f"{enr_low}% (Low)"],
              fontsize=6, facecolor=bg_ax,
              labelcolor="white" if dark else "black",
              edgecolor="darkslateblue" if dark else "lightgray",
              loc="lower right")
    
    #Plot saving
    if save_path:
        fig = ax.figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        print(f"Saved grid plot to -> {save_path}")
    
    return ax



# High-fidelity validation plot (shared GA + BO signature)
# **********************************************************

def validate_and_plot_hifi(
    cfg,
    hifi_ref:     dict,
    best_lofi:    dict,
    hifi_opt:     dict,
    enr_grid_ref: np.ndarray,
    enr_grid_opt: np.ndarray,
    filename:     str = "validation_comparison.png",
    algorithm:    str = "GA",
) -> plt.Figure:
    """
    Compare and visualise three designs:
      A — Reference enrichment layout         [HiFi]
      B — Best optimised layout at LoFi       [LoFi]
      C — Best optimised layout re-evaluated  [HiFi]

    algorithm    : "GA" or "BO" — used in the title
    """
    delta_keff_AC = hifi_opt["keff"]  - hifi_ref["keff"]
    delta_keff_BC = hifi_opt["keff"]  - best_lofi["keff"]
    delta_ppf_AC  = hifi_opt["ppf"]   - hifi_ref["ppf"]
    delta_ppf_BC  = hifi_opt["ppf"]   - best_lofi["ppf"]

    sigma_lo_est   = 1.0 / (cfg.n_particles * cfg.n_active) ** 0.5
    sigma_hifi     = hifi_opt["keff_std"]
    sigma_combined = (sigma_lo_est**2 + sigma_hifi**2) ** 0.5
    z_score_bc     = (abs(delta_keff_BC) / sigma_combined
                      if sigma_combined > 0 else 0.0)

    # Console comparison table
    print("═" * 72)
    print(f"  {algorithm} VALIDATION COMPARISON — Enrichment Layout Optimisation")
    print("═" * 72)
    print(f"{'Design':<44}  {'k∞':>10}  {'σ(k∞) pcm':>10}  {'PPF':>7}")
    print("-" * 72)
    for label, res, kstd in [
        ("A — Reference layout          [HiFi]", hifi_ref,  hifi_ref["keff_std"]),
        (f"B — {algorithm} optimal layout         [LoFi]", best_lofi, sigma_lo_est),
        (f"C — {algorithm} optimal layout (re-run)[HiFi]", hifi_opt,  hifi_opt["keff_std"]),
    ]:
        kstd_str = (f"{kstd*1e5:>10.2f}"
                    if isinstance(kstd, float) and kstd > 0
                    else f"{'~'+str(round(sigma_lo_est*1e5,1)):>10}")
        print(f"  {label:<42}  {res['keff']:>10.5f}  {kstd_str}  {res['ppf']:>7.4f}")
    print("═" * 72)
    print(f"\n  Δk∞  (A→C): {delta_keff_AC:+.5f}  ({delta_keff_AC*1e5:+.1f} pcm)")
    print(f"  ΔPPF (A→C): {delta_ppf_AC:+.4f}")
    print(f"  Δk∞  (B→C): {delta_keff_BC:+.5f}  |Z| = {z_score_bc:.2f}  "
          f"{' consistent' if z_score_bc < 2 else ' check particles'}")

    # Figure
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("midnightblue")
    gs     = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_opt = fig.add_subplot(gs[1, 0])
    ax_k   = fig.add_subplot(gs[0, 1])
    ax_ppf = fig.add_subplot(gs[0, 2])
    ax_dis = fig.add_subplot(gs[0, 3])
    ax_dk  = fig.add_subplot(gs[1, 1:])

    for ax in [ax_ref, ax_opt, ax_k, ax_ppf, ax_dis, ax_dk]:
        ax.set_facecolor("midnightblue")
        ax.tick_params(colors="silver")
        ax.xaxis.label.set_color("silver")
        ax.yaxis.label.set_color("silver")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("darkslateblue")

       

    plot_enr_grid(enr_grid_ref, "A — Reference",    ax=ax_ref, dark=True)
    plot_enr_grid(enr_grid_opt, f"C — {algorithm} optimal", ax=ax_opt, dark=True)

    labels = ["A\nReference\n[HiFi]", f"B\n{algorithm} best\n[LoFi]",
              f"C\n{algorithm} best\n(re-run)\n[HiFi]"]
    colors = ["steelblue", "crimson", "mediumseagreen"]
    keffs  = [hifi_ref["keff"],   best_lofi["keff"],  hifi_opt["keff"]]
    kerrs  = [hifi_ref["keff_std"], sigma_lo_est,      hifi_opt["keff_std"]]
    ppfs   = [hifi_ref["ppf"],    best_lofi["ppf"],    hifi_opt["ppf"]]
    x      = np.arange(3)

    bars = ax_k.bar(x, keffs, color=colors, alpha=0.85, width=0.5, zorder=2)
    ax_k.errorbar(x, keffs, yerr=[2 * e for e in kerrs],
                  fmt="none", color="white", capsize=6, lw=2, zorder=3)
    ax_k.axhline(cfg.k_target, color="indianred", ls="--", lw=1.5,
                 label=f"Target = {cfg.k_target}")
    ax_k.set_xticks(x); ax_k.set_xticklabels(labels, fontsize=7)
    ax_k.set_ylabel("k∞"); ax_k.set_title("k∞ comparison  (±2σ)")
    for bar, val in zip(bars, keffs):
        ax_k.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                  f"{val:.5f}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax_k.legend(fontsize=7, facecolor="midnightblue", labelcolor="white", edgecolor="darkslateblue")
    ax_k.grid(True, alpha=0.15, axis="y")

    bars = ax_ppf.bar(x, ppfs, color=colors, alpha=0.85, width=0.5, zorder=2)
    if cfg.ppf_target is not None:
        ax_ppf.axhline(cfg.ppf_target, color="indianred", ls="--", lw=1.5,
                       label=f"Limit = {cfg.ppf_target}")
    ax_ppf.set_xticks(x); ax_ppf.set_xticklabels(labels, fontsize=7)
    ax_ppf.set_ylabel("PPF"); ax_ppf.set_title("Power Peaking Factor")
    for bar, val in zip(bars, ppfs):
        ax_ppf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax_ppf.legend(fontsize=7, facecolor="midnightblue", labelcolor="white", edgecolor="darkslateblue")
    ax_ppf.grid(True, alpha=0.15, axis="y")

    dk_pcm = delta_keff_BC * 1e5
    ax_dis.barh([1, 0], [dk_pcm, delta_ppf_BC],
                color=["mediumseagreen", "darkorange"], alpha=0.85, height=0.4)
    ax_dis.axvline(0, color="white", lw=1)
    ax_dis.axvline(+2 * sigma_combined * 1e5, color="indianred", ls="--", lw=1,
                   label="±2σ (k∞)")
    ax_dis.axvline(-2 * sigma_combined * 1e5, color="indianred", ls="--", lw=1)
    ax_dis.set_yticks([0, 1])
    ax_dis.set_yticklabels(["ΔPPF (B->C)", "Δk∞ pcm (B->C)"],
                            color="silver", fontsize=8)
    ax_dis.set_title(
        f"LoFi vs HiFi discrepancy\n|Z| = {z_score_bc:.2f}"
        f"  {' consistent' if z_score_bc < 2 else ' check'}",
        color="white",
    )
    ax_dis.legend(fontsize=7, facecolor="midnightblue", labelcolor="white",
                  edgecolor="darkslateblue")
    ax_dis.grid(True, alpha=0.15, axis="x")

    dk_vals = [abs(cfg.k_target - v) for v in keffs]
    bars = ax_dk.bar(x, dk_vals, color=colors, alpha=0.85, width=0.5, zorder=2)
    ax_dk.set_xticks(x); ax_dk.set_xticklabels(labels, fontsize=8)
    ax_dk.set_ylabel(f"|k∞ − {cfg.k_target}|")
    ax_dk.set_title(f"Distance from k∞ target — smaller is better")
    for bar, val in zip(bars, dk_vals):
        ax_dk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                   f"{val:.5f}", ha="center", va="bottom", fontsize=8, color="white")
    ax_dk.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        f"{cfg.model_name} — {algorithm} High-Fidelity Validation\n"
        "A = Reference   B = Optimised [LoFi]   C = Optimised [HiFi re-run]",
        color="white", fontsize=11, y=1.01,
    )

    save_dir = getattr(cfg, "results_dir", "results")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"\nSaved → {filepath}")
    return fig
