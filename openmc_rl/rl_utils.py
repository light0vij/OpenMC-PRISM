"""
rl_utils.py — Shared for the RL package
---------------------------------------
Random Valid Chromosome -> Builds a random layout that strictly follows inventory & corner rules.


"Reference layout" for validation, or a quick smoke test before training.

Kept out of `rl_env.py` to generate generate layouts instantly without spinning up a Gym environment.

Similar to `make_random_individual() in openmc_ga

"""

from __future__ import annotations

import numpy as np

from .rl_config import RLConfig, high_units_reachable


def make_random_individual(cfg: RLConfig, rng: np.random.Generator) -> np.ndarray:
    n_sym = cfg.n_sym_rods
    chrom = np.zeros(n_sym, dtype=int)
    corner_idx = cfg.get_corner_indices()

    order = list(range(n_sym))
    rng.shuffle(order)

    remaining_high       = cfg.n_high_rods
    usable_diag_left    = sum(1 for i in cfg.diag_idx    if i not in corner_idx)
    usable_offdiag_left = sum(1 for i in cfg.offdiag_idx if i not in corner_idx)

    for idx in order:
        r, c = cfg.sym_index[idx]
        is_diag   = (r == c)
        is_corner = idx in corner_idx
        mult      = 1 if is_diag else 2

        usable_diag_after    = usable_diag_left
        usable_offdiag_after = usable_offdiag_left
        if not is_corner:
            if is_diag:
                usable_diag_after -= 1
            else:
                usable_offdiag_after -= 1

        can_low = high_units_reachable(remaining_high, usable_diag_after, usable_offdiag_after)

        can_high = (not is_corner) and (remaining_high - mult >= 0) and \
            high_units_reachable(remaining_high - mult, usable_diag_after, usable_offdiag_after)

        if can_high and (not can_low or rng.random() < 0.5):
            chrom[idx] = 1
            remaining_high -= mult
        else:
            chrom[idx] = 0

        if not is_corner:
            if is_diag:
                usable_diag_left -= 1
            else:
                usable_offdiag_left -= 1

    return chrom
