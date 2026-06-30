"""
rl_env.py — BWR NxN layout - Gymnasium environment
---------------------------------------------------

Episode -> building one complete chromosome, 
           one symmetry position at a time, in cfg.sym_index order. 
           
    reward -> only at the terminal step, 
              after the full grid is built->openmc run once per episode. 


Action Masking: A step-by-step formulation allows us to structurally enforce inventory constraints (e.g., exact HIGH rod counts) via action masking. The agent cannot physically place invalid rods, allowing it to focus entirely on spatial optimization instead of learning counting rules.

Contract with the rest of the platform (environments):
  evaluate_fn : injected by the caller via functools.partial, like in openmc_ga / openmc_bo (from environments.rnv.py)
                  

"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .rl_config import RLConfig, high_units_reachable


class BWREnrichmentEnv(gym.Env):
    """
    Gymnasium environment for sequential, inventory-constrained BWR enrichment-layout construction.

    Observation -> Flat float32 vector containing the current grid state (one-hot: undecided/low/high), current step coordinates (row, col, flags), and remaining rod                     inventory.
    Action      : Discrete(2) -> 0 = LOW, 1 = HIGH (for the current symmetry position).
    
    Reward      : 0.0 at every non-terminal step unless use_adjacency_penalty
                  is on ; 
                  at the terminal step, reward = -fitness
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: RLConfig, evaluate_fn: Callable[[np.ndarray], dict]):
        super().__init__()
        self.cfg = cfg
        self.evaluate_fn = evaluate_fn
        self._corner_idx = cfg.get_corner_indices()

        n_sym = cfg.n_sym_rods
        obs_dim = 3 * n_sym + 9
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        '''
        Corners force-masked to LOW. Exclude future corners from feasibility lookahead so the action mask doesn't falsely allocate remaining 
        HIGH inventory to unreachable slots.
        '''
        self._n_usable_diag    = sum(1 for i in cfg.diag_idx    if i not in self._corner_idx)
        self._n_usable_offdiag = sum(1 for i in cfg.offdiag_idx if i not in self._corner_idx)

        # Episode state — (re)initialised in reset()
        self.chrom: Optional[np.ndarray] = None
        self.full_grid: Optional[np.ndarray] = None
        self.step_idx = 0
        self.remaining_high = 0
        self.remaining_low = 0
        self.diag_left = 0             # raw remaining diagonal slots (observation only)
        self.offdiag_left = 0          # raw remaining off-diagonal slots (observation only)
        self.usable_diag_left = 0      # remaining NON-CORNER diagonal slots (drives the mask)
        self.usable_offdiag_left = 0   # remaining NON-CORNER off-diagonal slots (drives the mask)

    
#************************ Gymnasium ***********************************
   

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        cfg = self.cfg

        self.chrom     = np.full(cfg.n_sym_rods, -1, dtype=int)
        self.full_grid = np.full((cfg.n_rods_side, cfg.n_rods_side), -1, dtype=int)
        self.step_idx  = 0

        self.remaining_high = cfg.n_high_rods
        self.remaining_low  = cfg.n_rods_total - cfg.n_high_rods
        self.diag_left      = cfg.n_diag
        self.offdiag_left   = cfg.n_offdiag
        self.usable_diag_left    = self._n_usable_diag
        self.usable_offdiag_left = self._n_usable_offdiag

        return self._build_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        action = int(action)
        idx = self.step_idx
        r, c = cfg.sym_index[idx]
        is_diag = (r == c)
        mult = 1 if is_diag else 2

        
        mask = self.action_masks()
        '''
         Eventhough a properly wrapped MaskablePPO agent will never request a masked-out action, Sometimes a random test or an untrained agent will try to 
         make an illegal move (for example in the smoke test). If that happens, this code will just override it with a valid choice so the final design always follows          the rules and acts like a failsafe. 
        '''
        if not mask[action]:
            action = int(np.argmax(mask))

        self.chrom[idx] = action
        self.full_grid[r, c] = action
        self.full_grid[c, r] = action

        if action == 1:
            self.remaining_high -= mult
        else:
            self.remaining_low -= mult

        if is_diag:
            self.diag_left -= 1
            if idx not in self._corner_idx:
                self.usable_diag_left -= 1
        else:
            self.offdiag_left -= 1
            if idx not in self._corner_idx:
                self.usable_offdiag_left -= 1

        reward = self._shaping_reward(r, c, action)

        self.step_idx += 1
        terminated = self.step_idx == cfg.n_sym_rods
        truncated = False
        info: dict = {}

        if terminated:
            enr_grid = cfg.decode(self.chrom)
            res  = self.evaluate_fn(self.chrom)
            keff = float(res["keff"])
            ppf  = float(res.get("ppf", 0.0))
            fit  = cfg.fitness(keff, ppf)
            if cfg.use_adjacency_penalty:
                fit += cfg.adj_penalty_weight * float(res.get("adj_high_rods", 0))
            reward += -fit

            info = dict(
                keff       = keff,
                keff_std   = float(res.get("keff_std", 0.0)),
                ppf        = ppf,
                fitness    = float(fit),
                chromosome = self.chrom.copy(),
                enr_grid   = enr_grid,
            )

        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self):
        n = self.cfg.n_rods_side
        rows = []
        for r in range(n):
            row = []
            for c in range(n):
                v = self.full_grid[r, c]
                row.append("H" if v == 1 else ("L" if v == 0 else "."))
            rows.append(" ".join(row))
        text = "\n".join(rows)
        print(text)
        return text

    def close(self):
        pass

    
#***************Action masking — (required by sb3_contrib.MaskablePPO via sb3_contrib.common.wrappers.ActionMasker)******************

    def action_masks(self) -> np.ndarray:
        """
        
        Action Feasibility Mask -> Returns [low_is_legal, high_is_legal]

        - Lookahead   -> Count ONLY "usable" (non-corner) slots for HIGH rod budget.
        - Corner Trap -> Future corners = forced LOW. Counting them => false capacity => fatal trap.
        - Hard Mask   -> If current step == corner => block HIGH outright.
        - Guarantee   -> Strict accounting => agent never stuck (always >= 1 valid action).

        """
        cfg = self.cfg
        if self.step_idx >= cfg.n_sym_rods:
            return np.array([True, True])  # episode already terminated

        idx = self.step_idx
        r, c = cfg.sym_index[idx]
        is_diag = (r == c)
        mult = 1 if is_diag else 2
        is_corner = idx in self._corner_idx

        '''
        Update Usable Slots -> Only subtract 1 if this slot is a normal (non-corner) space.
        Skip Corners        -> They were not a part of the "usable" budget, so filling one doesn't eat into the remaining capacity.
        
        '''
        usable_diag_after    = self.usable_diag_left
        usable_offdiag_after = self.usable_offdiag_left
        if not is_corner:
            if is_diag:
                usable_diag_after -= 1
            else:
                usable_offdiag_after -= 1

        mask = np.zeros(2, dtype=bool)
        mask[0] = high_units_reachable(self.remaining_high, usable_diag_after, usable_offdiag_after)

        if not (cfg.use_corner_masking and is_corner):
            remaining_after_high = self.remaining_high - mult
            mask[1] = remaining_after_high >= 0 and high_units_reachable(
                remaining_after_high, usable_diag_after, usable_offdiag_after
            )

        return mask

    
#******************************** Helpers ***********************************
    
    def _shaping_reward(self, r: int, c: int, action: int) -> float:
    
        '''
            Reward Shaping -> Soft per-step penalty applied to the partial grid.

                    - Decided Only   -> Ignores empty/undecided neighbors.
                    - Adjacency Only -> Corners are hard-masked elsewhere, so this only penalizes bad rod adjacencies.
                    - Scaled Down    -> Kept small to give the agent early hints (gradient) without overpowering the final terminal score.
    
       '''
        if not (self.cfg.use_adjacency_penalty and action == 1):
            return 0.0

        n = self.cfg.n_rods_side
        penalty = 0.0
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < n and 0 <= cc < n and self.full_grid[rr, cc] == 1:
                penalty += 1.0

        return -self.cfg.adj_penalty_weight * self.cfg.shaping_scale * penalty

    def _build_observation(self) -> np.ndarray:
        cfg = self.cfg
        n_sym = cfg.n_sym_rods

        onehot = np.zeros((n_sym, 3), dtype=np.float32)
        undecided = self.chrom < 0
        low_mask  = self.chrom == 0
        high_mask = self.chrom == 1
        onehot[undecided, 0] = 1.0
        onehot[low_mask,  1] = 1.0
        onehot[high_mask, 2] = 1.0

        if self.step_idx < n_sym:
            r, c = cfg.sym_index[self.step_idx]
            is_diag   = 1.0 if r == c else 0.0
            is_corner = 1.0 if self.step_idx in self._corner_idx else 0.0
            denom = max(cfg.n_rods_side - 1, 1)
            r_norm, c_norm = r / denom, c / denom
        else:
            r_norm = c_norm = is_diag = is_corner = 0.0

        tail = np.array([
            self.step_idx / n_sym,
            r_norm, c_norm, is_diag, is_corner,
            self.remaining_high / cfg.n_rods_total,
            self.remaining_low  / cfg.n_rods_total,
            self.diag_left    / max(cfg.n_diag, 1),
            self.offdiag_left / max(cfg.n_offdiag, 1),
        ], dtype=np.float32)

        return np.concatenate([onehot.ravel(), tail]).astype(np.float32)
