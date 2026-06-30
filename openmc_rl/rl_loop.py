"""
rl_loop.py — Training loop for the RL agent  (similar to ga_loop.py / bo_loop.py)
--------------------------------------------------------------------------------

run_rl(cfg, evaluate_fn)        -> trains a MaskablePPO policy, 
                                   
run_rl_smoke_test(cfg, evaluate_fn) -> one untrained, random-but-feasible
                                    episode at smoke particle counts, to
                                    verify the env <-> evaluate_fn <-> OpenMC
                                    wiring before spending any training
                                    budget — same role as
                                    run_ga_smoke_test() / run_smoke_test().

evaluate_fn is injected by the caller via functools.partial, exactly like
openmc_ga / openmc_bo:
    evaluate_fn = partial(evaluate_for_ga, cfg=cfg_rl)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .rl_config import RLConfig
from .rl_env import BWREnrichmentEnv
from .rl_policy import build_agent, policy_summary


def _mask_fn(env: BWREnrichmentEnv) -> np.ndarray:
    return env.action_masks()


def make_env(cfg: RLConfig, evaluate_fn: Callable[[np.ndarray], dict]):
    """
    Build the wrapped environment MaskablePPO expects: raw BWREnrichmentEnv -> ActionMasker (exposes action_masks() to the algorithm) -> Monitor
    (so SB3's callback `infos`/`dones` plumbing works as used below).
    """
    env = BWREnrichmentEnv(cfg, evaluate_fn)
    env = ActionMasker(env, _mask_fn)
    env = Monitor(env)
    return env


class _EpisodeLogger(BaseCallback):
    """
    Collects one row per completed episode (== one OpenMC evaluation, since each episode terminates after exactly cfg.n_sym_rods steps). 
    Reads the terminal `info` dict BWREnrichmentEnv.step() attaches (keff, ppf, fitness, chromosome).

    Coded for a single environment (n_envs=1, the default in run_rl()).
    To extend it to vectorised/parallel envs requires only iterating `self.locals["infos"]`/`self.locals["dones"]` as lists. 
    """

    def __init__(self, cfg: RLConfig, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.all_chromosomes: list = []
        self.all_keff: list = []
        self.all_keff_std: list = []
        self.all_ppf: list = []
        self.all_fitness: list = []
        self.window_best_fitness: list = []
        self.window_mean_fitness: list = []
        self.window_best_keff: list = []       # k_inf of the best-fitness episode in each window
        self.window_best_keff_std: list = []   #  OpenMC statistical uncertainty
        self.window_best_ppf: list = []        # PPF (no uncertainty tracked)
        self._window_buffer: list = []
        self._window_buffer_keff: list = []
        self._window_buffer_keff_std: list = []
        self._window_buffer_ppf: list = []
 
        self.hof_chromosome: Optional[np.ndarray] = None
        self.hof_keff = np.inf
        self.hof_keff_std = 0.0
        self.hof_ppf = np.inf
        self.hof_fitness = np.inf

#********************************** need to cross check for bugs ************************
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done or "fitness" not in info:
                continue

            # STORE GLOBAL HISTORY
            self.all_chromosomes.append(np.array(info["chromosome"]))
            self.all_keff.append(info["keff"])
            self.all_keff_std.append(info.get("keff_std", 0.0))
            self.all_ppf.append(info["ppf"])
            self.all_fitness.append(info["fitness"])
            
            # ADD DIRECTLY FROM 'info' DICT TO THE WINDOW BUFFERS
            self._window_buffer.append(info["fitness"])
            self._window_buffer_keff.append(info["keff"])            
            self._window_buffer_keff_std.append(info.get("keff_std", 0.0))   
            self._window_buffer_ppf.append(info.get("ppf", 0.0))          

            # HALL OF FAME (HOF) UPDATES
            if info["fitness"] < self.hof_fitness:
                self.hof_fitness    = float(info["fitness"])
                self.hof_keff       = float(info["keff"])
                self.hof_keff_std   = float(info.get("keff_std", 0.0))
                self.hof_ppf        = float(info["ppf"])
                self.hof_chromosome = np.array(info["chromosome"])

            # PRINT AND CLEAR BUFFERS EVERY N EPISODES
            if len(self._window_buffer) >= self.cfg.log_every_n_episodes:
                self.window_best_fitness.append(float(np.min(self._window_buffer)))
                self.window_mean_fitness.append(float(np.mean(self._window_buffer)))
                
                if self.verbose:
                    n_ep = len(self.all_fitness)
                    lo = n_ep - len(self._window_buffer) + 1
                    
                    # MUST calculate these variables before printing them. 
                    best_fit = np.min(self._window_buffer)
                    mean_fit = np.mean(self._window_buffer)
                    
                    # Best episode in the window, to get its physics values
                    best_idx = np.argmin(self._window_buffer)
                    best_k = self._window_buffer_keff[best_idx]
                    best_k_err = self._window_buffer_keff_std[best_idx]
                    best_ppf = self._window_buffer_ppf[best_idx]
                    
                    print(f"Episodes {lo:4d}-{n_ep:<4d} : "
                          f"best_fit={best_fit:.5f}  mean_fit={mean_fit:.5f}  "
                          f"k∞={best_k:.5f} +/- {best_k_err:.5f}, PPF={best_ppf:.4f}  "
                          f"[HOF fit={self.hof_fitness:.5f}]")

                # CLEAR ALL BUFFERS for the next batch
                self._window_buffer.clear()
                self._window_buffer_keff.clear()       
                self._window_buffer_keff_std.clear()   
                self._window_buffer_ppf.clear()        
                    
        return True

#$***************************************************************************************
    
def run_rl(
    cfg: RLConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    verbose: bool = True,
    #progress_bar: bool = False,
) -> dict:
    '''
    Train MaskablePPO -> Builds BWR core layouts.

        - verbose      -> Toggles local print logs (0 dependencies).
        - progress_bar -> Toggles SB3's tqdm bar (requires `tqdm rich`, defaults False).
        - Returns Dict -> Matches GA/BO output format (history, best_design, eval_count, k∞, Δk∞, ppf) 
                  + includes the trained `model` for saving/prediction.
    '''

    env = make_env(cfg, evaluate_fn)
    model = build_agent(env, cfg, verbose=0)

    if verbose:
        print(cfg.summary())
        print()
        print(policy_summary(model, cfg))
        print("=" * 70)
        print(f"  REINFORCEMENT LEARNING — {cfg.model_name}")
        print(f"  Assembly   : {cfg.n_rods_side}x{cfg.n_rods_side}  ({cfg.n_sym_rods} sym positions)")
        print(f"  Algorithm  : MaskablePPO  (sb3-contrib)")
        print("=" * 70)

    logger_cb = _EpisodeLogger(cfg, verbose=1 if verbose else 0)
    model.learn(total_timesteps=cfg.total_timesteps, callback=logger_cb, progress_bar=verbose)  #––––––––––––––––> 

    n_episodes = len(logger_cb.all_fitness)
    if verbose:
        _print_rl_summary(cfg, logger_cb, n_episodes)

    return dict(
        model               = model,
        env                 = env,
        all_chromosomes     = np.array(logger_cb.all_chromosomes),
        all_keff            = np.array(logger_cb.all_keff),
        all_keff_std        = np.array(logger_cb.all_keff_std),
        all_ppf             = np.array(logger_cb.all_ppf),
        all_fitness         = np.array(logger_cb.all_fitness),
        window_best_fitness = np.array(logger_cb.window_best_fitness),
        window_mean_fitness = np.array(logger_cb.window_mean_fitness),
        hof_chromosome      = logger_cb.hof_chromosome,
        hof_keff            = logger_cb.hof_keff,
        hof_keff_std        = logger_cb.hof_keff_std,
        hof_ppf             = logger_cb.hof_ppf,
        hof_fitness         = logger_cb.hof_fitness,
        n_evaluations       = n_episodes,
        total_timesteps     = cfg.total_timesteps,
    )

#*******************************88 #SMOKE TEST ***************************************\

def run_rl_smoke_test(cfg: RLConfig, evaluate_fn: Callable[[np.ndarray], dict]) -> dict:
    
    _prod = (cfg.n_particles, cfg.n_inactive, cfg.n_active)
    cfg.n_particles = cfg.n_particles_smoke
    cfg.n_inactive  = cfg.n_inactive_smoke
    cfg.n_active    = cfg.n_active_smoke

    env = BWREnrichmentEnv(cfg, evaluate_fn)
    rng = np.random.default_rng(cfg.random_seed)
    env.reset(seed=cfg.random_seed)

    print("=" * 70)
    print(f"  RL SMOKE TEST — {cfg.model_name}  (untrained / random-feasible policy)")
    print(f"  Assembly : {cfg.n_rods_side}x{cfg.n_rods_side}  ({cfg.n_sym_rods} sym positions)")
    print("=" * 70)

    terminated = False
    info: dict = {}
    while not terminated:
        mask = env.action_masks()
        action = int(rng.choice(np.flatnonzero(mask)))
        _, _, terminated, _, info = env.step(action)

    enr_grid = info["enr_grid"]
    n_high   = int((np.round(enr_grid, 4) == round(cfg.enr_high, 4)).sum())
    print(f"  Decoded grid : {n_high} x {cfg.enr_high}%  +  "
          f"{cfg.n_rods_total - n_high} x {cfg.enr_low}%  (avg {enr_grid.mean():.4f} wt%)")
    print(f"  Particles    : {cfg.n_particles}  ({cfg.n_inactive} inactive + {cfg.n_active} active)")
    print(f"  k_inf    = {info['keff']:.5f}   (target = {cfg.k_target})")
    print(f"  |dk_inf| = {abs(cfg.k_target - info['keff']):.5f}")
    if cfg.ppf_target is not None:
        feas = "feasible" if info["ppf"] <= cfg.ppf_target else "NOT feasible"
        print(f"  PPF      = {info['ppf']:.4f}   (limit <= {cfg.ppf_target})  {feas}")
    print(f"  Fitness  = {info['fitness']:.5f}  (lower is better; 0 = perfect)")
    print("=" * 70)

    cfg.n_particles, cfg.n_inactive, cfg.n_active = _prod
    return info


#Print Summary

def _print_rl_summary(cfg: RLConfig, logger_cb: _EpisodeLogger, n_episodes: int) -> None:
    print("\n" + "=" * 70)
    print(f"  RL SUMMARY — {cfg.model_name}  [{cfg.n_rods_side}x{cfg.n_rods_side}]")
    print("=" * 70)
    print(f"  Best k∞         : {logger_cb.hof_keff:.5f}")
    print(f"  k∞ target       : {cfg.k_target}")
    print(f"  |Δk∞|           : {abs(cfg.k_target - logger_cb.hof_keff):.5f}")
    if cfg.ppf_target is not None:
        sat = "SATISFIED" if logger_cb.hof_ppf <= cfg.ppf_target else "NOT SATISFIED"
        print(f"  PPF                : {logger_cb.hof_ppf:.4f}  (limit <= {cfg.ppf_target})  {sat}")
    print(f"  Best fitness       : {logger_cb.hof_fitness:.6f}")
    print(f"  Total OpenMC calls : {n_episodes}  "
          f"({cfg.total_timesteps:,} timesteps / {cfg.n_sym_rods} steps per episode)")
    print(f"  Best chromosome    : {list(logger_cb.hof_chromosome) if logger_cb.hof_chromosome is not None else None}")
    print("=" * 70)
