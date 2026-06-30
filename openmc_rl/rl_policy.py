"""
rl_policy.py — MaskablePPO construction
-----------------------------------------

Only file in the package that imports ––> torch / sb3_contrib / stable_baselines3. 

 - This package doesn't hand-roll its own neural network or PPO update rule 
 - sb3-contrib implements PPO + action masking.
 - rl_policy.py––> translates an RLConfig into the right constructor arguments for MaskablePPO.

Policy network ––> Standard, small MLP (shared actor/critic heads)
                   (Small since state/action spaces are tiny (tens of inputs, 2 outputs).) 
                
"""

from __future__ import annotations

from typing import Optional
#*********************************************

import sys as _sys

if not hasattr(_sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300

    _sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]

if not hasattr(_sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        return None

    _sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]


#******************************************************

import gymnasium as gym
from sb3_contrib import MaskablePPO

from .rl_config import RLConfig


def build_agent(
    env: gym.Env,
    cfg: RLConfig,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
) -> MaskablePPO:
    """
    Construct a MaskablePPO agent wired to `env` 
    """
    policy_kwargs = dict(net_arch=list(cfg.policy_hidden_sizes))

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        policy_kwargs=policy_kwargs,
        seed=cfg.random_seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
    return model


def policy_summary(model: MaskablePPO, cfg: RLConfig) -> str:
    n_params = sum(p.numel() for p in model.policy.parameters())
    n_trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    lines = [
        f"MaskablePPO policy — {cfg.model_name}",
        f"  Hidden layers (shared actor/critic MLP) : {cfg.policy_hidden_sizes}",
        f"  Total parameters                        : {n_params:,}  ({n_trainable:,} trainable)",
        f"  Learning rate                           : {cfg.learning_rate}",
        f"  Rollout length / batch / epochs          : {cfg.n_steps} / {cfg.batch_size} / {cfg.n_epochs}",
        f"  Gamma / GAE-lambda / clip range          : {cfg.gamma} / {cfg.gae_lambda} / {cfg.clip_range}",
        f"  Entropy coef / value-fn coef             : {cfg.ent_coef} / {cfg.vf_coef}",
        f"  Total training timesteps                : {cfg.total_timesteps:,}",
    ]
    return "\n".join(lines)
