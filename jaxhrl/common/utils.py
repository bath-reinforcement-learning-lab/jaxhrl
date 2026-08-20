import argparse
import os.path as osp
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium
import numpy as np
import yaml
import brll_core.environments.common.wrappers
from brll_core.algorithms.common.logger import Logger
from brll_core.environments.common.protocols import ImplementsGetActionMask


class Transition(NamedTuple):
    state: object
    action: object
    reward: float
    next_state: object
    terminated: bool
    truncated: bool


def parse_config() -> dict:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Select from `configs/*.yaml`",
    )
    args.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Overwrite config seed.",
    )
    args = args.parse_args()

    with open(Path(args.config)) as file:
        config = yaml.safe_load(file)

    if hasattr(args, "seed") and args.seed is not None:
        config["seed"] = args.seed

    return config


def make_env(
    make_kwargs: dict,
    wrappers: dict[str, dict],
    seed: int,
    rank: str | None = None,
    record_video: bool = False,
    logger: Logger | None = None,
) -> gymnasium.Env:
    assert (not record_video) or (record_video and Logger), "Logger must be provided if record_video is True."  # fmt: off

    env = gymnasium.make(**make_kwargs)

    if record_video:
        save_dir = osp.join(logger.get_artifact_path(), "videos")
        env = brll_core.environments.common.wrappers.ToggleableRecordVideo(
            env,
            video_folder=save_dir,
            name_prefix=f"env-{rank}" if rank is not None else "video",
            episode_trigger=lambda x: True,
        )

    for wrapper_name, wrapper_kwargs in wrappers.items():
        if hasattr(gymnasium.wrappers, wrapper_name):
            wrapper = getattr(gymnasium.wrappers, wrapper_name)
        elif hasattr(brll_core.environments.common.wrappers, wrapper_name):
            wrapper = getattr(brll_core.environments.common.wrappers, wrapper_name)
        else:
            raise ValueError(f"Wrapper '{wrapper_name}' not found in gymnasium or brll_core wrappers.")
        env = wrapper(env, **wrapper_kwargs)

    env.action_space.seed(seed)
    return env


def get_make_env_fn(*args, **kwargs) -> Callable:
    """
    Returns a function that can be used to create an environment with the given arguments.
    This is useful for passing to vectorized environments or other APIs that require a callable.
    """
    return lambda: make_env(*args, **kwargs)





def discounted_return(rewards: list[float], gamma: float) -> float:
    """
    Given a list of rewards and a discount factor, computes the discounted sum of those rewards.

    Args:
        rewards (List[Number]): The list of rewards, where rewards[i] is the reward at time step i.
        gamma (float): The discount factor.

    Returns:
        Number: The discounted sum of rewards.

    Remarks:
        After testing many different variations of this function, I found that this
        pure-Python implementation is the fastest. Yes, faster than any reasonable NumPy equivalent.
        It is also the most readable. Do not try to "optimise" it without good reason.
    """
    discounted_sum_of_rewards = 0.0
    gamma_power = 1.0

    for reward in rewards:
        discounted_sum_of_rewards += reward * gamma_power
        gamma_power *= gamma

    return discounted_sum_of_rewards





def has_wrapper(
    env: gymnasium.Env | gymnasium.vector.SyncVectorEnv,
    wrapper_class: type[gymnasium.Wrapper],
) -> bool:
    """Check recursively if env is wrapped with wrapper_type."""
    current = env if isinstance(env, gymnasium.Env) else env.envs[0]
    while isinstance(current, gymnasium.Wrapper):
        if isinstance(current, wrapper_class):
            return True
        current = current.env
    return False


def get_action_mask(env: gymnasium.Env, obs: object) -> np.ndarray:
    """_summary_

    Args:
        env (Env): _description_
        obs (object): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: _description_
    """
    if isinstance(env.unwrapped, ImplementsGetActionMask):
        return env.unwrapped.get_action_mask(obs)
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        return np.ones((env.action_space.n,), dtype=bool)
    else:
        raise NotImplementedError(
            f"{env} can not be used with QLearning.get_action_mask(). Either use the ImplementsGetActionMask protocol or a different learning algorithm. "
        )
