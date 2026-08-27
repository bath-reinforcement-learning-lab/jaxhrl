"""
Custom stand-in for the HiPPO paper's Block/Gather environments.

`jaxhrl/HiPPO.py` only ever samples actions via `jax.random.categorical` --
it is discrete/categorical-action only, wired to gymnax/Craftax-style envs.
The paper's own test environments (Block Hopper/Half Cheetah, Snake/Ant
Gather) are continuous-action MuJoCo robots, so they aren't reachable here
without either changing the algorithm (out of scope for verification) or
building full MuJoCo clones. This environment is built instead to isolate,
as cleanly as possible, the specific property the paper's own time-
commitment ablation (Section 5.2, Figure 3) demonstrates: a persistent
decision (HiPPO's skill index, held fixed for a whole commitment period)
carries information forward through time that a purely reactive,
memoryless per-step decision cannot.

Each episode: a target direction (one of `NUM_DIRECTIONS`) is drawn at
random and revealed in the observation for only the first `REVEAL_STEPS`
steps; after that the observation goes blank (an all-zero cue) for the rest
of the episode, which lasts up to `HORIZON` steps. The agent moves by a unit
step in one of `NUM_DIRECTIONS` per action, with NO position/velocity
feedback at all -- the only way to reach the target (`GOAL_DISTANCE` steps
away, within `HORIZON`) is to correctly read the brief cue and then keep
repeating that same action for the remaining, much longer blank stretch,
open-loop.

This makes the environment a genuine POMDP that structurally favors HiPPO's
mechanism: a manager that commits to a skill for a whole episode-length
commitment period (`p_min` set safely above `REVEAL_STEPS`) reads the cue
once, and the *skill index itself* -- not anything in the observation --
carries the right answer forward for the rest of the episode. A manager
forced to redecide every single step (`p=1`) is, like the skill network
itself, purely reactive: once the cue goes blank it has no information left
to distinguish this episode from any other, so it can only fall back to one
global default action for the remaining, uninformative steps -- exactly the
mechanism the paper points to when its own p=1 ablation collapses towards
flat-PPO-like performance. A flat, non-hierarchical policy (no manager, no
persistent skill index at all) faces exactly the same limitation.

Matches gymnax's `Environment.step` auto-reset-on-done convention (used
directly as reset_fn/step_fn elsewhere in this repo's
jaxhrl/common/wrappers.py), so it's a drop-in for HiPPO.py's rollout code.
Episode boundaries (`done`) also force a fresh manager decision in HiPPO.py's
own rollout logic, so the cue-reveal window and a mandatory redecision are
naturally aligned at the start of every episode regardless of `p`.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple

NUM_DIRECTIONS = 4
REVEAL_STEPS = 2
GOAL_DISTANCE = 5
HORIZON = 8

NUM_ACTIONS = NUM_DIRECTIONS
STATE_DIM = NUM_DIRECTIONS  # the cue only -- blank (all zero) once t >= REVEAL_STEPS

_DIRS = jnp.array(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=jnp.float32
)


class CompassState(NamedTuple):
    pos: jnp.ndarray         # (2,) float32 -- hidden from the observation
    target_dir: jnp.ndarray  # scalar int32
    t: jnp.ndarray           # scalar int32, steps elapsed this episode


def _obs_of(state):
    cue_visible = (state.t < REVEAL_STEPS).astype(jnp.float32)
    return jax.nn.one_hot(state.target_dir, NUM_DIRECTIONS) * cue_visible


def reset_fn(key):
    target_dir = jax.random.randint(key, (), 0, NUM_DIRECTIONS)
    state = CompassState(
        pos=jnp.zeros(2, dtype=jnp.float32),
        target_dir=target_dir,
        t=jnp.array(0, dtype=jnp.int32),
    )
    return _obs_of(state), state


def step_fn(key, state, action):
    new_pos = state.pos + _DIRS[action]
    new_t = state.t + 1

    target_pos = _DIRS[state.target_dir] * GOAL_DISTANCE
    reached = jnp.linalg.norm(new_pos - target_pos) < 0.5
    reward = reached.astype(jnp.float32)
    done = reached | (new_t >= HORIZON)

    stepped_state = CompassState(pos=new_pos, target_dir=state.target_dir, t=new_t)
    stepped_obs = _obs_of(stepped_state)

    reset_key, _ = jax.random.split(key)
    reset_obs, reset_state = reset_fn(reset_key)

    final_state = jax.tree_util.tree_map(
        lambda r, s: jnp.where(done, r, s), reset_state, stepped_state
    )
    final_obs = jnp.where(done, reset_obs, stepped_obs)
    infos = {"reached_goal": reached}
    return final_obs, final_state, reward, done, infos
