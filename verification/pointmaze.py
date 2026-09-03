"""Continuous Four Rooms for HAC verification.

A 2D point mass in [-1, 1]^2 divided into four rooms by a vertical wall at
x = 0 and a horizontal wall at y = 0, each pierced by two doorways. The agent
always starts in the bottom-left room; the end goal is sampled uniformly over
the whole arena, so most goals sit in a different room and can only be reached
by routing through doorways.

This is the point-mass reduction of Levy et al.'s "ant four rooms" -- the
domain their multi-level result is strongest on. The dynamics are deliberately
trivial (the action moves the agent directly) so that nothing about low-level
motor control confounds the measurement: the only difficulty is temporally
extended credit assignment over a sparse reward, which is exactly the
difficulty HAC's hierarchy claims to address.

Reward is never used by HAC -- it builds its own sparse goal reward from a
projection of the observation -- and `done` is always False, so the algorithm
owns the episode boundary just as it does on brax.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

BOUND = 1.0        # arena is [-BOUND, BOUND]^2
STEP_SCALE = 0.01  # units moved per unit action -> ~200 steps to cross the arena
DOOR = 0.15        # half-width of each doorway
DOOR_POS = 0.5     # doorways centred at +-DOOR_POS along each wall
START = jnp.array([-0.75, -0.75])   # bottom-left room
START_NOISE = 0.1


class MazeState(NamedTuple):
    pos: jax.Array   # (2,)


def _through_doorway(coord):
    """True where a wall crossing at this coordinate lands in a doorway."""
    return (jnp.abs(coord - DOOR_POS) < DOOR) | (jnp.abs(coord + DOOR_POS) < DOOR)


def step_pos(pos, action):
    """Move, blocking each axis independently if it would cross a wall outside
    a doorway. Axis-independent blocking means the agent slides along walls
    rather than sticking, which keeps the task about navigation, not contact."""
    target = pos + STEP_SCALE * jnp.clip(action, -1.0, 1.0)

    # Crossing the vertical wall at x = 0 is allowed only through a doorway in y.
    crosses_x = jnp.sign(pos[0]) != jnp.sign(target[0])
    block_x = crosses_x & ~_through_doorway(target[1])
    new_x = jnp.where(block_x, pos[0], target[0])

    # Crossing the horizontal wall at y = 0 is allowed only through a doorway in x.
    crosses_y = jnp.sign(pos[1]) != jnp.sign(target[1])
    block_y = crosses_y & ~_through_doorway(new_x)
    new_y = jnp.where(block_y, pos[1], target[1])

    return jnp.clip(jnp.array([new_x, new_y]), -BOUND, BOUND)


def reset_fn(key):
    pos = START + START_NOISE * jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
    pos = jnp.clip(pos, -BOUND, BOUND)
    return pos, MazeState(pos=pos)


def step_fn(key, state, action):
    pos = step_pos(state.pos, action)
    return pos, MazeState(pos=pos), jnp.float32(0.0), jnp.bool_(False), {}


def make_wrapped_env(JaxWrappedEnv):
    """Build the repo's JaxWrappedEnv around this maze. Takes the NamedTuple
    class as an argument so this module stays importable without the repo."""
    return JaxWrappedEnv(
        env=None, env_params=None, state_dim=2, num_actions=0,
        reset_fn=reset_fn, step_fn=step_fn,
        cumulant_fn=None, goal_fn=None, num_goals=0,
        goal_kind=None, goal_target=None, goal_reached_fn=None,
        action_dim=2,
        action_low=-jnp.ones((2,), jnp.float32),
        action_high=jnp.ones((2,), jnp.float32),
    )


def room_of(pos):
    """Room index 0..3 for a position -- used to report how often the end goal
    lies outside the agent's starting room."""
    return (pos[..., 0] > 0).astype(jnp.int32) + 2 * (pos[..., 1] > 0).astype(jnp.int32)
