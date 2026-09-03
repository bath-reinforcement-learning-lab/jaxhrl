"""
Four-rooms *navigation* task for the Option-Critic verification.

This is the environment from the Option-Critic paper (Bacon, Harb & Precup,
AAAI 2017, Section 5.1 / Figures 3-4): a 13x13 four-rooms gridworld where the
agent must reach a fixed goal cell, the goal is relocated partway through
training, and the paper measures how fast a hierarchical option-learning agent
recovers relative to a flat one.

It shares the exact wall layout / free-cell enumeration of the DCEO oracle
(`verification/fourrooms.py`) but adds a goal, a terminating +1 goal reward,
the paper's 1/3 action noise, and gymnax-style vmap-able `reset_fn` / `step_fn`
with auto-reset-on-done, so the repo's real `batch_select_option_critic_action`
can roll it out unchanged.

The goal cell is passed to `reset_fn` / `step_fn` as a traced `goal_idx`
argument (not baked into a closure), so the training loop can relocate the goal
without triggering a JIT recompile. `GOAL_A` / `GOAL_B` are the pre- and
post-relocation goal cells.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

N = 13
ACTION_NOISE = 1.0 / 3.0          # paper: with prob 1/3 the move is a random direction
HORIZON = 150
GAMMA = 0.99

# up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_ACTIONS = 4


def _build_walls():
    walls = np.zeros((N, N), dtype=bool)
    walls[0, :] = True
    walls[N - 1, :] = True
    walls[:, 0] = True
    walls[:, N - 1] = True
    walls[1:N - 1, 6] = True   # vertical divider
    walls[6, 1:N - 1] = True   # horizontal divider
    for (r, c) in [(3, 6), (9, 6), (6, 3), (6, 9)]:  # one doorway per arm of the cross
        walls[r, c] = False
    return walls


WALLS = _build_walls()
FREE_CELLS = [(r, c) for r in range(N) for c in range(N) if not WALLS[r, c]]
NUM_STATES = len(FREE_CELLS)                       # 104
CELL_TO_IDX = {cell: i for i, cell in enumerate(FREE_CELLS)}
STATE_DIM = NUM_STATES

DOORWAYS = [(3, 6), (9, 6), (6, 3), (6, 9)]
DOORWAY_IDX = [CELL_TO_IDX[c] for c in DOORWAYS]

# The paper's four-rooms transfer test: the goal starts in the east doorway (a
# bottleneck -- options that learn to reach it stay reusable as navigation
# skills) and, once the agent has learned, relocates into the lower-right room.
# After the move the flat policy's saturated softmax points the wrong way with
# only per-step action noise to escape, whereas Option-Critic can also explore
# at the option level (epsilon-greedy over Q_Omega, committing to a whole
# option per episode) -- the benefit the paper credits to options.
GOAL_A = (6, 9)      # east doorway (between the two right-hand rooms)
GOAL_B = (10, 10)    # lower-right room


def _transition_table():
    """`nxt[state_idx, action]` -> next state_idx, with wall bumps staying put."""
    nxt = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
    for cell, i in CELL_TO_IDX.items():
        r, c = cell
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and not WALLS[nr, nc]:
                nxt[i, a] = CELL_TO_IDX[(nr, nc)]
            else:
                nxt[i, a] = i
    return nxt


_NXT = jnp.asarray(_transition_table())


class NavState(NamedTuple):
    pos: jnp.ndarray   # scalar int32 -- free-cell index
    t: jnp.ndarray     # scalar int32 -- steps elapsed this episode


def _one_hot(idx):
    return jax.nn.one_hot(idx, NUM_STATES, dtype=jnp.float32)


def reset_fn(key, goal_idx):
    # Start states are drawn uniformly over the free cells other than the goal.
    # The paper uses a fixed start; uniform starts act as a curriculum that
    # lets the deep network learn the task in a tractable number of steps
    # (value bootstraps back from envs that start near the goal). The transfer
    # claim under test is unaffected.
    pos = jax.random.randint(key, (), 0, NUM_STATES)
    pos = jnp.where(pos == goal_idx, (pos + 1) % NUM_STATES, pos)   # never start on the goal
    return _one_hot(pos), NavState(pos=pos, t=jnp.array(0, dtype=jnp.int32))


def step_fn(key, state, action, goal_idx):
    noise_key, rand_key, reset_key = jax.random.split(key, 3)
    slipped = jax.random.uniform(noise_key) < ACTION_NOISE
    rand_action = jax.random.randint(rand_key, (), 0, NUM_ACTIONS)
    eff_action = jnp.where(slipped, rand_action, action)

    new_pos = _NXT[state.pos, eff_action]
    new_t = state.t + 1
    reached = new_pos == goal_idx
    reward = reached.astype(jnp.float32)
    done = reached | (new_t >= HORIZON)

    stepped = NavState(pos=new_pos, t=new_t)
    reset_obs, reset_state = reset_fn(reset_key, goal_idx)
    final_state = jax.tree_util.tree_map(
        lambda r, s: jnp.where(done, r, s), reset_state, stepped
    )
    final_obs = jnp.where(done, reset_obs, _one_hot(new_pos))
    return final_obs, final_state, reward, done, {"reached_goal": reached}


GOAL_A_IDX = CELL_TO_IDX[GOAL_A]
GOAL_B_IDX = CELL_TO_IDX[GOAL_B]
