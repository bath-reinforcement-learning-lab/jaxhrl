"""
Reward-free 13x13 FourRooms for the METRA verification.

METRA is unsupervised: it never sees a task reward, only its own intrinsic
skill reward `(phi(s') - phi(s)) . z`. So this environment exposes no goal and
no reward -- just one-hot observations, deterministic 4-connected movement,
and a fixed-horizon auto-reset. It is the gridworld analogue of the paper's
locomotion domains: the thing METRA should recover is a low-dimensional map of
this state space whose geometry matches the *temporal* (shortest-path)
distance between cells -- which, thanks to the doorways, is very different from
Euclidean distance in the grid.

Layout / adjacency / free-cell indexing are reused from `fourrooms.py` (the
DCEO oracle). `graph_dist` is the exact all-pairs shortest-path matrix over the
4-connected free cells -- the ground-truth temporal-distance metric for Test B.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

import fourrooms as fr

N = fr.N
NUM_STATES = fr.NUM_STATES          # 104
FREE_CELLS = fr.FREE_CELLS
CELL_TO_IDX = fr.CELL_TO_IDX
WALLS = fr.WALLS
ACTIONS = fr.ACTIONS                # up, down, left, right
NUM_ACTIONS = 4
STATE_DIM = NUM_STATES
HORIZON = 50

# rooms, for colouring Test B plots: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
ROOM_OF = np.array([(0 if r < 6 else 2) + (0 if c < 6 else 1) for (r, c) in FREE_CELLS])


def _transition_table():
    nxt = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
    for (r, c), i in CELL_TO_IDX.items():
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            nxt[i, a] = CELL_TO_IDX[(nr, nc)] if (nr, nc) in CELL_TO_IDX else i
    return nxt


_NXT = jnp.asarray(_transition_table())


def graph_dist():
    """Exact all-pairs shortest-path distance over the 4-connected free cells."""
    A = fr.build_adjacency()
    INF = NUM_STATES + 1
    D = np.where(A > 0, 1, INF).astype(np.int32)
    np.fill_diagonal(D, 0)
    for k in range(NUM_STATES):            # Floyd-Warshall (104^3, fine once)
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    return D


GRAPH_DIST = graph_dist()


class OpenState(NamedTuple):
    pos: jnp.ndarray   # scalar int32 free-cell index
    t: jnp.ndarray     # scalar int32 steps this episode


def _one_hot(idx):
    return jax.nn.one_hot(idx, NUM_STATES, dtype=jnp.float32)


def reset_fn(key):
    pos = jax.random.randint(key, (), 0, NUM_STATES)
    return _one_hot(pos), OpenState(pos=pos, t=jnp.array(0, dtype=jnp.int32))


def step_fn(key, state, action):
    new_pos = _NXT[state.pos, action]
    new_t = state.t + 1
    done = new_t >= HORIZON
    stepped = OpenState(pos=new_pos, t=new_t)
    reset_obs, reset_state = reset_fn(jax.random.fold_in(key, 0))
    final_state = jax.tree_util.tree_map(
        lambda rr, ss: jnp.where(done, rr, ss), reset_state, stepped
    )
    final_obs = jnp.where(done, reset_obs, _one_hot(new_pos))
    return final_obs, final_state, jnp.float32(0.0), done, {"pos": new_pos}


def make_wrapped_env(JaxWrappedEnv):
    """Wrap this env in the repo's `JaxWrappedEnv` NamedTuple so a patched
    `make_jax_env` can hand it to METRA's real training loop unchanged."""
    return JaxWrappedEnv(
        env=None, env_params=None, state_dim=STATE_DIM, num_actions=NUM_ACTIONS,
        reset_fn=reset_fn, step_fn=step_fn,
        cumulant_fn=None, goal_fn=None, num_goals=0,
        goal_kind=None, goal_target=None, goal_reached_fn=None,
        action_dim=0,
    )
