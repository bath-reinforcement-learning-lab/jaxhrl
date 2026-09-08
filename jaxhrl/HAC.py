# HAC in JAX
# From "Learning Multi-Level Hierarchies with Hindsight"
from typing import Any, NamedTuple
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from jaxhrl.common.utils import parse_config
from jaxhrl.common.logger import Logger
from jaxhrl.common.wrappers import make_jax_env


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nnx.Module):
    def __init__(self, obs_dim: int, goal_dim: int, out_dim: int,
                 out_low, out_high, obs_scale, goal_scale,
                 hidden_dim: int = 256, *, rngs: nnx.Rngs):
        self.out_low = tuple(float(x) for x in out_low)
        self.out_high = tuple(float(x) for x in out_high)
        self.obs_scale = tuple(float(x) for x in obs_scale)
        self.goal_scale = tuple(float(x) for x in goal_scale)

        self.linear1 = nnx.Linear(obs_dim + goal_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, obs: jax.Array, goal: jax.Array) -> jax.Array:
        x = jnp.concatenate([
            obs.astype(jnp.float32) / jnp.asarray(self.obs_scale),
            goal.astype(jnp.float32) / jnp.asarray(self.goal_scale),
        ], axis=-1)
        x = nnx.relu(self.ln1(self.linear1(x)))
        x = nnx.relu(self.ln2(self.linear2(x)))
        low, high = jnp.asarray(self.out_low), jnp.asarray(self.out_high)
        centre, half_range = (high + low) / 2.0, (high - low) / 2.0
        return centre + half_range * nnx.tanh(self.linear3(x))


class Critic(nnx.Module):

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, q_limit: float,
                 obs_scale, goal_scale, act_scale,
                 hidden_dim: int = 256, *, rngs: nnx.Rngs):
        self.q_limit = float(q_limit)
        self.obs_scale = tuple(float(x) for x in obs_scale)
        self.goal_scale = tuple(float(x) for x in goal_scale)
        self.act_scale = tuple(float(x) for x in act_scale)

        self.linear1 = nnx.Linear(obs_dim + goal_dim + act_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs: jax.Array, goal: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([
            obs.astype(jnp.float32) / jnp.asarray(self.obs_scale),
            goal.astype(jnp.float32) / jnp.asarray(self.goal_scale),
            action.astype(jnp.float32) / jnp.asarray(self.act_scale),
        ], axis=-1)
        x = nnx.relu(self.ln1(self.linear1(x)))
        x = nnx.relu(self.ln2(self.linear2(x)))
        return -nnx.sigmoid(self.linear3(x)).squeeze(-1) * self.q_limit


# ---------------------------------------------------------------------------
# Goal space
# ---------------------------------------------------------------------------

def project(obs: jax.Array, goal_indices: jax.Array) -> jax.Array:
    """State -> goal-space projection. Used UNCLIPPED for the achievement test
    and CLIPPED (see clip_to_goal_space) when stored as a hindsight action."""
    return obs[..., goal_indices]


def clip_to_goal_space(proj: jax.Array, goal_low: jax.Array, goal_high: jax.Array) -> jax.Array:
    """Levy's project_state_to_subgoal clips to the subgoal bounds. Without
    this the agent wanders outside the box, the stored hindsight action lands
    outside the actor's tanh-reachable range, and the deterministic policy
    gradient pins the actor to the boundary forever."""
    return jnp.clip(proj, goal_low, goal_high)


def goal_reached(achieved: jax.Array, goal: jax.Array, threshold: jax.Array) -> jax.Array:
    return jnp.all(jnp.abs(achieved - goal) < threshold, axis=-1)


def select_action(actor, obs, goal, low, high, deterministic, key,
                  noise_perc: float, random_perc: float):
    """Levy's exploration: with probability random_perc a uniform sample over
    the level's bounds, otherwise policy + N(0, noise_perc*(high-low)/2). A
    level under subgoal test acts deterministically, and that propagates to
    every level beneath it."""
    a = actor(obs, goal)
    noise_key, uniform_key, branch_key = jax.random.split(key, 3)

    sigma = noise_perc * (high - low) / 2.0
    noisy = a + sigma * jax.random.normal(noise_key, a.shape)
    random = jax.random.uniform(uniform_key, a.shape, minval=low, maxval=high)

    use_random = jax.random.uniform(branch_key, (a.shape[0],)) < random_perc
    out = jnp.where(use_random[:, None], random, noisy)
    out = jnp.where(deterministic[:, None], a, out)
    return jnp.clip(out, low, high)


# ---------------------------------------------------------------------------
# DDPG losses and update (module scope, so verification/repo_loader.py can
# import them without running __main__)
# ---------------------------------------------------------------------------

def critic_loss_fn(critic, target_critic, target_actor, batch, gamma: float, q_limit: float):
    next_action = target_actor(batch["next_obs"], batch["goal"])
    q_next = target_critic(batch["next_obs"], batch["goal"], next_action)
    # discount is 0 iff the goal was reached (or this is a subgoal-test penalty).
    # Hitting a level's action horizon is a timeout, not a terminal, so those
    # transitions bootstrap normally.
    target = batch["reward"] + gamma * batch["discount"] * q_next
    target = jnp.clip(jax.lax.stop_gradient(target), -q_limit, 0.0)
    q_pred = critic(batch["obs"], batch["goal"], batch["action"])
    return jnp.mean((q_pred - target) ** 2)


def actor_loss_fn(actor, critic, batch):
    action = actor(batch["obs"], batch["goal"])
    return -jnp.mean(critic(batch["obs"], batch["goal"], action))


def soft_update(target, online, tau: float):
    nnx.update(target, jax.tree.map(
        lambda t, o: tau * o + (1.0 - tau) * t,
        nnx.state(target, nnx.Param), nnx.state(online, nnx.Param),
    ))


def train_level_step(actor, critic, target_actor, target_critic,
                     opt_actor, opt_critic, batch,
                     gamma: float, q_limit: float, tau: float):
    """One DDPG update for a single level: critic first, then the actor against
    the freshly-updated critic, then both Polyak updates."""
    critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(
        critic, target_critic, target_actor, batch, gamma, q_limit
    )
    opt_critic.update(critic, critic_grads)

    actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(actor, critic, batch)
    opt_actor.update(actor, actor_grads)

    soft_update(target_critic, critic, tau)
    soft_update(target_actor, actor, tau)
    return critic_loss, actor_loss


# ---------------------------------------------------------------------------
# Replay: a masked ring buffer per level.
#
# HAC's writes are bursty and asynchronous across envs -- up to H+2 rows per
# env per step (one hindsight-action transition, one subgoal-test penalty, and
# at attempt end a burst of up to H goal-relabelled window rows). No fixed-rate
# buffer add expresses that: a one-slot-per-step add with a validity mask
# leaves the top level's buffer ~1% valid, and an outbox of depth one silently
# drops the relabelled burst. The scatter below writes exactly the valid rows.
# ---------------------------------------------------------------------------

class Ring(NamedTuple):
    data: Any        # pytree, leaves (capacity, ...)
    ptr: jax.Array   # () int32
    size: jax.Array  # () int32


def ring_init(row_spec, capacity: int) -> Ring:
    data = jax.tree.map(lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), row_spec)
    return Ring(data, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))


def ring_add(ring: Ring, rows, mask: jax.Array, capacity: int) -> Ring:
    """Compact `rows` down to the entries `mask` selects and append them.
    Invalid rows are aimed at the out-of-bounds index `capacity`, which
    `mode="drop"` discards."""
    counts = mask.astype(jnp.int32)
    n_valid = jnp.sum(counts)
    dest = jnp.where(mask, (ring.ptr + jnp.cumsum(counts) - 1) % capacity, capacity)
    data = jax.tree.map(lambda buf, row: buf.at[dest].set(row, mode="drop"), ring.data, rows)
    return Ring(data, (ring.ptr + n_valid) % capacity,
                jnp.minimum(ring.size + n_valid, capacity))


def ring_sample(ring: Ring, key: jax.Array, batch_size: int):
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(ring.size, 1))
    return jax.tree.map(lambda buf: buf[idx], ring.data)


class Window(NamedTuple):
    """The current attempt of one level, held per env so that hindsight-goal
    relabelling can draw its goal from the achieved states of the WHOLE
    attempt (Levy's finalize_goal_replay). Relabelling on the fly using only
    past achieved states would make every relabelled row a -1, so the critic
    would never see the reward=0/discount=0 case. Reward, discount and goal are recomputed at flush and so are
    not stored."""
    obs: jax.Array       # (num_envs, H, obs_dim)
    action: jax.Array    # (num_envs, H, act_dim)
    next_obs: jax.Array  # (num_envs, H, obs_dim)
    achieved: jax.Array  # (num_envs, H, goal_dim)  clipped projection
    count: jax.Array     # (num_envs,) int32


def where_per_env(done_mask, a, b):
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


class LoopCarry(NamedTuple):
    nnx_states: tuple           # length k; per level (actor, critic, t_actor, t_critic, opt_a, opt_c)
    rings: tuple                # length k; Ring
    windows: tuple              # length k; Window
    env_state: Any
    obs: jax.Array              # (num_envs, obs_dim)
    goal_stack: jax.Array       # (num_envs, k, goal_dim)   [:, k-1] is the end goal
    action_start_obs: jax.Array # (num_envs, k, obs_dim)    obs when level i began its CURRENT ACTION
    attempt_count: jax.Array    # (num_envs, k) int32       ACTIONS taken by level i this attempt
    testing: jax.Array          # (num_envs, k) bool        level i acts deterministically
    needs_new_goal: jax.Array   # (num_envs, k) bool        `end` carried out of the previous step
    ep_step: jax.Array          # (num_envs,) int32
    cum_reward: jax.Array       # (num_envs,) float32       native brax reward, logged not trained
    stats: jax.Array            # (k, 2) float32            cumulative [attempts, successes] per level
    step: jax.Array
    rng: jax.Array


if __name__ == "__main__":

    config = parse_config()
    logger = Logger(config)

    framework_type = config["env"].get("framework", "brax")
    env_id = config["env"]["make"]["id"].split("/")[-1]
    env_kwargs = config["env"].get("kwargs", {}) or {}

    n_steps = config["training"].get("n_steps", 200_000)
    chunk_size = config["training"].get("chunk_size", 200)
    num_envs = config["training"].get("num_envs", 512)
    k = config["training"].get("num_levels", 3)
    H = config["training"].get("H", 10)
    # Per-level action budgets. Levy holds the sub-levels at a constant
    # time_scale and lets only the TOP level's budget absorb the episode
    # horizon.
    horizon = config["training"].get("horizon", None)
    if horizon is not None:
        top_H = max(1, int(round(horizon / (H ** (k - 1)))))
    else:
        top_H = H
    H_levels = [H] * (k - 1) + [top_H]
    batch_size = config["training"].get("batch_size", 1024)
    buffer_size = config["training"].get("buffer_size", 200_000)
    subgoal_test_perc = config["training"].get("subgoal_test_perc", 0.3)
    random_action_perc = config["training"].get("random_action_perc", 0.2)
    noise_perc = config["training"].get("noise_perc", 0.1)
    tau = config["training"].get("tau", 0.05)
    lr_actor = config["training"].get("lr_actor", 1e-3)
    lr_critic = config["training"].get("lr_critic", 1e-3)
    hidden_dim = config.get("network", {}).get("hidden_dim", 256)

    # gamma and the critic's output bound are DERIVED from H, not configured --
    # Levy sets q_limit = -H and gamma = 1 - 1/H so that the reachable return
    # range of the -1-per-step reward exactly fills the critic's output bound.
    gammas = [1.0 - 1.0 / h for h in H_levels]
    q_limits = [float(h) for h in H_levels]

    goal_cfg = config["goal"]
    goal_indices = jnp.asarray(goal_cfg["indices"], dtype=jnp.int32)
    goal_low = jnp.asarray(goal_cfg["low"], dtype=jnp.float32)
    goal_high = jnp.asarray(goal_cfg["high"], dtype=jnp.float32)
    goal_threshold = jnp.asarray(goal_cfg["threshold"], dtype=jnp.float32)
    goal_dim = goal_low.shape[0]

    # Every level trains on the same schedule. Levy calls learn(num_updates)
    # on EVERY layer at each episode end, so the upper levels get as many
    # gradient steps as level 0 and simply replay their smaller buffers more.
    train_every = [config["training"].get("train_every", 1)] * k

    wrapped = make_jax_env(framework_type, env_id, cumulant_dim=goal_dim, **env_kwargs)
    obs_dim = wrapped.state_dim
    action_dim = wrapped.action_dim
    action_low = wrapped.action_low if wrapped.action_low is not None else -jnp.ones((action_dim,))
    action_high = wrapped.action_high if wrapped.action_high is not None else jnp.ones((action_dim,))

    assert action_dim > 0, "HAC is continuous-control only; use a brax environment."
    assert buffer_size >= num_envs * (max(H_levels) + 2), (
        f"buffer_size must be >= num_envs*(max(H)+2) = {num_envs * (max(H_levels) + 2)}, else one "
        f"step's writes alias within themselves and the scatter is nondeterministic."
    )

    # Per-level action space: primitives at level 0, subgoals above it.
    level_low = [action_low] + [goal_low] * (k - 1)
    level_high = [action_high] + [goal_high] * (k - 1)
    level_act_dim = [action_dim] + [goal_dim] * (k - 1)

    # Static input normalisation
    goal_scale = np.maximum(np.abs(np.asarray(goal_low)), np.abs(np.asarray(goal_high)))
    goal_scale = np.where(goal_scale > 0, goal_scale, 1.0)
    obs_scale = np.ones((obs_dim,), dtype=np.float32)
    obs_scale[np.asarray(goal_cfg["indices"])] = goal_scale
    level_act_scale = [np.maximum(np.abs(np.asarray(level_low[i])), np.abs(np.asarray(level_high[i])))
                       for i in range(k)]
    level_act_scale = [np.where(s > 0, s, 1.0) for s in level_act_scale]

    seed = config["seed"]
    key = jax.random.PRNGKey(seed)

    # ---- Per-level networks, optimizers and graphdefs ----
    # Levels have different action dims, so their parameters cannot be stacked
    # and vmapped; a Python list unrolled at trace time.
    graphdefs, level_states = [], []
    for i in range(k):
        key, ka, kc, kta, ktc = jax.random.split(key, 5)
        actor = Actor(obs_dim, goal_dim, level_act_dim[i], level_low[i], level_high[i],
                      obs_scale, goal_scale, hidden_dim=hidden_dim, rngs=nnx.Rngs(ka))
        target_actor = Actor(obs_dim, goal_dim, level_act_dim[i], level_low[i], level_high[i],
                             obs_scale, goal_scale, hidden_dim=hidden_dim, rngs=nnx.Rngs(kta))
        nnx.update(target_actor, jax.tree.map(jnp.copy, nnx.state(actor)))

        critic = Critic(obs_dim, goal_dim, level_act_dim[i], q_limits[i],
                        obs_scale, goal_scale, level_act_scale[i],
                        hidden_dim=hidden_dim, rngs=nnx.Rngs(kc))
        target_critic = Critic(obs_dim, goal_dim, level_act_dim[i], q_limits[i],
                               obs_scale, goal_scale, level_act_scale[i],
                               hidden_dim=hidden_dim, rngs=nnx.Rngs(ktc))
        nnx.update(target_critic, jax.tree.map(jnp.copy, nnx.state(critic)))

        opt_actor = nnx.Optimizer(
            actor, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_actor)), wrt=nnx.Param)
        opt_critic = nnx.Optimizer(
            critic, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_critic)), wrt=nnx.Param)

        gd, st = nnx.split((actor, critic, target_actor, target_critic, opt_actor, opt_critic))
        graphdefs.append(gd)
        level_states.append(st)

    # ---- Rings and windows ----
    rings0, windows0 = [], []
    for i in range(k):
        row_spec = {
            "obs": jnp.zeros((obs_dim,), jnp.float32),
            "goal": jnp.zeros((goal_dim,), jnp.float32),
            "action": jnp.zeros((level_act_dim[i],), jnp.float32),
            "reward": jnp.zeros((), jnp.float32),
            "next_obs": jnp.zeros((obs_dim,), jnp.float32),
            "discount": jnp.zeros((), jnp.float32),
        }
        rings0.append(ring_init(row_spec, buffer_size))
        windows0.append(Window(
            obs=jnp.zeros((num_envs, H_levels[i], obs_dim), jnp.float32),
            action=jnp.zeros((num_envs, H_levels[i], level_act_dim[i]), jnp.float32),
            next_obs=jnp.zeros((num_envs, H_levels[i], obs_dim), jnp.float32),
            achieved=jnp.zeros((num_envs, H_levels[i], goal_dim), jnp.float32),
            count=jnp.zeros((num_envs,), jnp.int32),
        ))

    vmap_reset = jax.jit(jax.vmap(wrapped.reset_fn))
    vmap_step = jax.jit(jax.vmap(wrapped.step_fn, in_axes=(0, 0, 0)))

    env_idx = jnp.arange(num_envs)

    def sample_end_goals(rng_key):
        return jax.random.uniform(rng_key, (num_envs, goal_dim), minval=goal_low, maxval=goal_high)

    # ---- Training Loop Core ----
    def scan_body(carry: LoopCarry, step_key):
        rng, select_key, test_key, reset_key, train_key = jax.random.split(carry.rng, 5)

        obs = carry.obs
        goal_stack = carry.goal_stack
        testing = carry.testing
        needs = carry.needs_new_goal   # `end` from the previous step


        acted = jnp.concatenate(
            [jnp.ones((num_envs, 1), jnp.bool_), needs[:, :k - 1]], axis=1)

        # ---- Top-down goal refresh, i = k-2 .. 0, strictly descending ----
        # goal_stack[i] comes from actor_{i+1}(obs, goal_stack[i+1]), and
        # goal_stack[i+1] may itself have just been refreshed this step, so the
        # order matters and this loop cannot be vectorised over levels.
        select_keys = jax.random.split(select_key, k)
        test_keys = jax.random.split(test_key, k)
        actors = [nnx.merge(graphdefs[i], carry.nnx_states[i])[0] for i in range(k)]

        for i in range(k - 2, -1, -1):
            parent = i + 1
            proposed = select_action(
                actors[parent], obs, goal_stack[:, parent], goal_low, goal_high,
                testing[:, parent], select_keys[parent], noise_perc, random_action_perc)
            refresh = needs[:, i]
            goal_stack = goal_stack.at[:, i].set(
                jnp.where(refresh[:, None], proposed, goal_stack[:, i]))
            # A level is under test if its parent is, or on a fresh coin flip.
            new_testing = jnp.logical_or(
                testing[:, parent],
                jax.random.uniform(test_keys[i], (num_envs,)) < subgoal_test_perc)
            testing = testing.at[:, i].set(jnp.where(refresh, new_testing, testing[:, i]))

        # Counters reset when a level is given a fresh goal, then increment for
        # the level's own new action. Comparison against H is post-increment;
        # comparing pre-increment would grant H+1 actions.
        attempt_count = jnp.where(needs, 0, carry.attempt_count) + acted.astype(jnp.int32)
        # This is the obs at the start of the level's current ACTION, which is
        # what its transition is anchored on -- not the start of its attempt.
        action_start_obs = jnp.where(acted[:, :, None], obs[:, None, :], carry.action_start_obs)

        # ---- Level 0 acts, environment steps ----
        action = select_action(
            actors[0], obs, goal_stack[:, 0], action_low, action_high,
            testing[:, 0], select_keys[0], noise_perc, random_action_perc)

        step_keys = jax.random.split(step_key, num_envs)
        obs2, env_state2, extrinsic_r, env_done, info = vmap_step(step_keys, carry.env_state, action)

        raw_proj = project(obs2, goal_indices)                              # (N, goal_dim)
        achieved_row = clip_to_goal_space(raw_proj, goal_low, goal_high)
        # Achievement is tested on the UNCLIPPED projection, at every level, on
        # every primitive step -- Levy's check_goals unwinds the whole stack the
        # instant any level's goal is hit, not just at action boundaries.
        achieved = goal_reached(raw_proj[:, None, :], goal_stack, goal_threshold)   # (N, k)

        # ---- Close-out, two passes ----
        # Pass 1 (ascending) uses local_end[i-1], not end[i-1]: end[i-1] depends
        # on end[i] depends on level i's action completing, which is end[i-1] --
        # circular. The ancestor-unwind case is picked up by the suffix-OR in
        # pass 2 instead.
        local_end = []
        prev_local = jnp.ones((num_envs,), jnp.bool_)
        for i in range(k):
            out_of_actions = jnp.logical_and(prev_local, attempt_count[:, i] >= H_levels[i])
            le = achieved[:, i] | env_done | out_of_actions
            local_end.append(le)
            prev_local = le
        local_end = jnp.stack(local_end, axis=1)

        end = [None] * k
        suffix = jnp.zeros((num_envs,), jnp.bool_)
        for i in range(k - 1, -1, -1):
            suffix = jnp.logical_or(local_end[:, i], suffix)
            end[i] = suffix
        end = jnp.stack(end, axis=1)                                        # (N, k)
        episode_done = end[:, k - 1]   # the only episode boundary; <= H**k steps

        # `completes[i]` -- level i's current action finished this step -- is
        # end[i-1], with end[-1] := True since a level-0 action is one step.
        completes = jnp.concatenate(
            [jnp.ones((num_envs, 1), jnp.bool_), end[:, :k - 1]], axis=1)

        level_rewards = jnp.where(achieved, 0.0, -1.0)
        level_discounts = jnp.where(achieved, 0.0, 1.0)

        # ---- Per-level buffer writes ----
        new_rings, new_windows = [], []
        for i in range(k):
            window = carry.windows[i]
            # For i > 0 the stored action is the subgoal ACTUALLY reached, not
            # the one proposed -- the hindsight action transition. Level 0's
            # action is a primitive, so there is nothing to relabel.
            stored_action = action if i == 0 else achieved_row

            # Push the completed action into the window BEFORE flushing, or
            # every attempt's last and most informative action is dropped from
            # goal replay.
            push = completes[:, i]
            slot = jnp.clip(window.count, 0, H_levels[i] - 1)

            def push_leaf(buf, new):
                current = buf[env_idx, slot]
                return buf.at[env_idx, slot].set(
                    jnp.where(push.reshape((-1,) + (1,) * (new.ndim - 1)), new, current))

            window = Window(
                obs=push_leaf(window.obs, action_start_obs[:, i]),
                action=push_leaf(window.action, stored_action),
                next_obs=push_leaf(window.next_obs, obs2),
                achieved=push_leaf(window.achieved, achieved_row),
                count=window.count + push.astype(jnp.int32),
            )

            rows, masks = [], []

            # 1. Hindsight action transition.
            rows.append({
                "obs": action_start_obs[:, i],
                "goal": goal_stack[:, i],
                "action": stored_action,
                "reward": level_rewards[:, i],
                "next_obs": obs2,
                "discount": level_discounts[:, i],
            })
            masks.append(completes[:, i])

            # 2. Subgoal testing transition. Levy penalises only when the child
            # exhausted its attempts without reaching the subgoal (`maxed_out`).
            # Gating merely on "not achieved" would also punish subgoals cut
            # short by an ancestor unwinding, injecting spurious -H targets.
            if i > 0:
                child_maxed = jnp.logical_and(
                    attempt_count[:, i - 1] >= H_levels[i - 1],
                    jnp.logical_not(achieved[:, i - 1]))
                rows.append({
                    "obs": action_start_obs[:, i],
                    "goal": goal_stack[:, i],
                    "action": goal_stack[:, i - 1],   # the PROPOSED subgoal
                    "reward": jnp.full((num_envs,), -float(H_levels[i]), jnp.float32),
                    "next_obs": obs2,
                    "discount": jnp.zeros((num_envs,), jnp.float32),
                })
                masks.append(completes[:, i] & testing[:, i - 1] & child_maxed)

            # 3. Hindsight goal transitions: the whole attempt relabelled with a
            # goal drawn uniformly from the achieved states of that attempt.
            # Levy's finalize_goal_replay forces its last relabel index to
            # num_trans-1, so with the default single replay goal the goal is
            # ALWAYS the attempt's final achieved state. Sampling the index
            # uniformly instead (an earlier version of this file) leaves every
            # transition after the drawn index at -1 with no terminal
            # transition in the window at all -- and since this scheme yields
            # only ~1 reward-0 transition per window of H, losing it guts the
            # relabelling's entire learning signal.
            relabel_idx = jnp.maximum(window.count - 1, 0)
            relabel_goal = window.achieved[env_idx, relabel_idx]             # (N, goal_dim)
            relabel_reached = goal_reached(
                window.achieved, relabel_goal[:, None, :], goal_threshold)   # (N, H)
            slot_valid = jnp.arange(H_levels[i])[None, :] < window.count[:, None]
            rows.append({
                "obs": window.obs,
                "goal": jnp.broadcast_to(relabel_goal[:, None, :], (num_envs, H_levels[i], goal_dim)),
                "action": window.action,
                "reward": jnp.where(relabel_reached, 0.0, -1.0),
                "next_obs": window.next_obs,
                "discount": jnp.where(relabel_reached, 0.0, 1.0),
            })
            masks.append(jnp.logical_and(end[:, i][:, None], slot_valid))

            # Flatten each row group to (M, ...) and concatenate into one
            # scatter of M = num_envs * (H + 2) candidate rows.
            flat = {name: [] for name in rows[0]}
            flat_mask = []
            for row, mask in zip(rows, masks):
                lead = mask.size
                for name, value in row.items():
                    spec = rings0[i].data[name].shape[1:]
                    flat[name].append(value.reshape((lead,) + spec))
                flat_mask.append(mask.reshape((lead,)))
            flat = {name: jnp.concatenate(parts, axis=0) for name, parts in flat.items()}
            flat_mask = jnp.concatenate(flat_mask, axis=0)

            new_rings.append(ring_add(carry.rings[i], flat, flat_mask, buffer_size))
            new_windows.append(window._replace(
                count=jnp.where(end[:, i], 0, window.count)))

        # ---- Per-level DDPG updates ----
        train_keys = jax.random.split(train_key, k)
        step_next = carry.step + 1
        new_states, critic_losses, actor_losses, trained_flags = [], [], [], []
        for i in range(k):
            should_train = jnp.logical_and(
                step_next % train_every[i] == 0,
                new_rings[i].size >= batch_size,
            )

            def do_train(state, _i=i, _ring=new_rings[i], _key=train_keys[i]):
                a, c, ta, tc, oa, oc = nnx.merge(graphdefs[_i], state)
                batch = ring_sample(_ring, _key, batch_size)
                c_loss, a_loss = train_level_step(a, c, ta, tc, oa, oc, batch,
                                                  gammas[_i], q_limits[_i], tau)
                _, next_state = nnx.split((a, c, ta, tc, oa, oc))
                return next_state, c_loss, a_loss

            def skip_train(state):
                return state, jnp.array(0.0), jnp.array(0.0)

            state_next, c_loss, a_loss = jax.lax.cond(
                should_train, do_train, skip_train, carry.nnx_states[i])
            new_states.append(state_next)
            critic_losses.append(c_loss)
            actor_losses.append(a_loss)
            trained_flags.append(should_train)

        # ---- Episode reset (after every emission, using the pre-reset obs) ----
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_state = vmap_reset(reset_keys)
        rng, goal_key = jax.random.split(rng)
        fresh_end_goals = sample_end_goals(goal_key)

        obs_next = where_per_env(episode_done, reset_obs, obs2)
        env_state_next = jax.tree.map(
            lambda a, b: where_per_env(episode_done, a, b), reset_state, env_state2)
        goal_stack_next = goal_stack.at[:, k - 1].set(
            jnp.where(episode_done[:, None], fresh_end_goals, goal_stack[:, k - 1]))
        action_start_obs_next = jnp.where(
            episode_done[:, None, None], reset_obs[:, None, :], action_start_obs)
        attempt_count_next = jnp.where(episode_done[:, None], 0, attempt_count)
        testing_next = jnp.where(episode_done[:, None], False, testing)
        new_windows = [w._replace(count=jnp.where(episode_done, 0, w.count)) for w in new_windows]

        cum_reward_new = carry.cum_reward + extrinsic_r
        completed_return = jnp.where(episode_done, cum_reward_new, 0.0)
        ep_len_completed = jnp.where(episode_done, carry.ep_step + 1, 0)

        attempts_delta = jnp.sum(end.astype(jnp.float32), axis=0)
        success_delta = jnp.sum(jnp.logical_and(end, achieved).astype(jnp.float32), axis=0)
        stats_next = carry.stats + jnp.stack([attempts_delta, success_delta], axis=-1)

        new_carry = carry._replace(
            nnx_states=tuple(new_states),
            rings=tuple(new_rings),
            windows=tuple(new_windows),
            env_state=env_state_next,
            obs=obs_next,
            goal_stack=goal_stack_next,
            action_start_obs=action_start_obs_next,
            attempt_count=attempt_count_next,
            testing=testing_next,
            needs_new_goal=end,
            ep_step=jnp.where(episode_done, 0, carry.ep_step + 1),
            cum_reward=jnp.where(episode_done, 0.0, cum_reward_new),
            stats=stats_next,
            step=step_next,
            rng=rng,
        )

        metrics = {
            "critic_losses": jnp.stack(critic_losses),
            "actor_losses": jnp.stack(actor_losses),
            "trained": jnp.stack(trained_flags),
            "episode_done": episode_done,
            "completed_return": completed_return,
            "ep_len_completed": ep_len_completed,
            "end_goal_reached": jnp.logical_and(episode_done, achieved[:, k - 1]),
            "subgoal_tests": jnp.sum(jnp.logical_and(completes[:, 1:], testing[:, :k - 1]).astype(jnp.float32)),
            "ring_sizes": jnp.stack([r.size for r in new_rings]),
        }
        return new_carry, metrics

    # XLA Buffer Donation on pure state
    @partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, keys):
        return jax.lax.scan(scan_body, carry, keys)

    key, reset_key0, goal_key0 = jax.random.split(key, 3)
    obs0, state0 = vmap_reset(jax.random.split(reset_key0, num_envs))

    goal_stack0 = jnp.zeros((num_envs, k, goal_dim), jnp.float32)
    goal_stack0 = goal_stack0.at[:, k - 1].set(sample_end_goals(goal_key0))

    carry = LoopCarry(
        nnx_states=tuple(level_states),
        rings=tuple(rings0),
        windows=tuple(windows0),
        env_state=state0,
        obs=obs0,
        goal_stack=goal_stack0,
        action_start_obs=jnp.broadcast_to(obs0[:, None, :], (num_envs, k, obs_dim)),
        attempt_count=jnp.zeros((num_envs, k), jnp.int32),
        testing=jnp.zeros((num_envs, k), jnp.bool_),
        # Everyone selects on the very first step.
        needs_new_goal=jnp.ones((num_envs, k), jnp.bool_),
        ep_step=jnp.zeros((num_envs,), jnp.int32),
        cum_reward=jnp.zeros((num_envs,), jnp.float32),
        stats=jnp.zeros((k, 2), jnp.float32),
        step=jnp.array(0, jnp.int32),
        rng=key,
    )

    def run_and_log(carry, key, n, step0):
        prev_stats = jax.device_get(carry.stats)

        keys = jax.random.split(key, n)
        carry, metrics = run_chunk(carry, keys)
        metrics = jax.device_get(metrics)

        chunk_stats = jax.device_get(carry.stats) - prev_stats
        attempts, successes = chunk_stats[:, 0], chunk_stats[:, 1]

        env_steps0 = step0 * num_envs
        total_episodes = np.sum(metrics["episode_done"])

        chunk_metrics = {}
        for i in range(k):
            trained_i = metrics["trained"][:, i]
            n_trained = max(np.sum(trained_i), 1)
            chunk_metrics[f"train/level_{i}/critic_loss"] = float(
                np.sum(metrics["critic_losses"][:, i]) / n_trained)
            chunk_metrics[f"train/level_{i}/actor_loss"] = float(
                np.sum(metrics["actor_losses"][:, i]) / n_trained)
            chunk_metrics[f"train/level_{i}/updates"] = int(np.sum(trained_i))
            chunk_metrics[f"train/level_{i}/buffer_size"] = int(metrics["ring_sizes"][-1, i])
            chunk_metrics[f"levels/attempts_level_{i}"] = float(attempts[i])
            chunk_metrics[f"levels/success_rate_level_{i}"] = float(
                successes[i] / max(attempts[i], 1))

        mean_return = float(np.sum(metrics["completed_return"]) / max(total_episodes, 1))
        mean_ep_len = float(np.sum(metrics["ep_len_completed"]) / max(total_episodes, 1))
        end_goal_rate = float(np.sum(metrics["end_goal_reached"]) / max(total_episodes, 1))

        chunk_metrics.update({
            "train/native_return_mean": mean_return,     # brax's own reward, logged not trained
            "train/episode_len_mean": mean_ep_len,
            "train/episodes": int(total_episodes),
            "train/end_goal_success_rate": end_goal_rate,
            "train/subgoal_tests_per_step": float(np.mean(metrics["subgoal_tests"])),
        })
        logger.log_metrics(chunk_metrics, step=env_steps0)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} "
              f"env-steps) | End-goal success: {end_goal_rate:.3f} | Episodes: {int(total_episodes)} "
              f"| Critic losses: "
              + ", ".join(f"L{i}={chunk_metrics[f'train/level_{i}/critic_loss']:.4f}" for i in range(k)))
        return carry

    # Main Execution and Evaluation Loop
    from jaxhrl.common.utils import StepScheduler
    from jaxhrl.common.wrappers import run_offpolicy_eval_stage
    eval_config = config.get("eval", {})
    checkpoint_config = config.get("checkpoint", {})
    eval_sched = StepScheduler(n_steps, eval_config.get("interval_pct", 0.05), eval_config.get("enabled", False))
    ckpt_sched = StepScheduler(n_steps, checkpoint_config.get("interval_pct", 0.2), checkpoint_config.get("enabled", True))

    @jax.jit
    def eval_step(nnx_states, obs, goal_stack, attempt_count, needs, end_goal):
        """One greedy hierarchical step for a single unbatched env. Mirrors
        scan_body's scheduling with a batch axis of 1 and no exploration."""
        obs_b = obs[None, :]
        goal_stack = goal_stack.at[k - 1].set(end_goal)
        acted = jnp.concatenate([jnp.ones((1,), jnp.bool_), needs[:k - 1]])

        for i in range(k - 2, -1, -1):
            parent = i + 1
            actor = nnx.merge(graphdefs[parent], nnx_states[parent])[0]
            proposed = actor(obs_b, goal_stack[parent][None, :])[0]
            goal_stack = goal_stack.at[i].set(
                jnp.where(needs[i], proposed, goal_stack[i]))

        attempt_count = jnp.where(needs, 0, attempt_count) + acted.astype(jnp.int32)
        actor0 = nnx.merge(graphdefs[0], nnx_states[0])[0]
        action = actor0(obs_b, goal_stack[0][None, :])[0]
        return action, goal_stack, attempt_count

    def make_eval_policy_fn(nnx_states):
        """Factory per episode: the hierarchy's goal stack and attempt counters
        are per-episode state and must reset between the eval runs, so they live
        in a closure rather than in the jitted step (which stays outside so it
        is compiled once, not retraced per episode)."""
        state = {
            "goal_stack": jnp.zeros((k, goal_dim), jnp.float32),
            "attempt_count": jnp.zeros((k,), jnp.int32),
            "needs": jnp.ones((k,), jnp.bool_),
            "end_goal": None,
        }

        def policy_fn(_params, single_obs, eval_key):
            if state["end_goal"] is None:
                state["end_goal"] = jax.random.uniform(
                    eval_key, (goal_dim,), minval=goal_low, maxval=goal_high)

            action, goal_stack, attempt_count = eval_step(
                nnx_states, single_obs, state["goal_stack"],
                state["attempt_count"], state["needs"], state["end_goal"])

            achieved = goal_reached(
                project(single_obs, goal_indices)[None, :], goal_stack, goal_threshold)
            local_end, prev_local = [], jnp.array(True)
            for i in range(k):
                le = achieved[i] | jnp.logical_and(prev_local, attempt_count[i] >= H_levels[i])
                local_end.append(le)
                prev_local = le
            end, suffix = [None] * k, jnp.array(False)
            for i in range(k - 1, -1, -1):
                suffix = jnp.logical_or(local_end[i], suffix)
                end[i] = suffix

            state["goal_stack"] = goal_stack
            state["attempt_count"] = attempt_count
            state["needs"] = jnp.stack(end)
            return action, goal_stack[0]

        return policy_fn

    for step_idx in range(0, n_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)

        current_env_step = (step_idx + chunk_size) * num_envs

        if ckpt_sched.due(step_idx):
            logger.save_checkpoint(jax.device_get(carry.nnx_states), current_env_step)

        if eval_sched.due(step_idx):
            print(f"\n--- Running Evaluation at Step {step_idx} ---")
            eval_states = carry.nnx_states
            key = run_offpolicy_eval_stage(
                logger, wrapped, lambda: make_eval_policy_fn(eval_states),
                eval_states, key, current_env_step, eval_config)
            print(f"--- Evaluation Complete ---\n")

    logger.close()
