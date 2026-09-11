# HierQ in JAX
# From "Learning Multi-Level Hierarchies with Hindsight", Algorithm 2 (Appendix)
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



class MultiGoalQNetwork(nnx.Module):
    """Q(s, g, a) for every goal g at once: (batch, num_goals, num_actions).

    The head is linear, not sigmoid-bounded: a pessimistic bias inside a
    sigmoid would saturate it and starve the subgoal outputs of gradient.
    Targets are clipped to [-q_limit, 0] in the loss instead.
    """

    def __init__(self, obs_dim: int, num_goals: int, num_actions: int, init_value: float,
                 hidden_dim: int = 512, *, rngs: nnx.Rngs):
        self.num_goals = int(num_goals)
        self.num_actions = int(num_actions)
        self.linear1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        # Zero weights + constant bias: every Q starts at `init_value`, and an
        # output that is never trained stays there exactly.
        self.head = nnx.Linear(hidden_dim, self.num_goals * self.num_actions,
                               kernel_init=nnx.initializers.zeros_init(),
                               bias_init=nnx.initializers.constant(init_value),
                               rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32)
        x = nnx.relu(self.ln1(self.linear1(x)))
        x = nnx.relu(self.ln2(self.linear2(x)))
        q = self.head(x)
        return q.reshape(q.shape[:-1] + (self.num_goals, self.num_actions))


def eps_greedy(q_rows, key, epsilon, explore_mask, deterministic):
    """epsilon-greedy over Q(s, g, .) for one goal per env.
    """
    n = q_rows.shape[0]
    tie_key, rand_key, coin_key = jax.random.split(key, 3)
    greedy = jnp.argmax(q_rows + jax.random.uniform(tie_key, q_rows.shape) * 1e-6, axis=-1)
    logits = jnp.where(explore_mask, 0.0, -1e9)
    random = jax.random.categorical(rand_key, logits, shape=(n,))
    explore = (jax.random.uniform(coin_key, (n,)) < epsilon) & ~deterministic
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def level_loss(model, target_model, batch, achieved_fn, gamma, q_limit, subgoal_level):
    """All-goals TD loss for one level.
    """
    obs = batch["obs"].astype(jnp.float32)
    nxt = batch["next_obs"].astype(jnp.float32)
    ach_next = achieved_fn(nxt)                                   # (B, G) bool
    terminal = ach_next | batch["done"][:, None]
    bootstrap = jnp.max(target_model(nxt), axis=-1)               # (B, G)
    target = (jnp.where(ach_next, 0.0, -1.0)
              + jnp.where(terminal, 0.0, gamma) * bootstrap)      # (B, G)
    q = model(obs)                                                # (B, G, A)

    if subgoal_level:
        pen = batch["penalty"]
        target = jnp.where(pen[:, None], -q_limit, target)
        target = jnp.clip(jax.lax.stop_gradient(target), -q_limit, 0.0)
        onehot = jax.nn.one_hot(batch["action"], q.shape[-1])
        act_mask = jnp.where(pen[:, None], onehot, ach_next.astype(jnp.float32))  # (B, A)
        err = (q - target[:, :, None]) ** 2 * act_mask[:, None, :]
        return jnp.sum(err) / jnp.maximum(jnp.sum(act_mask) * q.shape[1], 1.0)

    target = jnp.clip(jax.lax.stop_gradient(target), -q_limit, 0.0)
    idx = jnp.broadcast_to(batch["action"][:, None, None], (q.shape[0], q.shape[1], 1))
    q_sel = jnp.take_along_axis(q, idx, axis=-1)[..., 0]          # (B, G)
    return jnp.mean((q_sel - target) ** 2)


def soft_update(target, online, tau: float):
    nnx.update(target, jax.tree.map(
        lambda t, o: tau * o + (1.0 - tau) * t,
        nnx.state(target, nnx.Param), nnx.state(online, nnx.Param),
    ))


def train_level_step(model, target_model, opt, batch, achieved_fn,
                     gamma: float, q_limit: float, tau: float, subgoal_level: bool):
    def loss_fn(m, t, b):
        return level_loss(m, t, b, achieved_fn, gamma, q_limit, subgoal_level)

    loss, grads = nnx.value_and_grad(loss_fn)(model, target_model, batch)
    opt.update(model, grads)
    soft_update(target_model, model, tau)
    return loss


# ---------------------------------------------------------------------------
# Replay: a masked ring buffer per level
# ---------------------------------------------------------------------------

class Ring(NamedTuple):
    data: Any
    ptr: jax.Array
    size: jax.Array


def ring_init(row_spec, capacity: int) -> Ring:
    data = jax.tree.map(lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), row_spec)
    return Ring(data, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))


def ring_add(ring: Ring, rows, mask: jax.Array, capacity: int) -> Ring:
    counts = mask.astype(jnp.int32)
    n_valid = jnp.sum(counts)
    dest = jnp.where(mask, (ring.ptr + jnp.cumsum(counts) - 1) % capacity, capacity)
    data = jax.tree.map(lambda buf, row: buf.at[dest].set(row, mode="drop"), ring.data, rows)
    return Ring(data, (ring.ptr + n_valid) % capacity,
                jnp.minimum(ring.size + n_valid, capacity))


def ring_sample(ring: Ring, key: jax.Array, batch_size: int):
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(ring.size, 1))
    return jax.tree.map(lambda buf: buf[idx], ring.data)


class ObsWindow(NamedTuple):
    """PrevStates_i: the last prod(H_levels[:i]) observations since the world
    last reset -- exactly the states from which the current one was reachable
    within one level-i action."""
    obs: jax.Array     # (num_envs, capacity, obs_dim) uint8
    count: jax.Array   # (num_envs,) int32


def where_per_env(done_mask, a, b):
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


class LoopCarry(NamedTuple):
    nnx_states: tuple           # length k; per level (model, target, optimizer)
    rings: tuple                # length k
    windows: tuple              # length k-1; ObsWindow for levels 1..k-1
    env_state: Any
    obs: jax.Array              # (num_envs, obs_dim) float32
    goal_stack: jax.Array       # (num_envs, k) int32 goal indices; [:, k-1] is the commanded goal
    action_start_obs: jax.Array # (num_envs, k, obs_dim) uint8 -- obs when level i began its current action
    attempt_count: jax.Array    # (num_envs, k) int32
    testing: jax.Array          # (num_envs, k) bool -- level i acts deterministically
    needs_new_goal: jax.Array   # (num_envs, k) bool -- `end` from the previous step
    seen: jax.Array             # (num_goals,) bool -- goals ever achieved
    ep_step: jax.Array          # (num_envs,) int32 -- steps into the current goal attempt
    stats: jax.Array            # (k, 2) cumulative [attempts, successes]
    step: jax.Array
    rng: jax.Array


if __name__ == "__main__":

    config = parse_config()
    logger = Logger(config)

    framework_type = config["env"].get("framework", "gc_craftax")
    env_id = config["env"]["make"]["id"].split("/")[-1]
    env_kwargs = config["env"].get("kwargs", {}) or {}

    tc = config["training"]
    n_steps = tc.get("n_steps", 30_000)
    chunk_size = tc.get("chunk_size", 300)
    num_envs = tc.get("num_envs", 256)
    k = tc.get("num_levels", 2)
    H = tc.get("H", 10)
    batch_size = tc.get("batch_size", 256)
    buffer_size = tc.get("buffer_size", 200_000)
    lr = tc.get("lr", 3e-4)
    tau = tc.get("tau", 0.01)
    subgoal_test_perc = tc.get("subgoal_test_perc", 0.3)
    hidden_dim = tc.get("hidden_dim", 512)

    # Exploration: linear decay, as LEO's optax.linear_schedule.
    epsilon_start = tc.get("epsilon_start", 0.2)
    epsilon_finish = tc.get("epsilon_finish", 0.01)
    epsilon_decay_steps = max(1, int(tc.get("epsilon_decay", 0.2) * n_steps))

    # LEO's greedy test protocol.
    test_interval_chunks = tc.get("test_interval_chunks", 5)
    test_num_repeats = tc.get("test_num_repeats", 16)
    test_num_steps = tc.get("test_num_steps", 512)

    wrapped = make_jax_env(framework_type, env_id, cumulant_dim=1, **env_kwargs)
    assert wrapped.goal_fn is not None and wrapped.num_goals > 0, (
        "HierQ needs a finite goal set: the env must provide goal_fn(obs) -> (num_goals,) bool")
    obs_dim = wrapped.state_dim
    n_actions = wrapped.num_actions
    G = wrapped.num_goals
    goal_names = getattr(wrapped.env, "goal_names", None) or [f"goal_{g}" for g in range(G)]
    goal_fn_batch = jax.vmap(wrapped.goal_fn)

    # Per-level action budgets: sub-levels hold H, the top level absorbs the rest
    # of `horizon` 
    horizon_cfg = tc.get("horizon", None)
    top_H = max(1, int(round(horizon_cfg / (H ** (k - 1))))) if horizon_cfg else H
    H_levels = [H] * (k - 1) + [top_H]
    horizon = int(np.prod(H_levels))
    gammas = [1.0 - 1.0 / h for h in H_levels]
    q_limits = [float(h) for h in H_levels]
    level_actions = [n_actions] + [G] * (k - 1)     # subgoals are goal indices
    window_caps = [int(np.prod(H_levels[:i])) for i in range(1, k)]
    assert buffer_size >= 2 * num_envs, "buffer_size must hold one step's writes (2 x num_envs)"

    seed = config["seed"]
    key = jax.random.PRNGKey(seed)

    # Level 0 starts optimistic at 0, as Algorithm 2 prescribes pessimism only
    # for i > 0; subgoal levels start at their floor -H_i.
    graphdefs, level_states = [], []
    for i in range(k):
        key, km, kt = jax.random.split(key, 3)
        init_value = 0.0 if i == 0 else -q_limits[i]
        model = MultiGoalQNetwork(obs_dim, G, level_actions[i], init_value, hidden_dim, rngs=nnx.Rngs(km))
        target = MultiGoalQNetwork(obs_dim, G, level_actions[i], init_value, hidden_dim, rngs=nnx.Rngs(kt))
        nnx.update(target, jax.tree.map(jnp.copy, nnx.state(model)))
        opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr)),
                            wrt=nnx.Param)
        gd, st = nnx.split((model, target, opt))
        graphdefs.append(gd)
        level_states.append(st)

    def row_spec(i):
        spec = {"obs": jnp.zeros((obs_dim,), jnp.uint8),
                "action": jnp.zeros((), jnp.int32),
                "next_obs": jnp.zeros((obs_dim,), jnp.uint8),
                "done": jnp.zeros((), jnp.bool_)}
        if i > 0:
            spec["penalty"] = jnp.zeros((), jnp.bool_)
        return spec

    rings0 = [ring_init(row_spec(i), buffer_size) for i in range(k)]
    windows0 = [ObsWindow(obs=jnp.zeros((num_envs, c, obs_dim), jnp.uint8),
                          count=jnp.zeros((num_envs,), jnp.int32)) for c in window_caps]

    vmap_reset = jax.jit(jax.vmap(wrapped.reset_fn))
    vmap_step = jax.jit(jax.vmap(wrapped.step_fn, in_axes=(0, 0, 0)))
    env_idx = jnp.arange(num_envs)

    def epsilon_at(step):
        frac = jnp.clip(step.astype(jnp.float32) / epsilon_decay_steps, 0.0, 1.0)
        return epsilon_start + (epsilon_finish - epsilon_start) * frac

    def sample_task_goals(skey, seen, n):
        """LEO's autocurriculum: uniform over goals seen at least once."""
        return jax.random.categorical(skey, jnp.where(seen, 0.0, -1e9), shape=(n,)).astype(jnp.int32)

    def close_out(achieved, env_done, attempt_count, n):
        local_end = []
        prev_local = jnp.ones((n,), jnp.bool_)
        for i in range(k):
            out_of_actions = jnp.logical_and(prev_local, attempt_count[:, i] >= H_levels[i])
            le = achieved[:, i] | env_done | out_of_actions
            local_end.append(le)
            prev_local = le
        end = [None] * k
        suffix = jnp.zeros((n,), jnp.bool_)
        for i in range(k - 1, -1, -1):
            suffix = jnp.logical_or(local_end[i], suffix)
            end[i] = suffix
        return jnp.stack(end, axis=1)

    # ---- Training Loop Core ----
    def scan_body(carry: LoopCarry, step_key):
        (rng, select_key, test_key, goal_key,
         win_key, train_key) = jax.random.split(carry.rng, 6)
        eps_now = epsilon_at(carry.step)

        obs = carry.obs
        goal_stack = carry.goal_stack
        testing = carry.testing
        needs = carry.needs_new_goal
        models = [nnx.merge(graphdefs[i], carry.nnx_states[i])[0] for i in range(k)]

        # Level i takes a new action iff the previous step ended level i-1's
        # attempt; level 0 acts every primitive step. Level i gets a fresh goal
        # iff the previous step ended its own attempt.
        acted = jnp.concatenate(
            [jnp.ones((num_envs, 1), jnp.bool_), needs[:, :k - 1]], axis=1)

        # ---- Top-down subgoal selection, i = k-2 .. 0 (strictly descending) ----
        select_keys = jax.random.split(select_key, k)
        test_keys = jax.random.split(test_key, k)
        for i in range(k - 2, -1, -1):
            parent = i + 1
            rows = models[parent](obs)[env_idx, goal_stack[:, parent]]        # (N, G)
            proposed = eps_greedy(rows, select_keys[parent], eps_now, carry.seen,
                                  testing[:, parent])
            refresh = needs[:, i]
            goal_stack = goal_stack.at[:, i].set(jnp.where(refresh, proposed, goal_stack[:, i]))
            new_testing = jnp.logical_or(
                testing[:, parent],
                jax.random.uniform(test_keys[i], (num_envs,)) < subgoal_test_perc)
            testing = testing.at[:, i].set(jnp.where(refresh, new_testing, testing[:, i]))

        attempt_count = jnp.where(needs, 0, carry.attempt_count) + acted.astype(jnp.int32)
        obs_u8 = obs.astype(jnp.uint8)
        action_start_obs = jnp.where(acted[:, :, None], obs_u8[:, None, :], carry.action_start_obs)

        # ---- Level 0 acts, environment steps (auto-resetting on termination) ----
        rows0 = models[0](obs)[env_idx, goal_stack[:, 0]]                     # (N, A)
        action = eps_greedy(rows0, select_keys[0], eps_now,
                            jnp.ones((n_actions,), jnp.bool_), testing[:, 0])
        step_keys = jax.random.split(step_key, num_envs)
        obs2, env_state2, _, env_done, _ = vmap_step(step_keys, carry.env_state, action)
        obs2_u8 = obs2.astype(jnp.uint8)

        ach_now = goal_fn_batch(obs2)                                          # (N, G)
        achieved = jnp.take_along_axis(ach_now, goal_stack, axis=1)            # (N, k)
        end = close_out(achieved, env_done, attempt_count, num_envs)
        # The commanded goal's attempt is over: reached, out of actions, or the
        # world terminated. A new goal is commanded from the current state.
        goal_done = end[:, k - 1]
        completes = jnp.concatenate(
            [jnp.ones((num_envs, 1), jnp.bool_), end[:, :k - 1]], axis=1)

        # ---- Replay writes ----
        new_rings = [ring_add(carry.rings[0],
                              {"obs": obs_u8, "action": action, "next_obs": obs2_u8,
                               "done": env_done},
                              jnp.ones((num_envs,), jnp.bool_), buffer_size)]
        new_windows = []
        win_keys = jax.random.split(win_key, k)
        for i in range(1, k):
            cap = window_caps[i - 1]
            w = carry.windows[i - 1]
            w = ObsWindow(obs=w.obs.at[env_idx, w.count % cap].set(obs2_u8), count=w.count + 1)
            fill = jnp.minimum(w.count, cap)
            # One state per step from PrevStates_i, uniformly: an unbiased
            # stochastic form of Algorithm 2's "for each s in PrevStates_i".
            slot = jax.random.randint(win_keys[i], (num_envs,), 0, jnp.maximum(fill, 1))
            prev_obs = w.obs[env_idx, slot]
            child_maxed = (attempt_count[:, i - 1] >= H_levels[i - 1]) & ~achieved[:, i - 1]
            pen_mask = completes[:, i] & testing[:, i - 1] & child_maxed
            rows = {
                "obs": jnp.concatenate([prev_obs, action_start_obs[:, i]]),
                "action": jnp.concatenate([jnp.full((num_envs,), -1, jnp.int32), goal_stack[:, i - 1]]),
                "next_obs": jnp.concatenate([obs2_u8, obs2_u8]),
                "done": jnp.concatenate([env_done, env_done]),
                "penalty": jnp.concatenate([jnp.zeros((num_envs,), jnp.bool_),
                                            jnp.ones((num_envs,), jnp.bool_)]),
            }
            mask = jnp.concatenate([fill > 0, pen_mask])
            new_rings.append(ring_add(carry.rings[i], rows, mask, buffer_size))
            # A state from before the world reset is not a predecessor of the
            # current one, so the window is only cleared on true termination.
            new_windows.append(w._replace(count=jnp.where(env_done, 0, w.count)))

        # ---- Per-level updates ----
        train_keys = jax.random.split(train_key, k)
        new_states, losses, trained = [], [], []
        for i in range(k):
            should_train = new_rings[i].size >= batch_size

            def do_train(state, _i=i, _ring=new_rings[i], _key=train_keys[i]):
                m, t, o = nnx.merge(graphdefs[_i], state)
                batch = ring_sample(_ring, _key, batch_size)
                loss = train_level_step(m, t, o, batch, goal_fn_batch, gammas[_i],
                                        q_limits[_i], tau, _i > 0)
                _, next_state = nnx.split((m, t, o))
                return next_state, loss

            def skip_train(state):
                return state, jnp.array(0.0, jnp.float32)

            st, loss = jax.lax.cond(should_train, do_train, skip_train, carry.nnx_states[i])
            new_states.append(st)
            losses.append(loss)
            trained.append(should_train)

        # ---- Goal chaining: command a new goal wherever the attempt ended ----
        seen_next = carry.seen | ach_now.any(axis=0)
        new_goals = sample_task_goals(goal_key, seen_next, num_envs)
        goal_stack_next = goal_stack.at[:, k - 1].set(
            jnp.where(goal_done, new_goals, goal_stack[:, k - 1]))

        attempts_delta = jnp.sum(end.astype(jnp.float32), axis=0)
        success_delta = jnp.sum(jnp.logical_and(end, achieved).astype(jnp.float32), axis=0)

        new_carry = carry._replace(
            nnx_states=tuple(new_states),
            rings=tuple(new_rings),
            windows=tuple(new_windows),
            env_state=env_state2,
            obs=obs2,
            goal_stack=goal_stack_next,
            action_start_obs=action_start_obs,
            attempt_count=jnp.where(goal_done[:, None], 0, attempt_count),
            testing=jnp.where(goal_done[:, None], False, testing),
            needs_new_goal=end,
            seen=seen_next,
            ep_step=jnp.where(goal_done, 0, carry.ep_step + 1),
            stats=carry.stats + jnp.stack([attempts_delta, success_delta], axis=-1),
            step=carry.step + 1,
            rng=rng,
        )
        penalties = jnp.zeros((), jnp.float32)
        for i in range(1, k):
            penalties = penalties + jnp.sum(
                (completes[:, i] & testing[:, i - 1]
                 & (attempt_count[:, i - 1] >= H_levels[i - 1]) & ~achieved[:, i - 1]
                 ).astype(jnp.float32))
        metrics = {
            "goal_done": goal_done,
            "goal_success": jnp.logical_and(goal_done, achieved[:, k - 1]),
            "env_done": env_done,
            "attempt_len": jnp.where(goal_done, carry.ep_step + 1, 0),
            "losses": jnp.stack(losses),
            "trained": jnp.stack(trained),
            "penalties": penalties,
            "epsilon": eps_now,
        }
        return new_carry, metrics

    @partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, keys):
        return jax.lax.scan(scan_body, carry, keys)

    # Main Execution and Evaluation Loop
    from jaxhrl.common.utils import StepScheduler
    from jaxhrl.common.wrappers import run_offpolicy_eval_stage
    eval_config = config.get("eval", {})
    checkpoint_config = config.get("checkpoint", {})
    eval_sched = StepScheduler(n_steps, eval_config.get("interval_pct", 0.05), eval_config.get("enabled", False))
    ckpt_sched = StepScheduler(n_steps, checkpoint_config.get("interval_pct", 0.2), checkpoint_config.get("enabled", True))

    @jax.jit
    def eval_step(nnx_states, obs, goal_stack, attempt_count, needs):
        """One greedy hierarchical step for a single unbatched env."""
        models = [nnx.merge(graphdefs[i], nnx_states[i])[0] for i in range(k)]
        acted = jnp.concatenate([jnp.ones((1,), jnp.bool_), needs[:k - 1]])
        for i in range(k - 2, -1, -1):
            parent = i + 1
            proposed = jnp.argmax(models[parent](obs[None])[0, goal_stack[parent]]).astype(jnp.int32)
            goal_stack = goal_stack.at[i].set(jnp.where(needs[i], proposed, goal_stack[i]))
        attempt_count = jnp.where(needs, 0, attempt_count) + acted.astype(jnp.int32)
        action = jnp.argmax(models[0](obs[None])[0, goal_stack[0]]).astype(jnp.int32)
        return action, goal_stack, attempt_count

    def make_eval_policy_fn(nnx_states, seen):
        """Per-episode closure: the goal stack and attempt counters must reset
        between eval runs; the jitted step stays outside so it compiles once."""
        state = {"goal_stack": None, "attempt_count": jnp.zeros((k,), jnp.int32),
                 "needs": jnp.ones((k,), jnp.bool_)}

        def policy_fn(_params, single_obs, eval_key):
            if state["goal_stack"] is None:
                task = sample_task_goals(eval_key, seen, 1)[0]
                state["goal_stack"] = jnp.zeros((k,), jnp.int32).at[k - 1].set(task)
            action, goal_stack, attempt_count = eval_step(
                nnx_states, single_obs, state["goal_stack"], state["attempt_count"], state["needs"])
            achieved = wrapped.goal_fn(single_obs)[goal_stack][None]
            end = close_out(achieved, jnp.zeros((1,), jnp.bool_), attempt_count[None], 1)[0]
            state.update(goal_stack=goal_stack, attempt_count=attempt_count, needs=end)
            return action, goal_stack[0]

        return policy_fn

    print(f"HierQ (deep): k={k} H_levels={H_levels} gammas={[round(g, 3) for g in gammas]} "
          f"per-goal budget={horizon} | obs {obs_dim}, {n_actions} actions, {G} goals "
          f"({int(seen0.sum())} seen at reset) | eps {epsilon_start}->{epsilon_finish} over "
          f"{epsilon_decay_steps} steps | test every {test_interval_chunks} chunks: "
          f"{G} goals x {test_num_repeats} x {test_num_steps} steps")

    key, tkey = jax.random.split(key)
    run_and_log_test(carry.nnx_states, tkey, 0)
    for chunk_idx, step_idx in enumerate(range(0, n_steps, chunk_size)):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)
        current_env_step = (step_idx + chunk_size) * num_envs

        if (chunk_idx + 1) % test_interval_chunks == 0 or step_idx + chunk_size >= n_steps:
            key, tkey = jax.random.split(key)
            run_and_log_test(carry.nnx_states, tkey, current_env_step)

        if ckpt_sched.due(step_idx):
            logger.save_checkpoint(jax.device_get(carry.nnx_states), current_env_step)

        if eval_sched.due(step_idx):
            print(f"\n--- Running Evaluation at Step {step_idx} ---")
            eval_states, eval_seen = carry.nnx_states, carry.seen
            key = run_offpolicy_eval_stage(
                logger, wrapped, lambda: make_eval_policy_fn(eval_states, eval_seen),
                eval_states, key, current_env_step, eval_config)
            print(f"--- Evaluation Complete ---\n")

    logger.close()
