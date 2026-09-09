# HierQ in JAX
# From "Learning Multi-Level Hierarchies with Hindsight", Algorithm 2 (Appendix)
from typing import Any, NamedTuple
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from jaxhrl.common.utils import parse_config
from jaxhrl.common.logger import Logger
from jaxhrl.common.wrappers import make_jax_env



def eps_greedy(q_rows: jax.Array, key: jax.Array, epsilon: float, n_actions: int):
    """epsilon-greedy over Q_i(s, g, .).

    Ties are broken randomly. levels above 0 start pessimistically initialised, so every entry of a row is
    identical and a plain argmax would make every level propose action 0
    forever until the first update happens to land on that row.
    """
    tie_key, rand_key, coin_key = jax.random.split(key, 3)
    tie_break = jax.random.uniform(tie_key, q_rows.shape) * 1e-8
    greedy = jnp.argmax(q_rows + tie_break, axis=-1)
    random = jax.random.randint(rand_key, (q_rows.shape[0],), 0, n_actions)
    explore = jax.random.uniform(coin_key, (q_rows.shape[0],)) < epsilon
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def update_level0(q0, states, actions, next_states, alpha, gamma, n_states):
    """Algorithm 2's all-goals update for the primitive level:

        for each state s_goal in S:
            Q0(s, s_goal, a) <- (1-a).Q0(s, s_goal, a)
                                + a.[R0 + g.max_a' Q0(s', s_goal, a')]

    One primitive transition updates the row for EVERY goal simultaneously --
    the tabular form of hindsight goal transitions, done exhaustively instead of
    by sampling a relabelled goal.
    """
    goals = jnp.arange(n_states)                                  # (S,)
    reached = next_states[:, None] == goals[None, :]              # (N, S)
    reward = jnp.where(reached, 0.0, -1.0)
    # Shortest path scheme: discount collapses to 0 exactly when the goal is hit.
    discount = jnp.where(reached, 0.0, gamma)
    bootstrap = jnp.max(q0[next_states], axis=-1)                 # (N, S)
    target = reward + discount * bootstrap

    cur = q0[states[:, None], goals[None, :], actions[:, None]]   # (N, S)
    delta = alpha * (target - cur)
    return q0.at[states[:, None], goals[None, :], actions[:, None]].add(delta)


def update_level_i(qi, window, window_valid, next_states, alpha, gamma, n_states):
    """Algorithm 2's PrevStates update for a subgoal level:

        for each state s in PrevStates_i:
            for each goal s_goal in S:
                Qi(s, s_goal, s') <- (1-a).Qi(s, s_goal, s')
                                     + a.[Ri + g.max_a Qi(s', s_goal, a)]
    """
    goals = jnp.arange(n_states)
    reached = next_states[:, None] == goals[None, :]              # (N, S)
    reward = jnp.where(reached, 0.0, -1.0)
    discount = jnp.where(reached, 0.0, gamma)
    bootstrap = jnp.max(qi[next_states], axis=-1)                 # (N, S)
    target = (reward + discount * bootstrap)[:, None, :]          # (N, 1, S)

    prev = window[:, :, None]                                     # (N, W, 1)
    goal_ax = goals[None, None, :]                                # (1, 1, S)
    act = next_states[:, None, None]                              # (N, 1, 1)
    cur = qi[prev, goal_ax, act]                                  # (N, W, S)
    delta = alpha * (target - cur) * window_valid[:, :, None]
    return qi.at[prev, goal_ax, act].add(delta)


class Window(NamedTuple):
    """PrevStates_i: a rolling record of the last H**i states visited this
    episode. Length H**i is exactly how many primitive steps one level-i action
    may span, so every state still in the window is one from which the current
    state was reachable inside this level's horizon."""
    states: jax.Array   # (num_envs, H**i) int32
    count: jax.Array    # (num_envs,) int32 -- valid entries, capped at H**i


def push_window(window: Window, next_states: jax.Array, capacity: int, env_idx: jax.Array):
    # `count` must stay monotonic: it is both the write cursor (count % capacity)
    # and the fill level. 
    slot = window.count % capacity
    return Window(
        states=window.states.at[env_idx, slot].set(next_states),
        count=window.count + 1,
    )


def window_mask(window: Window, capacity: int, slot_idx: jax.Array):
    return (slot_idx[None, :] < jnp.minimum(window.count, capacity)[:, None]).astype(jnp.float32)


def where_per_env(done_mask, a, b):
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


class LoopCarry(NamedTuple):
    q_tables: tuple             # length k; Q0 is (S,S,A), Qi>0 is (S,S,S)
    windows: tuple              # length k-1; Window for levels 1..k-1
    env_state: Any
    obs: jax.Array              # (num_envs,) int32 state index
    goal_stack: jax.Array       # (num_envs, k) int32; [:, k-1] is the task goal
    attempt_count: jax.Array    # (num_envs, k) int32 -- actions taken this attempt
    needs_new_goal: jax.Array   # (num_envs, k) bool  -- `end` from the previous step
    ep_step: jax.Array          # (num_envs,) int32
    stats: jax.Array            # (k, 2) float32 cumulative [attempts, successes]
    step: jax.Array
    rng: jax.Array


if __name__ == "__main__":

    config = parse_config()
    logger = Logger(config)

    framework_type = config["env"].get("framework", "gridworld")
    env_id = config["env"]["make"]["id"].split("/")[-1]

    n_steps = config["training"].get("n_steps", 20_000)
    chunk_size = config["training"].get("chunk_size", 500)
    num_envs = config["training"].get("num_envs", 64)
    k = config["training"].get("num_levels", 3)
    H = config["training"].get("H", 5)
    alpha = config["training"].get("alpha", 0.1)
    # Default None -> derive gamma per level from that level's horizon.
    gamma = config["training"].get("gamma", None)
    epsilon = config["training"].get("epsilon", 0.2)

    wrapped = make_jax_env(framework_type, env_id, cumulant_dim=1)
    n_states = wrapped.num_goals
    n_actions = wrapped.num_actions
    assert n_states > 0 and n_actions > 0, "HierQ is tabular; use a discrete env."

    # Per-level action budgets. Levy holds the sub-levels at a constant
    # time_scale and lets only the TOP level's budget absorb the episode
    # horizon.
    #
    # The paper's outer loop runs the top level until the task goal is hit; a
    # fixed cap is required here for static shapes, and bounds the episode in
    # practice either way.
    horizon_cfg = config["training"].get("horizon", None)
    if horizon_cfg is not None:
        top_H = max(1, int(round(horizon_cfg / (H ** (k - 1)))))
    else:
        top_H = H
    H_levels = [H] * (k - 1) + [top_H]
    horizon = int(np.prod(H_levels))

    # "Set Qi(s, g, a), i > 0 to some pessimistic initialization." Any value
    # strictly below the worst return a reachable subgoal can earn (-H, i.e. H
    # actions each costing -1) leaves unreachable subgoals ranked last forever,
    # which is what replaces HAC's subgoal-testing penalties.
    pessimistic_cfg = config["training"].get("pessimistic_init", None)
    gammas = [gamma if gamma is not None else 1.0 - 1.0 / h for h in H_levels]
    pessimistics = [pessimistic_cfg if pessimistic_cfg is not None
                    else -1.0 / (1.0 - g) for g in gammas]

    seed = config["seed"]
    key = jax.random.PRNGKey(seed)

    # Level 0's actions are primitive; every level above acts in the state space
    # (A_i = S), so its table is (S, S, S). Level 0 starts optimistic at 0, as
    # Algorithm 2 prescribes pessimism only for i > 0.
    q_tables0 = [jnp.zeros((n_states, n_states, n_actions), jnp.float32)]
    for i in range(1, k):
        q_tables0.append(jnp.full((n_states, n_states, n_states), pessimistics[i], jnp.float32))

    windows0 = []
    window_caps = []
    for i in range(1, k):
        # PrevStates_i spans one level-i ACTION, i.e. the product of the action
        # budgets of every level BELOW i -- H**i only when every level shares H.
        cap = int(np.prod(H_levels[:i]))
        window_caps.append(cap)
        windows0.append(Window(
            states=jnp.zeros((num_envs, cap), jnp.int32),
            count=jnp.zeros((num_envs,), jnp.int32),
        ))

    vmap_reset = jax.jit(jax.vmap(wrapped.reset_fn))
    vmap_step = jax.jit(jax.vmap(wrapped.step_fn, in_axes=(0, 0, 0)))

    env_idx = jnp.arange(num_envs)
    slot_idxs = [jnp.arange(c) for c in window_caps]

    def sample_task_goals(rng_key):
        return jax.random.randint(rng_key, (num_envs,), 0, n_states)

    # ---- Training Loop Core ----
    def scan_body(carry: LoopCarry, step_key):
        rng, select_key, reset_key, goal_key = jax.random.split(carry.rng, 4)

        obs = carry.obs
        goal_stack = carry.goal_stack
        needs = carry.needs_new_goal
        q = list(carry.q_tables)

        # Level i takes a new action iff the previous step ended level i-1's
        # attempt; level 0 acts every primitive step. Level i is given a fresh
        # goal iff the previous step ended its own attempt. Since end[i] implies
        # end[i-1], the two stay consistent with no extra bookkeeping.
        acted = jnp.concatenate(
            [jnp.ones((num_envs, 1), jnp.bool_), needs[:, :k - 1]], axis=1)

        # ---- Top-down subgoal selection, i = k-2 .. 0 (strictly descending) ----
        # goal_stack[i] is chosen by level i+1 conditioned on goal_stack[i+1],
        # which may itself have just been refreshed, so order matters.
        select_keys = jax.random.split(select_key, k)
        for i in range(k - 2, -1, -1):
            parent = i + 1
            rows = q[parent][obs, goal_stack[:, parent]]          # (N, S)
            proposed = eps_greedy(rows, select_keys[parent], epsilon, n_states)
            refresh = needs[:, i]
            goal_stack = goal_stack.at[:, i].set(
                jnp.where(refresh, proposed, goal_stack[:, i]))

        attempt_count = jnp.where(needs, 0, carry.attempt_count) + acted.astype(jnp.int32)

        # ---- Level 0 acts, environment steps ----
        rows0 = q[0][obs, goal_stack[:, 0]]                       # (N, A)
        action = eps_greedy(rows0, select_keys[0], epsilon, n_actions)
        step_keys = jax.random.split(step_key, num_envs)
        obs2, env_state2, _, env_done, _ = vmap_step(step_keys, carry.env_state, action)
        s2 = obs2[:, 0].astype(jnp.int32)

        achieved = s2[:, None] == goal_stack                      # (N, k)

        # ---- Close-out, two passes ----
        # Pass 1 uses local_end[i-1] rather than end[i-1]: end[i-1] depends on
        # end[i], which depends on level i's action completing, which is
        # end[i-1] -- circular. Ancestor unwinding is applied by the suffix-OR.
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
        end = jnp.stack(end, axis=1)                              # (N, k)
        episode_done = end[:, k - 1]

        # ---- Q updates ----
        q[0] = update_level0(q[0], obs, action, s2, alpha, gammas[0], n_states)

        new_windows = []
        for i in range(1, k):
            cap = window_caps[i - 1]
            w = push_window(carry.windows[i - 1], s2, cap, env_idx)
            valid = window_mask(w, cap, slot_idxs[i - 1])
            q[i] = update_level_i(q[i], w.states, valid, s2, alpha, gammas[i], n_states)
            new_windows.append(w)

        # ---- Episode reset (after the updates, on the pre-reset state) ----
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_state = vmap_reset(reset_keys)
        fresh_goals = sample_task_goals(goal_key)

        obs_next = jnp.where(episode_done, reset_obs[:, 0].astype(jnp.int32), s2)
        env_state_next = jax.tree.map(
            lambda a, b: where_per_env(episode_done, a, b), reset_state, env_state2)
        goal_stack_next = goal_stack.at[:, k - 1].set(
            jnp.where(episode_done, fresh_goals, goal_stack[:, k - 1]))
        attempt_count_next = jnp.where(episode_done[:, None], 0, attempt_count)
        # PrevStates is per-episode: a state from the previous episode is not one
        # the current state was reached from.
        new_windows = [w._replace(count=jnp.where(episode_done, 0, w.count))
                       for w in new_windows]

        ep_len_completed = jnp.where(episode_done, carry.ep_step + 1, 0)
        attempts_delta = jnp.sum(end.astype(jnp.float32), axis=0)
        success_delta = jnp.sum(jnp.logical_and(end, achieved).astype(jnp.float32), axis=0)
        stats_next = carry.stats + jnp.stack([attempts_delta, success_delta], axis=-1)

        new_carry = carry._replace(
            q_tables=tuple(q),
            windows=tuple(new_windows),
            env_state=env_state_next,
            obs=obs_next,
            goal_stack=goal_stack_next,
            attempt_count=attempt_count_next,
            needs_new_goal=end,
            ep_step=jnp.where(episode_done, 0, carry.ep_step + 1),
            stats=stats_next,
            step=carry.step + 1,
            rng=rng,
        )

        metrics = {
            "episode_done": episode_done,
            "ep_len_completed": ep_len_completed,
            "end_goal_reached": jnp.logical_and(episode_done, achieved[:, k - 1]),
        }
        return new_carry, metrics

    @partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, keys):
        return jax.lax.scan(scan_body, carry, keys)

    key, reset_key0, goal_key0 = jax.random.split(key, 3)
    obs0, state0 = vmap_reset(jax.random.split(reset_key0, num_envs))

    goal_stack0 = jnp.zeros((num_envs, k), jnp.int32)
    goal_stack0 = goal_stack0.at[:, k - 1].set(sample_task_goals(goal_key0))

    carry = LoopCarry(
        q_tables=tuple(q_tables0),
        windows=tuple(windows0),
        env_state=state0,
        obs=obs0[:, 0].astype(jnp.int32),
        goal_stack=goal_stack0,
        attempt_count=jnp.zeros((num_envs, k), jnp.int32),
        needs_new_goal=jnp.ones((num_envs, k), jnp.bool_),   # everyone selects on step 1
        ep_step=jnp.zeros((num_envs,), jnp.int32),
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
        success_rate = float(np.sum(metrics["end_goal_reached"]) / max(total_episodes, 1))
        mean_ep_len = float(np.sum(metrics["ep_len_completed"]) / max(total_episodes, 1))

        chunk_metrics = {
            "train/end_goal_success_rate": success_rate,
            "train/episode_len_mean": mean_ep_len,
            "train/episodes": int(total_episodes),
        }
        for i in range(k):
            chunk_metrics[f"levels/attempts_level_{i}"] = float(attempts[i])
            chunk_metrics[f"levels/success_rate_level_{i}"] = float(
                successes[i] / max(attempts[i], 1))
        logger.log_metrics(chunk_metrics, step=env_steps0)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-"
              f"{env_steps0 + n * num_envs} env-steps) | Success: {success_rate:.3f} "
              f"| Episodes: {int(total_episodes)} | Ep len: {mean_ep_len:.1f}")
        return carry

    # Main Execution and Evaluation Loop
    from jaxhrl.common.utils import StepScheduler
    from jaxhrl.common.wrappers import run_offpolicy_eval_stage
    eval_config = config.get("eval", {})
    checkpoint_config = config.get("checkpoint", {})
    eval_sched = StepScheduler(n_steps, eval_config.get("interval_pct", 0.05), eval_config.get("enabled", False))
    ckpt_sched = StepScheduler(n_steps, checkpoint_config.get("interval_pct", 0.2), checkpoint_config.get("enabled", True))

    @jax.jit
    def eval_step(q_tables, s, goal_stack, attempt_count, needs, task_goal):
        """One greedy hierarchical step for a single unbatched env."""
        goal_stack = goal_stack.at[k - 1].set(task_goal)
        acted = jnp.concatenate([jnp.ones((1,), jnp.bool_), needs[:k - 1]])
        for i in range(k - 2, -1, -1):
            parent = i + 1
            proposed = jnp.argmax(q_tables[parent][s, goal_stack[parent]])
            goal_stack = goal_stack.at[i].set(
                jnp.where(needs[i], proposed.astype(jnp.int32), goal_stack[i]))
        attempt_count = jnp.where(needs, 0, attempt_count) + acted.astype(jnp.int32)
        action = jnp.argmax(q_tables[0][s, goal_stack[0]]).astype(jnp.int32)
        return action, goal_stack, attempt_count

    def make_eval_policy_fn(q_tables):
        """Factory per episode: the goal stack and attempt counters are
        per-episode state and must reset between eval runs, so they live in a
        closure while the jitted step stays outside and is compiled once."""
        state = {"goal_stack": jnp.zeros((k,), jnp.int32),
                 "attempt_count": jnp.zeros((k,), jnp.int32),
                 "needs": jnp.ones((k,), jnp.bool_),
                 "task_goal": None}

        def policy_fn(_params, single_obs, eval_key):
            s = single_obs[0].astype(jnp.int32)
            if state["task_goal"] is None:
                state["task_goal"] = jax.random.randint(eval_key, (), 0, n_states)
            action, goal_stack, attempt_count = eval_step(
                q_tables, s, state["goal_stack"], state["attempt_count"],
                state["needs"], state["task_goal"])

            achieved = s == goal_stack
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

    print(f"HierQ: k={k} H_levels={H_levels} gammas={[round(g,3) for g in gammas]} horizon={horizon} | {n_states} states, {n_actions} actions "
          f"| tables: Q0 {q_tables0[0].shape}" +
          (f", Qi {q_tables0[1].shape}" if k > 1 else ""))

    for step_idx in range(0, n_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)
        current_env_step = (step_idx + chunk_size) * num_envs

        if ckpt_sched.due(step_idx):
            logger.save_checkpoint(jax.device_get(carry.q_tables), current_env_step)

        if eval_sched.due(step_idx):
            print(f"\n--- Running Evaluation at Step {step_idx} ---")
            eval_tables = carry.q_tables
            key = run_offpolicy_eval_stage(
                logger, wrapped, lambda: make_eval_policy_fn(eval_tables),
                eval_tables, key, current_env_step, eval_config)
            print(f"--- Evaluation Complete ---\n")

    logger.close()
