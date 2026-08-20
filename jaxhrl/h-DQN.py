# h-DQN in JAX
# From "Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation"
from typing import Any, NamedTuple
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import flashbax as fbx
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from jaxhrl.common.utils import parse_config
from jaxhrl.common.logger import Logger
from jaxhrl.common.jax_wrappers import make_jax_env, run_eval_episode


class QNetwork(nnx.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 1024, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        x = nnx.relu(self.ln1(self.linear1(x)))
        x = nnx.relu(self.ln2(self.linear2(x)))
        return self.linear3(x)

def train_controller_step(q1, target1, opt1, buffer1_state, key, controller_buffer, gamma1=0.99):
    batch1 = controller_buffer.sample(buffer1_state, key).experience

    def loss1_fn(model):
        q_pred = model(batch1.first["obs"])
        q_pred_a = jnp.take_along_axis(
            q_pred, batch1.first["action"][:, None], axis=-1
        ).squeeze(-1)

        q_next_online = model(batch1.second["obs"])
        next_action = jnp.argmax(q_next_online, axis=-1)
        q_next_target = target1(batch1.second["obs"])
        q_next_sel = jnp.take_along_axis(
            q_next_target, next_action[:, None], axis=-1
        ).squeeze(-1)

        not_done = 1.0 - batch1.first["done"].astype(jnp.float32)
        target = batch1.first["reward"] + gamma1 * not_done * jax.lax.stop_gradient(q_next_sel)
        return jnp.mean((q_pred_a - jax.lax.stop_gradient(target)) ** 2)

    loss1, grads1 = nnx.value_and_grad(loss1_fn)(q1)
    opt1.update(q1, grads1)
    return loss1


def train_meta_step(q2, target2, opt2, buffer2_state, key, meta_controller_buffer, gamma2=0.99):
    batch2 = meta_controller_buffer.sample(buffer2_state, key).experience

    def loss2_fn(model):
        q_pred = model(batch2.first["obs"])
        q_pred_g = jnp.take_along_axis(
            q_pred, batch2.first["goal"][:, None], axis=-1
        ).squeeze(-1)

        q_next_online = model(batch2.second["obs"])
        next_goal = jnp.argmax(q_next_online, axis=-1)
        q_next_target = target2(batch2.second["obs"])
        q_next_sel = jnp.take_along_axis(
            q_next_target, next_goal[:, None], axis=-1
        ).squeeze(-1)

        not_terminal = 1.0 - batch2.first["terminal"].astype(jnp.float32)
        gamma_macro = gamma2 ** batch2.first["duration"]
        target = batch2.first["F"] + gamma_macro * not_terminal * jax.lax.stop_gradient(q_next_sel)

        valid = batch2.first["valid"].astype(jnp.float32)
        sq_err = (q_pred_g - jax.lax.stop_gradient(target)) ** 2
        return jnp.sum(sq_err * valid) / jnp.maximum(jnp.sum(valid), 1.0)

    loss2, grads2 = nnx.value_and_grad(loss2_fn)(q2)
    opt2.update(q2, grads2)
    return loss2


def where_per_env(done_mask, a, b):
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


class LoopCarry(NamedTuple):    
    goal_duration: jax.Array 
    env_state: Any
    obs: jax.Array
    goal: jax.Array               
    goal_start_obs: jax.Array     
    F: jax.Array                  
    nnx_state: nnx.State          
    buffer1_state: Any
    buffer2_state: Any
    epsilon_1: jax.Array          
    epsilon_2: jax.Array          
    goal_success_stats: jax.Array 
    step: jax.Array
    rng: jax.Array
    cum_reward: jnp.ndarray


if __name__ == "__main__":

    # Parse config
    config = parse_config()
    logger = Logger(config)

    framework_type = config["env"].get("framework", "gymnax")
    env_id = config["env"]["make"]["id"].split("/")[-1]
    n_steps = config["training"].get("n_steps", 20_000)
    chunk_size = config["training"].get("chunk_size", 100)
    eval_freq = config["training"].get("eval_freq", chunk_size)
    num_envs = config["training"].get("num_envs", 1024)
    min_epsilon = config["training"].get("min_epsilon", 0.05)
    eps1_decay = config["training"].get("epsilon_1_decay", 0.995)   # per-attempt decay for controller epsilon
    eps2_decay = config["training"].get("epsilon_2_decay", config["training"].get("epsilon_decay", 0.99999))  # per-step decay for meta epsilon
    train_every_controller = config["training"].get("train_every_controller", 1)
    train_every_meta = config["training"].get("train_every_meta", 8)  # much less frequent
    target_update_every = config["training"].get("target_update_every", 1000)
    cumulant_dim = config["training"].get("cumulant_dim", 16)
    gamma1 = config["training"].get("gamma1")
    gamma2 = config["training"].get("gamma2")

    # define pacing
    initial_goals = config["training"].get("initial_goals")
    fast_unlock_steps = config["training"].get("fast_unlock_steps", 100_000)
    fast_unlock_target = config["training"].get("fast_unlock_target", 40)

    wrapped = make_jax_env(framework_type, env_id, cumulant_dim)
    num_goals = wrapped.num_goals
    state_dim = wrapped.state_dim
    num_actions = wrapped.num_actions

    seed = config["seed"]
    key = jax.random.PRNGKey(seed)
    key, k1_c, k1_t, k2_c, k2_t = jax.random.split(key, 5)

    epsilon_1 = jnp.ones(num_goals)
    epsilon_2 = jnp.array(1.0)

    # Initialize Controller (Q1)
    controller = QNetwork(state_dim + num_goals, num_actions, hidden_dim=1024, rngs=nnx.Rngs(k1_c))
    target1 = QNetwork(state_dim + num_goals, num_actions, hidden_dim=1024, rngs=nnx.Rngs(k1_t))
    nnx.update(target1, jax.tree.map(jnp.copy, nnx.state(controller)))

    # Initialize Meta-Controller (Q2)
    meta_controller = QNetwork(state_dim, num_goals, hidden_dim=1024, rngs=nnx.Rngs(k2_c))
    target2 = QNetwork(state_dim, num_goals, hidden_dim=1024, rngs=nnx.Rngs(k2_t))
    nnx.update(target2, jax.tree.map(jnp.copy, nnx.state(meta_controller)))

    opt1 = nnx.Optimizer(
        controller,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
        wrt=nnx.Param,
    )
    opt2 = nnx.Optimizer(
        meta_controller,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
        wrt=nnx.Param,
    )

    graphdef, initial_nnx_state = nnx.split((controller, target1, opt1, meta_controller, target2, opt2))

    train_batch_size = config["training"].get("batch_size", 64)
    controller_buffer = fbx.make_flat_buffer(
        max_length=config["training"].get("controller_buffer_size", 300_000),
        min_length=train_batch_size,
        sample_batch_size=train_batch_size,
        add_batch_size=num_envs,  
    )

    meta_controller_buffer = fbx.make_flat_buffer(
        max_length=config["training"].get("meta_buffer_size", 30_000),
        min_length=train_batch_size,
        sample_batch_size=train_batch_size,
        add_batch_size=num_envs,  
    )

    dummy_controller_transition = {
        "obs": jnp.zeros((state_dim + num_goals,), jnp.float32),
        "action": jnp.zeros((), jnp.int32),
        "reward": jnp.zeros((), jnp.float32),
        "done": jnp.zeros((), jnp.bool_),
    }
    dummy_meta_transition = {
        "obs": jnp.zeros((state_dim,), jnp.float32),
        "goal": jnp.zeros((), jnp.int32),
        "F": jnp.zeros((), jnp.float32),
        "valid": jnp.zeros((), jnp.bool_),
        "terminal": jnp.zeros((), jnp.bool_),
        "duration": jnp.zeros((), jnp.float32),
    }
    controller_buffer_state0 = controller_buffer.init(dummy_controller_transition)
    meta_controller_buffer_state0 = meta_controller_buffer.init(dummy_meta_transition)

    vmap_reset = jax.jit(jax.vmap(wrapped.reset_fn))
    vmap_step = jax.jit(jax.vmap(wrapped.step_fn, in_axes=(0, 0, 0)))

    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    obs0, state0 = vmap_reset(reset_keys0)

    # ---- Training Loop Core ----
    def scan_body(carry: LoopCarry, step_key):
        rng, action_key, sample1_key, sample2_key, goal_key, reset_key, train_key = jax.random.split(carry.rng, 7)

        q1, target1, opt1, q2, target2, opt2 = nnx.merge(graphdef, carry.nnx_state)

        # Controller action: eps-greedy on Q1({s,g})
        goal_onehot = jax.nn.one_hot(carry.goal, num_goals)
        aug_obs = jnp.concatenate([carry.obs, goal_onehot], axis=-1)
        
        q1_values = q1(aug_obs) 
        greedy_action = jnp.argmax(q1_values, axis=-1)

        action_keys = jax.random.split(action_key, num_envs)
        random_action = jax.vmap(lambda k: jax.random.randint(k, (), 0, num_actions))(action_keys)
        eps_per_env = carry.epsilon_1[carry.goal]
        explore = jax.random.uniform(action_key, (num_envs,)) < eps_per_env
        action = jnp.where(explore, random_action, greedy_action)

        duration_new = carry.goal_duration + 1.0

        # Step envs
        step_keys = jax.random.split(step_key, num_envs)
        obs2, env_state2, extrinsic_r, done, info = vmap_step(step_keys, carry.env_state, action)

        cum_reward_new = carry.cum_reward + extrinsic_r
        completed_return = jnp.where(done, cum_reward_new, 0.0)
        cum_reward_next = jnp.where(done, 0.0, cum_reward_new)

        # Intrinsic reward + goal reached from internal critic
        intrinsic_r, goal_reached = wrapped.goal_reached_fn(carry.obs, carry.goal, obs2, action, info)
        intrinsic_r = jnp.clip(intrinsic_r, -1.0, 1.0) # clip in case reward spike on rare events
        goal_onehot2 = jax.nn.one_hot(carry.goal, num_goals) 
        aug_obs2 = jnp.concatenate([obs2, goal_onehot2], axis=-1)

        controller_done = jnp.logical_or(done, goal_reached)

        controller_transition = {
            "obs": aug_obs,
            "action": action,
            "reward": intrinsic_r,
            "done": controller_done,
        }
        buffer1_state = controller_buffer.add(carry.buffer1_state, controller_transition)

        # Accumulate extrinsic reward for meta controller
        F_new = carry.F + extrinsic_r
        hierarchy_done = jnp.logical_or(done, goal_reached)

        meta_transition = {
            "obs": carry.goal_start_obs,
            "goal": carry.goal,
            "F": F_new,
            "valid": hierarchy_done, 
            "terminal": done,
            "duration": duration_new,
        }
        buffer2_state = meta_controller_buffer.add(carry.buffer2_state, meta_transition)
    

        fast_rate = (fast_unlock_target - initial_goals) / fast_unlock_steps
        slow_rate = (num_goals - fast_unlock_target) / max(n_steps - fast_unlock_steps, 1)

        step_f = carry.step.astype(jnp.float32)
        unlocked_fast = initial_goals + fast_rate * jnp.minimum(step_f, fast_unlock_steps)
        unlocked_slow = fast_unlock_target + slow_rate * jnp.maximum(step_f - fast_unlock_steps, 0.0)
        current_max_goal = jnp.clip(
            jnp.where(step_f < fast_unlock_steps, unlocked_fast, unlocked_slow).astype(jnp.int32),
            initial_goals, num_goals,
        )

        # Pick new goals where current one finished
        q2_values = q2(obs2)
        goal_indices = jnp.arange(num_goals)
        valid_goal_mask = goal_indices < current_max_goal
        q2_values_masked = jnp.where(valid_goal_mask, q2_values, -jnp.inf)
        greedy_goal = jnp.argmax(q2_values_masked, axis=-1)
        
        # 2. Restrict random exploration to only the unlocked subset
        goal_keys = jax.random.split(goal_key, num_envs)
        random_goal = jax.vmap(lambda k: jax.random.randint(k, (), 0, current_max_goal))(goal_keys)
        
        explore_goal = jax.random.uniform(goal_key, (num_envs,)) < carry.epsilon_2
        sampled_goal = jnp.where(explore_goal, random_goal, greedy_goal)

        goal_next = jnp.where(hierarchy_done, sampled_goal, carry.goal)
        goal_start_obs_next = where_per_env(hierarchy_done, obs2, carry.goal_start_obs)
        F_next = jnp.where(hierarchy_done, 0.0, F_new)
        
        # Reset fully done envs
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_state = vmap_reset(reset_keys)

        obs_next = where_per_env(done, reset_obs, obs2)
        env_state_next = jax.tree.map(lambda a, b: where_per_env(done, a, b), reset_state, env_state2)
        goal_next = jnp.where(done, sampled_goal, goal_next)
        goal_start_obs_next = where_per_env(done, reset_obs, goal_start_obs_next)
        F_next = jnp.where(done, 0.0, F_next)
        goal_duration_next = jnp.where(hierarchy_done, 0.0, duration_new)
        goal_duration_next = jnp.where(done, 0.0, goal_duration_next)

        # Adaptive per goal epsilon annealing based on success rate
        attempted = hierarchy_done.astype(jnp.float32)
        succeeded = jnp.logical_and(hierarchy_done, goal_reached).astype(jnp.float32)
        attempt_delta = jnp.zeros((num_goals,)).at[carry.goal].add(attempted)
        success_delta = jnp.zeros((num_goals,)).at[carry.goal].add(succeeded)
        goal_success_stats_next = carry.goal_success_stats + jnp.stack([success_delta, attempt_delta], axis=-1)

        success_rate = success_delta / jnp.maximum(attempt_delta, 1.0)
        ema_mask = attempt_delta > 0
        per_goal_decay = jnp.where(
            ema_mask,
            eps1_decay + (1 - eps1_decay) * (1.0 - success_rate),  # closer to 1.0 (slower decay) when success_rate is low
            1.0,  # goal not attempted this step -> epsilon_1 for it stays unchanged
        )
        epsilon_1_next = jnp.maximum(min_epsilon, carry.epsilon_1 * per_goal_decay)
        epsilon_2_next = jnp.maximum(min_epsilon, carry.epsilon_2 * eps2_decay)
        step_next = carry.step + 1

        _, current_nnx_state = nnx.split((q1, target1, opt1, q2, target2, opt2))

        train_key1, train_key2 = jax.random.split(train_key)

        should_train_1 = jnp.logical_and(
            step_next % train_every_controller == 0,
            controller_buffer.can_sample(buffer1_state),
        )
        should_train_2 = jnp.logical_and(
            step_next % train_every_meta == 0,
            meta_controller_buffer.can_sample(buffer2_state),
        )

        # --- Controller (Q1) update ---
        def do_train_1(state):
            _q1, _t1, _o1, _q2, _t2, _o2 = nnx.merge(graphdef, state)
            l1 = train_controller_step(
                _q1, _t1, _o1, buffer1_state, train_key1, controller_buffer, gamma1=gamma1
            )
            _, next_state = nnx.split((_q1, _t1, _o1, _q2, _t2, _o2))
            return next_state, l1

        def skip_train_1(state):
            return state, 0.0

        state_after_train1, loss1 = jax.lax.cond(
            should_train_1, do_train_1, skip_train_1, current_nnx_state
        )

        # --- Meta-controller (Q2) update ---
        def do_train_2(state):
            _q1, _t1, _o1, _q2, _t2, _o2 = nnx.merge(graphdef, state)
            l2 = train_meta_step(
                _q2, _t2, _o2, buffer2_state, train_key2, meta_controller_buffer, gamma2=gamma2
            )
            _, next_state = nnx.split((_q1, _t1, _o1, _q2, _t2, _o2))
            return next_state, l2

        def skip_train_2(state):
            return state, 0.0

        state_after_train, loss2 = jax.lax.cond(
            should_train_2, do_train_2, skip_train_2, state_after_train1
        )

        # Periodic Hard Target Update
        def hard_update(state):
            _q1, _t1, _o1, _q2, _t2, _o2 = nnx.merge(graphdef, state)
            
            nnx.update(_t1, jax.tree.map(jnp.copy, nnx.state(_q1)))
            nnx.update(_t2, jax.tree.map(jnp.copy, nnx.state(_q2)))
            
            _, next_state = nnx.split((_q1, _t1, _o1, _q2, _t2, _o2))
            return next_state

        def skip_update(state):
            return state

        final_nnx_state = jax.lax.cond(
            step_next % target_update_every == 0,
            hard_update, skip_update, state_after_train
        )

        new_carry = carry._replace(
            goal_duration=goal_duration_next,
            env_state=env_state_next,
            obs=obs_next,
            goal=goal_next,
            goal_start_obs=goal_start_obs_next,
            F=F_next,
            nnx_state=final_nnx_state,
            buffer1_state=buffer1_state,
            buffer2_state=buffer2_state,
            epsilon_1=epsilon_1_next,
            epsilon_2=epsilon_2_next,
            goal_success_stats=goal_success_stats_next,
            step=step_next,
            rng=rng,
            cum_reward=cum_reward_next,
        )

        metrics = {
            "loss1": loss1,
            "loss2": loss2,
            "trained_1": should_train_1,
            "trained_2": should_train_2,
            "done": done,
            "goal_reached": goal_reached,
            "hierarchy_done": hierarchy_done,
            "completed_return": completed_return,
            "goals": carry.goal,
            "infos": info,
        }
        return new_carry, metrics

    # XLA Buffer Donation on pure state
    @partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, keys):
        return jax.lax.scan(scan_body, carry, keys)

    key, goal_key0 = jax.random.split(key)
    random_goal0 = jax.vmap(lambda k: jax.random.randint(k, (), 0, initial_goals))(
        jax.random.split(goal_key0, num_envs)
    )
    goal0 = random_goal0

    # Initialize carry with pure arrays/state
    carry = LoopCarry(
        goal_duration=jnp.zeros((num_envs), dtype=jnp.float32),
        env_state=state0,
        obs=obs0,
        goal=goal0,
        goal_start_obs=jnp.copy(obs0),
        F=jnp.zeros((num_envs,)),
        nnx_state=initial_nnx_state,
        buffer1_state=controller_buffer_state0,
        buffer2_state=meta_controller_buffer_state0,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        goal_success_stats=jnp.zeros((num_goals, 2)),
        step=jnp.array(0, dtype=jnp.int32),
        rng=key,
        cum_reward=jnp.zeros((num_envs,)),
    )

    def run_and_log(carry, key, n, step0):
        prev_stats = jax.device_get(carry.goal_success_stats)  # snapshot before this chunk

        keys = jax.random.split(key, n)
        carry, metrics = run_chunk(carry, keys)
        metrics = jax.device_get(metrics)

        new_stats = jax.device_get(carry.goal_success_stats)
        chunk_stats = new_stats - prev_stats          # successes/attempts *this chunk only*
        chunk_successes, chunk_attempts = chunk_stats[:, 0], chunk_stats[:, 1]

        attempted_mask = chunk_attempts > 0
        per_goal_rate = np.where(attempted_mask, chunk_successes / np.maximum(chunk_attempts, 1), 0.0)

        # macro average: each attempted goal counts equally
        mean_success_rate = float(per_goal_rate[attempted_mask].mean()) if attempted_mask.any() else 0.0
        # micro average: weighted by how often each goal was attempted
        weighted_success_rate = float(chunk_successes.sum() / max(chunk_attempts.sum(), 1))

        # Per-goal success rate and raw counts, only for goals actually attempted this chunk
        # (skip un-attempted goals to avoid spamming the dashboard with dead 0.0 lines)
        per_goal_success_metrics = {}
        for g_idx in np.nonzero(attempted_mask)[0]:
            per_goal_success_metrics[f"goals/success_rate_goal_{g_idx}"] = float(per_goal_rate[g_idx])
            per_goal_success_metrics[f"goals/attempts_goal_{g_idx}"] = float(chunk_attempts[g_idx])

        trained_1 = metrics["trained_1"]
        trained_2 = metrics["trained_2"]
        loss_1 = float(np.sum(metrics["loss1"]) / max(np.sum(trained_1), 1))
        loss_2 = float(np.sum(metrics["loss2"]) / max(np.sum(trained_2), 1))
        total_dones = np.sum(metrics["done"])
        sum_returns = np.sum(metrics["completed_return"])
        true_mean_return = float(np.where(total_dones > 0, sum_returns / total_dones, 0.0))

        env_steps0 = step0 * num_envs

        chunk_metrics = {
            "train/loss_controller": loss_1,
            "train/loss_meta_controller": loss_2,
            "train/return_mean": true_mean_return,
            "goals/mean_success_rate": mean_success_rate,
            "goals/weighted_success_rate": weighted_success_rate,
            "goals/num_goals_attempted": int(attempted_mask.sum()),
            **per_goal_success_metrics,
        }
        logger.log_metrics(chunk_metrics, step=env_steps0)

        # 2. Log Granular Timestep Data
        for t in range(n):
            current_t_step = env_steps0 + (t * num_envs)
            t_metrics = {}

            goals_at_t = metrics["goals"][t]
            for g_idx in range(num_goals):
                t_metrics[f"goals/usage_goal_{g_idx}"] = float(np.mean(goals_at_t == g_idx))

            for key_name, value in metrics["infos"].items():
                if isinstance(value, np.ndarray):
                    t_metrics[f"achievements/{key_name}"] = float(np.mean(value[t]))

            logger.log_metrics(t_metrics, step=current_t_step)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} env-steps) "
              f"| Meta Controller Loss: {loss_2:.4f} | Controller Loss: {loss_1:.4f} | True Return: {true_mean_return:.2f}")
        return carry

    # Main Execution and Evaluation Loop
    eval_config = config.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = int(n_steps * eval_config.get("interval_pct", 0.05))
    eval_max_steps = eval_config.get("max_steps", 2000)
    next_eval_step = eval_interval

    @jax.jit
    def greedy_eval_policy(eval_state, single_obs, eval_key):
        _q1, _, _, _q2, _, _ = nnx.merge(graphdef, eval_state)
        q2_vals = _q2(single_obs[None, :])
        eval_goal = jnp.argmax(q2_vals, axis=-1)[0]
        
        goal_onehot = jax.nn.one_hot(eval_goal, num_goals)
        aug_obs = jnp.concatenate([single_obs, goal_onehot], axis=-1)
        q1_vals = _q1(aug_obs[None, :])
        eval_action = jnp.argmax(q1_vals, axis=-1)[0]
        
        return eval_action, eval_goal

    for step_idx in range(0, n_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)

        # Evaluation Hook
        if do_eval and step_idx >= next_eval_step:
            print(f"\n--- Running Evaluation at Step {step_idx} ---")

            current_env_step = (step_idx + chunk_size) * num_envs

            # ONLY use device_get for saving the checkpoint
            logger.save_checkpoint(jax.device_get(carry.nnx_state), current_env_step)

            # Pass the ON-DEVICE carry.nnx_state to the evaluation policy
            key, eval_key = jax.random.split(key)
            trajectory = run_eval_episode(
                env_wrapper=wrapped,
                policy_fn=greedy_eval_policy,
                params=carry.nnx_state,  # <-- Fixed!
                key=eval_key,
                max_steps=eval_max_steps
            )

            logger.log_eval_trajectory(current_env_step, trajectory)

            next_eval_step += eval_interval
            print(f"--- Evaluation Complete. Total Reward: {sum(trajectory['reward']):.2f} ---\n")

    logger.close()
