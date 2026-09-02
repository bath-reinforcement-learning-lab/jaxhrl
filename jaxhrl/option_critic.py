# Option-Critic Architecture (Bacon, Harb & Precup, 2017)
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flashbax as fbx
import functools
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from brll_core.algorithms.common.utils import parse_config
from brll_core.algorithms.common.logger import Logger
from brll_core.algorithms.common.jax_wrappers import make_jax_env


class OptionCriticNetwork(nn.Module):
    num_options: int
    num_actions: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        q_w = nn.Dense(self.num_options)(x)  # Q_Omega(s, w)
        beta_logits = nn.Dense(self.num_options, bias_init=nn.initializers.constant(-2.0))(x)  # beta_w(s)
        action_logits = nn.Dense(self.num_options * self.num_actions)(x)
        action_logits = action_logits.reshape((x.shape[0], self.num_options, self.num_actions))

        return q_w, beta_logits, action_logits


def select_option_critic_action(key, obs, option, done, params, net, config):
    """Call-and-return: if the previously active option just terminated (or
    this is the first step of an episode), sample a new option epsilon-
    greedily over Q_Omega; otherwise keep it. Then sample an action from the
    (possibly-new) option's intra-option policy."""
    q_w, beta_logits, action_logits = net.apply(params, obs[None, :])
    q_w, beta_logits, action_logits = q_w[0], beta_logits[0], action_logits[0]

    option = jnp.where(done, -1, option)
    beta = jax.nn.sigmoid(beta_logits)

    key, term_key, opt_key, eps_key, act_key = jax.random.split(key, 5)
    current_beta = jnp.where(option >= 0, beta[jnp.maximum(option, 0)], 1.0)
    terminate = (option < 0) | jax.random.bernoulli(term_key, current_beta)

    greedy_option = jnp.argmax(q_w)
    random_option = jax.random.randint(opt_key, (), 0, config['num_options'])
    is_random = jax.random.uniform(eps_key) < config['epsilon']
    sampled_option = jnp.where(is_random, random_option, greedy_option)

    new_option = jnp.where(terminate, sampled_option, option)

    logits_o = action_logits[new_option]
    action = jax.random.categorical(act_key, logits_o)
    logp = jax.nn.log_softmax(logits_o)[action]

    return action, new_option, logp, q_w[new_option], terminate, key


def batch_select_option_critic_action(keys, obs, options, dones, params, net, config):
    batched_step_fn = jax.vmap(
        select_option_critic_action, in_axes=(0, 0, 0, 0, None, None, None)
    )
    return batched_step_fn(keys, obs, options, dones, params, net, config)


def option_critic_loss_fn(params, target_params, net, batch, config):
    obs = batch.first["obs"]
    next_obs = batch.second["obs"]
    actions = batch.first["action"]
    options = batch.first["option"]
    rewards = batch.first["reward"]
    dones = batch.first["done"].astype(jnp.float32)
    batch_idx = jnp.arange(obs.shape[0])

    q_w, _, action_logits = net.apply(params, obs)
    q_selected = q_w[batch_idx, options]

    logits_o = action_logits[batch_idx, options]
    log_probs_all = jax.nn.log_softmax(logits_o, axis=-1)
    logp = jnp.take_along_axis(log_probs_all, actions[:, None], axis=-1)[:, 0]
    probs = jnp.exp(log_probs_all)
    entropy = jnp.mean(-jnp.sum(probs * log_probs_all, axis=-1))

    # Bootstrap target uses the (Polyak-averaged) target network.
    next_q_target, next_beta_logits_target, _ = net.apply(target_params, next_obs)
    next_beta_target = jax.nn.sigmoid(next_beta_logits_target)
    next_beta_o_target = next_beta_target[batch_idx, options]
    v_next_target = jnp.max(next_q_target, axis=-1)
    q_next_o_target = next_q_target[batch_idx, options]

    bootstrap = (1.0 - next_beta_o_target) * q_next_o_target + next_beta_o_target * v_next_target
    target = rewards + config['gamma'] * (1.0 - dones) * bootstrap
    target = jax.lax.stop_gradient(target)

    critic_loss = jnp.mean((q_selected - target) ** 2)

    advantage = jax.lax.stop_gradient(target - q_selected)
    actor_loss = -jnp.mean(logp * advantage)

    # Termination gradient (Bacon et al., eq. 4), evaluated with the ONLINE
    # network on s' -- this is what actually trains beta_w.
    next_q_online, next_beta_logits_online, _ = net.apply(params, next_obs)
    beta_next_online = jax.nn.sigmoid(next_beta_logits_online)
    beta_next_o = beta_next_online[batch_idx, options]
    v_next_online = jnp.max(next_q_online, axis=-1)
    q_next_o_online = next_q_online[batch_idx, options]

    termination_advantage = jax.lax.stop_gradient(
        q_next_o_online - v_next_online + config['delib_cost']
    )
    nonterminal = 1.0 - dones
    termination_loss = jnp.mean(nonterminal * beta_next_o * termination_advantage)

    total_loss = (actor_loss + config['value_coef'] * critic_loss
                  - config['entropy_coef'] * entropy + termination_loss)
    aux = {
        "actor_loss": actor_loss, "critic_loss": critic_loss,
        "entropy": entropy, "termination_loss": termination_loss,
    }
    return total_loss, aux


# =============================================================================
# Main Execution & JIT Training Logic
# =============================================================================


if __name__ == "__main__":
    config_raw = parse_config()
    logger = Logger(config_raw)

    framework_type = config_raw["env"].get("framework", "gymnax")
    env_id = config_raw["env"]["make"]["id"].split("/")[-1]
    env = make_jax_env(framework_type, env_id, 1)

    config = {
        'num_steps': config_raw["training"].get("num_steps", 1_000_000),
        'num_envs': config_raw["training"].get("num_envs", 64),
        'num_actions': env.num_actions,
        'num_options': config_raw["training"].get("num_options", 8),
        'epsilon': config_raw["training"].get("epsilon", 0.1),
        'buffer_size': config_raw["training"].get("buffer_size", 100_000),
        'batch_size': config_raw["training"].get("batch_size", 256),
        'warmup_steps': config_raw["training"].get("warmup_steps", 5_000),
        'chunk_size': config_raw["training"].get("chunk_size", 100),  # steps per host sync
        'gamma': config_raw["training"].get("gamma", 0.99),
        'lr': config_raw["training"].get("lr", 1e-4),
        'target_tau': config_raw["training"].get("target_tau", 0.005),  # Polyak coefficient
        'delib_cost': config_raw["training"].get("delib_cost", 0.0),
        'value_coef': config_raw["training"].get("value_coef", 0.5),
        'entropy_coef': config_raw["training"].get("entropy_coef", 0.01),
        'hidden_dim': config_raw["training"].get("hidden_dim", 256),
    }
    config['obs_dim'] = env.state_dim
    num_envs = config['num_envs']

    seed = config_raw["seed"]
    key = jax.random.PRNGKey(seed)

    # Initialize Network
    dummy_obs = jnp.zeros((1, config['obs_dim']), dtype=jnp.float32)
    net = OptionCriticNetwork(
        num_options=config['num_options'], num_actions=config['num_actions'],
        hidden_dim=config['hidden_dim'],
    )
    key, init_key = jax.random.split(key)
    params = net.init(init_key, dummy_obs)
    target_params = jax.tree.map(jnp.copy, params)

    # Optimizer
    optimizer = optax.adam(config['lr'])
    opt_state = optimizer.init(params)

    # Initialize Flashbax Buffer
    buffer = fbx.make_flat_buffer(
        max_length=config['buffer_size'],
        min_length=max(config['warmup_steps'], config['batch_size']),
        sample_batch_size=config['batch_size'],
        add_batch_size=num_envs,
    )
    dummy_transition = {
        "obs": jnp.zeros((config['obs_dim'],), dtype=jnp.float32),
        "action": jnp.zeros((), dtype=jnp.int32),
        "option": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=bool),
    }
    buffer_state = buffer.init(dummy_transition)

    # Initialize Environment & Agent States
    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    vmap_reset = jax.vmap(env.reset_fn)
    obs0, state0 = vmap_reset(reset_keys0)

    dones = jnp.ones(num_envs, dtype=bool)  # forces an option decision on the first step
    options = jnp.full(num_envs, -1, dtype=jnp.int32)
    ep_returns = jnp.zeros(num_envs, dtype=jnp.float32)
    global_step = jnp.array(0, dtype=jnp.int32)

    # Initial Carry
    carry = (
        state0, obs0, dones, options, ep_returns,
        buffer_state, params, target_params, opt_state, global_step,
    )

    # -----------------------------------------------------
    # JITTED Core Execution Step
    # -----------------------------------------------------

    def polyak_update(online, target, tau):
        return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

    @functools.partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, step_keys):
        """Scans over n steps fully on accelerator."""

        def scan_step(c, rng):
            (env_states, obs, dones, options, ep_returns,
             buffer_state, params, target_params, opt_state, step) = c

            rng, action_rng = jax.random.split(rng)
            action_keys = jax.random.split(action_rng, num_envs)
            actions, next_options, logps, values, terminated, _ = batch_select_option_critic_action(
                action_keys, obs, options, dones, params, net, config
            )

            rng, step_rng_env = jax.random.split(rng)
            step_env_keys = jax.random.split(step_rng_env, num_envs)
            batch_step_fn = jax.vmap(env.step_fn, in_axes=(0, 0, 0))
            next_obs, next_env_states, rewards, next_dones, infos = batch_step_fn(
                step_env_keys, env_states, actions
            )

            next_ep_returns_running = ep_returns + rewards
            completed_return = jnp.where(next_dones, next_ep_returns_running, 0.0)
            next_ep_returns = jnp.where(next_dones, 0.0, next_ep_returns_running)

            transitions = {
                "obs": obs, "action": actions, "option": next_options,
                "reward": rewards, "done": next_dones,
            }
            buffer_state = buffer.add(buffer_state, transitions)

            # --- Optimization step (skipped via lax.cond until buffer is warm) ---
            can_train = buffer.can_sample(buffer_state)

            def do_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                rng, sample_key = jax.random.split(rng)
                batch = buffer.sample(buffer_state, sample_key).experience

                def loss(p):
                    return option_critic_loss_fn(p, target_params, net, batch, config)

                (loss_val, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                new_target_params = polyak_update(new_params, target_params, config['target_tau'])
                return new_params, new_target_params, new_opt_state, loss_val, aux

            def skip_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                zero = jnp.array(0.0)
                aux = {"actor_loss": zero, "critic_loss": zero, "entropy": zero, "termination_loss": zero}
                return params, target_params, opt_state, zero, aux

            rng, train_rng = jax.random.split(rng)
            train_operands = (train_rng, buffer_state, params, target_params, opt_state)
            params, target_params, opt_state, loss_val, aux = jax.lax.cond(
                can_train, do_train, skip_train, train_operands
            )

            next_step = step + num_envs

            metrics = {
                "loss": loss_val, "done": next_dones, "completed_return": completed_return,
                "options": next_options, "terminated": terminated,
                **aux,
            }
            next_c = (
                next_env_states, next_obs, next_dones, next_options, next_ep_returns,
                buffer_state, params, target_params, opt_state, next_step,
            )
            return next_c, metrics

        return jax.lax.scan(scan_step, carry, step_keys)

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    def run_and_log(carry, rng_key, n, step0):
        keys = jax.random.split(rng_key, n)
        carry, metrics = run_chunk(carry, keys)
        metrics = jax.device_get(metrics)

        total_dones = np.sum(metrics["done"])
        sum_returns = np.sum(metrics["completed_return"])
        true_mean_return = float(np.where(total_dones > 0, sum_returns / total_dones, 0.0))

        env_steps0 = step0 * num_envs

        chunk_metrics = {
            "train/loss": float(np.mean(metrics["loss"])),
            "train/actor_loss": float(np.mean(metrics["actor_loss"])),
            "train/critic_loss": float(np.mean(metrics["critic_loss"])),
            "train/entropy": float(np.mean(metrics["entropy"])),
            "train/termination_loss": float(np.mean(metrics["termination_loss"])),
            "train/termination_rate": float(np.mean(metrics["terminated"])),
            "train/return_mean": true_mean_return,
        }
        logger.log_metrics(chunk_metrics, step=env_steps0)

        num_options = config['num_options']
        for t in range(n):
            current_t_step = env_steps0 + (t * num_envs)
            opts_at_t = metrics["options"][t]
            t_metrics = {
                f"options/usage_{opt_idx}": float(np.mean(opts_at_t == opt_idx))
                for opt_idx in range(num_options)
            }
            logger.log_metrics(t_metrics, step=current_t_step)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} env-steps) "
              f"| Loss: {chunk_metrics['train/loss']:.4f} | True Return: {true_mean_return:.2f}")

        return carry

    # -----------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------
    print("Starting Option-Critic parallel rollouts...")

    total_steps = config['num_steps']
    chunk_size = config['chunk_size']

    eval_config = config_raw.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = int(total_steps * eval_config.get("interval_pct", 0.05))
    eval_max_steps = eval_config.get("max_steps", 2000)
    next_eval_step = eval_interval

    @jax.jit
    def greedy_eval_policy(eval_params, single_obs, eval_key):
        # Greedy option each step (run_eval_episode's policy_fn is stateless
        # across steps, so option persistence can't be threaded through it --
        # same constraint as DCEO.py's eval policy) with zero exploration.
        eval_conf = dict(config)
        eval_conf['epsilon'] = 0.0
        act, opt, _, _, _, _ = select_option_critic_action(
            key=eval_key, obs=single_obs, option=jnp.array(-1, dtype=jnp.int32),
            done=jnp.array(True), params=eval_params, net=net, config=eval_conf,
        )
        return act, opt

    for step_idx in range(0, total_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)

        if do_eval and step_idx >= next_eval_step:
            print(f"\n--- Running Evaluation at Step {step_idx} ---")

            frozen_params = jax.device_get(carry[6])
            current_env_step = (step_idx + chunk_size) * num_envs

            logger.save_checkpoint(frozen_params, current_env_step)

            key, eval_key = jax.random.split(key)
            from brll_core.algorithms.common.jax_wrappers import run_eval_episode
            trajectory = run_eval_episode(
                env_wrapper=env,
                policy_fn=greedy_eval_policy,
                params=frozen_params,
                key=eval_key,
                max_steps=eval_max_steps,
            )

            logger.log_eval_trajectory(current_env_step, trajectory)

            next_eval_step += eval_interval
            print(f"--- Evaluation Complete. Total Reward: {sum(trajectory['reward']):.2f} ---\n")

    print("Training completed.")
    logger.close()
