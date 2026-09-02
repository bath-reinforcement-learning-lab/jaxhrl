# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction
# (Park, Rybkin & Levine, 2023). A skill encoder phi(s) learns a metric
# embedding of the state space; the agent is rewarded for moving phi(s) in
# the direction of a sampled skill vector z, subject to a Lagrangian
# constraint that keeps consecutive phi-steps roughly unit-length (so z
# indexes genuinely different, temporally-extended behaviors instead of
# collapsing). The skill-conditioned policy itself is trained with
# discrete-action SAC using the METRA reward as the (only) reward signal.
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


class Encoder(nn.Module):
    z_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = state.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.z_dim)(x)


class Actor(nn.Module):
    hidden_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, obs, z):
        x = jnp.concatenate([obs.astype(jnp.float32), z], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)


class Critic(nn.Module):
    hidden_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, obs, z):
        x = jnp.concatenate([obs.astype(jnp.float32), z], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)


def sample_z(key, num_envs, z_dim, unit_z):
    z = jax.random.normal(key, (num_envs, z_dim))
    if unit_z:
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    return z


def metra_action(key, obs, z, actor_params, actor_net, greedy=False):
    logits = actor_net.apply(actor_params, obs[None, :], z[None, :])[0]
    return jnp.where(greedy, jnp.argmax(logits), jax.random.categorical(key, logits))


def batch_metra_action(keys, obs, z, actor_params, actor_net):
    batched_step_fn = jax.vmap(metra_action, in_axes=(0, 0, 0, None, None))
    return batched_step_fn(keys, obs, z, actor_params, actor_net)


def metra_components(phi_params, phi_net, obs, next_obs, z, nonterminal, config):
    phi_obs = phi_net.apply(phi_params, obs)
    phi_next = phi_net.apply(phi_params, next_obs)
    phi_diff = phi_next - phi_obs

    raw_r = jnp.sum(phi_diff * z, axis=-1)
    r = raw_r * nonterminal

    sq_dist_unmasked = jnp.mean(jnp.square(phi_diff), axis=-1)
    sq_dist = sq_dist_unmasked * nonterminal

    cst_penalty = jnp.minimum(1.0 - sq_dist_unmasked, config['lagrange_eps']) * nonterminal
    phi_delta_norm = jnp.linalg.norm(phi_diff, axis=-1)

    return {
        "raw_r": raw_r, "r": r, "sq_dist": sq_dist,
        "sq_dist_unmasked": sq_dist_unmasked, "cst_penalty": cst_penalty,
        "phi_delta_norm": phi_delta_norm,
    }


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
        'z_dim': config_raw["training"].get("z_dim", 8),
        'unit_z': config_raw["training"].get("unit_z", True),
        'buffer_size': config_raw["training"].get("buffer_size", 100_000),
        'batch_size': config_raw["training"].get("batch_size", 256),
        'warmup_steps': config_raw["training"].get("warmup_steps", 5_000),
        'chunk_size': config_raw["training"].get("chunk_size", 100),  # steps per host sync
        'gamma': config_raw["training"].get("gamma", 0.99),
        'lr': config_raw["training"].get("lr", 3e-4),
        'target_tau': config_raw["training"].get("target_tau", 0.005),  # Polyak coefficient
        'lagrange_eps': config_raw["training"].get("lagrange_eps", 1e-3),
        'alpha_init': config_raw["training"].get("alpha_init", 0.001),
        'lambda_init': config_raw["training"].get("lambda_init", 30.0),
        'target_entropy': config_raw["training"].get("target_entropy", 0.9),
        'hidden_dim': config_raw["training"].get("hidden_dim", 256),
    }
    config['obs_dim'] = env.state_dim
    num_envs = config['num_envs']

    seed = config_raw["seed"]
    key = jax.random.PRNGKey(seed)

    # Initialize Networks
    dummy_obs = jnp.zeros((1, config['obs_dim']), dtype=jnp.float32)
    dummy_z = jnp.zeros((1, config['z_dim']), dtype=jnp.float32)

    actor_net = Actor(hidden_dim=config['hidden_dim'], num_actions=config['num_actions'])
    q1_net = Critic(hidden_dim=config['hidden_dim'], num_actions=config['num_actions'])
    q2_net = Critic(hidden_dim=config['hidden_dim'], num_actions=config['num_actions'])
    phi_net = Encoder(z_dim=config['z_dim'], hidden_dim=config['hidden_dim'])

    key, actor_key, q1_key, q2_key, phi_key = jax.random.split(key, 5)
    params = {
        'actor': actor_net.init(actor_key, dummy_obs, dummy_z),
        'q1': q1_net.init(q1_key, dummy_obs, dummy_z),
        'q2': q2_net.init(q2_key, dummy_obs, dummy_z),
        'phi': phi_net.init(phi_key, dummy_obs),
        'log_alpha': jnp.array(jnp.log(config['alpha_init']), dtype=jnp.float32),
        'log_lambda': jnp.array(jnp.log(config['lambda_init']), dtype=jnp.float32),
    }
    target_params = {
        'q1': jax.tree.map(jnp.copy, params['q1']),
        'q2': jax.tree.map(jnp.copy, params['q2']),
    }

    # Optimizers (one Adam instance per component, all at the same LR --
    actor_optimizer = optax.adam(config['lr'])
    critic_optimizer = optax.adam(config['lr'])
    phi_optimizer = optax.adam(config['lr'])
    alpha_optimizer = optax.adam(config['lr'])
    lambda_optimizer = optax.adam(config['lr'])

    opt_state = {
        'actor': actor_optimizer.init(params['actor']),
        'q1': critic_optimizer.init(params['q1']),
        'q2': critic_optimizer.init(params['q2']),
        'phi': phi_optimizer.init(params['phi']),
        'log_alpha': alpha_optimizer.init(params['log_alpha']),
        'log_lambda': lambda_optimizer.init(params['log_lambda']),
    }

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
        "z": jnp.zeros((config['z_dim'],), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=bool),
    }
    buffer_state = buffer.init(dummy_transition)

    # Initialize Environment & Agent States
    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    vmap_reset = jax.vmap(env.reset_fn)
    obs0, state0 = vmap_reset(reset_keys0)

    key, z_key = jax.random.split(key)
    z0 = sample_z(z_key, num_envs, config['z_dim'], config['unit_z'])

    ep_returns = jnp.zeros(num_envs, dtype=jnp.float32)
    global_step = jnp.array(0, dtype=jnp.int32)

    # Initial Carry
    carry = (state0, obs0, z0, ep_returns, buffer_state, params, target_params, opt_state, global_step)

    # -----------------------------------------------------
    # JITTED Core Execution Step
    # -----------------------------------------------------

    def polyak_update(online, target, tau):
        return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

    @functools.partial(jax.jit, donate_argnums=0)
    def run_chunk(carry, step_keys):
        """Scans over n steps fully on accelerator."""

        def scan_step(c, rng):
            (env_states, obs, z, ep_returns, buffer_state,
             params, target_params, opt_state, step) = c

            rng, action_rng = jax.random.split(rng)
            action_keys = jax.random.split(action_rng, num_envs)
            actions = batch_metra_action(action_keys, obs, z, params['actor'], actor_net)

            rng, step_rng_env = jax.random.split(rng)
            step_env_keys = jax.random.split(step_rng_env, num_envs)
            batch_step_fn = jax.vmap(env.step_fn, in_axes=(0, 0, 0))
            next_obs, next_env_states, rewards, next_dones, infos = batch_step_fn(
                step_env_keys, env_states, actions
            )

            # Track true environment return for logging only -- METRA is
            # unsupervised and never trains on `rewards`.
            next_ep_returns_running = ep_returns + rewards
            completed_return = jnp.where(next_dones, next_ep_returns_running, 0.0)
            next_ep_returns = jnp.where(next_dones, 0.0, next_ep_returns_running)

            transitions = {"obs": obs, "action": actions, "z": z, "done": next_dones}
            buffer_state = buffer.add(buffer_state, transitions)

            # Resample z only for envs whose episode just ended.
            rng, z_rng = jax.random.split(rng)
            new_z_sample = sample_z(z_rng, num_envs, config['z_dim'], config['unit_z'])
            next_z = jnp.where(next_dones[:, None], new_z_sample, z)

            # --- Optimization step (skipped via lax.cond until buffer is warm) ---
            can_train = buffer.can_sample(buffer_state)

            def do_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                rng, sample_key = jax.random.split(rng)
                batch = buffer.sample(buffer_state, sample_key).experience
                obs_b = batch.first["obs"]
                next_obs_b = batch.second["obs"]
                action_b = batch.first["action"]
                z_b = batch.first["z"]
                nonterminal = 1.0 - batch.first["done"].astype(jnp.float32)
                valid_denom = jnp.maximum(jnp.sum(nonterminal), 1.0)
                batch_idx = jnp.arange(obs_b.shape[0])

                def phi_loss_fn(phi_params):
                    comp = metra_components(phi_params, phi_net, obs_b, next_obs_b, z_b, nonterminal, config)
                    lambda_ = jnp.exp(params['log_lambda'])
                    objective = 10.0 * comp['r'] + jax.lax.stop_gradient(lambda_) * comp['cst_penalty']
                    loss = -jnp.sum(objective) / valid_denom
                    aux = {
                        "metra_reward": jnp.sum(comp['r']) / valid_denom,
                        "cst_penalty": jnp.sum(comp['cst_penalty']) / valid_denom,
                        "phi_sq_dist": jnp.sum(comp['sq_dist']) / valid_denom,
                    }
                    return loss, aux

                (phi_loss, phi_aux), phi_grad = jax.value_and_grad(phi_loss_fn, has_aux=True)(params['phi'])
                phi_updates, new_phi_opt = phi_optimizer.update(phi_grad, opt_state['phi'], params['phi'])
                new_phi_params = optax.apply_updates(params['phi'], phi_updates)

                def lambda_loss_fn(log_lambda):
                    comp = metra_components(new_phi_params, phi_net, obs_b, next_obs_b, z_b, nonterminal, config)
                    mean_cst = jax.lax.stop_gradient(jnp.sum(comp['cst_penalty'])) / valid_denom
                    return log_lambda * mean_cst

                lambda_loss, lambda_grad = jax.value_and_grad(lambda_loss_fn)(params['log_lambda'])
                lambda_updates, new_lambda_opt = lambda_optimizer.update(
                    lambda_grad, opt_state['log_lambda'], params['log_lambda']
                )
                new_log_lambda = optax.apply_updates(params['log_lambda'], lambda_updates)

                def critic_loss_fn(q1_params, q2_params):
                    alpha_sg = jax.lax.stop_gradient(jnp.exp(params['log_alpha']))
                    q1_values = q1_net.apply(q1_params, obs_b, z_b)
                    q2_values = q2_net.apply(q2_params, obs_b, z_b)
                    q1_sel = q1_values[batch_idx, action_b]
                    q2_sel = q2_values[batch_idx, action_b]

                    next_logits = actor_net.apply(params['actor'], next_obs_b, z_b)
                    next_probs = jax.nn.softmax(next_logits, axis=-1)
                    next_log_probs = jax.nn.log_softmax(next_logits, axis=-1)

                    target_q1 = q1_net.apply(target_params['q1'], next_obs_b, z_b)
                    target_q2 = q2_net.apply(target_params['q2'], next_obs_b, z_b)
                    target_q = jnp.minimum(target_q1, target_q2)
                    next_v = jnp.sum(next_probs * (target_q - alpha_sg * next_log_probs), axis=-1)

                    comp = metra_components(new_phi_params, phi_net, obs_b, next_obs_b, z_b, nonterminal, config)
                    intrinsic_r = jax.lax.stop_gradient(comp['r'])
                    target = jax.lax.stop_gradient(intrinsic_r + config['gamma'] * nonterminal * next_v)

                    q1_loss = 0.5 * jnp.mean((q1_sel - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_sel - target) ** 2)
                    aux = {"q1_mean": jnp.mean(q1_sel), "q2_mean": jnp.mean(q2_sel), "target_q_mean": jnp.mean(target)}
                    return q1_loss + q2_loss, aux

                (critic_loss, critic_aux), (q1_grad, q2_grad) = jax.value_and_grad(
                    critic_loss_fn, argnums=(0, 1), has_aux=True
                )(params['q1'], params['q2'])
                q1_updates, new_q1_opt = critic_optimizer.update(q1_grad, opt_state['q1'], params['q1'])
                new_q1_params = optax.apply_updates(params['q1'], q1_updates)
                q2_updates, new_q2_opt = critic_optimizer.update(q2_grad, opt_state['q2'], params['q2'])
                new_q2_params = optax.apply_updates(params['q2'], q2_updates)

                def actor_loss_fn(actor_params):
                    logits = actor_net.apply(actor_params, obs_b, z_b)
                    probs = jax.nn.softmax(logits, axis=-1)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    entropy = jnp.mean(-jnp.sum(probs * log_probs, axis=-1))

                    q_values = jax.lax.stop_gradient(jnp.minimum(
                        q1_net.apply(new_q1_params, obs_b, z_b), q2_net.apply(new_q2_params, obs_b, z_b)
                    ))
                    expected_q = jnp.mean(jnp.sum(probs * q_values, axis=-1))
                    loss = jnp.mean(jnp.sum(
                        probs * (jnp.exp(params['log_alpha']) * log_probs - q_values), axis=-1
                    ))
                    aux = {"entropy": entropy, "expected_q": expected_q}
                    return loss, aux

                (actor_loss, actor_aux), actor_grad = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(params['actor'])
                actor_updates, new_actor_opt = actor_optimizer.update(
                    actor_grad, opt_state['actor'], params['actor']
                )
                new_actor_params = optax.apply_updates(params['actor'], actor_updates)

                def alpha_loss_fn(log_alpha):
                    logits = actor_net.apply(new_actor_params, obs_b, z_b)
                    probs = jax.nn.softmax(logits, axis=-1)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    entropy = -jnp.sum(probs * log_probs, axis=-1)
                    return jnp.mean(jnp.exp(log_alpha) * jax.lax.stop_gradient(entropy - config['target_entropy']))

                alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(params['log_alpha'])
                alpha_updates, new_alpha_opt = alpha_optimizer.update(
                    alpha_grad, opt_state['log_alpha'], params['log_alpha']
                )
                new_log_alpha = optax.apply_updates(params['log_alpha'], alpha_updates)

                new_params = {
                    'actor': new_actor_params, 'q1': new_q1_params, 'q2': new_q2_params,
                    'phi': new_phi_params, 'log_alpha': new_log_alpha, 'log_lambda': new_log_lambda,
                }
                new_opt_state = {
                    'actor': new_actor_opt, 'q1': new_q1_opt, 'q2': new_q2_opt,
                    'phi': new_phi_opt, 'log_alpha': new_alpha_opt, 'log_lambda': new_lambda_opt,
                }
                new_target_params = {
                    'q1': polyak_update(new_q1_params, target_params['q1'], config['target_tau']),
                    'q2': polyak_update(new_q2_params, target_params['q2'], config['target_tau']),
                }
                total_loss = phi_loss + lambda_loss + critic_loss + actor_loss + alpha_loss
                aux = {
                    "phi_loss": phi_loss, "lambda_loss": lambda_loss, "critic_loss": critic_loss,
                    "actor_loss": actor_loss, "alpha_loss": alpha_loss,
                    "metra_reward": phi_aux["metra_reward"], "cst_penalty": phi_aux["cst_penalty"],
                    "phi_sq_dist": phi_aux["phi_sq_dist"],
                    "entropy": actor_aux["entropy"], "expected_q": actor_aux["expected_q"],
                    "alpha": jnp.exp(new_log_alpha), "lambda": jnp.exp(new_log_lambda),
                    "q1_mean": critic_aux["q1_mean"], "q2_mean": critic_aux["q2_mean"],
                }
                return new_params, new_target_params, new_opt_state, total_loss, aux

            def skip_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                zero = jnp.array(0.0)
                aux = {
                    "phi_loss": zero, "lambda_loss": zero, "critic_loss": zero,
                    "actor_loss": zero, "alpha_loss": zero, "metra_reward": zero,
                    "cst_penalty": zero, "phi_sq_dist": zero, "entropy": zero, "expected_q": zero,
                    "alpha": jnp.exp(params['log_alpha']), "lambda": jnp.exp(params['log_lambda']),
                    "q1_mean": zero, "q2_mean": zero,
                }
                return params, target_params, opt_state, zero, aux

            rng, train_rng = jax.random.split(rng)
            train_operands = (train_rng, buffer_state, params, target_params, opt_state)
            params, target_params, opt_state, total_loss, aux = jax.lax.cond(
                can_train, do_train, skip_train, train_operands
            )

            next_step = step + num_envs

            metrics = {
                "loss": total_loss, "done": next_dones, "completed_return": completed_return, **aux,
            }
            next_c = (
                next_env_states, next_obs, next_z, next_ep_returns, buffer_state,
                params, target_params, opt_state, next_step,
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
            "train/phi_loss": float(np.mean(metrics["phi_loss"])),
            "train/lambda_loss": float(np.mean(metrics["lambda_loss"])),
            "train/critic_loss": float(np.mean(metrics["critic_loss"])),
            "train/actor_loss": float(np.mean(metrics["actor_loss"])),
            "train/alpha_loss": float(np.mean(metrics["alpha_loss"])),
            "train/metra_reward": float(np.mean(metrics["metra_reward"])),
            "train/cst_penalty": float(np.mean(metrics["cst_penalty"])),
            "train/phi_sq_dist": float(np.mean(metrics["phi_sq_dist"])),
            "train/entropy": float(np.mean(metrics["entropy"])),
            "train/expected_q": float(np.mean(metrics["expected_q"])),
            "train/alpha": float(metrics["alpha"][-1]),
            "train/lambda": float(metrics["lambda"][-1]),
            "train/env_return_mean": true_mean_return,
        }
        logger.log_metrics(chunk_metrics, step=env_steps0)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} env-steps) "
              f"| Loss: {chunk_metrics['train/loss']:.4f} | METRA Reward: {chunk_metrics['train/metra_reward']:.4f} "
              f"| Env Return: {true_mean_return:.2f}")

        return carry

    # -----------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------
    print("Starting METRA parallel rollouts...")

    total_steps = config['num_steps']
    chunk_size = config['chunk_size']

    eval_config = config_raw.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = int(total_steps * eval_config.get("interval_pct", 0.05))
    eval_max_steps = eval_config.get("max_steps", 2000)
    next_eval_step = eval_interval

    for step_idx in range(0, total_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)

        if do_eval and step_idx >= next_eval_step:
            print(f"\n--- Running Evaluation at Step {step_idx} ---")

            frozen_params = jax.device_get(carry[5])
            current_env_step = (step_idx + chunk_size) * num_envs

            logger.save_checkpoint(frozen_params, current_env_step)

            # A skill is fixed for the whole evaluation episode (that's the
            # point of a skill-conditioned policy) -- sample it once here and
            # close over it, since run_eval_episode's policy_fn is stateless
            # across steps.
            key, z_eval_key, eval_key = jax.random.split(key, 3)
            eval_z = sample_z(z_eval_key, 1, config['z_dim'], config['unit_z'])[0]

            def greedy_eval_policy(eval_params, single_obs, step_key):
                act = metra_action(step_key, single_obs, eval_z, eval_params['actor'], actor_net, greedy=True)
                return act, eval_z

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
