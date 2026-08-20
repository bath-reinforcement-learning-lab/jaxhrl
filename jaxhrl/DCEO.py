# Deep Covering Eigen Options in JAX
# From "Deep Laplacian-based Options for Temporally-Extended Exploration"
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flashbax as fbx
from typing import Dict, Any
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from jaxhrl.common.utils import parse_config
from haxhrl.common.logger import Logger
from jaxhrl.common.wrappers import make_jax_env

# Flax Neural Networks

class SymbolicEncoder(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x

class LaplacianRepresentationNetwork(nn.Module):
    latent_dim: int = 256
    rep_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = SymbolicEncoder(hidden_dim=self.latent_dim)(x)
        phi = nn.Dense(features=self.rep_dim)(x)
        phi = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
        return phi

class OptionQNetwork(nn.Module):
    num_actions: int
    num_options: int
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x, option_idx):
        x = SymbolicEncoder(hidden_dim=self.latent_dim)(x)
        # option_idx == -1 (the "task" context) yields an all-zero one-hot,
        # giving the network a distinct context for the extrinsic-reward policy.
        option_one_hot = jax.nn.one_hot(option_idx, self.num_options)
        x = jnp.concatenate([x, option_one_hot], axis=-1)
        x = nn.Dense(features=self.latent_dim)(x)
        x = nn.relu(x)
        q_values = nn.Dense(features=self.num_actions)(x)
        return q_values

# Helper Functions for Action Selection

def sample_option_policy(q_params, q_net, obs, option_idx, key, num_actions):
    """Greedy action under the given option's Q-function.

    `key` is accepted (and currently unused) so callers can later swap in an
    intra-option epsilon-greedy policy without changing the call signature.
    """
    q_values = q_net.apply(q_params, obs[None, :], jnp.asarray(option_idx)[None])[0]
    return jnp.argmax(q_values)

def get_greedy_action(q_params, q_net, obs, num_actions):
    """Greedy action under the task policy (option index -1)."""
    q_values = q_net.apply(q_params, obs[None, :], jnp.array([-1], dtype=jnp.int32))[0]
    return jnp.argmax(q_values)

def select_dceo_action(key, obs, done, tau, option, q_params, q_net, config, current_mu):
    D = config['D']
    epsilon = config['epsilon']
    num_options = config['num_options']
    num_actions = config['num_actions']

    tau = jnp.where(done, True, tau)
    option = jnp.where(done, -1, option)

    key, subkey = jax.random.split(key)
    tau = tau | (jax.random.uniform(subkey) < (1.0 / D))

    key, eps_key, mu_key, opt_key, act_opt_key, act_uni_key, act_curr_key = jax.random.split(key, 7)

    is_eps = jax.random.uniform(eps_key) < epsilon
    is_mu = jax.random.uniform(mu_key) < current_mu

    sampled_option = jax.random.randint(opt_key, (), 0, num_options)
    action_sampled_opt = sample_option_policy(q_params, q_net, obs, sampled_option, act_opt_key, num_actions)
    action_uniform = jax.random.randint(act_uni_key, (), 0, num_actions)
    action_greedy = get_greedy_action(q_params, q_net, obs, num_actions)
    action_curr_opt = sample_option_policy(q_params, q_net, obs, option, act_curr_key, num_actions)

    opt_if_eps = jnp.where(is_mu, sampled_option, -1)
    tau_if_eps = jnp.where(is_mu, False, True)
    act_if_eps = jnp.where(is_mu, action_sampled_opt, action_uniform)

    opt_if_tau = jnp.where(is_eps, opt_if_eps, option)
    tau_if_tau = jnp.where(is_eps, tau_if_eps, tau)
    act_if_tau = jnp.where(is_eps, act_if_eps, action_greedy)

    final_option = jnp.where(tau, opt_if_tau, option)
    final_tau = jnp.where(tau, tau_if_tau, tau)
    final_act = jnp.where(tau, act_if_tau, action_curr_opt)

    return final_act, final_option, final_tau, key

def batch_select_dceo_action(keys, obs, dones, taus, options, q_params, q_net, config, mu):
    batched_step_fn = jax.vmap(
        select_dceo_action, in_axes=(0, 0, 0, 0, 0, None, None, None, None)
    )
    return batched_step_fn(keys, obs, dones, taus, options, q_params, q_net, config, mu)


# Losses

def laplacian_loss_fn(laplacian_params, laplacian_net, obs_a, obs_b, obs_i, obs_j, beta):
    phi_a = laplacian_net.apply(laplacian_params, obs_a)
    phi_b = laplacian_net.apply(laplacian_params, obs_b)
    attractive = jnp.mean(jnp.sum((phi_a - phi_b) ** 2, axis=-1))

    phi_i = laplacian_net.apply(laplacian_params, obs_i)
    phi_j = laplacian_net.apply(laplacian_params, obs_j)
    batch_size = phi_i.shape[0]
    gram = (phi_i.T @ phi_j) / batch_size  # [rep_dim, rep_dim]
    rep_dim = gram.shape[0]
    identity = jnp.eye(rep_dim)
    # Lower-triangular (incl. diagonal) mask
    # each pair (i, j) with i >= j is penalized once.
    mask = jnp.tril(jnp.ones((rep_dim, rep_dim)))
    orthogonality = jnp.sum(((gram - identity) ** 2) * mask) / jnp.sum(mask)

    loss = attractive + beta * orthogonality
    return loss, {"lap_attractive": attractive, "lap_orthogonality": orthogonality}

def q_loss_fn(q_params, q_net, target_q_params, target_laplacian_params, laplacian_net, batch, config):
    obs = batch.first["obs"]
    next_obs = batch.second["obs"]
    actions = batch.first["action"]
    dones = batch.first["done"].astype(jnp.float32)
    options = batch.first["option"]
    ext_rewards = batch.first["reward"]

    is_task = options < 0
    eig_idx = jnp.where(options >= 0, options // 2, 0)
    sign = jnp.where(options % 2 == 0, 1.0, -1.0)

    phi = laplacian_net.apply(target_laplacian_params, obs)
    phi_next = laplacian_net.apply(target_laplacian_params, next_obs)
    phi_k = jnp.take_along_axis(phi, eig_idx[:, None], axis=-1)[:, 0]
    phi_k_next = jnp.take_along_axis(phi_next, eig_idx[:, None], axis=-1)[:, 0]
    intrinsic_reward = sign * (phi_k_next - phi_k)

    reward = jnp.where(is_task, ext_rewards, intrinsic_reward)
    reward = jax.lax.stop_gradient(reward)

    q_values = q_net.apply(q_params, obs, options)
    q_sa = jnp.take_along_axis(q_values, actions[:, None], axis=-1)[:, 0]

    q_next_values = q_net.apply(target_q_params, next_obs, options)
    q_next_max = jax.lax.stop_gradient(jnp.max(q_next_values, axis=-1))

    target = reward + config['gamma'] * (1.0 - dones) * q_next_max
    loss = jnp.mean((q_sa - target) ** 2)
    return loss, {"q_loss": loss, "mean_reward": jnp.mean(reward)}


# 4. Main Execution & JIT Training Logic


if __name__ == "__main__":
    config_raw = parse_config()
    logger = Logger(config_raw)

    framework_type = config_raw["env"].get("framework", "gymnax")
    env_id = config_raw["env"]["make"]["id"].split("/")[-1]
    env = make_jax_env(framework_type, env_id, 1)

    # Map parameters
    config = {
        'num_steps': config_raw["training"].get("n_steps", 20_000),
        'epsilon': config_raw["training"].get("epsilon", 0.1),
        'num_envs': config_raw["training"].get("num_envs", 1024),
        'D': config_raw["training"].get("D", 100),
        'num_options': config_raw["training"].get("num_options", 10),
        'num_actions': env.num_actions,
        'mu': config_raw["training"].get("mu"),
        'buffer_size': config_raw["training"].get("buffer_size"),
        'batch_size': config_raw["training"].get("batch_size", 256),
        'chunk_size': config_raw["training"].get("chunk_size", 100),  # Steps per host sync
        'gamma': config_raw["training"].get("gamma", 0.99),
        'lr_laplacian': config_raw["training"].get("lr_laplacian", 1e-4),
        'lr_q': config_raw["training"].get("lr_q", 1e-4),
        'beta': config_raw["training"].get("beta", 1.0),  # orthogonality weight
        'target_tau': config_raw["training"].get("target_tau", 0.005),  # Polyak coefficient
    }

    if config['num_options'] % 2 != 0:
        raise ValueError(
            f"num_options must be even (each eigenvector contributes a +/- pair of "
            f"options), got {config['num_options']}"
        )
    config['num_eigenvectors'] = config['num_options'] // 2

    config['obs_dim'] = env.state_dim
    env_params = getattr(env, 'default_params', None)
    num_envs = config['num_envs']

    seed = config_raw["seed"]
    key = jax.random.PRNGKey(seed)

    # Initialize Networks
    dummy_obs = jnp.zeros((1, config['obs_dim']), dtype=jnp.float32)
    dummy_option_idx = jnp.zeros((1,), dtype=jnp.int32)

    laplacian_net = LaplacianRepresentationNetwork(latent_dim=256, rep_dim=config['num_eigenvectors'])
    q_net = OptionQNetwork(num_actions=config['num_actions'], num_options=config['num_options'], latent_dim=256)

    init_rng_laplacian, init_rng_q = jax.random.split(key, 2)
    params = {
        'laplacian': laplacian_net.init(init_rng_laplacian, dummy_obs),
        'q_network': q_net.init(init_rng_q, dummy_obs, dummy_option_idx)
    }
    target_params = jax.tree.map(jnp.copy, params)  # deep copy of the pytree

    # Optimizers
    laplacian_optimizer = optax.adam(config['lr_laplacian'])
    q_optimizer = optax.adam(config['lr_q'])
    opt_state = {
        'laplacian': laplacian_optimizer.init(params['laplacian']),
        'q_network': q_optimizer.init(params['q_network']),
    }

    # Initialize Flashbax Buffer
    buffer = fbx.make_flat_buffer(
    max_length=config['buffer_size'],
    min_length=config['batch_size'],
    sample_batch_size=config['batch_size'],
    add_batch_size=config['num_envs']  # Tells Flashbax to expect a batch of n parallel steps
)

    dummy_transition = {
    "obs": jnp.zeros((config['obs_dim'],), dtype=jnp.float32),
    "action": jnp.zeros((), dtype=jnp.int32),
    "reward": jnp.zeros((), dtype=jnp.float32),
    "done": jnp.zeros((), dtype=bool),
    "option": jnp.full((), -1, dtype=jnp.int32),
}
    buffer_state = buffer.init(dummy_transition)

    # Initialize Environment & Agent States
    key, env_rng = jax.random.split(key)
    env_keys = jax.random.split(env_rng, num_envs)

    vmap_reset = jax.vmap(env.reset_fn)
    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    obs0, state0 = vmap_reset(reset_keys0)

    dones = jnp.zeros(num_envs, dtype=bool)
    taus = jnp.ones(num_envs, dtype=bool)
    options = jnp.full(num_envs, -1, dtype=jnp.int32)
    ep_returns = jnp.zeros(num_envs, dtype=jnp.float32)
    global_step = jnp.array(0, dtype=jnp.int32)

    # Initial Carry
    carry = (
        state0, obs0, dones, taus, options, ep_returns,
        buffer_state, params, target_params, opt_state, global_step,
    )

    # -----------------------------------------------------
    # JITTED Core Execution Step
    # -----------------------------------------------------

    def polyak_update(online, target, tau):
        return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

    @jax.jit(donate_argnums=0)
    def run_chunk(carry, step_keys):
        """Scans over n steps fully on accelerator."""

        def scan_step(c, rng):
            (env_states, obs, dones, taus, options, ep_returns,
             buffer_state, params, target_params, opt_state, step) = c

            # dynamic mu
            mu = config['mu']

            rng, step_rng = jax.random.split(rng)
            env_action_keys = jax.random.split(step_rng, num_envs)

            # Action Selection
            actions, next_options, next_taus, _ = batch_select_dceo_action(
                env_action_keys, obs, dones, taus, options,
                params['q_network'], q_net, config, mu
            )

            # Env Step
            batch_step_fn = jax.vmap(env.step_fn, in_axes=(0, 0, 0))
            rng, step_rng_env = jax.random.split(rng)
            step_env_keys = jax.random.split(step_rng_env, num_envs)

            next_obs, next_env_states, rewards, next_dones, infos = batch_step_fn(
                step_env_keys, env_states, actions
            )

            # Return Tracking
            ep_returns = ep_returns + rewards
            completed_return = jnp.where(next_dones, ep_returns, 0.0)
            next_ep_returns = jnp.where(next_dones, 0.0, ep_returns)

            # Store in Buffer (store the option that was ACTIVE for this step)
            transitions = {
                "obs": obs,
                "action": actions,
                "reward": rewards,
                "done": next_dones,
                "option": options,
            }
            buffer_state = buffer.add(buffer_state, transitions)

            # --- Optimization step (skipped via lax.cond until buffer is warm) ---
            can_train = buffer.can_sample(buffer_state)

            def do_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                rng, k_batch, k_indep_a, k_indep_b = jax.random.split(rng, 4)
                batch = buffer.sample(buffer_state, k_batch).experience
                batch_indep_a = buffer.sample(buffer_state, k_indep_a).experience
                batch_indep_b = buffer.sample(buffer_state, k_indep_b).experience

                def lap_loss(p):
                    return laplacian_loss_fn(
                        p, laplacian_net,
                        batch.first['obs'], batch.second['obs'],
                        batch_indep_a.first['obs'], batch_indep_b.first['obs'],
                        config['beta'],
                    )

                (loss_laplacian, lap_aux), lap_grads = jax.value_and_grad(lap_loss, has_aux=True)(
                    params['laplacian']
                )
                lap_updates, new_lap_opt_state = laplacian_optimizer.update(
                    lap_grads, opt_state['laplacian'], params['laplacian']
                )
                new_lap_params = optax.apply_updates(params['laplacian'], lap_updates)

                def qloss(p):
                    return q_loss_fn(
                        p, q_net, target_params['q_network'], target_params['laplacian'],
                        laplacian_net, batch, config,
                    )

                (loss_q, q_aux), q_grads = jax.value_and_grad(qloss, has_aux=True)(
                    params['q_network']
                )
                q_updates, new_q_opt_state = q_optimizer.update(
                    q_grads, opt_state['q_network'], params['q_network']
                )
                new_q_params = optax.apply_updates(params['q_network'], q_updates)

                new_params = {'laplacian': new_lap_params, 'q_network': new_q_params}
                new_opt_state = {'laplacian': new_lap_opt_state, 'q_network': new_q_opt_state}
                new_target_params = {
                    'laplacian': polyak_update(new_lap_params, target_params['laplacian'], config['target_tau']),
                    'q_network': polyak_update(new_q_params, target_params['q_network'], config['target_tau']),
                }
                return new_params, new_target_params, new_opt_state, loss_laplacian, loss_q

            def skip_train(operands):
                rng, buffer_state, params, target_params, opt_state = operands
                return params, target_params, opt_state, jnp.array(0.0), jnp.array(0.0)

            rng, train_rng = jax.random.split(rng)
            train_operands = (train_rng, buffer_state, params, target_params, opt_state)
            params, target_params, opt_state, loss_laplacian, loss_q = jax.lax.cond(
                can_train, do_train, skip_train, train_operands
            )

            next_step = step + num_envs

            # Compile metrics for this step
            metrics = {
                "loss1": loss_laplacian,
                "loss2": loss_q,
                "done": next_dones,
                "completed_return": completed_return
            }

            metrics = {
                            "loss1": loss_laplacian,
                            "loss2": loss_q,
                            "done": next_dones,
                            "completed_return": completed_return,
                            "options": options,  # Shape will become (chunk_size, num_envs)
                            "infos": infos       # PyTree of shape (chunk_size, num_envs)
                        }
            next_c = (
                next_env_states, next_obs, next_dones, next_taus, next_options, next_ep_returns,
                buffer_state, params, target_params, opt_state, next_step,
            )
            return next_c, metrics
        return jax.lax.scan(scan_step, carry, step_keys)

    # Logging 

    # Logging 
    def run_and_log(carry, rng_key, n, step0):
        import numpy as np
        keys = jax.random.split(rng_key, n)
        carry, metrics = run_chunk(carry, keys)
        metrics = jax.device_get(metrics)

        loss_1 = float(np.mean(metrics["loss1"]))
        loss_2 = float(np.mean(metrics["loss2"]))

        total_dones = np.sum(metrics["done"])
        sum_returns = np.sum(metrics["completed_return"])
        true_mean_return = float(np.where(total_dones > 0, sum_returns / total_dones, 0.0))

        env_steps0 = step0 * num_envs
        
        # 1. Log High-Level Chunk Metrics together
        chunk_metrics = {
            "train/loss_laplacian": loss_1,
            "train/loss_q": loss_2,
            "train/return_mean": true_mean_return
        }
        logger.log_metrics(chunk_metrics, step=env_steps0)

        # 2. Log Granular Timestep Data
        num_options = config['num_options']
        
        for t in range(n):
            current_t_step = env_steps0 + (t * num_envs)
            t_metrics = {}
            
            opts_at_t = metrics["options"][t]
            t_metrics["options/task_policy_usage"] = float(np.mean(opts_at_t == -1))
            
            for opt_idx in range(num_options):
                t_metrics[f"options/usage_opt_{opt_idx}"] = float(np.mean(opts_at_t == opt_idx))

            for key, value in metrics["infos"].items():
                if isinstance(value, np.ndarray):
                    t_metrics[f"achievements/{key}"] = float(np.mean(value[t]))
            
            # Log the entire timestep dictionary at once, pinned to the exact step
            logger.log_metrics(t_metrics, step=current_t_step)

        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} env-steps) "
              f"| Laplacian Loss: {loss_1:.4f} | Q Loss: {loss_2:.4f} | True Return: {true_mean_return:.2f}")

        return carry

    # -----------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------
    print("Starting DCEO parallel rollouts...")

    total_steps = config['num_steps']
    chunk_size = config['chunk_size']
    
    # Evaluation Configuration Setup
    eval_config = config_raw.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = int(total_steps * eval_config.get("interval_pct", 0.05))
    eval_max_steps = eval_config.get("max_steps", 2000)
    next_eval_step = eval_interval

    # Create the greedy evaluation policy (no epsilon, no mu)
    @jax.jit
    def greedy_eval_policy(eval_params, single_obs, eval_key):
        # By passing epsilon=0.0 and current_mu=0.0 to the config dict inside the wrapper, 
        # it forces the agent to use its learned options deterministically.
        eval_conf = config.copy()
        eval_conf['epsilon'] = 0.0  
        
        act, opt, _, _ = select_dceo_action(
            key=eval_key, 
            obs=single_obs, 
            done=jnp.array(False), 
            tau=jnp.array(True), 
            option=jnp.array(-1), 
            q_params=eval_params['q_network'], 
            q_net=q_net, 
            config=eval_conf, 
            current_mu=0.0
        )
        return act, opt

    for step_idx in range(0, total_steps, chunk_size):
        key, chunk_key = jax.random.split(key)
        carry = run_and_log(carry, chunk_key, chunk_size, step_idx)
        
        # --- EVALUATION HOOK ---
        if do_eval and step_idx >= next_eval_step:
            print(f"\n--- Running Evaluation at Step {step_idx} ---")
            
            frozen_params = jax.device_get(carry[7])
            current_env_step = (step_idx + chunk_size) * num_envs  # Sync the eval timeline here!
            
            # Save Checkpoint
            logger.save_checkpoint(frozen_params, current_env_step)
            
            # Run Trajectory
            key, eval_key = jax.random.split(key)
            from brll_core.algorithms.common.jax_wrappers import run_eval_episode
            trajectory = run_eval_episode(
                env_wrapper=env, 
                policy_fn=greedy_eval_policy, 
                params=frozen_params, 
                key=eval_key, 
                max_steps=eval_max_steps
            )
            
            # Upload Trajectory
            logger.log_eval_trajectory(current_env_step, trajectory)
            
            next_eval_step += eval_interval
            print(f"--- Evaluation Complete. Total Reward: {sum(trajectory['reward']):.2f} ---\n")

    print("Training completed.")
    logger.close()
