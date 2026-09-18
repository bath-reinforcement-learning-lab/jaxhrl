# Proximal Policy Option-Critic (PPOC)
# Klissarov, Bacon, Harb & Precup, "Learnings Options End-to-End for
# Continuous Action Tasks" (NeurIPS 2017 Deep RL workshop, arXiv:1712.00004)
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flashbax as fbx
import functools
from typing import Dict, Any, NamedTuple
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from jaxhrl.common.utils import parse_config
from jaxhrl.common.logger import Logger
from jaxhrl.common.wrappers import make_jax_env


class RolloutCarry(NamedTuple):
    """The state threaded through run_rollout_chunk's scan and back out to
    the main training loop."""
    env_states: Any
    obs: jnp.ndarray
    dones: jnp.ndarray
    options: jnp.ndarray
    ep_returns: jnp.ndarray
    buffer_state: Any
    params: Dict[str, Any]
    opt_state: Dict[str, Any]
    global_step: jnp.ndarray


# Networks
class PolicyNetwork(nn.Module):
    """intra-option Gaussian policies and the
    softmax policy over options
    """
    num_options: int
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))

        mu_logits = nn.Dense(self.num_options)(x)             # policy-over-options logits
        action_mean = nn.Dense(self.num_options * self.action_dim)(x)
        action_mean = action_mean.reshape((x.shape[0], self.num_options, self.action_dim))

        log_std = self.param('log_std', nn.initializers.zeros, (self.num_options, self.action_dim))

        return mu_logits, action_mean, log_std


class ValueNetwork(nn.Module):
    num_options: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))

        q_omega = nn.Dense(self.num_options)(x)
        beta_logits = nn.Dense(self.num_options, bias_init=nn.initializers.constant(-2.0))(x)
        return q_omega, beta_logits


# Call-and-return action selection (Algorithm 1)


def select_ppoc_action(key, obs, option, done, policy_params, policy_net,
                        value_params, value_net, config):
    """One environment step. If the previously active option just terminated
    (or `done`), sample a new option o ~ mu(.|s) (softmax over options). Otherwise keep it. Then sample a continuous
    action from the (possibly-new) option's Gaussian intra-option policy;"""
    mu_logits, action_mean, log_std = policy_net.apply(policy_params, obs[None, :])
    mu_logits, action_mean = mu_logits[0], action_mean[0]
    q_omega, beta_logits = value_net.apply(value_params, obs[None, :])
    q_omega, beta_logits = q_omega[0], beta_logits[0]
    beta = jax.nn.sigmoid(beta_logits)

    option = jnp.where(done, -1, option)
    key, term_key, opt_key, act_key = jax.random.split(key, 4)

    current_beta = jnp.where(option >= 0, beta[jnp.maximum(option, 0)], 1.0)
    terminate = (option < 0) | jax.random.bernoulli(term_key, current_beta)

    sampled_option = jax.random.categorical(opt_key, mu_logits)
    new_option = jnp.where(terminate, sampled_option, option)
    decision_flag = terminate

    option_logp = jax.nn.log_softmax(mu_logits)[new_option]

    mean_o = action_mean[new_option]
    std_o = jnp.exp(log_std[new_option])
    raw_action = mean_o + std_o * jax.random.normal(act_key, mean_o.shape)
    action_logp = jnp.sum(
        -0.5 * ((raw_action - mean_o) / std_o) ** 2 - log_std[new_option]
        - 0.5 * jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    action = jnp.clip(raw_action, config['action_low'], config['action_high'])
    q_value = q_omega[new_option]

    return (action, raw_action, new_option, action_logp, option_logp, q_value,
            decision_flag, key)


def batch_select_ppoc_action(keys, obs, options, dones, policy_params, policy_net,
                              value_params, value_net, config):
    batched_step_fn = jax.vmap(
        select_ppoc_action, in_axes=(0, 0, 0, 0, None, None, None, None, None)
    )
    return batched_step_fn(keys, obs, options, dones, policy_params, policy_net,
                            value_params, value_net, config)


def where_per_env(done_mask, a, b):
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


# Advantage estimation

def compute_option_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    def scan_fn(carry, x):
        next_value, next_adv = carry
        r_t, v_t, d_t = x
        not_done = 1.0 - d_t.astype(jnp.float32)
        delta = r_t + gamma * next_value * not_done - v_t
        adv = delta + gamma * gae_lambda * not_done * next_adv
        return (v_t, adv), adv

    (_, _), advantages_rev = jax.lax.scan(
        scan_fn, (bootstrap_value, jnp.zeros_like(bootstrap_value)),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


# PPO losses (Algorithm 1: PPOC)

def ppoc_policy_loss_fn(policy_params, policy_net, value_params, value_net,
                         obs, options, raw_actions, old_action_logp, adv,
                         decision_mask, clip_eps, entropy_coef):
    mu_logits, action_mean, log_std = policy_net.apply(policy_params, obs)
    batch_idx = jnp.arange(obs.shape[0])

    # --- intra-option PPO-clip actor ---
    mean_o = action_mean[batch_idx, options]
    log_std_o = log_std[options]
    std_o = jnp.exp(log_std_o)
    new_action_logp = jnp.sum(
        -0.5 * ((raw_actions - mean_o) / std_o) ** 2 - log_std_o - 0.5 * jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    norm_adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    ratio = jnp.exp(new_action_logp - old_action_logp)
    unclipped = ratio * norm_adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
    intra_option_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # Differential entropy of a diagonal Gaussian, closed form.
    gauss_entropy = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std_o, axis=-1))
    approx_kl = jnp.mean(old_action_logp - new_action_logp)


    q_omega, _ = value_net.apply(value_params, obs)
    v_omega = jnp.sum(jax.nn.softmax(mu_logits, axis=-1) * q_omega, axis=-1)
    option_advantage = jax.lax.stop_gradient(q_omega[batch_idx, options] - v_omega)

    log_mu_all = jax.nn.log_softmax(mu_logits, axis=-1)
    new_option_logp = jnp.take_along_axis(log_mu_all, options[:, None], axis=-1)[:, 0]

    mask_f = decision_mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask_f), 1.0)
    oadv_mean = jnp.sum(option_advantage * mask_f) / denom
    oadv_var = jnp.sum(mask_f * (option_advantage - oadv_mean) ** 2) / denom
    norm_oadv = (option_advantage - oadv_mean) / (jnp.sqrt(oadv_var) + 1e-8)
    policy_over_options_loss = -jnp.sum(new_option_logp * norm_oadv * mask_f) / denom

    mu_probs = jnp.exp(log_mu_all)
    mu_entropy_all = -jnp.sum(mu_probs * log_mu_all, axis=-1)
    mu_entropy = jnp.sum(mu_entropy_all * mask_f) / denom

    total_loss = (intra_option_loss + policy_over_options_loss
                  - entropy_coef * gauss_entropy - entropy_coef * mu_entropy)

    aux = {
        "intra_option_loss": intra_option_loss,
        "policy_over_options_loss": policy_over_options_loss,
        "gauss_entropy": gauss_entropy,
        "mu_entropy": mu_entropy,
        "approx_kl": approx_kl,
    }
    return total_loss, aux


def ppoc_value_loss_fn(value_params, value_net, policy_params, policy_net,
                        obs, next_obs, options, nonterminal, returns, old_values,
                        clip_eps, value_coef, delib_cost):
    """combined loss: value-clipped critic MSE plus the termination gradient"""
    q_omega, _ = value_net.apply(value_params, obs)
    batch_idx = jnp.arange(obs.shape[0])
    q_selected = q_omega[batch_idx, options]

    value_clipped = old_values + jnp.clip(q_selected - old_values, -clip_eps, clip_eps)
    value_loss_unclipped = (q_selected - returns) ** 2
    value_loss_clipped = (value_clipped - returns) ** 2
    critic_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))

    q_next, beta_logits_next = value_net.apply(value_params, next_obs)
    mu_logits_next, _, _ = policy_net.apply(policy_params, next_obs)
    mu_next = jax.nn.softmax(mu_logits_next, axis=-1)
    v_omega_next = jnp.sum(mu_next * q_next, axis=-1)

    beta_next = jax.nn.sigmoid(beta_logits_next)
    beta_next_o = beta_next[batch_idx, options]
    q_next_o = q_next[batch_idx, options]

    termination_advantage = jax.lax.stop_gradient(q_next_o - v_omega_next + delib_cost)
    termination_loss = jnp.mean(nonterminal * beta_next_o * termination_advantage)

    total_loss = value_coef * critic_loss + termination_loss
    aux = {
        "critic_loss": critic_loss,
        "termination_loss": termination_loss,
        "beta_mean": jnp.mean(beta_next_o),
    }
    return total_loss, aux



# Main Execution & JIT Training Logic


if __name__ == "__main__":
    config_raw = parse_config()
    logger = Logger(config_raw)

    framework_type = config_raw["env"].get("framework", "brax")
    env_id = config_raw["env"]["make"]["id"].split("/")[-1]
    env_kwargs = config_raw["env"].get("kwargs", {}) or {}
    env = make_jax_env(framework_type, env_id, 1, **env_kwargs)

    assert env.action_dim > 0, "PPOC is continuous-control only; use a brax environment."

    config = {
        'num_steps': config_raw["training"].get("num_steps", 2_000_000),
        'num_envs': config_raw["training"].get("num_envs", 256),
        'num_options': config_raw["training"].get("num_options", 2),
        'rollout_horizon': config_raw["training"].get("rollout_horizon", 128),
        'gamma': config_raw["training"].get("gamma", 0.99),
        'gae_lambda': config_raw["training"].get("gae_lambda", 0.95),
        'clip_eps': config_raw["training"].get("clip_eps", 0.2),
        'entropy_coef': config_raw["training"].get("entropy_coef", 0.01),
        'value_coef': config_raw["training"].get("value_coef", 0.5),
        'max_grad_norm': config_raw["training"].get("max_grad_norm", 0.5),
        'ppo_epochs': config_raw["training"].get("ppo_epochs", 4),
        'num_minibatches': config_raw["training"].get("num_minibatches", 8),
        'lr_policy': config_raw["training"].get("lr_policy", 3e-4),
        'lr_value': config_raw["training"].get("lr_value", 3e-4),
        'delib_cost': config_raw["training"].get("delib_cost", 0.0),
        'hidden_dim': config_raw["training"].get("hidden_dim", 64),
        # Paper Section 4: "In the case of options, we also divide the reward
        # by 10 to reduce the scale of the value functions, and therefore the
        # termination probability gradient, making it more stable." Applied
        # only to options runs (num_options > 1), not the primitive-action
        # baseline -- left at 1.0 (no-op) by default so a plain flat-PPO run
        # (num_options=1) is unaffected unless explicitly configured.
        'reward_scale': config_raw["training"].get("reward_scale", 1.0),
    }

    config['obs_dim'] = env.state_dim
    config['action_dim'] = env.action_dim
    config['action_low'] = env.action_low if env.action_low is not None else -jnp.ones((env.action_dim,))
    config['action_high'] = env.action_high if env.action_high is not None else jnp.ones((env.action_dim,))

    num_envs = config['num_envs']
    rollout_horizon = config['rollout_horizon']
    num_options = config['num_options']
    obs_dim = config['obs_dim']
    action_dim = config['action_dim']

    seed = config_raw["seed"]
    key = jax.random.PRNGKey(seed)

    # Initialize networks
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    policy_net = PolicyNetwork(num_options=num_options, action_dim=action_dim, hidden_dim=config['hidden_dim'])
    value_net = ValueNetwork(num_options=num_options, hidden_dim=config['hidden_dim'])

    key, init_key_policy, init_key_value = jax.random.split(key, 3)
    params = {
        'policy': policy_net.init(init_key_policy, dummy_obs),
        'value': value_net.init(init_key_value, dummy_obs),
    }

    # Optimizers (global-norm clipping, fully separate networks/optimizers)
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adam(config['lr_policy']),
    )
    value_optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adam(config['lr_value']),
    )
    opt_state = {
        'policy': policy_optimizer.init(params['policy']),
        'value': value_optimizer.init(params['value']),
    }

    buffer = fbx.make_flat_buffer(
        max_length=(rollout_horizon + 1) * num_envs,
        min_length=rollout_horizon * num_envs,
        sample_batch_size=num_envs,  # unused for random sampling; required by the API
        add_batch_size=num_envs,
    )

    dummy_transition = {
        "obs": jnp.zeros((obs_dim,), dtype=jnp.float32),
        "next_obs": jnp.zeros((obs_dim,), dtype=jnp.float32),
        "raw_action": jnp.zeros((action_dim,), dtype=jnp.float32),
        "option": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=bool),
        "decision_flag": jnp.zeros((), dtype=bool),
        "action_logp": jnp.zeros((), dtype=jnp.float32),
        "option_logp": jnp.zeros((), dtype=jnp.float32),
        "q_value": jnp.zeros((), dtype=jnp.float32),
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

    carry = RolloutCarry(
        env_states=state0, obs=obs0, dones=dones, options=options,
        ep_returns=ep_returns, buffer_state=buffer_state, params=params,
        opt_state=opt_state, global_step=global_step,
    )

    # JITTED Rollout Collection (Algorithm 1)

    @functools.partial(jax.jit, donate_argnums=0)
    def run_rollout_chunk(carry, step_keys):
        """Scans over `rollout_horizon` steps fully on accelerator; pure
        rollout collection, no gradient steps (PPO trains once per full
        chunk, on-policy)."""

        def scan_step(c, rng):
            rng, action_rng = jax.random.split(rng)
            action_keys = jax.random.split(action_rng, num_envs)

            (actions, raw_actions, next_options, action_logp, option_logp,
             q_value, decision_flags, _) = batch_select_ppoc_action(
                action_keys, c.obs, c.options, c.dones,
                c.params['policy'], policy_net, c.params['value'], value_net, config,
            )

            batch_step_fn = jax.vmap(env.step_fn, in_axes=(0, 0, 0))
            rng, step_rng_env = jax.random.split(rng)
            step_env_keys = jax.random.split(step_rng_env, num_envs)
            next_obs, next_env_states, rewards, next_dones, infos = batch_step_fn(
                step_env_keys, c.env_states, actions
            )

            next_ep_returns_running = c.ep_returns + rewards
            completed_return = jnp.where(next_dones, next_ep_returns_running, 0.0)
            next_ep_returns = jnp.where(next_dones, 0.0, next_ep_returns_running)

            # The transition stores the RAW (pre-reset) next_obs -- the true
            # terminal observation, which is what the termination loss /
            # value bootstrap need -- while the carry moving on to the next
            # scan iteration gets the RESET obs/state for any env that just
            # finished
            transitions = {
                "obs": c.obs, "next_obs": next_obs, "raw_action": raw_actions,
                "option": next_options, "reward": rewards, "done": next_dones,
                "decision_flag": decision_flags, "action_logp": action_logp,
                "option_logp": option_logp, "q_value": q_value,
            }
            buffer_state = buffer.add(c.buffer_state, transitions)

            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            reset_obs, reset_env_states = vmap_reset(reset_keys)
            carry_obs = where_per_env(next_dones, reset_obs, next_obs)
            carry_env_states = jax.tree.map(
                lambda a, b: where_per_env(next_dones, a, b), reset_env_states, next_env_states
            )

            metrics = {
                "done": next_dones, "completed_return": completed_return,
                "options": next_options, "decision_flag": decision_flags,
            }
            next_c = RolloutCarry(
                env_states=carry_env_states, obs=carry_obs, dones=next_dones,
                options=next_options, ep_returns=next_ep_returns,
                buffer_state=buffer_state, params=c.params, opt_state=c.opt_state,
                global_step=c.global_step + num_envs,
            )
            return next_c, metrics

        return jax.lax.scan(scan_step, carry, step_keys)

    # JITTED PPO Update

    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def ppo_update(rng, params, opt_state, buffer_state, bootstrap_obs, bootstrap_option):
        raw = buffer_state.experience
        raw = jax.tree.map(lambda a: a[:, :rollout_horizon, ...], raw)
        batch = jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), raw)  # (T, N, ...)

        q_omega_bootstrap, _ = value_net.apply(params['value'], bootstrap_obs)
        bootstrap_value = q_omega_bootstrap[jnp.arange(bootstrap_obs.shape[0]), bootstrap_option]

        # Deliberation cost (Harb et al. 2018): the reward actually trained
        # on charges eta to the step where a new option was just started, on
        # top of the paper's Section 4 reward/10 rescaling (reward_scale,
        # 1.0 i.e. a no-op unless configured -- see config comment above).
        r_hat = (batch["reward"] * config['reward_scale']
                 - config['delib_cost'] * batch["decision_flag"].astype(jnp.float32))

        adv, ret = jax.vmap(
            compute_option_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
        )(r_hat, batch["q_value"], batch["done"], bootstrap_value, config['gamma'], config['gae_lambda'])

        flat = {
            "obs": batch["obs"].reshape((-1, obs_dim)),
            "next_obs": batch["next_obs"].reshape((-1, obs_dim)),
            "option": batch["option"].reshape(-1),
            "raw_action": batch["raw_action"].reshape((-1, action_dim)),
            "action_logp": batch["action_logp"].reshape(-1),
            "q_value": batch["q_value"].reshape(-1),
            "decision_flag": batch["decision_flag"].reshape(-1),
            "nonterminal": (1.0 - batch["done"].astype(jnp.float32)).reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
        }
        total_samples = rollout_horizon * num_envs
        minibatch_size = total_samples // config['num_minibatches']

        def epoch_step(carry, epoch_rng):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_rng, total_samples)

            def mb_step(carry2, mb_idx):
                params, opt_state = carry2
                idx = jax.lax.dynamic_slice_in_dim(perm, mb_idx * minibatch_size, minibatch_size)
                mb = jax.tree.map(lambda a: a[idx], flat)

                def p_loss(p):
                    return ppoc_policy_loss_fn(
                        p, policy_net, params['value'], value_net,
                        mb["obs"], mb["option"], mb["raw_action"], mb["action_logp"],
                        mb["adv"], mb["decision_flag"], config['clip_eps'], config['entropy_coef'],
                    )
                (loss_policy, policy_aux), grads_policy = jax.value_and_grad(p_loss, has_aux=True)(params['policy'])
                policy_updates, new_policy_opt = policy_optimizer.update(
                    grads_policy, opt_state['policy'], params['policy']
                )
                new_policy_params = optax.apply_updates(params['policy'], policy_updates)

                def v_loss(p):
                    return ppoc_value_loss_fn(
                        p, value_net, params['policy'], policy_net,
                        mb["obs"], mb["next_obs"], mb["option"], mb["nonterminal"],
                        mb["ret"], mb["q_value"], config['clip_eps'], config['value_coef'],
                        config['delib_cost'],
                    )
                (loss_value, value_aux), grads_value = jax.value_and_grad(v_loss, has_aux=True)(params['value'])
                value_updates, new_value_opt = value_optimizer.update(
                    grads_value, opt_state['value'], params['value']
                )
                new_value_params = optax.apply_updates(params['value'], value_updates)

                new_params = {'policy': new_policy_params, 'value': new_value_params}
                new_opt_state = {'policy': new_policy_opt, 'value': new_value_opt}
                mb_metrics = {
                    "loss_policy": loss_policy, "loss_value": loss_value,
                    "intra_option_loss": policy_aux["intra_option_loss"],
                    "policy_over_options_loss": policy_aux["policy_over_options_loss"],
                    "gauss_entropy": policy_aux["gauss_entropy"],
                    "mu_entropy": policy_aux["mu_entropy"],
                    "approx_kl": policy_aux["approx_kl"],
                    "critic_loss": value_aux["critic_loss"],
                    "termination_loss": value_aux["termination_loss"],
                    "beta_mean": value_aux["beta_mean"],
                }
                return (new_params, new_opt_state), mb_metrics

            (params, opt_state), mb_metrics = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(config['num_minibatches'])
            )
            return (params, opt_state), mb_metrics

        epoch_keys = jax.random.split(rng, config['ppo_epochs'])
        (params, opt_state), all_metrics = jax.lax.scan(epoch_step, (params, opt_state), epoch_keys)

        update_metrics = jax.tree.map(jnp.mean, all_metrics)
        return params, opt_state, update_metrics

    # Logging
    def run_and_log(carry, rng_key, step0):
        keys = jax.random.split(rng_key, rollout_horizon)
        carry, rollout_metrics = run_rollout_chunk(carry, keys)

        train_rng, next_key = jax.random.split(rng_key)
        params, opt_state, update_metrics = ppo_update(
            train_rng, carry.params, carry.opt_state, carry.buffer_state,
            carry.obs, carry.options,
        )

        # Buffer contents were fully consumed; re-init for the next chunk.
        carry = carry._replace(
            buffer_state=buffer.init(dummy_transition), params=params, opt_state=opt_state
        )

        rollout_metrics = jax.device_get(rollout_metrics)
        update_metrics = jax.device_get(update_metrics)

        total_dones = np.sum(rollout_metrics["done"])
        sum_returns = np.sum(rollout_metrics["completed_return"])
        true_mean_return = float(np.where(total_dones > 0, sum_returns / total_dones, 0.0))

        env_steps0 = step0 * num_envs

        chunk_metrics = {
            "train/loss_policy": float(update_metrics["loss_policy"]),
            "train/loss_value": float(update_metrics["loss_value"]),
            "train/intra_option_loss": float(update_metrics["intra_option_loss"]),
            "train/policy_over_options_loss": float(update_metrics["policy_over_options_loss"]),
            "train/gauss_entropy": float(update_metrics["gauss_entropy"]),
            "train/mu_entropy": float(update_metrics["mu_entropy"]),
            "train/approx_kl": float(update_metrics["approx_kl"]),
            "train/critic_loss": float(update_metrics["critic_loss"]),
            "train/termination_loss": float(update_metrics["termination_loss"]),
            "train/beta_mean": float(update_metrics["beta_mean"]),
            "train/return_mean": true_mean_return,
        }
        logger.log_metrics(chunk_metrics, step=env_steps0)

        option_usage = {f"options/usage_{opt_idx}": float(np.mean(rollout_metrics["options"] == opt_idx))
                         for opt_idx in range(num_options)}
        option_usage["options/decision_rate"] = float(np.mean(rollout_metrics["decision_flag"]))

        for t in range(rollout_horizon):
            current_t_step = env_steps0 + (t * num_envs)
            opts_at_t = rollout_metrics["options"][t]
            decisions_at_t = rollout_metrics["decision_flag"][t]
            t_metrics = {f"options/usage_{opt_idx}": float(np.mean(opts_at_t == opt_idx))
                         for opt_idx in range(num_options)}
            t_metrics["options/decision_rate"] = float(np.mean(decisions_at_t))
            logger.log_metrics(t_metrics, step=current_t_step)

        print(f"Steps {step0}-{step0 + rollout_horizon} (x{num_envs} envs = "
              f"{env_steps0}-{env_steps0 + rollout_horizon * num_envs} env-steps) "
              f"| Policy Loss: {chunk_metrics['train/loss_policy']:.4f} "
              f"| Value Loss: {chunk_metrics['train/loss_value']:.4f} "
              f"| True Return: {true_mean_return:.2f}")

        return carry, next_key, {**chunk_metrics, **option_usage}

    # -----------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------
    print("Starting PPOC parallel rollouts...")

    total_steps = config['num_steps']

    eval_config = config_raw.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = max(int(total_steps * eval_config.get("interval_pct", 0.05)), rollout_horizon)
    next_eval_step = eval_interval

    checkpoint_config = config_raw.get("checkpoint", {})
    do_ckpt = checkpoint_config.get("enabled", True)
    ckpt_interval = max(int(total_steps * checkpoint_config.get("interval_pct", 0.2)), rollout_horizon)
    next_ckpt_step = ckpt_interval

    for step_idx in range(0, total_steps, rollout_horizon):
        key, chunk_key = jax.random.split(key)
        carry, key, chunk_snapshot = run_and_log(carry, chunk_key, step_idx)

        current_env_step = (step_idx + rollout_horizon) * num_envs

        if do_ckpt and step_idx >= next_ckpt_step:
            logger.save_checkpoint(jax.device_get(carry.params), current_env_step)
            next_ckpt_step += ckpt_interval

        if do_eval and step_idx >= next_eval_step:
            logger.save_eval_metrics(current_env_step, chunk_snapshot)
            print(f"--- Eval snapshot written at Step {step_idx} (env step {current_env_step}) ---")
            next_eval_step += eval_interval

    print("Training completed.")
    logger.close()
