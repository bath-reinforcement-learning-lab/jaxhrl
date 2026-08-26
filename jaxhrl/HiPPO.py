import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flashbax as fbx
import functools
from typing import Dict, Any, Tuple, NamedTuple
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
    commitment_left: jnp.ndarray
    skills: jnp.ndarray
    ep_returns: jnp.ndarray
    ep_achievements: Any
    buffer_state: Any
    params: Dict[str, Any]
    opt_state: Dict[str, Any]
    global_step: jnp.ndarray


# Networks

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


class ManagerActorCritic(nn.Module):
    """pi_theta_h(z | s): the manager policy over skills, plus V_h(s).

    Li et al. 2020 (HiPPO), Appendix A: with a randomized time-commitment
    (the only mode this file implements), "we ... provide the number of
    timesteps until the next latent selection as an input into both the
    manager and skill networks" -- `time_remaining` below.
    """
    num_skills: int
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x, time_remaining):
        x = x.astype(jnp.float32).reshape((x.shape[0], -1))
        time_remaining = time_remaining.astype(jnp.float32).reshape((-1, 1))
        h = SymbolicEncoder(hidden_dim=self.latent_dim)(jnp.concatenate([x, time_remaining], axis=-1))
        logits = nn.Dense(features=self.num_skills)(h)
        value = nn.Dense(features=1)(h)
        return logits, jnp.squeeze(value, axis=-1)


class SkillActorCritic(nn.Module):
    """pi_theta_l(a | s, z): the intra-option/skill policy, plus V_l(s, z).

    Also takes `time_remaining` -- see ManagerActorCritic docstring.
    """
    num_actions: int
    num_skills: int
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x, skill_idx, time_remaining):
        x = x.astype(jnp.float32).reshape((x.shape[0], -1))
        time_remaining = time_remaining.astype(jnp.float32).reshape((-1, 1))
        h = SymbolicEncoder(hidden_dim=self.latent_dim)(jnp.concatenate([x, time_remaining], axis=-1))
        skill_one_hot = jax.nn.one_hot(skill_idx, self.num_skills)
        h = jnp.concatenate([h, skill_one_hot], axis=-1)
        h = nn.Dense(features=self.latent_dim)(h)
        h = nn.relu(h)
        logits = nn.Dense(features=self.num_actions)(h)
        value = nn.Dense(features=1)(h)
        return logits, jnp.squeeze(value, axis=-1)



# Helper Functions for Action Selection (Algorithm 1: HiPPO Rollout)

def select_hippo_action(key, obs, commitment_left, skill, manager_params, manager_net,
                         skill_params, skill_net, config, greedy=False):
    """One environment step of Algorithm 1.

    `commitment_left <= 0` signals the current time-commitment p has expired
    (or this is the first step of an episode), so the manager samples a new
    p ~ Cat([P_min, P_max]) and a new skill z ~ pi_theta_h(.|s). Otherwise the
    previously sampled skill/commitment is kept and only the skill policy
    acts. `p` is a fixed hyperparameter distribution (not learned), so it
    carries no log-prob.
    """
    p_min, p_max = config['p_min'], config['p_max']
    need_decision = commitment_left <= 0

    key, p_key, z_key, a_key = jax.random.split(key, 4)

    # p is drawn from a fixed prior independent of any network params, so it
    # (and the resulting new_commitment) can be resolved before the manager
    # forward pass -- letting both networks condition on the time remaining
    # until the next decision, per Appendix A.
    sampled_p = p_min + jax.random.randint(p_key, (), 0, p_max - p_min + 1)
    new_commitment = jnp.where(need_decision, sampled_p, commitment_left)
    time_remaining = new_commitment.astype(jnp.float32) / p_max

    manager_logits, manager_value = manager_net.apply(manager_params, obs[None, :], time_remaining[None])
    manager_logits, manager_value = manager_logits[0], manager_value[0]

    sampled_skill = jnp.where(
        greedy, jnp.argmax(manager_logits), jax.random.categorical(z_key, manager_logits)
    )

    new_skill = jnp.where(need_decision, sampled_skill, skill)
    manager_logp = jax.nn.log_softmax(manager_logits)[new_skill]

    skill_logits, skill_value = skill_net.apply(skill_params, obs[None, :], new_skill[None], time_remaining[None])
    skill_logits, skill_value = skill_logits[0], skill_value[0]
    action = jnp.where(
        greedy, jnp.argmax(skill_logits), jax.random.categorical(a_key, skill_logits)
    )
    skill_logp = jax.nn.log_softmax(skill_logits)[action]

    return (action, new_skill, new_commitment, need_decision,
            manager_logp, manager_value, skill_logp, skill_value, key, time_remaining)


def batch_select_hippo_action(keys, obs, commitment_left, skills, manager_params, manager_net,
                               skill_params, skill_net, config):
    batched_step_fn = jax.vmap(
        select_hippo_action,
        in_axes=(0, 0, 0, 0, None, None, None, None, None),
    )
    return batched_step_fn(keys, obs, commitment_left, skills, manager_params, manager_net,
                            skill_params, skill_net, config)


# Advantage Estimation

def compute_skill_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    """Standard per-step GAE(lambda) for the skill/intra-option policy.

    rewards, values, dones: (T,) for a single env. bootstrap_value: scalar,
    V_l(s_T) for whatever skill is active as the chunk ends.
    """
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


def compute_manager_smdp_targets(rewards, dones, decision_flags, manager_values,
                                  bootstrap_value, gamma):
    """

    rewards, dones, decision_flags, manager_values: (T,).
    decision_flags[t] == True marks t as the first step of a new commitment
    (a fresh skill z_t was sampled at t). Returns (target, valid), each (T,);
    only entries where `valid` (== decision_flags) is True are meaningful —
    those are the manager's actual SMDP training transitions. The manager's
    "reward" for choosing z_t is the gamma-discounted sum of rewards over the
    whole commitment window, bootstrapped by V_h at the *next* decision (or
    0 if the episode ended first).
    """
    T = rewards.shape[0]

    #    segment ends: last step a commitment stays active, i.e. either the
    #    episode ends there or the next step starts a fresh decision.
    next_is_decision = jnp.concatenate([decision_flags[1:], jnp.array([True])])
    is_segment_end = dones | next_is_decision

    #    forward pass: discounted reward-so-far within the current segment,
    #    and the running discount power gamma^k applied to r_t so far.
    def fwd(carry, x):
        run_r, run_g = carry
        r_t, start_t = x
        run_r = jnp.where(start_t, 0.0, run_r)
        run_g = jnp.where(start_t, 1.0, run_g)
        run_r = run_r + run_g * r_t
        new_run_g = run_g * gamma
        return (run_r, new_run_g), (run_r, new_run_g)

    _, (partial_return, gamma_pow_after) = jax.lax.scan(
        fwd, (jnp.array(0.0), jnp.array(1.0)), (rewards, decision_flags)
    )

    # at segment-end steps, close out the macro-transition target.
    next_manager_value = jnp.concatenate([manager_values[1:], bootstrap_value[None]])
    target_at_end = partial_return + gamma_pow_after * next_manager_value * (1.0 - dones.astype(jnp.float32))
    target_at_end = jnp.where(is_segment_end, target_at_end, 0.0)

    # backward-fill the segment-end target to the decision (segment-start)
    #    step, so each manager decision gets exactly one training target.
    def bwd(carry, x):
        is_end, val = x
        carry = jnp.where(is_end, val, carry)
        return carry, carry

    _, filled = jax.lax.scan(bwd, jnp.array(0.0), (is_segment_end, target_at_end), reverse=True)

    return filled, decision_flags


# PPO Losses (Algorithm 2: HiPPO)

def ppo_actor_critic_loss(logits, values, old_logp, actions, advantages, returns,
                           old_values, clip_eps, entropy_coef, value_coef, mask=None):
    log_probs_all = jax.nn.log_softmax(logits)
    new_logp = jnp.take_along_axis(log_probs_all, actions[:, None], axis=-1)[:, 0]

    if mask is not None:
        mask_f = mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(mask_f), 1.0)
        adv_mean = jnp.sum(advantages * mask_f) / denom
        adv_var = jnp.sum(mask_f * (advantages - adv_mean) ** 2) / denom
        norm_adv = (advantages - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
    else:
        norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    ratio = jnp.exp(new_logp - old_logp)
    unclipped = ratio * norm_adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
    policy_loss = -jnp.minimum(unclipped, clipped)

    value_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    value_loss_unclipped = (values - returns) ** 2
    value_loss_clipped = (value_clipped - returns) ** 2
    value_loss = 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped)

    probs = jnp.exp(log_probs_all)
    entropy = -jnp.sum(probs * log_probs_all, axis=-1)

    per_sample_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    if mask is not None:
        loss = jnp.sum(per_sample_loss * mask_f) / denom
        approx_kl = jnp.sum((old_logp - new_logp) * mask_f) / denom
        mean_entropy = jnp.sum(entropy * mask_f) / denom
    else:
        loss = jnp.mean(per_sample_loss)
        approx_kl = jnp.mean(old_logp - new_logp)
        mean_entropy = jnp.mean(entropy)

    return loss, {"policy_loss": jnp.mean(policy_loss), "value_loss": jnp.mean(value_loss),
                  "entropy": mean_entropy, "approx_kl": approx_kl}


def skill_loss_fn(skill_params, skill_net, obs, skills, actions, old_logp, advantages,
                   returns, old_values, time_remaining, config):
    logits, values = skill_net.apply(skill_params, obs, skills, time_remaining)
    return ppo_actor_critic_loss(
        logits, values, old_logp, actions, advantages, returns, old_values,
        config['clip_eps'], config['entropy_coef'], config['value_coef'],
    )


def manager_loss_fn(manager_params, manager_net, obs, skills, old_logp, advantages,
                     returns, old_values, valid_mask, time_remaining, config):
    logits, values = manager_net.apply(manager_params, obs, time_remaining)
    return ppo_actor_critic_loss(
        logits, values, old_logp, skills, advantages, returns, old_values,
        config['clip_eps'], config['entropy_coef'], config['value_coef'], mask=valid_mask,
    )


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
        'num_steps': config_raw["training"].get("num_steps", 20_000),
        'num_envs': config_raw["training"].get("num_envs", 1024),
        'p_min': config_raw["training"].get("p_min", 5),
        'p_max': config_raw["training"].get("p_max", 20),
        'num_skills': config_raw["training"].get("num_skills", 8),
        'num_actions': env.num_actions,
        'rollout_horizon': config_raw["training"].get("rollout_horizon", 128),  # horizon H per PPO update
        'gamma': config_raw["training"].get("gamma", 0.99),
        'gae_lambda': config_raw["training"].get("gae_lambda", 0.95),
        'clip_eps': config_raw["training"].get("clip_eps", 0.2),
        'entropy_coef': config_raw["training"].get("entropy_coef", 0.01),
        'value_coef': config_raw["training"].get("value_coef", 0.5),
        'max_grad_norm': config_raw["training"].get("max_grad_norm", 0.5),
        'ppo_epochs': config_raw["training"].get("ppo_epochs", 4),
        'num_minibatches': config_raw["training"].get("num_minibatches", 8),
        'lr_manager': config_raw["training"].get("lr_manager", 3e-4),
        'lr_skill': config_raw["training"].get("lr_skill", 3e-4),
    }

    if config['p_max'] < config['p_min']:
        raise ValueError(
            f"p_max ({config['p_max']}) must be >= p_min ({config['p_min']})"
        )

    config['obs_dim'] = env.state_dim
    num_envs = config['num_envs']
    rollout_horizon = config['rollout_horizon']

    seed = config_raw["seed"]
    key = jax.random.PRNGKey(seed)

    # Initialize
    dummy_obs = jnp.zeros((1, config['obs_dim']), dtype=jnp.float32)
    dummy_skill_idx = jnp.zeros((1,), dtype=jnp.int32)
    dummy_time_remaining = jnp.zeros((1,), dtype=jnp.float32)

    manager_net = ManagerActorCritic(num_skills=config['num_skills'], latent_dim=256)
    skill_net = SkillActorCritic(num_actions=config['num_actions'], num_skills=config['num_skills'], latent_dim=256)

    init_rng_manager, init_rng_skill = jax.random.split(key, 2)
    params = {
        'manager': manager_net.init(init_rng_manager, dummy_obs, dummy_time_remaining),
        'skill': skill_net.init(init_rng_skill, dummy_obs, dummy_skill_idx, dummy_time_remaining),
    }

    # Optimizers (global-norm clipping)
    manager_optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adam(config['lr_manager']),
    )
    skill_optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adam(config['lr_skill']),
    )
    opt_state = {
        'manager': manager_optimizer.init(params['manager']),
        'skill': skill_optimizer.init(params['skill']),
    }

    buffer = fbx.make_flat_buffer(
        max_length=(rollout_horizon + 1) * num_envs,
        min_length=rollout_horizon * num_envs,
        sample_batch_size=num_envs,  # unused for random sampling; required by the API
        add_batch_size=num_envs,
    )

    dummy_transition = {
        "obs": jnp.zeros((config['obs_dim'],), dtype=jnp.float32),
        "action": jnp.zeros((), dtype=jnp.int32),
        "skill": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=bool),
        "decision_flag": jnp.zeros((), dtype=bool),
        "skill_logp": jnp.zeros((), dtype=jnp.float32),
        "skill_value": jnp.zeros((), dtype=jnp.float32),
        "manager_logp": jnp.zeros((), dtype=jnp.float32),
        "manager_value": jnp.zeros((), dtype=jnp.float32),
        "time_remaining": jnp.zeros((), dtype=jnp.float32),
    }
    buffer_state = buffer.init(dummy_transition)

    # ---------------------------------------------------------------
    # Initialize Environment & Agent States
    # ---------------------------------------------------------------
    key, env_rng = jax.random.split(key)
    env_keys = jax.random.split(env_rng, num_envs)

    vmap_reset = jax.vmap(env.reset_fn)
    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    obs0, state0 = vmap_reset(reset_keys0)

    dones = jnp.zeros(num_envs, dtype=bool)
    commitment_left = jnp.zeros(num_envs, dtype=jnp.int32)  # forces a decision on the first step
    skills = jnp.full(num_envs, -1, dtype=jnp.int32)
    ep_returns = jnp.zeros(num_envs, dtype=jnp.float32)
    global_step = jnp.array(0, dtype=jnp.int32)

    # Discover the `infos` pytree returned by env.step_fn (achievement
    # flags/counters, etc.) with a throwaway step, purely to get its
    # structure/dtypes -- JAX needs the carry's pytree structure to be
    # identical on every scan iteration, so `ep_achievements` below has to
    # be built with matching keys/shapes/dtypes up front. This doesn't
    # touch `state0`/`obs0`/the main `key` stream, so it has no effect on
    # the actual rollout.
    _probe_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    _probe_actions = jnp.zeros((num_envs,), dtype=jnp.int32)
    _, _, _, _, dummy_infos = jax.vmap(env.step_fn, in_axes=(0, 0, 0))(
        _probe_keys, state0, _probe_actions
    )
    ep_achievements = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x, dtype=jnp.float32), dummy_infos
    )

    carry = RolloutCarry(
        env_states=state0, obs=obs0, dones=dones, commitment_left=commitment_left,
        skills=skills, ep_returns=ep_returns, ep_achievements=ep_achievements,
        buffer_state=buffer_state, params=params, opt_state=opt_state, global_step=global_step,
    )

    # JITTED Rollout Collection (Algorithm 1)

    @functools.partial(jax.jit, donate_argnums=0)
    def run_rollout_chunk(carry, step_keys):
        """Scans over `rollout_horizon` steps fully on accelerator; pure rollout
        collection, no gradient steps (PPO trains once per full chunk, not
        per-step, since it is on-policy and needs the whole batch for its
        epoch/minibatch updates)."""

        def scan_step(c, rng):
            rng, step_rng = jax.random.split(rng)
            action_keys = jax.random.split(step_rng, num_envs)

            (actions, next_skills, next_commitment, decision_flags,
             manager_logp, manager_value, skill_logp, skill_value, _,
             time_remaining) = batch_select_hippo_action(
                action_keys, c.obs, c.commitment_left, c.skills,
                c.params['manager'], manager_net, c.params['skill'], skill_net, config,
            )

            batch_step_fn = jax.vmap(env.step_fn, in_axes=(0, 0, 0))
            rng, step_rng_env = jax.random.split(rng)
            step_env_keys = jax.random.split(step_rng_env, num_envs)
            next_obs, next_env_states, rewards, next_dones, infos = batch_step_fn(
                step_env_keys, c.env_states, actions
            )

            remaining = jnp.where(next_dones, 0, next_commitment - 1)

            next_ep_returns_running = c.ep_returns + rewards
            completed_return = jnp.where(next_dones, next_ep_returns_running, 0.0)
            next_ep_returns = jnp.where(next_dones, 0.0, next_ep_returns_running)

            next_ep_achievements_running = jax.tree_util.tree_map(
                lambda acc, flag: jnp.maximum(acc, flag.astype(jnp.float32)),
                c.ep_achievements, infos,
            )
            completed_achievements = jax.tree_util.tree_map(
                lambda x: jnp.where(next_dones, x, jnp.zeros_like(x)),
                next_ep_achievements_running,
            )
            next_ep_achievements = jax.tree_util.tree_map(
                lambda x: jnp.where(next_dones, jnp.zeros_like(x), x),
                next_ep_achievements_running,
            )

            transitions = {
                "obs": c.obs,
                "action": actions,
                "skill": next_skills,
                "reward": rewards,
                "done": next_dones,
                "decision_flag": decision_flags,
                "skill_logp": skill_logp,
                "skill_value": skill_value,
                "manager_logp": manager_logp,
                "manager_value": manager_value,
                "time_remaining": time_remaining,
            }
            buffer_state = buffer.add(c.buffer_state, transitions)

            metrics = {
                "done": next_dones,
                "completed_return": completed_return,
                "completed_achievements": completed_achievements,
                "skills": next_skills,
                "decision_flag": decision_flags,
            }
            next_c = RolloutCarry(
                env_states=next_env_states, obs=next_obs, dones=next_dones,
                commitment_left=remaining, skills=next_skills, ep_returns=next_ep_returns,
                ep_achievements=next_ep_achievements, buffer_state=buffer_state,
                params=c.params, opt_state=c.opt_state, global_step=c.global_step + num_envs,
            )
            return next_c, metrics

        return jax.lax.scan(scan_step, carry, step_keys)

    # JITTED PPO Update 

    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def ppo_update(rng, params, opt_state, buffer_state, bootstrap_obs, bootstrap_skill,
                    bootstrap_time_remaining):
        raw = buffer_state.experience
        raw = jax.tree.map(lambda a: a[:, :rollout_horizon, ...], raw)
        batch = jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), raw)

        manager_bootstrap_logits, manager_bootstrap_value = manager_net.apply(
            params['manager'], bootstrap_obs, bootstrap_time_remaining
        )
        skill_bootstrap_logits, skill_bootstrap_value = skill_net.apply(
            params['skill'], bootstrap_obs, bootstrap_skill, bootstrap_time_remaining
        )

        skill_adv, skill_ret = jax.vmap(
            compute_skill_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
        )(batch["reward"], batch["skill_value"], batch["done"],
          skill_bootstrap_value, config['gamma'], config['gae_lambda'])

        manager_target, manager_valid = jax.vmap(
            compute_manager_smdp_targets, in_axes=(1, 1, 1, 1, 0, None), out_axes=1
        )(batch["reward"], batch["done"], batch["decision_flag"], batch["manager_value"],
          manager_bootstrap_value, config['gamma'])
        manager_adv = manager_target - batch["manager_value"]

        flat = {
            "obs": batch["obs"].reshape((-1, config['obs_dim'])),
            "action": batch["action"].reshape(-1),
            "skill": batch["skill"].reshape(-1),
            "skill_logp": batch["skill_logp"].reshape(-1),
            "skill_value": batch["skill_value"].reshape(-1),
            "skill_adv": skill_adv.reshape(-1),
            "skill_ret": skill_ret.reshape(-1),
            "manager_logp": batch["manager_logp"].reshape(-1),
            "manager_value": batch["manager_value"].reshape(-1),
            "manager_adv": manager_adv.reshape(-1),
            "manager_ret": manager_target.reshape(-1),
            "manager_valid": manager_valid.reshape(-1),
            "time_remaining": batch["time_remaining"].reshape(-1),
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

                def s_loss(p):
                    return skill_loss_fn(
                        p, skill_net, mb["obs"], mb["skill"], mb["action"],
                        mb["skill_logp"], mb["skill_adv"], mb["skill_ret"], mb["skill_value"],
                        mb["time_remaining"], config,
                    )
                (loss_skill, skill_aux), grads_skill = jax.value_and_grad(s_loss, has_aux=True)(params['skill'])
                skill_updates, new_skill_opt = skill_optimizer.update(
                    grads_skill, opt_state['skill'], params['skill']
                )
                new_skill_params = optax.apply_updates(params['skill'], skill_updates)

                def m_loss(p):
                    return manager_loss_fn(
                        p, manager_net, mb["obs"], mb["skill"], mb["manager_logp"],
                        mb["manager_adv"], mb["manager_ret"], mb["manager_value"], mb["manager_valid"],
                        mb["time_remaining"], config,
                    )
                (loss_manager, manager_aux), grads_manager = jax.value_and_grad(m_loss, has_aux=True)(params['manager'])
                manager_updates, new_manager_opt = manager_optimizer.update(
                    grads_manager, opt_state['manager'], params['manager']
                )
                new_manager_params = optax.apply_updates(params['manager'], manager_updates)

                new_params = {'skill': new_skill_params, 'manager': new_manager_params}
                new_opt_state = {'skill': new_skill_opt, 'manager': new_manager_opt}
                mb_metrics = {
                    "loss_skill": loss_skill, "loss_manager": loss_manager,
                    "skill_entropy": skill_aux["entropy"], "manager_entropy": manager_aux["entropy"],
                    "skill_kl": skill_aux["approx_kl"], "manager_kl": manager_aux["approx_kl"],
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

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    def run_and_log(carry, rng_key, step0):
        keys = jax.random.split(rng_key, rollout_horizon)
        carry, rollout_metrics = run_rollout_chunk(carry, keys)

        train_rng, next_key = jax.random.split(rng_key)
        bootstrap_time_remaining = carry.commitment_left.astype(jnp.float32) / config['p_max']
        params, opt_state, update_metrics = ppo_update(
            train_rng, carry.params, carry.opt_state, carry.buffer_state, carry.obs, carry.skills,
            bootstrap_time_remaining,
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
            "train/loss_skill": float(update_metrics["loss_skill"]),
            "train/loss_manager": float(update_metrics["loss_manager"]),
            "train/entropy_skill": float(update_metrics["skill_entropy"]),
            "train/entropy_manager": float(update_metrics["manager_entropy"]),
            "train/approx_kl_skill": float(update_metrics["skill_kl"]),
            "train/approx_kl_manager": float(update_metrics["manager_kl"]),
            "train/return_mean": true_mean_return,
        }

        # Per-completed-episode achievement rate, not a raw per-timestep
        # average over (possibly mid-episode) envs -- see completed_return
        # above for the identical pattern. `total_dones` is shared with
        # return_mean since both are "per completed episode in this chunk".
        for k, v in rollout_metrics["completed_achievements"].items():
            chunk_metrics[f"achievements/{k}"] = float(np.where(total_dones > 0, np.sum(v) / total_dones, 0.0))

        logger.log_metrics(chunk_metrics, step=env_steps0)

        num_skills = config['num_skills']
        for t in range(rollout_horizon):
            current_t_step = env_steps0 + (t * num_envs)
            t_metrics = {}

            skills_at_t = rollout_metrics["skills"][t]
            decisions_at_t = rollout_metrics["decision_flag"][t]
            for skill_idx in range(num_skills):
                t_metrics[f"skills/usage_{skill_idx}"] = float(np.mean(skills_at_t == skill_idx))
            t_metrics["skills/decision_rate"] = float(np.mean(decisions_at_t))

            logger.log_metrics(t_metrics, step=current_t_step)

        print(f"Steps {step0}-{step0 + rollout_horizon} (x{num_envs} envs = "
              f"{env_steps0}-{env_steps0 + rollout_horizon * num_envs} env-steps) "
              f"| Skill Loss: {chunk_metrics['train/loss_skill']:.4f} "
              f"| Manager Loss: {chunk_metrics['train/loss_manager']:.4f} "
              f"| True Return: {true_mean_return:.2f}")

        return carry, next_key


    # Main Training Loop
    print("Starting HiPPO parallel rollouts...")

    total_steps = config['num_steps']

    eval_config = config_raw.get("eval", {})
    do_eval = eval_config.get("enabled", False)
    eval_interval = int(total_steps * eval_config.get("interval_pct", 0.05))
    eval_max_steps = eval_config.get("max_steps", 2000)
    next_eval_step = eval_interval

    @jax.jit
    def greedy_eval_policy(eval_params, single_obs, commitment_left, skill, eval_key):
        # Greedy manager + greedy skill; the time-commitment p is still
        # sampled since it is a fixed hyperparameter distribution, not a
        # learned quantity, so there is nothing to make "greedy" about it.
        # commitment_left/skill are threaded in by the caller (run_eval_episode's
        # policy_fn is stateless across steps, so persistence has to happen
        # outside this jitted step -- see eval_policy_fn below, matching
        # HIRO.py's eval_hier_state closure pattern).
        act, new_skill, new_commitment, _, _, _, _, _, _, _ = select_hippo_action(
            key=eval_key,
            obs=single_obs,
            commitment_left=commitment_left,
            skill=skill,
            manager_params=eval_params['manager'],
            manager_net=manager_net,
            skill_params=eval_params['skill'],
            skill_net=skill_net,
            config=config,
            greedy=True,
        )
        return act, new_skill, new_commitment

    for step_idx in range(0, total_steps, rollout_horizon):
        key, chunk_key = jax.random.split(key)
        carry, key = run_and_log(carry, chunk_key, step_idx)

        if do_eval and step_idx >= next_eval_step:
            print(f"\n--- Running Evaluation at Step {step_idx} ---")

            frozen_params = jax.device_get(carry.params)
            current_env_step = (step_idx + rollout_horizon) * num_envs

            logger.save_checkpoint(frozen_params, current_env_step)

            key, eval_key = jax.random.split(key)
            from jaxhrl.common.wrappers import run_eval_episode

            # run_eval_episode calls policy_fn(params, obs, key) with no
            # persisted state, but the manager's decision must hold for
            # commitment_left steps -- track commitment_left/skill across
            # calls in a closure, reset fresh for this eval episode.
            eval_hippo_state = {
                "commitment_left": jnp.array(0, dtype=jnp.int32),
                "skill": jnp.array(-1, dtype=jnp.int32),
            }

            def eval_policy_fn(eval_params, single_obs, step_eval_key):
                action, new_skill, new_commitment = greedy_eval_policy(
                    eval_params, single_obs,
                    eval_hippo_state["commitment_left"], eval_hippo_state["skill"],
                    step_eval_key,
                )
                eval_hippo_state["commitment_left"] = new_commitment - 1
                eval_hippo_state["skill"] = new_skill
                return action, new_skill

            trajectory = run_eval_episode(
                env_wrapper=env,
                policy_fn=eval_policy_fn,
                params=frozen_params,
                key=eval_key,
                max_steps=eval_max_steps,
            )

            logger.log_eval_trajectory(current_env_step, trajectory)

            next_eval_step += eval_interval
            print(f"--- Evaluation Complete. Total Reward: {sum(trajectory['reward']):.2f} ---\n")

    print("Training completed.")
    logger.close()