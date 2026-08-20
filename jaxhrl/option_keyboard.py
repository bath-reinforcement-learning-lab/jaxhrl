# Option Keyboard in JAX
# From "The Option Keyboard: Combining Skills in Reinforcement Learning"
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import flashbax as fbx

from jaxhrl.common.utils import parse_config
from jaxhrl.common.logger import Logger
from jaxhrl.common.jax_wrappers import make_jax_env


class GRUCell(nnx.Module): 
    def __init__(self, din: int, dhidden: int, *, rngs: nnx.Rngs):
        self.dhidden = dhidden
        self.r_w = nnx.Linear(din + dhidden, dhidden, rngs=rngs)
        self.z_w = nnx.Linear(din + dhidden, dhidden, rngs=rngs)
        self.n_w = nnx.Linear(din + dhidden, dhidden, rngs=rngs)

    def initialize_carry(self, batch_size: int = 1):
        return jnp.zeros((batch_size, self.dhidden))

    def __call__(self, carry, x):
        concat = jnp.concatenate([x, carry], axis=-1)
        r = jax.nn.sigmoid(self.r_w(concat))
        z = jax.nn.sigmoid(self.z_w(concat))
        concat_n = jnp.concatenate([x, r * carry], axis=-1)
        n = jnp.tanh(self.n_w(concat_n))
        next_carry = (1 - z) * n + z * carry
        return next_carry, next_carry


class RecurrentEncoder(nnx.Module):
    """
    maps u(h, a, s') -> h (hidden state)
    """
    def __init__(self, din: int, dhidden: int, dout: int, num_actions: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.linear1 = nnx.Linear(din + num_actions, dhidden, rngs=rngs)
        self.gru = GRUCell(dhidden, dout, rngs=rngs)

    def __call__(self, carry, prev_action, x):
        a_onehot = jax.nn.one_hot(prev_action, self.num_actions)
        x = jnp.concatenate([x, a_onehot], axis=-1)
        x = nnx.relu(self.linear1(x))
        next_carry, h = self.gru(carry, x)
        return h, next_carry


class SFHead(nnx.Module):
    """
    maps h -> psi 
    take hidden state successor features)
    """
    def __init__(self, dh: int, dhidden: int, num_actions: int, num_options: int, cumulant_dim: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.num_options = num_options
        self.cumulant_dim = cumulant_dim
        self.linear1 = nnx.Linear(dh, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, num_actions * num_options * cumulant_dim, rngs=rngs)

    def __call__(self, h):
        x = nnx.relu(self.linear1(h))
        x = self.linear2(x)
        out_shape = x.shape[:-1] + (self.num_options, self.num_actions, self.cumulant_dim)
        return x.reshape(out_shape)


class TerminationHead(nnx.Module):
    """
    maps h -> beta (probability distribution of terminating each option)
    """
    def __init__(self, dh: int, dhidden: int, num_options: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dh, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, num_options, rngs=rngs)

    def __call__(self, h):
        x = nnx.relu(self.linear1(h))
        return jax.nn.sigmoid(self.linear2(x))


class Agent(nnx.Module):
    """
    encapsulates encoder, sf head and termination head
    """
    def __init__(self, state_dim, hidden_dim, h_dim, num_actions, cumulant_dim, num_options, *, rngs):
        self.encoder = RecurrentEncoder(state_dim, hidden_dim, h_dim, num_actions, rngs=rngs)
        self.sf_head = SFHead(h_dim, hidden_dim, num_actions, num_options, cumulant_dim, rngs=rngs)
        self.term_head = TerminationHead(h_dim, hidden_dim, num_options, rngs=rngs)

    def __call__(self, carry, prev_action, s):
        h, next_carry = self.encoder(carry, prev_action, s)
        psi = self.sf_head(h)
        beta = self.term_head(h)
        return psi, beta, next_carry


def sf_loss(agent, target_agent, W_options, carry_batch, next_carry_batch, prev_action_batch, s_batch, a_batch,
            phi_batch, s_next_batch, done_batch, gamma):
    psi_all, _beta, _ = agent(carry_batch, prev_action_batch, s_batch)
    batch_indices = jnp.arange(s_batch.shape[0])
    psi_sa = psi_all[batch_indices, :, a_batch, :]  # (batch, options, cumulant_dim)

    psi_next_all, _beta_next, _ = target_agent(next_carry_batch, a_batch, s_next_batch)  # (batch, options, actions, cumulant_dim)

    # each option picks its own greedy action using its own w_i, not a shared w
    q_per_option = jnp.einsum('boad,od->boa', psi_next_all, W_options)  # (batch, options, actions)
    a_star = jnp.argmax(q_per_option, axis=-1)  # (batch, options)

    def get_max_psi(psi_n, a_s):
        return psi_n[jnp.arange(psi_n.shape[0]), a_s, :]

    psi_next = jax.vmap(get_max_psi)(psi_next_all, a_star)  # (batch, options, cumulant_dim)
    target = phi_batch[:, None, :] + gamma * psi_next * (1.0 - done_batch[:, None, None])
    target = jax.lax.stop_gradient(target)
    return jnp.mean((psi_sa - target) ** 2)


def reward_loss(w, phi, true_reward):
    # phi: (num_envs, cumulant_dim), true_reward: (num_envs,) -- jnp.dot handles
    # this as a batched matrix-vector product, giving (num_envs,) predictions.
    predicted_reward = jnp.dot(phi, w)
    return jnp.mean((predicted_reward - true_reward) ** 2)


def where_per_env(done_mask, a, b):
    """jnp.where with `done_mask` (shape (num_envs,)) broadcast across any
    trailing per-env feature dims of a/b."""
    reshaped = done_mask.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(reshaped, a, b)


class LoopCarry(NamedTuple):
    models_state: Any
    w: jnp.ndarray
    buffer_state: Any
    carry_h: jnp.ndarray
    prev_action: jnp.ndarray
    obs: jnp.ndarray
    env_state: Any
    cum_reward: jnp.ndarray
    step_count: jnp.ndarray


if __name__ == "__main__":

    # parse config
    config = parse_config()
    logger = Logger(config)

    framework_type = config["env"].get("framework", "gymnax")
    env_id = config["env"]["make"]["id"].split("/")[-1]
    cumulant_dim = config["training"].get("cumulant_dim", 3)
    wrapped = make_jax_env(framework_type, env_id, cumulant_dim)

    n_steps = config["training"].get("n_steps", 20_000)
    gamma = config["training"].get("gamma", 0.99)
    eval_freq = config["training"].get("eval_freq", 500)
    epsilon = config["training"].get("epsilon", 0.1)
    target_sync_every = config["training"].get("target_sync_every", 200)
    num_envs = config["training"].get("num_envs", 1024)

    state_dim = wrapped.state_dim
    num_actions = wrapped.num_actions
    num_options = config["training"].get("num_options", 10)

    seed = config["seed"]

    # initialise
    key = jax.random.PRNGKey(seed)
    key, k1, k2, w_key = jax.random.split(key, 4)

    w0 = jax.random.normal(w_key, (cumulant_dim,))
    key, w_opts_key = jax.random.split(key)
    raw = jax.random.normal(w_opts_key, (num_options, cumulant_dim))
    W_options = raw / (jnp.linalg.norm(raw, axis=-1, keepdims=True) + 1e-8)  # (num_options, cumulant_dim)

    network_cfg = config.get("network", {})
    hidden_dim = network_cfg.get("encoder", {}).get("hidden_dim", 64)
    h_dim = network_cfg.get("gru", {}).get("h_dim", 32)

    agent = Agent(state_dim, hidden_dim, h_dim, num_actions, cumulant_dim, num_options, rngs=nnx.Rngs(k1))
    target_agent = Agent(state_dim, hidden_dim, h_dim, num_actions, cumulant_dim, num_options, rngs=nnx.Rngs(k2))
    nnx.update(target_agent, jax.tree_util.tree_map(jnp.copy, nnx.state(agent)))

    optimizer = nnx.Optimizer(
        agent,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
        wrt=nnx.Param,
    )

    train_batch_size = config["training"].get("batch_size", 64)
    buffer = fbx.make_flat_buffer(
        max_length=config["training"].get("buffer_size", 200_000),
        min_length=train_batch_size,
        sample_batch_size=train_batch_size,
        add_batch_size=num_envs,  
    )

    dummy_transition = {
        "obs": jnp.zeros((state_dim,), dtype=jnp.float32),
        "carry": jnp.zeros((h_dim,), dtype=jnp.float32),
        "prev_action": jnp.array(-1, dtype=jnp.int32),
        "action": jnp.array(0, dtype=jnp.int32),
        "phi": jnp.zeros((cumulant_dim,), dtype=jnp.float32),
        "done": jnp.array(0.0, dtype=jnp.float32),
    }
    buffer_state0 = buffer.init(dummy_transition)

    vmap_reset = jax.vmap(wrapped.reset_fn)
    vmap_step = jax.vmap(wrapped.step_fn, in_axes=(0, 0, 0))
    # Update in_axes to account for the two new state arguments
    vmap_cumulant = jax.vmap(wrapped.cumulant_fn, in_axes=(0, 0, 0, 0, 0, 0))

    key, reset_key = jax.random.split(key)
    reset_keys0 = jax.random.split(reset_key, num_envs)
    obs0, state0 = vmap_reset(reset_keys0)
    carry_h0 = agent.encoder.gru.initialize_carry(batch_size=num_envs)
    prev_action0 = jnp.full((num_envs,), -1, dtype=jnp.int32)  # no previous action at episode start

    graphdef, models_state0 = nnx.split((agent, target_agent, optimizer))

    # training loop
    def scan_body(carry: LoopCarry, step_key):
        akey, skey, sample_key, reset_key = jax.random.split(step_key, 4)
        akey_explore, akey_random = jax.random.split(akey)

        # batched action selection 
        cur_agent, _cur_target, _cur_opt = nnx.merge(graphdef, carry.models_state)
        psi_all, _beta, next_carry_h = cur_agent(carry.carry_h, carry.prev_action, carry.obs)  # obs: (num_envs, state_dim)
        q_all = jnp.einsum('boad,d->boa', psi_all, carry.w)  # (num_envs, options, actions)
        best_option = jnp.argmax(jnp.max(q_all, axis=-1), axis=-1)  # (num_envs,)
        gpi_act = jnp.argmax(q_all[jnp.arange(num_envs), best_option], axis=-1)  # (num_envs,)

        explore = jax.random.uniform(akey_explore, shape=(num_envs,)) < epsilon
        random_action = jax.random.randint(akey_random, shape=(num_envs,), minval=0, maxval=num_actions)
        action = jnp.where(explore, random_action, gpi_act).astype(jnp.int32)

        # --- vectorized env step (pure, vmapped) ---
        step_keys = jax.random.split(skey, num_envs)
        next_obs, next_env_state, reward, done, _info = vmap_step(step_keys, carry.env_state, action)
        
        # Pass the newly required states to the cumulant function
        phi = vmap_cumulant(carry.obs, carry.env_state, action, reward, next_obs, next_env_state)

        transition = {
            "obs": carry.obs,
            "carry": carry.carry_h,
            "prev_action": carry.prev_action,
            "action": action,
            "phi": phi,
            "done": done.astype(jnp.float32),
        }
        new_buffer_state = buffer.add(carry.buffer_state, transition)

        # --- w update, aggregated over all num_envs in this step ---
        w_loss_val, w_grad = jax.value_and_grad(reward_loss)(carry.w, phi, reward)
        new_w = carry.w - 0.01 * w_grad

        # --- SF training step on a sampled minibatch, conditional on buffer readiness ---
        can_sample = buffer.can_sample(new_buffer_state)

        def do_train(models_state):
            a, t, opt = nnx.merge(graphdef, models_state)
            batch = buffer.sample(new_buffer_state, sample_key).experience
            loss_val, grads = nnx.value_and_grad(sf_loss)(
                a, t, W_options,
                batch.first["carry"], batch.second["carry"], batch.first["prev_action"],
                batch.first["obs"], batch.first["action"],
                batch.first["phi"], batch.second["obs"],
                batch.first["done"], gamma,
            )
            opt.update(a, grads)
            return nnx.state((a, t, opt)), loss_val

        def skip_train(models_state):
            return models_state, jnp.array(0.0)

        new_models_state, sf_loss_val = jax.lax.cond(
            can_sample, do_train, skip_train, carry.models_state
        )

        # --- periodic target network sync ---
        do_sync = (carry.step_count % target_sync_every) == 0

        def sync(models_state):
            a, t, opt = nnx.merge(graphdef, models_state)
            nnx.update(t, jax.tree_util.tree_map(jnp.copy, nnx.state(a)))
            return nnx.state((a, t, opt))

        def no_sync(models_state):
            return models_state

        new_models_state = jax.lax.cond(do_sync, sync, no_sync, new_models_state)

        # --- per-env reset on done ---
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_env_state = vmap_reset(reset_keys)

        out_obs = where_per_env(done, reset_obs, next_obs)
        out_env_state = jax.tree_util.tree_map(
            lambda r, n: where_per_env(done, r, n), reset_env_state, next_env_state
        )
        out_carry_h = where_per_env(done, jnp.zeros_like(carry.carry_h), next_carry_h)
        # On done: no previous action for the fresh episode (-1 sentinel).
        # Otherwise: the action just taken (a_t) becomes next step's prev_action.
        out_prev_action = jnp.where(done, -1, action).astype(jnp.int32)
        out_cum_reward = jnp.where(done, 0.0, carry.cum_reward + reward)

        new_carry = LoopCarry(
            models_state=new_models_state,
            w=new_w,
            buffer_state=new_buffer_state,
            carry_h=out_carry_h,
            prev_action=out_prev_action,
            obs=out_obs,
            env_state=out_env_state,
            cum_reward=out_cum_reward,
            step_count=carry.step_count + 1,
        )
        completed_return = carry.cum_reward + reward
        metrics = {
            "sf_loss": sf_loss_val, 
            "w_loss": w_loss_val, 
            "done": done.astype(jnp.float32),
            "completed_return": jnp.where(done, completed_return, 0.0)
        }
        return new_carry, metrics

    @jax.jit(donate_argnums=0)
    def run_chunk(carry, keys):
        return jax.lax.scan(scan_body, carry, keys)

    carry = LoopCarry(
        models_state=models_state0,
        w=w0,
        buffer_state=buffer_state0,
        carry_h=carry_h0,
        prev_action=prev_action0,
        obs=obs0,
        env_state=state0,
        cum_reward=jnp.zeros((num_envs,)),
        step_count=jnp.array(0, dtype=jnp.int32),
    )

    def run_and_log(carry, key, n, step0):
        keys = jax.random.split(key, n)
        carry, metrics = run_chunk(carry, keys)
        
        mean_sf_loss = float(jnp.mean(metrics["sf_loss"]))
        mean_w_loss = float(jnp.mean(metrics["w_loss"]))
        
        # Calculate the true episode return
        total_dones = jnp.sum(metrics["done"])
        sum_returns = jnp.sum(metrics["completed_return"])
        # Avoid division by zero if no episodes finished in this chunk
        true_mean_return = float(jnp.where(total_dones > 0, sum_returns / total_dones, 0.0))
        
        env_steps0 = step0 * num_envs
        logger.log_metric("train/return_mean", true_mean_return, env_steps0)
        logger.log_metric("objectives/sf_loss", mean_sf_loss, env_steps0)
        
        print(f"Steps {step0}-{step0 + n} (x{num_envs} envs = {env_steps0}-{env_steps0 + n * num_envs} env-steps) "
              f"| SF Loss: {mean_sf_loss:.4f} | W Loss: {mean_w_loss:.4f} | True Return: {true_mean_return:.2f}")
        return carry

    n_chunks = n_steps // eval_freq
    for chunk_idx in range(n_chunks):
        key, subkey = jax.random.split(key)
        carry = run_and_log(carry, subkey, eval_freq, chunk_idx * eval_freq)

    remainder = n_steps % eval_freq
    if remainder:
        key, subkey = jax.random.split(key)
        carry = run_and_log(carry, subkey, remainder, n_chunks * eval_freq)

    logger.close()
