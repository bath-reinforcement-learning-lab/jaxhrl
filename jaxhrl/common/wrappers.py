import functools
import jax
import jax.numpy as jnp
from typing import Tuple, Any, NamedTuple, Callable

class JaxWrappedEnv(NamedTuple):
    env: Any
    env_params: Any
    state_dim: int
    num_actions: int
    reset_fn: Any
    step_fn: Any
    cumulant_fn: Any
    goal_fn: Any
    num_goals: int
    goal_kind: jax.Array
    goal_target: jax.Array
    goal_reached_fn: Callable
    # Continuous-action envs (e.g. brax) only: num_actions/goal_* above are
    # discrete-action machinery and stay at their harmless defaults (0/None)
    # for these. action_low/action_high default to None rather than being
    # omitted (NamedTuple fields can't vary per-instance) -- callers that
    # want the "unset -> fall back to a config default" behaviour must check
    # for None explicitly, since getattr(env, "action_low", default) will
    # always find the field present and return None, not the default.
    action_dim: int = 0
    action_low: Any = None
    action_high: Any = None


_CRAFTAX_FULL_OFFSETS = dict(
    wood=8217, stone=8218, coal=8219, iron=8220, diamond=8221,
    sapphire=8222, ruby=8223,
    pickaxe_level=8228, sword_level=8229,
    health=8239, food=8240, drink=8241, energy=8242, mana=8243,
    current_floor=8265,
)

def _craftax_goal_fn(obs):
    o = _CRAFTAX_FULL_OFFSETS
    return jnp.array([
        obs[o["wood"]], obs[o["stone"]], obs[o["coal"]], obs[o["iron"]], obs[o["diamond"]],
        obs[o["pickaxe_level"]], obs[o["sword_level"]],
        obs[o["health"]], obs[o["food"]], obs[o["drink"]], obs[o["energy"]],
        obs[o["current_floor"]],
    ])

_CRAFTAX_GOAL_KIND = jnp.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
_CRAFTAX_GOAL_TARGET = jnp.array([1., 1., 1., 1., 1., 1., 1., 9., 9., 9., 9., 1.])

def _cartpole_goal_fn(obs):
    return jnp.array([obs[0], obs[2]])  

_CARTPOLE_GOAL_KIND = jnp.array([0, 0])          
_CARTPOLE_GOAL_TARGET = jnp.array([0.0, 0.0])    

def _generic_goal_fn(obs, num_goals):
    flat = obs.flatten()
    pad_len = max(0, num_goals - flat.shape[0])
    return jnp.pad(flat, (0, pad_len))[:num_goals]


def _make_goal_reached_fn(goal_fn, goal_kind, goal_target, threshold):
    """
    Generic reached/intrinsic-reward function.
    Now accepts 'info' as a 5th argument to prevent TypeErrors from hierarchical algorithms,
    even though the legacy environments ignore it.
    """
    def goal_reached_fn(obs_prev, goal, obs_next, action, info=None):
        feats_prev = jax.vmap(goal_fn)(obs_prev)          
        feats_next = jax.vmap(goal_fn)(obs_next)          

        feat_prev = jnp.take_along_axis(feats_prev, goal[:, None], axis=-1).squeeze(-1)
        feat_next = jnp.take_along_axis(feats_next, goal[:, None], axis=-1).squeeze(-1)

        kind = goal_kind[goal]      
        target = goal_target[goal]  

        reached_value = jnp.abs(feat_next - target) < threshold
        reached_delta = (feat_next - feat_prev) >= target

        goal_reached = jnp.where(kind == 0, reached_value, reached_delta)
        intrinsic_r = goal_reached.astype(jnp.float32)  

        return intrinsic_r, goal_reached

    return goal_reached_fn

_CRAFTAX_DENSE_IDX = jnp.array(list(_CRAFTAX_FULL_OFFSETS.values()))

def _craftax_dense_stats(obs):
    # Resource counts, tool levels, intrinsics, dungeon floor -- these change on
    # most steps, unlike the achievement flags below which are rare one-off events.
    return obs[_CRAFTAX_DENSE_IDX]

def _craftax_cumulant_fn(obs, state, action, reward, next_obs, next_state, cumulant_dim):
    # Calculate which achievements were unlocked in this exact step
    current_achievements = state.achievements.astype(jnp.float32)
    next_achievements = next_state.achievements.astype(jnp.float32)
    achievement_delta = next_achievements - current_achievements

    dense_delta = _craftax_dense_stats(next_obs) - _craftax_dense_stats(obs)

    phi = jnp.concatenate([achievement_delta, dense_delta])

    # Pad or truncate to ensure it perfectly matches cumulant_dim
    pad_len = max(0, cumulant_dim - phi.shape[0])
    phi = jnp.pad(phi, (0, pad_len))
    return phi[:cumulant_dim]

def make_jax_env(framework: str, env_id: str, cumulant_dim: int, goal_threshold: float = 0.1, **env_kwargs) -> JaxWrappedEnv:
    
    def cumulant_fn(obs, state, action, reward, next_obs, next_state):
        if "CartPole" in env_id:
            phi = jnp.array([reward, -jnp.abs(next_obs[2]), -jnp.abs(next_obs[0])])
        elif "Snake" in env_id:
            phi = jnp.array([reward, next_obs[2], next_obs[4]])
        elif "Craftax" in env_id or framework == "craftax":
            phi = _craftax_cumulant_fn(obs, state, action, reward, next_obs, next_state, cumulant_dim)
            return phi
        else:
            flat_obs = next_obs.flatten()
            pad_len = max(0, cumulant_dim - 1 - flat_obs.shape[0])
            padded_obs = jnp.pad(flat_obs, (0, pad_len))
            phi = jnp.concatenate([jnp.array([reward]), padded_obs])
        return phi[:cumulant_dim]


    
    if "Craftax" in env_id or framework == "craftax":
        goal_fn = _craftax_goal_fn
        goal_kind = _CRAFTAX_GOAL_KIND
        goal_target = _CRAFTAX_GOAL_TARGET
        num_goals = goal_kind.shape[0]
    elif "CartPole" in env_id:
        goal_fn = _cartpole_goal_fn
        goal_kind = _CARTPOLE_GOAL_KIND
        goal_target = _CARTPOLE_GOAL_TARGET
        num_goals = goal_kind.shape[0]
    else:
        num_goals = 8  
        goal_fn = lambda obs: _generic_goal_fn(obs, num_goals)
        goal_kind = jnp.zeros((num_goals,), dtype=jnp.int32)   
        goal_target = jnp.zeros((num_goals,))

    goal_reached_fn = _make_goal_reached_fn(goal_fn, goal_kind, goal_target, goal_threshold)

    if framework == "gymnax":
        import gymnax
        env, env_params = gymnax.make(env_id)
        try:
            state_dim = env.observation_space(env_params).shape[0]
            num_actions = env.action_space(env_params).n
        except TypeError:
            state_dim = env.observation_space.shape[0]
            num_actions = env.action_space.n

        def reset_fn(key):
            return env.reset(key, env_params)

        def step_fn(key, state, action):
            return env.step(key, state, action, env_params)

        return JaxWrappedEnv(env, env_params, state_dim, num_actions, reset_fn, step_fn,
                              cumulant_fn, goal_fn, num_goals, goal_kind, goal_target, goal_reached_fn)

    elif framework == "jumanji":
        import jumanji
        env = jumanji.make(env_id)

        def flatten_obs(obs_tree):
            leaves = jax.tree_util.tree_leaves(obs_tree)
            return jnp.concatenate([jnp.asarray(x).flatten().astype(jnp.float32) for x in leaves])

        dummy_state, dummy_timestep = env.reset(jax.random.PRNGKey(0))
        flat_obs = flatten_obs(dummy_timestep.observation)
        state_dim = flat_obs.shape[0]
        num_actions = env.action_spec.num_values

        @jax.jit
        def reset_fn(key):
            state, timestep = env.reset(key)
            return flatten_obs(timestep.observation), state

        @jax.jit
        def step_fn(key, state, action):
            next_state, timestep = env.step(state, action)
            reward = timestep.reward
            done = timestep.last()
            info = {}
            return flatten_obs(timestep.observation), next_state, reward, done, info

        return JaxWrappedEnv(env, None, state_dim, num_actions, reset_fn, step_fn,
                              cumulant_fn, goal_fn, num_goals, goal_kind, goal_target, goal_reached_fn)

    elif framework == "craftax":
        from craftax.craftax_env import make_craftax_env_from_name
        actual_env_id = "Craftax-Classic-Symbolic-v1" if "classic" in env_id.lower() else "Craftax-Symbolic-v1"
        env = make_craftax_env_from_name(actual_env_id, auto_reset=True)
        env_params = env.default_params
        state_dim = env.observation_space(env_params).shape[0]
        num_actions = env.action_space(env_params).n

        @jax.jit
        def reset_fn(key):
            return env.reset(key, env_params)

        @jax.jit
        def step_fn(key, state, action):
            return env.step(key, state, action, env_params)

        return JaxWrappedEnv(env, env_params, state_dim, num_actions, reset_fn, step_fn,
                              cumulant_fn, goal_fn, num_goals, goal_kind, goal_target, goal_reached_fn)

    elif framework == "brax":
        # Continuous-action locomotion envs (ant, humanoid, halfcheetah, ...).
        import brax.envs

        env_kwargs = dict(env_kwargs)
        env_kwargs.setdefault("backend", "positional")
        brax_env = brax.envs.get_environment(env_id, **env_kwargs)

        state_dim = brax_env.observation_size
        action_dim = brax_env.action_size
        action_low = -jnp.ones((action_dim,), dtype=jnp.float32)
        action_high = jnp.ones((action_dim,), dtype=jnp.float32)

        @jax.jit
        def reset_fn(key):
            state = brax_env.reset(key)
            return state.obs, state

        @jax.jit
        def step_fn(key, state, action):
            # brax's own step(state, action) is deterministic given state;
            # `key` is accepted (unused) only to match every other branch's
            # (key, state, action) -> (obs, state, reward, done, info) signature.
            next_state = brax_env.step(state, action)
            return (next_state.obs, next_state, next_state.reward,
                   next_state.done.astype(bool), next_state.info)

        return JaxWrappedEnv(brax_env, None, state_dim, 0, reset_fn, step_fn,
                              cumulant_fn, None, 0, None, None, None,
                              action_dim=action_dim, action_low=action_low, action_high=action_high)

    else:
        raise ValueError(f"Unknown JAX environment framework wrapper target: {framework}")




@functools.lru_cache(maxsize=None)
def _jit_step_and_reset(step_fn, reset_fn):
    """Compiles step_fn/reset_fn once per (step_fn, reset_fn) identity and
    caches the result, so repeated eval episodes reuse the same compiled
    executable instead of retracing (and leaking device memory) every call."""
    return jax.jit(step_fn), jax.jit(reset_fn)


def run_eval_episode(env_wrapper, policy_fn, params, key, max_steps=2000):
    import numpy as np

    step_fn_jit, reset_fn_jit = _jit_step_and_reset(env_wrapper.step_fn, env_wrapper.reset_fn)

    key, reset_key = jax.random.split(key)
    obs, state = reset_fn_jit(reset_key)
    
    trajectory = {"reward": [], "option": []}
    
    for t in range(max_steps):
        key, action_key = jax.random.split(key)
        
        action, option = policy_fn(params, obs, action_key)
        
        obs, state, reward, done, info = step_fn_jit(key, state, action)
        
        trajectory["reward"].append(float(jax.device_get(reward)))
        # `option` is a scalar discrete index for option/subgoal-index
        # policies (e.g. h-DQN) but a continuous vector for goal-space
        # policies (e.g. HIRO's sampled goal) -- tolist() handles both,
        # unlike int() which only works for the scalar case.
        trajectory["option"].append(np.asarray(jax.device_get(option)).tolist())
        
        if done:
            break

    return trajectory


def run_offpolicy_eval_stage(logger, env_wrapper, make_policy_fn, params, key, step, eval_config):
    """One eval stage for an off-policy algorithm: runs `eval.num_runs`
    (default 10) greedy episodes and writes their returns to a JSON file via
    logger.save_eval_returns. Also forwards the last episode's trajectory to
    logger.log_eval_trajectory (a no-op unless use_wandb is set), preserving
    the existing W&B table/video behaviour. Returns the advanced RNG key.

    `make_policy_fn` is a zero-arg factory returning a fresh policy_fn(params,
    obs, key) -> (action, option) for one episode -- required because a
    recurrent policy's hidden state must reset between the `num_runs`
    episodes, not just once per eval stage. Stateless policies can pass
    `lambda: greedy_eval_policy`.
    """
    num_runs = eval_config.get("num_runs", 10)
    max_steps = eval_config.get("max_steps", 2000)

    returns = []
    trajectory = None
    for _ in range(num_runs):
        key, run_key = jax.random.split(key)
        trajectory = run_eval_episode(env_wrapper, make_policy_fn(), params, run_key, max_steps=max_steps)
        returns.append(sum(trajectory["reward"]))

    logger.save_eval_returns(step, returns)
    logger.log_eval_trajectory(step, trajectory)
    return key