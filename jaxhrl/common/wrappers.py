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

def make_jax_env(framework: str, env_id: str, cumulant_dim: int, goal_threshold: float = 0.1) -> JaxWrappedEnv:
    
    def cumulant_fn(obs, state, action, reward, next_obs, next_state):
        if "CartPole" in env_id:
            phi = jnp.array([reward, -jnp.abs(next_obs[2]), -jnp.abs(next_obs[0])])
        elif "Snake" in env_id:
            phi = jnp.array([reward, next_obs[2], next_obs[4]])
        elif "Craftax" in env_id or framework == "craftax":
            # Pass state and next_state instead of obs
            phi = _craftax_cumulant_fn(state, action, reward, next_state, cumulant_dim)
            return phi
        else:
            flat_obs = next_obs.flatten()
            pad_len = max(0, cumulant_dim - 1 - flat_obs.shape[0])
            padded_obs = jnp.pad(flat_obs, (0, pad_len))
            phi = jnp.concatenate([jnp.array([reward]), padded_obs])
        return phi[:cumulant_dim]

        @jax.jit
        def step_fn(key, state, action):
            obs_dict, next_state, reward, done, info = env.step(key, state, action, env_params)
            flat_obs = jnp.concatenate([jnp.asarray(x).flatten() for x in jax.tree_util.tree_leaves(obs_dict)])
            
            # Pack the raw dictionary into info so native_goal_reached_fn can use it later
            info["base_observation_dict"] = obs_dict
            
            return flat_obs, next_state, reward, done, info

        def native_goal_reached_fn(obs_prev_flat, goal_idx, obs_next_flat, action, info_dict):
            obs_dict = info_dict["base_observation_dict"]
            actual_goal = goal_indexes_to_goals(all_goals, goal_idx)
            success = jax.vmap(goal_achieved)(obs_dict, actual_goal)


            intrinsic_reward = success.astype(jnp.float32)
            return intrinsic_reward, success

        return JaxWrappedEnv(env, env_params, state_dim, num_actions, reset_fn, step_fn,
                             cumulant_fn, None, num_goals, None, None, native_goal_reached_fn)

    
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

    else:
        raise ValueError(f"Unknown JAX environment framework wrapper target: {framework}")
