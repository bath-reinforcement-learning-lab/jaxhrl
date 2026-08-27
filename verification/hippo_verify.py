"""
HiPPO empirical verification.

Reproduces two of the paper's own reported tests against the
`jaxhrl/HiPPO.py` code. The only new code is the environment (`sparse_compass.py`
-- a stand-in for the paper's continuous-action MuJoCo Block/Gather tasks,
which this categorical-action-only repo can't run directly)

Test B (Figure 3 -- learning from scratch + time-commitment ablation): all
four conditions below reuse the exact same training loop and the exact same
`ManagerActorCritic`/`SkillActorCritic`/loss functions; only `num_skills`,
`p_min`, and `p_max` change, exactly mirroring the paper's own ablation
structure (HiPPO with randomized period, HiPPO with fixed period, HiPPO with
p=1, and a flat, non-hierarchical policy).

  - "HiPPO fixed p":   paper's "HiPPO p=10"-style condition.
  - "HiPPO p=1":       paper's degenerate ablation -- manager redecides
                        every step.
  - "Flat PPO":        num_skills=1, p_min=p_max=1 -- the skill index never
                        carries information, so the SkillActorCritic is
                        acting as a plain flat policy. This reuses the exact
                        same code path as the hierarchical conditions rather
                        than a separately-written flat-PPO implementation.

Test A (Table 2 -- skill-diversity / gradient-approximation diagnostic): run
on a batch of real rollout data from the trained "HiPPO random p" policy,
computing the same two quantities the paper reports in Table 2.
"""
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from repo_loader import load_hippo
import sparse_compass as env

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
NUM_ENVS = 1024
ROLLOUT_HORIZON = 100
NUM_ITERATIONS = 400
LATENT_DIM = 32
LR = 3e-4
GAMMA = 0.97
GAE_LAMBDA = 0.9
CLIP_EPS = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
NUM_MINIBATCHES = 4

CONDITIONS = {
    "HiPPO random p (8-12)": {"num_skills": 4, "p_min": 8, "p_max": 12},
    "HiPPO fixed p=10":      {"num_skills": 4, "p_min": 10, "p_max": 10},
    "HiPPO p=1 (ablation)":  {"num_skills": 4, "p_min": 1, "p_max": 1},
    "Flat PPO":              {"num_skills": 1, "p_min": 1, "p_max": 1},
}


def build_networks(hippo, num_skills):
    manager_net = hippo.ManagerActorCritic(num_skills=num_skills, latent_dim=LATENT_DIM)
    skill_net = hippo.SkillActorCritic(
        num_actions=env.NUM_ACTIONS, num_skills=num_skills, latent_dim=LATENT_DIM
    )
    return manager_net, skill_net


def init_params(key, manager_net, skill_net, num_skills):
    dummy_obs = jnp.zeros((1, env.STATE_DIM), dtype=jnp.float32)
    dummy_skill = jnp.zeros((1,), dtype=jnp.int32)
    dummy_time = jnp.zeros((1,), dtype=jnp.float32)
    km, ks = jax.random.split(key)
    return {
        "manager": manager_net.init(km, dummy_obs, dummy_time),
        "skill": skill_net.init(ks, dummy_obs, dummy_skill, dummy_time),
    }


def make_train_iteration(hippo, config, manager_net, skill_net, manager_opt, skill_opt):
    """Builds one jitted function that runs ROLLOUT_HORIZON env steps with
    the REAL `batch_select_hippo_action`, then PPO_EPOCHS epochs of updates
    with the REAL `skill_loss_fn`/`manager_loss_fn`, closely mirroring
    `HiPPO.py`'s own `run_rollout_chunk` + `ppo_update`. `config` is closed
    over as plain Python constants, exactly as `HiPPO.py` itself does (its
    jitted functions reference the outer `config` dict by closure, not as a
    traced argument).
    """

    def scan_step(carry, step_key):
        action_keys = jax.random.split(step_key, NUM_ENVS)
        (actions, next_skills, next_commitment, decision_flags,
         manager_logp, manager_value, skill_logp, skill_value, _,
         time_remaining) = hippo.batch_select_hippo_action(
            action_keys, carry["obs"], carry["commitment_left"], carry["skill"],
            carry["params"]["manager"], manager_net, carry["params"]["skill"], skill_net, config,
        )
        env_key = jax.random.fold_in(step_key, 1)
        env_keys = jax.random.split(env_key, NUM_ENVS)
        next_obs, next_state, reward, done, infos = jax.vmap(env.step_fn)(
            env_keys, carry["env_state"], actions
        )

        remaining = jnp.where(done, 0, next_commitment - 1)
        next_ep_running = carry["ep_return"] + reward
        completed_return = jnp.where(done, next_ep_running, 0.0)
        next_ep_return = jnp.where(done, 0.0, next_ep_running)

        transition = {
            "obs": carry["obs"], "action": actions, "skill": next_skills,
            "reward": reward, "done": done, "decision_flag": decision_flags,
            "skill_logp": skill_logp, "skill_value": skill_value,
            "manager_logp": manager_logp, "manager_value": manager_value,
            "time_remaining": time_remaining,
        }
        metrics = {"done": done, "completed_return": completed_return, "reached": infos["reached_goal"]}
        next_carry = dict(carry)
        next_carry.update({
            "env_state": next_state, "obs": next_obs, "commitment_left": remaining,
            "skill": next_skills, "ep_return": next_ep_return,
        })
        return next_carry, (transition, metrics)

    def rollout(carry, rng):
        keys = jax.random.split(rng, ROLLOUT_HORIZON)
        final_carry, (batch, metrics) = jax.lax.scan(scan_step, carry, keys)
        return final_carry, batch, metrics

    def ppo_update(params, opt_state, batch, final_carry, rng):
        bootstrap_time = final_carry["commitment_left"].astype(jnp.float32) / config["p_max"]
        _, manager_bootstrap_value = manager_net.apply(params["manager"], final_carry["obs"], bootstrap_time)
        _, skill_bootstrap_value = skill_net.apply(
            params["skill"], final_carry["obs"], final_carry["skill"], bootstrap_time
        )

        skill_adv, skill_ret = jax.vmap(
            hippo.compute_skill_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
        )(batch["reward"], batch["skill_value"], batch["done"], skill_bootstrap_value, GAMMA, GAE_LAMBDA)

        manager_target, manager_valid = jax.vmap(
            hippo.compute_manager_smdp_targets, in_axes=(1, 1, 1, 1, 0, None), out_axes=1
        )(batch["reward"], batch["done"], batch["decision_flag"], batch["manager_value"],
          manager_bootstrap_value, GAMMA)
        manager_adv = manager_target - batch["manager_value"]

        flat = {
            "obs": batch["obs"].reshape((-1, env.STATE_DIM)),
            "action": batch["action"].reshape(-1),
            "skill": batch["skill"].reshape(-1),
            "skill_logp": batch["skill_logp"].reshape(-1),
            "skill_value": batch["skill_value"].reshape(-1),
            "manager_logp": batch["manager_logp"].reshape(-1),
            "manager_value": batch["manager_value"].reshape(-1),
            "time_remaining": batch["time_remaining"].reshape(-1),
            "manager_valid": manager_valid.reshape(-1),
            "skill_adv": skill_adv.reshape(-1),
            "skill_ret": skill_ret.reshape(-1),
            "manager_adv": manager_adv.reshape(-1),
            "manager_ret": manager_target.reshape(-1),
        }

        total = ROLLOUT_HORIZON * NUM_ENVS
        mb_size = total // NUM_MINIBATCHES

        def epoch_step(carry2, epoch_key):
            params, opt_state = carry2
            perm = jax.random.permutation(epoch_key, total)

            def mb_step(carry3, i):
                params, opt_state = carry3
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb_size, mb_size)
                mb = jax.tree_util.tree_map(lambda a: a[idx], flat)

                def s_loss(p):
                    return hippo.skill_loss_fn(
                        p, skill_net, mb["obs"], mb["skill"], mb["action"],
                        mb["skill_logp"], mb["skill_adv"], mb["skill_ret"],
                        mb["skill_value"], mb["time_remaining"], config,
                    )
                (l_s, aux_s), g_s = jax.value_and_grad(s_loss, has_aux=True)(params["skill"])
                upd_s, new_opt_s = skill_opt.update(g_s, opt_state["skill"], params["skill"])
                new_skill = optax.apply_updates(params["skill"], upd_s)

                def m_loss(p):
                    return hippo.manager_loss_fn(
                        p, manager_net, mb["obs"], mb["skill"], mb["manager_logp"],
                        mb["manager_adv"], mb["manager_ret"], mb["manager_value"],
                        mb["manager_valid"], mb["time_remaining"], config,
                    )
                (l_m, aux_m), g_m = jax.value_and_grad(m_loss, has_aux=True)(params["manager"])
                upd_m, new_opt_m = manager_opt.update(g_m, opt_state["manager"], params["manager"])
                new_manager = optax.apply_updates(params["manager"], upd_m)

                new_params = {"skill": new_skill, "manager": new_manager}
                new_opt_state = {"skill": new_opt_s, "manager": new_opt_m}
                return (new_params, new_opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(NUM_MINIBATCHES)
            )
            return (params, opt_state), None

        epoch_keys = jax.random.split(rng, PPO_EPOCHS)
        (params, opt_state), _ = jax.lax.scan(epoch_step, (params, opt_state), epoch_keys)
        return params, opt_state

    @jax.jit
    def train_iteration(carry, rng):
        rollout_rng, update_rng = jax.random.split(rng)
        new_carry, batch, metrics = rollout(carry, rollout_rng)
        new_params, new_opt_state = ppo_update(
            carry["params"], carry["opt_state"], batch, new_carry, update_rng
        )
        new_carry["params"] = new_params
        new_carry["opt_state"] = new_opt_state
        return new_carry, metrics, batch

    return train_iteration


def run_condition(hippo, cond_name, cond_cfg, seed):
    config = dict(cond_cfg)
    config.update({
        "clip_eps": CLIP_EPS,
        "entropy_coef": ENTROPY_COEF,
        "value_coef": VALUE_COEF,
    })
    key = jax.random.PRNGKey(seed)

    manager_net, skill_net = build_networks(hippo, config["num_skills"])
    key, init_key = jax.random.split(key)
    params = init_params(init_key, manager_net, skill_net, config["num_skills"])

    manager_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    skill_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    opt_state = {"manager": manager_opt.init(params["manager"]), "skill": skill_opt.init(params["skill"])}

    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs0, state0 = jax.vmap(env.reset_fn)(reset_keys)

    carry = {
        "env_state": state0, "obs": obs0,
        "commitment_left": jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
        "skill": jnp.full((NUM_ENVS,), -1, dtype=jnp.int32),
        "ep_return": jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
        "params": params, "opt_state": opt_state,
    }

    train_iteration = make_train_iteration(hippo, config, manager_net, skill_net, manager_opt, skill_opt)

    return_history = []
    hit_rate_history = []
    last_batch = None
    for it in range(NUM_ITERATIONS):
        key, iter_key = jax.random.split(key)
        carry, metrics, batch = train_iteration(carry, iter_key)
        metrics = jax.device_get(metrics)
        total_dones = np.sum(metrics["done"])
        mean_return = float(np.sum(metrics["completed_return"]) / total_dones) if total_dones > 0 else 0.0
        hit_rate = float(np.mean(metrics["reached"]))
        return_history.append(mean_return)
        hit_rate_history.append(hit_rate)
        last_batch = batch
        if it % 10 == 0 or it == NUM_ITERATIONS - 1:
            print(f"    [{cond_name} seed={seed}] iter {it:3d}  mean_return={mean_return:.3f}  "
                  f"per-step goal-hit-rate={hit_rate:.4f}", flush=True)

    return {
        "return_history": return_history,
        "hit_rate_history": hit_rate_history,
        "params": carry["params"],
        "manager_net": manager_net,
        "skill_net": skill_net,
        "config": config,
        "last_batch": jax.device_get(last_batch),
    }


def skill_diversity_diagnostic(hippo, run_result):
    """Table 2's own diagnostic, computed at the granularity Lemma 1 actually
    concerns: eps = max prob of the sampled action under a DIFFERENT skill
    (per-step, as the paper defines it), the probability under the
    actually-sampled skill, and the cosine similarity between the exact
    gradient (Eq. 3's mixture over skills of each whole commitment-period's
    *joint* sub-trajectory probability, summed with logsumexp for numerical
    stability) and the approximate gradient (Eq. 4 -- what the repo actually
    differentiates: only the sampled skill's log-prob). Computing the exact
    side per-step instead of per-block would systematically dilute it by a
    spurious extra factor of 1/num_skills per step that Lemma 1's own proof
    does not have (the mixture there is over whole-block joint
    probabilities, where a wrong skill's contribution is suppressed by
    roughly eps^block_length, not eps^1) -- so blocks (contiguous runs of a
    single held skill, found via `decision_flag`) are the correct unit here.
    """
    skill_net = run_result["skill_net"]
    params = run_result["params"]["skill"]
    config = run_result["config"]
    num_skills = config["num_skills"]
    batch = run_result["last_batch"]
    all_skill_ids = jnp.arange(num_skills)

    obs_all = batch["obs"]
    action_all = batch["action"]
    skill_all = batch["skill"]
    decision_all = batch["decision_flag"]
    time_all = batch["time_remaining"]
    T, N = action_all.shape

    # A block runs from one `decision_flag` to the next (a whole commitment
    # period under one held skill). Lemma 1's skill-diversity assumption
    # (0 < pi_l(a|s,z_j) < eps for j != actual) is about behavior that is
    # actually *driven by z* -- it doesn't apply to this environment's brief
    # cue-visible window at the very start of each block (`env.REVEAL_STEPS`
    # steps), where the observation alone tells every skill the right
    # action, so all of them agree regardless of z by construction. That
    # window is excluded from both the eps/taken-prob sample and the
    # gradient comparison below, so the diagnostic measures what the
    # assumption actually concerns: the memory-driven portion of the block.
    max_envs_to_scan = min(64, N)
    blocks = []
    for env_idx in range(max_envs_to_scan):
        starts = np.nonzero(np.asarray(decision_all[:, env_idx]))[0]
        for i, s in enumerate(starts):
            e = int(starts[i + 1]) if i + 1 < len(starts) else T
            s2 = s + getattr(env, "REVEAL_STEPS", 0)
            if e - s2 < 1:
                continue
            blocks.append((
                jnp.asarray(obs_all[s2:e, env_idx]),
                jnp.asarray(action_all[s2:e, env_idx]),
                int(skill_all[s, env_idx]),
                jnp.asarray(time_all[s2:e, env_idx]),
            ))
        if len(blocks) >= 150:
            break
    blocks = blocks[:150]

    # --- eps / taken-prob: Table 2's own per-step quantities, sampled from
    # the same post-reveal steps as the gradient comparison below ---
    o_s = jnp.concatenate([ob for ob, ac, z, tm in blocks], axis=0)
    a_s = jnp.concatenate([ac for ob, ac, z, tm in blocks], axis=0)
    t_s = jnp.concatenate([tm for ob, ac, z, tm in blocks], axis=0)
    z_s = jnp.concatenate([jnp.full((ac.shape[0],), z, dtype=jnp.int32) for ob, ac, z, tm in blocks], axis=0)
    n_sample = min(512, o_s.shape[0])
    o_s, a_s, z_s, t_s = o_s[:n_sample], a_s[:n_sample], z_s[:n_sample], t_s[:n_sample]

    def probs_under_every_skill(o, a, t):
        tiled_o = jnp.tile(o[None, :], (num_skills, 1))
        tiled_t = jnp.full((num_skills,), t)
        logits, _ = skill_net.apply(params, tiled_o, all_skill_ids, tiled_t)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return jnp.exp(logp[:, a])

    probs_all_skills = jax.vmap(probs_under_every_skill)(o_s, a_s, t_s)  # (n, num_skills)
    taken_prob = jnp.take_along_axis(probs_all_skills, z_s[:, None], axis=1)[:, 0]
    mask = jax.nn.one_hot(z_s, num_skills).astype(bool)
    eps = jnp.max(jnp.where(mask, -jnp.inf, probs_all_skills), axis=1)

    def approx_total_logp(p):
        total = 0.0
        for ob, ac, z, tm in blocks:
            logits, _ = skill_net.apply(p, ob, jnp.full((ob.shape[0],), z, dtype=jnp.int32), tm)
            logp = jax.nn.log_softmax(logits, axis=-1)
            total = total + jnp.sum(jnp.take_along_axis(logp, ac[:, None], axis=1))
        return total

    def exact_total_logp(p):
        total = 0.0
        for ob, ac, z, tm in blocks:
            def joint_logp_given_z(zz):
                logits, _ = skill_net.apply(p, ob, jnp.full((ob.shape[0],), zz, dtype=jnp.int32), tm)
                logp = jax.nn.log_softmax(logits, axis=-1)
                return jnp.sum(jnp.take_along_axis(logp, ac[:, None], axis=1))
            joint_logp_per_z = jax.vmap(joint_logp_given_z)(all_skill_ids)
            total = total + (jax.scipy.special.logsumexp(joint_logp_per_z) - jnp.log(num_skills))
        return total

    approx_grad = jax.grad(approx_total_logp)(params)
    exact_grad = jax.grad(exact_total_logp)(params)

    approx_flat = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(approx_grad)])
    exact_flat = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(exact_grad)])
    cosine_sim = jnp.dot(approx_flat, exact_flat) / (
        jnp.linalg.norm(approx_flat) * jnp.linalg.norm(exact_flat) + 1e-12
    )

    return {
        "eps_mean": float(jnp.mean(eps)),
        "eps_std": float(jnp.std(eps)),
        "taken_prob_mean": float(jnp.mean(taken_prob)),
        "taken_prob_std": float(jnp.std(taken_prob)),
        "cosine_similarity": float(cosine_sim),
        "n_sample": int(n_sample),
        "n_blocks": len(blocks),
    }


def smoothed(x, window=5):
    x = np.asarray(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    t0 = time.time()
    hippo = load_hippo()

    print(f"=== Test B: learning from scratch + time-commitment ablation "
          f"(sparse_compass: goal_distance={env.GOAL_DISTANCE}, horizon={env.HORIZON}) ===")

    all_results = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        seed_returns = []
        seed_hitrates = []
        run_for_diag = None
        for seed in SEEDS:
            print(f"  -- condition '{cond_name}', seed {seed} --")
            result = run_condition(hippo, cond_name, cond_cfg, seed)
            seed_returns.append(result["return_history"])
            seed_hitrates.append(result["hit_rate_history"])
            if seed == SEEDS[0] and cond_name == "HiPPO random p (8-12)":
                run_for_diag = result
        all_results[cond_name] = {
            "return_history_per_seed": seed_returns,
            "hit_rate_history_per_seed": seed_hitrates,
        }
        if run_for_diag is not None:
            all_results[cond_name]["_diag_source"] = run_for_diag

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond_name, res in all_results.items():
        curves = np.array(res["hit_rate_history_per_seed"])
        mean_curve = curves.mean(axis=0)
        axes[0].plot(smoothed(mean_curve), label=cond_name)
        hitcurves = np.array(res["hit_rate_history_per_seed"])
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("per-step goal-hit rate (smoothed)")
    axes[0].set_title("Sparse-compass: goal discovery rate")
    axes[0].legend(fontsize=8)

    for cond_name, res in all_results.items():
        curves = np.array(res["return_history_per_seed"])
        mean_curve = curves.mean(axis=0)
        axes[1].plot(smoothed(mean_curve), label=cond_name)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mean episode return (smoothed)")
    axes[1].set_title("Sparse-compass: learning curves")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hippo_learning_curves.png", dpi=130)
    plt.close(fig)

    summary = {"conditions": {}}
    for cond_name, res in all_results.items():
        returns = np.array(res["return_history_per_seed"])
        hits = np.array(res["hit_rate_history_per_seed"])
        summary["conditions"][cond_name] = {
            "config": CONDITIONS[cond_name],
            "final_10iter_mean_return": float(returns[:, -10:].mean()),
            "auc_mean_return": float(returns.mean()),
            "final_10iter_hit_rate": float(hits[:, -10:].mean()),
            "auc_hit_rate": float(hits.mean()),
        }

    print("\n=== Test A: Table 2 skill-diversity / gradient-approximation diagnostic ===")
    diag_source = all_results["HiPPO random p (8-12)"]["_diag_source"]
    diag = skill_diversity_diagnostic(hippo, diag_source)
    print(f"  eps (max prob under a wrong skill)      = {diag['eps_mean']:.4f} +/- {diag['eps_std']:.4f}")
    print(f"  prob under the actually-sampled skill    = {diag['taken_prob_mean']:.4f} +/- {diag['taken_prob_std']:.4f}")
    print(f"  cosine sim(exact grad, approximate grad) = {diag['cosine_similarity']:.4f}")
    summary["skill_diversity_diagnostic"] = diag

    summary["runtime_sec"] = time.time() - t0
    with open(RESULTS_DIR / "hippo_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['runtime_sec']:.1f}s. "
          f"Summary written to {RESULTS_DIR / 'hippo_verification_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
