"""
PPOC empirical verification.

Reproduces the central empirical claim of Klissarov, Bacon, Harb & Precup,
"Learnings Options End-to-End for Continuous Action Tasks" (NeurIPS 2017 Deep
RL workshop, arXiv:1712.00004), Section 4 / Figure 1: on continuous-control
locomotion, options trained with a deliberation cost beat primitive actions
(the paper's own "Primitives" baseline), and the size of that improvement is
NOT monotonic in the deliberation cost eta ("the increase in performance is
not directly proportional to eta").

Environment choice: the paper's own tasks are Hopper, Walker2d, HalfCheetah,
and a custom HopperIceBlock (no public implementation / no brax equivalent --
skipped, noted in REPORT.md)

Run: python verification/ppoc_verify.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import brax.envs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from repo_loader import load_ppoc

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_ENVS = 256
ROLLOUT = 128                # env-steps per on-policy PPO iteration
HIDDEN_DIM = 64               # paper's own hidden size (Sec. 4)
GRAD_CLIP = 0.5
EPOCHS = 4
NUM_MINIBATCHES = 8
LOG_EVERY = 10                # iterations per logged/aggregated point
EPISODE_LEN_CAP = 1000        # time-limit truncation -- see scan_step for why
REWARD_SCALE_OPTIONS = 0.1    # paper Sec. 4's reward/10 trick, options runs only

BASE_CONFIG = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "lr": 3e-4,
}

# Primary sweep: hopper, varying num_options / deliberation cost eta. Includes
# a num_options=2 arm alongside num_options=4: an early 4-option pass showed a
# lot of seeds collapsing to decision_rate in {0, 1} (the known Option-Critic
# degenerate-termination failure mode), wasting most of the
# 4 options' capacity; fewer options is the standard mitigation to try before
# concluding options don't help (see this session's debug-then-tune note).
HOPPER_ITERS = 400            # x256 envs x128 steps = 13.1M env-steps/run
HOPPER_SEEDS = [0, 1, 2]
HOPPER_CONDITIONS = [
    ("Primitive actions (1 option)", 1, 0.0),
    ("Options (4), eta=0.0",          4, 0.0),
    ("Options (4), eta=0.01",         4, 0.01),
    ("Options (4), eta=0.05",         4, 0.05),
    ("Options (2), eta=0.01",         2, 0.01),
    ("Options (2), eta=0.05",         2, 0.05),
]

# Cross-environment check: does "options beat primitives" hold beyond hopper?
# The "Options" condition is picked dynamically in main() as whichever
# non-primitive condition scored best in the hopper sweep above.
CROSS_ENVS = ["walker2d", "halfcheetah"]
CROSS_ITERS = 400             # x256 envs x128 steps = 13.1M env-steps/run
CROSS_SEEDS = [0, 1]


def make_env_fns(env_id):
    env = brax.envs.get_environment(env_id, backend="positional")

    def reset_fn(key):
        state = env.reset(key)
        return state.obs, state

    def step_fn(key, state, action):
        next_state = env.step(state, action)
        return (next_state.obs, next_state, next_state.reward,
                jnp.asarray(next_state.done).astype(bool), next_state.info)

    return env, reset_fn, step_fn


def where_per_env(mask, a, b):
    """See jaxhrl/PPOC.py's `where_per_env` -- brax envs don't auto-reset,
    so any env whose episode just ended has to be explicitly reset before
    the next step, or it stays "dead" (done=True forever) for the rest of
    training."""
    return jnp.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)


def build_agent(ppoc, num_options, obs_dim, act_dim, key):
    policy_net = ppoc.PolicyNetwork(num_options=num_options, action_dim=act_dim, hidden_dim=HIDDEN_DIM)
    value_net = ppoc.ValueNetwork(num_options=num_options, hidden_dim=HIDDEN_DIM)
    kp, kv = jax.random.split(key)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = {"policy": policy_net.init(kp, dummy_obs), "value": value_net.init(kv, dummy_obs)}
    policy_opt = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(BASE_CONFIG["lr"]))
    value_opt = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(BASE_CONFIG["lr"]))
    opt_state = {"policy": policy_opt.init(params["policy"]), "value": value_opt.init(params["value"])}
    return policy_net, value_net, params, opt_state, policy_opt, value_opt


def make_train_iter(ppoc, reset_fn, step_fn, policy_net, value_net, policy_opt, value_opt,
                     config, obs_dim, act_dim):
    """One jitted on-policy iteration: ROLLOUT env-steps with the real
    `batch_select_ppoc_action` (+ explicit reset-on-done), then
    EPOCHS x NUM_MINIBATCHES gradient steps with the real
    `ppoc_policy_loss_fn` / `ppoc_value_loss_fn`."""
    total = ROLLOUT * NUM_ENVS
    mb_size = total // NUM_MINIBATCHES

    def scan_step(carry, rng):
        params, env_state, obs, dones, options, ep_return, ep_len = carry

        rng, act_rng = jax.random.split(rng)
        act_keys = jax.random.split(act_rng, NUM_ENVS)
        (actions, raw_actions, next_options, action_logp, option_logp,
         q_value, decision_flags, _) = ppoc.batch_select_ppoc_action(
            act_keys, obs, options, dones, params["policy"], policy_net,
            params["value"], value_net, config,
        )

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_env_state, rewards, next_dones, infos = jax.vmap(step_fn)(
            step_keys, env_state, actions
        )

        # Time limit truncation: brax halfcheetah (unlike hopper/walker2d)
        # never sets done=True on its own (no "unhealthy" state to terminate
        # on), so without an explicit cap episodes never "complete" and every
        # completion-gated metric below would silently read 0 forever. Treat
        # hitting EPISODE_LEN_CAP steps as an episode boundary for ALL envs,
        # uniformly, for both bookkeeping and the training signal (done ->
        # GAE/value target) 
        running_len = ep_len + 1
        episode_done = next_dones | (running_len >= EPISODE_LEN_CAP)

        running = ep_return + rewards
        completed_return = jnp.where(episode_done, running, 0.0)
        next_ep_return = jnp.where(episode_done, 0.0, running)
        completed_len = jnp.where(episode_done, running_len, 0)
        next_ep_len = jnp.where(episode_done, 0, running_len)

        rng, reset_rng = jax.random.split(rng)
        reset_obs, reset_state = jax.vmap(reset_fn)(jax.random.split(reset_rng, NUM_ENVS))
        carry_obs = where_per_env(episode_done, reset_obs, next_obs)
        carry_state = jax.tree.map(lambda a, b: where_per_env(episode_done, a, b), reset_state, next_env_state)

        trans = {
            "obs": obs, "next_obs": next_obs, "raw_action": raw_actions, "option": next_options,
            "reward": rewards, "done": episode_done, "decision_flag": decision_flags,
            "action_logp": action_logp, "q_value": q_value,
        }
        metrics = {
            "completed_return": completed_return, "completed_len": completed_len,
            "n_done": jnp.sum(episode_done), "decision_flag": decision_flags, "options": next_options,
        }
        carry = (params, carry_state, carry_obs, episode_done, next_options, next_ep_return, next_ep_len)
        return carry, (trans, metrics)

    @jax.jit
    def train_iter(carry, params, opt_state, rng):
        roll_rng, epoch_rng = jax.random.split(rng)
        env_state, obs, dones, options, ep_return, ep_len = carry
        roll_carry = (params, env_state, obs, dones, options, ep_return, ep_len)
        roll_carry, (batch, metrics) = jax.lax.scan(
            scan_step, roll_carry, jax.random.split(roll_rng, ROLLOUT)
        )
        (_, env_state, obs, dones, options, ep_return, ep_len) = roll_carry
        carry = (env_state, obs, dones, options, ep_return, ep_len)

        q_bootstrap, _ = value_net.apply(params["value"], obs)
        bootstrap_value = q_bootstrap[jnp.arange(NUM_ENVS), options]
        r_hat = (batch["reward"] * config["reward_scale"]
                 - config["delib_cost"] * batch["decision_flag"].astype(jnp.float32))
        adv, ret = jax.vmap(
            ppoc.compute_option_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
        )(r_hat, batch["q_value"], batch["done"], bootstrap_value, config["gamma"], config["gae_lambda"])

        flat = {
            "obs": batch["obs"].reshape((-1, obs_dim)),
            "next_obs": batch["next_obs"].reshape((-1, obs_dim)),
            "option": batch["option"].reshape(-1),
            "raw_action": batch["raw_action"].reshape((-1, act_dim)),
            "action_logp": batch["action_logp"].reshape(-1),
            "q_value": batch["q_value"].reshape(-1),
            "decision_flag": batch["decision_flag"].reshape(-1),
            "nonterminal": (1.0 - batch["done"].astype(jnp.float32)).reshape(-1),
            "adv": adv.reshape(-1), "ret": ret.reshape(-1),
        }

        def epoch(state, ekey):
            params, opt_state = state
            perm = jax.random.permutation(ekey, total)

            def mb_step(state, i):
                params, opt_state = state
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb_size, mb_size)
                mb = jax.tree_util.tree_map(lambda a: a[idx], flat)

                def p_loss(p):
                    return ppoc.ppoc_policy_loss_fn(
                        p, policy_net, params["value"], value_net,
                        mb["obs"], mb["option"], mb["raw_action"], mb["action_logp"], mb["adv"],
                        mb["decision_flag"], config["clip_eps"], config["entropy_coef"],
                    )
                (lp, aux_p), gp = jax.value_and_grad(p_loss, has_aux=True)(params["policy"])
                up, new_policy_opt = policy_opt.update(gp, opt_state["policy"], params["policy"])
                new_policy_params = optax.apply_updates(params["policy"], up)

                def v_loss(p):
                    return ppoc.ppoc_value_loss_fn(
                        p, value_net, params["policy"], policy_net,
                        mb["obs"], mb["next_obs"], mb["option"], mb["nonterminal"], mb["ret"], mb["q_value"],
                        config["clip_eps"], config["value_coef"], config["delib_cost"],
                    )
                (lv, aux_v), gv = jax.value_and_grad(v_loss, has_aux=True)(params["value"])
                uv, new_value_opt = value_opt.update(gv, opt_state["value"], params["value"])
                new_value_params = optax.apply_updates(params["value"], uv)

                new_params = {"policy": new_policy_params, "value": new_value_params}
                new_opt_state = {"policy": new_policy_opt, "value": new_value_opt}
                mb_metrics = {"loss_policy": lp, "loss_value": lv, **aux_p, **aux_v}
                return (new_params, new_opt_state), mb_metrics

            (params, opt_state), mb_metrics = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(NUM_MINIBATCHES)
            )
            return (params, opt_state), mb_metrics

        (params, opt_state), all_metrics = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(epoch_rng, EPOCHS)
        )
        update_metrics = jax.tree.map(jnp.mean, all_metrics)
        return carry, params, opt_state, metrics, update_metrics

    return train_iter


def fresh_env_carry(reset_fn, num_options, key):
    reset_keys = jax.random.split(key, NUM_ENVS)
    obs0, state0 = jax.vmap(reset_fn)(reset_keys)
    return (
        state0, obs0,
        jnp.ones(NUM_ENVS, dtype=bool),                 # force an option decision on step 1
        jnp.full(NUM_ENVS, -1, dtype=jnp.int32),
        jnp.zeros(NUM_ENVS, dtype=jnp.float32),
        jnp.zeros(NUM_ENVS, dtype=jnp.int32),
    )


def run_condition(ppoc, env_id, num_options, delib_cost, seed, n_iters):
    env, reset_fn, step_fn = make_env_fns(env_id)
    obs_dim, act_dim = env.observation_size, env.action_size

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    policy_net, value_net, params, opt_state, policy_opt, value_opt = build_agent(
        ppoc, num_options, obs_dim, act_dim, init_key
    )

    config = dict(BASE_CONFIG)
    config.update({
        "num_options": num_options, "delib_cost": delib_cost,
        "action_low": -jnp.ones((act_dim,)), "action_high": jnp.ones((act_dim,)),
        # Paper Section 4: "In the case of options, we also divide the reward
        # by 10 ... making [the termination probability gradient] more
        # stable." Applied only to options runs, not the primitive baseline.
        "reward_scale": REWARD_SCALE_OPTIONS if num_options > 1 else 1.0,
    })
    train_iter = make_train_iter(
        ppoc, reset_fn, step_fn, policy_net, value_net, policy_opt, value_opt, config, obs_dim, act_dim
    )

    key, reset_key = jax.random.split(key)
    carry = fresh_env_carry(reset_fn, num_options, reset_key)

    history = []
    acc = {"ret_num": 0.0, "ret_den": 0.0, "len_num": 0.0, "len_den": 0.0, "decision_rate": [], "n": 0}
    env_steps = 0
    for it in range(n_iters):
        key, ik = jax.random.split(key)
        carry, params, opt_state, metrics, update_metrics = train_iter(carry, params, opt_state, ik)
        m = jax.device_get(metrics)
        n_done = float(np.sum(m["n_done"]))
        acc["ret_num"] += float(np.sum(m["completed_return"]))
        acc["ret_den"] += n_done
        acc["len_num"] += float(np.sum(m["completed_len"]))
        acc["len_den"] += n_done
        acc["decision_rate"].append(float(np.mean(m["decision_flag"])))
        acc["n"] += 1
        env_steps += ROLLOUT * NUM_ENVS
        if acc["n"] >= LOG_EVERY:
            history.append((
                env_steps,
                acc["ret_num"] / max(acc["ret_den"], 1.0),
                acc["len_num"] / max(acc["len_den"], 1.0),
                float(np.mean(acc["decision_rate"])),
            ))
            acc = {"ret_num": 0.0, "ret_den": 0.0, "len_num": 0.0, "len_den": 0.0, "decision_rate": [], "n": 0}

    return {"history": history}


def summarize(history):
    ret = np.array([h[1] for h in history])
    length = np.array([h[2] for h in history])
    decision_rate = np.array([h[3] for h in history])
    k = max(len(ret) // 4, 1)
    return {
        "final_return": float(ret[-k:].mean()),
        "auc_return": float(ret.mean()),
        "final_ep_len": float(length[-k:].mean()),
        "final_decision_rate": float(decision_rate[-k:].mean()),
    }


def plot_curves(all_results, title, path):
    plt.figure(figsize=(8, 5))
    for cond_name, seed_hists in all_results.items():
        xs = np.array([h[0] for h in seed_hists[0]])
        ret = np.array([[h[1] for h in hist] for hist in seed_hists])
        m, se = ret.mean(axis=0), ret.std(axis=0) / np.sqrt(ret.shape[0])
        plt.plot(xs, m, label=cond_name)
        plt.fill_between(xs, m - se, m + se, alpha=0.15)
    plt.xlabel("environment steps")
    plt.ylabel("episode return")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    t0 = time.time()
    ppoc = load_ppoc()
    summary = {"config": {"num_envs": NUM_ENVS, "rollout": ROLLOUT, **BASE_CONFIG}, "hopper_sweep": {}, "cross_env": {}}

    # -------------------------------------------------------------------
    # Primary: hopper, num_options / deliberation-cost sweep
    # -------------------------------------------------------------------
    print(f"=== Hopper: primitives vs. options, deliberation-cost sweep "
          f"({len(HOPPER_SEEDS)} seeds, {HOPPER_ITERS} iters = "
          f"{HOPPER_ITERS * ROLLOUT * NUM_ENVS:,} env-steps/run) ===", flush=True)
    hopper_results = {}
    for cond_name, num_options, delib_cost in HOPPER_CONDITIONS:
        seed_hists = []
        for seed in HOPPER_SEEDS:
            t1 = time.time()
            res = run_condition(ppoc, "hopper", num_options, delib_cost, seed, HOPPER_ITERS)
            seed_hists.append(res["history"])
            last = res["history"][-1]
            print(f"  {cond_name}, seed {seed}: final return={last[1]:.1f} "
                  f"ep_len={last[2]:.1f} decision_rate={last[3]:.3f} ({time.time()-t1:.0f}s)", flush=True)
        hopper_results[cond_name] = seed_hists
        stats = [summarize(h) for h in seed_hists]
        summary["hopper_sweep"][cond_name] = {
            "num_options": num_options, "delib_cost": delib_cost,
            "final_return_mean": float(np.mean([s["final_return"] for s in stats])),
            "final_return_sem": float(np.std([s["final_return"] for s in stats]) / np.sqrt(len(stats))),
            "auc_return_mean": float(np.mean([s["auc_return"] for s in stats])),
            "final_ep_len_mean": float(np.mean([s["final_ep_len"] for s in stats])),
            "final_decision_rate_mean": float(np.mean([s["final_decision_rate"] for s in stats])),
            "per_seed_final_return": [s["final_return"] for s in stats],
        }

    plot_curves(hopper_results, "PPOC on brax Hopper: primitives vs. options (deliberation-cost sweep)",
                RESULTS_DIR / "ppoc_hopper_sweep.png")

    print("\n=== Hopper sweep summary ===")
    print(f"{'condition':<32} {'final return':>14} {'AUC return':>12} {'ep len':>8} {'decision rate':>14}")
    for cond_name, s in summary["hopper_sweep"].items():
        print(f"{cond_name:<32} {s['final_return_mean']:>10.1f}±{s['final_return_sem']:<3.0f} "
              f"{s['auc_return_mean']:>12.1f} {s['final_ep_len_mean']:>8.1f} {s['final_decision_rate_mean']:>14.3f}")

    # -------------------------------------------------------------------
    # Cross-environment check: primitives vs. best options condition,
    # on two more brax locomotion tasks. "Best options condition" is picked
    # dynamically from the hopper sweep above (by final_return_mean, among
    # the non-primitive conditions), rather than hardcoded.
    # -------------------------------------------------------------------
    best_options_name = max(
        (name for name in summary["hopper_sweep"] if name != "Primitive actions (1 option)"),
        key=lambda name: summary["hopper_sweep"][name]["final_return_mean"],
    )
    best_opts = summary["hopper_sweep"][best_options_name]
    CROSS_CONDITIONS = [
        ("Primitive actions (1 option)", 1, 0.0),
        (f"Options ({best_opts['num_options']}), eta={best_opts['delib_cost']} [hopper-best]",
         best_opts["num_options"], best_opts["delib_cost"]),
    ]
    print(f"\n=== Cross-environment check: {CROSS_ENVS} "
          f"({len(CROSS_SEEDS)} seeds, {CROSS_ITERS} iters); best hopper options condition "
          f"was '{best_options_name}' ===", flush=True)
    for env_id in CROSS_ENVS:
        env_results = {}
        for cond_name, num_options, delib_cost in CROSS_CONDITIONS:
            seed_hists = []
            for seed in CROSS_SEEDS:
                t1 = time.time()
                res = run_condition(ppoc, env_id, num_options, delib_cost, seed, CROSS_ITERS)
                seed_hists.append(res["history"])
                last = res["history"][-1]
                print(f"  [{env_id}] {cond_name}, seed {seed}: final return={last[1]:.1f} "
                      f"ep_len={last[2]:.1f} ({time.time()-t1:.0f}s)", flush=True)
            env_results[cond_name] = seed_hists
            stats = [summarize(h) for h in seed_hists]
            summary["cross_env"].setdefault(env_id, {})[cond_name] = {
                "num_options": num_options, "delib_cost": delib_cost,
                "final_return_mean": float(np.mean([s["final_return"] for s in stats])),
                "final_return_sem": float(np.std([s["final_return"] for s in stats]) / np.sqrt(len(stats))),
                "per_seed_final_return": [s["final_return"] for s in stats],
            }
        plot_curves(env_results, f"PPOC on brax {env_id}: primitives vs. options",
                    RESULTS_DIR / f"ppoc_{env_id}_curves.png")

    print("\n=== Cross-environment summary ===")
    for env_id, conds in summary["cross_env"].items():
        print(f"  {env_id}:")
        for cond_name, s in conds.items():
            print(f"    {cond_name:<32} final return={s['final_return_mean']:.1f}±{s['final_return_sem']:.1f}")

    summary["runtime_sec"] = time.time() - t0
    with open(RESULTS_DIR / "ppoc_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary -> "
          f"{RESULTS_DIR / 'ppoc_verification_summary.json'}")


if __name__ == "__main__":
    main()
