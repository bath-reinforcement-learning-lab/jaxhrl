"""
MOC (Multi-updates Option-Critic) empirical verification.

Reproduces the central four-rooms experiment of "Flexible Option Learning"
(Klissarov & Precup, NeurIPS 2021), which names the method MOC (Multi-updates
Option Critic):

  Test A -- non-stationary four-rooms (paper Figure 1b): all option components
    learned from scratch, then the goal is relocated mid-training. The paper's
    claims are (i) MOC and OC both beat a flat actor-critic, (ii) MOC reaches
    OC's performance in about half the episodes, and (iii) MOC has lower
    seed-to-seed variance. Same task, same on-policy loop, same 16 seeds as
    `option_critic_verify.py`; the flat baseline is the shared code path with
    `num_options=1`.

  Test B -- option usage / diversity (paper Figures 1c and 6): the paper
    reports two opposite regimes -- in the tabular case (Figure 1c) full
    multi-updating (eta = 1) *reduces* option diversity / information radius,
    while in deep MiniGrid (Figure 6) MOC separates the state space better than
    OC's single-dominant-option solution. Measured here as the entropy of the
    greedy-option distribution over states, the dominant option's share, and
    the information radius (inter-option policy divergence).
"""
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
from repo_loader import load_moc, load_option_critic
import fourrooms_nav as frn

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = list(range(16))
NUM_ENVS = 128
ROLLOUT = 32
HIDDEN_DIM = 128
STEPS_PHASE_A = 1_000_000       # learn from scratch
STEPS_PHASE_B = 1_500_000       # after the goal relocation
GRAD_CLIP = 1.0
EPOCHS = 4
NUM_MINIBATCHES = 4
TARGET_TAU_PER_ITER = 0.05
LOG_EVERY = 8

BASE_CONFIG = {
    "epsilon": 0.1,
    "gamma": frn.GAMMA,
    "lr": 7e-4,
    "clip_eps": 0.2,           # PPO-style clip on MOC's importance ratio
    "delib_cost": 0.03,
    "value_coef": 0.5,
    "entropy_coef": 0.02,
}

# name -> (algorithm, num_options)
CONDITIONS = {
    "Flat actor-critic (1 option)": ("oc", 1),
    "Option-Critic (4 options)": ("oc", 4),
    "MOC (4 options)": ("moc", 4),
}


def polyak_update(online, target, tau):
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)


def build_agent(net_mod, num_options, obs_dim, key):
    net = net_mod.OptionCriticNetwork(
        num_options=num_options, num_actions=frn.NUM_ACTIONS, hidden_dim=HIDDEN_DIM
    )
    params = net.init(key, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    target_params = jax.tree.map(jnp.copy, params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP), optax.adam(BASE_CONFIG["lr"])
    )
    opt_state = optimizer.init(params)
    return net, params, target_params, optimizer, opt_state


def make_train_iter(net_mod, loss_fn, net, optimizer, config):
    """One jitted on-policy iteration: scan ROLLOUT env-steps with the real
    `batch_select_option_critic_action`, then EPOCHS x NUM_MINIBATCHES gradient
    steps on the freshly collected batch with `loss_fn` (the repo's real
    `option_critic_loss_fn` or `moc_loss_fn`). `goal_idx` is threaded as a
    traced value so relocating the goal doesn't recompile."""
    total = ROLLOUT * NUM_ENVS
    mb_size = total // NUM_MINIBATCHES

    def scan_step(carry, rng):
        params, env_state, obs, dones, options, ep_return, ep_len, goal_idx = carry

        rng, act_rng = jax.random.split(rng)
        act_keys = jax.random.split(act_rng, NUM_ENVS)
        actions, next_options, logps, _, terminated, _ = net_mod.batch_select_option_critic_action(
            act_keys, obs, options, dones, params, net, config
        )

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_env_state, rewards, next_dones, infos = jax.vmap(
            frn.step_fn, in_axes=(0, 0, 0, None)
        )(step_keys, env_state, actions, goal_idx)

        running = ep_return + rewards
        completed_return = jnp.where(next_dones, running, 0.0)
        next_ep_return = jnp.where(next_dones, 0.0, running)
        running_len = ep_len + 1
        completed_len = jnp.where(next_dones, running_len, 0)
        next_ep_len = jnp.where(next_dones, 0, running_len)

        trans = {
            "obs": obs, "action": actions, "option": next_options, "logp": logps,
            "reward": rewards, "done": next_dones, "next_obs": next_obs,
        }
        metrics = {
            "completed_return": completed_return, "completed_len": completed_len,
            "n_done": jnp.sum(next_dones), "ep_len_sum": jnp.sum(completed_len),
            "terminated": terminated, "options": next_options,
        }
        carry = (params, next_env_state, next_obs, next_dones, next_options,
                 next_ep_return, next_ep_len, goal_idx)
        return carry, (trans, metrics)

    def loss_on(params, target_params, mb):
        class Batch:
            first = {"obs": mb["obs"], "action": mb["action"], "option": mb["option"],
                     "logp": mb["logp"], "reward": mb["reward"], "done": mb["done"]}
            second = {"obs": mb["next_obs"]}
        return loss_fn(params, target_params, net, Batch, config)

    @jax.jit
    def train_iter(carry, params, target_params, opt_state, rng):
        roll_rng, epoch_rng = jax.random.split(rng)
        env_state, obs, dones, options, ep_return, ep_len, goal_idx = carry
        roll_carry = (params, env_state, obs, dones, options, ep_return, ep_len, goal_idx)
        roll_carry, (batch, metrics) = jax.lax.scan(
            scan_step, roll_carry, jax.random.split(roll_rng, ROLLOUT)
        )
        (_, env_state, obs, dones, options, ep_return, ep_len, goal_idx) = roll_carry
        carry = (env_state, obs, dones, options, ep_return, ep_len, goal_idx)
        flat = jax.tree_util.tree_map(lambda a: a.reshape((total,) + a.shape[2:]), batch)

        def epoch(state, ekey):
            params, opt_state = state
            perm = jax.random.permutation(ekey, total)

            def mb_step(state, i):
                params, opt_state = state
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb_size, mb_size)
                mb = jax.tree_util.tree_map(lambda a: a[idx], flat)
                (loss_val, aux), grads = jax.value_and_grad(loss_on, has_aux=True)(
                    params, target_params, mb
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss_val

            (params, opt_state), losses = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(NUM_MINIBATCHES)
            )
            return (params, opt_state), jnp.mean(losses)

        (params, opt_state), _ = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(epoch_rng, EPOCHS)
        )
        target_params = polyak_update(params, target_params, TARGET_TAU_PER_ITER)
        return carry, params, target_params, opt_state, metrics

    return train_iter


def fresh_env_carry(goal_idx, key):
    reset_keys = jax.random.split(key, NUM_ENVS)
    obs0, state0 = jax.vmap(frn.reset_fn, in_axes=(0, None))(reset_keys, goal_idx)
    return (
        state0, obs0,
        jnp.ones(NUM_ENVS, dtype=bool),
        jnp.full(NUM_ENVS, -1, dtype=jnp.int32),
        jnp.zeros(NUM_ENVS, dtype=jnp.float32),
        jnp.zeros(NUM_ENVS, dtype=jnp.int32),
        jnp.asarray(goal_idx, dtype=jnp.int32),
    )


def run_condition(net_mod, loss_fn, num_options, seed):
    key = jax.random.PRNGKey(seed)
    obs_dim = frn.STATE_DIM
    key, init_key = jax.random.split(key)
    net, params, target_params, optimizer, opt_state = build_agent(
        net_mod, num_options, obs_dim, init_key
    )

    config = dict(BASE_CONFIG)
    config["num_options"] = num_options
    config["num_actions"] = frn.NUM_ACTIONS
    train_iter = make_train_iter(net_mod, loss_fn, net, optimizer, config)

    goal_a = frn.CELL_TO_IDX[frn.GOAL_A]
    goal_b = frn.CELL_TO_IDX[frn.GOAL_B]

    def run_phase(carry, params, target_params, opt_state, env_steps, key, n_steps):
        history = []
        acc = {"ret_num": 0.0, "ret_den": 0.0, "len_sum": 0.0, "term": [], "opt": [], "n": 0}
        for _ in range(n_steps // (ROLLOUT * NUM_ENVS)):
            key, k = jax.random.split(key)
            carry, params, target_params, opt_state, metrics = train_iter(
                carry, params, target_params, opt_state, k
            )
            m = jax.device_get(metrics)
            acc["ret_num"] += float(np.sum(m["completed_return"]))
            acc["ret_den"] += float(np.sum(m["n_done"]))
            acc["len_sum"] += float(np.sum(m["ep_len_sum"]))
            acc["term"].append(float(np.mean(m["terminated"])))
            acc["opt"].append(np.bincount(np.asarray(m["options"]).ravel(), minlength=num_options))
            acc["n"] += 1
            env_steps += ROLLOUT * NUM_ENVS
            if acc["n"] >= LOG_EVERY:
                den = max(acc["ret_den"], 1.0)
                opt_counts = np.sum(acc["opt"], axis=0).astype(float)
                opt_frac = opt_counts / max(opt_counts.sum(), 1.0)
                usage_entropy = float(-np.sum(opt_frac * np.log(opt_frac + 1e-12)) / np.log(num_options)) \
                    if num_options > 1 else 0.0
                history.append((env_steps, acc["ret_num"] / den,
                                acc["len_sum"] / den, float(np.mean(acc["term"])),
                                usage_entropy))
                acc = {"ret_num": 0.0, "ret_den": 0.0, "len_sum": 0.0, "term": [], "opt": [], "n": 0}
        return carry, params, target_params, opt_state, env_steps, history, key

    key, k_reset, k_phase = jax.random.split(key, 3)
    carry = fresh_env_carry(goal_a, k_reset)
    carry, params, target_params, opt_state, steps, hist_a, k_phase = run_phase(
        carry, params, target_params, opt_state, 0, k_phase, STEPS_PHASE_A
    )

    key, k_reset = jax.random.split(key)
    carry = fresh_env_carry(goal_b, k_reset)
    carry, params, target_params, opt_state, steps, hist_b, k_phase = run_phase(
        carry, params, target_params, opt_state, steps, k_phase, STEPS_PHASE_B
    )

    return {
        "history": hist_a + hist_b,
        "params": jax.device_get(params),
        "net": net,
        "num_options": num_options,
    }


# ---------------------------------------------------------------------------
# Test B -- option usage / diversity
# ---------------------------------------------------------------------------
def option_usage_diagnostic(run_result):
    net = run_result["net"]
    params = run_result["params"]
    num_options = run_result["num_options"]

    all_obs = jnp.eye(frn.STATE_DIM, dtype=jnp.float32)
    q_w, _, action_logits = net.apply(params, all_obs)
    q_w = np.asarray(q_w)
    pi = np.asarray(jax.nn.softmax(action_logits, axis=-1))        # (S, O, A)
    greedy_option = q_w.argmax(axis=1)

    frac = np.array([np.mean(greedy_option == o) for o in range(num_options)])
    usage_entropy = float(-np.sum(frac * np.log(frac + 1e-12)) / np.log(num_options))
    dominant_option_share = float(frac.max())

    # pairwise total-variation distance between option policies, over states
    tv = []
    for i in range(num_options):
        for j in range(i + 1, num_options):
            tv.append(0.5 * np.abs(pi[:, i, :] - pi[:, j, :]).sum(axis=-1))
    inter_option_tv = float(np.mean(tv))

    # information radius (generalised Jensen-Shannon divergence) between the
    # option policies, averaged over states -- the paper's Figure 1c diversity
    # measure
    mixture = pi.mean(axis=1)                                      # (S, A)
    def _ent(p):
        return -np.sum(p * np.log(p + 1e-12), axis=-1)
    info_radius = float(np.mean(_ent(mixture) - _ent(pi).mean(axis=1)))

    return {
        "option_usage_entropy": usage_entropy,
        "dominant_option_share": dominant_option_share,
        "option_usage_fractions": frac.tolist(),
        "inter_option_tv": inter_option_tv,
        "information_radius": info_radius,
    }


# ---------------------------------------------------------------------------
def _smooth(mat, split, w=5):
    out = np.empty_like(mat, dtype=float)
    k = np.ones(w) / w
    p = w // 2
    for a, b in ((0, split), (split, mat.shape[1])):
        seg = mat[:, a:b]
        if seg.shape[1] < w:
            out[:, a:b] = seg
        else:
            padded = np.pad(seg, ((0, 0), (p, p)), mode="edge")
            out[:, a:b] = np.array([np.convolve(r, k, mode="valid") for r in padded])
    return out


def plot_curves(all_results, switch_step, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cond_name, seed_hists in all_results.items():
        xs = np.array([h[0] for h in seed_hists[0]])
        split = int(np.sum(xs <= switch_step))
        ret = _smooth(np.array([[h[1] for h in hist] for hist in seed_hists]), split)
        lens = _smooth(np.array([[h[2] for h in hist] for hist in seed_hists]), split)
        for ax, data in ((axes[0], ret), (axes[1], lens)):
            m, se = data.mean(axis=0), data.std(axis=0) / np.sqrt(data.shape[0])
            ax.plot(xs, m, label=cond_name)
            ax.fill_between(xs, m - se, m + se, alpha=0.18)
    for ax in axes:
        ax.axvline(switch_step, color="k", ls="--", lw=1)
        ax.set_xlabel("env steps")
    axes[0].set_ylabel("episode return (= goal-reach rate)")
    axes[0].set_title("Four-rooms from scratch + relocation (dashed line)")
    axes[1].set_ylabel("steps per episode")
    axes[1].set_title("Four-rooms: steps per episode")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def stats(seed_hists, switch_step):
    ret = np.array([[h[1] for h in hist] for hist in seed_hists])
    xs = np.array([h[0] for h in seed_hists[0]])
    pre, post = xs <= switch_step, xs > switch_step
    pre_auc = ret[:, pre].mean(axis=1)          # sample efficiency on the initial task
    post_auc = ret[:, post].mean(axis=1)        # recovery after the goal moves
    fin = ret[:, post][:, -3:].mean(axis=1)

    def _to(mask, thr):
        # env-steps within the segment to first reach `thr` mean return, per seed
        out = []
        seg_x = xs[mask]
        for row in ret[:, mask]:
            hit = np.where(row >= thr)[0]
            out.append(float(seg_x[hit[0]] - (0 if mask is pre else switch_step)) if hit.size else np.nan)
        return out

    pre_to_08 = _to(pre, 0.8)
    return {
        "n_seeds": len(seed_hists),
        "pre_switch_auc_return": float(pre_auc.mean()),
        "pre_switch_auc_return_sem": float(pre_auc.std() / np.sqrt(len(pre_auc))),
        "pre_switch_steps_to_0.8_mean": float(np.nanmean(pre_to_08)),
        "pre_switch_seeds_reaching_0.8": int(np.sum(~np.isnan(pre_to_08))),
        "post_switch_auc_return": float(post_auc.mean()),
        "post_switch_auc_return_sem": float(post_auc.std() / np.sqrt(len(post_auc))),
        "post_switch_final_return": float(fin.mean()),
        "post_switch_final_return_std": float(fin.std()),
        "n_seeds_recovered_final_gt_0.8": int((fin > 0.8).sum()),
    }


def main():
    t0 = time.time()
    moc = load_moc()
    oc = load_option_critic()
    loss_fns = {"moc": moc.moc_loss_fn, "oc": oc.option_critic_loss_fn}
    switch_step = STEPS_PHASE_A

    print(f"=== Test A: four-rooms from scratch + relocation "
          f"(goal {frn.GOAL_A} -> {frn.GOAL_B} at {switch_step} env-steps, "
          f"{len(SEEDS)} seeds) ===", flush=True)

    all_results = {}
    diag = {}
    for cond_name, (algo, num_options) in CONDITIONS.items():
        seed_hists = []
        for seed in SEEDS:
            print(f"  -- {cond_name}, seed {seed} --", flush=True)
            res = run_condition(moc, loss_fns[algo], num_options, seed)
            seed_hists.append(res["history"])
            last = res["history"][-1]
            print(f"     final: return={last[1]:.3f}  steps/ep={last[2]:.1f}  "
                  f"usage_entropy={last[4]:.3f}", flush=True)
            if seed == SEEDS[0] and num_options > 1:
                diag[cond_name] = res
        all_results[cond_name] = seed_hists

    plot_curves(all_results, switch_step, RESULTS_DIR / "moc_transfer_curves.png")

    summary = {
        "config": {
            "seeds": SEEDS, "num_envs": NUM_ENVS, "rollout": ROLLOUT,
            "steps_phase_a": STEPS_PHASE_A, "steps_phase_b": STEPS_PHASE_B,
            "goal_a": frn.GOAL_A, "goal_b": frn.GOAL_B, **BASE_CONFIG,
        },
        "transfer": {},
        "option_usage": {},
    }
    for cond_name, seed_hists in all_results.items():
        s = stats(seed_hists, switch_step)
        s["curve_env_steps"] = [h[0] for h in seed_hists[0]]
        s["return_per_seed"] = [[h[1] for h in hist] for hist in seed_hists]
        summary["transfer"][cond_name] = s
        print(f"  {cond_name}: pre_auc={s['pre_switch_auc_return']:.3f}  "
              f"post_auc={s['post_switch_auc_return']:.3f}  "
              f"final={s['post_switch_final_return']:.3f}+/-{s['post_switch_final_return_std']:.3f}  "
              f"recovered={s['n_seeds_recovered_final_gt_0.8']}/{s['n_seeds']}")

    print("\n=== Test B: option usage / diversity (seed-0 trained policies) ===")
    for cond_name, res in diag.items():
        d = option_usage_diagnostic(res)
        summary["option_usage"][cond_name] = d
        print(f"  {cond_name}: usage_entropy={d['option_usage_entropy']:.3f}  "
              f"dominant_share={d['dominant_option_share']:.3f}  "
              f"inter_option_tv={d['inter_option_tv']:.3f}  "
              f"info_radius={d['information_radius']:.4f}")

    summary["runtime_sec"] = time.time() - t0
    with open(RESULTS_DIR / "moc_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary -> "
          f"{RESULTS_DIR / 'moc_verification_summary.json'}")


if __name__ == "__main__":
    main()
