"""
Option-Critic empirical verification.


The training loop here is on-policy (fresh rollouts, a few gradient epochs per
rollout, slow target-net tracking) rather than `option_critic.py`'s own
replay-buffer loop: Option-Critic's intra-option policy gradient is an
on-policy estimator (Bacon et al. use on-policy actor-critic updates), and the
repo's `option_critic_loss_fn` computes exactly that -- TD critic target +
intra-option PG + the termination gradient -- on whatever batch of
consecutive-step transitions it is handed, so feeding it fresh rollouts stays
faithful to the loss under test while matching the paper's own update regime.

Reproduces the Option-Critic paper's four-rooms experiments (Bacon, Harb &
Precup, AAAI 2017, Section 5.1):

  Test A -- non-stationary transfer (Figure 3): train to reach one goal, then
    relocate the goal to a different room and keep training. The flat baseline
    is the SAME code path with `num_options=1` (which collapses the option
    machinery to a plain advantage actor-critic).

  Test B -- option specialization (Figure 4): sweep every state through the
    trained multi-option network and measure how the options carve up the
    grid -- the greedy option per state should be spatially coherent
    (contiguous regions, not scatter) rather than uniform-random-looking.
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
from repo_loader import load_option_critic
import fourrooms_nav as frn

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = list(range(16))       
NUM_ENVS = 128
ROLLOUT = 32                    # env-steps per on-policy iteration
HIDDEN_DIM = 128
STEPS_PHASE_A = 1_000_000       # env-steps before the goal relocation
STEPS_PHASE_B = 2_000_000       # env-steps after the goal relocation (recovery)
GRAD_CLIP = 1.0
EPOCHS = 4
NUM_MINIBATCHES = 4
TARGET_TAU_PER_ITER = 0.05      # slow target-net tracking, applied once per iteration
LOG_EVERY = 8                   # iterations per logged point

BASE_CONFIG = {
    "epsilon": 0.1,             # epsilon-greedy over options (Q_Omega)
    "gamma": frn.GAMMA,
    "lr": 7e-4,
    # Bacon et al. 2017 (Sec. 3) subtract a small constant from the termination
    # advantage to stop options collapsing to one-step primitives; `delib_cost`
    # is exactly that knob in `option_critic.py`.
    "delib_cost": 0.03,
    "value_coef": 0.5,
    "entropy_coef": 0.02,
}

CONDITIONS = {
    "Flat actor-critic (1 option)": 1,
    "Option-Critic (4 options)": 4,
    "Option-Critic (8 options)": 8,
}


def polyak_update(online, target, tau):
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)


def build_agent(oc, num_options, obs_dim, key):
    net = oc.OptionCriticNetwork(
        num_options=num_options, num_actions=frn.NUM_ACTIONS, hidden_dim=HIDDEN_DIM
    )
    params = net.init(key, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    target_params = jax.tree.map(jnp.copy, params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP), optax.adam(BASE_CONFIG["lr"])
    )
    opt_state = optimizer.init(params)
    return net, params, target_params, optimizer, opt_state


def make_train_iter(oc, net, optimizer, config):
    """One jitted on-policy iteration: scan ROLLOUT env-steps with the real
    `batch_select_option_critic_action`, then EPOCHS x NUM_MINIBATCHES gradient
    steps on the freshly collected batch with the real `option_critic_loss_fn`.
    `goal_idx` is threaded as a traced value so relocating the goal doesn't
    recompile."""
    total = ROLLOUT * NUM_ENVS
    mb_size = total // NUM_MINIBATCHES

    def scan_step(carry, rng):
        params, env_state, obs, dones, options, ep_return, ep_len, goal_idx = carry

        rng, act_rng = jax.random.split(rng)
        act_keys = jax.random.split(act_rng, NUM_ENVS)
        actions, next_options, _, _, terminated, _ = oc.batch_select_option_critic_action(
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
            "obs": obs, "action": actions, "option": next_options,
            "reward": rewards, "done": next_dones, "next_obs": next_obs,
        }
        metrics = {
            "completed_return": completed_return, "completed_len": completed_len,
            "n_done": jnp.sum(next_dones), "ep_len_sum": jnp.sum(completed_len),
            "terminated": terminated,
        }
        carry = (params, next_env_state, next_obs, next_dones, next_options,
                 next_ep_return, next_ep_len, goal_idx)
        return carry, (trans, metrics)

    def loss_on(params, target_params, mb):
        class Batch:
            first = {"obs": mb["obs"], "action": mb["action"], "option": mb["option"],
                     "reward": mb["reward"], "done": mb["done"]}
            second = {"obs": mb["next_obs"]}
        return oc.option_critic_loss_fn(params, target_params, net, Batch, config)

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
        jnp.ones(NUM_ENVS, dtype=bool),                 # force an option decision on step 1
        jnp.full(NUM_ENVS, -1, dtype=jnp.int32),
        jnp.zeros(NUM_ENVS, dtype=jnp.float32),
        jnp.zeros(NUM_ENVS, dtype=jnp.int32),
        jnp.asarray(goal_idx, dtype=jnp.int32),
    )


def run_condition(oc, num_options, seed):
    key = jax.random.PRNGKey(seed)
    obs_dim = frn.STATE_DIM
    key, init_key = jax.random.split(key)
    net, params, target_params, optimizer, opt_state = build_agent(
        oc, num_options, obs_dim, init_key
    )

    config = dict(BASE_CONFIG)
    config["num_options"] = num_options
    config["num_actions"] = frn.NUM_ACTIONS
    train_iter = make_train_iter(oc, net, optimizer, config)

    goal_a = frn.CELL_TO_IDX[frn.GOAL_A]
    goal_b = frn.CELL_TO_IDX[frn.GOAL_B]

    def run_phase(carry, params, target_params, opt_state, env_steps, key, n_steps):
        history = []
        acc = {"ret_num": 0.0, "ret_den": 0.0, "len_sum": 0.0, "term": [], "n": 0}
        for it in range(n_steps // (ROLLOUT * NUM_ENVS)):
            key, k = jax.random.split(key)
            carry, params, target_params, opt_state, metrics = train_iter(
                carry, params, target_params, opt_state, k
            )
            m = jax.device_get(metrics)
            acc["ret_num"] += float(np.sum(m["completed_return"]))
            acc["ret_den"] += float(np.sum(m["n_done"]))
            acc["len_sum"] += float(np.sum(m["ep_len_sum"]))
            acc["term"].append(float(np.mean(m["terminated"])))
            acc["n"] += 1
            env_steps += ROLLOUT * NUM_ENVS
            if acc["n"] >= LOG_EVERY:
                den = max(acc["ret_den"], 1.0)
                history.append((env_steps, acc["ret_num"] / den,
                                acc["len_sum"] / den, float(np.mean(acc["term"]))))
                acc = {"ret_num": 0.0, "ret_den": 0.0, "len_sum": 0.0, "term": [], "n": 0}
        return carry, params, target_params, opt_state, env_steps, history, key

    key, k_reset, k_phase = jax.random.split(key, 3)
    carry = fresh_env_carry(goal_a, k_reset)
    carry, params, target_params, opt_state, steps, hist_a, k_phase = run_phase(
        carry, params, target_params, opt_state, 0, k_phase, STEPS_PHASE_A
    )

    # relocate the goal -- keep params / optimizer / target, re-reset the envs
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
# Test B -- option specialization
# ---------------------------------------------------------------------------
def option_specialization(oc, run_result):
    net = run_result["net"]
    params = run_result["params"]
    num_options = run_result["num_options"]

    all_obs = jnp.eye(frn.STATE_DIM, dtype=jnp.float32)
    q_w, beta_logits, action_logits = net.apply(params, all_obs)
    q_w = np.asarray(q_w)
    beta = np.asarray(jax.nn.sigmoid(beta_logits))                 # (S, O)
    pi = np.asarray(jax.nn.softmax(action_logits, axis=-1))        # (S, O, A)
    greedy_option = q_w.argmax(axis=1)                             # (S,)

    tv = []
    for i in range(num_options):
        for j in range(i + 1, num_options):
            tv.append(0.5 * np.abs(pi[:, i, :] - pi[:, j, :]).sum(axis=-1))
    inter_option_tv = float(np.mean(tv)) if tv else 0.0

    cell_to_idx = frn.CELL_TO_IDX

    def coherence(assignment):
        agree = []
        for s, (r, c) in enumerate(frn.FREE_CELLS):
            nbrs = [cell_to_idx[(r + dr, c + dc)]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if (r + dr, c + dc) in cell_to_idx]
            if not nbrs:
                continue
            vals, counts = np.unique(assignment[nbrs], return_counts=True)
            agree.append(float(assignment[s] == vals[counts.argmax()]))
        return float(np.mean(agree))

    spatial_coherence = coherence(greedy_option)
    rng = np.random.default_rng(0)
    random_coherence = float(np.mean([
        coherence(rng.integers(0, num_options, size=frn.STATE_DIM)) for _ in range(100)
    ]))

    doorway_beta = float(np.mean(beta[frn.DOORWAY_IDX, :]))
    interior_idx = [i for i in range(frn.STATE_DIM) if i not in frn.DOORWAY_IDX]
    interior_beta = float(np.mean(beta[interior_idx, :]))

    return {
        "inter_option_tv": inter_option_tv,
        "greedy_option_spatial_coherence": spatial_coherence,
        "random_option_spatial_coherence": random_coherence,
        "mean_beta_doorways": doorway_beta,
        "mean_beta_interior": interior_beta,
        "option_usage": [float(np.mean(greedy_option == o)) for o in range(num_options)],
    }, {"greedy_option": greedy_option, "beta": beta}


def _grid(values, fill=np.nan):
    g = np.full((frn.N, frn.N), fill, dtype=float)
    for s, (r, c) in enumerate(frn.FREE_CELLS):
        g[r, c] = values[s]
    return g


def plot_specialization(maps, num_options, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(_grid(maps["greedy_option"]), cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title(f"greedy option per state ({num_options} options)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(_grid(maps["beta"].mean(axis=1)), cmap="viridis")
    axes[1].set_title("mean termination prob. beta(s)")
    for (r, c) in frn.DOORWAYS:
        axes[1].plot(c, r, "rx")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _smooth(mat, split, w=5):
    """Moving average, applied to the pre- and post-switch segments separately
    so the relocation discontinuity isn't smeared across, with edge padding so
    the endpoints aren't dragged toward zero."""
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


def plot_transfer_curves(all_results, switch_step, path):
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
    axes[0].set_title("Four-rooms transfer: return (goal relocated at dashed line)")
    axes[1].set_ylabel("steps per episode")
    axes[1].set_title("Four-rooms transfer: steps per episode")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def recovery_stats(seed_hists, switch_step):
    ret = np.array([[h[1] for h in hist] for hist in seed_hists])
    xs = np.array([h[0] for h in seed_hists[0]])
    pre, post = xs <= switch_step, xs > switch_step
    auc = ret[:, post].mean(axis=1)              # per-seed post-switch area under curve
    fin = ret[:, post][:, -3:].mean(axis=1)      # per-seed return at the end of the window
    # mean return this many env-steps after the relocation (robust "how far
    # has it recovered by now" -- a single-crossing time-to-threshold is too
    # noisy on curves this wide)
    recovery_curve = {}
    for off in (250_000, 500_000, 1_000_000, 1_500_000, 2_000_000):
        idx = np.where(xs >= switch_step + off)[0]
        if idx.size:
            recovery_curve[f"return_at_+{off // 1000}k"] = float(ret[:, idx[0]].mean())
    return {
        "n_seeds": len(seed_hists),
        "pre_switch_final_return": float(ret[:, pre][:, -3:].mean()),
        "post_switch_auc_return": float(auc.mean()),
        "post_switch_auc_return_median": float(np.median(auc)),
        "post_switch_auc_return_sem": float(auc.std() / np.sqrt(len(auc))),
        "post_switch_final_return": float(fin.mean()),
        "post_switch_final_return_median": float(np.median(fin)),
        "n_seeds_recovered_final_gt_0.8": int((fin > 0.8).sum()),
        **recovery_curve,
    }


def main():
    t0 = time.time()
    oc = load_option_critic()
    switch_step = STEPS_PHASE_A

    print(f"=== Test A: four-rooms non-stationary transfer "
          f"(goal {frn.GOAL_A} -> {frn.GOAL_B} at {switch_step} env-steps, "
          f"{len(SEEDS)} seeds) ===", flush=True)

    all_results = {}
    diag_run = None
    for cond_name, num_options in CONDITIONS.items():
        seed_hists = []
        for seed in SEEDS:
            print(f"  -- {cond_name}, seed {seed} --", flush=True)
            res = run_condition(oc, num_options, seed)
            seed_hists.append(res["history"])
            last = res["history"][-1]
            print(f"     final: return={last[1]:.3f}  steps/ep={last[2]:.1f}  "
                  f"term_rate={last[3]:.3f}", flush=True)
            # keep the first seed of the (last) multi-option condition for Test B
            if num_options > 1 and seed == SEEDS[0] and (
                diag_run is None or num_options == 4
            ):
                diag_run = res
        all_results[cond_name] = seed_hists

    plot_transfer_curves(all_results, switch_step,
                         RESULTS_DIR / "option_critic_transfer_curves.png")

    summary = {
        "config": {
            "seeds": SEEDS, "num_envs": NUM_ENVS, "rollout": ROLLOUT,
            "steps_phase_a": STEPS_PHASE_A, "steps_phase_b": STEPS_PHASE_B,
            "goal_a": frn.GOAL_A,
            "goal_b": frn.GOAL_B, **BASE_CONFIG,
        },
        "transfer": {},
    }
    for cond_name, seed_hists in all_results.items():
        summary["transfer"][cond_name] = recovery_stats(seed_hists, switch_step)
        summary["transfer"][cond_name]["curve_env_steps"] = [h[0] for h in seed_hists[0]]
        summary["transfer"][cond_name]["return_per_seed"] = [
            [h[1] for h in hist] for hist in seed_hists
        ]
        s = summary["transfer"][cond_name]
        print(f"  {cond_name}: post_switch_auc={s['post_switch_auc_return']:.3f}  "
              f"return @+0.5M/+1M/+1.5M/final = {s.get('return_at_+500k', 0):.2f} / "
              f"{s.get('return_at_+1000k', 0):.2f} / {s.get('return_at_+1500k', 0):.2f} / "
              f"{s['post_switch_final_return']:.2f}")

    print("\n=== Test B: option specialization (trained 4-option policy) ===")
    spec, maps = option_specialization(oc, diag_run)
    plot_specialization(maps, diag_run["num_options"],
                        RESULTS_DIR / "option_critic_options.png")
    for k, v in spec.items():
        print(f"  {k}: {v}")
    summary["option_specialization"] = spec

    summary["runtime_sec"] = time.time() - t0
    with open(RESULTS_DIR / "option_critic_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary -> "
          f"{RESULTS_DIR / 'option_critic_verification_summary.json'}")


if __name__ == "__main__":
    main()
