"""
DCEO paper-fidelity verification.

Uses the ACTUAL classes/functions imported straight from jaxhrl/DCEO.py
(LaplacianRepresentationNetwork, laplacian_loss_fn, OptionQNetwork, q_loss_fn,
select_dceo_action) -- nothing here is a reimplementation of the algorithm.

Three checks, matching "Deep Laplacian-based Options for Temporally-Extended
Exploration" (Klissarov & Machado, 2020):

  1. Eigenvector recovery: train laplacian_net on FourRooms (104 states, exact
     ground-truth Laplacian eigenvectors available via eigh) and check the
     learned representation matches the true smallest-eigenvalue eigenvectors
     up to sign, via best-matched cosine similarity.
  2. Orthogonality ablation: beta=0 should reproduce the known failure mode
     where all learned directions collapse onto the dominant eigenvector.
  3. Option behavior: train OptionQNetwork + run select_dceo_action against
     the (now-verified) frozen Laplacian representation, and confirm each
     option's greedy rollouts actually drive the agent toward the extreme
     (max for sign=+1, min for sign=-1) of its assigned eigenvector.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from repo_loader import load_dceo
import fourrooms as fr

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 0
NUM_EIGENVECTORS = 4
LATENT_DIM = 64
LAP_STEPS = 6000
LAP_BATCH = 256
LAP_LR = 3e-4


def polyak_update(online, target, tau):
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)


def make_all_obs():
    return jnp.stack([fr.one_hot_obs(i) for i in range(fr.NUM_STATES)])


def train_laplacian(dceo, beta, key, log_every=500):
    laplacian_net = dceo.LaplacianRepresentationNetwork(latent_dim=LATENT_DIM, rep_dim=NUM_EIGENVECTORS)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, fr.NUM_STATES), dtype=jnp.float32)
    params = laplacian_net.init(init_key, dummy_obs)
    opt = optax.adam(LAP_LR)
    opt_state = opt.init(params)

    env = fr.FourRoomsEnv(horizon=10_000, rng=np.random.default_rng(SEED))
    walk = env.random_walk(200_000)  # long single random walk -> consecutive-pair pool
    all_obs = np.stack([fr.one_hot_obs(i) for i in range(fr.NUM_STATES)]).astype(np.float32)

    rng = np.random.default_rng(SEED + 1)
    n_pairs = len(walk) - 1

    @jax.jit
    def train_step(params, opt_state, obs_a, obs_b, obs_i, obs_j):
        (loss, aux), grads = jax.value_and_grad(dceo.laplacian_loss_fn, has_aux=True)(
            params, laplacian_net, obs_a, obs_b, obs_i, obs_j, beta
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    losses = []
    for step in range(LAP_STEPS):
        pair_idx = rng.integers(0, n_pairs, size=LAP_BATCH)
        obs_a = jnp.asarray(all_obs[walk[pair_idx]])
        obs_b = jnp.asarray(all_obs[walk[pair_idx + 1]])
        idx_i = rng.integers(0, fr.NUM_STATES, size=LAP_BATCH)
        idx_j = rng.integers(0, fr.NUM_STATES, size=LAP_BATCH)
        obs_i = jnp.asarray(all_obs[idx_i])
        obs_j = jnp.asarray(all_obs[idx_j])

        params, opt_state, loss, aux = train_step(params, opt_state, obs_a, obs_b, obs_i, obs_j)
        losses.append(float(loss))
        if step % log_every == 0 or step == LAP_STEPS - 1:
            print(f"    [beta={beta}] step {step:5d}  loss={float(loss):.4f}  "
                  f"attractive={float(aux['lap_attractive']):.4f}  ortho={float(aux['lap_orthogonality']):.4f}")

    phi_all = laplacian_net.apply(params, jnp.asarray(all_obs))  # (NUM_STATES, num_eig)
    return params, laplacian_net, np.asarray(phi_all), losses


def best_match(learned, true):
    """Hungarian-match learned columns to true columns on |cosine similarity|,
    return (matched_sim[num_eig], sign[num_eig], perm[num_eig])."""
    learned_n = learned / (np.linalg.norm(learned, axis=0, keepdims=True) + 1e-12)
    true_n = true / (np.linalg.norm(true, axis=0, keepdims=True) + 1e-12)
    cos = learned_n.T @ true_n  # (learned_k, true_k)
    cost = -np.abs(cos)
    row_ind, col_ind = linear_sum_assignment(cost)
    sims = np.abs(cos[row_ind, col_ind])
    signs = np.sign(cos[row_ind, col_ind])
    return sims, signs, col_ind, row_ind


def collapse_score(learned):
    """Average |cosine similarity| between distinct learned columns.
    ~1.0 = fully collapsed onto one direction, ~0 = well-spread/orthogonal."""
    n = learned / (np.linalg.norm(learned, axis=0, keepdims=True) + 1e-12)
    gram = np.abs(n.T @ n)
    k = gram.shape[0]
    off_diag = gram[~np.eye(k, dtype=bool)]
    return float(off_diag.mean())


def plot_eigenvector_grid(true_vecs, learned_vecs, signs, perm, out_path):
    k = true_vecs.shape[1]
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6))
    grid_true = np.full((fr.N, fr.N), np.nan)
    for eig_i in range(k):
        vals_true = true_vecs[:, eig_i]
        vals_learned = learned_vecs[:, perm[eig_i]] * signs[eig_i]
        gt = grid_true.copy()
        gl = grid_true.copy()
        for state_idx, (r, c) in enumerate(fr.FREE_CELLS):
            gt[r, c] = vals_true[state_idx]
            gl[r, c] = vals_learned[state_idx]
        axes[0, eig_i].imshow(gt, cmap="RdBu")
        axes[0, eig_i].set_title(f"true eig #{eig_i}")
        axes[0, eig_i].axis("off")
        axes[1, eig_i].imshow(gl, cmap="RdBu")
        axes[1, eig_i].set_title(f"learned (matched)")
        axes[1, eig_i].axis("off")
    fig.suptitle("DCEO Laplacian representation vs. true graph-Laplacian eigenvectors (FourRooms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


import flax.linen as nn


class OracleLaplacianNet(nn.Module):
    """Not repo code -- a diagnostic stand-in that returns the exact true
    graph-Laplacian eigenvectors via table lookup (obs is one-hot, so
    obs @ table selects the matching row). Used only to check whether the
    downstream option-learning code (q_loss_fn / select_dceo_action /
    OptionQNetwork, all real repo code) behaves correctly *given* a valid
    Laplacian representation -- isolating that from whether laplacian_loss_fn
    itself can learn one."""
    true_vecs: np.ndarray

    @nn.compact
    def __call__(self, x):
        table = self.variable("consts", "table", lambda: jnp.asarray(self.true_vecs, dtype=jnp.float32))
        return x @ table.value


def run_option_behavior_check(dceo, laplacian_params, laplacian_net, key):
    """End-to-end check: train OptionQNetwork with the ACTUAL q_loss_fn /
    select_dceo_action from the repo against the frozen, already-verified
    Laplacian representation, then confirm each option seeks the extreme of
    its assigned eigenvector."""
    num_options = 2 * NUM_EIGENVECTORS
    num_actions = 4
    config = {
        "D": 20, "epsilon": 0.15, "num_options": num_options, "num_actions": num_actions,
        "gamma": 0.98,
    }
    q_net = dceo.OptionQNetwork(num_actions=num_actions, num_options=num_options, latent_dim=LATENT_DIM)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, fr.NUM_STATES), dtype=jnp.float32)
    dummy_opt = jnp.zeros((1,), dtype=jnp.int32)
    q_params = q_net.init(init_key, dummy_obs, dummy_opt)
    target_q_params = jax.tree.map(jnp.copy, q_params)
    opt = optax.adam(5e-4)
    opt_state = opt.init(q_params)

    @jax.jit
    def train_step(q_params, target_q_params, opt_state, batch_first, batch_second):
        class Batch:
            first = batch_first
            second = batch_second
        (loss, aux), grads = jax.value_and_grad(dceo.q_loss_fn, has_aux=True)(
            q_params, q_net, target_q_params, laplacian_params, laplacian_net, Batch, config
        )
        updates, opt_state = opt.update(grads, opt_state, q_params)
        q_params = optax.apply_updates(q_params, updates)
        target_q_params = polyak_update(q_params, target_q_params, 0.01)
        return q_params, target_q_params, opt_state, loss

    select_action_jit = jax.jit(
        lambda key, obs, done, tau, option, q_params, mu: dceo.select_dceo_action(
            key, obs, done, tau, option, q_params, q_net, config, mu
        )
    )

    env = fr.FourRoomsEnv(horizon=1_000_000, rng=np.random.default_rng(SEED + 2))
    obs, idx = env.reset()
    obs = jnp.asarray(obs)
    done = jnp.array(False)
    tau = jnp.array(True)
    option = jnp.array(-1, dtype=jnp.int32)
    buffer = []  # list of dicts: obs, action, reward, done, option

    n_train_steps = 12_000
    batch_size = 128
    rng = np.random.default_rng(SEED + 3)
    losses = []
    for step in range(n_train_steps):
        key, act_key = jax.random.split(key)
        action, next_option, next_tau, _ = select_action_jit(
            act_key, obs, done, tau, option, q_params, jnp.asarray(0.9)
        )
        action = int(action)
        next_obs_np, next_idx, reward, done_np = env.step(action)

        buffer.append({
            "obs": np.asarray(obs), "action": action, "reward": reward,
            "done": done_np, "option": int(option),
        })

        obs = jnp.asarray(next_obs_np)
        done = jnp.array(done_np)
        tau = next_tau
        option = next_option

        if len(buffer) > batch_size + 1:
            idxs = rng.integers(0, len(buffer) - 1, size=batch_size)
            batch_first = {
                "obs": jnp.asarray(np.stack([buffer[i]["obs"] for i in idxs])),
                "action": jnp.asarray(np.array([buffer[i]["action"] for i in idxs]), dtype=jnp.int32),
                "reward": jnp.asarray(np.array([buffer[i]["reward"] for i in idxs]), dtype=jnp.float32),
                "done": jnp.asarray(np.array([buffer[i]["done"] for i in idxs]), dtype=jnp.float32),
                "option": jnp.asarray(np.array([buffer[i]["option"] for i in idxs]), dtype=jnp.int32),
            }
            batch_second = {
                "obs": jnp.asarray(np.stack([buffer[i + 1]["obs"] for i in idxs])),
            }
            q_params, target_q_params, opt_state, loss = train_step(
                q_params, target_q_params, opt_state, batch_first, batch_second
            )
            losses.append(float(loss))
            if len(buffer) > 60_000:
                buffer = buffer[-40_000:]
        if step % 2000 == 0:
            recent = np.mean(losses[-200:]) if losses else float("nan")
            print(f"    [options] step {step:6d}  q_loss(last200)={recent:.4f}")

    # ---- Evaluate: does each option's greedy rollout seek the extreme of its eigenvector? ----
    all_obs = make_all_obs()
    phi_all = np.asarray(laplacian_net.apply(laplacian_params, all_obs))  # (NUM_STATES, num_eig)

    n_starts = 12
    rollout_len = 40
    rng2 = np.random.default_rng(SEED + 4)
    results = {}
    for option_idx in range(num_options):
        eig_idx = option_idx // 2
        sign = 1.0 if option_idx % 2 == 0 else -1.0
        deltas = []
        for _ in range(n_starts):
            env_eval = fr.FourRoomsEnv(horizon=1_000_000, rng=np.random.default_rng(int(rng2.integers(0, 1e9))))
            obs0, idx0 = env_eval.reset()
            phi_start = phi_all[idx0, eig_idx]
            cur_obs = jnp.asarray(obs0)
            cur_idx = idx0
            for t in range(rollout_len):
                a = dceo.sample_option_policy(q_params, q_net, cur_obs, option_idx, jax.random.PRNGKey(0), num_actions)
                _, cur_idx, _, _ = env_eval.step(int(a))
                cur_obs = jnp.asarray(fr.one_hot_obs(cur_idx))
            phi_end = phi_all[cur_idx, eig_idx]
            deltas.append(sign * (phi_end - phi_start))
        results[option_idx] = {
            "eig_idx": int(eig_idx), "sign": sign,
            "mean_signed_delta": float(np.mean(deltas)),
            "frac_positive": float(np.mean(np.array(deltas) > 0)),
        }
        print(f"    option {option_idx} (eig {eig_idx}, sign {sign:+.0f}): "
              f"mean signed delta={results[option_idx]['mean_signed_delta']:.4f}  "
              f"frac_positive={results[option_idx]['frac_positive']:.2f}")

    return results, losses


def main():
    t0 = time.time()
    dceo = load_dceo()
    key = jax.random.PRNGKey(SEED)
    true_vecs, true_vals = fr.true_eigenvectors(NUM_EIGENVECTORS)

    print("=== [1/3] Training Laplacian representation (beta=1.0, orthogonality on) ===")
    key, k1 = jax.random.split(key)
    params_beta1, laplacian_net, phi_beta1, losses_beta1 = train_laplacian(dceo, beta=1.0, key=k1)

    print("=== [2/3] Ablation: training with beta=0.0 (orthogonality off) ===")
    key, k2 = jax.random.split(key)
    params_beta0, _, phi_beta0, losses_beta0 = train_laplacian(dceo, beta=0.0, key=k2)

    sims1, signs1, perm1, row1 = best_match(phi_beta1, true_vecs)
    sims0, signs0, perm0, row0 = best_match(phi_beta0, true_vecs)
    collapse1 = collapse_score(phi_beta1)
    collapse0 = collapse_score(phi_beta0)

    print(f"\nbeta=1.0: matched cosine sims = {np.round(sims1, 3)}  mean={sims1.mean():.3f}  collapse={collapse1:.3f}")
    print(f"beta=0.0: matched cosine sims = {np.round(sims0, 3)}  mean={sims0.mean():.3f}  collapse={collapse0:.3f}")

    plot_eigenvector_grid(true_vecs, phi_beta1, signs1, perm1, RESULTS_DIR / "dceo_eigenvectors_beta1.png")
    plot_eigenvector_grid(true_vecs, phi_beta0, signs0, perm0, RESULTS_DIR / "dceo_eigenvectors_beta0_ablation.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(losses_beta1, label="beta=1.0", alpha=0.8)
    ax[0].plot(losses_beta0, label="beta=0.0", alpha=0.8)
    ax[0].set_title("Laplacian loss")
    ax[0].set_xlabel("step")
    ax[0].legend()
    gram1 = np.abs((phi_beta1 / np.linalg.norm(phi_beta1, axis=0, keepdims=True)).T @
                   (phi_beta1 / np.linalg.norm(phi_beta1, axis=0, keepdims=True)))
    gram0 = np.abs((phi_beta0 / np.linalg.norm(phi_beta0, axis=0, keepdims=True)).T @
                   (phi_beta0 / np.linalg.norm(phi_beta0, axis=0, keepdims=True)))
    im = ax[1].imshow(np.concatenate([gram1, np.ones((NUM_EIGENVECTORS, 1)), gram0], axis=1), cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title("|cosine| Gram: beta=1.0 (left) vs beta=0.0 (right)")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dceo_loss_and_collapse.png", dpi=130)
    plt.close(fig)

    print("\n=== [3/3] Training OptionQNetwork with select_dceo_action/q_loss_fn ===")
    print("--- 3a: against the LEARNED (trained) Laplacian representation ---")
    key, k3 = jax.random.split(key)
    option_results_learned, _ = run_option_behavior_check(dceo, params_beta1, laplacian_net, k3)
    frac_correct_learned = np.mean([1.0 if r["mean_signed_delta"] > 0 else 0.0 for r in option_results_learned.values()])

    print("--- 3b: diagnostic -- against an ORACLE representation (true eigenvectors by table lookup, "
          "not repo code) to check whether q_loss_fn/select_dceo_action/OptionQNetwork behave correctly "
          "*given* a valid representation, independent of whether laplacian_loss_fn can learn one ---")
    oracle_net = OracleLaplacianNet(true_vecs=true_vecs)
    oracle_params = oracle_net.init(jax.random.PRNGKey(0), jnp.zeros((1, fr.NUM_STATES), jnp.float32))
    key, k3b = jax.random.split(key)
    option_results_oracle, _ = run_option_behavior_check(dceo, oracle_params, oracle_net, k3b)
    frac_correct_oracle = np.mean([1.0 if r["mean_signed_delta"] > 0 else 0.0 for r in option_results_oracle.values()])

    option_results = option_results_learned
    frac_correct = frac_correct_learned

    summary = {
        "eigenvector_recovery": {
            "true_eigenvalues": true_vals.tolist(),
            "beta1_matched_cosine_sim": sims1.tolist(),
            "beta1_mean_matched_cosine_sim": float(sims1.mean()),
            "beta1_collapse_score": collapse1,
            "beta0_matched_cosine_sim": sims0.tolist(),
            "beta0_mean_matched_cosine_sim": float(sims0.mean()),
            "beta0_collapse_score": collapse0,
            "beta1_final_loss": losses_beta1[-1],
            "beta0_final_loss": losses_beta0[-1],
        },
        "option_behavior_learned_representation": option_results_learned,
        "option_behavior_learned_frac_correct": float(frac_correct_learned),
        "option_behavior_oracle_representation_diagnostic": option_results_oracle,
        "option_behavior_oracle_frac_correct": float(frac_correct_oracle),
        "runtime_sec": time.time() - t0,
    }
    with open(RESULTS_DIR / "dceo_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary written to {RESULTS_DIR / 'dceo_verification_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
