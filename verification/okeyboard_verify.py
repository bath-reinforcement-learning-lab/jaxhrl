"""
Option Keyboard paper-fidelity verification.

Uses the ACTUAL Agent / sf_loss / reward_loss / GRUCell / RecurrentEncoder /
SFHead / TerminationHead classes imported straight from
jaxhrl/option_keyboard.py -- the successor-feature network and its TD loss
are not reimplemented here.

Reproduces the central empirical claim of Barreto et al. 2019, "The Option
Keyboard: Combining Skills in Reinforcement Learning" (NeurIPS), using their
own "Foraging World" domain (foragingworld.py) and their own worked example
(Appendix E.1, Scenario A2): pretrain successor features for the paper's
basis cumulant set W0 = {(1,0), (0,1)} (one skill per nutrient) ONLY, then
check whether Generalised Policy Evaluation/Improvement (Eq. 6-7 in the
paper) can combine those two skills, with *no further learning*, into
optimal behavior for a novel weight vector -- the paper's own headline case
is w=(1,-1) ("seek nutrient 1, avoid nutrient 2"), which is never trained on
directly but is exactly recoverable as a linear combination of the two
trained cumulants.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from repo_loader import load_option_keyboard
import foragingworld as fw

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 0
HIDDEN_DIM = 64
H_DIM = 32
GAMMA = 0.95
NUM_ENVS = 256             # parallel environments, matching the repo's own vmapped training design
N_MACRO_STEPS = 180_000    # each macro-step advances all NUM_ENVS envs by one step -> ~46M total env-steps
N_MACRO_SAMPLES = 4        # how many past macro-steps to draw per training minibatch (batch size = this * NUM_ENVS)
BUFFER_CAP_MACRO = 2_000   # capacity in macro-steps (-> BUFFER_CAP_MACRO * NUM_ENVS transitions)
TARGET_SYNC_EVERY = 200

# Barreto et al.'s own basis set W0 = {(1,0), (0,1)} -- one option per nutrient.
W_OPTIONS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
NUM_OPTIONS = 2


class MacroStepBuffer:
    """Ring buffer of whole macro-step snapshots (each holding NUM_ENVS
    transitions at once, since all envs step in lockstep). Sampling a
    training minibatch means picking a handful of past macro-step indices
    and concatenating all NUM_ENVS transitions from each -- this keeps the
    gather fully vectorized (no per-transition Python loop) while still
    drawing from a wide window of past experience for decorrelation."""

    def __init__(self, cap=BUFFER_CAP_MACRO):
        self.cap = cap
        self.data = []

    def add(self, snapshot):
        self.data.append(snapshot)
        if len(self.data) > self.cap:
            self.data.pop(0)

    def sample(self, rng, n_macro_samples):
        idxs = rng.integers(0, len(self.data) - 1, size=n_macro_samples)
        keys = self.data[0].keys()
        first = {k: np.concatenate([self.data[i][k] for i in idxs], axis=0) for k in keys}
        second = {k: np.concatenate([self.data[i + 1][k] for i in idxs], axis=0) for k in keys}
        return first, second


def gpi_action(ok, agent, carry_h, prev_action, obs, w):
    """Verbatim arithmetic from option_keyboard.py's scan_body (the repo's
    own GPI action-selection), factored out so it can be reused for
    evaluation with arbitrary (including untrained) w vectors."""
    psi_all, _beta, next_carry_h = agent(carry_h, prev_action, obs)  # (batch, options, actions, cumulant_dim)
    q_all = jnp.einsum('boad,d->boa', psi_all, w)  # (batch, options, actions)
    best_option = jnp.argmax(jnp.max(q_all, axis=-1), axis=-1)  # (batch,)
    batch_idx = jnp.arange(obs.shape[0])
    action = jnp.argmax(q_all[batch_idx, best_option], axis=-1)
    return action, next_carry_h


def single_option_action(ok, agent, carry_h, prev_action, obs, option_idx):
    """'Play one key' baseline: greedy action under a single pretrained
    option's own SF-derived Q, ignoring GPI combination entirely."""
    psi_all, _beta, next_carry_h = agent(carry_h, prev_action, obs)
    w = jnp.asarray(W_OPTIONS[option_idx])
    q = jnp.einsum('bad,d->ba', psi_all[:, option_idx], w)
    action = jnp.argmax(q, axis=-1)
    return action, next_carry_h


def train(ok, key, num_envs=NUM_ENVS, n_macro_steps=N_MACRO_STEPS, eps_min=1.0):
    """Runs `num_envs` parallel Foraging World environments (numpy-vectorized
    stepping) so each jitted action/train call processes a full batch at
    once -- this is both far higher-throughput than stepping one env at a
    time in a Python loop, and closer to how the repo's own __main__ script
    actually trains (via jax.vmap over many parallel envs)."""
    agent = ok.Agent(fw.STATE_DIM, HIDDEN_DIM, H_DIM, fw.NUM_ACTIONS, fw.CUMULANT_DIM, NUM_OPTIONS,
                      rngs=nnx.Rngs(key))
    target_agent = ok.Agent(fw.STATE_DIM, HIDDEN_DIM, H_DIM, fw.NUM_ACTIONS, fw.CUMULANT_DIM, NUM_OPTIONS,
                             rngs=nnx.Rngs(jax.random.fold_in(key, 1)))
    nnx.update(target_agent, jax.tree.map(jnp.copy, nnx.state(agent)))
    opt = nnx.Optimizer(agent, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)), wrt=nnx.Param)

    W = jnp.asarray(W_OPTIONS)

    @nnx.jit
    def train_step(agent, target_agent, opt, carry_b, next_carry_b, prev_action_b, obs_b, action_b,
                    phi_b, next_obs_b, done_b):
        def loss_fn(model):
            return ok.sf_loss(model, target_agent, W, carry_b, next_carry_b, prev_action_b,
                               obs_b, action_b, phi_b, next_obs_b, done_b, GAMMA)
        grads = nnx.grad(loss_fn)(agent)
        opt.update(agent, grads)

    @nnx.jit
    def act_jit(agent, carry_h, prev_action, obs, w_explore, eps_key, use_random):
        greedy_action, next_carry_h = gpi_action(ok, agent, carry_h, prev_action, obs, w_explore)
        random_action = jax.random.randint(eps_key, greedy_action.shape, 0, fw.NUM_ACTIONS)
        action = jnp.where(use_random, random_action, greedy_action)
        return action, next_carry_h

    rng = np.random.default_rng(SEED + 2)
    buf = MacroStepBuffer()

    # w_explore=[1,1] is only used to break ties when eps_min < 1 (GPI-greedy
    # component of behavior). With eps_min=1.0 (the default) it's never actually
    # invoked. pure uniform-random exploration is used for the entire run. an earlier version used a GPI-greedy behavior policy
    # w.r.t. w_explore=[1,1] once eps decayed, which converged to camping at the
    # single cell (y3, composition (1,1)) that jointly maximizes both nutrients --
    # starving y1/y2-specific data and causing the very SF estimates GPI depends on
    # to *degrade* with more training (see REPORT.md). Pure random exploration keeps
    # visitation of all 3 food cells flat (~2% hit rate each) for the entire run,
    # confirmed empirically, regardless of training duration.
    w_explore = jnp.array([1.0, 1.0])
    eps = 1.0
    eps_decay = 0.99997  # irrelevant when eps_min=1.0; kept for optional lower-eps_min experiments

    obs, pos, t_arr = fw.reset_batch(rng, num_envs)
    carry_h = jnp.zeros((num_envs, H_DIM), dtype=jnp.float32)
    prev_action = jnp.full((num_envs,), -1, dtype=jnp.int32)

    key_loop = jax.random.PRNGKey(SEED + 3)
    t0 = time.time()
    total_env_steps = 0
    food_hits_window = np.zeros(3, dtype=np.int64)  # coverage diagnostic: y1/y2/y3 hits since last log
    window_env_steps = 0
    for step in range(n_macro_steps):
        key_loop, eps_key = jax.random.split(key_loop)
        use_random = jnp.asarray(rng.random(num_envs) < eps)
        action, next_carry_h = act_jit(agent, carry_h, prev_action, jnp.asarray(obs), w_explore, eps_key, use_random)
        action_np = np.asarray(action)

        next_obs, next_pos, phi, done, next_t = fw.step_batch(pos, action_np, t_arr, horizon=40)
        total_env_steps += num_envs
        window_env_steps += num_envs
        food_hits_window[0] += int(np.sum((phi[:, 0] == 1) & (phi[:, 1] == 0)))
        food_hits_window[1] += int(np.sum((phi[:, 0] == 0) & (phi[:, 1] == 1)))
        food_hits_window[2] += int(np.sum((phi[:, 0] == 1) & (phi[:, 1] == 1)))

        buf.add({
            "obs": obs, "carry": np.asarray(carry_h), "prev_action": np.asarray(prev_action),
            "action": action_np, "phi": phi, "done": done.astype(np.float32),
        })

        pos, t_arr = next_pos, next_t
        obs = next_obs
        prev_action = action
        carry_h = next_carry_h
        if done.any():
            reset_obs, reset_pos, _ = fw.reset_batch(rng, num_envs)
            obs = np.where(done[:, None], reset_obs, obs)
            pos = np.where(done[:, None], reset_pos, pos)
            t_arr = np.where(done, 0, t_arr)
            carry_h = jnp.where(jnp.asarray(done)[:, None], 0.0, carry_h)
            prev_action = jnp.where(jnp.asarray(done), -1, prev_action)

        eps = max(eps_min, eps * eps_decay)

        if len(buf.data) > N_MACRO_SAMPLES + 1:
            first, second = buf.sample(rng, N_MACRO_SAMPLES)
            train_step(
                agent, target_agent, opt,
                jnp.asarray(first["carry"]), jnp.asarray(second["carry"]),
                jnp.asarray(first["prev_action"]), jnp.asarray(first["obs"]),
                jnp.asarray(first["action"]), jnp.asarray(first["phi"]),
                jnp.asarray(second["obs"]), jnp.asarray(first["done"]),
            )

        if step % TARGET_SYNC_EVERY == 0:
            nnx.update(target_agent, jax.tree.map(jnp.copy, nnx.state(agent)))

        if step % 10_000 == 0:
            rates = food_hits_window / max(window_env_steps, 1)
            print(f"    [SF pretrain] macro-step {step:6d}  env-steps={total_env_steps:9d}  "
                  f"eps={eps:.3f}  elapsed={time.time()-t0:.1f}s  "
                  f"hit-rate(y1,y2,y3)=({rates[0]:.4f},{rates[1]:.4f},{rates[2]:.4f})", flush=True)
            food_hits_window[:] = 0
            window_env_steps = 0

    print(f"    [SF pretrain] finished: {total_env_steps} total env-steps in {time.time()-t0:.1f}s")
    return agent


def rollout_avg_reward(ok, agent, action_fn_jit, w_task, n_episodes=50, horizon=40, seed=1000):
    """Greedy rollout (no exploration); per-step task reward = w_task . cumulant."""
    rewards = []
    for ep in range(n_episodes):
        env = fw.ForagingWorldEnv(horizon=horizon, rng=np.random.default_rng(seed + ep))
        obs = env.reset()
        carry_h = jnp.zeros((1, H_DIM), dtype=jnp.float32)
        prev_action = jnp.array([-1], dtype=jnp.int32)
        ep_reward = 0.0
        for t in range(horizon):
            action, carry_h = action_fn_jit(agent, carry_h, prev_action, jnp.asarray(obs[None, :]))
            action = int(action[0])
            obs, phi, done = env.step(action)
            ep_reward += float(np.dot(w_task, phi))
            prev_action = jnp.array([action], dtype=jnp.int32)
        rewards.append(ep_reward / horizon)
    return float(np.mean(rewards)), float(np.std(rewards))


def sf_accuracy_check(ok, agent, seed=2000, n_episodes=20, horizon=40):
    all_pred, all_empirical = [], []
    w_explore = jnp.array([1.0, 1.0])

    @nnx.jit
    def step_jit(agent, carry_h, prev_action, obs, w_explore):
        psi_all, _beta, next_carry_h = agent(carry_h, prev_action, obs)
        action, _ = gpi_action(ok, agent, carry_h, prev_action, obs, w_explore)
        return psi_all, action, next_carry_h

    for ep in range(n_episodes):
        env = fw.ForagingWorldEnv(horizon=horizon, rng=np.random.default_rng(seed + ep))
        obs = env.reset()
        carry_h = jnp.zeros((1, H_DIM), dtype=jnp.float32)
        prev_action = jnp.array([-1], dtype=jnp.int32)
        traj_phi = []
        traj_psi_pred = []
        for t in range(horizon):
            psi_all, action, next_carry_h = step_jit(agent, carry_h, prev_action, jnp.asarray(obs[None, :]), w_explore)
            a = int(action[0])
            traj_psi_pred.append(np.asarray(psi_all[0, :, a, :]))  # (num_options, cumulant_dim) at time t
            obs, phi, done = env.step(a)
            traj_phi.append(phi)
            prev_action = jnp.array([a], dtype=jnp.int32)
            carry_h = next_carry_h

        traj_phi = np.stack(traj_phi)  # (horizon, cumulant_dim)
        discounted = np.zeros_like(traj_phi)
        running = np.zeros(fw.CUMULANT_DIM, dtype=np.float32)
        for t in reversed(range(horizon)):
            running = traj_phi[t] + GAMMA * running
            discounted[t] = running
        for t in range(horizon):
            for opt_idx in range(NUM_OPTIONS):
                all_pred.append(traj_psi_pred[t][opt_idx])
                all_empirical.append(discounted[t])
    return np.array(all_pred), np.array(all_empirical)


def main():
    t0 = time.time()
    ok = load_option_keyboard()
    key = jax.random.PRNGKey(SEED)

    print("=== Pretraining successor features for W0 = {(1,0), (0,1)} (paper's exact basis set) ===")
    agent = train(ok, key)

    print("\n=== SF accuracy check: psi predictions vs. empirical discounted cumulant returns ===")
    pred, empirical = sf_accuracy_check(ok, agent)
    corr_per_dim = [float(np.corrcoef(pred[:, d], empirical[:, d])[0, 1]) for d in range(fw.CUMULANT_DIM)]
    mae = float(np.mean(np.abs(pred - empirical)))
    print(f"    correlation(psi, empirical) per cumulant dim: {np.round(corr_per_dim, 3)}  mean_abs_error={mae:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for d in range(fw.CUMULANT_DIM):
        axes[d].scatter(empirical[:, d], pred[:, d], s=4, alpha=0.3)
        lo, hi = empirical[:, d].min(), empirical[:, d].max()
        axes[d].plot([lo, hi], [lo, hi], "r--", lw=1)
        axes[d].set_xlabel("empirical discounted cumulant")
        axes[d].set_ylabel("psi prediction")
        axes[d].set_title(f"nutrient {d+1}  (r={corr_per_dim[d]:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ok_sf_accuracy.png", dpi=130)
    plt.close(fig)

    print("\n=== GPI zero-shot combination test (paper's Scenario A2 setup) ===")
    test_ws = {
        "w=(1,0) [trained basis]": np.array([1.0, 0.0]),
        "w=(0,1) [trained basis]": np.array([0.0, 1.0]),
        "w=(1,-1) [NOVEL -- paper's key example]": np.array([1.0, -1.0]),
        "w=(-1,1) [NOVEL]": np.array([-1.0, 1.0]),
        "w=(1,1) [NOVEL]": np.array([1.0, 1.0]),
    }

    single_fns_jit = [
        nnx.jit(lambda agent, carry_h, prev_action, obs, opt_idx=opt_idx:
                single_option_action(ok, agent, carry_h, prev_action, obs, opt_idx))
        for opt_idx in range(NUM_OPTIONS)
    ]

    results = {}
    for name, w in test_ws.items():
        w_j = jnp.asarray(w, dtype=jnp.float32)
        gpi_fn_jit = nnx.jit(lambda agent, carry_h, prev_action, obs, w_j=w_j:
                              gpi_action(ok, agent, carry_h, prev_action, obs, w_j))

        gpi_mean, gpi_std = rollout_avg_reward(ok, agent, gpi_fn_jit, w)
        opt_val = fw.optimal_reward_per_step(w)

        best_single, best_single_val = None, -1e9
        for opt_idx in range(NUM_OPTIONS):
            m, s = rollout_avg_reward(ok, agent, single_fns_jit[opt_idx], w)
            if m > best_single_val:
                best_single_val, best_single = m, opt_idx

        results[name] = {
            "w": w.tolist(),
            "gpi_mean_reward_per_step": gpi_mean,
            "gpi_std": gpi_std,
            "optimal_reward_per_step": opt_val,
            "best_single_option_idx": best_single,
            "best_single_option_reward_per_step": best_single_val,
        }
        print(f"    {name:42s}  GPI={gpi_mean:+.3f}  best-single-option={best_single_val:+.3f}  "
              f"optimal={opt_val:+.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.27
    ax.bar(x - width, [results[n]["gpi_mean_reward_per_step"] for n in names], width, label="GPI (combined)")
    ax.bar(x, [results[n]["best_single_option_reward_per_step"] for n in names], width, label="best single option")
    ax.bar(x + width, [results[n]["optimal_reward_per_step"] for n in names], width, label="optimal")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" [")[0] for n in names], rotation=20)
    ax.set_ylabel("avg task reward / step")
    ax.set_title("Option Keyboard: GPI zero-shot combination vs. single pretrained option")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ok_gpi_zeroshot.png", dpi=130)
    plt.close(fig)

    novel_keys = [k for k in results if "NOVEL" in k]
    novel_gpi_near_optimal = np.mean([
        results[k]["gpi_mean_reward_per_step"] >= 0.7 * results[k]["optimal_reward_per_step"]
        if results[k]["optimal_reward_per_step"] > 0 else
        results[k]["gpi_mean_reward_per_step"] >= results[k]["optimal_reward_per_step"] - 0.3
        for k in novel_keys
    ])

    summary = {
        "config": {"W_options": W_OPTIONS.tolist(), "num_envs": NUM_ENVS, "n_macro_steps": N_MACRO_STEPS,
                    "total_env_steps": NUM_ENVS * N_MACRO_STEPS, "gamma": GAMMA,
                    "hidden_dim": HIDDEN_DIM, "h_dim": H_DIM},
        "sf_accuracy": {"correlation_per_cumulant_dim": corr_per_dim, "mean_abs_error": mae},
        "gpi_zero_shot": results,
        "frac_novel_w_gpi_near_optimal": float(novel_gpi_near_optimal),
        "runtime_sec": time.time() - t0,
    }
    with open(RESULTS_DIR / "okeyboard_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary written to "
          f"{RESULTS_DIR / 'okeyboard_verification_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
