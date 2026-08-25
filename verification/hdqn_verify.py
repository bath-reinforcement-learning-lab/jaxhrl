"""
h-DQN paper-fidelity verification.

Uses the ACTUAL QNetwork / train_controller_step / train_meta_step imported
straight from jaxhrl/h-DQN.py -- nothing about the hierarchical agent's
network or update rules is reimplemented here.

Reproduces the core empirical claim of Kulkarni et al. 2016: on a small
discrete stochastic decision process with a delayed, order-dependent sparse
reward (toychain.py), a hierarchical agent using intrinsic-motivation
subgoals learns far faster than a flat epsilon-greedy DQN agent given the
same environment-step budget.

The flat-DQN baseline reuses the SAME QNetwork class from the repo (so the
architecture is identical) with a standard hand-written 1-step Q-learning
update -- only the hierarchical machinery (goals, intrinsic reward,
controller/meta-controller split) differs between the two agents.
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
from repo_loader import load_hdqn
import toychain as tc

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 0
N_EPISODES = 2500
EVAL_EVERY = 100
GAMMA = 0.95
BATCH_SIZE = 32
BUFFER_CAP = 20_000


class _FixedBatchBuffer:
    """train_controller_step/train_meta_step call `buffer.sample(state, key).experience`.
    This stand-in just returns a pre-built batch, so we can drive the ACTUAL
    repo loss functions with a plain Python replay list instead of flashbax."""
    def __init__(self, batch):
        self._batch = batch

    def sample(self, state, key):
        class Result:
            experience = self._batch
        return Result()


def make_jit_controller_step(hdqn, gamma):
    @nnx.jit
    def step(controller, target1, opt1, obs, action, reward, done, next_obs):
        class B:
            first = {"obs": obs, "action": action, "reward": reward, "done": done}
            second = {"obs": next_obs}
        hdqn.train_controller_step(
            controller, target1, opt1, buffer1_state=None, key=None,
            controller_buffer=_FixedBatchBuffer(B), gamma1=gamma,
        )
    return step


def make_jit_meta_step(hdqn, gamma):
    @nnx.jit
    def step(meta, target2, opt2, obs, goal, F, terminal, duration, next_obs):
        class B:
            first = {"obs": obs, "goal": goal, "F": F, "terminal": terminal,
                      "duration": duration, "valid": jnp.ones_like(terminal)}
            second = {"obs": next_obs}
        hdqn.train_meta_step(
            meta, target2, opt2, buffer2_state=None, key=None,
            meta_controller_buffer=_FixedBatchBuffer(B), gamma2=gamma,
        )
    return step


def make_jit_forward(net_attr=None):
    @nnx.jit
    def forward(model, obs):
        return model(obs)
    return forward


class ReplayBuffer:
    def __init__(self, cap=BUFFER_CAP):
        self.cap = cap
        self.data = []

    def add(self, item):
        self.data.append(item)
        if len(self.data) > self.cap:
            self.data.pop(0)

    def sample(self, rng, batch_size):
        idxs = rng.integers(0, len(self.data), size=min(batch_size, len(self.data)))
        return [self.data[i] for i in idxs]


def run_hdqn(hdqn, seed):
    key = jax.random.PRNGKey(seed)
    key, k1c, k1t, k2c, k2t = jax.random.split(key, 5)
    num_goals = tc.N_STATES
    num_actions = 2

    controller = hdqn.QNetwork(tc.N_STATES + num_goals, num_actions, hidden_dim=64, rngs=nnx.Rngs(k1c))
    target1 = hdqn.QNetwork(tc.N_STATES + num_goals, num_actions, hidden_dim=64, rngs=nnx.Rngs(k1t))
    nnx.update(target1, jax.tree.map(jnp.copy, nnx.state(controller)))

    meta = hdqn.QNetwork(tc.N_STATES, num_goals, hidden_dim=64, rngs=nnx.Rngs(k2c))
    target2 = hdqn.QNetwork(tc.N_STATES, num_goals, hidden_dim=64, rngs=nnx.Rngs(k2t))
    nnx.update(target2, jax.tree.map(jnp.copy, nnx.state(meta)))

    opt1 = nnx.Optimizer(controller, optax.adam(1e-3), wrt=nnx.Param)
    opt2 = nnx.Optimizer(meta, optax.adam(1e-3), wrt=nnx.Param)

    controller_step = make_jit_controller_step(hdqn, GAMMA)
    meta_step = make_jit_meta_step(hdqn, GAMMA)
    forward = make_jit_forward()

    buf1 = ReplayBuffer()  # controller transitions
    buf2 = ReplayBuffer()  # meta-controller transitions

    eps1 = 1.0
    eps2 = 1.0
    eps_min = 0.05
    eps1_decay = 0.999
    eps2_decay = 0.999

    env = tc.ToyChainEnv(rng=np.random.default_rng(seed + 1000))
    rng = np.random.default_rng(seed + 2000)

    success_history = []
    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        goal = int(rng.integers(0, num_goals))
        goal_start_obs = obs
        F = 0.0
        duration = 0
        got_big_reward = False

        while not done:
            goal_onehot = tc.one_hot(goal)
            aug_obs = np.concatenate([obs, goal_onehot])
            if rng.random() < eps1:
                action = int(rng.integers(0, num_actions))
            else:
                q = np.asarray(forward(controller, jnp.asarray(aug_obs[None, :])))[0]
                action = int(np.argmax(q))

            next_obs, next_pos, reward, done = env.step(action)
            if reward >= tc.BIG_REWARD:
                got_big_reward = True

            goal_reached = (next_pos == goal)
            intrinsic_r = 1.0 if goal_reached else 0.0
            duration += 1

            controller_done = done or goal_reached
            aug_next_obs = np.concatenate([next_obs, goal_onehot])
            buf1.add({"obs": aug_obs, "action": action, "reward": intrinsic_r,
                      "done": float(controller_done), "next_obs": aug_next_obs})

            F += reward
            hierarchy_done = done or goal_reached
            if hierarchy_done:
                buf2.add({"obs": goal_start_obs, "goal": goal, "F": F,
                          "terminal": float(done), "duration": float(duration),
                          "next_obs": next_obs})

            obs = next_obs
            if hierarchy_done and not done:
                if rng.random() < eps2:
                    goal = int(rng.integers(0, num_goals))
                else:
                    q2 = np.asarray(forward(meta, jnp.asarray(obs[None, :])))[0]
                    goal = int(np.argmax(q2))
                goal_start_obs = obs
                F = 0.0
                duration = 0

            if len(buf1.data) >= BATCH_SIZE:
                batch = buf1.sample(rng, BATCH_SIZE)
                controller_step(
                    controller, target1, opt1,
                    jnp.asarray(np.stack([b["obs"] for b in batch])),
                    jnp.asarray(np.array([b["action"] for b in batch]), dtype=jnp.int32),
                    jnp.asarray(np.array([b["reward"] for b in batch]), dtype=jnp.float32),
                    jnp.asarray(np.array([b["done"] for b in batch]), dtype=jnp.float32),
                    jnp.asarray(np.stack([b["next_obs"] for b in batch])),
                )
            if len(buf2.data) >= BATCH_SIZE:
                batch = buf2.sample(rng, BATCH_SIZE)
                meta_step(
                    meta, target2, opt2,
                    jnp.asarray(np.stack([b["obs"] for b in batch])),
                    jnp.asarray(np.array([b["goal"] for b in batch]), dtype=jnp.int32),
                    jnp.asarray(np.array([b["F"] for b in batch]), dtype=jnp.float32),
                    jnp.asarray(np.array([b["terminal"] for b in batch]), dtype=jnp.float32),
                    jnp.asarray(np.array([b["duration"] for b in batch]), dtype=jnp.float32),
                    jnp.asarray(np.stack([b["next_obs"] for b in batch])),
                )

        eps1 = max(eps_min, eps1 * eps1_decay)
        eps2 = max(eps_min, eps2 * eps2_decay)

        if ep % 200 == 0:
            nnx.update(target1, jax.tree.map(jnp.copy, nnx.state(controller)))
            nnx.update(target2, jax.tree.map(jnp.copy, nnx.state(meta)))

        success_history.append(1.0 if got_big_reward else 0.0)
        if ep % 500 == 0:
            recent = np.mean(success_history[-500:])
            print(f"    [h-DQN] episode {ep:5d}  recent success rate={recent:.3f}", flush=True)

    return np.array(success_history)


def run_flat_dqn(hdqn, seed):
    """Standard 1-step DQN baseline, same QNetwork class, same env-step budget,
    same total episode count -- only the hierarchy/intrinsic-motivation is
    removed, isolating that as the variable under test."""
    key = jax.random.PRNGKey(seed)
    net = hdqn.QNetwork(tc.N_STATES, 2, hidden_dim=64, rngs=nnx.Rngs(key))
    target = hdqn.QNetwork(tc.N_STATES, 2, hidden_dim=64, rngs=nnx.Rngs(jax.random.PRNGKey(seed + 1)))
    nnx.update(target, jax.tree.map(jnp.copy, nnx.state(net)))
    opt = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

    buf = ReplayBuffer()
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.999

    env = tc.ToyChainEnv(rng=np.random.default_rng(seed + 1000))
    rng = np.random.default_rng(seed + 2000)
    forward = make_jit_forward()

    @nnx.jit
    def train_step(net, target, opt, obs, action, reward, done, next_obs):
        def loss_fn(model):
            q_pred = model(obs)
            q_pred_a = jnp.take_along_axis(q_pred, action[:, None], axis=-1).squeeze(-1)
            q_next_online = model(next_obs)
            next_action = jnp.argmax(q_next_online, axis=-1)
            q_next_target = target(next_obs)
            q_next_sel = jnp.take_along_axis(q_next_target, next_action[:, None], axis=-1).squeeze(-1)
            tgt = reward + GAMMA * (1.0 - done) * jax.lax.stop_gradient(q_next_sel)
            return jnp.mean((q_pred_a - jax.lax.stop_gradient(tgt)) ** 2)
        grads = nnx.grad(loss_fn)(net)
        opt.update(net, grads)

    success_history = []
    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        got_big_reward = False
        while not done:
            if rng.random() < eps:
                action = int(rng.integers(0, 2))
            else:
                q = np.asarray(forward(net, jnp.asarray(obs[None, :])))[0]
                action = int(np.argmax(q))
            next_obs, next_pos, reward, done = env.step(action)
            if reward >= tc.BIG_REWARD:
                got_big_reward = True
            buf.add({"obs": obs, "action": action, "reward": reward, "done": float(done), "next_obs": next_obs})
            obs = next_obs

            if len(buf.data) >= BATCH_SIZE:
                batch = buf.sample(rng, BATCH_SIZE)
                o = jnp.asarray(np.stack([b["obs"] for b in batch]))
                a = jnp.asarray(np.array([b["action"] for b in batch]), dtype=jnp.int32)
                r = jnp.asarray(np.array([b["reward"] for b in batch]), dtype=jnp.float32)
                d = jnp.asarray(np.array([b["done"] for b in batch]), dtype=jnp.float32)
                no = jnp.asarray(np.stack([b["next_obs"] for b in batch]))
                train_step(net, target, opt, o, a, r, d, no)

        eps = max(eps_min, eps * eps_decay)
        if ep % 200 == 0:
            nnx.update(target, jax.tree.map(jnp.copy, nnx.state(net)))
        success_history.append(1.0 if got_big_reward else 0.0)
        if ep % 500 == 0:
            recent = np.mean(success_history[-500:])
            print(f"    [flat DQN] episode {ep:5d}  recent success rate={recent:.3f}", flush=True)

    return np.array(success_history)


def smoothed(x, window=100):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    t0 = time.time()
    hdqn = load_hdqn()

    print(f"=== Running h-DQN on the Kulkarni et al. toy chain "
          f"({tc.N_STATES} states, p_left={tc.P_LEFT_SUCCEED}, p_right={tc.P_RIGHT_SUCCEED}, "
          f"horizon={tc.HORIZON}) ===")
    hdqn_success = run_hdqn(hdqn, seed=SEED)
    print(f"  h-DQN overall success rate: {hdqn_success.mean():.3f}  "
          f"(last 500 eps: {hdqn_success[-500:].mean():.3f})")

    print("=== Running flat DQN baseline (same network class, same env-step budget) ===")
    flat_success = run_flat_dqn(hdqn, seed=SEED)
    print(f"  flat DQN overall success rate: {flat_success.mean():.3f}  "
          f"(last 500 eps: {flat_success[-500:].mean():.3f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(smoothed(hdqn_success), label="h-DQN (hierarchical)")
    ax.plot(smoothed(flat_success), label="flat DQN baseline")
    ax.set_xlabel("episode")
    ax.set_ylabel("success rate (100-ep rolling mean)")
    ax.set_title("h-DQN vs. flat DQN on Kulkarni et al. toy stochastic MDP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hdqn_vs_flat_success.png", dpi=130)
    plt.close(fig)

    summary = {
        "env": {"n_states": tc.N_STATES, "p_left_succeed": tc.P_LEFT_SUCCEED,
                "p_right_succeed": tc.P_RIGHT_SUCCEED, "horizon": tc.HORIZON,
                "start_state": tc.START_STATE, "n_episodes": N_EPISODES},
        "hdqn_overall_success_rate": float(hdqn_success.mean()),
        "hdqn_last500_success_rate": float(hdqn_success[-500:].mean()),
        "flat_dqn_overall_success_rate": float(flat_success.mean()),
        "flat_dqn_last500_success_rate": float(flat_success[-500:].mean()),
        "runtime_sec": time.time() - t0,
    }
    with open(RESULTS_DIR / "hdqn_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']:.1f}s. Summary written to {RESULTS_DIR / 'hdqn_verification_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
