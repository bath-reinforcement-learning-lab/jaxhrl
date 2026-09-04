"""
METRA paper-fidelity verification.

Reproduces METRA's central claims (Park, Rybkin & Levine, ICLR 2024) at
gridworld scale, with a 2-D continuous skill space:

  Test A -- skills are directed and diverse: a skill conditioned on z actually
    moves the learned abstraction phi in the direction of z (the paper's
    objective, Eq. 7), so cos(phi(s_end) - phi(s_start), z) is strongly
    positive -- something a random policy does not do -- and different skills
    fan out to different parts of the grid.

  Test B -- phi recovers the temporal-distance metric (Theorem 4.1: linear
    squared METRA ~ PCA under the temporal-distance metric): the pairwise
    distance in phi-space tracks the exact shortest-path distance between
    cells (Spearman correlation), and phi laid out in 2-D matches the
    classical-MDS embedding of the shortest-path matrix (Procrustes).

  Test C -- zero-shot goal reaching (paper Section 5.3 / Figure 8): setting
    z = (phi(g) - phi(s)) / ||.|| and running the skill policy greedily
    reaches the goal cell, with no goal-conditioned policy ever trained.
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = HERE / "results"

SEEDS = [0, 1, 2, 3, 4]
NUM_ENVS = 128
Z_DIM = 2
CHUNK = 250
NUM_STEPS = 100_000         # scan-steps -> NUM_STEPS * NUM_ENVS env-steps (~12.8M)
HIDDEN_DIM = 128


def write_config(seed, path):
    path.write_text(f"""seed: {seed}
experiment: METRAV_seed{seed}
save_json: false
use_wandb: false
env:
  framework: gridworld
  make: {{id: fourrooms_open}}
training:
  num_steps: {NUM_STEPS}
  num_envs: {NUM_ENVS}
  chunk_size: {CHUNK}
  z_dim: {Z_DIM}
  unit_z: true
  batch_size: 256
  warmup_steps: 2000
  buffer_size: 100000
  lr: 3.0e-4
  gamma: 0.99
  target_tau: 0.005
  lagrange_eps: 1.0e-3
  alpha_init: 0.05
  lambda_init: 30.0
  target_entropy: 0.9
  hidden_dim: {HIDDEN_DIM}
eval:
  enabled: false
""")


# ---------------------------------------------------------------------------
# Child: run the repo's real METRA training loop, dump the trained params.
# ---------------------------------------------------------------------------
def run_child(config_path, params_out):
    import types
    import yaml
    import runpy
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO_ROOT))

    import jaxhrl.common.wrappers as W          # the REAL wrapper (for JaxWrappedEnv)
    import fourrooms_open as fo

    cfg = yaml.safe_load(open(config_path))

    class _Logger:
        def __init__(self, *a, **k): self.rows = []
        def log_metrics(self, metrics, step=None):
            self.rows.append({"step": step, **{k: float(v) for k, v in metrics.items()}})
        def save_checkpoint(self, *a, **k): pass
        def log_eval_trajectory(self, *a, **k): pass
        def close(self):
            Path(str(config_path) + ".metrics.json").write_text(json.dumps(self.rows))

    def _make_jax_env(*a, **k):
        return fo.make_wrapped_env(W.JaxWrappedEnv)

    # METRA.py imports parse_config / Logger / make_jax_env from brll_core only.
    def _stub(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
    for parent in ("brll_core", "brll_core.algorithms", "brll_core.algorithms.common"):
        _stub(parent)
    _stub("brll_core.algorithms.common.utils", parse_config=lambda: cfg)
    _stub("brll_core.algorithms.common.logger", Logger=_Logger)
    _stub("brll_core.algorithms.common.jax_wrappers",
          make_jax_env=_make_jax_env, run_eval_episode=lambda *a, **k: None)

    import jax
    ns = runpy.run_path(str(REPO_ROOT / "jaxhrl" / "METRA.py"), run_name="__main__")
    final_params = jax.device_get(ns["carry"][5])
    with open(params_out, "wb") as f:
        pickle.dump({"params": final_params, "config": ns["config"]}, f)


# ---------------------------------------------------------------------------
# Parent: analysis of the trained phi / skill policy.
# ---------------------------------------------------------------------------
def _analyse(params_blob):
    import jax, jax.numpy as jnp
    sys.path.insert(0, str(HERE))
    from repo_loader import load_metra
    import fourrooms_open as fo
    metra = load_metra()

    params, config = params_blob["params"], params_blob["config"]
    z_dim = config["z_dim"]
    phi_net = metra.Encoder(z_dim=z_dim, hidden_dim=config["hidden_dim"])
    actor_net = metra.Actor(hidden_dim=config["hidden_dim"], num_actions=fo.NUM_ACTIONS)

    all_obs = jnp.eye(fo.NUM_STATES, dtype=jnp.float32)
    phi_all = np.asarray(phi_net.apply(params["phi"], all_obs))          # (S, z_dim)

    # ---- Test B: phi vs temporal-distance metric ----
    D = fo.GRAPH_DIST.astype(float)
    iu = np.triu_indices(fo.NUM_STATES, k=1)
    phi_pdist = np.linalg.norm(phi_all[iu[0]] - phi_all[iu[1]], axis=-1)
    d_true = D[iu]
    # Spearman without scipy: correlation of ranks
    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    spearman = _spearman(phi_pdist, d_true)

    # classical MDS of the shortest-path matrix -> 2-D, then Procrustes vs phi
    D2 = D ** 2
    Jc = np.eye(fo.NUM_STATES) - np.ones((fo.NUM_STATES, fo.NUM_STATES)) / fo.NUM_STATES
    B = -0.5 * Jc @ D2 @ Jc
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1][:2]
    mds = evecs[:, order] * np.sqrt(np.maximum(evals[order], 0))
    def _procrustes_disparity(X, Y):
        X = X - X.mean(0); Y = Y - Y.mean(0)
        X = X / (np.linalg.norm(X) + 1e-12); Y = Y / (np.linalg.norm(Y) + 1e-12)
        U, s, Vt = np.linalg.svd(Y.T @ X)
        return float(1.0 - s.sum() ** 2), (X @ (U @ Vt).T)
    procrustes_disp, phi_aligned = _procrustes_disparity(mds[:, :z_dim] if z_dim >= 2 else mds, phi_all)

    # ---- rollouts of the skill policy ----
    @jax.jit
    def rollout(phi_params, actor_params, z, key, start_pos):
        def body(carry, k):
            obs, state = carry
            a = metra.metra_action(k, obs, z, actor_params, actor_net, greedy=True)
            nobs, nstate, _, _, info = fo.step_fn(k, state, a)
            return (nobs, nstate), info["pos"]
        obs0 = jax.nn.one_hot(start_pos, fo.NUM_STATES, dtype=jnp.float32)
        state0 = fo.OpenState(pos=start_pos, t=jnp.array(0, jnp.int32))
        (_, _), positions = jax.lax.scan(body, (obs0, state0), jax.random.split(key, fo.HORIZON))
        return positions

    @jax.jit
    def rollout_to_goal(phi_params, actor_params, goal_phi, key, start_pos):
        """Closed-loop zero-shot: recompute z = (phi(g) - phi(s_t)) / ||.|| each step."""
        def body(carry, k):
            obs, state = carry
            z = goal_phi - phi_net.apply(phi_params, obs[None, :])[0]
            z = z / (jnp.linalg.norm(z) + 1e-9)
            a = metra.metra_action(k, obs, z, actor_params, actor_net, greedy=True)
            nobs, nstate, _, _, info = fo.step_fn(k, state, a)
            return (nobs, nstate), info["pos"]
        obs0 = jax.nn.one_hot(start_pos, fo.NUM_STATES, dtype=jnp.float32)
        state0 = fo.OpenState(pos=start_pos, t=jnp.array(0, jnp.int32))
        (_, _), positions = jax.lax.scan(body, (obs0, state0), jax.random.split(key, fo.HORIZON))
        return positions

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(12345)

    # ---- Test A: directedness + diversity ----
    K = 64
    zs = rng.normal(size=(K, z_dim)); zs = zs / np.linalg.norm(zs, axis=1, keepdims=True)
    starts = rng.integers(0, fo.NUM_STATES, size=K)
    cos_metra, cos_random, end_positions, path_lens = [], [], [], []
    for i in range(K):
        key, k1, k2 = jax.random.split(key, 3)
        pos = rollout(params["phi"], params["actor"], jnp.asarray(zs[i], jnp.float32),
                      k1, jnp.asarray(starts[i], jnp.int32))
        pos = np.asarray(pos)
        dphi = phi_all[pos[-1]] - phi_all[starts[i]]
        cos_metra.append(float(dphi @ zs[i] / (np.linalg.norm(dphi) + 1e-9)))
        end_positions.append(pos[-1])
        path_lens.append(int(D[starts[i], pos[-1]]))
        # random-action baseline from the same start
        p = starts[i]
        for _ in range(fo.HORIZON):
            p = int(np.asarray(fo._NXT)[p, rng.integers(0, 4)])
        dphi_r = phi_all[p] - phi_all[starts[i]]
        cos_random.append(float(dphi_r @ zs[i] / (np.linalg.norm(dphi_r) + 1e-9)))
    end_cells = np.array([fo.FREE_CELLS[e] for e in end_positions])
    endpoint_spread = float(np.mean(np.std(end_cells, axis=0)))

    # ---- Test C: zero-shot goal reaching ----
    M = 60
    starts_c = rng.integers(0, fo.NUM_STATES, size=M)
    goals_c = rng.integers(0, fo.NUM_STATES, size=M)
    final_d_fixed, final_d_closed, final_d_randz, start_d = [], [], [], []
    for i in range(M):
        s, g = int(starts_c[i]), int(goals_c[i])
        start_d.append(int(D[s, g]))
        zc = phi_all[g] - phi_all[s]
        zc = zc / (np.linalg.norm(zc) + 1e-9)
        key, k1, k2, k3 = jax.random.split(key, 4)
        pf = np.asarray(rollout(params["phi"], params["actor"], jnp.asarray(zc, jnp.float32),
                                k1, jnp.asarray(s, jnp.int32)))
        final_d_fixed.append(int(D[pf[-1], g]))
        pc = np.asarray(rollout_to_goal(params["phi"], params["actor"],
                                        jnp.asarray(phi_all[g], jnp.float32), k2, jnp.asarray(s, jnp.int32)))
        # closed loop: report the closest approach, since it may pass through g
        final_d_closed.append(int(np.min([D[p, g] for p in pc])))
        zr = rng.normal(size=z_dim); zr = zr / np.linalg.norm(zr)
        pr = np.asarray(rollout(params["phi"], params["actor"], jnp.asarray(zr, jnp.float32),
                                k3, jnp.asarray(s, jnp.int32)))
        final_d_randz.append(int(D[pr[-1], g]))
    final_d_fixed = np.array(final_d_fixed); final_d_closed = np.array(final_d_closed)
    start_d = np.array(start_d); final_d_randz = np.array(final_d_randz)

    return {
        "phi_all": phi_all, "mds": mds, "phi_aligned": phi_aligned,
        "metrics": {
            "spearman_phidist_vs_graphdist": spearman,
            "procrustes_disparity_vs_mds": procrustes_disp,
            "test_a_cos_dphi_z_metra": float(np.mean(cos_metra)),
            "test_a_cos_dphi_z_random": float(np.mean(cos_random)),
            "test_a_mean_start_to_end_graphdist": float(np.mean(path_lens)),
            "test_a_endpoint_spread_cells": endpoint_spread,
            "test_c_start_graphdist_mean": float(start_d.mean()),
            "test_c_final_graphdist_fixed_z": float(final_d_fixed.mean()),
            "test_c_final_graphdist_closed_loop_z": float(final_d_closed.mean()),
            "test_c_final_graphdist_random_z": float(final_d_randz.mean()),
            "test_c_closed_loop_success_within_2": float(np.mean(final_d_closed <= 2)),
            "test_c_fixed_z_frac_closer_than_start": float(np.mean(final_d_fixed < start_d)),
        },
    }


def plot(per_seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fourrooms_open as fo
    n = len(per_seed)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8))
    axes = np.atleast_2d(axes)
    for j, res in enumerate(per_seed):
        ph = res["phi_aligned"]
        axes[0, j].scatter(ph[:, 0], ph[:, 1], c=fo.ROOM_OF, cmap="tab10", s=18)
        axes[0, j].set_title(f"seed {j}: learned phi (Procrustes-aligned)\n"
                             f"Spearman={res['metrics']['spearman_phidist_vs_graphdist']:.2f}")
        axes[0, j].set_aspect("equal"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        md = res["mds"]
        axes[1, j].scatter(md[:, 0], md[:, 1], c=fo.ROOM_OF, cmap="tab10", s=18)
        axes[1, j].set_title("classical MDS of shortest-path matrix")
        axes[1, j].set_aspect("equal"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg_dir = RESULTS / "_configs"
    cfg_dir.mkdir(exist_ok=True)
    t0 = time.time()

    per_seed = []
    for seed in SEEDS:
        cfg = cfg_dir / f"metra_seed{seed}.yaml"
        params_out = cfg_dir / f"metra_seed{seed}.params.pkl"
        write_config(seed, cfg)
        print(f"--- METRA seed {seed}: {NUM_STEPS * NUM_ENVS:,} env-steps ---", flush=True)
        ts = time.time()
        proc = subprocess.run(
            [sys.executable, str(HERE / "metra_verify.py"), "--run", str(cfg), str(params_out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-4000:]); print(proc.stderr[-4000:])
            raise RuntimeError(f"seed {seed} failed")
        with open(params_out, "rb") as f:
            blob = pickle.load(f)
        res = _analyse(blob)
        per_seed.append(res)
        m = res["metrics"]
        print(f"    done in {time.time() - ts:.0f}s  "
              f"Spearman={m['spearman_phidist_vs_graphdist']:.3f}  "
              f"cos(dphi,z) metra={m['test_a_cos_dphi_z_metra']:.3f} vs random={m['test_a_cos_dphi_z_random']:.3f}  "
              f"zero-shot d: {m['test_c_start_graphdist_mean']:.1f} -> "
              f"fixed {m['test_c_final_graphdist_fixed_z']:.1f} / closed {m['test_c_final_graphdist_closed_loop_z']:.1f} "
              f"/ rand {m['test_c_final_graphdist_random_z']:.1f}",
              flush=True)

    plot(per_seed, RESULTS / "metra_phi_map.png")

    agg = {}
    keys = per_seed[0]["metrics"].keys()
    for k in keys:
        vals = [r["metrics"][k] for r in per_seed]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}

    summary = {
        "config": {"seeds": SEEDS, "num_envs": NUM_ENVS, "z_dim": Z_DIM,
                   "env_steps_per_seed": NUM_STEPS * NUM_ENVS, "horizon": None},
        "results": agg, "runtime_sec": time.time() - t0,
    }
    (RESULTS / "metra_verification_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== METRA verification summary (mean over seeds) ===")
    for k, v in agg.items():
        print(f"  {k:42s} {v['mean']:.3f} +/- {v['std']:.3f}")
    print(f"\nArtifacts: {RESULTS/'metra_phi_map.png'}, {RESULTS/'metra_verification_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, default=None, metavar=("CONFIG", "PARAMS_OUT"))
    args = ap.parse_args()
    if args.run:
        run_child(Path(args.run[0]), Path(args.run[1]))
    else:
        main()
