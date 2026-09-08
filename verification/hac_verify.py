"""HAC paper-fidelity verification.

Reproduces the central claim of Levy et al., "Learning Multi-Level Hierarchies
with Hindsight" (ICLR 2019): on a sparse-reward, temporally extended
goal-reaching task, an agent with MORE levels of hierarchy learns markedly
faster than one with fewer, and a flat agent barely learns at all.


The episode horizon is held FIXED at 729 primitive steps for every arm
(H_levels = [729] / [27,27] / [9,9,9]), so each agent sees the same environment
budget
and the only variable is hierarchy depth. Without that the comparison would be
rigged: H**k would hand deeper agents longer episodes.

Two arms beyond the depth sweep test the paper's second claim, that subgoal
testing is what keeps a level's subgoals reachable:
  * k=3 with subgoal_test_perc = 0.0 (ablation -- expected to degrade)
  * k=3 with the default 0.3 (the main arm)

Run: python verification/hac_verify.py
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = HERE / "results"

HORIZON = 729         
NUM_ENVS = 128
N_STEPS = 21870       
CHUNK = HORIZON


ARMS = [
    ("flat (k=1)",           1, 729, 0.3),   
    ("2-level HAC",          2, 27,  0.3),   
    ("3-level HAC",          3, 9,   0.3),   
    ("3-level, no subgoal testing", 3, 9, 0.0),
]


def arm_slug(label):
    return "HACV_" + label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")


def write_config(label, k, H, test_perc, path):
    HORIZON_ = HORIZON
    path.write_text(f"""experiment: {arm_slug(label)}
seed: 0
save_json: True
use_wandb: False
overwrite: True
env:
  framework: pointmaze
  make: {{id: fourrooms}}
goal:
  indices: [0, 1]
  low: [-1.0, -1.0]
  high: [1.0, 1.0]
  threshold: [0.03, 0.03]
training:
  num_levels: {k}
  H: {H}
  n_steps: {N_STEPS}
  num_envs: {NUM_ENVS}
  chunk_size: {CHUNK}
  batch_size: 512
  buffer_size: 200000
  subgoal_test_perc: {test_perc}
  random_action_perc: 0.2
  noise_perc: 0.1
  tau: 0.05
  lr_actor: 0.001
  lr_critic: 0.001
network: {{hidden_dim: 128}}
eval: {{enabled: False}}
checkpoint: {{enabled: False}}
""")


# --------------------------------------------------------------------------
# Child mode: patch the env factory, then run the repo's real training loop.
# --------------------------------------------------------------------------

def run_child(config_path):
    import runpy
    sys.path.insert(0, str(REPO_ROOT))
    import jaxhrl.common.wrappers as W
    import pointmaze

    def _make(framework, env_id, cumulant_dim, goal_threshold=0.1, **kwargs):
        return pointmaze.make_wrapped_env(W.JaxWrappedEnv)

    W.make_jax_env = _make
    sys.argv = ["HAC.py", "--config", str(config_path)]
    runpy.run_path(str(REPO_ROOT / "jaxhrl" / "HAC.py"), run_name="__main__")


# --------------------------------------------------------------------------
# Parent mode
# --------------------------------------------------------------------------

def read_curve(label):
    run_dir = REPO_ROOT / "results" / arm_slug(label) / "runs"
    files = sorted(run_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"no metrics written for {label}")
    m = json.load(open(files[-1]))
    m = m.get("metrics", m)
    return np.asarray(m["train/end_goal_success_rate"], dtype=float)


def steps_to(curve, threshold):
    """Env-steps to first reach `threshold` success, or None if never."""
    idx = np.nonzero(curve >= threshold)[0]
    return int((idx[0] + 1) * CHUNK * NUM_ENVS) if len(idx) else None


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg_dir = RESULTS / "_configs"
    cfg_dir.mkdir(exist_ok=True)

    curves = {}
    for label, k, H, tp in ARMS:
        cfg = cfg_dir / f"{arm_slug(label)}.yaml"
        write_config(label, k, H, tp, cfg)
        # Fresh process per arm so JAX/Logger state cannot leak between them.
        shutil.rmtree(REPO_ROOT / "results" / arm_slug(label), ignore_errors=True)
        t0 = time.time()
        print(f"--- {label}: k={k} H={H} (horizon {H**k}) subgoal_test={tp} ---", flush=True)
        proc = subprocess.run(
            [sys.executable, str(HERE / "hac_verify.py"), "--run", str(cfg)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
            raise RuntimeError(f"{label} failed")
        curves[label] = read_curve(label)
        print(f"    done in {time.time() - t0:.0f}s, final success {curves[label][-1]:.3f}", flush=True)

    # ---- plot ----
    x = (np.arange(1, len(next(iter(curves.values()))) + 1)) * CHUNK * NUM_ENVS
    plt.figure(figsize=(8, 5))
    styles = {"flat (k=1)": dict(color="tab:red", ls="--"),
              "2-level HAC": dict(color="tab:orange"),
              "3-level HAC": dict(color="tab:blue", lw=2.5),
              "3-level, no subgoal testing": dict(color="tab:gray", ls=":")}
    for label, c in curves.items():
        plt.plot(x[:len(c)], c, label=label, **styles.get(label, {}))
    plt.xlabel("environment steps")
    plt.ylabel("end-goal success rate")
    plt.title(f"HAC on continuous Four Rooms (episode horizon fixed at {HORIZON} steps)")
    plt.legend(); plt.grid(alpha=0.3); plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / "hac_levels_comparison.png", dpi=130)

    # ---- summary ----
    summary = {
        "horizon": HORIZON, "num_envs": NUM_ENVS,
        "env_steps": int(N_STEPS * NUM_ENVS),
        "arms": {},
    }
    for label, c in curves.items():
        summary["arms"][label] = {
            "final_success": float(c[-1]),
            "best_success": float(c.max()),
            "mean_last_quarter": float(c[-len(c) // 4:].mean()),
            "env_steps_to_0.5": steps_to(c, 0.5),
            "env_steps_to_0.8": steps_to(c, 0.8),
        }
    (RESULTS / "hac_verification_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== HAC verification summary ===")
    print(f"{'arm':<32} {'final':>7} {'best':>7} {'->0.5':>12} {'->0.8':>12}")
    for label, a in summary["arms"].items():
        f = lambda v: f"{v:,}" if v is not None else "never"
        print(f"{label:<32} {a['final_success']:>7.3f} {a['best_success']:>7.3f} "
              f"{f(a['env_steps_to_0.5']):>12} {f(a['env_steps_to_0.8']):>12}")
    print(f"\nArtifacts: {RESULTS/'hac_levels_comparison.png'}, "
          f"{RESULTS/'hac_verification_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="internal: run one arm")
    args = ap.parse_args()
    if args.run:
        run_child(Path(args.run))
    else:
        main()
