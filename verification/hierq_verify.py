"""HierQ (deep) paper-fidelity verification.

Reproduces the discrete-task claim of Levy et al., "Learning Multi-Level
Hierarchies with Hindsight" (ICLR 2019), Figure 4: on grid world tasks, an
agent with more levels of hierarchy learns faster than one with fewer, and
both beat a flat agent --

    "In all tasks, the 3-level agent outperformed the 2-level agent, and the
     2-level agent outperformed the flat agent."

The flat discrete baseline is "Q-learning with HER"; `num_levels: 1` here IS
that baseline, since HierQ's level-0 loss already relabels every goal in the
state space on every transition (exhaustive HER), no separate implementation
needed.


Reported metric: `train/end_goal_success_rate`, the fraction of episodes
whose commanded goal was reached -- HierQ.py samples the task goal uniformly
over the WHOLE goal set every episode (Algorithm 2's g_(k-1) <- G_(k-1)), so
every logged chunk already averages over a representative mix of goals; no
separate held-out test is needed.

The episode horizon is held fixed at 125 primitive steps for every arm, with
the sub-level budget H=5 constant across depths and the top level absorbing
the remainder (H_levels = [125] / [5,25] / [5,5,5]), so every agent gets the
same per-goal-attempt budget and the same level-0 reach -- depth is the only
variable.

Run: python verification/hierq_verify.py
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

HORIZON = 125
SUBLEVEL_H = 5
NUM_ENVS = 64
N_STEPS = 40_000
CHUNK = 500
EPSILON = 0.2
SEEDS = [0, 1, 2]
TASKS = ["fourrooms25"]

ARMS = [("flat (k=1)", 1), ("2-level HierQ", 2), ("3-level HierQ", 3)]
STYLES = {"flat (k=1)": dict(color="tab:green", ls="--"),
          "2-level HierQ": dict(color="tab:blue"),
          "3-level HierQ": dict(color="tab:red", lw=2.5)}


def slug(task, label, seed):
    tag = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    return f"HQV_{task}_{tag}_s{seed}"


def write_config(task, label, k, seed, path):
    path.write_text(f"""experiment: {slug(task, label, seed)}
seed: {seed}
save_json: True
use_wandb: False
overwrite: True
env:
  framework: gridworld
  make: {{id: {task}}}
  kwargs: {{one_hot: True}}
training:
  num_levels: {k}
  H: {SUBLEVEL_H}
  horizon: {HORIZON}
  n_steps: {N_STEPS}
  num_envs: {NUM_ENVS}
  chunk_size: {CHUNK}
  batch_size: 256
  buffer_size: 100000
  lr: 0.0003
  tau: 0.02
  hidden_dim: 128
  subgoal_test_perc: 0.3
  epsilon: {EPSILON}
eval: {{enabled: False}}
checkpoint: {{enabled: False}}
""")


def run_child(config_path):
    import runpy
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv = ["HierQ.py", "--config", str(config_path)]
    runpy.run_path(str(REPO_ROOT / "jaxhrl" / "HierQ.py"), run_name="__main__")


def read_run(task, label, seed):
    d = REPO_ROOT / "results" / slug(task, label, seed) / "runs"
    files = sorted(d.glob("*.json"))
    if not files:
        raise RuntimeError(f"no metrics for {task}/{label}/seed{seed}")
    m = json.load(open(files[-1]))
    m = m.get("metrics", m)
    success = np.asarray(m["train/end_goal_success_rate"], float)
    steps = np.arange(1, len(success) + 1) * CHUNK * NUM_ENVS
    return success, steps


def steps_to(success, steps, thresh):
    """Env-steps to first reach `thresh`, smoothed over 3 chunks so a single
    lucky/unlucky chunk of episodes (small-sample per-chunk success rate)
    does not count as convergence on its own."""
    if len(success) < 3:
        return None
    sm = np.convolve(success, np.ones(3) / 3, mode="valid")
    idx = np.nonzero(sm >= thresh)[0]
    return float(steps[idx[0] + 2]) if len(idx) else None


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg_dir = RESULTS / "_hierq_configs"
    cfg_dir.mkdir(exist_ok=True)

    curves = {}
    for task in TASKS:
        for label, k in ARMS:
            per_seed = []
            for seed in SEEDS:
                cfg = cfg_dir / f"{slug(task, label, seed)}.yaml"
                write_config(task, label, k, seed, cfg)
                shutil.rmtree(REPO_ROOT / "results" / slug(task, label, seed), ignore_errors=True)
                t0 = time.time()
                proc = subprocess.run(
                    [sys.executable, str(HERE / "hierq_verify.py"), "--run", str(cfg)],
                    cwd=str(REPO_ROOT), capture_output=True, text=True)
                if proc.returncode != 0:
                    print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
                    raise RuntimeError(f"{task}/{label}/seed{seed} failed")
                per_seed.append(read_run(task, label, seed))
                print(f"  {task:10} {label:15} seed {seed}  "
                      f"final {per_seed[-1][0][-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            curves[(task, label)] = per_seed

    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 4.5), squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        for label, _ in ARMS:
            runs = curves[(task, label)]
            n = min(len(s) for s, _ in runs)
            S = np.stack([s[:n] for s, _ in runs])
            E = np.mean([e[:n] for _, e in runs], axis=0)
            mu, sd = S.mean(0), S.std(0)
            ax.plot(E, mu, label=label, **STYLES[label])
            ax.fill_between(E, mu - sd, mu + sd, alpha=0.15, color=STYLES[label].get("color"))
        ax.set_xlabel("environment steps"); ax.set_ylabel("end-goal success rate")
        ax.set_title(f"{task}  (per-goal-attempt horizon {HORIZON})")
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "hierq_levels_comparison.png", dpi=130)

    summary = {"horizon": HORIZON, "num_envs": NUM_ENVS, "seeds": SEEDS, "tasks": {}}
    for task in TASKS:
        summary["tasks"][task] = {}
        for label, _ in ARMS:
            runs = curves[(task, label)]
            finals = [s[-1] for s, _ in runs]
            e50 = [steps_to(s, e, 0.5) for s, e in runs]
            e80 = [steps_to(s, e, 0.8) for s, e in runs]
            summary["tasks"][task][label] = {
                "final_success_mean": float(np.mean(finals)),
                "final_success_std": float(np.std(finals)),
                "env_steps_to_0.5": None if any(v is None for v in e50) else float(np.mean(e50)),
                "env_steps_to_0.8": None if any(v is None for v in e80) else float(np.mean(e80)),
            }
    (RESULTS / "hierq_verification_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== HierQ (deep) verification summary ({len(SEEDS)} seeds, "
          f"horizon {HORIZON} for every arm) ===")
    for task in TASKS:
        print(f"\n{task}")
        print(f"  {'arm':<16}{'final':>14}{'steps->0.5':>13}{'steps->0.8':>13}")
        for label, _ in ARMS:
            a = summary["tasks"][task][label]
            f = lambda v: f"{v:,.0f}" if v is not None else "never"
            print(f"  {label:<16}{a['final_success_mean']:>8.3f}+-{a['final_success_std']:<4.3f}"
                  f"{f(a['env_steps_to_0.5']):>13}{f(a['env_steps_to_0.8']):>13}")

    print("\nPaper's claim: 3-level > 2-level > flat.")
    for task in TASKS:
        order = [summary["tasks"][task][l]["env_steps_to_0.8"] for l, _ in ARMS]
        if any(o is None for o in order):
            verdict = "incomplete (an arm never reached 0.8)"
        else:
            verdict = "HOLDS" if order[2] < order[1] < order[0] else "does not hold"
        print(f"  {task:12} env-steps to 0.8 (flat, 2-level, 3-level) = "
              f"{['%.0f' % o if o else 'never' for o in order]} -> {verdict}")
    print(f"\nArtifacts: {RESULTS/'hierq_levels_comparison.png'}, "
          f"{RESULTS/'hierq_verification_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    if args.run:
        run_child(Path(args.run))
    else:
        main()
