"""HierQ paper-fidelity verification.

Reproduces the discrete-task claim of Levy et al., "Learning Multi-Level
Hierarchies with Hindsight" (ICLR 2019), Figure 4: on grid world tasks, an
agent with more levels of hierarchy learns faster than one with fewer, and
both beat a flat agent --

    "In all tasks, the 3-level agent outperformed the 2-level agent, and the
     2-level agent outperformed the flat agent."

The paper's flat discrete baseline is "Q-learning with HER", which is exactly
`num_levels: 1` here: HierQ's level-0 update already relabels every goal in the
state space on every transition, so a 1-level HierQ *is* Q-learning with
(exhaustive) hindsight experience replay. No separate baseline is needed.

Both of the paper's discrete domains are run: the open 10x10 grid world and
Four Rooms.

The episode horizon is held FIXED at 125 primitive steps for every arm, with
the sub-level budget H=5 constant across depths and the top level absorbing the
remainder (H_levels = [125] / [5,25] / [5,5,5]), so each agent gets both the
same environment budget and the same level-0 reach. The x-axis is training episodes, matching the paper's figure.

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

# Sub-level action budget, held CONSTANT across depths (Levy keeps his
# time_scale fixed and lets only the top level absorb the horizon).
HORIZON = 125         # primitive steps per episode, identical for every arm
SUBLEVEL_H = 5
NUM_ENVS = 16
N_STEPS = 12500       # scan steps -> 100 chunks of one episode each
CHUNK = HORIZON
SEEDS = [0, 1, 2]
TASKS = ["fourrooms", "10x10"]

# (label, num_levels). H and horizon are the same for every arm; HierQ derives
# the per-level budgets from them.
ARMS = [
    ("flat (k=1)",     1),
    ("2-level HierQ",  2),
    ("3-level HierQ",  3),
]

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
training:
  num_levels: {k}
  H: {SUBLEVEL_H}
  horizon: {HORIZON}
  n_steps: {N_STEPS}
  num_envs: {NUM_ENVS}
  chunk_size: {CHUNK}
  alpha: 0.1
  epsilon: 0.2
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
    episodes = np.cumsum(np.asarray(m["train/episodes"], float))
    return success, episodes


def episodes_to(success, episodes, thresh):
    """Training episodes until success first reaches `thresh` (smoothed over 3
    points so a single lucky chunk does not count as convergence)."""
    if len(success) < 3:
        return None
    sm = np.convolve(success, np.ones(3) / 3, mode="valid")
    idx = np.nonzero(sm >= thresh)[0]
    return float(episodes[idx[0] + 2]) if len(idx) else None


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg_dir = RESULTS / "_hierq_configs"
    cfg_dir.mkdir(exist_ok=True)

    curves = {}   # (task, label) -> list of (success, episodes) per seed
    for task in TASKS:
        for label, k in ARMS:
            per_seed = []
            for seed in SEEDS:
                cfg = cfg_dir / f"{slug(task, label, seed)}.yaml"
                write_config(task, label, k, seed, cfg)
                shutil.rmtree(REPO_ROOT / "results" / slug(task, label, seed),
                              ignore_errors=True)
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

    # ---- plot: one panel per task, mean +- 1 std across seeds ----
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 4.5), squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        for label, _ in ARMS:
            runs = curves[(task, label)]
            n = min(len(s) for s, _ in runs)
            S = np.stack([s[:n] for s, _ in runs])
            E = np.mean([e[:n] for _, e in runs], axis=0)
            mu, sd = S.mean(0), S.std(0)
            ax.plot(E, mu, label=label, **STYLES[label])
            ax.fill_between(E, mu - sd, mu + sd, alpha=0.15,
                            color=STYLES[label].get("color"))
        ax.set_xlabel("training episodes"); ax.set_ylabel("success rate")
        ax.set_title(f"{task}  (episode horizon {HORIZON})")
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "hierq_levels_comparison.png", dpi=130)

    # ---- summary ----
    summary = {"horizon": HORIZON, "num_envs": NUM_ENVS, "seeds": SEEDS, "tasks": {}}
    for task in TASKS:
        summary["tasks"][task] = {}
        for label, _ in ARMS:
            runs = curves[(task, label)]
            finals = [s[-1] for s, _ in runs]
            e50 = [episodes_to(s, e, 0.5) for s, e in runs]
            e80 = [episodes_to(s, e, 0.8) for s, e in runs]
            summary["tasks"][task][label] = {
                "final_success_mean": float(np.mean(finals)),
                "final_success_std": float(np.std(finals)),
                "episodes_to_0.5": None if any(v is None for v in e50) else float(np.mean(e50)),
                "episodes_to_0.8": None if any(v is None for v in e80) else float(np.mean(e80)),
            }
    (RESULTS / "hierq_verification_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== HierQ verification summary "
          f"({len(SEEDS)} seeds, horizon {HORIZON} for every arm) ===")
    for task in TASKS:
        print(f"\n{task}")
        print(f"  {'arm':<16}{'final':>14}{'eps->0.5':>11}{'eps->0.8':>11}")
        for label, _ in ARMS:
            a = summary["tasks"][task][label]
            f = lambda v: f"{v:,.0f}" if v is not None else "never"
            print(f"  {label:<16}{a['final_success_mean']:>8.3f}+-{a['final_success_std']:<4.3f}"
                  f"{f(a['episodes_to_0.5']):>11}{f(a['episodes_to_0.8']):>11}")

    print("\nPaper's claim: 3-level > 2-level > flat.")
    for task in TASKS:
        order = [summary["tasks"][task][l]["episodes_to_0.8"] for l, _ in ARMS]
        if any(o is None for o in order):
            verdict = "incomplete (an arm never reached 0.8)"
        else:
            verdict = "HOLDS" if order[2] < order[1] < order[0] else "does not hold"
        print(f"  {task:12} episodes to 0.8 (flat, 2-level, 3-level) = "
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
