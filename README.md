# jaxhrl

Open Source Hierarchical Reinforcement Learning algorithms implemented in pure JAX.

## Why JAX

Every algorithm is written in pure JAX and fully JITTed. That allows them to run end-to-end with fully JITTed environments such as craftax/gymnax, running  huge numbers of environments in parallel via `jax.vmap`.
Most algorithms here default to `num_envs: 1024`. Swapping in a different
environment doesn't require touching the algorithm code — only the wrapper
in [`jaxhrl/common/wrappers.py`](jaxhrl/common/wrappers.py) needs adjusting
to expose that environment's `reset_fn`/`step_fn`/observation and action
shapes in the form each algorithm expects.

Our implementations run very fast. HiPPO runs at ~73000 env-steps/s on a single NVIDIA GeForce RTX 4090.

## Algorithms

| Algorithm | Paper | Status |
|---|---|---|
| [DCEO](jaxhrl/DCEO.py) | Klissarov & Machado, *"Deep Laplacian-based Options for Temporally-Extended Exploration"* (ICML 2023) | Verified — the Laplacian representation correctly recovers the true graph eigenvectors |
| [h-DQN](jaxhrl/h-DQN.py) | Kulkarni et al., *"Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation"* (2016) | Verified — matches the paper's own toy-MDP result |
| [Option Keyboard](jaxhrl/option_keyboard.py) | Barreto et al., *"The Option Keyboard: Combining Skills in Reinforcement Learning"* (NeurIPS 2019) | GPI's zero-shot skill combination matches the paper's own worked example |
| [HiPPO](jaxhrl/HiPPO.py) | Li, Florensa, Clavera & Abbeel, *"Sub-Policy Adaptation for Hierarchical Reinforcement Learning"* (ICLR 2020) | Verified — reproduces the paper's own time-commitment ablation and skill-diversity diagnostic |
| [HAC](jaxhrl/HAC.py) | Levy et al., *"Learning Multi-Level Hierarchies with Hindsight"* | Partially verified — 2-level HAC reproduces the paper's sample-efficiency claim over a flat agent (2.7x fewer steps); the 3-level claim does not reproduce |
| [option_critic](jaxhrl/option_critic.py) | Bacon, Harb & Precup, *"The Option-Critic Architecture"* (AAAI 2017) |  Verified — reproduces the paper's four-rooms transfer direction (Figure 3, options recover faster after the goal moves) and option specialization (Figure 4) |
| [MOC](jaxhrl/MOC.py) | Klissarov & Precup, *"Flexible Option Learning"* (NeurIPS 2021) |  Verified — reproduces the paper's four-rooms result (Figure 1b): multi-updating recovers from the goal relocation far faster than vanilla Option-Critic and with ~10x lower seed variance |

## Verification

[`verification/`](verification/)
contains standalone scripts that import each algorithm's actual network and
loss code and test it against toy environments from the original papers. Full write-ups, plots, and numbers are in
[`verification/REPORT.md`](verification/REPORT.md). Summary:

- **DCEO**: the Laplacian representation network recovers the true graph
  Laplacian eigenvectors (cosine similarity ~0.7+ against an exact
  ground-truth eigendecomposition on FourRooms, on the well-separated
  eigenvalues).
- **h-DQN**: reproduced Kulkarni et al.'s own toy stochastic decision
  process — the hierarchical agent learns it, a flat DQN baseline with the
  same network and step budget doesn't.
- **Option Keyboard**: reproduced Barreto et al.'s own "Foraging World"
  worked example at scale.
- **HiPPO**: reproduced Li et al.'s own time-commitment ablation (Figure 3)
  and skill-diversity/gradient-approximation diagnostic (Table 2) on a small
  custom POMDP standing in for the paper's MuJoCo environments — HiPPO
  (randomized or fixed period) solves the task while a p=1 ablation and flat
  PPO both plateau at the same no-memory ceiling, and the approximate vs.
  exact policy gradient stay in the same close-agreement regime the paper
  reports.
- **HAC**: partially verified. On a continuous Four Rooms task with the
  episode budget held identical across arms, 2-level HAC reaches 50% success
  2.7x faster and 80% success 2.2x faster than a flat agent — the paper's
  sample-efficiency claim. The 3-level claim does **not** reproduce, and the
  cause is open: two candidate explanations (per-level horizon allocation, and
  level-0 reach margin) were each tested and neither survived. The machinery
  itself is sound at every depth — 1/2/3-level agents reach 1.000/0.998/0.955
  success on an open point-mass, and the DDPG core recovers a known-optimal Q
  function in isolation — so what is unverified is the paper's *advantage* from
  a third level, not the third level working at all.
- **Option-Critic**: reproduced Bacon et al.'s four-rooms transfer test
  (Figure 3) — options cost nothing on the stationary task (learning curves
  superimposed on a flat actor-critic built from the same code path with
  `num_options=1`), and after the goal is relocated the Option-Critic agents
  recover faster (post-switch AUC 0.51 / 0.56 for 4 / 8 options vs 0.42 flat,
  16 seeds). The learned options partition the
  grid into spatially-coherent regions (Figure 4).
- **MOC** (*Flexible Option Learning*): reproduced the paper's four-rooms
  Figure 1b on the same task and harness as the Option-Critic check above,
  swapping only the loss function. Learning the initial task from scratch, all
  of flat / OC / MOC converge together; after the goal is relocated MOC
  recovers to 0.99 return while OC reaches 0.80 and a flat agent 0.56 in the
  same budget, and MOC's final-return seed std is 0.01 vs OC's 0.13 (16 seeds,
  all 16 MOC seeds recover vs 8/16 OC). MOC does this by leaning on fewer
  options (usage entropy 0.14 vs OC's 0.99) — the diversity/performance
  trade-off the paper's η hyperparameter is meant to control.

Rerun any check with e.g. `python verification/dceo_verify.py`.

## Running an algorithm

```bash
pip install jax flax optax flashbax gymnax pyyaml  # + wandb/mlflow if logging with them
python -m jaxhrl.HiPPO --config jaxhrl/configs/HiPPO.craftax.yaml
```

Every algorithm takes the same `--config <yaml>` (and optional `--seed`)
interface. A config needs at minimum:

```yaml
seed: 0
experiment: my_run_name
env:
  framework: gymnax        # only framework currently wired up
  make:
    id: "MyEnv-v0"
training:
  num_envs: 1024            # parallel environments, vmapped end-to-end
  n_steps: 20000
  # ...algorithm-specific hyperparameters (learning rates, num_options,
  # buffer sizes, etc.) -- see the `config_raw["training"].get(...)` calls
  # near the top of each script's __main__ block for the full list and
  # defaults.
```

[`jaxhrl/`](jaxhrl/configs/) contains example configs for craftax.

## Logging

Every algorithm shares one `Logger` ([`jaxhrl/common/logger.py`](jaxhrl/common/logger.py)),
turned on entirely from the config YAML — no code changes needed, and you
can enable more than one backend at once:

```yaml
experiment: my_dceo_run   # required by Logger regardless of backend
save_json: true           # local JSON, no extra service needed
use_mlflow: true          # logs params/metrics to an MLflow experiment
use_wandb: true            # logs to Weights & Biases
project: my-wandb-project  # only used if use_wandb
entity: my-team             # optional, falls back to your personal workspace if inaccessible
overwrite: false            # set true to wipe a previous run with the same experiment name
```

- **`save_json`** — metrics are buffered in memory and written to
  `results/<experiment>/runs/<timestamp>.json` on close; a `config.yaml` is
  saved alongside and checked for consistency on reruns of the same
  `experiment` name (mismatches raise, unless `overwrite: true`).
- **`use_mlflow`** — standard `mlflow.log_params`/`log_metrics` under an
  experiment named after `experiment`.
- **`use_wandb`** — standard `wandb.init`/`wandb.log`. W&B is also currently
  the *only* backend wired up for two extras that each algorithm's periodic
  eval hook (`eval.enabled` in the config) calls into:
  - `logger.save_checkpoint(params, step)` — uploads Flax params (msgpack)
    as a W&B Artifact.
  - `logger.log_eval_trajectory(step, trajectory, frames=None)` — logs a
    per-timestep eval table (reward, cumulative reward, option/skill chosen)
    and an optional rollout video.

  Both are no-ops if `use_wandb` isn't set — so with only `save_json` and/or
  `use_mlflow` enabled, periodic checkpoints and eval-trajectory logging
  are silently skipped even if `eval.enabled: true`.

If none of the three backends are enabled, `Logger` prints a warning (not an
error) and just runs without logging anywhere.

