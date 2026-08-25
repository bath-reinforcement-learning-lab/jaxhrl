# jaxhrl

Open Source Hierarchical Reinforcement Learning algorithms implemented in pure JAX.

## Why JAX

Every algorithm here — environment stepping, agent forward/backward passes,
replay, and the training loop itself — is fully JITTed
and run under `jax.jit`/`jax.lax.scan`. There's no per-step Python/host
round-trip. That allows them to run end-to-end with fully JITTed environments such as craftax/gymnax, running 
huge numbers of environments in parallel via `jax.vmap`.
Most algorithms here default to `num_envs: 1024`. Swapping in a different
environment doesn't require touching the algorithm code — only the wrapper
in [`jaxhrl/common/wrappers.py`](jaxhrl/common/wrappers.py) needs adjusting
to expose that environment's `reset_fn`/`step_fn`/observation and action
shapes in the form each algorithm expects.

## Algorithms

| Algorithm | Paper | Status |
|---|---|---|
| [DCEO](jaxhrl/DCEO.py) | Klissarov & Machado, *"Deep Laplacian-based Options for Temporally-Extended Exploration"* (ICML 2023) | ✅ Verified — the Laplacian representation correctly recovers the true graph eigenvectors |
| [h-DQN](jaxhrl/h-DQN.py) | Kulkarni et al., *"Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation"* (2016) | ✅ Verified — matches the paper's own toy-MDP result |
| [Option Keyboard](jaxhrl/option_keyboard.py) | Barreto et al., *"The Option Keyboard: Combining Skills in Reinforcement Learning"* (NeurIPS 2019) | ✅ GPI's zero-shot skill combination matches the paper's own worked example |
| [HiPPO](jaxhrl/HiPPO.py) | Manager/skill hierarchical PPO (learned skills, SMDP-level manager) | ⬜ Not yet verified |
| [HAC](jaxhrl/HAC.py) | Levy et al., *"Learning Multi-Level Hierarchies with Hindsight"* | 🚧 Stub — not yet implemented |
| [HIRO](jaxhrl/HIRO.py) | Nachum et al., *"Data-Efficient Hierarchical Reinforcement Learning"* | 🚧 Stub — not yet implemented |
| [option_critic](jaxhrl/option_critic.py) | Bacon, Harb & Precup, *"The Option-Critic Architecture"* | 🚧 Stub — not yet implemented |

## Verification

[`verification/`](verification/)
contains standalone scripts that import each algorithm's actual network and
loss code and test it against toy environments from the original papers.. Full write-ups, plots, and numbers are in
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
- HiPPO / HAC / HIRO / option_critic: **TODO**.

Rerun any check with e.g. `python verification/dceo_verify.py`.

## Running an algorithm

```bash
pip install jax flax optax flashbax gymnax pyyaml  # + wandb/mlflow if logging with them
python jaxhrl/DCEO.py --config path/to/config.yaml [--seed 0]
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

No example config YAMLs ship yet — check each script's `__main__` block for
the exact keys it reads.

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

