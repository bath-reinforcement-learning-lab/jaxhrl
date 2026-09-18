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

| Algorithm | Paper | Description |
|---|---|---|
| [DCEO](jaxhrl/DCEO.py) | Klissarov & Machado, *"Deep Laplacian-based Options for Temporally-Extended Exploration"* (ICML 2023) | Extends Eigenoptions to continuous domains by using a neural approximation of the graph Laplacian. |
| [h-DQN](jaxhrl/h-DQN.py) | Kulkarni et al., *"Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation"* (2016) | Extends DQN to a two-level hierarchy, where a meta-controller sets goals and a lower-level controller pursues them. Both levels are trained with goal-conditioned deep Q-learning. |
| [Option Keyboard](jaxhrl/option_keyboard.py) | Barreto et al., *"The Option Keyboard: Combining Skills in Reinforcement Learning"* (NeurIPS 2019) | Combines a set of base options into new ones by linearly mixing their cumulants with Successor Features and GPI. |
| [Option Critic](jaxhrl/option_critic.py) | Bacon, Harb & Precup, *"The Option-Critic Architecture"* (AAAI 2017) |  Extends policy-gradient theorem to options to learn them end-to-end. |
| [MOC](jaxhrl/MOC.py) | Klissarov & Precup, *"Flexible Option Learning"* (NeurIPS 2021) |  Flexible Option Critic extends intra-option learning to update all options consistent with primitive action chosen, boosting data efficiency. |
| [PPO Option Critic](jaxhrl/PPOC.py) | Klissarov et al., *"Learning Options End-to-End for Continuous Action Tasks"* (NeurIPS 2017 worskhop) | Extends option critic to continuous environments using PPO. |
| [HiPPO](jaxhrl/HiPPO.py) | Li, Florensa, Clavera & Abbeel, *"Sub-Policy Adaptation for Hierarchical Reinforcement Learning"* (ICLR 2020) | Approximates policy gradient by assuming options are maximally diverse, skills are fixed length sampled from a cat distribution, no learnt termination function. |
| [HAC](jaxhrl/HAC.py) | Levy et al., *"Learning Multi-Level Hierarchies with Hindsight"* | Uses hindsight experience replay to relabel unsuccessful trajectories as achieving a different goal to help sparsity.|
| [HierQ](jaxhrl/HierQ.py) | Levy et al., *"Learning Multi-Level Hierarchies with Hindsight"* (Algorithm 2, Appendix) | From the HAC paper appendix for discrete environments |
| [METRA](jaxhrl/METRA.py) | Park, Rybkin & Levine, *"Scalable Unsupervised RL with Metric-Aware Abstraction"* (ICLR 2024) | Discovers diverse scalable set of skills unsupervised that cover as much of the state space as possible by maximising temporal distance between skills.|

## Verification

[`verification/`](verification/)
contains standalone scripts that import each algorithm's actual network and
loss code and test it against toy environments from the original papers. Full write-ups, plots, and numbers are in
[`verification/REPORT.md`](verification/REPORT.md).

Rerun any check with e.g. `python verification/<algorithm>_verify.py`.

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

