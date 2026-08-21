# jaxhrl

Hierarchical Reinforcement Learning algorithms implemented in pure JAX.

## Why JAX

Every algorithm here — environment stepping, agent forward/backward passes,
replay, and the training loop itself — is written to live entirely on-device
and run under `jax.jit`/`jax.lax.scan`. There's no per-step Python/host
round-trip. That has one big practical consequence: **training scales to
hundreds or thousands of parallel environments for free**, via `jax.vmap`,
as long as the environment's `reset`/`step` functions are themselves
jittable (e.g. a [gymnax](https://github.com/RobertTLange/gymnax) env).
Most algorithms here default to `num_envs: 1024`. Swapping in a different
environment doesn't require touching the algorithm code — only the wrapper
in [`jaxhrl/common/wrappers.py`](jaxhrl/common/wrappers.py) needs adjusting
to expose that environment's `reset_fn`/`step_fn`/observation and action
shapes in the form each algorithm expects.

## Algorithms

| Algorithm | Paper | Status |
|---|---|---|
| [DCEO](jaxhrl/DCEO.py) | Klissarov & Machado, *"Deep Laplacian-based Options for Temporally-Extended Exploration"* (ICML 2023) | ✅ Verified — the Laplacian representation correctly recovers the true graph eigenvectors; see [`verification/REPORT.md`](verification/REPORT.md) |
| [h-DQN](jaxhrl/h-DQN.py) | Kulkarni et al., *"Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation"* (2016) | ✅ Verified — matches the paper's own toy-MDP result |
| [Option Keyboard](jaxhrl/option_keyboard.py) | Barreto et al., *"The Option Keyboard: Combining Skills in Reinforcement Learning"* (NeurIPS 2019) | ✅ Verified at scale (46M env-steps) — GPI's zero-shot skill combination matches the paper's own worked example |
| [HiPPO](jaxhrl/HiPPO.py) | Manager/skill hierarchical PPO (learned skills, SMDP-level manager) | ⬜ Not yet verified |
| [HAC](jaxhrl/HAC.py) | Levy et al., *"Learning Multi-Level Hierarchies with Hindsight"* | 🚧 Stub — not yet implemented |
| [HIRO](jaxhrl/HIRO.py) | Nachum et al., *"Data-Efficient Hierarchical Reinforcement Learning"* | 🚧 Stub — not yet implemented |
| [option_critic](jaxhrl/option_critic.py) | Bacon, Harb & Precup, *"The Option-Critic Architecture"* | 🚧 Stub — not yet implemented |

## Verification

Paper fidelity isn't assumed — it's checked. [`verification/`](verification/)
contains standalone scripts that import each algorithm's actual network and
loss code (not reimplementations) and test it against toy environments with
known ground truth — exact Laplacian eigenvectors, a paper's own worked
example, that kind of thing. Full write-ups, plots, and numbers are in
[`verification/REPORT.md`](verification/REPORT.md). Summary:

- **DCEO**: the Laplacian representation network recovers the true graph
  Laplacian eigenvectors (cosine similarity ~0.7+ against an exact
  ground-truth eigendecomposition on FourRooms, on the well-separated
  eigenvalues).
- **h-DQN**: reproduced Kulkarni et al.'s own toy stochastic decision
  process — the hierarchical agent learns it, a flat DQN baseline with the
  same network and step budget doesn't.
- **Option Keyboard**: reproduced Barreto et al.'s own "Foraging World"
  worked example at scale — GPI combines two pretrained skills into
  optimal-ish behavior for a weight vector neither was trained on.
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

### Known issue

`jaxhrl/h-DQN.py` and `jaxhrl/option_keyboard.py` both import
`jaxhrl.common.jax_wrappers`, a module that doesn't exist in this repo (only
`jaxhrl.common.wrappers` does) — so as of now, neither can run end-to-end via
the command above until that import is fixed or the module is added.
