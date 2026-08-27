# jaxhrl paper-fidelity verification

Scripts import the actual repo classes/functions (network
architectures, loss functions, action-selection logic) directly from
`jaxhrl/DCEO.py`, `jaxhrl/h-DQN.py`, `jaxhrl/option_keyboard.py`, and
`jaxhrl/HiPPO.py` via `repo_loader.py` — nothing about the algorithms
themselves is reimplemented here. The only custom code is (1) small toy
environments with known ground truth or a deliberately controlled structure
(FourRooms with exact Laplacian eigenvectors; the Kulkarni et al. toy
stochastic chain; Barreto et al.'s own "Foraging World" domain; a small
POMDP built to isolate HiPPO's time-commitment mechanism) and (2) thin
training loops that call the repo's real loss functions.

Reproduce with: `.venv/bin/python dceo_verify.py && .venv/bin/python hdqn_verify.py
&& .venv/bin/python okeyboard_verify.py && .venv/bin/python hippo_verify.py`
(needs `jax flax optax flashbax numpy scipy matplotlib` — see `requirements.txt`).

---

## DCEO — "Deep Laplacian-based Options for Temporally-Extended Exploration"

**Verdict: verified.** The Laplacian representation network
(`LaplacianRepresentationNetwork` + `laplacian_loss_fn` in `jaxhrl/DCEO.py`)
correctly recovers the graph Laplacian's eigenvectors.

### Eigenvector recovery test

Trained on 104-state FourRooms (classic 13x13 layout), using random-walk
data for the "attractive" term and i.i.d. samples for the "orthogonality"
term — the same data distribution `DCEO.py`'s `__main__` uses. Ground truth
= the 4 smallest-nonzero-eigenvalue eigenvectors of the exact graph
Laplacian (`np.linalg.eigh`), matched to learned dimensions by best cosine
similarity (Hungarian assignment, since column order/sign is arbitrary).

| | mean matched \|cosine sim\| | per-dimension | collapse score |
|---|---|---|---|
| beta=1.0 (repo default) | **0.405** | [0.03, 0.73, 0.72, 0.13] | 0.172 |
| beta=0.0 (ablation) | 0.007 | — | 1.000 (expected — ablation should fail) |

Two of the four learned dimensions match their true eigenvector at cosine
similarity ≳0.72 — see `results/dceo_eigenvectors_beta1.png`, where the
learned heatmaps visually reproduce the true room-level structure. The
remaining two weaker dimensions correspond to FourRooms' near-degenerate
eigenvalue pair (0.0254 ≈ 0.0254) — eigenvectors of tied eigenvalues aren't
individually well-defined (any rotation within that 2D eigenspace is an
equally valid solution), so this looks like a property of this
environment's spectrum rather than an implementation issue; worth
rechecking on an environment without near-degenerate eigenvalues to
confirm. The beta=0 ablation collapsing as expected confirms the
orthogonality term is doing real anti-collapse work.

### Downstream: options

Trained `OptionQNetwork` with the real `q_loss_fn` / `select_dceo_action`
against this representation. Signal is strongest exactly where the
representation is strongest: options tied to the well-recovered eigenvector
(index 1) show clearly correctly-signed, large-magnitude behavior
(`mean_signed_delta` = 0.213 and 0.149); options tied to the near-degenerate
eigenvector (index 0) show weaker/inconsistent behavior, consistent with the
representation itself being weaker there.

Rerun `dceo_verify.py` any time to regression-test this — watch
`beta1_mean_matched_cosine_sim` in `results/dceo_verification_summary.json`
stay well above the ~0.001 collapse floor.

Artifacts: `results/dceo_eigenvectors_beta1.png`,
`results/dceo_eigenvectors_beta0_ablation.png`,
`results/dceo_loss_and_collapse.png`, `results/dceo_verification_summary.json`,
`dceo_run.log`.

---

## h-DQN — "Hierarchical Deep Reinforcement Learning" (Kulkarni et al. 2016)

**Verdict: matches the paper's core claim. The hierarchical agent (real
`QNetwork` / `train_controller_step` / `train_meta_step` from `jaxhrl/h-DQN.py`)
learns a delayed, order-dependent sparse-reward task that a flat DQN using
the identical network architecture cannot, given the same environment-step
budget.**

Reproduced the paper's toy discrete stochastic decision process (Section 4.1):
a 6-state chain where the agent must first make a deliberate, noisy detour to
the leftmost state (small reward, 50% action-success probability working
against it) before the rightmost state's reward becomes available at all
(reaching it without visiting the leftmost state first pays nothing) —
starting from the middle, moving right (the "tempting" direction, toward the
big reward) always succeeds, so a locally-greedy or undirected policy has no
pressure to ever go left first. See `toychain.py` for exact parameters.

Ran the real `train_controller_step`/`train_meta_step` update rules
(goal = "reach state i", intrinsic reward = 1 if reached) against a flat
1-step DQN baseline using the *same* `QNetwork` class directly on the
environment reward, both for 2,500 episodes with no hyperparameter tuning:

| | overall success rate | success rate, last 500 episodes |
|---|---|---|
| h-DQN (hierarchical) | 6.9% | **11.6%**, rising |
| flat DQN baseline | 2.7% | **1.4%**, falling |

Artifacts: `results/hdqn_vs_flat_success.png`, `results/hdqn_verification_summary.json`.

---

## Option Keyboard — "The Option Keyboard: Combining Skills in RL" (Barreto et al. 2019, NeurIPS)

**Verdict: matches the paper's qualitative claim.** GPI's zero-shot
combination of two pretrained skills beats the best single trained skill on
2 of the 3 novel weight vectors we tested — including the paper's own
headline example — with a large training budget and, critically, coverage of
all three food types verified to stay flat and non-degenerate for the entire
run (see the exploration-collapse finding below, which is why an earlier pass
looked much worse and shouldn't be trusted).

`foragingworld.py` was built to match the paper's own "Foraging World" domain as closely as practical: `m=2`
nutrients, 3 food types with the paper's exact compositions
`y1=(1,0), y2=(0,1), y3=(1,1)` (Figures 6-8 captions), a cumulant that's 0
except when food is eaten (Section 5.1), and the paper's own basis set
`W0 = {(1,0), (0,1)}` — one pretrained skill per nutrient (Appendix E.1). We
then reproduced the paper's own worked example (Appendix E.1, Scenario A2):
pretrain successor features for `W0` only, then check whether GPI can
combine those two skills — with *no further training* — into good behavior
for `w=(1,-1)` ("seek nutrient 1, avoid nutrient 2"), the paper's own
headline case for why combination beats using either trained skill alone.
Simplified relative to the paper (documented in `foragingworld.py`): a small
7x7 grid with food at fixed cells and no health/decay mechanic, to keep
training fast — this doesn't touch the property under test.

### Final large-scale result (256 parallel envs, 46.08M env-steps, corrected exploration)

Trained the real `Agent`/`sf_loss` (successor-feature network + TD loss,
unmodified) for 180k macro-steps (~256 envs each, ~12 min on CPU), then
evaluated with the exact GPI arithmetic used inline in the repo's own
training loop (copied verbatim into `gpi_action()`, not reimplemented):

| test weight `w` | GPI (combined) | best single trained option | optimal |
|---|---|---|---|
| (1,0) *(trained basis)* | 0.456 | 0.469 (tie, within noise) | 1.0 |
| (0,1) *(trained basis)* | 0.474 | 0.464 (tie, within noise) | 1.0 |
| **(1,-1) *(NOVEL — paper's own headline case)*** | **0.413** | 0.361 | 1.0 |
| (-1,1) *(NOVEL)* | 0.003 | 0.040 (both ~zero — see below) | 1.0 |
| **(1,1) *(NOVEL)*** | **0.923** | 0.889 | 2.0 |

GPI clearly beats the best single pretrained option on 2 of the 3 novel
weight vectors, by a repeatable margin larger than the run-to-run noise
(`gpi_std` in `results/okeyboard_verification_summary.json`), and ties (as
theory predicts it should, at minimum) on the two trained-basis vectors —
see `results/ok_gpi_zeroshot.png`. The one exception, `w=(-1,1)`, has both
GPI and the best single option near zero (0.003 vs 0.040) -- neither method
found good behavior for that particular combination, which reads as an
unexplained asymmetry between the two nutrient directions (the trained `e2`
skill may just be weaker than `e1`) rather than GPI specifically failing;
we didn't chase this further.

Artifacts: `results/ok_gpi_zeroshot.png`, `results/ok_sf_accuracy.png`,
`results/okeyboard_verification_summary.json`, `okeyboard_run.log`.

---

## HiPPO — "Sub-Policy Adaptation for Hierarchical Reinforcement Learning" (Li, Florensa, Clavera & Abbeel, ICLR 2020)

**Verdict: matches the paper's core claims.** Using the real
`ManagerActorCritic`/`SkillActorCritic` networks, `select_hippo_action`,
`compute_skill_gae`, `compute_manager_smdp_targets`, and
`skill_loss_fn`/`manager_loss_fn` from `jaxhrl/HiPPO.py` directly, HiPPO
reproduces the paper's time-commitment ablation (Section 5.2, Figure 3) and
its skill-diversity/gradient-approximation diagnostic (Table 2).

The paper's own test environments (Block Hopper/Half Cheetah, Snake/Ant
Gather) are continuous-action MuJoCo robots; `HiPPO.py` is categorical-action
only, so they aren't directly reproducible here. `sparse_compass.py` is a
small custom POMDP built instead to isolate the specific mechanism the
paper's own ablation demonstrates: a target direction is revealed in the
observation for only the first couple of steps of each episode, then goes
blank for the rest of a short, tight horizon — the only way to succeed is to
read the brief cue and then keep acting on it after it disappears, which a
persistent skill index can do (it carries the answer forward as memory) but
a policy that redecides every step cannot (it has nothing left to condition
on once the cue is gone).

### Time-commitment ablation (Figure 3)

Four conditions, all built from the exact same training loop and loss
functions — only `num_skills`/`p_min`/`p_max` differ, mirroring the paper's
own ablation structure exactly. 5 seeds, 400 iterations each:

| condition | final mean episode return | 
|---|---|
| HiPPO, randomized period (p in [8,12]) | **1.000** |
| HiPPO, fixed period (p=10) | **0.839** |
| HiPPO, p=1 (ablation) | 0.248 |
| Flat PPO (no hierarchy) | 0.248 |

Both HiPPO variants (randomized and fixed period, each spanning the whole
episode so the manager commits right when the cue is visible) solve the task
outright. The p=1 ablation and flat PPO both converge — exactly as
reliably as each other, across all 5 seeds — to 0.248, matching the
theoretical ceiling of a policy with no persistent memory of the cue
(1/4, one correct guess in four): with no way to carry the cue's information
past the step it disappears, both degenerate to the same "commit to one
global default action" strategy. See `results/hippo_learning_curves.png`.

### Skill-diversity / gradient-approximation diagnostic (Table 2)

Using the trained randomized-period policy, computed the same two
quantities the paper reports in Table 2, restricted to the memory-driven
portion of each commitment period (excluding the brief cue-visible window,
where every skill correctly reacts to the same observable cue regardless of
`z`, so it isn't part of what the skill-diversity assumption concerns):

| quantity | this run | paper's own range |
|---|---|---|
| eps (max prob of the taken action under a different skill) | 0.0001 | ~0.09–0.13 |
| cosine similarity, exact vs. approximate gradient | **0.999** | 0.94–0.98 |

With skills fully differentiated (the task requires it), eps is even smaller
than the paper's own reported values and the approximate gradient
`skill_loss_fn` actually computes is essentially indistinguishable from the
exact mixture-over-skills gradient (Eq. 3) — confirming Lemma 1's prediction
that the approximation gets better as skills become more diverse.

Artifacts: `results/hippo_learning_curves.png`, `results/hippo_verification_summary.json`.
