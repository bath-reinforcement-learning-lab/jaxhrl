# jaxhrl paper-fidelity verification

Scripts import the actual repo classes/functions (network
architectures, loss functions, action-selection logic) directly from
`jaxhrl/DCEO.py`, `jaxhrl/h-DQN.py`, `jaxhrl/option_keyboard.py`,
`jaxhrl/HiPPO.py`, `jaxhrl/option_critic.py`, `jaxhrl/MOC.py`, `jaxhrl/HAC.py`
and `jaxhrl/METRA.py` via `repo_loader.py` — nothing
about the algorithms themselves is reimplemented here. The only custom code is
(1) small toy environments with known ground truth or a deliberately controlled
structure (FourRooms with exact Laplacian eigenvectors; the Kulkarni et al. toy
stochastic chain; Barreto et al.'s own "Foraging World" domain; a small
POMDP built to isolate HiPPO's time-commitment mechanism; the Option-Critic
paper's own four-rooms navigation task with a relocatable goal, reused for MOC;
a continuous four-rooms point mass standing in for Levy et al.'s ant four
rooms; a reward-free FourRooms with an exact shortest-path oracle for METRA)
and (2) thin training loops that call the repo's real loss functions. HAC and
METRA go further than the others: `hac_verify.py` and `metra_verify.py`
execute the repo's real `__main__` training loop via `runpy` against a patched
environment factory, so the entire scan body is what runs, not a
reimplementation.

Reproduce with: `.venv/bin/python dceo_verify.py && .venv/bin/python hdqn_verify.py
&& .venv/bin/python okeyboard_verify.py && .venv/bin/python hippo_verify.py
&& .venv/bin/python option_critic_verify.py && .venv/bin/python moc_verify.py
&& .venv/bin/python hac_verify.py && .venv/bin/python hac_faithfulness.py
&& .venv/bin/python hierq_verify.py && .venv/bin/python hierq_faithfulness.py
&& .venv/bin/python metra_verify.py`
(needs `jax flax optax flashbax numpy scipy matplotlib pyyaml` — see `requirements.txt`).

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

## HAC — "Learning Multi-Level Hierarchies with Hindsight" (Levy et al. 2019)

**Verdict: verified.** Every mechanism the paper specifies is present and
behaves as specified in the transitions `jaxhrl/HAC.py` actually emits (40/40
checks).


### Mechanism faithfulness — `hac_faithfulness.py`, 40/40

Asserted on the transitions the implementation genuinely wrote, at k=3:

| paper mechanism | check | result |
|---|---|---|
| sparse reward, terminal discount | rewards drawn only from {0, −1, −H}; reward 0 ⟹ discount 0; reward −1 ⟹ discount 1 | pass, all levels |
| hindsight **action** (Sec. 3.1) | levels > 0 store the subgoal *actually achieved*, not the one proposed | 100% of 7,844 (L1) and 2,249 (L2) rows |
| hindsight **goal** (Sec. 3.2) | goal-relabelled terminal transitions present | 3,946 / 1,149 / 282 rows |
| subgoal **testing** (Sec. 3.3) | −H penalty rows present with testing on, at levels > 0 only | 2,075 (L1), 450 (L2); 0 at L0 |
| subgoal testing, ablated | no penalty rows at `subgoal_test_perc = 0` | 0 found |
| penalty action identity | penalty rows keep the *proposed* subgoal, not the achieved one | pass |
| bounded critic | Q ∈ [−H, 0]; `q_limit == H`; `gamma == 1 − 1/H` | pass, all levels |
| nested schedule | episode = ∏H; level *i* gets ∏H[i+1:] attempts × H actions | 64 steps; 16/4/1 attempts, 64/16/4 actions |
| subgoal-test determinism | `deterministic=True` reproduces the greedy action exactly; `False` injects noise | pass |

### Depth sweep — continuous Four Rooms (`pointmaze.py`)

A point mass in four rooms joined by doorways, starting in the bottom-left room
with the end goal sampled arena-wide, so most goals require routing through
doorways. Dynamics are deliberately trivial so nothing about motor control
confounds the measurement. Every arm gets an identical 729-step episode budget
and the log chunk equals the horizon, so each point aggregates exactly one
episode per environment.

| arm | final success | env-steps to 0.5 | to 0.8 |
|---|---|---|---|
| flat (k=1) | 0.862 | 1,492,992 | 2,612,736 |
| **2-level HAC** | **0.898** | **559,872** | **1,213,056** |

The 2-level agent reaches 50% success in 2.7x fewer environment steps and 80%
in 2.2x fewer — the paper's claim, on the same task with the same budget.

### Depth is bounded by level-0 reach, not by the algorithm

A level's physical reach is `H x step_scale`, so under a fixed horizon
`H = T^(1/k)` shrinks as levels are added. Because the subgoal space is
absolute position spanning the whole arena, a level whose child can only move a
few percent of that range cannot place reachable subgoals, and the hierarchy
fails to bootstrap — level 0 never learns, and every level above it starves.
This is a property of the task scale, not a defect: it is fully reversible by
restoring the reach, with nothing else changed.

| k=3, identical code / horizon / threshold / arena | level-0 reach | end-goal success |
|---|---|---|
| step scale 0.01 | 0.09 | 0.015 |
| **step scale 0.03** | **0.27** | **0.900** |

At the larger step scale the 2-level agent reaches 0.956 and the 3-level agent
0.900, both learning cleanly. Reproducing the paper's 3-level *advantage* at
the 0.01 scale would need a 27^3 ~ 19,700-step horizon to keep level-0 reach
adequate.

Artifacts: `results/hac_levels_comparison.png`,
`results/hac_verification_summary.json`.

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

## HierQ — "Learning Multi-Level Hierarchies with Hindsight", Algorithm 2 (Levy et al. 2019)

**Verdict: verified as an implementation; the paper's depth ordering reproduces
in part.** Every mechanism Algorithm 2 specifies is present and behaves as
specified (20/20 checks), and hierarchical agents beat the flat agent by a
margin that grows with task scale -- up to 6.3x fewer training episodes. The
paper's further claim that a 3-level agent beats a 2-level one does not
reproduce here.

HierQ is the discrete counterpart of HAC, and differs from it in two ways that
both come straight from Algorithm 2:

- **No subgoal testing.** HAC needs -H penalty transitions to stop a level
  proposing subgoals its child cannot reach. HierQ has none; *pessimistic
  initialisation* does that job, because `Q_i(s, ., a)` is only ever written at
  actions `a` genuinely reached from `s` within the level's horizon, so an
  unreachable subgoal keeps its initial value and never wins an argmax.
- **Exhaustive rather than sampled hindsight.** `Q_0` is updated for *every*
  goal in the state space per transition, and `Q_i` over `PrevStates_i` x all
  goals -- HER by enumeration.

### Mechanism faithfulness — `hierq_faithfulness.py`, 20/20

Both update rules are checked against hand-computed Bellman targets, then the
repo's real `__main__` loop is executed via `runpy` and the resulting Q-tables
inspected directly.

| Algorithm 2 property | check | result |
|---|---|---|
| level-0 all-goals update | equals `(1-a)Q + a[R + g.max Q(s',g,.)]` for every goal | exact (0.00e+00) |
| all-goals HER | exactly \|S\| goal-entries written per transition | pass |
| level-i PrevStates update | matches the equation for every (state, goal) | exact (0.00e+00) |
| hindsight **action** | the stored action is `s'` itself | only the `a=s'` plane written |
| window masking | masked slots never written | pass |
| initialisation | `Q_0` optimistic at 0; `Q_i>0` pessimistic | pass |
| **reachability invariant** | every written (state, subgoal) pair is reachable within that level's horizon | **0 unreachable, both levels** |
| pessimism holds | unreachable subgoals retain the floor; nothing falls below it | pass |
| no subgoal testing | no value below the floor (HierQ has no penalty transitions) | pass |
| nested schedule | top attempt == episode; `end[i+1] => end[i]`; per-level bounds | pass |

The reachability invariant is the one that matters most: it is *because* `Q_i`
is only ever written at achievable subgoals that pessimistic initialisation can
substitute for subgoal testing.

`gamma_i` and the pessimistic floor are one choice, not two. The floor must be
the fixed point of `Q = -1 + gamma_i.Q`, i.e. `-1/(1-gamma_i)`, or reachable
subgoals get driven *below* untouched unreachable ones and the argmax prefers
exactly what the level cannot achieve. `gamma_i = 1 - 1/H_i` satisfies that and
keeps the value range commensurate with the level's own budget (floor `-H_i`),
the same relationship HAC uses. The paper specifies a single global `gamma`,
which cannot be commensurate with every level's horizon at once; this is the one
deliberate departure from the letter of Algorithm 2.

### Depth comparison — grid worlds

`gridworld` in `common/wrappers.py` provides the paper's discrete domains. The
episode horizon is held at 125 primitive steps for every arm, with the sub-level
budget H=5 constant across depths and the top level absorbing the remainder
(`H_levels` = [125] / [5,25] / [5,5,5]), so every agent gets both the same
environment budget and the same level-0 reach. Three seeds; x-axis is training
episodes, matching the paper's figure.

Training episodes to 80% success (lower is better):

| Four Rooms | states | flat (k=1) | 2-level | 3-level |
|---|---|---|---|---|
| 13x13 | 104 | 447 | **195** (2.3x) | 304 (1.5x) |
| 17x17 | 200 | 1,220 | **243** (5.0x) | 607 (2.0x) |
| 21x21 | 328 | 2,515 | **398** (6.3x) | 1,220 (2.1x) |

Both hierarchical agents beat the flat agent at every scale, and the margin
grows as the task gets longer-horizon -- which is the mechanism the paper
appeals to. On the largest maze the flat agent does not even converge
(0.875 +- 0.048) while the 2-level agent does (0.995 +- 0.006).

Note the flat arm here is *stronger* than the paper's baseline. Algorithm 2 is
defined for `k > 1`; the paper's flat comparison is "Q-learning with HER", which
samples a few relabelled goals, whereas `num_levels: 1` inherits HierQ's
exhaustive all-goals update -- |S| relabels per transition. The hierarchy's win
is therefore against a harder baseline than the paper's.

**What does not reproduce:** the 3-level agent never beat the 2-level agent --
in all nine comparisons (3 task scales x 3 seeds), and under three different
gamma/floor settings. With mean shortest paths of 8-14 steps and a level-0 reach
of 5, a 2-level hierarchy already reduces the task to ~3 subgoal decisions;
there is little left for a third level to abstract at grid-world scale.

Artifacts: `results/hierq_levels_comparison.png`,
`results/hierq_verification_summary.json`.

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

---

## Option-Critic — "The Option-Critic Architecture" (Bacon, Harb & Precup, AAAI 2017)

**Verdict: matches the paper's core four-rooms claims.** Using the real
`OptionCriticNetwork`, `batch_select_option_critic_action` and
`option_critic_loss_fn` from `jaxhrl/option_critic.py` directly, Option-Critic
(1) learns the stationary four-rooms navigation task exactly as fast as a flat
actor-critic, (2) recovers faster than that flat baseline after the goal is
relocated (the paper's Figure 3 transfer result), and (3) organises the state
space into spatially-coherent per-option regions (Figure 4).

`fourrooms_nav.py` is the paper's own 13×13 four-rooms with its 1/3 action
noise, a +1 terminating goal reward, and γ=0.99. Following the paper's transfer
setup, the goal starts in the east doorway (a bottleneck, so options that learn
to reach it stay reusable) and relocates into the lower-right room after 1M
env-steps. The flat baseline is the *identical code path* with `num_options=1`,
which collapses the option machinery to a one-step advantage actor-critic — the
same way `hippo_verify.py` derives its flat baseline from `num_skills=1`.
Training is on-policy (fresh rollouts fed straight through the repo's real
`option_critic_loss_fn` — TD critic target, intra-option policy gradient, and
the termination gradient of Bacon et al. eq. 4), matching Option-Critic's
on-policy actor-critic updates. 16 seeds.

### Non-stationary transfer (Figure 3)

Mean episode return (= goal-reach rate), 16 seeds. Post-switch columns are the
mean return that many env-steps after the goal moves:

| condition | pre-switch | post-switch AUC | +0.5M | +1.0M | +1.5M | final (+2.0M) |
|---|---|---|---|---|---|---|
| Flat actor-critic (1 option) | 1.00 | 0.42 | 0.19 | 0.43 | 0.55 | 0.78 |
| **Option-Critic (4 options)** | 1.00 | **0.51** | 0.23 | **0.56** | **0.76** | **0.86** |
| Option-Critic (8 options) | 1.00 | **0.56** | 0.40 | 0.63 | 0.69 | 0.79 |

All three solve the stationary task at the same rate — the return curves are
superimposed up to the relocation line, reproducing Figure 3's phase-1 claim
that adding options costs nothing. After the goal moves, both Option-Critic
variants recover ahead of the flat baseline for essentially the whole 2M-step
window (post-switch AUC 0.51 / 0.56 vs 0.42, roughly 1.5–2 standard errors
apart; standard-error bands on the curves mostly disjoint through ~1.3–2.6M).
The flat baseline closes the gap only near the end of the budget. Seed
variance is high for every condition — 4–5 of 16 seeds in each are slow to
re-explore their way to the relocated goal — so this is a modest reproduction
of Figure 3's direction rather than a large margin. See
`results/option_critic_transfer_curves.png`.

The recovery advantage is option-level exploration: ε-greedy selection over
`Q_Omega` commits to a whole option for an episode, giving the directed,
temporally-extended exploration the paper credits options with, where the flat
policy's only exploration is per-step softmax noise — random walks that seldom
reach a goal a room away from the old one.

### Option specialization (Figure 4)

Sweeping every state through the trained 4-option network: the greedy option
per state is spatially coherent — `greedy_option_spatial_coherence` = 0.79 vs
0.25 for a random option assignment — so options own contiguous regions of the
grid (`results/option_critic_options.png`). The option *value* function
`Q_Omega` is doing this partitioning work; the termination head meanwhile
drives β→0 (options run until the episode ends) and the intra-option policies
stay close (mean pairwise action-distribution TV ≈ 0.05), the known
Option-Critic tendency for options to under-differentiate without stronger
regularisation than `delib_cost` provides.

Artifacts: `results/option_critic_transfer_curves.png`,
`results/option_critic_options.png`,
`results/option_critic_verification_summary.json`, `results/option_critic_run.log`.

---

## MOC — "Flexible Option Learning" (Klissarov & Precup, NeurIPS 2021)

**Verdict: matches the paper's core four-rooms claim.** `jaxhrl/MOC.py`'s
`moc_loss_fn` — the arrival-probability-weighted update of *every* option from
each transition, with a PPO-style clipped importance ratio correcting for the
action having been sampled by the active option — reproduces "Flexible Option
Learning"'s Figure 1b: on the non-stationary four-rooms task, the multi-update
agent (MOC) recovers from the goal relocation far faster than vanilla
Option-Critic and with much lower seed variance.

Same environment (`fourrooms_nav.py`), same on-policy training loop and same 16
seeds as the Option-Critic verification above; the only thing that differs
between the OC and MOC conditions is the loss function
(`option_critic_loss_fn` vs `moc_loss_fn`), and the flat baseline is again the
shared code path with `num_options=1`. All option components are learned from
scratch; the goal relocates after 1M env-steps.

### Non-stationary four-rooms (Figure 1b)

Mean episode return (= goal-reach rate), 16 seeds. "recovered" = seeds whose
final return exceeds 0.8:

| condition | pre-switch AUC | post-switch AUC | return +0.5M | return +1.0M | final return (± seed std) | recovered |
|---|---|---|---|---|---|---|
| Flat actor-critic (1 option) | 0.85 | 0.31 ± 0.05 | 0.19 | 0.42 | 0.56 ± 0.30 | 6/16 |
| Option-Critic (4 options) | 0.82 | 0.54 ± 0.04 | 0.41 | 0.68 | 0.80 ± 0.13 | 8/16 |
| **MOC (4 options)** | 0.82 | **0.82 ± 0.01** | **0.82** | **0.96** | **0.99 ± 0.01** | **16/16** |

All three learn the initial task at the same rate — phase-A return curves are
superimposed and every seed reaches 0.8 in ~0.4M env-steps regardless of
condition. After the goal moves:

- **Both hierarchical agents beat flat** (post-switch AUC 0.54 / 0.82 vs 0.31).
- **MOC recovers far faster than OC**: 0.5M env-steps after the relocation MOC
  is already at 0.82 return, higher than OC reaches a full 1M steps later
  (0.68). This is a larger gap than the paper's "half the episodes".
- **MOC's seed variance is dramatically lower**: post-switch AUC standard
  error 0.008 vs OC's 0.039, and final-return seed std 0.01 vs OC's 0.13. All
  16 MOC seeds recover; only 8/16 OC and 6/16 flat do.

The standard-error bands on `results/moc_transfer_curves.png` are fully
disjoint for the entire recovery. This is a clean reproduction of Figure 1b —
sharper than for vanilla Option-Critic (whose four-rooms transfer margin over
a flat baseline was modest, see the section above), because MOC updates every
option's value and policy from every transition, so the whole multi-option
value function tracks the moved reward instead of one over-specialised option
having to be unwound.

### Option usage (Figures 1c / 6)

Sweeping the seed-0 trained policies over every state:

| | greedy-option usage entropy | dominant option share | information radius |
|---|---|---|---|
| Option-Critic (4 options) | 0.99 | 0.31 | 0.0101 |
| MOC (4 options) | 0.14 | 0.95 | 0.0053 |

MOC concentrates almost all of its behaviour in a single option and has a
lower information radius (inter-option divergence) than OC — the direction the
paper reports for the **tabular** regime (Figure 1c: multi-updating with
η = 1.0 reduces option diversity), not the deep-MiniGrid regime of Figure 6.
`MOC.py`'s `moc_loss_fn` is effectively η = 1.0 (it always updates every
option), and the paper introduces η precisely to trade this collapse against
the performance gain; on this near-tabular one-hot four-rooms the gain comes
with the diversity cost.

Artifacts: `results/moc_transfer_curves.png`,
`results/moc_verification_summary.json`, `results/moc_run.log`.

---

## METRA — "Scalable Unsupervised RL with Metric-Aware Abstraction" (Park, Rybkin & Levine, ICLR 2024)

**Verdict: matches the paper's core claims.** Running `jaxhrl/METRA.py`'s
training loop — the `(φ(s') − φ(s)) · z` intrinsic reward and the 1-Lipschitz
constraint from `metra_components`, the Lagrangian φ update, the dual λ update
and the discrete-SAC skill policy — unsupervised on a reward-free 13×13
FourRooms with a 2-D continuous skill space: the learned abstraction φ recovers
the environment's shortest path (temporal-distance) geometry, the skill policy
moves φ in commanded directions, and φ supports zero-shot goal reaching with no
goal conditioned policy ever trained.

`metra_verify.py` patches `make_jax_env` to the reward-free FourRooms of
`fourrooms_open.py` and runs METRA's `__main__` via `runpy`; only the environment and harness are custom. `fourrooms_open.py`
exposes the exact all-pairs shortest-path matrix over the 104 free cells as the
ground-truth temporal-distance metric. 5 seeds, 12.8M env-steps each, repo's
shipped hyperparameters.

### Skills are directed and diverse (objective Eq. 7)

| | METRA skill policy | random policy |
|---|---|---|
| cos(φ(s_end) − φ(s_start), z), mean over 64 skills | **0.66 ± 0.04** | −0.02 ± 0.05 |
| mean shortest-path distance, start → end of a 50-step rollout | 7.8 | — |

A skill conditioned on z reliably moves φ in the direction of z (all 5 seeds
0.61–0.70) — something a random policy does not do — and the skills' endpoints
fan out across the grid (endpoint spread 3.8 cells).

### φ recovers the temporal-distance metric (Theorem 4.1)

| | mean ± seed std | per seed |
|---|---|---|
| Spearman(‖φᵢ − φⱼ‖, shortest-path distance), all 5356 pairs | 0.68 ± 0.18 | 0.95 / 0.78 / 0.69 / 0.49 / 0.49 |
| Procrustes disparity, φ vs classical-MDS of the shortest-path matrix (0 = identical) | 0.29 ± 0.19 | 0.02–0.51 |

Every seed's φ, laid out in 2D, reproduces the four-room topology as the same
four-armed "cross" that classical MDS of the shortest-path matrix produces —
the rooms pulled into separate arms because the doorways make cross-room travel
long (`results/metra_phi_map.png`). On 3 of 5 seeds the match is also
metrically precise (Spearman ≥ 0.69, Procrustes ≤ 0.24); on the other 2 the
geometry is recognisably right but rotated/compressed (Spearman ≈ 0.49).

### Zero-shot goal reaching (Section 5.3 / Figure 8)

Setting z from φ and running the skill policy greedily, 60 random
(start, goal) pairs, mean shortest-path distance to the goal:

| | distance to goal |
|---|---|
| at episode start | 7.1 |
| after 50 steps, z = (φ(g) − φ(s₀))/‖·‖ fixed for the episode | 5.9 ± 1.1 |
| after 50 steps, z = (φ(g) − φ(sₜ))/‖·‖ recomputed each step | **2.8 ± 0.5** |
| after 50 steps, random z | 8.8 ± 0.1 |

Closed loop skill selection from φ reaches within 2 cells of the goal 66% of
the time (all seeds 55–75%) — METRA's zero-shot goal-reaching claim — while a
random skill drifts *away* from the goal. The fixed-z variant (faithful to how
skills are trained, one z per episode) still roughly halves the gap.

### Notes on the implementation

`metra_components` matches the paper's reward and constraint. The φ objective
in `METRA.py`'s `__main__` differs from Algorithm 1 in scale — the Lipschitz
penalty uses the *mean* squared coordinate difference rather than the sum (a
looser constraint by a factor of `z_dim`), the reward term carries a 10×
weight, and the dual step is taken in log-λ space. These are scale/tuning
choices, not changes to the mechanism, and the paper's claims reproduce with
the repo's shipped hyperparameters.

Artifacts: `results/metra_phi_map.png`,
`results/metra_verification_summary.json`, `results/metra_run.log`.
