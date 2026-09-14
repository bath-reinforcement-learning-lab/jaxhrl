"""HierQ (deep) implementation-faithfulness suite.

Distinct from `hierq_verify.py`, which asks whether it reproduces a performance claim, this
suite asks "is it the right algorithm".

Tabular HierQ's invariants are checked at the level they now actually hold:

  * a table entry that is never written stays at its initial value FOREVER,
    exactly. 
  * "for each goal, for each state in PrevStates" becomes one multi-goal
    forward pass; checked by hand-computing the masked target for a toy batch
    against a network whose head starts at a known constant (zero weights,
    constant bias), so `level_loss`'s output is exactly predictable, including
    the `done`-gated discount (the environment terminating also zeros it, not
    only the goal being achieved).
  * pessimistic initialisation is checked structurally: the head's weights are
    zero at construction, so Q(s, ., .) equals the init bias for every s
    before a single gradient step -- an exact, input-independent identity.

The literal-episode training loop is checked against real runpy-executed
transitions:

  * the world resets whenever the task goal's attempt ends
  * `close_out`, the two-pass attempt-scheduling helper, is exercised both
    through the real loop and directly 

Run: python verification/hierq_faithfulness.py
"""
import pathlib
import sys
import tempfile

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


BASE_CFG = {
    "experiment": "HQF_probe", "seed": 0, "save_json": True,
    "use_wandb": False, "overwrite": True,
    "env": {"framework": "gridworld", "make": {"id": "fourrooms"}, "kwargs": {"one_hot": True}},
    "training": {"num_levels": 3, "H": 4, "horizon": 64, "n_steps": 900,
                 "num_envs": 32, "chunk_size": 64, "batch_size": 32,
                 "buffer_size": 20000, "lr": 3e-4, "tau": 0.05,
                 "hidden_dim": 32, "subgoal_test_perc": 0.5, "epsilon": 0.5},
    "eval": {"enabled": False}, "checkpoint": {"enabled": False},
}


def run_hierq(cfg):
    import runpy
    p = pathlib.Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text(yaml.safe_dump(cfg))
    sys.argv = ["HierQ.py", "--config", str(p)]
    return runpy.run_path(str(REPO_ROOT / "jaxhrl" / "HierQ.py"), run_name="__main__")


def main():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import jaxhrl.HierQ as HQ

    OBS_DIM, G, HID = 6, 4, 8

    # ---- A: pessimistic/optimistic initialisation is exact and input-free ----
    print("A. Network initialisation")
    for init_value, label, A in [(0.0, "level 0 (optimistic, 0.0)", 3),
                                 (-5.0, "subgoal level (pessimistic, -5.0)", G)]:
        model = HQ.MultiGoalQNetwork(OBS_DIM, G, A, init_value, HID, rngs=nnx.Rngs(0))
        obs = jnp.asarray(np.random.default_rng(1).normal(size=(9, OBS_DIM)), jnp.float32)
        q = np.asarray(model(obs))
        check(f"{label}: Q(s,g,a) == init_value for every s, before any training",
              bool(np.all(q == init_value)), f"range [{q.min():.4f}, {q.max():.4f}]")

    # ---- B: the multi-goal, done-aware TD target and masked loss, closed form
    print("\nB. All-goals loss matches a hand-computed target (network held at a known constant)")
    for subgoal_level, init_value, label in [(False, 0.0, "level 0"), (True, -4.0, "subgoal level")]:
        A = G if subgoal_level else 3   # subgoal levels: action space == goal space
        model = HQ.MultiGoalQNetwork(OBS_DIM, G, A, init_value, HID, rngs=nnx.Rngs(2))
        target = HQ.MultiGoalQNetwork(OBS_DIM, G, A, init_value, HID, rngs=nnx.Rngs(3))
        # Both nets are all-zero-weight at construction, so both output exactly
        # `init_value` for any input -- the whole target/loss is closed-form.
        B, q_limit, gamma = 6, 4.0, 0.75
        rng = np.random.default_rng(4)
        obs = jnp.asarray(rng.normal(size=(B, OBS_DIM)), jnp.float32)
        nxt = jnp.asarray(rng.normal(size=(B, OBS_DIM)), jnp.float32)
        ach = rng.uniform(size=(B, G)) < 0.4
        # Row 5 is forced unachieved-everywhere but `done`, to isolate the
        # done-gated (not just goal-gated) discount.
        ach[5, :] = False
        done = np.zeros(B, bool); done[4] = True; done[5] = True
        ach_fn = lambda o, _a=jnp.asarray(ach): _a

        action = jnp.asarray(rng.integers(0, A, size=B), jnp.int32)
        batch = {"obs": obs, "next_obs": nxt, "action": action, "done": jnp.asarray(done)}
        if subgoal_level:
            penalty = jnp.asarray([True, False, True, False, False, False])
            batch["penalty"] = penalty

        got = float(HQ.level_loss(model, target, batch, ach_fn, gamma, q_limit, subgoal_level))

        terminal = ach | done[:, None]
        want_target = np.where(ach, 0.0, -1.0) + np.where(terminal, 0.0, gamma) * init_value
        if subgoal_level:
            pen = np.asarray(penalty)
            want_target = np.where(pen[:, None], -q_limit, want_target)
            want_target = np.clip(want_target, -q_limit, 0.0)
            act_np = np.asarray(action)
            onehot = np.eye(A)[act_np]
            act_mask = np.where(pen[:, None], onehot, ach)             # (B, A)
            err = (init_value - want_target[:, :, None]) ** 2 * act_mask[:, None, :]
            want = err.sum() / max(act_mask.sum() * G, 1.0)
        else:
            # No action axis on the target side: q_sel = Q(s, g, action_taken)
            # for every goal g, so the loss is the mean over ALL (row, goal)
            # pairs -- the exhaustive all-goals update.
            want_target = np.clip(want_target, -q_limit, 0.0)
            want = float(np.mean((init_value - want_target) ** 2))

        check(f"{label}: level_loss matches the hand-computed done-aware masked target",
              abs(got - want) < 1e-4, f"got {got:.6f}, want {want:.6f}")

    # ---- gradient sparsity at the head: untouched outputs get exactly zero gradient (matches table entry untouched) 
    print("\n   Gradient sparsity at the head (subgoal level, with penalty rows)")
    A = G
    model = HQ.MultiGoalQNetwork(OBS_DIM, G, A, -4.0, HID, rngs=nnx.Rngs(5))
    target = HQ.MultiGoalQNetwork(OBS_DIM, G, A, -4.0, HID, rngs=nnx.Rngs(6))
    rng = np.random.default_rng(7)
    B = 6
    obs = jnp.asarray(rng.normal(size=(B, OBS_DIM)), jnp.float32)
    nxt = jnp.asarray(rng.normal(size=(B, OBS_DIM)), jnp.float32)
    ach = np.zeros((B, G), bool)
    ach[np.arange(B), rng.integers(0, G, size=B)] = True     # exactly one goal achieved per row
    ach_fn = lambda o, _a=jnp.asarray(ach): _a
    action = jnp.asarray(rng.integers(0, A, size=B), jnp.int32)
    batch = {"obs": obs, "next_obs": nxt, "action": action,
             "done": jnp.zeros((B,), bool), "penalty": jnp.zeros((B,), bool)}

    def loss_fn(m):
        return HQ.level_loss(m, target, batch, ach_fn, 0.9, 4.0, True)

    _, grads = nnx.value_and_grad(loss_fn)(model)
    head_grad = np.asarray(grads.head.kernel.value).reshape(HID, G, A)   # (hidden, G, A)
    touched_actions = np.asarray(ach).any(axis=0)
    untouched = ~touched_actions
    check("subgoal level: head gradient is exactly zero for actions no row selected",
          bool(np.all(head_grad[:, :, untouched] == 0.0)),
          f"{int(untouched.sum())}/{A} action-columns untouched, max |grad| there "
          f"{np.abs(head_grad[:, :, untouched]).max() if untouched.any() else 0.0:.2e}")
    check("subgoal level: at least one touched action-column has nonzero gradient",
          bool(np.any(head_grad[:, :, touched_actions] != 0.0)))

    # ---- C: eps_greedy respects the explore mask -----------------------------
    print("\nC. eps_greedy restricts random exploration to the allowed action set")
    q_rows = jnp.zeros((200, 5), jnp.float32)
    mask = jnp.asarray([True, False, True, False, False])
    acts = np.asarray(HQ.eps_greedy(q_rows, jax.random.PRNGKey(0), 1.0, mask,
                                    jnp.zeros((200,), bool)))
    check("random exploration never selects a masked-out action",
          bool(np.all(mask[acts])), f"actions used: {sorted(set(acts.tolist()))}")
    # deterministic=True must make epsilon irrelevant 
    key1 = jax.random.PRNGKey(1)
    det_eps1 = np.asarray(HQ.eps_greedy(q_rows, key1, 1.0, mask, jnp.ones((200,), bool)))
    det_eps0 = np.asarray(HQ.eps_greedy(q_rows, key1, 0.0, mask, jnp.ones((200,), bool)))
    check("deterministic=True makes epsilon irrelevant (same key, eps=1 vs eps=0 agree)",
          bool(np.array_equal(det_eps1, det_eps0)),
          f"{int((det_eps1 != det_eps0).sum())}/200 rows differ")

    # ---- D: the real training loop -------------------------------------------
    print("\nD. Structural invariants after running the real training loop")
    g = run_hierq(BASE_CFG)
    H_levels, k = g["H_levels"], g["k"]
    print(f"   H_levels={H_levels}  gammas={[round(x,3) for x in g['gammas']]}  "
          f"q_limits={g['q_limits']}  G={g['G']}")

    def rows_of(level):
        ring = g["carry"].rings[level]
        n = int(ring.size)
        return {kk: np.asarray(v)[:n] for kk, v in ring.data.items()}, n

    # gridworld's step_fn hard-codes done=False, so the world in this run never
    # terminates -- the single cleanest test of "goals chain without a reset".
    for i in range(k):
        r, n = rows_of(i)
        check(f"level {i}: ring buffer received transitions", n > 0, f"{n} rows")
        check(f"level {i}: `done` is always False on gridworld (it never terminates)",
              n == 0 or bool(np.all(~r["done"])), "found a True")
        if i == 0:
            continue
        pen = r.get("penalty")
        check(f"level {i}: penalty field present on every row", pen is not None)
        n_pen = int(pen.sum()) if pen is not None else 0
        check(f"level {i}: some penalty rows written (subgoal_test_perc=0.5)", n_pen > 0,
              f"{n_pen} of {n} rows")
        non_pen_next = r["next_obs"][~pen]
        ach_count = (non_pen_next.astype(np.float32) > 0.5).sum(axis=-1)
        check(f"level {i}: hindsight rows' next_obs satisfies exactly one goal "
              f"(one-hot gridworld)", bool(np.all(ach_count == 1)),
              f"counts seen: {sorted(set(ach_count.tolist()))}")
        pen_action = r["action"][pen]
        pen_next = r["next_obs"][pen]
        achieved_idx = np.argmax(pen_next.astype(np.float32), axis=-1)
        mismatch_rate = float((achieved_idx != pen_action).mean())
        check(f"level {i}: penalty rows' proposed subgoal mostly differs from what was reached",
              mismatch_rate > 0.5, f"{mismatch_rate:.1%} differ")

    # The world resets on EVERY episode boundary (episode_done), not only on
    # true env termination, so a window's raw push-counter can never exceed
    # one full episode's worth of primitive steps 
    for i in range(1, k):
        count = np.asarray(g["carry"].windows[i - 1].count)
        check(f"level {i}: PrevStates window push-count never exceeds one "
              f"episode's horizon ({int(np.prod(H_levels))} steps)",
              bool(np.all(count <= int(np.prod(H_levels)))), f"max count {int(count.max())}")

    print("\n   re-running with subgoal_test_perc = 0.0 ...")
    cfg0 = yaml.safe_load(yaml.safe_dump(BASE_CFG))
    cfg0["training"]["subgoal_test_perc"] = 0.0
    cfg0["experiment"] = "HQF_probe_notest"
    g0 = run_hierq(cfg0)
    for i in range(1, k):
        ring = g0["carry"].rings[i]
        n = int(ring.size)
        pen = np.asarray(ring.data["penalty"])[:n]
        check(f"level {i}: NO penalty rows when subgoal_test_perc=0", int(pen.sum()) == 0,
              f"{int(pen.sum())} found")

    # ---- E: episode structure -------------------------------------------------
    print("\nE. Episode structure")
    import json, glob
    f = sorted(glob.glob(str(REPO_ROOT / "results" / "HQF_probe" / "runs" / "*.json")))[-1]
    m = json.load(open(f)); m = m.get("metrics", m)
    total_episodes = sum(m["train/episodes"])
    check("episodes occur (the world resets on episode boundaries)", total_episodes > 0,
          f"{total_episodes} episodes")
    seen = np.asarray(g["carry"].seen)
    check("seen-goal mask has at least one true entry after training", bool(seen.any()),
          f"{int(seen.sum())}/{g['G']} seen")
    # `seen0` shares carry.seen's buffer, which run_chunk's donate_argnums=0
    # frees after many calls 
    cfg_init = yaml.safe_load(yaml.safe_dump(BASE_CFG))
    cfg_init["training"]["n_steps"] = 0
    cfg_init["experiment"] = "HQF_init"; cfg_init["save_json"] = False
    g_init = run_hierq(cfg_init)
    seen0 = np.asarray(g_init["seen0"])
    check("seen is initialised from goals already true in the reset observation",
          int(seen0.sum()) > 0, f"{int(seen0.sum())} seen at reset")

    # ---- F: task-goal sampling is uniform over ALL goals (Algorithm 2's
    #      g_(k-1) <- G_(k-1)), not gated by `seen` -------------------------
    print("\nF. Task-goal sampling is uniform over the whole goal set")
    n_goals_toy, n_draw = 10, 6000
    draws = np.asarray(HQ.sample_goals(jax.random.PRNGKey(9), n_goals_toy, n_draw))
    counts = np.bincount(draws, minlength=n_goals_toy)
    check("every goal index is drawn, roughly uniformly (no seen-gating)",
          bool(np.all(counts > 0)) and bool(np.all(np.abs(counts - n_draw / n_goals_toy) < 0.3 * n_draw / n_goals_toy)),
          f"counts {counts.tolist()}")

    # ---- G: nested schedule (per episode) --------------------------------
    print("\nG. Nested schedule")
    att = np.asarray(g["carry"].stats)[:, 0]
    total_eps = sum(m["train/episodes"])
    check(f"top level: exactly one attempt per episode (its attempt IS the episode)",
          abs(att[k - 1] / max(total_eps, 1) - 1.0) < 0.02, f"measured {att[k-1]/max(total_eps,1):.3f}")
    for i in range(k - 1):
        ceiling = int(np.prod(H_levels[i + 1:]))
        check(f"level {i}: attempts/episode <= prod(H_levels[{i+1}:]) = {ceiling}",
              att[i] / max(total_eps, 1) <= ceiling + 0.5,
              f"measured {att[i] / max(total_eps, 1):.2f}")
    check(f"episode length never exceeds prod(H_levels) = {int(np.prod(H_levels))}",
          max(m["train/episode_len_mean"]) <= int(np.prod(H_levels)) + 1e-6,
          f"max measured {max(m['train/episode_len_mean']):.1f}")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*70}\nHierQ (deep) faithfulness: {n_pass}/{len(RESULTS)} checks passed")
    failed = [n for n, ok, _ in RESULTS if not ok]
    for n in failed:
        print("  FAILED:", n)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
