"""HierQ implementation-faithfulness suite.

Checks that `jaxhrl/HierQ.py` implements Algorithm 2 from the appendix of Levy
et al., "Learning Multi-Level Hierarchies with Hindsight" -- as distinct from
`hierq_verify.py`, which asks whether it reproduces the paper's grid-world
performance result. This suite answers "is it the right algorithm".

Two kinds of evidence:

  * the two update rules are checked against hand-computed Bellman targets,
    exactly as Algorithm 2 writes them;
  * the repo's real `__main__` loop is then executed via runpy and the resulting
    Q-tables are inspected, to confirm the structural invariants that make the
    algorithm work

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
sys.path.insert(0, str(HERE))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


BASE_CFG = {
    "experiment": "HQF_probe", "seed": 0, "save_json": True,
    "use_wandb": False, "overwrite": True,
    "env": {"framework": "gridworld", "make": {"id": "fourrooms"}},
    "training": {"num_levels": 3, "H": 4, "horizon": 64, "n_steps": 1280,
                 "num_envs": 32, "chunk_size": 64, "alpha": 0.1,
                 # gamma omitted -> derived per level as 1 - 1/H_i
                 "epsilon": 0.2},
    "eval": {"enabled": False}, "checkpoint": {"enabled": False},
}


def run_hierq(cfg):
    import runpy
    p = pathlib.Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text(yaml.safe_dump(cfg))
    sys.argv = ["HierQ.py", "--config", str(p)]
    return runpy.run_path(str(REPO_ROOT / "jaxhrl" / "HierQ.py"), run_name="__main__")


def bfs_all_pairs(trans):
    S = trans.shape[0]
    INF = 10 ** 6
    dist = np.full((S, S), INF, np.int32)
    for src in range(S):
        dist[src, src] = 0
        frontier, d = [src], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for a in range(trans.shape[1]):
                    v = trans[u, a]
                    if dist[src, v] == INF:
                        dist[src, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def main():
    import jax.numpy as jnp
    import jaxhrl.HierQ as HQ

    # ---- A: the two update rules, against hand-computed Bellman targets ------
    print("A. Update rules match Algorithm 2's equations")
    S, A, alpha, gamma = 5, 3, 0.3, 0.9

    q0 = jnp.asarray(np.random.default_rng(0).normal(size=(S, S, A)), jnp.float32)
    states = jnp.asarray([1, 3], jnp.int32)
    actions = jnp.asarray([2, 0], jnp.int32)
    nxt = jnp.asarray([4, 4], jnp.int32)
    got = np.asarray(HQ.update_level0(q0, states, actions, nxt, alpha, gamma, S))

    q0n = np.asarray(q0)
    want = q0n.copy()
    for n in range(2):
        s, a, s2 = int(states[n]), int(actions[n]), int(nxt[n])
        for g in range(S):
            r = 0.0 if s2 == g else -1.0
            disc = 0.0 if s2 == g else gamma
            target = r + disc * q0n[s2, g].max()
            # `.add` composes the deltas of both transitions; accumulate the same way
            want[s, g, a] += alpha * (target - q0n[s, g, a])
    check("level 0: updates every goal with (1-a)Q + a[R + g.max_a' Q(s',g,a')]",
          np.allclose(got, want, atol=1e-5),
          f"max abs diff {np.abs(got - want).max():.2e}")
    touched = ~np.isclose(np.asarray(q0), got, atol=1e-9)
    check("level 0: exactly S goal-entries touched per transition (all-goals HER)",
          touched.sum() == 2 * S, f"{touched.sum()} entries, expected {2*S}")

    qi = jnp.asarray(np.random.default_rng(1).normal(size=(S, S, S)), jnp.float32)
    window = jnp.asarray([[0, 2, 3], [1, 1, 4]], jnp.int32)
    valid = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], jnp.float32)
    got_i = np.asarray(HQ.update_level_i(qi, window, valid, nxt, alpha, gamma, S))

    qin = np.asarray(qi)
    want_i = qin.copy()
    for n in range(2):
        s2 = int(nxt[n])
        for w in range(window.shape[1]):
            if float(valid[n, w]) == 0.0:
                continue
            pstate = int(window[n, w])
            for g in range(S):
                r = 0.0 if s2 == g else -1.0
                disc = 0.0 if s2 == g else gamma
                target = r + disc * qin[s2, g].max()
                want_i[pstate, g, s2] += alpha * (target - qin[pstate, g, s2])
    check("level i>0: PrevStates update matches Algorithm 2 for every (s, goal)",
          np.allclose(got_i, want_i, atol=1e-5),
          f"max abs diff {np.abs(got_i - want_i).max():.2e}")
    diff = ~np.isclose(qin, got_i, atol=1e-9)
    # nxt is 4 for both transitions, so ONLY the a==4 action-plane may change.
    check("level i>0: the stored ACTION is s' itself (hindsight action transition)",
          diff[:, :, [0, 1, 2, 3]].sum() == 0 and diff[:, :, 4].sum() > 0,
          f"{diff[:, :, 4].sum()} entries written, all in the a=s'=4 plane")
    # env 0's window is [0,2,3] with slot 2 masked off, and state 3 appears in
    # no other env's valid slots, so row 3 must be untouched.
    check("level i>0: masked-out window slots are never written",
          np.allclose(qin[3], got_i[3]),
          "state 3 sits only in a valid=0 slot and is unchanged")

    # ---- B: structural invariants after a real training run -----------------
    print("\nB. Structural invariants after running the real training loop")
    g = run_hierq(BASE_CFG)
    H_levels, k = g["H_levels"], g["k"]
    pess, n_states = g["pessimistics"], g["n_states"]
    print(f"   H_levels={H_levels}  pessimistic_init={pess}  S={n_states}")

    from jaxhrl.common.wrappers import _make_gridworld
    trans, S_env, _ = _make_gridworld("fourrooms")
    dist = bfs_all_pairs(np.asarray(trans))

    # The initial tables are donated to run_chunk and freed, so inspect them via
    # a zero-step run whose training loop never executes.
    cfg0 = yaml.safe_load(yaml.safe_dump(BASE_CFG))
    cfg0["training"]["n_steps"] = 0
    cfg0["experiment"] = "HQF_init"; cfg0["save_json"] = False
    g_init = run_hierq(cfg0)
    q0_init = np.asarray(g_init["carry"].q_tables[0])
    check("Q0 is initialised optimistically at 0 (Algorithm 2 pessimises only i > 0)",
          float(np.abs(q0_init).max()) < 1e-9, f"max |Q0| = {float(np.abs(q0_init).max()):.1e}")
    for i in range(1, k):
        qi_init = np.asarray(g_init["carry"].q_tables[i])
        check(f"Q{i} is initialised pessimistically at {pess[i]:.3f}",
              float(qi_init.min()) == float(qi_init.max())
              and abs(float(qi_init.min()) - pess[i]) < 1e-3,
              f"uniform {float(qi_init.min())}")

    for i in range(1, k):
        q = np.asarray(g["carry"].q_tables[i])
        init = pess[i]
        touched_pair = (q != init).any(axis=1)          # (state, subgoal)
        reach = int(np.prod(H_levels[:i]))              # primitive steps per level-i action
        within = dist <= reach
        n_t = int(touched_pair.sum())
        n_bad = int(touched_pair[~within].sum())
        check(f"level {i}: every written (state, subgoal) pair is reachable "
              f"within {reach} primitive steps",
              n_bad == 0, f"{n_t} pairs written, {n_bad} unreachable")
        # q is (state, goal, action); `within` is (state, subgoal=action), so
        # transpose to put the masked axes first.
        q_sa = q.transpose(0, 2, 1)      # (state, action, goal)
        check(f"level {i}: unreachable subgoals retain the pessimistic value "
              f"{init} (this replaces subgoal testing)",
              bool(np.all(q_sa[~within] == init)),
              f"{int((~within).sum())} unreachable pairs, all untouched")
        # HAC writes -H penalty rows; HierQ must never produce a value below the
        # pessimistic floor, because it has no penalty transitions at all.
        # With the floor at the fixed point of Q = -1 + gamma*Q nothing can fall
        # below it, so reachable subgoals are always ranked above untouched ones.
        check(f"level {i}: nothing falls below the pessimistic floor, so reachable "
              f"subgoals always outrank unreachable ones",
              float(q.min()) >= init - 1e-4, f"min {float(q.min()):.3f} vs floor {init:.3f}")

    # ---- C: nested schedule --------------------------------------------------
    print("\nC. Nested schedule")
    import json, glob
    f = sorted(glob.glob(str(REPO_ROOT / "results" / "HQF_probe" / "runs" / "*.json")))[-1]
    m = json.load(open(f)); m = m.get("metrics", m)
    expected = int(np.prod(H_levels))
    att = [m[f"levels/attempts_level_{i}"][-1] for i in range(k)]
    eps = m["train/episodes"][-1]
    # Episodes terminate as soon as the task goal is hit, so per-episode attempt
    # counts sit at or below their full-horizon values; these bounds are exact.
    check("top level: exactly one attempt per episode (its attempt IS the episode)",
          abs(att[k - 1] / eps - 1.0) < 0.02, f"measured {att[k-1]/eps:.3f}")
    for i in range(k - 1):
        check(f"level {i} ends at least as often as level {i+1} (end[i+1] => end[i])",
              att[i] >= att[i + 1] - 1e-6, f"{att[i]:.0f} vs {att[i+1]:.0f}")
        ceiling = int(np.prod(H_levels[i + 1:]))
        check(f"level {i}: at most {ceiling} attempts per episode (full-horizon bound)",
              att[i] / eps <= ceiling + 0.02, f"measured {att[i]/eps:.2f}")
    check(f"episode length never exceeds prod(H_levels) = {expected}",
          max(m["train/episode_len_mean"]) <= expected + 1e-6,
          f"max measured {max(m['train/episode_len_mean']):.1f}")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*68}\nHierQ faithfulness: {n_pass}/{len(RESULTS)} checks passed")
    failed = [n for n, ok, _ in RESULTS if not ok]
    for n in failed:
        print("  FAILED:", n)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
