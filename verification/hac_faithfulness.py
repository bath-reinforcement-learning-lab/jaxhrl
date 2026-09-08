"""HAC implementation-faithfulness suite.

Checks that `jaxhrl/HAC.py` implements the algorithm Levy et al. specify in
"Learning Multi-Level Hierarchies with Hindsight" (ICLR 2019) -- as distinct
from `hac_verify.py`, which asks whether it reproduces the paper's performance
results. This suite answers "is it the right algorithm", not "is it fast".

It asserts the paper's defining properties on the transitions the implementation
genuinely wrote:

  A  sparse reward / terminal-discount semantics
  B  hindsight ACTION transitions   (paper Sec. 3.1)
  C  hindsight GOAL transitions     (paper Sec. 3.2, HER)
  D  subgoal TESTING transitions    (paper Sec. 3.3)
  E  bounded critic + matched discount  (Levy's critic: Q in [-H,0], g=1-1/H)
  F  nested schedule                (level i gets H_i actions; H**k horizon)
  G  level 0 stores primitive actions, higher levels store subgoal states
  H  sub-policies act deterministically while a subgoal is under test

Run: python verification/hac_faithfulness.py
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

BASE_CFG = {
    "experiment": "HACF_probe", "seed": 0, "save_json": False,
    "use_wandb": False, "overwrite": True,
    "env": {"framework": "pointmaze", "make": {"id": "fourrooms"}},
    "goal": {"indices": [0, 1], "low": [-1.0, -1.0], "high": [1.0, 1.0],
             "threshold": [0.15, 0.15]},   # loose, so achievements actually occur
    "training": {
        "num_levels": 3, "H": 4, "n_steps": 400, "num_envs": 32,
        "chunk_size": 200, "batch_size": 32, "buffer_size": 40000,
        "subgoal_test_perc": 0.5, "random_action_perc": 0.2, "noise_perc": 0.1,
        "tau": 0.05, "lr_actor": 0.001, "lr_critic": 0.001,
    },
    "network": {"hidden_dim": 32},
    "eval": {"enabled": False}, "checkpoint": {"enabled": False},
}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def run_hac(cfg):
    """Execute the repo's real training loop; return its module globals."""
    import runpy
    import jaxhrl.common.wrappers as W
    import pointmaze
    W.make_jax_env = lambda *a, **k: pointmaze.make_wrapped_env(W.JaxWrappedEnv)
    p = pathlib.Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text(yaml.safe_dump(cfg))
    sys.argv = ["HAC.py", "--config", str(p)]
    return runpy.run_path(str(REPO_ROOT / "jaxhrl" / "HAC.py"), run_name="__main__")


def rows_of(g, level):
    """The transitions the implementation actually wrote for one level."""
    ring = g["carry"].rings[level]
    n = int(ring.size)
    return {k: np.asarray(v)[:n] for k, v in ring.data.items()}, n


def main():
    print("Running the repo's real HAC training loop (subgoal testing ON)...")
    g = run_hac(BASE_CFG)
    H_levels = g["H_levels"]
    k = len(H_levels)
    goal_idx = np.asarray(g["goal_indices"])
    glow, ghigh = np.asarray(g["goal_low"]), np.asarray(g["goal_high"])
    print(f"H_levels={H_levels}  gammas={[round(x,4) for x in g['gammas']]}  "
          f"q_limits={g['q_limits']}\n")

    def proj(obs):
        return np.clip(obs[..., goal_idx], glow, ghigh)

    # ---- A: reward / discount semantics -------------------------------------
    print("A. Sparse reward and terminal-discount semantics")
    for i in range(k):
        r, n = rows_of(g, i)
        if n == 0:
            check(f"level {i}: buffer non-empty", False, "no transitions written")
            continue
        rew, disc = r["reward"], r["discount"]
        allowed = {0.0, -1.0, -float(H_levels[i])}
        vals = set(np.unique(np.round(rew, 6)).tolist())
        check(f"level {i}: rewards drawn only from {{0, -1, -H}}",
              vals <= allowed, f"observed {sorted(vals)}")
        check(f"level {i}: reward 0 => discount 0 (achieved is terminal)",
              np.all(disc[rew == 0.0] == 0.0))
        check(f"level {i}: reward -1 => discount 1 (timeout bootstraps)",
              np.all(disc[rew == -1.0] == 1.0))

    # ---- B/G: hindsight action transitions ----------------------------------
    print("\nB/G. Hindsight ACTION transitions (Sec. 3.1)")
    for i in range(k):
        r, n = rows_of(g, i)
        if n == 0:
            continue
        achieved = proj(r["next_obs"])
        matches = np.all(np.isclose(r["action"], achieved, atol=1e-5), axis=-1)
        pen = r["reward"] == -float(H_levels[i])
        if i == 0:
            check("level 0: stores PRIMITIVE actions, not state projections",
                  matches.mean() < 0.5, f"{matches.mean():.1%} coincide")
        else:
            non_pen = ~pen
            frac = matches[non_pen].mean() if non_pen.any() else 0.0
            check(f"level {i}: non-penalty action == achieved subgoal state",
                  frac > 0.99, f"{frac:.1%} of {int(non_pen.sum())} rows")
            if pen.any():
                check(f"level {i}: penalty rows keep the PROPOSED subgoal",
                      matches[pen].mean() < 0.5, f"{matches[pen].mean():.1%} coincide")

    # ---- C: hindsight goal transitions --------------------------------------
    print("\nC. Hindsight GOAL transitions (Sec. 3.2)")
    for i in range(k):
        r, n = rows_of(g, i)
        if n == 0:
            continue
        achieved = proj(r["next_obs"])
        relabelled = np.all(np.isclose(r["goal"], achieved, atol=1e-5), axis=-1)
        terminal = relabelled & (r["reward"] == 0.0) & (r["discount"] == 0.0)
        check(f"level {i}: goal-relabelled terminal transitions present",
              terminal.sum() > 0, f"{terminal.sum()} of {n} rows")

    # ---- D: subgoal testing transitions -------------------------------------
    print("\nD. Subgoal TESTING transitions (Sec. 3.3)")
    for i in range(k):
        r, n = rows_of(g, i)
        if n == 0:
            continue
        pen = (r["reward"] == -float(H_levels[i])) & (r["discount"] == 0.0)
        if i == 0:
            check("level 0: no subgoal-test penalties (it proposes no subgoals)",
                  pen.sum() == 0, f"{pen.sum()} found")
        else:
            check(f"level {i}: penalty transitions present with testing on",
                  pen.sum() > 0, f"{pen.sum()} of {n} rows")

    print("\n   re-running with subgoal_test_perc = 0.0 ...")
    cfg0 = yaml.safe_load(yaml.safe_dump(BASE_CFG))
    cfg0["training"]["subgoal_test_perc"] = 0.0
    cfg0["experiment"] = "HACF_probe_notest"
    g0 = run_hac(cfg0)
    for i in range(1, k):
        r, n = rows_of(g0, i)
        if n == 0:
            continue
        pen = (r["reward"] == -float(H_levels[i])) & (r["discount"] == 0.0)
        check(f"level {i}: NO penalties when testing is disabled",
              pen.sum() == 0, f"{pen.sum()} found")

    # ---- E: bounded critic and matched discount -----------------------------
    print("\nE. Bounded critic and matched discount")
    import jax.numpy as jnp
    from flax import nnx
    for i in range(k):
        check(f"level {i}: q_limit == H ({H_levels[i]})",
              abs(g["q_limits"][i] - H_levels[i]) < 1e-9)
        check(f"level {i}: gamma == 1 - 1/H ({1 - 1/H_levels[i]:.4f})",
              abs(g["gammas"][i] - (1 - 1 / H_levels[i])) < 1e-9)
        _, critic, _, _, _, _ = nnx.merge(g["graphdefs"][i], g["carry"].nnx_states[i])
        obs_dim, goal_dim = g["obs_dim"], g["goal_dim"]
        rng = np.random.default_rng(0)
        q = np.asarray(critic(
            jnp.asarray(rng.normal(size=(256, obs_dim)), jnp.float32),
            jnp.asarray(rng.uniform(-1, 1, size=(256, goal_dim)), jnp.float32),
            jnp.asarray(rng.uniform(-1, 1, size=(256, g["level_act_dim"][i])), jnp.float32)))
        check(f"level {i}: Q bounded to [-H, 0]",
              q.min() >= -H_levels[i] - 1e-4 and q.max() <= 1e-6,
              f"observed [{q.min():.3f}, {q.max():.3f}]")

    # ---- F: nested schedule --------------------------------------------------
    print("\nF. Nested schedule")
    expected = int(np.prod(H_levels))
    cfgF = yaml.safe_load(yaml.safe_dump(BASE_CFG))
    cfgF["goal"]["threshold"] = [1e-9, 1e-9]      # unreachable -> pure horizon
    cfgF["training"].update(n_steps=expected * 3, chunk_size=expected)
    cfgF["experiment"] = "HACF_probe_sched"
    cfgF["save_json"] = True
    gF = run_hac(cfgF)
    import json, glob
    f = sorted(glob.glob(str(REPO_ROOT / "results" / "HACF_probe_sched" / "runs" / "*.json")))[-1]
    m = json.load(open(f)); m = m.get("metrics", m)
    ep_len = m["train/episode_len_mean"][-1]
    check(f"episode length == prod(H_levels) == {expected}",
          abs(ep_len - expected) < 1e-6, f"measured {ep_len}")
    for i in range(k):
        # `attempts_level_i` counts how many times level i's ATTEMPT ended, i.e.
        # how often its parent handed it a fresh goal -- prod(H_levels[i+1:]),
        # with an empty product of 1 for the top level. Each such attempt spans
        # H_levels[i] of that level's own actions, so actions per episode is
        # prod(H_levels[i:]); both are asserted below.
        want_attempts = int(np.prod(H_levels[i + 1:])) if i + 1 < k else 1
        got = m[f"levels/attempts_level_{i}"][-1] / m["train/episodes"][-1]
        check(f"level {i}: {want_attempts} attempts per episode",
              abs(got - want_attempts) / want_attempts < 0.02, f"measured {got:.1f}")
        want_actions = int(np.prod(H_levels[i:]))
        check(f"level {i}: {want_actions} actions per episode "
              f"({want_attempts} attempts x H={H_levels[i]})",
              abs(got * H_levels[i] - want_actions) / want_actions < 0.02,
              f"measured {got * H_levels[i]:.1f}")

    # ---- H: determinism under subgoal test -----------------------------------
    print("\nH. Sub-policies act deterministically while under test")
    import jaxhrl.HAC as HACmod   # module-scope import, no __main__
    import jax
    actor, _, _, _, _, _ = nnx.merge(g["graphdefs"][0], g["carry"].nnx_states[0])
    obs = jnp.asarray(np.random.default_rng(1).normal(size=(64, g["obs_dim"])), jnp.float32)
    goal = jnp.asarray(np.random.default_rng(2).uniform(-1, 1, size=(64, g["goal_dim"])), jnp.float32)
    low, high = jnp.asarray(g["action_low"]), jnp.asarray(g["action_high"])
    det = HACmod.select_action(actor, obs, goal, low, high,
                               jnp.ones((64,), bool), jax.random.PRNGKey(0), 0.1, 0.2)
    raw = jnp.clip(actor(obs, goal), low, high)
    check("deterministic=True reproduces the greedy action exactly",
          bool(jnp.allclose(det, raw, atol=1e-6)))
    stoch = HACmod.select_action(actor, obs, goal, low, high,
                                 jnp.zeros((64,), bool), jax.random.PRNGKey(0), 0.1, 0.2)
    check("deterministic=False injects exploration noise",
          not bool(jnp.allclose(stoch, raw, atol=1e-6)))

    # ---- summary -------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*66}\nHAC faithfulness: {n_pass}/{len(RESULTS)} checks passed")
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print("  -", n)
    else:
        print("Every property the paper specifies holds in the transitions the")
        print("implementation actually emits.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
