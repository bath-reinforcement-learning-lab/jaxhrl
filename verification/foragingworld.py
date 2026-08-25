



"""
The "Foraging World" domain from Barreto et al. 2019, "The Option Keyboard:
Combining Skills in Reinforcement Learning" (NeurIPS), Section 5.1 / Appendix E.1.

Faithful to the paper's specifics we could pin down exactly:
  - m=2 nutrients, 3 food types with compositions y1=(1,0), y2=(0,1), y3=(1,1)
    (Figure 6/7/8 captions).
  - A cumulant e_i(h,a,s') that is 0 unless a food item is consumed, in which
    case it equals the resulting increase in nutrient i (Section 5.1).
  - The paper's own basis set is W0 = {(1,0), (0,1)} -- one option per
    nutrient (Appendix E.1).
  - Reward is assumed linear in the cumulants, r(s,a,s') ~= w^T c(s,a,s')
    (Section 6, "Related work" -- this is the standing assumption GPE/GPI's
    guarantees rely on), so we define task reward as w_task . (nutrient gain
    this step).

Simplified relative to the paper (documented, not hidden): grid is small and
food sits at 3 fixed cells rather than being pixel-rendered and randomly
respawning, and there's no nutrient decay/health mechanic -- this keeps the
check fast while preserving exactly the property under test (does GPI
combine pretrained per-nutrient skills into a *novel* combination weight
vector without further learning).
"""
import numpy as np

GRID = 7
FOOD_POS = {
    "y1": (1, 1),  # nutrient composition (1, 0)
    "y2": (1, 5),  # nutrient composition (0, 1)
    "y3": (5, 3),  # nutrient composition (1, 1)
}
FOOD_VEC = {
    "y1": np.array([1.0, 0.0], dtype=np.float32),
    "y2": np.array([0.0, 1.0], dtype=np.float32),
    "y3": np.array([1.0, 1.0], dtype=np.float32),
}
FOOD_CELLS = list(FOOD_POS.values())
FOOD_VECS = np.stack([FOOD_VEC[k] for k in FOOD_POS])  # (3, 2)
STATE_DIM = 4 * GRID * GRID  # one-hot(agent) ++ one-hot(y1) ++ one-hot(y2) ++ one-hot(y3)
NUM_ACTIONS = 4  # up, down, left, right
CUMULANT_DIM = 2

_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _one_hot_cell(pos):
    v = np.zeros(GRID * GRID, dtype=np.float32)
    v[pos[0] * GRID + pos[1]] = 1.0
    return v


def obs_from_state(agent_pos):
    return np.concatenate([_one_hot_cell(agent_pos)] + [_one_hot_cell(p) for p in FOOD_CELLS])


class ForagingWorldEnv:
    """Food sits at 3 fixed cells and is always available -- standing on a
    food cell 'eats' it every step (dense reward signal, fast to learn from).
    This turns the qualitative test into a clean one: for weight vector w,
    the optimal steady-state policy is to camp at argmax_type(w . y_type)."""

    def __init__(self, horizon=40, rng=None):
        self.horizon = horizon
        self.rng = rng or np.random.default_rng(0)

    def reset(self):
        self.pos = (int(self.rng.integers(0, GRID)), int(self.rng.integers(0, GRID)))
        self.t = 0
        return obs_from_state(self.pos)

    def step(self, action):
        dr, dc = _DELTAS[int(action)]
        nr = int(np.clip(self.pos[0] + dr, 0, GRID - 1))
        nc = int(np.clip(self.pos[1] + dc, 0, GRID - 1))
        self.pos = (nr, nc)
        self.t += 1

        cumulant = np.zeros(CUMULANT_DIM, dtype=np.float32)
        for key, cell in FOOD_POS.items():
            if self.pos == cell:
                cumulant = FOOD_VEC[key]
                break

        done = self.t >= self.horizon
        return obs_from_state(self.pos), cumulant, done


_FOOD_OH_CONCAT = np.concatenate([_one_hot_cell(p) for p in FOOD_CELLS])  # (3*GRID*GRID,), constant


def obs_from_state_batch(pos_batch):
    """Vectorized version of obs_from_state for a batch of agent positions,
    shape (num_envs, 2) -> (num_envs, STATE_DIM). Food is at fixed cells so
    its one-hot block is the same constant for every env."""
    num_envs = pos_batch.shape[0]
    agent_idx = pos_batch[:, 0] * GRID + pos_batch[:, 1]
    agent_oh = np.zeros((num_envs, GRID * GRID), dtype=np.float32)
    agent_oh[np.arange(num_envs), agent_idx] = 1.0
    food_oh = np.tile(_FOOD_OH_CONCAT, (num_envs, 1))
    return np.concatenate([agent_oh, food_oh], axis=1)


def reset_batch(rng, num_envs):
    pos = np.stack([rng.integers(0, GRID, size=num_envs), rng.integers(0, GRID, size=num_envs)], axis=1)
    t = np.zeros(num_envs, dtype=np.int32)
    return obs_from_state_batch(pos), pos, t


def step_batch(pos_batch, actions, t_batch, horizon):
    """Vectorized env step for `num_envs` independent agents sharing the
    same fixed food layout. Does NOT auto-reset finished envs -- the caller
    is expected to do that per-env (mirrors the repo's own `where_per_env`
    reset pattern used throughout its __main__ training loops)."""
    deltas = np.array(_DELTAS)
    d = deltas[actions]
    new_pos = np.clip(pos_batch + d, 0, GRID - 1)
    new_t = t_batch + 1

    cumulant = np.zeros((pos_batch.shape[0], CUMULANT_DIM), dtype=np.float32)
    for key, cell in FOOD_POS.items():
        mask = (new_pos[:, 0] == cell[0]) & (new_pos[:, 1] == cell[1])
        cumulant[mask] = FOOD_VEC[key]

    done = new_t >= horizon
    obs = obs_from_state_batch(new_pos)
    return obs, new_pos, cumulant, done, new_t


def optimal_reward_per_step(w):
    """Ground truth: best achievable per-step reward under weight vector w is
    just camping at whichever food cell maximizes w . y_type."""
    return float(np.max(FOOD_VECS @ np.asarray(w)))


def food_type_for_weight(w):
    idx = int(np.argmax(FOOD_VECS @ np.asarray(w)))
    return list(FOOD_POS.keys())[idx], FOOD_CELLS[idx]
