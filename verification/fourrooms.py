"""Classic 13x13 FourRooms gridworld (104 free states), plus exact ground-truth
graph-Laplacian eigenvectors for that gridworld, used as the paper-fidelity
oracle for DCEO's Laplacian representation network.
"""
import numpy as np

N = 13


def _build_walls():
    walls = np.zeros((N, N), dtype=bool)
    walls[0, :] = True
    walls[N - 1, :] = True
    walls[:, 0] = True
    walls[:, N - 1] = True
    walls[1:N - 1, 6] = True   # vertical divider
    walls[6, 1:N - 1] = True   # horizontal divider
    # doorways (one per arm of the cross)
    for (r, c) in [(3, 6), (9, 6), (6, 3), (6, 9)]:
        walls[r, c] = False
    return walls


WALLS = _build_walls()
FREE_CELLS = [(r, c) for r in range(N) for c in range(N) if not WALLS[r, c]]
NUM_STATES = len(FREE_CELLS)
CELL_TO_IDX = {cell: i for i, cell in enumerate(FREE_CELLS)}

# 4 cardinal actions
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def _step_cell(cell, action):
    r, c = cell
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < N and 0 <= nc < N and not WALLS[nr, nc]:
        return (nr, nc)
    return cell  # bump into wall -> stay


def build_adjacency():
    """Undirected adjacency graph over free cells reachable by a single
    cardinal move (the graph used by the Laplacian-eigenoptions literature)."""
    A = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
    for cell in FREE_CELLS:
        i = CELL_TO_IDX[cell]
        for a in range(4):
            nxt = _step_cell(cell, a)
            if nxt != cell:
                j = CELL_TO_IDX[nxt]
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def true_eigenvectors(num_eigenvectors: int):
    """Smallest-nonzero-eigenvalue eigenvectors of the combinatorial graph
    Laplacian L = D - A. These are the ground-truth directions DCEO's
    self-supervised Laplacian loss is supposed to approximate."""
    A = build_adjacency()
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)  # ascending, symmetric-safe
    # eigvals[0] ~ 0 (trivial constant eigenvector) -> skip it
    idx = np.argsort(eigvals)
    order = idx[1:1 + num_eigenvectors]
    vecs = eigvecs[:, order]
    vals = eigvals[order]
    # normalize each to unit norm (already unit-norm from eigh, but be explicit)
    vecs = vecs / (np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-12)
    return vecs, vals  # (NUM_STATES, num_eigenvectors)


def one_hot_obs(idx):
    obs = np.zeros(NUM_STATES, dtype=np.float32)
    obs[idx] = 1.0
    return obs


class FourRoomsEnv:
    """Minimal single-instance stepping env with one-hot observations."""

    def __init__(self, horizon=200, rng=None):
        self.horizon = horizon
        self.rng = rng or np.random.default_rng(0)
        self.t = 0
        self.state_idx = None

    def reset(self):
        self.state_idx = self.rng.integers(0, NUM_STATES)
        self.t = 0
        return one_hot_obs(self.state_idx), self.state_idx

    def step(self, action):
        cell = FREE_CELLS[self.state_idx]
        next_cell = _step_cell(cell, int(action))
        self.state_idx = CELL_TO_IDX[next_cell]
        self.t += 1
        done = self.t >= self.horizon
        reward = 0.0
        if done:
            obs, idx = self.reset()
            return obs, idx, reward, True
        return one_hot_obs(self.state_idx), self.state_idx, reward, False

    def random_walk(self, n_steps):
        """Returns arrays of (state_idx_t, state_idx_t+1) for a uniform random
        walk -- the on-policy transition graph used for the 'attractive' term."""
        obs, idx = self.reset()
        idxs = np.empty(n_steps + 1, dtype=np.int64)
        idxs[0] = idx
        for t in range(n_steps):
            a = self.rng.integers(0, 4)
            _, idx, _, done = self.step(a)
            idxs[t + 1] = idx
            if done:
                idxs[t + 1] = self.state_idx  # already reset inside step()
        return idxs
