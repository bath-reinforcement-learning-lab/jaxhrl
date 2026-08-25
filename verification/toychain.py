"""
The discrete stochastic decision process from Kulkarni et al. 2016
("Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction
and Intrinsic Motivation"), Section 4.1 / Figure 1: a chain of N states
s1..sN. The agent starts near the middle. Each of the two actions (left,
right) succeeds with probability p_succeed and otherwise moves the agent the
opposite way. Reaching the leftmost state gives a small reward the first time
it happens; reaching the rightmost state gives a large reward, but ONLY if
the leftmost state was visited earlier in the same episode. This makes the
optimal policy walk *away* from the big reward first -- a delayed,
order-dependent sparse-reward task that's known to be hard for flat
epsilon-greedy agents and is the paper's canonical example of where
intrinsic-motivation-driven subgoals help.
"""
import numpy as np

N_STATES = 6
START_STATE = 2
HORIZON = 10
# Asymmetric slip probability: moving right (toward the tempting-but-conditional
# big reward) always works; moving left (the deliberate, reward-free detour that
# has to happen first) only succeeds P_LEFT_SUCCEED of the time. This keeps the
# action choice informative (unlike a symmetric 50/50 slip, which cancels out
# and makes both actions equivalent) while preserving the paper's core
# difficulty: reaching the necessary first subgoal takes deliberate, noisy
# effort against the easy/immediate path toward the big reward.
P_LEFT_SUCCEED = 0.5
P_RIGHT_SUCCEED = 1.0
SMALL_REWARD = 0.01
BIG_REWARD = 1.0


def one_hot(idx, n=N_STATES):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


class ToyChainEnv:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.pos = START_STATE
        self.t = 0
        self.visited_left = False
        return one_hot(self.pos)

    def step(self, action):
        intended = -1 if action == 0 else 1
        p_succeed = P_LEFT_SUCCEED if action == 0 else P_RIGHT_SUCCEED
        if self.rng.random() >= p_succeed:
            intended = -intended  # slip
        self.pos = int(np.clip(self.pos + intended, 0, N_STATES - 1))
        self.t += 1

        reward = 0.0
        if self.pos == 0:
            if not self.visited_left:
                reward = SMALL_REWARD
            self.visited_left = True
        elif self.pos == N_STATES - 1:
            reward = BIG_REWARD if self.visited_left else 0.0

        done = self.t >= HORIZON
        return one_hot(self.pos), self.pos, reward, done
