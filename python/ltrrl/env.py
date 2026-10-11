import random


class Env:
    @property
    def observation_dim(self):
        raise NotImplementedError

    @property
    def n_actions(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class BanditEnv(Env):
    def __init__(self, probs):
        self.probs = list(probs)

    @property
    def observation_dim(self):
        return 1

    @property
    def n_actions(self):
        return len(self.probs)

    def reset(self):
        return [0.0]

    def step(self, action):
        r = 1.0 if random.random() < self.probs[action] else 0.0
        return [0.0], r, True, {}


class GridWorldEnv(Env):
    def __init__(self, size=4):
        self.size = size
        self.pos = 0

    @property
    def observation_dim(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return 4

    def _obs(self):
        o = [0.0] * (self.size * self.size)
        o[self.pos] = 1.0
        return o

    def reset(self):
        self.pos = 0
        return self._obs()

    def step(self, action):
        r, c = divmod(self.pos, self.size)
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        else:
            c = min(self.size - 1, c + 1)
        self.pos = r * self.size + c
        done = self.pos == self.size * self.size - 1
        return self._obs(), 1.0 if done else -0.01, done, {}
