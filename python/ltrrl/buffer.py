import random
from collections import deque

import litetorch as lt


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((list(obs), int(action), float(reward), list(next_obs), bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        d = len(batch[0][0])
        obs = lt.Tensor.from_vector([v for t in batch for v in t[0]], [batch_size, d])
        act = lt.Tensor.from_vector([float(t[1]) for t in batch], [batch_size])
        rew = lt.Tensor.from_vector([t[2] for t in batch], [batch_size])
        nxt = lt.Tensor.from_vector([v for t in batch for v in t[3]], [batch_size, d])
        done = lt.Tensor.from_vector([1.0 if t[4] else 0.0 for t in batch], [batch_size])
        return {"obs": obs, "action": act, "reward": rew, "next_obs": nxt, "done": done}

    def __len__(self):
        return len(self.buf)
