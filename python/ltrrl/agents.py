import math
import random

import litetorch as lt
from litetorch import nn, optim


def _rowsum(x):
    n = x.shape[0]
    d = x.shape[1]
    ones = lt.Tensor.ones([n, 1])
    return lt.Ops.matmul(ones.transpose(0, 1), x).reshape([d])


def _colsum(x):
    b = x.shape[0]
    k = x.shape[1]
    ones = lt.Tensor.ones([k, 1])
    return lt.Ops.matmul(x, ones).reshape([b])


def _mean(x):
    n = x.numel()
    flat = x.reshape([n, 1])
    ones = lt.Tensor.ones([n, 1])
    s = lt.Ops.matmul(ones.transpose(0, 1), flat).reshape([1])
    return lt.Ops.mul(s, lt.Tensor.from_vector([1.0 / n], [1]))


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for m in self.layers:
            x = m.forward(x)
        return x

    def parameters(self):
        ps = []
        for m in self.layers:
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in self.layers:
            m.to(device)


def _one_hot(actions, n_actions):
    b = len(actions)
    m = [0.0] * (b * n_actions)
    for i, a in enumerate(actions):
        m[i * n_actions + int(a)] = 1.0
    return lt.Tensor.from_vector(m, [b, n_actions])


def _to_batch(obs_list):
    d = len(obs_list[0])
    flat = [float(v) for o in obs_list for v in o]
    return lt.Tensor.from_vector(flat, [len(obs_list), d])


class DQN:
    def __init__(self, obs_dim, n_actions, hidden=128, lr=1e-3, gamma=0.99):
        self.n_actions = n_actions
        self.gamma = gamma
        self.q = MLP([obs_dim, hidden, hidden, n_actions])
        self.target = MLP([obs_dim, hidden, hidden, n_actions])
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.sync_target()

    def sync_target(self):
        for p, tp in zip(self.q.parameters(), self.target.parameters()):
            tp.copy_(p)

    def act(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with lt.no_grad():
            t = _to_batch([obs])
            qv = self.q.forward(t).to_vector()
        return max(range(self.n_actions), key=lambda a: qv[a])

    def update(self, batch):
        b = batch["obs"].shape[0]
        q = self.q.forward(batch["obs"])
        acts = [int(v) for v in batch["action"].to_vector()]
        q_sa = _colsum(lt.Ops.mul(q, _one_hot(acts, self.n_actions)))
        with lt.no_grad():
            nq = self.target.forward(batch["next_obs"]).to_vector()
            maxn = [max(nq[i * self.n_actions:(i + 1) * self.n_actions]) for i in range(b)]
            rv = batch["reward"].to_vector()
            dv = batch["done"].to_vector()
            yv = [rv[i] + self.gamma * maxn[i] * (1.0 - dv[i]) for i in range(b)]
        y = lt.Tensor.from_vector(yv, [b])
        loss = lt.Ops.mse_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.to_vector()[0]


class PPO:
    def __init__(self, obs_dim, n_actions, hidden=128, lr=3e-4, clip_eps=0.2):
        self.n_actions = n_actions
        self.clip_eps = clip_eps
        self.actor = MLP([obs_dim, hidden, n_actions])
        self.critic = MLP([obs_dim, hidden, 1])
        self.opt = optim.Adam(self.actor.parameters() + self.critic.parameters(), lr=lr)

    def act(self, obs):
        with lt.no_grad():
            t = _to_batch([obs])
            logits = self.actor.forward(t).to_vector()
        mx = max(logits)
        ex = [math.exp(v - mx) for v in logits]
        s = sum(ex)
        probs = [v / s for v in ex]
        r = random.random()
        acc = 0.0
        for a, p in enumerate(probs):
            acc += p
            if r < acc:
                return a, math.log(max(p, 1e-12))
        return len(probs) - 1, math.log(max(probs[-1], 1e-12))

    def update(self, obs_list, actions, old_logps, advantages, returns):
        b = len(obs_list)
        obs = _to_batch(obs_list)
        logits = self.actor.forward(obs)
        probs = lt.Ops.softmax(logits, 1)
        lp = lt.Ops.log(lt.Ops.clamp(probs, 1e-8, 1.0))
        lp_a = _colsum(lt.Ops.mul(lp, _one_hot(actions, self.n_actions)))
        old = lt.Tensor.from_vector([float(v) for v in old_logps], [b])
        ratio = lt.Ops.exp(lt.Ops.sub(lp_a, old))
        adv = lt.Tensor.from_vector([float(v) for v in advantages], [b])
        s1 = lt.Ops.mul(ratio, adv)
        s2 = lt.Ops.mul(lt.Ops.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps), adv)
        diff = lt.Ops.sub(s1, s2)
        emin = lt.Ops.mul(lt.Ops.sub(lt.Ops.add(s1, s2), lt.Ops.abs(diff)),
                          lt.Tensor.from_vector([0.5], [1]))
        actor_loss = lt.Ops.mul(_mean(emin), lt.Tensor.from_vector([-1.0], [1]))
        v = self.critic.forward(obs).reshape([b])
        ret = lt.Tensor.from_vector([float(x) for x in returns], [b])
        critic_loss = lt.Ops.mse_loss(v, ret)
        loss = lt.Ops.add(actor_loss.reshape([1]), critic_loss.reshape([1]))
        self.opt.zero_grad()
        lt.Ops.sum(loss).backward()
        self.opt.step()
        return loss.to_vector()[0]


class REINFORCE:
    def __init__(self, obs_dim, n_actions, hidden=128, lr=1e-3):
        self.n_actions = n_actions
        self.policy = MLP([obs_dim, hidden, n_actions])
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs):
        with lt.no_grad():
            t = _to_batch([obs])
            logits = self.policy.forward(t).to_vector()
        mx = max(logits)
        ex = [math.exp(v - mx) for v in logits]
        s = sum(ex)
        probs = [v / s for v in ex]
        r = random.random()
        acc = 0.0
        for a, p in enumerate(probs):
            acc += p
            if r < acc:
                return a
        return len(probs) - 1

    def update(self, obs_list, actions, returns, gamma=1.0):
        b = len(obs_list)
        logits = self.policy.forward(_to_batch(obs_list))
        lp = lt.Ops.log(lt.Ops.clamp(lt.Ops.softmax(logits, 1), 1e-8, 1.0))
        lp_a = _colsum(lt.Ops.mul(lp, _one_hot(actions, self.n_actions)))
        g = lt.Tensor.from_vector([float(v) for v in returns], [b])
        loss = lt.Ops.mul(_mean(lt.Ops.mul(lp_a, g)), lt.Tensor.from_vector([-1.0], [1]))
        self.opt.zero_grad()
        lt.Ops.sum(loss).backward()
        self.opt.step()
        return loss.to_vector()[0]


def compute_returns(rewards, gamma=0.99):
    g = 0.0
    out = [0.0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out
