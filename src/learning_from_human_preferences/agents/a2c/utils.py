import numpy as np
import torch
import torch.nn as nn
from collections import deque


# =========================================================
# Sampling (Gumbel trick)
# =========================================================

def sample(logits):
    noise = torch.rand_like(logits)
    return torch.argmax(logits - torch.log(-torch.log(noise)), dim=1)


# =========================================================
# Entropy
# =========================================================

def cat_entropy(logits):

    a0 = logits - torch.max(logits, dim=1, keepdim=True).values

    ea0 = torch.exp(a0)

    z0 = torch.sum(ea0, dim=1, keepdim=True)

    p0 = ea0 / z0

    return torch.sum(p0 * (torch.log(z0) - a0), dim=1)


def cat_entropy_softmax(p0):

    return -torch.sum(p0 * torch.log(p0 + 1e-6), dim=1)


# =========================================================
# MSE
# =========================================================

def mse(pred, target):

    return (pred - target) ** 2 / 2.0


# =========================================================
# Orthogonal initialization
# =========================================================

def ortho_init(scale=1.0):

    def _init(tensor):

        shape = tensor.shape

        if len(shape) == 2:
            flat_shape = shape

        elif len(shape) == 4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])

        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)

        u, _, v = np.linalg.svd(a, full_matrices=False)

        q = u if u.shape == flat_shape else v

        q = q.reshape(shape)

        q = scale * q[:shape[0], :shape[1]]

        tensor.data.copy_(torch.tensor(q, dtype=torch.float32))

    return _init


# =========================================================
# Conv layer
# =========================================================

def conv(in_channels, nf, rf, stride, pad=0, init_scale=1.0):

    layer = nn.Conv2d(in_channels, nf, rf, stride=stride, padding=pad)

    ortho_init(init_scale)(layer.weight)

    nn.init.zeros_(layer.bias)

    return layer


# =========================================================
# Fully connected
# =========================================================

def fc(in_features, nh, init_scale=1.0):

    layer = nn.Linear(in_features, nh)

    ortho_init(init_scale)(layer.weight)

    nn.init.zeros_(layer.bias)

    return layer


# =========================================================
# batch_to_seq
# =========================================================

def batch_to_seq(h, nbatch, nsteps, flat=False):

    if flat:
        h = h.view(nbatch, nsteps)
    else:
        h = h.view(nbatch, nsteps, -1)

    return [h[:, i] for i in range(nsteps)]


# =========================================================
# seq_to_batch
# =========================================================

def seq_to_batch(h, flat=False):

    if not flat:

        nh = h[0].shape[-1]

        return torch.cat(h, dim=0).view(-1, nh)

    else:

        return torch.stack(h).view(-1)


# =========================================================
# LSTM (identical equations)
# =========================================================

def lstm(xs, ms, s, wx, wh, b):

    c, h = torch.chunk(s, 2, dim=1)

    outputs = []

    for x, m in zip(xs, ms):

        c = c * (1 - m)

        h = h * (1 - m)

        z = x @ wx + h @ wh + b

        i, f, o, u = torch.chunk(z, 4, dim=1)

        i = torch.sigmoid(i)

        f = torch.sigmoid(f)

        o = torch.sigmoid(o)

        u = torch.tanh(u)

        c = f * c + i * u

        h = o * torch.tanh(c)

        outputs.append(h)

    s = torch.cat([c, h], dim=1)

    return outputs, s


# =========================================================
# LayerNorm
# =========================================================

def layer_norm(x, g, b, eps=1e-5):

    mean = x.mean(dim=1, keepdim=True)

    var = x.var(dim=1, keepdim=True)

    x = (x - mean) / torch.sqrt(var + eps)

    return x * g + b


# =========================================================
# LN-LSTM
# =========================================================

def lnlstm(xs, ms, s, wx, wh, gx, bx, gh, bh, b, gc, bc):

    c, h = torch.chunk(s, 2, dim=1)

    outputs = []

    for x, m in zip(xs, ms):

        c = c * (1 - m)

        h = h * (1 - m)

        z = (
            layer_norm(x @ wx, gx, bx)
            + layer_norm(h @ wh, gh, bh)
            + b
        )

        i, f, o, u = torch.chunk(z, 4, dim=1)

        i = torch.sigmoid(i)

        f = torch.sigmoid(f)

        o = torch.sigmoid(o)

        u = torch.tanh(u)

        c = f * c + i * u

        h = o * torch.tanh(layer_norm(c, gc, bc))

        outputs.append(h)

    s = torch.cat([c, h], dim=1)

    return outputs, s


# =========================================================
# conv_to_fc
# =========================================================

def conv_to_fc(x):

    return x.view(x.shape[0], -1)


# =========================================================
# Discount rewards
# =========================================================

def discount_with_dones(rewards, dones, gamma):

    discounted = []

    r = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):

        r = reward + gamma * r * (1.0 - done)

        discounted.append(r)

    return discounted[::-1]


# =========================================================
# Scheduler
# =========================================================

def constant(p):
    return 1


def linear(p):
    return 1 - p


schedules = {
    "linear": linear,
    "constant": constant,
}


class Scheduler:

    def __init__(self, v, nvalues, schedule):

        self.n = 0.0

        self.v = v

        self.nvalues = nvalues

        self.schedule = schedules[schedule]

    def value(self):

        current_value = self.v * self.schedule(self.n / self.nvalues)

        self.n += 1.0

        return current_value

    def value_steps(self, steps):

        return self.v * self.schedule(steps / self.nvalues)


# =========================================================
# Episode statistics
# =========================================================

class EpisodeStats:

    def __init__(self, nsteps, nenvs):

        self.episode_rewards = [[] for _ in range(nenvs)]

        self.lenbuffer = deque(maxlen=40)

        self.rewbuffer = deque(maxlen=40)

        self.nsteps = nsteps

        self.nenvs = nenvs

    def feed(self, rewards, masks):

        rewards = np.reshape(rewards, [self.nenvs, self.nsteps])

        masks = np.reshape(masks, [self.nenvs, self.nsteps])

        for i in range(self.nenvs):

            for j in range(self.nsteps):

                self.episode_rewards[i].append(rewards[i][j])

                if masks[i][j]:

                    l = len(self.episode_rewards[i])

                    s = sum(self.episode_rewards[i])

                    self.lenbuffer.append(l)

                    self.rewbuffer.append(s)

                    self.episode_rewards[i] = []

    def mean_length(self):

        return np.mean(self.lenbuffer) if self.lenbuffer else 0

    def mean_reward(self):

        return np.mean(self.rewbuffer) if self.rewbuffer else 0