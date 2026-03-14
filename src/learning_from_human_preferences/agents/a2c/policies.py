import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Initialization (Baselines equivalent)
# =========================================================

def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.zeros_(module.bias)


# =========================================================
# Utilities
# =========================================================

def sample(logits):
    probs = torch.softmax(logits, dim=1)
    return torch.multinomial(probs, 1).squeeze(1)


def nhwc_to_nchw(x):
    return x.permute(0, 3, 1, 2)


# =========================================================
# Nature CNN
# =========================================================

class NatureCNN(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc = nn.Linear(3136, 512)

        self.apply(init_weights)

    def forward(self, x):

        x = nhwc_to_nchw(x)
        x = x / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))

        return x


# =========================================================
# CNN Policy
# =========================================================

class CnnPolicy(nn.Module):

    def __init__(self, ob_space, ac_space, nstack):

        super().__init__()

        nh, nw, nc = ob_space.shape
        input_channels = nc * nstack

        self.cnn = NatureCNN(input_channels)

        self.pi = nn.Linear(512, ac_space.n)
        self.vf = nn.Linear(512, 1)

        init_weights(self.pi)
        init_weights(self.vf)

        self.initial_state = []

    def forward(self, x):

        features = self.cnn(x)

        logits = self.pi(features)
        value = self.vf(features)

        return logits, value

    def step(self, obs):

        obs = torch.tensor(obs).float()

        with torch.no_grad():

            logits, value = self.forward(obs)

            action = sample(logits)

        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            []
        )

    def value(self, obs):

        obs = torch.tensor(obs).float()

        with torch.no_grad():
            _, value = self.forward(obs)

        return value.cpu().numpy()


# =========================================================
# MLP Policy (MovingDot)
# =========================================================

class MlpPolicy(nn.Module):

    def __init__(self, ob_space, ac_space, nstack):

        super().__init__()

        nh, nw, nc = ob_space.shape

        input_dim = nh * nw

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.pi = nn.Linear(512, ac_space.n)
        self.vf = nn.Linear(512, 1)

        self.apply(init_weights)

        self.initial_state = []

    def forward(self, x):

        x = x[:, :, :, -1] / 255.0

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        logits = self.pi(x)
        value = self.vf(x)

        return logits, value

    def step(self, obs):

        obs = torch.tensor(obs).float()

        with torch.no_grad():

            logits, value = self.forward(obs)

            action = sample(logits)

        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            []
        )


# =========================================================
# LSTM Policy
# =========================================================

class LstmPolicy(nn.Module):

    def __init__(self, ob_space, ac_space, nstack, nlstm=256):

        super().__init__()

        nh, nw, nc = ob_space.shape
        input_channels = nc * nstack

        self.cnn = NatureCNN(input_channels)

        self.lstm = nn.LSTMCell(512, nlstm)

        self.pi = nn.Linear(nlstm, ac_space.n)
        self.vf = nn.Linear(nlstm, 1)

        self.nlstm = nlstm

        self.initial_state = np.zeros((1, nlstm * 2), dtype=np.float32)

    def forward(self, x, state, mask):

        features = self.cnn(x)

        h, c = torch.chunk(state, 2, dim=1)

        mask = mask.unsqueeze(1)

        h = h * mask
        c = c * mask

        h, c = self.lstm(features, (h, c))

        logits = self.pi(h)
        value = self.vf(h)

        new_state = torch.cat([h, c], dim=1)

        return logits, value, new_state

    def step(self, obs, state, mask):

        obs = torch.tensor(obs).float()
        state = torch.tensor(state).float()
        mask = torch.tensor(mask).float()

        with torch.no_grad():

            logits, value, new_state = self.forward(obs, state, mask)

            action = sample(logits)

        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            new_state.cpu().numpy()
        )


# =========================================================
# LayerNorm LSTM Policy
# =========================================================

class LayerNormLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size

        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

        self.ln_i = nn.LayerNorm(4 * hidden_size)
        self.ln_h = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

        self.apply(init_weights)

    def forward(self, x, hidden):

        h, c = hidden

        gates = self.ln_i(self.ih(x)) + self.ln_h(self.hh(h))

        i, f, g, o = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new))

        return h_new, c_new


class LnLstmPolicy(nn.Module):

    def __init__(self, ob_space, ac_space, nstack, nlstm=256):

        super().__init__()

        nh, nw, nc = ob_space.shape
        input_channels = nc * nstack

        self.cnn = NatureCNN(input_channels)

        self.lnlstm = LayerNormLSTMCell(512, nlstm)

        self.pi = nn.Linear(nlstm, ac_space.n)
        self.vf = nn.Linear(nlstm, 1)

        self.initial_state = np.zeros((1, nlstm * 2), dtype=np.float32)

    def forward(self, x, state, mask):

        features = self.cnn(x)

        h, c = torch.chunk(state, 2, dim=1)

        mask = mask.unsqueeze(1)

        h = h * mask
        c = c * mask

        h, c = self.lnlstm(features, (h, c))

        logits = self.pi(h)
        value = self.vf(h)

        new_state = torch.cat([h, c], dim=1)

        return logits, value, new_state