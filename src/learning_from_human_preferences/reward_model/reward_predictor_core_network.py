"""
Reward Predictor Core Networks
==============================

Faithful PyTorch translation of the TensorFlow implementation used in:

    Deep Reinforcement Learning from Human Preferences
    Christiano et al., 2017

The logic is preserved exactly:

    frame -> neural network -> r_t
    segment reward = sum(r_t)

Segment rewards are compared using a Bradley–Terry model.

This file replaces TensorFlow 1 code with PyTorch while keeping
the exact architecture and preprocessing steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MovingDot feature extraction
# ============================================================

def extract_moving_dot_features(observation: torch.Tensor):
    """
    Extract features exactly as done in the TensorFlow code.

    Features used:
        action
        x_position
        y_position

    Only the LAST frame of the stack is used.

    Features are normalized:
        action / 4
        x / 83
        y / 83
    """

    # observation shape: (batch, H, W, C)

    last_frame = observation[..., -1] # is equivalent to write: observation[:, :, :, -1],
    # selects last channel for each element in the batch

    action = observation[:, 0, 0, -1]

    x_projection = last_frame.sum(dim=1) # sums along the H dimension
    y_projection = last_frame.sum(dim=2) # sums along the W dimension

    x_pos = x_projection.argmax(dim=1).float()
    y_pos = y_projection.argmax(dim=1).float()

    action = action.float()

    # normalization exactly as original implementation
    action = action / 4.0
    x_pos = x_pos / 83.0
    y_pos = y_pos / 83.0

    features = torch.stack([action, x_pos, y_pos], dim=1)

    return features


# ============================================================
# MovingDot reward network
# ============================================================

class MovingDotRewardNetwork(nn.Module):
    """
    Original architecture:

        3 → 64 → 64 → 64 → 1
        ReLU activations
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, features):

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        reward = self.fc4(x)

        return reward.squeeze(-1) # squeeze(dim) removes the specified dimension if it is 1


# ============================================================
# Atari reward CNN
# ============================================================

class AtariRewardNetwork(nn.Module):
    """
    Exact CNN architecture from the original implementation.

    Input:  (N, 84, 84, 4)  — NHWC, uint8
    Output: (N,)             — scalar reward per frame

    Conv layers (NCHW after permute):
        Conv(7x7, stride=3, 16)  → 16 × 26 × 26
        Conv(5x5, stride=2, 16)  → 16 × 11 × 11
        Conv(3x3, stride=1, 16)  → 16 ×  9 ×  9
        Conv(3x3, stride=1, 16)  → 16 ×  7 ×  7  → flatten → 784

    Followed by:
        FC(784 → 64)
        FC(64  →  1)

    Activations:   LeakyReLU(0.01)
    Regularization: BatchNorm2d + Dropout after each conv (except last)
    """

    # Flattened size for standard 84×84×4 Atari input.
    _FLAT = 16 * 7 * 7  # = 784

    def __init__(self, input_channels=4, dropout_prob=0.5):

        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(self._FLAT, 64)
        self.fc2 = nn.Linear(64, 1)

        self.activation = nn.LeakyReLU(0.01)

    def forward(self, frames):

        # frames: (N, 84, 84, 4) — NHWC uint8
        # Conv2d requires NCHW → permute before processing.
        x = frames.float() / 255.0
        x = x.permute(0, 3, 1, 2)   # (N, 4, 84, 84)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)   # (N, 784)

        x = self.activation(self.fc1(x))

        reward = self.fc2(x)

        return reward.squeeze(-1)


# ============================================================
# Segment reward
# ============================================================

def compute_segment_reward(frame_rewards: torch.Tensor):
    """
    Segment reward defined as:

        R(segment) = sum_t r_t
    """

    return frame_rewards.sum(dim=1)


# ============================================================
# Bradley–Terry preference model
# ============================================================

def preference_probability(reward_a, reward_b):
    """
    Compute preference probability:

        P(A > B) = exp(R_A) / (exp(R_A) + exp(R_B))
    """

    exp_a = torch.exp(reward_a)
    exp_b = torch.exp(reward_b)

    return exp_a / (exp_a + exp_b)


# ============================================================
# Preference loss
# ============================================================

def preference_loss(reward_a, reward_b, labels):
    """
    Binary cross-entropy loss on preference predictions.
    """

    prob = preference_probability(reward_a, reward_b)

    return F.binary_cross_entropy(prob, labels.float())