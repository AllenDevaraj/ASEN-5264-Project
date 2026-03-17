#!/usr/bin/env python3
"""Learned world model for POMCP rollouts.

Predicts next belief state given current belief state and action.
Trained on transitions collected from Belief-Augmented PPO.

Architecture:
    Input (14D):  ee_pos[3], block_mu[3], block_sigma[3], gripper[1], action[4]
    Output (10D): delta_ee[3], delta_mu[3], delta_sigma[3], grasp_logit[1]

Usage:
    # Training
    model = WorldModel()
    model.train_on_buffer(transitions, epochs=100)
    model.save("world_model.pt")

    # Inference (for POMCP rollouts)
    model = WorldModel.load("world_model.pt")
    next_state, grasp_prob = model.predict(state, action)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class WorldModelNet(nn.Module):
    """MLP transition model: (state, action) -> (delta_state, grasp_logit)."""

    def __init__(self, state_dim=10, action_dim=4, hidden=128):
        super().__init__()
        input_dim = state_dim + action_dim  # 14D
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Separate heads for dynamics and grasp prediction
        self.dynamics_head = nn.Linear(hidden, 9)  # delta_ee[3], delta_mu[3], delta_sigma[3]
        self.grasp_head = nn.Linear(hidden, 1)     # grasp logit

    def forward(self, x):
        h = self.net(x)
        dynamics = self.dynamics_head(h)
        grasp_logit = self.grasp_head(h)
        return dynamics, grasp_logit


class WorldModel:
    """Wrapper for training and using the learned world model."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = WorldModelNet().to(self.device)
        self.trained = False

    def train_on_buffer(self, transitions, epochs=100, batch_size=256, lr=1e-3,
                        val_split=0.1, verbose=True):
        """Train world model on collected transitions.

        Args:
            transitions: dict with keys:
                'states': (N, 10) — [ee_pos[3], block_mu[3], block_sigma[3], gripper[1]]
                'actions': (N, 4) — [dx, dy, dz, gripper_cmd]
                'next_states': (N, 10) — same format as states
                'grasp_success': (N,) — binary (1 if grasp succeeded, 0 otherwise)
            epochs: Number of training epochs.
            batch_size: Minibatch size.
            lr: Learning rate.
            val_split: Fraction for validation.
            verbose: Print progress.
        """
        states = torch.FloatTensor(transitions['states']).to(self.device)
        actions = torch.FloatTensor(transitions['actions']).to(self.device)
        next_states = torch.FloatTensor(transitions['next_states']).to(self.device)
        grasp_labels = torch.FloatTensor(transitions['grasp_success']).to(self.device)

        # Compute deltas
        delta_states = next_states - states
        inputs = torch.cat([states, actions], dim=1)

        # Train/val split
        n = inputs.shape[0]
        n_val = max(1, int(n * val_split))
        perm = torch.randperm(n)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]

        train_dataset = TensorDataset(
            inputs[train_idx], delta_states[train_idx], grasp_labels[train_idx]
        )
        val_dataset = TensorDataset(
            inputs[val_idx], delta_states[val_idx], grasp_labels[val_idx]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            # Training
            self.net.train()
            train_losses = []
            for batch_in, batch_delta, batch_grasp in train_loader:
                dynamics_pred, grasp_pred = self.net(batch_in)
                loss_dyn = mse_loss(dynamics_pred, batch_delta[:, :9])
                loss_grasp = bce_loss(grasp_pred.squeeze(), batch_grasp)
                loss = loss_dyn + 0.1 * loss_grasp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.net.eval()
            val_losses = []
            with torch.no_grad():
                for batch_in, batch_delta, batch_grasp in val_loader:
                    dynamics_pred, grasp_pred = self.net(batch_in)
                    loss_dyn = mse_loss(dynamics_pred, batch_delta[:, :9])
                    loss_grasp = bce_loss(grasp_pred.squeeze(), batch_grasp)
                    val_losses.append((loss_dyn + 0.1 * loss_grasp).item())

            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train={np.mean(train_losses):.6f}, val={val_loss:.6f}")

        # Restore best
        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.trained = True

    def predict(self, state, action):
        """Predict next state and grasp probability.

        Args:
            state: (10,) array — [ee_pos[3], block_mu[3], block_sigma[3], gripper[1]]
            action: (4,) array — [dx, dy, dz, gripper_cmd]

        Returns:
            next_state: (10,) array
            grasp_prob: float in [0, 1]
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.concatenate([state, action])).unsqueeze(0).to(self.device)
            dynamics, grasp_logit = self.net(x)
            delta = dynamics[0].cpu().numpy()
            grasp_prob = torch.sigmoid(grasp_logit[0, 0]).cpu().item()

        next_state = state.copy()
        next_state[:9] += delta  # apply deltas to ee_pos, mu, sigma
        # Clamp sigma to be positive
        next_state[6:9] = np.maximum(next_state[6:9], 1e-4)
        return next_state, grasp_prob

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path, device=None):
        wm = cls(device=device)
        wm.net.load_state_dict(torch.load(path, map_location=wm.device, weights_only=True))
        wm.trained = True
        return wm
