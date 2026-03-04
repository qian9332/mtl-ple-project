"""
Expert Networks for Multi-Task Learning.
Implements shared and task-specific expert modules used in MMoE/CGC/PLE architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    """Single Expert Network with configurable architecture."""

    def __init__(self, input_dim: int, expert_dim: int, dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        layers = [nn.Linear(input_dim, expert_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(expert_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(expert_dim, expert_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(expert_dim))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateNetwork(nn.Module):
    """
    Gate Network with Temperature Annealing.
    Controls how task towers select and combine expert outputs.
    Temperature annealing suppresses 'winner-take-all' expert collapse.
    """

    def __init__(self, input_dim: int, num_experts: int,
                 initial_temperature: float = 1.0,
                 min_temperature: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts, bias=False)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.initial_temperature = initial_temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            gate_weights: (batch_size, num_experts), softmax-normalized weights
        """
        logits = self.fc(x) / max(self.temperature, self.min_temperature)
        return F.softmax(logits, dim=-1)

    def anneal_temperature(self, decay_rate: float = 0.995):
        """Apply temperature decay (called per epoch)."""
        self.temperature = max(
            self.min_temperature,
            self.temperature * decay_rate
        )

    def get_temperature(self) -> float:
        return self.temperature


class ExpertUtilizationMonitor:
    """
    Monitor expert utilization to detect and prevent 'winner-take-all' collapse.
    Tracks gate weight distribution across experts.
    """

    def __init__(self, num_experts: int, num_tasks: int):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.reset()

    def reset(self):
        self.utilization_accum = torch.zeros(self.num_tasks, self.num_experts)
        self.sample_count = 0

    def update(self, task_id: int, gate_weights: torch.Tensor):
        """
        Args:
            task_id: which task's gate
            gate_weights: (batch_size, num_experts)
        """
        with torch.no_grad():
            avg_weights = gate_weights.mean(dim=0).cpu()
            self.utilization_accum[task_id] += avg_weights
            if task_id == 0:
                self.sample_count += 1

    def get_utilization(self) -> torch.Tensor:
        """Returns (num_tasks, num_experts) utilization matrix."""
        if self.sample_count == 0:
            return self.utilization_accum
        return self.utilization_accum / self.sample_count

    def get_load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load balance regularization loss.
        Encourages uniform expert utilization.

        Args:
            gate_weights: (batch_size, num_experts)
        Returns:
            scalar loss
        """
        # Importance: sum of gate values per expert
        importance = gate_weights.sum(dim=0)  # (num_experts,)
        # Load: count of "selected" experts (hard assignment)
        load = (gate_weights > 1.0 / gate_weights.shape[1]).float().sum(dim=0)

        importance_loss = (importance.float().var() /
                           (importance.float().mean() ** 2 + 1e-10))
        load_loss = (load.float().var() /
                     (load.float().mean() ** 2 + 1e-10))

        return importance_loss + load_loss

    def detect_collapse(self, threshold: float = 0.5) -> dict:
        """
        Detect if any expert is dominating (winner-take-all).

        Args:
            threshold: if any single expert gets > threshold of total weight
        Returns:
            dict with collapse detection results
        """
        util = self.get_utilization()
        if util.sum() == 0:
            return {"collapsed": False, "details": "No data yet"}

        # Normalize per task
        util_normalized = util / (util.sum(dim=1, keepdim=True) + 1e-10)
        max_util = util_normalized.max(dim=1)[0]

        collapsed_tasks = (max_util > threshold).nonzero(as_tuple=True)[0].tolist()

        return {
            "collapsed": len(collapsed_tasks) > 0,
            "collapsed_tasks": collapsed_tasks,
            "max_utilization_per_task": max_util.tolist(),
            "utilization_matrix": util_normalized.tolist()
        }
