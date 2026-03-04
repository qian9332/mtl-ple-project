"""
Baseline Models: MMoE and CGC for comparison experiments.

MMoE: Multi-gate Mixture-of-Experts (Ma et al., KDD 2018)
CGC:  Customized Gate Control - simplified PLE with single extraction layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .experts import ExpertNetwork, GateNetwork


class MMoEModel(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE).
    All experts are shared; each task has its own gate.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_sparse_features = config.get("num_sparse_features", 20)
        self.sparse_feature_dims = config.get("sparse_feature_dims", [100] * 20)
        self.embedding_dim = config.get("embedding_dim", 8)
        self.num_dense_features = config.get("num_dense_features", 10)
        self.num_tasks = config.get("num_tasks", 2)
        num_experts = config.get("num_experts", 6)
        expert_dim = config.get("expert_dim", 128)
        dropout = config.get("dropout", 0.1)
        tower_hidden = config.get("tower_hidden_dim", 64)

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim)
            for dim in self.sparse_feature_dims
        ])
        self.dense_proj = nn.Linear(
            self.num_dense_features,
            self.num_dense_features * self.embedding_dim
        )
        total_embed_dim = (self.num_sparse_features + self.num_dense_features) * self.embedding_dim

        # Shared experts
        self.experts = nn.ModuleList([
            ExpertNetwork(total_embed_dim, expert_dim, dropout)
            for _ in range(num_experts)
        ])

        # Per-task gates
        self.gates = nn.ModuleList([
            GateNetwork(total_embed_dim, num_experts)
            for _ in range(self.num_tasks)
        ])

        # Task towers
        self.task_towers = nn.ModuleList()
        for _ in range(self.num_tasks):
            tower = nn.Sequential(
                nn.Linear(expert_dim, tower_hidden),
                nn.BatchNorm1d(tower_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, tower_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(tower_hidden // 2, 1)
            )
            self.task_towers.append(tower)

    def _embed_features(self, sparse_features, dense_features):
        sparse_embeds = [emb(sparse_features[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_concat = torch.cat(sparse_embeds, dim=1)
        dense_proj = self.dense_proj(dense_features)
        return torch.cat([sparse_concat, dense_proj], dim=1)

    def forward(self, sparse_features, dense_features, **kwargs) -> Dict[str, torch.Tensor]:
        embed = self._embed_features(sparse_features, dense_features)

        expert_outputs = [expert(embed) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        task_logits = []
        all_gate_weights = []
        for task_id in range(self.num_tasks):
            gate_weights = self.gates[task_id](embed)
            all_gate_weights.append(gate_weights)
            task_input = torch.bmm(gate_weights.unsqueeze(1), expert_stack).squeeze(1)
            logit = self.task_towers[task_id](task_input).squeeze(-1)
            task_logits.append(logit)

        ctr_pred = torch.sigmoid(task_logits[0])
        cvr_pred = torch.sigmoid(task_logits[1])

        return {
            "ctr_pred": ctr_pred,
            "cvr_pred": cvr_pred,
            "ctcvr_pred": ctr_pred * cvr_pred,
            "gate_weights": [all_gate_weights],
            "mask_loss": torch.tensor(0.0, device=embed.device)
        }


class CGCModel(nn.Module):
    """
    Customized Gate Control (CGC).
    Single extraction layer with task-specific + shared experts.
    Essentially PLE with 1 extraction layer.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_sparse_features = config.get("num_sparse_features", 20)
        self.sparse_feature_dims = config.get("sparse_feature_dims", [100] * 20)
        self.embedding_dim = config.get("embedding_dim", 8)
        self.num_dense_features = config.get("num_dense_features", 10)
        self.num_tasks = config.get("num_tasks", 2)
        num_task_experts = config.get("num_task_experts", 3)
        num_shared_experts = config.get("num_shared_experts", 2)
        expert_dim = config.get("expert_dim", 128)
        dropout = config.get("dropout", 0.1)
        tower_hidden = config.get("tower_hidden_dim", 64)

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim)
            for dim in self.sparse_feature_dims
        ])
        self.dense_proj = nn.Linear(
            self.num_dense_features,
            self.num_dense_features * self.embedding_dim
        )
        total_embed_dim = (self.num_sparse_features + self.num_dense_features) * self.embedding_dim

        total_experts = num_task_experts + num_shared_experts

        # Task-specific experts
        self.task_experts = nn.ModuleList()
        for _ in range(self.num_tasks):
            self.task_experts.append(nn.ModuleList([
                ExpertNetwork(total_embed_dim, expert_dim, dropout)
                for _ in range(num_task_experts)
            ]))

        # Shared experts
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(total_embed_dim, expert_dim, dropout)
            for _ in range(num_shared_experts)
        ])

        # Gates
        self.gates = nn.ModuleList([
            GateNetwork(total_embed_dim, total_experts)
            for _ in range(self.num_tasks)
        ])

        # Task towers
        self.task_towers = nn.ModuleList()
        for _ in range(self.num_tasks):
            tower = nn.Sequential(
                nn.Linear(expert_dim, tower_hidden),
                nn.BatchNorm1d(tower_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, tower_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(tower_hidden // 2, 1)
            )
            self.task_towers.append(tower)

    def _embed_features(self, sparse_features, dense_features):
        sparse_embeds = [emb(sparse_features[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_concat = torch.cat(sparse_embeds, dim=1)
        dense_proj = self.dense_proj(dense_features)
        return torch.cat([sparse_concat, dense_proj], dim=1)

    def forward(self, sparse_features, dense_features, **kwargs) -> Dict[str, torch.Tensor]:
        embed = self._embed_features(sparse_features, dense_features)

        shared_outputs = [expert(embed) for expert in self.shared_experts]

        task_logits = []
        all_gate_weights = []
        for task_id in range(self.num_tasks):
            task_outputs = [expert(embed) for expert in self.task_experts[task_id]]
            all_outputs = task_outputs + shared_outputs
            expert_stack = torch.stack(all_outputs, dim=1)

            gate_weights = self.gates[task_id](embed)
            all_gate_weights.append(gate_weights)

            task_input = torch.bmm(gate_weights.unsqueeze(1), expert_stack).squeeze(1)
            logit = self.task_towers[task_id](task_input).squeeze(-1)
            task_logits.append(logit)

        ctr_pred = torch.sigmoid(task_logits[0])
        cvr_pred = torch.sigmoid(task_logits[1])

        return {
            "ctr_pred": ctr_pred,
            "cvr_pred": cvr_pred,
            "ctcvr_pred": ctr_pred * cvr_pred,
            "gate_weights": [all_gate_weights],
            "mask_loss": torch.tensor(0.0, device=embed.device)
        }
