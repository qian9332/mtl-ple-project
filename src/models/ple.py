"""
Progressive Layered Extraction (PLE) Model.
Implements the full PLE architecture with:
- Per-task exclusive experts + shared experts at each extraction layer
- Gate networks with temperature annealing
- Expert utilization monitoring
- ESMM-style CVR full-exposure branch
- Feature-mask self-supervised auxiliary task

Reference:
    Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL)
    Model for Personalized Recommendations", RecSys 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .experts import ExpertNetwork, GateNetwork, ExpertUtilizationMonitor


class ExtractionLayer(nn.Module):
    """
    Single PLE Extraction Layer.
    Contains task-specific experts, shared experts, and per-task gates.
    """

    def __init__(self, input_dim: int, expert_dim: int,
                 num_task_experts: int, num_shared_experts: int,
                 num_tasks: int, dropout: float = 0.1,
                 initial_temperature: float = 1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_task_experts = num_task_experts
        self.num_shared_experts = num_shared_experts
        total_experts_per_task = num_task_experts + num_shared_experts

        # Task-specific experts: each task has its own set
        self.task_experts = nn.ModuleList()
        for _ in range(num_tasks):
            task_expert_group = nn.ModuleList([
                ExpertNetwork(input_dim, expert_dim, dropout)
                for _ in range(num_task_experts)
            ])
            self.task_experts.append(task_expert_group)

        # Shared experts
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_dim, dropout)
            for _ in range(num_shared_experts)
        ])

        # Per-task gates
        self.gates = nn.ModuleList([
            GateNetwork(input_dim, total_experts_per_task, initial_temperature)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: (batch_size, input_dim) - shared input or previous layer output

        Returns:
            task_outputs: list of (batch_size, expert_dim) for each task
            gate_weights_list: list of gate weights for monitoring
        """
        # Compute shared expert outputs
        shared_outputs = [expert(x) for expert in self.shared_experts]

        task_outputs = []
        gate_weights_list = []

        for task_id in range(self.num_tasks):
            # Task-specific expert outputs
            task_expert_outputs = [
                expert(x) for expert in self.task_experts[task_id]
            ]

            # Combine: [task_experts..., shared_experts...]
            all_expert_outputs = task_expert_outputs + shared_outputs
            # Stack: (batch_size, num_experts, expert_dim)
            expert_stack = torch.stack(all_expert_outputs, dim=1)

            # Gate
            gate_weights = self.gates[task_id](x)  # (batch, num_experts)
            gate_weights_list.append(gate_weights)

            # Weighted combination: (batch, 1, num_experts) @ (batch, num_experts, dim)
            task_output = torch.bmm(
                gate_weights.unsqueeze(1), expert_stack
            ).squeeze(1)  # (batch, expert_dim)

            task_outputs.append(task_output)

        return task_outputs, gate_weights_list

    def anneal_temperature(self, decay_rate: float = 0.995):
        for gate in self.gates:
            gate.anneal_temperature(decay_rate)


class PLEModel(nn.Module):
    """
    Full PLE Model for Multi-Task CTR/CVR Prediction.

    Architecture:
        Input → Embedding → [PLE Extraction Layers x N] → Task Towers → Outputs

    Features:
        - Progressive layered extraction with parameter isolation
        - Gate temperature annealing for expert utilization balance
        - ESMM-CVR full-exposure branch
        - Feature-mask self-supervised auxiliary task
        - Exclusive expert isolation for conflict reduction
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # === Feature Embedding ===
        self.num_sparse_features = config.get("num_sparse_features", 20)
        self.sparse_feature_dims = config.get("sparse_feature_dims", [100] * 20)
        self.embedding_dim = config.get("embedding_dim", 8)
        self.num_dense_features = config.get("num_dense_features", 10)
        self.num_tasks = config.get("num_tasks", 2)  # CTR, CVR

        # Sparse embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim)
            for dim in self.sparse_feature_dims
        ])

        # Dense feature projection
        self.dense_proj = nn.Linear(
            self.num_dense_features,
            self.num_dense_features * self.embedding_dim
        )

        total_embed_dim = (self.num_sparse_features + self.num_dense_features) * self.embedding_dim

        # === PLE Extraction Layers ===
        num_layers = config.get("num_extraction_layers", 3)
        expert_dim = config.get("expert_dim", 128)
        num_task_experts = config.get("num_task_experts", 3)
        num_shared_experts = config.get("num_shared_experts", 2)
        dropout = config.get("dropout", 0.1)
        initial_temp = config.get("initial_temperature", 2.0)

        self.extraction_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = total_embed_dim if i == 0 else expert_dim
            self.extraction_layers.append(
                ExtractionLayer(
                    input_dim=in_dim,
                    expert_dim=expert_dim,
                    num_task_experts=num_task_experts,
                    num_shared_experts=num_shared_experts,
                    num_tasks=self.num_tasks,
                    dropout=dropout,
                    initial_temperature=initial_temp
                )
            )

        # === Task Towers ===
        tower_hidden = config.get("tower_hidden_dim", 64)
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

        # === ESMM: CVR = CTR * CVR_given_click ===
        self.use_esmm = config.get("use_esmm", True)

        # === Feature Mask Self-Supervised Auxiliary Task ===
        self.use_feature_mask = config.get("use_feature_mask", True)
        self.mask_ratio = config.get("mask_ratio", 0.15)
        if self.use_feature_mask:
            self.mask_reconstructor = nn.Sequential(
                nn.Linear(expert_dim, expert_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(expert_dim * 2, total_embed_dim)
            )

        # === Expert Utilization Monitor ===
        total_experts = num_task_experts + num_shared_experts
        self.utilization_monitor = ExpertUtilizationMonitor(
            total_experts, self.num_tasks
        )

        self._total_embed_dim = total_embed_dim

    def _embed_features(self, sparse_features: torch.Tensor,
                         dense_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_features: (batch, num_sparse) LongTensor
            dense_features: (batch, num_dense) FloatTensor
        Returns:
            (batch, total_embed_dim)
        """
        sparse_embeds = []
        for i, emb in enumerate(self.embeddings):
            sparse_embeds.append(emb(sparse_features[:, i]))
        sparse_concat = torch.cat(sparse_embeds, dim=1)  # (batch, num_sparse * emb_dim)

        dense_proj = self.dense_proj(dense_features)  # (batch, num_dense * emb_dim)

        return torch.cat([sparse_concat, dense_proj], dim=1)

    def forward(self, sparse_features: torch.Tensor,
                dense_features: torch.Tensor,
                apply_mask: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict with keys:
                - 'ctr_pred': CTR prediction (batch,)
                - 'cvr_pred': CVR prediction (batch,) - ESMM if enabled
                - 'ctcvr_pred': CTCVR = CTR * CVR (batch,)
                - 'gate_weights': list of gate weights per layer per task
                - 'mask_loss': feature mask reconstruction loss (if apply_mask)
        """
        # Embedding
        embed = self._embed_features(sparse_features, dense_features)

        # Feature masking for self-supervised learning
        mask_loss = torch.tensor(0.0, device=embed.device)
        original_embed = embed.clone()
        if apply_mask and self.use_feature_mask and self.training:
            mask = torch.bernoulli(
                torch.full_like(embed, 1 - self.mask_ratio)
            )
            embed = embed * mask

        # PLE Extraction
        all_gate_weights = []
        layer_input = embed
        task_outputs = None
        for layer in self.extraction_layers:
            task_outputs, gate_weights = layer(layer_input)
            all_gate_weights.append(gate_weights)
            # For next layer, use average of task outputs as shared input
            layer_input = torch.stack(task_outputs, dim=0).mean(dim=0)

        # Task Towers
        task_logits = []
        for task_id in range(self.num_tasks):
            logit = self.task_towers[task_id](task_outputs[task_id]).squeeze(-1)
            task_logits.append(logit)

        ctr_pred = torch.sigmoid(task_logits[0])
        cvr_raw = torch.sigmoid(task_logits[1])

        # ESMM: P(conversion) = P(click) * P(conversion | click)
        if self.use_esmm:
            ctcvr_pred = ctr_pred * cvr_raw
            cvr_pred = ctcvr_pred  # full-exposure CVR
        else:
            cvr_pred = cvr_raw
            ctcvr_pred = ctr_pred * cvr_raw

        # Feature mask reconstruction loss
        if apply_mask and self.use_feature_mask and self.training:
            # Use the first task's extraction output for reconstruction
            reconstructed = self.mask_reconstructor(task_outputs[0])
            mask_loss = F.mse_loss(reconstructed, original_embed)

        # Update utilization monitor
        if self.training:
            for layer_gates in all_gate_weights:
                for task_id, gw in enumerate(layer_gates):
                    self.utilization_monitor.update(task_id, gw.detach())

        return {
            "ctr_pred": ctr_pred,
            "cvr_pred": cvr_pred,
            "ctcvr_pred": ctcvr_pred,
            "gate_weights": all_gate_weights,
            "mask_loss": mask_loss
        }

    def anneal_all_temperatures(self, decay_rate: float = 0.995):
        """Anneal temperature for all gate networks."""
        for layer in self.extraction_layers:
            layer.anneal_temperature(decay_rate)

    def get_gate_temperatures(self) -> List[List[float]]:
        """Return current temperatures for all gates."""
        temps = []
        for layer in self.extraction_layers:
            layer_temps = [g.get_temperature() for g in layer.gates]
            temps.append(layer_temps)
        return temps
