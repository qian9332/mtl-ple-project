"""
Loss Functions for Multi-Task Learning.

Implements:
1. Uncertainty Weighting (Kendall et al., CVPR 2018)
   - Only adds T scalar parameters (log_sigma per task)
   - Post-training: weights are frozen → zero inference overhead
2. Standard multi-task loss with manual weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class UncertaintyWeightLoss(nn.Module):
    """
    Uncertainty-based Multi-Task Loss Weighting.

    loss = sum_t [ (1 / (2 * sigma_t^2)) * L_t + log(sigma_t) ]

    Only adds T learnable scalar parameters (log_sigma for each task).
    After training, the learned weights can be frozen for zero inference overhead.

    Reference:
        Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self, num_tasks: int = 2, initial_log_sigma: float = 0.0):
        super().__init__()
        # Learnable log(sigma) for each task
        self.log_sigma = nn.Parameter(
            torch.full((num_tasks,), initial_log_sigma)
        )
        self.num_tasks = num_tasks

    def forward(self, task_losses: list) -> Dict[str, torch.Tensor]:
        """
        Args:
            task_losses: list of T scalar losses

        Returns:
            dict with:
                - 'total_loss': weighted sum
                - 'task_weights': current task weights (1/(2*sigma^2))
                - 'log_sigma': current log sigma values
        """
        assert len(task_losses) == self.num_tasks

        total_loss = torch.tensor(0.0, device=self.log_sigma.device)
        task_weights = []

        for i, loss in enumerate(task_losses):
            precision = torch.exp(-2 * self.log_sigma[i])  # 1/sigma^2
            weighted_loss = precision * loss + self.log_sigma[i]
            total_loss = total_loss + weighted_loss
            task_weights.append(precision.item())

        return {
            "total_loss": total_loss,
            "task_weights": task_weights,
            "log_sigma": self.log_sigma.detach().cpu().tolist()
        }

    def get_frozen_weights(self) -> list:
        """
        Get frozen weights for inference (zero overhead).
        Call after training is complete.
        """
        with torch.no_grad():
            weights = torch.exp(-2 * self.log_sigma).cpu().tolist()
        return weights


def _compute_load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
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


class MultiTaskLoss(nn.Module):
    """
    Standard multi-task loss with manual or learned weighting.
    Includes ESMM loss, feature mask loss, and load balance loss.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_tasks = config.get("num_tasks", 2)
        self.use_uncertainty_weight = config.get("use_uncertainty_weight", True)
        self.mask_loss_weight = config.get("mask_loss_weight", 0.1)
        self.load_balance_weight = config.get("load_balance_weight", 0.01)

        if self.use_uncertainty_weight:
            self.uncertainty_weight = UncertaintyWeightLoss(
                num_tasks=self.num_tasks,
                initial_log_sigma=config.get("initial_log_sigma", 0.0)
            )
        else:
            # Manual weights
            manual_weights = config.get("manual_task_weights", [1.0, 1.0])
            self.register_buffer(
                "task_weights",
                torch.tensor(manual_weights, dtype=torch.float32)
            )

    def forward(self, predictions: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor],
                gate_weights_list: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-task loss.

        Args:
            predictions: model output dict
            labels: dict with 'click' and 'conversion' labels
            gate_weights_list: nested list of gate weights for load balance loss
        """
        # CTR loss (binary cross-entropy)
        ctr_loss = F.binary_cross_entropy(
            predictions["ctr_pred"],
            labels["click"].float(),
            reduction="mean"
        )

        # CVR loss - ESMM style: computed on ALL samples (full exposure)
        cvr_loss = F.binary_cross_entropy(
            predictions["cvr_pred"],
            labels["conversion"].float(),
            reduction="mean"
        )

        task_losses = [ctr_loss, cvr_loss]

        # Weighted combination
        if self.use_uncertainty_weight:
            result = self.uncertainty_weight(task_losses)
            total_loss = result["total_loss"]
            task_weight_values = result["task_weights"]
            log_sigma = result["log_sigma"]
        else:
            total_loss = sum(w * l for w, l in zip(self.task_weights, task_losses))
            task_weight_values = self.task_weights.tolist()
            log_sigma = [0.0] * self.num_tasks

        # Feature mask reconstruction loss
        mask_loss = predictions.get("mask_loss", torch.tensor(0.0))
        total_loss = total_loss + self.mask_loss_weight * mask_loss

        # Load balance loss
        lb_loss = torch.tensor(0.0, device=total_loss.device)
        if gate_weights_list is not None and self.load_balance_weight > 0:
            for layer_gates in gate_weights_list:
                for gw in layer_gates:
                    lb_loss = lb_loss + _compute_load_balance_loss(gw)
            total_loss = total_loss + self.load_balance_weight * lb_loss

        return {
            "total_loss": total_loss,
            "ctr_loss": ctr_loss.item(),
            "cvr_loss": cvr_loss.item(),
            "mask_loss": mask_loss.item() if isinstance(mask_loss, torch.Tensor) else mask_loss,
            "load_balance_loss": lb_loss.item(),
            "task_weights": task_weight_values,
            "log_sigma": log_sigma
        }
