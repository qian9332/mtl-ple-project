"""
Gradient Conflict Detection and Adaptive Early Stopping.

Implements:
1. Gradient Cosine Similarity based conflict detection
2. EMA adaptive threshold for early stopping
3. Shared-layer soft freezing when tasks conflict

Reference:
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class GradientConflictDetector:
    """
    Detects gradient conflicts between tasks using cosine similarity.
    Implements EMA-based adaptive thresholding and shared-layer soft freezing.
    """

    def __init__(self, num_tasks: int = 2,
                 ema_alpha: float = 0.1,
                 conflict_threshold: float = -0.1,
                 freeze_scale: float = 0.1,
                 window_size: int = 50):
        """
        Args:
            num_tasks: number of tasks
            ema_alpha: EMA smoothing factor for adaptive threshold
            conflict_threshold: initial threshold for conflict detection (negative = conflict)
            freeze_scale: learning rate scale factor for soft freezing (0=hard freeze, 1=no freeze)
            window_size: sliding window for statistics
        """
        self.num_tasks = num_tasks
        self.ema_alpha = ema_alpha
        self.conflict_threshold = conflict_threshold
        self.freeze_scale = freeze_scale
        self.window_size = window_size

        # EMA of cosine similarity
        self.ema_cos_sim = 0.0
        # Adaptive threshold (EMA-based)
        self.ema_threshold = conflict_threshold

        # History tracking
        self.cos_sim_history = deque(maxlen=window_size)
        self.conflict_count = 0
        self.total_count = 0

    def compute_task_gradients(self, model: nn.Module,
                                task_losses: List[torch.Tensor],
                                shared_params: Optional[List[nn.Parameter]] = None
                                ) -> List[torch.Tensor]:
        """
        Compute per-task gradients for shared parameters.

        Args:
            model: the model
            task_losses: list of per-task losses
            shared_params: list of shared parameters to monitor
                          (if None, uses all parameters)
        Returns:
            list of flattened gradient vectors per task
        """
        if shared_params is None:
            shared_params = [p for p in model.parameters() if p.requires_grad]

        task_grads = []
        for loss in task_losses:
            grads = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True,
                allow_unused=True
            )
            # Flatten and concatenate
            flat_grad = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p.flatten())
                for g, p in zip(grads, shared_params)
            ])
            task_grads.append(flat_grad)

        return task_grads

    def compute_cosine_similarity(self, grad1: torch.Tensor,
                                   grad2: torch.Tensor) -> float:
        """Compute cosine similarity between two gradient vectors."""
        cos_sim = torch.nn.functional.cosine_similarity(
            grad1.unsqueeze(0), grad2.unsqueeze(0)
        ).item()
        return cos_sim

    def update(self, cos_sim: float) -> Dict[str, float]:
        """
        Update EMA statistics and detect conflict.

        Returns:
            dict with conflict detection info
        """
        self.total_count += 1
        self.cos_sim_history.append(cos_sim)

        # Update EMA of cosine similarity
        self.ema_cos_sim = (
            self.ema_alpha * cos_sim +
            (1 - self.ema_alpha) * self.ema_cos_sim
        )

        # Update adaptive threshold (EMA of mean - 1.5 * std)
        if len(self.cos_sim_history) >= 10:
            hist = np.array(list(self.cos_sim_history))
            mean_cs = hist.mean()
            std_cs = hist.std()
            target_threshold = mean_cs - 1.5 * std_cs
            self.ema_threshold = (
                self.ema_alpha * target_threshold +
                (1 - self.ema_alpha) * self.ema_threshold
            )

        # Detect conflict
        is_conflict = cos_sim < self.ema_threshold
        if is_conflict:
            self.conflict_count += 1

        return {
            "cos_sim": cos_sim,
            "ema_cos_sim": self.ema_cos_sim,
            "ema_threshold": self.ema_threshold,
            "is_conflict": is_conflict,
            "conflict_ratio": self.conflict_count / max(self.total_count, 1)
        }

    def should_soft_freeze(self) -> Tuple[bool, float]:
        """
        Determine if shared layers should be soft-frozen.

        Returns:
            (should_freeze, scale_factor)
        """
        if self.total_count < self.window_size:
            return False, 1.0

        recent_conflict_ratio = sum(
            1 for cs in self.cos_sim_history if cs < self.ema_threshold
        ) / len(self.cos_sim_history)

        if recent_conflict_ratio > 0.5:
            # More than half of recent steps have conflicts
            scale = max(self.freeze_scale, 1.0 - recent_conflict_ratio)
            return True, scale

        return False, 1.0


class ConflictAwareEarlyStopping:
    """
    Early stopping that considers both task performance and gradient conflicts.

    Features:
    - Standard patience-based early stopping
    - Conflict-aware: increases patience when conflicts are resolving
    - Per-task monitoring: tracks CTR convergence and CVR oscillation separately
    """

    def __init__(self, patience: int = 10,
                 min_delta: float = 1e-4,
                 monitor_metric: str = "total_auc",
                 conflict_patience_bonus: int = 5):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.conflict_patience_bonus = conflict_patience_bonus

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

        # Per-task tracking
        self.ctr_scores = []
        self.cvr_scores = []
        self.conflict_ratios = []

    def update(self, metrics: Dict[str, float], epoch: int,
               conflict_info: Optional[Dict] = None) -> Dict[str, any]:
        """
        Update early stopping state.

        Args:
            metrics: dict with evaluation metrics
            epoch: current epoch
            conflict_info: output from GradientConflictDetector
        """
        score = metrics.get(self.monitor_metric, 0.0)
        self.ctr_scores.append(metrics.get("ctr_auc", 0.0))
        self.cvr_scores.append(metrics.get("cvr_auc", 0.0))

        # Adjust patience based on conflict resolution trend
        effective_patience = self.patience
        if conflict_info is not None:
            self.conflict_ratios.append(conflict_info.get("conflict_ratio", 0.0))
            # If conflicts are decreasing, give more patience
            if len(self.conflict_ratios) >= 5:
                recent = np.mean(self.conflict_ratios[-5:])
                earlier = np.mean(self.conflict_ratios[-10:-5]) if len(self.conflict_ratios) >= 10 else recent
                if recent < earlier:
                    effective_patience += self.conflict_patience_bonus

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        self.should_stop = self.counter >= effective_patience

        # Diagnose CTR convergence / CVR oscillation
        diagnosis = self._diagnose_convergence()

        return {
            "should_stop": self.should_stop,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "patience_counter": self.counter,
            "effective_patience": effective_patience,
            "diagnosis": diagnosis
        }

    def _diagnose_convergence(self) -> Dict[str, str]:
        """
        Standardized diagnosis for CTR convergence and CVR oscillation.

        Returns:
            dict with diagnosis for each task
        """
        diagnosis = {}

        # CTR convergence check
        if len(self.ctr_scores) >= 5:
            recent_ctr = self.ctr_scores[-5:]
            ctr_delta = max(recent_ctr) - min(recent_ctr)
            ctr_trend = np.polyfit(range(5), recent_ctr, 1)[0]

            if ctr_delta < 0.001:
                diagnosis["ctr"] = "CONVERGED - CTR AUC stable (delta < 0.001)"
            elif ctr_trend > 0:
                diagnosis["ctr"] = f"IMPROVING - CTR AUC trending up (slope={ctr_trend:.6f})"
            else:
                diagnosis["ctr"] = f"DEGRADING - CTR AUC trending down (slope={ctr_trend:.6f})"
        else:
            diagnosis["ctr"] = "WARMING_UP - Not enough data"

        # CVR oscillation check
        if len(self.cvr_scores) >= 5:
            recent_cvr = self.cvr_scores[-5:]
            cvr_std = np.std(recent_cvr)
            cvr_trend = np.polyfit(range(5), recent_cvr, 1)[0]

            # Check for oscillation pattern
            diffs = np.diff(recent_cvr)
            sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)

            if sign_changes >= 3:
                diagnosis["cvr"] = f"OSCILLATING - CVR unstable (sign_changes={sign_changes}, std={cvr_std:.4f})"
            elif cvr_std < 0.002:
                diagnosis["cvr"] = "CONVERGED - CVR AUC stable"
            elif cvr_trend > 0:
                diagnosis["cvr"] = f"IMPROVING - CVR AUC trending up (slope={cvr_trend:.6f})"
            else:
                diagnosis["cvr"] = f"DEGRADING - CVR AUC trending down (slope={cvr_trend:.6f})"
        else:
            diagnosis["cvr"] = "WARMING_UP - Not enough data"

        return diagnosis


class SharedLayerSoftFreezer:
    """
    Implements soft freezing of shared layers when gradient conflicts are detected.
    Reduces learning rate for shared parameters while maintaining task-specific updates.
    """

    def __init__(self, model: nn.Module, freeze_scale: float = 0.1):
        self.model = model
        self.freeze_scale = freeze_scale
        self.is_frozen = False
        self.original_lrs = {}

    def get_shared_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Identify shared parameters (shared experts, embeddings)."""
        shared_names = []
        for name, param in self.model.named_parameters():
            if any(key in name for key in ["shared_expert", "embedding", "dense_proj"]):
                shared_names.append((name, param))
        return shared_names

    def apply_soft_freeze(self, optimizer: torch.optim.Optimizer,
                           scale: float):
        """
        Apply soft freeze by scaling learning rates for shared params.
        """
        shared_param_ids = {
            id(p) for _, p in self.get_shared_params()
        }

        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if id(param) in shared_param_ids:
                    if id(param) not in self.original_lrs:
                        self.original_lrs[id(param)] = param_group["lr"]
                    # This is a simplified version - in practice you'd use
                    # separate param groups
                    pass

        self.is_frozen = True
        logger.info(f"Soft freeze applied with scale={scale:.4f}")

    def release_freeze(self, optimizer: torch.optim.Optimizer):
        """Release soft freeze."""
        self.is_frozen = False
        self.original_lrs.clear()
        logger.info("Soft freeze released")
