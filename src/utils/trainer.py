"""
Training and Evaluation Engine for Multi-Task Learning Models.

Implements:
- Full training loop with gradient conflict detection
- Temperature annealing schedule
- Expert utilization monitoring and logging
- CTR/CVR convergence diagnostics
- Comprehensive metric logging
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, Optional, Tuple
import logging

from ..losses.uncertainty_weight import MultiTaskLoss
from ..utils.gradient_conflict import (
    GradientConflictDetector,
    ConflictAwareEarlyStopping,
    SharedLayerSoftFreezer
)

logger = logging.getLogger(__name__)


class MTLTrainer:
    """
    Multi-Task Learning Trainer with comprehensive monitoring.
    """

    def __init__(self, model: nn.Module, config: dict,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function
        self.criterion = MultiTaskLoss(config).to(device)

        # Optimizer
        lr = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 1e-5)

        # Separate param groups for uncertainty weights
        model_params = list(model.parameters())
        loss_params = list(self.criterion.parameters())

        self.optimizer = torch.optim.Adam([
            {"params": model_params, "lr": lr, "weight_decay": weight_decay},
            {"params": loss_params, "lr": lr * 0.1, "weight_decay": 0}
        ])

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 50),
            eta_min=lr * 0.01
        )

        # Gradient conflict detector
        self.conflict_detector = GradientConflictDetector(
            num_tasks=config.get("num_tasks", 2),
            ema_alpha=config.get("ema_alpha", 0.1),
            conflict_threshold=config.get("conflict_threshold", -0.1)
        )

        # Early stopping
        self.early_stopping = ConflictAwareEarlyStopping(
            patience=config.get("patience", 15),
            min_delta=config.get("min_delta", 1e-4)
        )

        # Soft freezer
        self.soft_freezer = SharedLayerSoftFreezer(model)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_log = []
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader,
                     apply_mask: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            "total_loss": 0.0,
            "ctr_loss": 0.0,
            "cvr_loss": 0.0,
            "mask_loss": 0.0,
            "load_balance_loss": 0.0,
            "num_batches": 0,
            "conflict_count": 0,
            "avg_cos_sim": 0.0
        }

        all_ctr_preds, all_ctr_labels = [], []
        all_cvr_preds, all_cvr_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            sparse = batch["sparse_features"].to(self.device)
            dense = batch["dense_features"].to(self.device)
            click = batch["click"].to(self.device)
            conversion = batch["conversion"].to(self.device)

            # Forward pass
            predictions = self.model(sparse, dense, apply_mask=apply_mask)

            # Compute loss
            labels = {"click": click, "conversion": conversion}
            loss_dict = self.criterion(
                predictions, labels,
                gate_weights_list=predictions.get("gate_weights")
            )

            total_loss = loss_dict["total_loss"]

            # Gradient conflict detection (every N steps)
            conflict_info = None
            if self.global_step % self.config.get("conflict_check_interval", 50) == 0:
                conflict_info = self._check_gradient_conflict(predictions, labels)
                if conflict_info and conflict_info["is_conflict"]:
                    epoch_metrics["conflict_count"] += 1
                if conflict_info:
                    epoch_metrics["avg_cos_sim"] += conflict_info["cos_sim"]

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("grad_clip", 1.0)
            )

            # Apply soft freeze if needed
            should_freeze, scale = self.conflict_detector.should_soft_freeze()
            if should_freeze:
                self._apply_grad_scaling(scale)

            self.optimizer.step()
            self.global_step += 1

            # Collect predictions for AUC
            with torch.no_grad():
                all_ctr_preds.extend(predictions["ctr_pred"].cpu().numpy())
                all_ctr_labels.extend(click.cpu().numpy())
                all_cvr_preds.extend(predictions["cvr_pred"].cpu().numpy())
                all_cvr_labels.extend(conversion.cpu().numpy())

            # Accumulate metrics
            epoch_metrics["total_loss"] += loss_dict["total_loss"].item()
            epoch_metrics["ctr_loss"] += loss_dict["ctr_loss"]
            epoch_metrics["cvr_loss"] += loss_dict["cvr_loss"]
            epoch_metrics["mask_loss"] += loss_dict["mask_loss"]
            epoch_metrics["load_balance_loss"] += loss_dict["load_balance_loss"]
            epoch_metrics["num_batches"] += 1

        # Compute epoch averages
        n = epoch_metrics["num_batches"]
        for key in ["total_loss", "ctr_loss", "cvr_loss", "mask_loss", "load_balance_loss"]:
            epoch_metrics[key] /= max(n, 1)

        # Compute AUC
        try:
            epoch_metrics["train_ctr_auc"] = roc_auc_score(all_ctr_labels, all_ctr_preds)
        except ValueError:
            epoch_metrics["train_ctr_auc"] = 0.5
        try:
            epoch_metrics["train_cvr_auc"] = roc_auc_score(all_cvr_labels, all_cvr_preds)
        except ValueError:
            epoch_metrics["train_cvr_auc"] = 0.5

        # Task weights from uncertainty weighting
        if hasattr(self.criterion, "uncertainty_weight"):
            uw = self.criterion.uncertainty_weight
            epoch_metrics["task_weight_ctr"] = torch.exp(-2 * uw.log_sigma[0]).item()
            epoch_metrics["task_weight_cvr"] = torch.exp(-2 * uw.log_sigma[1]).item()
            epoch_metrics["log_sigma_ctr"] = uw.log_sigma[0].item()
            epoch_metrics["log_sigma_cvr"] = uw.log_sigma[1].item()

        return epoch_metrics

    def _check_gradient_conflict(self, predictions, labels) -> Optional[Dict]:
        """Check gradient conflict between CTR and CVR tasks."""
        try:
            ctr_loss = nn.functional.binary_cross_entropy(
                predictions["ctr_pred"], labels["click"].float()
            )
            cvr_loss = nn.functional.binary_cross_entropy(
                predictions["cvr_pred"], labels["conversion"].float()
            )

            # Get shared params (embedding + shared experts)
            shared_params = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and ("shared" in n or "embed" in n or "dense_proj" in n)
            ]

            if not shared_params:
                return None

            task_grads = self.conflict_detector.compute_task_gradients(
                self.model, [ctr_loss, cvr_loss], shared_params
            )

            cos_sim = self.conflict_detector.compute_cosine_similarity(
                task_grads[0], task_grads[1]
            )

            return self.conflict_detector.update(cos_sim)
        except Exception as e:
            logger.warning(f"Gradient conflict check failed: {e}")
            return None

    def _apply_grad_scaling(self, scale: float):
        """Scale gradients of shared parameters for soft freezing."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and (
                "shared" in name or "embed" in name or "dense_proj" in name
            ):
                param.grad.data *= scale

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        self.model.eval()

        all_ctr_preds, all_ctr_labels = [], []
        all_cvr_preds, all_cvr_labels = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in eval_loader:
            sparse = batch["sparse_features"].to(self.device)
            dense = batch["dense_features"].to(self.device)
            click = batch["click"].to(self.device)
            conversion = batch["conversion"].to(self.device)

            predictions = self.model(sparse, dense, apply_mask=False)
            labels = {"click": click, "conversion": conversion}
            loss_dict = self.criterion(predictions, labels)

            total_loss += loss_dict["total_loss"].item()
            n_batches += 1

            all_ctr_preds.extend(predictions["ctr_pred"].cpu().numpy())
            all_ctr_labels.extend(click.cpu().numpy())
            all_cvr_preds.extend(predictions["cvr_pred"].cpu().numpy())
            all_cvr_labels.extend(conversion.cpu().numpy())

        metrics = {"val_loss": total_loss / max(n_batches, 1)}

        # AUC
        try:
            metrics["ctr_auc"] = roc_auc_score(all_ctr_labels, all_ctr_preds)
        except ValueError:
            metrics["ctr_auc"] = 0.5
        try:
            metrics["cvr_auc"] = roc_auc_score(all_cvr_labels, all_cvr_preds)
        except ValueError:
            metrics["cvr_auc"] = 0.5

        metrics["total_auc"] = (metrics["ctr_auc"] + metrics["cvr_auc"]) / 2

        # Log Loss
        try:
            metrics["ctr_logloss"] = log_loss(all_ctr_labels, np.clip(all_ctr_preds, 1e-7, 1-1e-7))
        except ValueError:
            metrics["ctr_logloss"] = float("inf")
        try:
            metrics["cvr_logloss"] = log_loss(all_cvr_labels, np.clip(all_cvr_preds, 1e-7, 1-1e-7))
        except ValueError:
            metrics["cvr_logloss"] = float("inf")

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, log_dir: str = "logs") -> Dict:
        """
        Full training loop.
        """
        os.makedirs(log_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting Multi-Task Learning Training")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")
        logger.info("=" * 80)

        training_history = []
        best_val_auc = 0.0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Temperature annealing
            if hasattr(self.model, "anneal_all_temperatures"):
                self.model.anneal_all_temperatures(
                    decay_rate=self.config.get("temp_decay_rate", 0.995)
                )

            # LR schedule
            self.scheduler.step()

            # Expert utilization check
            collapse_info = None
            if hasattr(self.model, "utilization_monitor"):
                collapse_info = self.model.utilization_monitor.detect_collapse()
                self.model.utilization_monitor.reset()

            # Early stopping
            conflict_info = {
                "conflict_ratio": self.conflict_detector.conflict_count /
                                  max(self.conflict_detector.total_count, 1)
            }
            es_result = self.early_stopping.update(val_metrics, epoch, conflict_info)

            # Save best model
            if val_metrics["total_auc"] > best_val_auc:
                best_val_auc = val_metrics["total_auc"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Compile epoch log
            epoch_time = time.time() - epoch_start
            epoch_log = {
                "epoch": epoch + 1,
                "time_seconds": round(epoch_time, 2),
                "train": train_metrics,
                "val": val_metrics,
                "gate_temperatures": (
                    self.model.get_gate_temperatures()
                    if hasattr(self.model, "get_gate_temperatures") else None
                ),
                "expert_collapse": collapse_info,
                "early_stopping": es_result,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "conflict_detector": {
                    "ema_cos_sim": self.conflict_detector.ema_cos_sim,
                    "ema_threshold": self.conflict_detector.ema_threshold,
                    "conflict_ratio": conflict_info["conflict_ratio"]
                }
            }
            training_history.append(epoch_log)

            # Log to console
            self._log_epoch(epoch_log)

            # Save checkpoint
            if (epoch + 1) % self.config.get("save_interval", 10) == 0:
                self._save_checkpoint(epoch, log_dir)

            # Early stopping check
            if es_result["should_stop"]:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best epoch: {es_result['best_epoch'] + 1}, "
                           f"Best AUC: {es_result['best_score']:.6f}")
                break

        # Save training history
        history_path = os.path.join(log_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2, default=str)

        # Final summary
        summary = self._generate_summary(training_history)
        summary_path = os.path.join(log_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Best Val AUC: {best_val_auc:.6f}")
        logger.info(f"Logs saved to: {log_dir}")
        logger.info("=" * 80)

        return {
            "history": training_history,
            "summary": summary,
            "best_val_auc": best_val_auc
        }

    def _log_epoch(self, log: Dict):
        """Pretty-print epoch results."""
        e = log["epoch"]
        t = log["time_seconds"]
        tr = log["train"]
        va = log["val"]
        es = log["early_stopping"]

        logger.info(
            f"Epoch {e:3d} | Time: {t:.1f}s | "
            f"Train Loss: {tr['total_loss']:.4f} | "
            f"Val Loss: {va['val_loss']:.4f} | "
            f"CTR AUC: {va['ctr_auc']:.4f} | "
            f"CVR AUC: {va['cvr_auc']:.4f} | "
            f"Total AUC: {va['total_auc']:.4f} | "
            f"LR: {log['learning_rate']:.6f}"
        )

        # Log task weights if available
        if "task_weight_ctr" in tr:
            logger.info(
                f"  Task Weights - CTR: {tr['task_weight_ctr']:.4f}, "
                f"CVR: {tr['task_weight_cvr']:.4f} | "
                f"log_σ - CTR: {tr['log_sigma_ctr']:.4f}, "
                f"CVR: {tr['log_sigma_cvr']:.4f}"
            )

        # Log gate temperatures
        if log.get("gate_temperatures"):
            temps = log["gate_temperatures"]
            logger.info(f"  Gate Temps: Layer0={temps[0]}, ...")

        # Log conflict info
        cd = log["conflict_detector"]
        logger.info(
            f"  Conflict: cos_sim_ema={cd['ema_cos_sim']:.4f}, "
            f"threshold={cd['ema_threshold']:.4f}, "
            f"ratio={cd['conflict_ratio']:.4f}"
        )

        # Log convergence diagnosis
        diagnosis = es.get("diagnosis", {})
        if diagnosis:
            logger.info(f"  Diagnosis: CTR={diagnosis.get('ctr', 'N/A')}")
            logger.info(f"  Diagnosis: CVR={diagnosis.get('cvr', 'N/A')}")

        # Expert collapse warning
        if log.get("expert_collapse", {}).get("collapsed"):
            logger.warning("  ⚠️ EXPERT COLLAPSE DETECTED! "
                         f"Tasks: {log['expert_collapse']['collapsed_tasks']}")

    def _save_checkpoint(self, epoch: int, log_dir: str):
        """Save model checkpoint."""
        ckpt_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

    def _generate_summary(self, history: list) -> Dict:
        """Generate training summary."""
        if not history:
            return {}

        best_epoch_idx = max(range(len(history)),
                             key=lambda i: history[i]["val"]["total_auc"])
        best = history[best_epoch_idx]

        # Get frozen uncertainty weights
        frozen_weights = None
        if hasattr(self.criterion, "uncertainty_weight"):
            frozen_weights = self.criterion.uncertainty_weight.get_frozen_weights()

        return {
            "model": self.model.__class__.__name__,
            "total_epochs": len(history),
            "best_epoch": best["epoch"],
            "best_val_metrics": best["val"],
            "best_train_metrics": {
                k: v for k, v in best["train"].items()
                if isinstance(v, (int, float))
            },
            "final_gate_temperatures": history[-1].get("gate_temperatures"),
            "frozen_uncertainty_weights": frozen_weights,
            "conflict_summary": {
                "final_conflict_ratio": history[-1]["conflict_detector"]["conflict_ratio"],
                "final_ema_cos_sim": history[-1]["conflict_detector"]["ema_cos_sim"]
            },
            "convergence_diagnosis": history[-1]["early_stopping"].get("diagnosis", {})
        }
