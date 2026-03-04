#!/usr/bin/env python3
"""
Main training script for Multi-Task Learning CTR/CVR Prediction.

Usage:
    python scripts/train.py --config configs/ple_config.json --model ple
    python scripts/train.py --config configs/ple_config.json --model mmoe
    python scripts/train.py --config configs/ple_config.json --model cgc
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.data.dataset import generate_synthetic_aliccp, prepare_dataloaders
from src.models.ple import PLEModel
from src.models.baselines import MMoEModel, CGCModel
from src.utils.trainer import MTLTrainer


def setup_logging(log_dir: str, model_name: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{model_name}_{timestamp}.log")

    # Reset handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def build_model(model_name: str, config: dict, data_info: dict):
    """Build model based on name."""
    model_config = {
        **config.get("model", {}),
        **config.get("training", {}),
        **config.get("gradient_conflict", {}),
        "num_sparse_features": data_info["num_sparse"],
        "sparse_feature_dims": [d + 1 for d in data_info["sparse_dims"]],
        "num_dense_features": data_info["num_dense"],
    }

    if model_name == "ple":
        return PLEModel(model_config), model_config
    elif model_name == "mmoe":
        model_config["num_experts"] = (
            model_config.get("num_task_experts", 3) +
            model_config.get("num_shared_experts", 2)
        )
        return MMoEModel(model_config), model_config
    elif model_name == "cgc":
        return CGCModel(model_config), model_config
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="MTL Training")
    parser.add_argument("--config", type=str, default="configs/ple_config.json")
    parser.add_argument("--model", type=str, default="ple",
                        choices=["ple", "mmoe", "cgc"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    # Setup
    model_log_dir = os.path.join(args.log_dir, args.model)
    log_file = setup_logging(model_log_dir, args.model)
    logger = logging.getLogger(__name__)

    logger.info(f"Log file: {log_file}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generate/load data
    logger.info("=" * 60)
    logger.info("Step 1: Data Generation")
    logger.info("=" * 60)
    data = generate_synthetic_aliccp(
        num_samples=config["data"]["num_samples"],
        num_sparse=config["data"]["num_sparse_features"],
        num_dense=config["data"]["num_dense_features"],
        seed=args.seed
    )

    # Save data stats
    data_stats = {
        "num_samples": config["data"]["num_samples"],
        "ctr_rate": float(data["click_labels"].mean()),
        "cvr_rate": float(data["conversion_labels"].mean()),
        "cvr_given_click": float(
            data["conversion_labels"][data["click_labels"] == 1].mean()
        ) if data["click_labels"].sum() > 0 else 0,
        "sparse_dims": data["sparse_dims"],
        "num_sparse": config["data"]["num_sparse_features"],
        "num_dense": config["data"]["num_dense_features"]
    }
    with open(os.path.join(model_log_dir, "data_stats.json"), "w") as f:
        json.dump(data_stats, f, indent=2)
    logger.info(f"Data stats: {json.dumps(data_stats, indent=2)}")

    # Create dataloaders
    train_loader, val_loader, test_loader, data_info = prepare_dataloaders(
        data,
        batch_size=config["data"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        num_workers=config["data"]["num_workers"],
        seed=args.seed
    )
    logger.info(f"DataLoaders created: train={data_info['train_size']}, "
                f"val={data_info['val_size']}, test={data_info['test_size']}")

    # Build model
    logger.info("=" * 60)
    logger.info(f"Step 2: Building {args.model.upper()} Model")
    logger.info("=" * 60)
    model, model_config = build_model(args.model, config, data_info)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Model architecture:\n{model}")

    # Build trainer
    trainer_config = {**model_config, **config.get("training", {}), **config.get("gradient_conflict", {})}
    trainer = MTLTrainer(model, trainer_config)

    # Train
    logger.info("=" * 60)
    logger.info("Step 3: Training")
    logger.info("=" * 60)
    start_time = time.time()

    result = trainer.train(
        train_loader, val_loader,
        num_epochs=config["training"]["num_epochs"],
        log_dir=model_log_dir
    )

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.1f}s")

    # Test evaluation
    logger.info("=" * 60)
    logger.info("Step 4: Test Evaluation")
    logger.info("=" * 60)

    # Load best model
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    # Save test results
    with open(os.path.join(model_log_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save frozen uncertainty weights
    if hasattr(trainer.criterion, "uncertainty_weight"):
        frozen = trainer.criterion.uncertainty_weight.get_frozen_weights()
        logger.info(f"Frozen uncertainty weights (zero inference overhead): {frozen}")
        with open(os.path.join(model_log_dir, "frozen_weights.json"), "w") as f:
            json.dump({"frozen_task_weights": frozen}, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
