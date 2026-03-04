#!/usr/bin/env python3
"""
Comparison experiment: MMoE vs CGC vs PLE.
Runs all three models and generates a comparison report.

Usage:
    python scripts/run_comparison.py --config configs/ple_config.json --epochs 30
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.data.dataset import generate_synthetic_aliccp, prepare_dataloaders
from src.models.ple import PLEModel
from src.models.baselines import MMoEModel, CGCModel
from src.utils.trainer import MTLTrainer


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"comparison_{timestamp}.log")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ple_config.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--log-dir", type=str, default="logs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    config["training"]["num_epochs"] = args.epochs

    log_file = setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Comparison Experiment Started | Log: {log_file}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data (shared across models)
    data = generate_synthetic_aliccp(
        num_samples=config["data"]["num_samples"],
        num_sparse=config["data"]["num_sparse_features"],
        num_dense=config["data"]["num_dense_features"],
        seed=args.seed
    )
    train_loader, val_loader, test_loader, data_info = prepare_dataloaders(
        data,
        batch_size=config["data"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        num_workers=config["data"]["num_workers"],
        seed=args.seed
    )

    model_config_base = {
        **config.get("model", {}),
        **config.get("training", {}),
        **config.get("gradient_conflict", {}),
        "num_sparse_features": data_info["num_sparse"],
        "sparse_feature_dims": [d + 1 for d in data_info["sparse_dims"]],
        "num_dense_features": data_info["num_dense"],
    }

    results = {}
    models = {
        "mmoe": MMoEModel,
        "cgc": CGCModel,
        "ple": PLEModel,
    }

    for model_name, ModelClass in models.items():
        logger.info("=" * 80)
        logger.info(f"Training {model_name.upper()}")
        logger.info("=" * 80)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        mc = dict(model_config_base)
        if model_name == "mmoe":
            mc["num_experts"] = mc.get("num_task_experts", 3) + mc.get("num_shared_experts", 2)

        model = ModelClass(mc)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{model_name.upper()} params: {total_params:,}")

        model_log_dir = os.path.join(args.log_dir, model_name)
        trainer_config = {**mc, **config.get("training", {}), **config.get("gradient_conflict", {})}
        trainer = MTLTrainer(model, trainer_config)

        start = time.time()
        train_result = trainer.train(
            train_loader, val_loader,
            num_epochs=args.epochs,
            log_dir=model_log_dir
        )
        train_time = time.time() - start

        # Test
        if trainer.best_model_state:
            model.load_state_dict(trainer.best_model_state)
        test_metrics = trainer.evaluate(test_loader)

        results[model_name] = {
            "test_metrics": test_metrics,
            "best_val_auc": train_result["best_val_auc"],
            "total_params": total_params,
            "train_time_seconds": round(train_time, 1),
            "summary": train_result["summary"]
        }

        logger.info(f"\n{model_name.upper()} Test Results: {json.dumps(test_metrics, indent=2)}")

    # Generate comparison report
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)

    header = f"{'Model':<10} {'CTR AUC':>10} {'CVR AUC':>10} {'Total AUC':>10} {'CTR LogLoss':>12} {'CVR LogLoss':>12} {'Params':>10} {'Time(s)':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, res in results.items():
        tm = res["test_metrics"]
        logger.info(
            f"{name.upper():<10} "
            f"{tm['ctr_auc']:>10.6f} "
            f"{tm['cvr_auc']:>10.6f} "
            f"{tm['total_auc']:>10.6f} "
            f"{tm['ctr_logloss']:>12.6f} "
            f"{tm['cvr_logloss']:>12.6f} "
            f"{res['total_params']:>10,} "
            f"{res['train_time_seconds']:>8.1f}"
        )

    # Save comparison
    comparison_path = os.path.join(args.log_dir, "comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
