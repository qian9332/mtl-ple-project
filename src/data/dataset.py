"""
Data loading and preprocessing for Ali-CCP (Alibaba Click and Conversion Prediction) dataset.

This module handles:
1. Downloading the public Ali-CCP dataset
2. Feature engineering and preprocessing
3. Train/val/test splitting
4. PyTorch DataLoader creation
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, Dict, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


class AliCCPDataset(Dataset):
    """
    Ali-CCP (Alibaba Click and Conversion Prediction) Dataset.

    Features:
    - Sparse features: user/item/context categorical features
    - Dense features: statistical features
    - Labels: click (CTR) and conversion (CVR)
    """

    def __init__(self, sparse_features: np.ndarray, dense_features: np.ndarray,
                 click_labels: np.ndarray, conversion_labels: np.ndarray):
        self.sparse_features = torch.LongTensor(sparse_features)
        self.dense_features = torch.FloatTensor(dense_features)
        self.click_labels = torch.FloatTensor(click_labels)
        self.conversion_labels = torch.FloatTensor(conversion_labels)

    def __len__(self):
        return len(self.click_labels)

    def __getitem__(self, idx):
        return {
            "sparse_features": self.sparse_features[idx],
            "dense_features": self.dense_features[idx],
            "click": self.click_labels[idx],
            "conversion": self.conversion_labels[idx]
        }


def generate_synthetic_aliccp(num_samples: int = 500000,
                                num_sparse: int = 20,
                                num_dense: int = 10,
                                seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data mimicking Ali-CCP dataset structure.
    This provides a realistic simulation when the original dataset is unavailable.

    The generation follows:
    - Sparse features: categorical with varying cardinalities
    - Dense features: normalized continuous features
    - CTR ~ 5%, CVR ~ 2% (realistic click/conversion rates)
    - CVR is conditioned on CTR (conversion only happens after click)
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic Ali-CCP data: {num_samples} samples")

    # Sparse feature cardinalities (simulating user/item/context features)
    sparse_dims = [
        1000,  # user_id_bucket
        500,   # item_id_bucket
        50,    # item_category
        20,    # item_brand_bucket
        100,   # user_age_bucket
        5,     # user_gender
        30,    # user_city_level
        200,   # user_occupation
        100,   # context_page_id
        24,    # context_hour
        7,     # context_weekday
        12,    # context_month
        50,    # user_historical_ctr_bucket
        50,    # user_historical_cvr_bucket
        30,    # item_historical_ctr_bucket
        30,    # item_historical_cvr_bucket
        10,    # position_id
        20,    # match_type
        15,    # campaign_id_bucket
        8,     # adgroup_id_bucket
    ]

    # Ensure we have the right number of sparse features
    sparse_dims = sparse_dims[:num_sparse]
    while len(sparse_dims) < num_sparse:
        sparse_dims.append(50)

    # Generate sparse features
    sparse_features = np.zeros((num_samples, num_sparse), dtype=np.int64)
    for i, dim in enumerate(sparse_dims):
        sparse_features[:, i] = np.random.randint(0, dim, size=num_samples)

    # Generate dense features (normalized)
    dense_features = np.random.randn(num_samples, num_dense).astype(np.float32)
    # Apply min-max to [0, 1]
    scaler = MinMaxScaler()
    dense_features = scaler.fit_transform(dense_features)

    # Generate labels with realistic patterns
    # Create a latent score based on features
    user_affinity = np.random.randn(1000)  # per user bucket
    item_quality = np.random.randn(500)    # per item bucket

    click_score = (
        user_affinity[sparse_features[:, 0]] * 0.3 +
        item_quality[sparse_features[:, 1]] * 0.3 +
        dense_features[:, 0] * 0.2 +
        dense_features[:, 1] * 0.1 +
        np.random.randn(num_samples) * 0.5
    )

    # CTR ~ 5%
    click_prob = 1.0 / (1.0 + np.exp(-(click_score - np.percentile(click_score, 95))))
    click_labels = (np.random.random(num_samples) < click_prob).astype(np.float32)

    # CVR conditioned on click (conversion only if clicked)
    conversion_score = (
        user_affinity[sparse_features[:, 0]] * 0.2 +
        item_quality[sparse_features[:, 1]] * 0.4 +
        dense_features[:, 2] * 0.2 +
        np.random.randn(num_samples) * 0.3
    )
    cvr_prob = 1.0 / (1.0 + np.exp(-(conversion_score - np.percentile(conversion_score, 98))))
    # ESMM: conversion label on full exposure (but mostly 0 for non-clicks)
    conversion_labels = (
        click_labels * (np.random.random(num_samples) < cvr_prob)
    ).astype(np.float32)

    logger.info(f"CTR rate: {click_labels.mean():.4f}, CVR rate: {conversion_labels.mean():.4f}")
    logger.info(f"CVR|Click rate: {conversion_labels[click_labels==1].mean():.4f}")

    return {
        "sparse_features": sparse_features,
        "dense_features": dense_features.astype(np.float32),
        "click_labels": click_labels,
        "conversion_labels": conversion_labels,
        "sparse_dims": sparse_dims
    }


def prepare_dataloaders(data: Dict[str, np.ndarray],
                         batch_size: int = 4096,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         num_workers: int = 4,
                         seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    dataset = AliCCPDataset(
        data["sparse_features"],
        data["dense_features"],
        data["click_labels"],
        data["conversion_labels"]
    )

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    info = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "sparse_dims": data["sparse_dims"],
        "num_sparse": data["sparse_features"].shape[1],
        "num_dense": data["dense_features"].shape[1],
        "ctr_rate": data["click_labels"].mean(),
        "cvr_rate": data["conversion_labels"].mean()
    }

    logger.info(f"Data split: train={train_size}, val={val_size}, test={test_size}")
    return train_loader, val_loader, test_loader, info
