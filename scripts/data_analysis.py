#!/usr/bin/env python3
"""
Comprehensive Data Analysis Script.
Generates EDA report for the Ali-CCP synthetic dataset.

Usage:
    python scripts/data_analysis.py --output data/analysis_report
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import generate_synthetic_aliccp

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def analyze_data(output_dir: str, num_samples: int = 500000):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Task Learning Dataset - Comprehensive Analysis")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_aliccp(num_samples=num_samples)

    sparse = data["sparse_features"]
    dense = data["dense_features"]
    click = data["click_labels"]
    conv = data["conversion_labels"]
    sparse_dims = data["sparse_dims"]

    # Column names
    sparse_names = [
        "user_id_bucket", "item_id_bucket", "item_category", "item_brand",
        "user_age", "user_gender", "user_city_level", "user_occupation",
        "context_page", "context_hour", "context_weekday", "context_month",
        "user_hist_ctr", "user_hist_cvr", "item_hist_ctr", "item_hist_cvr",
        "position_id", "match_type", "campaign_id", "adgroup_id"
    ]
    dense_names = [f"dense_feat_{i}" for i in range(dense.shape[1])]

    report = {}

    # ============================================================
    # 1. Basic Statistics
    # ============================================================
    print("\n📊 1. Basic Statistics")
    print("-" * 40)

    basic_stats = {
        "total_samples": int(num_samples),
        "num_sparse_features": sparse.shape[1],
        "num_dense_features": dense.shape[1],
        "click_rate (CTR)": f"{click.mean():.4f} ({click.mean()*100:.2f}%)",
        "conversion_rate (CVR)": f"{conv.mean():.4f} ({conv.mean()*100:.2f}%)",
        "cvr_given_click": f"{conv[click==1].mean():.4f} ({conv[click==1].mean()*100:.2f}%)" if click.sum() > 0 else "N/A",
        "click_count": int(click.sum()),
        "conversion_count": int(conv.sum()),
        "non_click_count": int((1 - click).sum()),
    }
    report["basic_statistics"] = basic_stats
    for k, v in basic_stats.items():
        print(f"  {k}: {v}")

    # ============================================================
    # 2. Label Distribution
    # ============================================================
    print("\n📊 2. Label Distribution Analysis")
    print("-" * 40)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CTR distribution
    labels = ["No Click", "Click"]
    sizes = [int((1-click).sum()), int(click.sum())]
    axes[0].pie(sizes, labels=labels, autopct="%1.2f%%", colors=["#ff9999", "#66b3ff"])
    axes[0].set_title("CTR Label Distribution")

    # CVR distribution
    labels = ["No Conversion", "Conversion"]
    sizes = [int((1-conv).sum()), int(conv.sum())]
    axes[1].pie(sizes, labels=labels, autopct="%1.2f%%", colors=["#ff9999", "#99ff99"])
    axes[1].set_title("CVR Label Distribution (Full Exposure)")

    # Click → Conversion funnel
    click_cnt = int(click.sum())
    conv_cnt = int(conv.sum())
    no_click_cnt = int((1-click).sum())
    click_no_conv = click_cnt - conv_cnt

    funnel = pd.DataFrame({
        "Stage": ["Impression", "Click", "Conversion"],
        "Count": [num_samples, click_cnt, conv_cnt]
    })
    axes[2].bar(funnel["Stage"], funnel["Count"], color=["#4ECDC4", "#45B7D1", "#96CEB4"])
    axes[2].set_title("Conversion Funnel")
    axes[2].set_ylabel("Count")
    for i, v in enumerate(funnel["Count"]):
        axes[2].text(i, v + num_samples * 0.01, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: label_distribution.png")

    # ============================================================
    # 3. Sparse Feature Analysis
    # ============================================================
    print("\n📊 3. Sparse Feature Analysis")
    print("-" * 40)

    sparse_stats = []
    for i in range(sparse.shape[1]):
        name = sparse_names[i] if i < len(sparse_names) else f"sparse_{i}"
        unique = len(np.unique(sparse[:, i]))
        ctr_per_val = {}
        for v in np.unique(sparse[:, i])[:10]:
            mask = sparse[:, i] == v
            ctr_per_val[int(v)] = float(click[mask].mean())

        sparse_stats.append({
            "feature": name,
            "cardinality": unique,
            "max_cardinality": int(sparse_dims[i]),
            "coverage": f"{unique/sparse_dims[i]*100:.1f}%",
            "top_ctr_values": dict(sorted(ctr_per_val.items(), key=lambda x: -x[1])[:3])
        })
        print(f"  {name}: cardinality={unique}/{sparse_dims[i]}, coverage={unique/sparse_dims[i]*100:.1f}%")

    report["sparse_feature_stats"] = sparse_stats

    # Cardinality chart
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [s["feature"] for s in sparse_stats]
    cards = [s["cardinality"] for s in sparse_stats]
    max_cards = [s["max_cardinality"] for s in sparse_stats]

    x = np.arange(len(names))
    ax.bar(x - 0.2, max_cards, 0.4, label="Max Cardinality", alpha=0.6, color="#FF6B6B")
    ax.bar(x + 0.2, cards, 0.4, label="Actual Unique", alpha=0.8, color="#4ECDC4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Sparse Feature Cardinality")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparse_cardinality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sparse_cardinality.png")

    # ============================================================
    # 4. Dense Feature Analysis
    # ============================================================
    print("\n📊 4. Dense Feature Analysis")
    print("-" * 40)

    dense_df = pd.DataFrame(dense, columns=dense_names)
    dense_stats_dict = dense_df.describe().to_dict()
    report["dense_feature_stats"] = {
        k: {kk: round(float(vv), 6) for kk, vv in v.items()}
        for k, v in dense_stats_dict.items()
    }

    print(dense_df.describe().to_string())

    # Dense feature distributions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < dense.shape[1]:
            ax.hist(dense[:, i], bins=50, alpha=0.7, color="#45B7D1", edgecolor="white")
            ax.set_title(dense_names[i], fontsize=10)
            ax.set_xlabel("")
    plt.suptitle("Dense Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dense_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dense_distributions.png")

    # Correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_data = np.column_stack([dense, click, conv])
    corr_names = dense_names + ["click", "conversion"]
    corr_df = pd.DataFrame(corr_data, columns=corr_names)
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: correlation_matrix.png")

    # ============================================================
    # 5. Label Correlation Analysis (ESMM Motivation)
    # ============================================================
    print("\n📊 5. ESMM Label Correlation Analysis")
    print("-" * 40)

    esmm_stats = {
        "P(click)": float(click.mean()),
        "P(conversion)": float(conv.mean()),
        "P(conversion|click)": float(conv[click == 1].mean()) if click.sum() > 0 else 0,
        "P(conversion|no_click)": float(conv[click == 0].mean()),
        "CTCVR = P(click) * P(conv|click)": float(click.mean() * (conv[click == 1].mean() if click.sum() > 0 else 0)),
        "click_conversion_correlation": float(np.corrcoef(click, conv)[0, 1]),
    }
    report["esmm_analysis"] = esmm_stats
    for k, v in esmm_stats.items():
        print(f"  {k}: {v:.6f}")

    # ============================================================
    # 6. Feature Importance Proxy (CTR variation)
    # ============================================================
    print("\n📊 6. Feature Importance (CTR Variation)")
    print("-" * 40)

    importance = []
    for i in range(sparse.shape[1]):
        name = sparse_names[i] if i < len(sparse_names) else f"sparse_{i}"
        unique_vals = np.unique(sparse[:, i])
        if len(unique_vals) > 1:
            ctr_by_val = [click[sparse[:, i] == v].mean() for v in unique_vals[:50]]
            var = np.var(ctr_by_val)
        else:
            var = 0
        importance.append({"feature": name, "ctr_variance": float(var)})
        print(f"  {name}: CTR variance = {var:.6f}")

    importance.sort(key=lambda x: -x["ctr_variance"])
    report["feature_importance"] = importance

    fig, ax = plt.subplots(figsize=(12, 6))
    imp_df = pd.DataFrame(importance)
    ax.barh(imp_df["feature"], imp_df["ctr_variance"], color="#96CEB4")
    ax.set_xlabel("CTR Variance Across Values")
    ax.set_title("Feature Importance (by CTR Variation)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance.png")

    # ============================================================
    # 7. Sample Sparsity Analysis
    # ============================================================
    print("\n📊 7. Sample Sparsity & Class Imbalance")
    print("-" * 40)

    sparsity = {
        "click_positive_ratio": f"{click.mean()*100:.2f}%",
        "click_imbalance_ratio": f"1:{int(1/click.mean()) if click.mean() > 0 else 'inf'}",
        "conversion_positive_ratio": f"{conv.mean()*100:.2f}%",
        "conversion_imbalance_ratio": f"1:{int(1/conv.mean()) if conv.mean() > 0 else 'inf'}",
        "conversion_given_click_ratio": f"{conv[click==1].mean()*100:.2f}%" if click.sum() > 0 else "N/A",
    }
    report["sparsity_analysis"] = sparsity
    for k, v in sparsity.items():
        print(f"  {k}: {v}")

    # ============================================================
    # Save Report
    # ============================================================
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Full report saved to: {report_path}")

    # Generate markdown summary
    md_report = generate_markdown_report(report, output_dir)
    md_path = os.path.join(output_dir, "ANALYSIS_REPORT.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"✅ Markdown report saved to: {md_path}")

    return report


def generate_markdown_report(report: dict, output_dir: str) -> str:
    md = []
    md.append("# Multi-Task Learning Dataset Analysis Report\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("## 1. Basic Statistics\n")
    for k, v in report.get("basic_statistics", {}).items():
        md.append(f"- **{k}**: {v}")
    md.append("")

    md.append("## 2. Label Distribution\n")
    md.append("![Label Distribution](label_distribution.png)\n")

    md.append("## 3. Sparse Feature Cardinality\n")
    md.append("![Sparse Features](sparse_cardinality.png)\n")
    md.append("| Feature | Cardinality | Max | Coverage |")
    md.append("|---------|-------------|-----|----------|")
    for s in report.get("sparse_feature_stats", []):
        md.append(f"| {s['feature']} | {s['cardinality']} | {s['max_cardinality']} | {s['coverage']} |")
    md.append("")

    md.append("## 4. Dense Feature Distributions\n")
    md.append("![Dense Features](dense_distributions.png)\n")

    md.append("## 5. Feature Correlation\n")
    md.append("![Correlation](correlation_matrix.png)\n")

    md.append("## 6. ESMM Analysis\n")
    for k, v in report.get("esmm_analysis", {}).items():
        md.append(f"- **{k}**: {v:.6f}")
    md.append("")

    md.append("## 7. Feature Importance\n")
    md.append("![Importance](feature_importance.png)\n")

    md.append("## 8. Class Imbalance\n")
    for k, v in report.get("sparsity_analysis", {}).items():
        md.append(f"- **{k}**: {v}")

    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/analysis_report")
    parser.add_argument("--samples", type=int, default=500000)
    args = parser.parse_args()
    analyze_data(args.output, args.samples)
