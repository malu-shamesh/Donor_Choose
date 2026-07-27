"""Plotting helpers that save figures to an output directory instead of displaying inline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["project_is_approved"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, labels=["Approved", "Not Approved"], colors=["#1f77b4", "#d62728"], autopct="%1.1f%%", startangle=140)
    ax.set_title("Project Approval Distribution")
    _save(fig, output_dir, "class_balance.png")


def plot_approval_by_state(df: pd.DataFrame, output_dir: Path) -> None:
    approved = df[df["project_is_approved"] == 1].groupby("school_state").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    approved.plot(kind="bar", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Approved Project Count")
    ax.set_title("Approved Projects by State")
    _save(fig, output_dir, "approval_by_state.png")


def plot_price_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[df["project_is_approved"] == 1]["price"].hist(bins=50, ax=axes[0])
    axes[0].set_title("Price Distribution (Approved)")
    df[df["project_is_approved"] == 0]["price"].hist(bins=50, ax=axes[1])
    axes[1].set_title("Price Distribution (Not Approved)")
    _save(fig, output_dir, "price_distribution.png")


def plot_roc_curve(fpr, tpr, roc_auc: float, output_dir: Path, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    _save(fig, output_dir, filename)


def plot_confusion_matrix(matrix_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Confusion Matrix")
    _save(fig, output_dir, filename)


def plot_grid_search_heatmap(cv_results: pd.DataFrame, param_x: str, param_y: str, output_dir: Path) -> None:
    score_cols = ["mean_test_score", "mean_train_score"]
    scores = (
        cv_results[[f"param_{param_y}", f"param_{param_x}", *score_cols]]
        .groupby([f"param_{param_y}", f"param_{param_x}"])
        .max()
        .unstack()[score_cols]
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(scores["mean_train_score"], annot=True, fmt=".4g", ax=axes[0])
    sns.heatmap(scores["mean_test_score"], annot=True, fmt=".4g", ax=axes[1], cmap="YlGnBu")
    axes[0].set_title("Train Set")
    axes[1].set_title("CV Set")
    _save(fig, output_dir, "grid_search_heatmap.png")
