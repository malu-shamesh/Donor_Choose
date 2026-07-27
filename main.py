"""CLI entrypoint for the DonorsChoose project-approval prediction pipeline.

Usage:
    python main.py --data-path data/Preprocessed_DonorsChoose_dataset.csv
    python main.py --data-path data/sample.csv --vectorizer tfidf
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.data_loader import LABEL_COLUMN, load_donor_data
from src.encoding import fit_encoding_pipeline, transform_features
from src.evaluate import evaluate_and_report
from src.model import tune_model
from src.visualize import (
    plot_approval_by_state,
    plot_class_balance,
    plot_confusion_matrix,
    plot_grid_search_heatmap,
    plot_price_distribution,
    plot_roc_curve,
)

VECTORIZERS = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DonorsChoose project-approval prediction pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the preprocessed DonorsChoose CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/xgboost_donorchoose.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--vectorizer", choices=list(VECTORIZERS), default="bow", help="Text feature vectorizer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading DonorsChoose data...")
    df = load_donor_data(args.data_path)
    print(f"Loaded {len(df)} rows.")

    print("Generating exploratory plots...")
    plot_class_balance(df, args.output_dir)
    plot_approval_by_state(df, args.output_dir)
    plot_price_distribution(df, args.output_dir)

    print(f"Splitting data (test_size={args.test_size}, stratified)...")
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df[LABEL_COLUMN]
    )

    print(f"Fitting {args.vectorizer} encoding pipeline...")
    pipeline = fit_encoding_pipeline(train_df, text_vectorizer_cls=VECTORIZERS[args.vectorizer])
    X_train = transform_features(train_df, pipeline)
    X_test = transform_features(test_df, pipeline)
    y_train, y_test = train_df[LABEL_COLUMN], test_df[LABEL_COLUMN]

    print("Tuning XGBoost via GridSearchCV...")
    search = tune_model(X_train, y_train)
    print(f"Best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    joblib.dump({"model": search.best_estimator_, "pipeline": pipeline}, args.model_path)
    print(f"Saved model to {args.model_path}")

    plot_grid_search_heatmap(pd.DataFrame(search.cv_results_), "max_depth", "n_estimators", args.output_dir)

    print("Evaluating on train and test sets...")
    train_results = evaluate_and_report(search.best_estimator_, X_train, y_train, args.output_dir, "train")
    test_results = evaluate_and_report(search.best_estimator_, X_test, y_test, args.output_dir, "test")
    print(f"Train ROC-AUC: {train_results['roc_auc']:.4f}, Test ROC-AUC: {test_results['roc_auc']:.4f}")

    plot_roc_curve(test_results["fpr"], test_results["tpr"], test_results["roc_auc"], args.output_dir, "roc_test.png", "Test ROC Curve")

    test_matrix = pd.read_csv(args.output_dir / "confusion_matrix_test.csv", index_col=0)
    plot_confusion_matrix(test_matrix, args.output_dir, "confusion_matrix_test.png")

    print(f"Saved evaluation report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
