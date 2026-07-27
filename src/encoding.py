"""Feature encoding: categorical one-hot, numeric scaling, and text vectorization.

The original notebook duplicated this entire encoding process twice, once for a
BoW feature set ("Case 1") and once for a TF-IDF feature set ("Case 2"), with
each cell copy-pasted and only the vectorizer class changed. This module takes
the vectorizer type as a parameter instead, so there's one code path for both.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import NUMERIC_COLUMNS, fit_numeric_scaler, scale_numeric_columns

CATEGORICAL_COLUMNS = [
    "project_subject_categories",
    "project_subject_subcategories",
    "school_state",
    "teacher_prefix",
    "project_grade_category",
]
TEXT_COLUMNS = ["cleaned_essays", "cleaned_titles"]


@dataclass
class EncodingPipeline:
    categorical_vectorizers: dict
    text_vectorizer_essays: object
    text_vectorizer_titles: object
    numeric_scaler: MinMaxScaler


def fit_encoding_pipeline(train_df: pd.DataFrame, text_vectorizer_cls=CountVectorizer) -> EncodingPipeline:
    """Fit one-hot encoders for categoricals, a text vectorizer for essays/titles, and a numeric scaler."""
    categorical_vectorizers = {}
    for col in CATEGORICAL_COLUMNS:
        vectorizer = CountVectorizer(lowercase=False, binary=True)
        vectorizer.fit(train_df[col].values)
        categorical_vectorizers[col] = vectorizer

    text_vectorizer_essays = text_vectorizer_cls(ngram_range=(2, 2), min_df=10, max_features=5000)
    text_vectorizer_essays.fit(train_df["cleaned_essays"])

    text_vectorizer_titles = text_vectorizer_cls(ngram_range=(2, 2), min_df=10, max_features=5000)
    text_vectorizer_titles.fit(train_df["cleaned_titles"])

    numeric_scaler = fit_numeric_scaler(train_df)

    return EncodingPipeline(categorical_vectorizers, text_vectorizer_essays, text_vectorizer_titles, numeric_scaler)


def transform_features(df: pd.DataFrame, pipeline: EncodingPipeline):
    """Encode categoricals, text, and numeric columns into a single sparse feature matrix."""
    parts = [
        pipeline.text_vectorizer_essays.transform(df["cleaned_essays"]),
        pipeline.text_vectorizer_titles.transform(df["cleaned_titles"]),
    ]
    for col in CATEGORICAL_COLUMNS:
        parts.append(pipeline.categorical_vectorizers[col].transform(df[col].values))

    scaled_df = scale_numeric_columns(df, pipeline.numeric_scaler)
    parts.append(scaled_df[NUMERIC_COLUMNS].to_numpy())

    return hstack(parts).tocsr()
