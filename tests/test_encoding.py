import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.encoding import fit_encoding_pipeline, transform_features


def _make_df(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cleaned_essays": [f"students need books help learning classroom {i}" for i in range(n)],
            "cleaned_titles": [f"help our classroom {i}" for i in range(n)],
            "project_subject_categories": ["math_science"] * n,
            "project_subject_subcategories": ["mathematics"] * n,
            "school_state": ["ca", "ny"] * (n // 2),
            "teacher_prefix": ["mrs", "mr"] * (n // 2),
            "project_grade_category": ["grades_3_5"] * n,
            "price": [float(i) for i in range(n)],
            "quantity": [1.0] * n,
            "teacher_number_of_previously_posted_projects": [0.0] * n,
            "project_is_approved": [1, 0] * (n // 2),
        }
    )


def test_transform_features_works_with_bow_vectorizer():
    df = _make_df()
    pipeline = fit_encoding_pipeline(df, text_vectorizer_cls=CountVectorizer)
    X = transform_features(df, pipeline)

    assert X.shape[0] == len(df)


def test_transform_features_works_with_tfidf_vectorizer():
    df = _make_df()
    pipeline = fit_encoding_pipeline(df, text_vectorizer_cls=TfidfVectorizer)
    X = transform_features(df, pipeline)

    assert X.shape[0] == len(df)
