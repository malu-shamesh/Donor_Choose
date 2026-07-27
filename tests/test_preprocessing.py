import pandas as pd
import pytest

from src.preprocessing import fit_numeric_scaler, scale_numeric_columns


def test_scale_numeric_columns_scales_each_feature_independently():
    """Regression test for the original Normalizer(.values.reshape(1,-1)) bug.

    That bug scaled an entire column as one sample, so scaling one column's
    values would depend on values in a completely different column. A correct
    per-feature scaler must produce results independent of other columns.
    """
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "quantity": [1.0, 1.0, 1.0],
            "teacher_number_of_previously_posted_projects": [0.0, 5.0, 10.0],
        }
    )
    scaler = fit_numeric_scaler(df)
    scaled = scale_numeric_columns(df, scaler)

    assert scaled["price"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert scaled["quantity"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_scale_numeric_columns_does_not_mutate_input():
    df = pd.DataFrame({"price": [10.0, 20.0], "quantity": [1.0, 2.0], "teacher_number_of_previously_posted_projects": [0.0, 1.0]})
    scaler = fit_numeric_scaler(df)
    scale_numeric_columns(df, scaler)

    assert df["price"].tolist() == [10.0, 20.0]
