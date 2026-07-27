"""Numeric feature scaling for the DonorsChoose dataset.

The original notebook scaled price/quantity/previous-projects with sklearn's
Normalizer called on `.values.reshape(1, -1)` - that reshapes an entire column
into a single sample with N features, so Normalizer computes one L2 norm across
the whole column rather than scaling each value independently. This uses
MinMaxScaler on each column properly shaped as (n_samples, 1), which is what a
per-feature scaler is meant to do. Also fixes a SettingWithCopyWarning from the
original notebook's `donor_data_na[df_norm.columns] = df_norm` in-place mutation
of a frame produced by dropna() (a copy of a slice), by working on an explicit copy.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

NUMERIC_COLUMNS = ["price", "quantity", "teacher_number_of_previously_posted_projects"]


def fit_numeric_scaler(df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df[NUMERIC_COLUMNS])
    return scaler


def scale_numeric_columns(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
    return df
