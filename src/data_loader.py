"""Load the preprocessed DonorsChoose dataset from a local CSV file.

This is a custom-cleaned version of the DonorsChoose.org Application Screening
dataset (columns like cleaned_titles/cleaned_essays are already preprocessed),
with no verifiable canonical Kaggle listing - the original notebook loaded it
from a hardcoded /kaggle/input path. See README.md for where to obtain it.
"""

from pathlib import Path

import pandas as pd

LABEL_COLUMN = "project_is_approved"


def load_donor_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.dropna()
