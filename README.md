# DonorsChoose Project Approval Prediction

Predicts whether a DonorsChoose.org classroom project proposal will be approved, using its text (title, essay), category, teacher, and school metadata. Originally a Kaggle notebook tuning a Decision Tree via GridSearchCV (topping out around 0.65 test AUC) with heavily duplicated code between a "Case 1" (BoW) and "Case 2" (TF-IDF) run; rewritten as a runnable pipeline with XGBoost and one parameterized encoding path for both vectorizer types.

## Pipeline

1. **Data loading** ([src/data_loader.py](src/data_loader.py)) - reads the preprocessed DonorsChoose CSV from a local path (see **Data** below).
2. **Preprocessing** ([src/preprocessing.py](src/preprocessing.py)) - min-max scales `price`, `quantity`, and `teacher_number_of_previously_posted_projects`.
3. **Encoding** ([src/encoding.py](src/encoding.py)) - one-hot encodes categorical columns (subject categories/subcategories, state, teacher prefix, grade), vectorizes `cleaned_essays`/`cleaned_titles` with either bag-of-words or TF-IDF (selectable via `--vectorizer`), and concatenates everything with scaled numeric features into one sparse matrix.
4. **Model** ([src/model.py](src/model.py)) - tunes an `XGBClassifier` over `max_depth`/`n_estimators`/`learning_rate` via `GridSearchCV` scored on ROC-AUC.
5. **Evaluation** ([src/evaluate.py](src/evaluate.py)) - computes ROC-AUC and a confusion matrix on both train and test sets.
6. **Visualization** ([src/visualize.py](src/visualize.py)) - saves the approval class balance, approvals-by-state bar chart, price distribution, grid search heatmap, ROC curve, and confusion matrix as PNGs.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Data

This is a custom-cleaned version of the [DonorsChoose.org Application Screening](https://www.kaggle.com/c/donorschoose-application-screening) dataset (columns like `cleaned_essays`/`cleaned_titles` are already preprocessed text), with no verifiable canonical Kaggle listing for this exact preprocessed file - the original notebook loaded it from a hardcoded `/kaggle/input` path. Place `Preprocessed_DonorsChoose_dataset.csv` under `data/`, or point `--data-path` elsewhere.

## Usage

```bash
# Bag-of-words features (default)
python main.py --data-path data/Preprocessed_DonorsChoose_dataset.csv

# TF-IDF features instead
python main.py --data-path data/Preprocessed_DonorsChoose_dataset.csv --vectorizer tfidf
```

Outputs land in `outputs/` (plots, ROC curve CSVs, confusion matrices for both splits), and the tuned model plus its encoding pipeline are saved to `--model-path` via `joblib`.

## Tests

```bash
pytest
```

## Why these changes

**Model**: the notebook tuned a `DecisionTreeClassifier` over `max_depth`/`min_samples_split`, reaching ~0.65 test AUC on both the BoW and TF-IDF feature sets - a single tree struggles with this kind of high-dimensional sparse text+categorical data. `XGBClassifier` handles it substantially better.

**Duplicated code**: the notebook ran the entire encoding-and-modeling process twice - "Case 1" with `CountVectorizer`, "Case 2" with `TfidfVectorizer" - with each cell copy-pasted and only the vectorizer changed. `fit_encoding_pipeline()` takes the vectorizer class as a parameter, so both runs share one code path (`--vectorizer bow|tfidf`).

**Normalizer bug**: the notebook scaled `price`/`quantity`/`teacher_number_of_previously_posted_projects` with `Normalizer().fit(X_train['price'].values.reshape(1,-1))`. Reshaping a column to `(1, -1)` turns the entire column into a single sample with N features, so `Normalizer` computes one L2 norm across all of a column's values combined - not what a per-feature scaler is supposed to do, and it silently produces different scaling depending on what else happens to be reshaped alongside it. `MinMaxScaler` is used instead, on properly shaped `(n_samples, n_features)` data, and a test (`test_scale_numeric_columns_scales_each_feature_independently`) verifies each column now scales independently of the others.

**`SettingWithCopyWarning`**: the notebook mutated `donor_data_na[df_norm.columns] = df_norm` in place on a DataFrame produced by `.dropna()` (a copy of a slice), which pandas flags as ambiguous. `scale_numeric_columns()` works on an explicit `.copy()` instead.

**Verification**: the encoding pipeline was tested with both `CountVectorizer` and `TfidfVectorizer` to confirm the deduplicated code path works for both cases the notebook ran separately.
