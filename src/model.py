"""XGBoost classifier with GridSearchCV hyperparameter tuning.

Replaces the notebook's DecisionTreeClassifier (tuned over max_depth/min_samples_split,
topping out around 0.65 test AUC) with XGBClassifier, which handles this kind of
high-dimensional sparse text+categorical feature set far better than a single tree.
"""

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}


def tune_model(X_train, y_train, param_grid: dict = PARAM_GRID, cv: int = 3) -> GridSearchCV:
    base_model = XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42)
    search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring="roc_auc", verbose=1, return_train_score=True, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search
