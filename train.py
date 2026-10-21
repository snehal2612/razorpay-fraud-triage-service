
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from prepare import clean_transactions, add_features


RANDOM_STATE = 42
EXCLUDED = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]

NUMERIC_COLUMNS = [
    "amount_inr",
    "hour_of_day",
    "txn_count_last_24h",
    "avg_amount_30d_inr",
    "distance_from_home_km",
    "is_new_device",
    "account_age_days",
]

ENGINEERED_COLUMNS = [
    "amount_vs_avg_ratio",
    "is_night",
    "high_value",
]

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "city",
    "payment_mode",
]


def load_data():
    raw = pd.read_csv("transactions_raw.csv")
    clean = clean_transactions(raw)
    featured = add_features(clean)

    # Numeric features first, then engineered features,
    # followed by one-hot encoded categorical features.
    numeric = featured[NUMERIC_COLUMNS + ENGINEERED_COLUMNS].copy()
    categorical = pd.get_dummies(
        featured[CATEGORICAL_COLUMNS],
        dtype=int
    )

    X = pd.concat([numeric, categorical], axis=1)
    y = featured["is_fraud"].astype(int)

    return featured, X, y


def split_data(X, y):
    # First: 60% training, 40% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    # Second: divide the temporary 40% equally.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    # Learn the median from training data only.
    median = X_train["distance_from_home_km"].median()

    datasets = []
    for part in [X_train, X_val, X_test]:
        part = part.copy()
        part["distance_from_home_km"] = (
            part["distance_from_home_km"].fillna(median)
        )
        datasets.append(part)

    X_train, X_val, X_test = datasets

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        median,
    )


def make_logistic(class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=42,
        )),
    ])


def print_metrics(name, model, X, y, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{name}: "
        f"Precision={precision_score(y, predictions, zero_division=0):.3f}, "
        f"Recall={recall_score(y, predictions, zero_division=0):.3f}, "
        f"F1={f1_score(y, predictions, zero_division=0):.3f}, "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


def build_final_model():
    """Build and return the final model and its inference settings."""
    _, X, y = load_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        median,
    ) = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    param_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    }

    search = GridSearchCV(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    val_prob = model.predict_proba(X_val)[:, 1]

    thresholds = [
        round(t, 1) for t in np.arange(0.1, 1.0, 0.1)
    ]

    eligible = []
    for threshold in thresholds:
        pred = (val_prob >= threshold).astype(int)
        rec = recall_score(y_val, pred, zero_division=0)

        if rec >= 0.65:
            eligible.append(threshold)

    threshold = max(eligible) if eligible else 0.1

    return {
        "model": model,
        "columns": list(X.columns),
        "median": float(median),
        "threshold": float(threshold),
    }


def main():
    featured, X, y = load_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        median,
    ) = split_data(X, y)

    print("\n========== SPLIT CHECKPOINTS ==========")
    for name, features, target in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        print(
            f"{name}: rows={len(features)}, "
            f"fraud={int(target.sum())}"
        )

    print("Training distance median:", median)
    print("Input columns:", list(X.columns))
    print("Number of input columns:", len(X.columns))

    # 1. Imbalance comparison.
    print("\n========== IMBALANCE COMPARISON ==========")

    plain = make_logistic()
    balanced = make_logistic(class_weight="balanced")

    plain.fit(X_train, y_train)
    balanced.fit(X_train, y_train)

    # Oversample fraud rows from training data only.
    train_table = X_train.copy()
    train_table["is_fraud"] = y_train

    majority = train_table[train_table["is_fraud"] == 0]
    minority = train_table[train_table["is_fraud"] == 1]

    sampled_fraud = minority.sample(
        n=len(majority),
        replace=True,
        random_state=42,
    )

    oversampled = pd.concat(
        [majority, sampled_fraud],
        ignore_index=True,
    )

    X_over = oversampled.drop(columns="is_fraud")
    y_over = oversampled["is_fraud"].astype(int)

    over_model = make_logistic()
    over_model.fit(X_over, y_over)

    print("Oversampled training rows:", len(oversampled))

    print_metrics("Plain Logistic", plain, X_val, y_val)
    print_metrics("Balanced Logistic", balanced, X_val, y_val)
    print_metrics("Oversampled Logistic", over_model, X_val, y_val)

    # 2. Cross-validation on training split only.
    print("\n========== CROSS-VALIDATION ==========")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    lr_scores = cross_val_score(
        make_logistic(class_weight="balanced"),
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    rf_base = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )

    rf_scores = cross_val_score(
        rf_base,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    print(
        f"Balanced Logistic CV F1: "
        f"{lr_scores.mean():.3f} +/- {lr_scores.std():.3f}"
    )
    print(
        f"Untuned Forest CV F1: "
        f"{rf_scores.mean():.3f} +/- {rf_scores.std():.3f}"
    )

    # 3. Grid search on training data only.
    print("\n========== RANDOM FOREST GRID SEARCH ==========")

    grid = GridSearchCV(
        rf_base,
        {
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [5, 10, 20],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print("Grid combinations:", len(grid.cv_results_["params"]))
    print("Best parameters:", grid.best_params_)
    print(f"Best mean CV F1: {grid.best_score_:.3f}")

    # 4. Validation threshold table.
    print("\n========== VALIDATION THRESHOLDS ==========")

    val_prob = model.predict_proba(X_val)[:, 1]
    eligible = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold = round(float(threshold), 1)
        pred = (val_prob >= threshold).astype(int)

        precision = precision_score(
            y_val, pred, zero_division=0
        )
        recall = recall_score(
            y_val, pred, zero_division=0
        )
        f1 = f1_score(y_val, pred, zero_division=0)
        flagged = int(pred.sum())

        print(
            f"t={threshold:.1f}: "
            f"precision={precision:.3f}, "
            f"recall={recall:.3f}, "
            f"F1={f1:.3f}, flagged={flagged}"
        )

        if recall >= 0.65:
            eligible.append(threshold)

    threshold = max(eligible) if eligible else 0.1
    print("Selected threshold:", threshold)

    # 5. Final test: evaluate once at selected threshold.
    print("\n========== FINAL TEST ==========")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    print(
        f"Precision={precision_score(y_test, test_pred, zero_division=0):.3f}"
    )
    print(
        f"Recall={recall_score(y_test, test_pred, zero_division=0):.3f}"
    )
    print(f"F1={f1_score(y_test, test_pred, zero_division=0):.3f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("Payments flagged:", int(test_pred.sum()))
    print(
        "Legitimate payments held per fraud caught:",
        round(fp / tp, 3) if tp else "undefined (no fraud caught)",
    )
    print("Frauds missed:", fn)

    # 6. Leakage experiment: chargeback is deliberately added.
    print("\n========== LEAKAGE EXPERIMENT ==========")

    X_leaky = X.copy()
    X_leaky["chargeback_filed"] = featured[
        "chargeback_filed"
    ].astype(int)

    (
        XL_train, XL_val, _,
        yl_train, yl_val, _,
        _,
    ) = split_data(X_leaky, y)

    leaky_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        **grid.best_params_,
    )
    leaky_model.fit(XL_train, yl_train)

    clean_pred_05 = (
        model.predict_proba(X_val)[:, 1] >= 0.5
    ).astype(int)
    leaky_pred_05 = (
        leaky_model.predict_proba(XL_val)[:, 1] >= 0.5
    ).astype(int)

    print(
        "Clean tuned forest validation F1:",
        round(f1_score(y_val, clean_pred_05), 3),
    )
    print(
        "Leaky forest validation F1:",
        round(f1_score(yl_val, leaky_pred_05), 3),
    )

    # 7. Feature importance.
    print("\n========== TOP FIVE FEATURES ==========")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print(importance.head(5).to_string())

    # 8. Risk committee numbers.
    print("\n========== RISK COMMITTEE CHECK ==========")
    print(
        f"At threshold {threshold:.1f}, "
        f"{int(test_pred.sum())} test payments were flagged; "
        f"{fp} legitimate payments were false positives, "
        f"{tp} frauds were caught and {fn} frauds were missed."
    )

    print("\n========== REUSABLE BUILDER CHECK ==========")
    result = build_final_model()
    print("Returned keys:", list(result.keys()))
    print("Returned column count:", len(result["columns"]))
    print("Returned median:", result["median"])
    print("Returned threshold:", result["threshold"])


if __name__ == "__main__":
    main()
