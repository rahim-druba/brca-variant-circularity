"""
Phase 4 of the BRCA1/2 reproduction pipeline: classical ML on the
internal ML-ready pool (complete-case, 12 scores).

Same architecture as the original methodology (chat_history.md SS3):
RF, SVM (RBF), GradientBoosting, XGBoost, sklearn Pipelines with median
imputation (belt-and-suspenders -- the set is already complete-case, so
this only matters if a future re-run relaxes that), GridSearchCV with
5-fold StratifiedKFold, scoring=roc_auc, 80/20 stratified split
(random_state=42, matching her split for comparability).

Difference from the original: class_weight='balanced' throughout,
since the internal ML-ready set is 3.6% positive (chat_history.md SS18)
-- confirmed with user rather than training naively on that imbalance.

2026-07-25 (Stage 4): added BayesDel_noAF_score as a 12th score -- see
new_scripts/04_filter_and_report.py's docstring for the decision rationale.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "RandomForest": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]},
    ),
    "SVM": (
        Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
        {"clf__C": [1, 10], "clf__gamma": [0.01, 0.1]},
    ),
    "GradientBoosting": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", GradientBoostingClassifier(random_state=42))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [2, 3]},
    ),
    "XGBoost": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", XGBClassifier(eval_metric="logloss", random_state=42))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5]},
    ),
}


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv")
    X = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} pathogenic)  "
          f"Test: {len(X_test)} ({y_test.sum()} pathogenic)")

    # GradientBoosting/XGBoost don't take class_weight directly -- use sample_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    results = {}
    fitted = {}
    for name, (pipe, grid) in MODELS.items():
        print(f"\n=== {name} ===")
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1)
        if name in ("GradientBoosting", "XGBoost"):
            gs.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            gs.fit(X_train, y_train)

        best = gs.best_estimator_
        fitted[name] = best
        proba = best.predict_proba(X_test)[:, 1]
        pred = best.predict(X_test)

        results[name] = {
            "best_params": gs.best_params_,
            "cv_roc_auc": round(gs.best_score_, 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, pred, zero_division=0), 4),
        }
        print(json.dumps(results[name], indent=2))
        joblib.dump(best, f"pipeline/models/model_{name.lower()}.pkl")

    pd.DataFrame(results).T.to_csv("pipeline/table_internal_comparison.csv")
    print("\n=== Summary (also saved to pipeline/table_internal_comparison.csv) ===")
    print(pd.DataFrame(results).T)

    # persist the split for downstream phases (ensembles reuse the same train/test)
    X_train.assign(label=y_train).to_csv("pipeline/internal_train_split.csv", index=False)
    X_test.assign(label=y_test).to_csv("pipeline/internal_test_split.csv", index=False)


if __name__ == "__main__":
    import os
    os.makedirs("pipeline/models", exist_ok=True)
    main()
