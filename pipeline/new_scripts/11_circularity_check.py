"""
Stage 5: circularity audit, first concrete check. Not a new model to add to
the comparison table -- a diagnostic to see how much of the ~0.97-0.99 AUC
cluster reported across every model this session actually depends on
features trained on the same kind of clinical labels (ClinVar-derived
pathogenic/benign) that our own internal pool's ground truth also comes
from, versus features with no such training-label exposure at all.

Split of the current 13 SCORE_COLS, per Rastogi et al. 2025 (CAGI6
Annotate-All-Missense benchmark, articles/47 Rastogi.pdf -- the classification
is theirs, independently derived, not ours):
  - ROBUST (no clinical-label training exposure): AlphaMissense_score,
    CADD_raw, CADD_phred, ESM1b_score, PrimateAI_score, Eigen_raw_coding
    (Eigen added 2026-07-26 re-run -- unsupervised by construction per
    Ionita-Laza et al. 2016, articles/53 IonitaLaza.pdf, added to the
    feature set in Stage 4 after this script was first written).
  - CLINICAL-LABEL-BIASED (trained on ClinVar/HGMD-derived P/B labels):
    ClinPred_score, MetaRNN_score, REVEL_score, MetaSVM_score, MetaLR_score,
    VEST4_score, BayesDel_noAF_score.

Trains the same XGBoost architecture (same hyperparameters as
05_classical_ml.py's GridSearchCV-selected best model) on three feature
sets -- full 12, robust-only, biased-only -- on the identical train/test
split used everywhere else this session, and compares internal + external
performance. A large drop for robust-only would say a meaningful share of
the full model's reported AUC is riding on training-label overlap rather
than independent signal; a small drop would be reassuring.

This does NOT resolve circularity by itself (it can't quantify the harder
question of exact training-variant overlap with each individual predictor's
undocumented training set), but it's a direct, checkable first pass rather
than an assumption either way.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

ROBUST_COLS = ["AlphaMissense_score", "CADD_raw", "CADD_phred", "ESM1b_score", "PrimateAI_score",
               "Eigen_raw_coding"]
BIASED_COLS = ["ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score",
               "MetaLR_score", "VEST4_score", "BayesDel_noAF_score"]
FULL_COLS = ROBUST_COLS + BIASED_COLS

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GRID = {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5]}


def fit_and_eval(feature_set, name, X_train, y_train, X_test, y_test, X_ext, y_ext):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", XGBClassifier(eval_metric="logloss", random_state=42))])
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    gs = GridSearchCV(pipe, GRID, scoring="roc_auc", cv=CV, n_jobs=-1)
    gs.fit(X_train[feature_set], y_train, clf__sample_weight=sample_weight)
    best = gs.best_estimator_

    def scores(X, y):
        proba = best.predict_proba(X[feature_set])[:, 1]
        pred = best.predict(X[feature_set])
        return {
            "n_features": len(feature_set),
            "cv_roc_auc": round(gs.best_score_, 4),
            "roc_auc": round(roc_auc_score(y, proba), 4),
            "auprc": round(average_precision_score(y, proba), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
        }

    return {f"{name}_internal": scores(X_test, y_test), f"{name}_external": scores(X_ext, y_ext)}


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv")
    X_all = df[FULL_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    ext = pd.read_csv("pipeline/external_ml_ready.csv")
    X_ext = ext[FULL_COLS].apply(pd.to_numeric, errors="coerce")
    y_ext = ext["label"].astype(int)

    results = {}
    for feature_set, name in [(FULL_COLS, "full13"), (ROBUST_COLS, "robust6"), (BIASED_COLS, "biased7")]:
        print(f"\n=== {name} ({len(feature_set)} features: {feature_set}) ===")
        r = fit_and_eval(feature_set, name, X_train, y_train, X_test, y_test, X_ext, y_ext)
        for k, v in r.items():
            print(f"  {k}: {v}")
        results.update(r)

    out = pd.DataFrame(results).T
    out.to_csv("pipeline/table_circularity_check.csv")
    print("\n=== Circularity check summary (saved to pipeline/table_circularity_check.csv) ===")
    print(out)


if __name__ == "__main__":
    main()
