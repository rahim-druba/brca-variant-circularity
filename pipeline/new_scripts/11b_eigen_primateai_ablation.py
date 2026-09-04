"""
Stage 4: empirical basis for the Eigen/EigenPC and PrimateAI decisions.
Companion to 11_circularity_check.py -- same XGBoost architecture, same
train/test split, same internal_ml_ready.csv/external_ml_ready.csv sources.

Eigen/EigenPC (Ionita-Laza et al. 2016, articles/53 IonitaLaza.pdf): unsupervised,
no clinical-label training exposure at all -- the "most robust" category, even
more so than AlphaMissense/CADD/PrimateAI/ESM1b (those still use population
allele-frequency signal; Eigen uses only the annotation covariance structure).
Question: does adding it to the feature set add real incremental signal, or is
it redundant with the conservation/protein-function information already
captured by the existing robust features?

PrimateAI (Sundaram et al. 2018, articles/46 Sundaram.pdf): already one of our
12 features. Its own authors explicitly asked that its scores not be
incorporated as input to other classifiers, to avoid compounding circularity
in the field's benchmark ecosystem. Checked separately (citation_log.md) that
none of our other 11 features use PrimateAI as a training-time component, so
this isn't a within-feature-set double-counting problem -- it's a scholarly-
practice question about whether dropping it costs real performance or not.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

BASE12 = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score",
]
EIGEN_COLS = ["Eigen_raw_coding", "Eigen_PC_raw_coding"]

VARIANTS = {
    "full12_baseline": BASE12,
    "full12_plus_eigen": BASE12 + EIGEN_COLS,
    "full12_minus_primateai": [c for c in BASE12 if c != "PrimateAI_score"],
    "full12_minus_primateai_plus_eigen": [c for c in BASE12 if c != "PrimateAI_score"] + EIGEN_COLS,
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GRID = {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5]}


def fit_and_eval(feature_set, X_train, y_train, X_test, y_test, X_ext, y_ext):
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

    return scores(X_test, y_test), scores(X_ext, y_ext)


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv")
    all_cols = BASE12 + EIGEN_COLS
    X_all = df[all_cols].apply(pd.to_numeric, errors="coerce")
    y_all = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    ext = pd.read_csv("pipeline/external_ml_ready.csv")
    X_ext = ext[all_cols].apply(pd.to_numeric, errors="coerce")
    y_ext = ext["label"].astype(int)

    # correlation check: is Eigen redundant with the existing robust features?
    robust_and_eigen = ["AlphaMissense_score", "CADD_raw", "ESM1b_score", "PrimateAI_score"] + EIGEN_COLS
    print("=== Correlation matrix (robust features + Eigen/EigenPC, internal train, pairwise-complete) ===")
    print(X_train[robust_and_eigen].corr().round(2))
    print()

    results = {}
    for name, cols in VARIANTS.items():
        print(f"=== {name} ({len(cols)} features) ===")
        int_scores, ext_scores = fit_and_eval(cols, X_train, y_train, X_test, y_test, X_ext, y_ext)
        results[f"{name}_internal"] = int_scores
        results[f"{name}_external"] = ext_scores
        print(f"  internal: {int_scores}")
        print(f"  external: {ext_scores}")

    out = pd.DataFrame(results).T
    out.to_csv("pipeline/table_eigen_primateai_ablation.csv")
    print("\n=== Summary (saved to pipeline/table_eigen_primateai_ablation.csv) ===")
    print(out)


if __name__ == "__main__":
    main()
