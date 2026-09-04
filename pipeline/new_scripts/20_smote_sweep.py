"""
Rebuild of the SMOTE sweep -- like Hybrid-B, the ACMG calibration, and the
LOF rule engine, this script no longer existed anywhere in the repo before
this session; only `table_smote_comparison.csv` (root and `2/results/`) and
`docs/memory/project_smote_rejected.md`'s description survived (documented
as a known gap in `CLAUDE.md` since early in the project).

Methodology reconstructed from the surviving table + memory note, verified
by reverse-engineering the exact ratio semantics from the surviving numbers
before writing this: SMOTE "X:1" means oversampling the minority class up to
minority_target = majority_count / X (e.g. 6:1 on a 6636-benign training set
-> target 1106 pathogenic -> 855 synthetic added; this exactly reproduces
the old table's synthetic_added values for 6:1/3:1/1:1, confirming the
formula before trusting it).

Scope note: the original plan (per CLAUDE.md's "Known gaps") was RF/SVM/GB/
XGBoost/ensembles, but what shipped before was RF+XGBoost only. This rebuild
covers all 4 base models (RF/SVM/GB/XGBoost) -- closing that documented
shortfall -- but explicitly excludes ensembles (voting/stacking) from this
pass to bound scope; a disclosed limitation, not a silent one.

Run on the CURRENT frozen 13-feature internal_ml_ready.csv (BayesDel +
Eigen included), not the older feature set the original table used -- so
these numbers supersede the old table rather than reproducing it exactly;
the old table's role here was purely to reverse-engineer the correct
methodology, not to be matched number-for-number.

Bootstrap CIs: 1000 resamples of the (fixed, already-scored) test set
indices with replacement, recomputing each metric per resample, taking the
2.5th/97.5th percentiles -- standard non-parametric bootstrap, not a normal-
approximation CI (safer given F1/precision are bounded and not symmetric).
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, brier_score_loss)
from xgboost import XGBClassifier

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
ARMS = {"baseline (class-weight)": None, "SMOTE 6:1": 6, "SMOTE 3:1": 3, "SMOTE 1:1": 1}
N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
        "SVM": SVC(kernel="rbf", C=1, gamma=0.01, probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, eval_metric="logloss", random_state=42),
    }


def resample_arm(X_train, y_train, ratio):
    """None -> no resampling (class-weighted models handle imbalance directly).
    Integer X -> SMOTE to minority_target = majority_count / X."""
    if ratio is None:
        return X_train, y_train, 0, len(X_train)
    n_majority = (y_train == 0).sum()
    n_minority = (y_train == 1).sum()
    target = int(n_majority / ratio)
    if target <= n_minority:
        return X_train, y_train, 0, len(X_train)
    sm = SMOTE(sampling_strategy={1: target}, random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res, target - n_minority, len(X_res)


def fit_model(name, model, X_train, y_train, is_baseline):
    pipe_steps = [("impute", SimpleImputer(strategy="median"))]
    if name == "SVM":
        pipe_steps.append(("scale", StandardScaler()))
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(pipe_steps + [("clf", model)])

    if is_baseline:
        # baseline arm uses class_weight='balanced' (RF/SVM) or sample_weight (GB/XGB),
        # matching the rest of this project's convention -- SMOTE arms use plain fitting
        # on the already-resampled (rebalanced) data instead.
        if name in ("RandomForest", "SVM"):
            pipe.named_steps["clf"].set_params(class_weight="balanced")
            pipe.fit(X_train, y_train)
        else:
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            sw = np.where(y_train == 1, pos_weight, 1.0)
            pipe.fit(X_train, y_train, clf__sample_weight=sw)
    else:
        pipe.fit(X_train, y_train)
    return pipe


def bootstrap_ci(y_true, y_proba, metric_fn, n=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    idx_all = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y_true[idx], y_proba[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "roc_auc_CI": bootstrap_ci(y_true, proba, roc_auc_score),
        "auprc": round(average_precision_score(y_true, proba), 4),
        "auprc_CI": bootstrap_ci(y_true, proba, average_precision_score),
        "f1@0.5": round(f1_score(y_true, pred, zero_division=0), 4),
        "f1_CI": bootstrap_ci(y_true, proba, lambda y, p: f1_score(y, (p >= threshold).astype(int), zero_division=0)),
        "precision@0.5": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall@0.5": round(recall_score(y_true, pred, zero_division=0), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
    }


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv", low_memory=False)
    X_all = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    ext = pd.read_csv("pipeline/external_ml_ready.csv", low_memory=False)
    X_ext = ext[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_ext = ext["label"].astype(int)

    # SMOTE needs no missing values -- impute before resampling (median, on train only)
    imputer_for_smote = SimpleImputer(strategy="median").fit(X_train)
    X_train_imputed = pd.DataFrame(imputer_for_smote.transform(X_train), columns=SCORE_COLS)

    rows = []
    for model_name, base_model_factory in [(n, f) for n, f in make_models().items()]:
        for arm_name, ratio in ARMS.items():
            is_baseline = ratio is None
            X_res, y_res, n_synth, n_after = resample_arm(X_train_imputed, y_train.reset_index(drop=True), ratio)
            model = make_models()[model_name]
            fitted = fit_model(model_name, model, X_res, y_res, is_baseline)

            for eval_name, X_eval, y_eval in [("internal-test", X_test, y_test), ("external", X_ext, y_ext)]:
                proba = fitted.predict_proba(X_eval)[:, 1]
                metrics = evaluate(y_eval, proba)
                row = {"model": model_name, "arm": arm_name, "eval_set": eval_name,
                       "synthetic_added": n_synth, "train_after_resample": n_after, **metrics}
                rows.append(row)
                print(f"{model_name:16s} {arm_name:22s} {eval_name:14s} "
                      f"AUC={metrics['roc_auc']:.4f} {metrics['roc_auc_CI']:16s} "
                      f"F1={metrics['f1@0.5']:.4f} Brier={metrics['brier']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv("pipeline/table_smote_comparison_13score_frozen.csv", index=False)
    print("\nSaved to pipeline/table_smote_comparison_13score_frozen.csv")

    print("\n=== Brier score trend by arm (should degrade monotonically with more aggressive SMOTE, per the original finding) ===")
    print(out.groupby(["model", "arm"])["brier"].mean().unstack(level=0))


if __name__ == "__main__":
    main()
