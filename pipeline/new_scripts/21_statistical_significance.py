"""
Blocker #3 from the 2026-07-26 publishability assessment: "no confidence
intervals, no significance testing between models whose AUCs sit in a tight
0.97-0.99 band. Right now 'Model A beats Model B' isn't a statistically
supported claim anywhere in this project."

Two things, both added here:
  1. Bootstrap 95% CIs for AUC/AUPRC/F1 for every tabular + Hybrid-B model,
     on both the internal test set and the external set (same non-parametric
     percentile-bootstrap method already used in 20_smote_sweep.py).
  2. Pairwise significance testing between every pair of models, via a
     PAIRED bootstrap test on the AUC difference -- not the classical DeLong
     analytical formula. Chosen deliberately: DeLong's test requires
     estimating a structural covariance matrix and is easy to get subtly
     wrong; a paired bootstrap (resample the SAME indices for both models
     each iteration, so the comparison stays paired/correlated exactly like
     DeLong's does) gives an equivalent, more transparently-correct answer
     with less room for implementation error. Two-sided p-value from the
     bootstrap distribution's proportion crossing zero, not a normal
     approximation.

Scope: covers every model that does NOT require reconstructing raw sequence
windows (RandomForest, SVM, GradientBoosting, XGBoost, ML_VotingEnsemble,
StackingEnsemble, Hybrid-B) -- these are exactly the models in the tight
0.97-0.99 AUC band the assessment flagged. The BiLSTM-sequence hybrid and
standalone CNN/BiLSTM are excluded from the pairwise significance matrix
(already shown elsewhere, table_external_validation.csv, to be near
coin-flip standalone and not meaningfully different from tabular-only when
hybridized) -- a disclosed scope limit, not an oversight.
"""
import itertools
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOTSTRAP = 1000

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, in_dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(device))).cpu().numpy()


def get_all_predictions(X, X_raw_for_tree_models):
    """Returns {model_name: proba array} for every tabular + Hybrid-B model."""
    preds = {}
    tabular_models = {
        "RandomForest": "model_randomforest.pkl", "SVM": "model_svm.pkl",
        "GradientBoosting": "model_gradientboosting.pkl", "XGBoost": "model_xgboost.pkl",
        "ML_VotingEnsemble": "model_voting.pkl", "StackingEnsemble": "model_stacking.pkl",
    }
    for name, fname in tabular_models.items():
        model = joblib.load(f"pipeline/models/{fname}")
        preds[name] = model.predict_proba(X_raw_for_tree_models)[:, 1]

    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))
    X_arr = preproc["scaler"].transform(preproc["imputer"].transform(X_raw_for_tree_models))
    p_xgb = xgb_final.predict_proba(X_raw_for_tree_models)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_arr)
    preds["Hybrid-B"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]
    return preds


def bootstrap_ci_metric(y_true, proba, metric_fn, n=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    idx_all = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y_true[idx], proba[idx]))
        except ValueError:
            continue
    point = metric_fn(y_true, proba)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def paired_bootstrap_auc_diff(y_true, proba_a, proba_b, n=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    idx_all = np.arange(len(y_true))
    diffs = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        try:
            diffs.append(roc_auc_score(y_true[idx], proba_a[idx]) - roc_auc_score(y_true[idx], proba_b[idx]))
        except ValueError:
            continue
    diffs = np.array(diffs)
    observed = roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_value = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return observed, lo, hi, p_value


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

    preds_internal = get_all_predictions(X_test, X_test)
    preds_external = get_all_predictions(X_ext, X_ext)

    # --- per-model CIs ---
    ci_rows = []
    for eval_name, y_eval, preds in [("internal-test", y_test, preds_internal), ("external", y_ext, preds_external)]:
        for model_name, proba in preds.items():
            auc, auc_lo, auc_hi = bootstrap_ci_metric(y_eval, proba, roc_auc_score)
            auprc, auprc_lo, auprc_hi = bootstrap_ci_metric(y_eval, proba, average_precision_score)
            f1, f1_lo, f1_hi = bootstrap_ci_metric(
                y_eval, proba, lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0))
            ci_rows.append({
                "eval_set": eval_name, "model": model_name,
                "roc_auc": round(auc, 4), "roc_auc_CI": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
                "auprc": round(auprc, 4), "auprc_CI": f"[{auprc_lo:.3f}, {auprc_hi:.3f}]",
                "f1@0.5": round(f1, 4), "f1_CI": f"[{f1_lo:.3f}, {f1_hi:.3f}]",
            })
            print(f"{eval_name:14s} {model_name:18s} AUC={auc:.4f} [{auc_lo:.3f},{auc_hi:.3f}]  "
                  f"F1={f1:.4f} [{f1_lo:.3f},{f1_hi:.3f}]")
    ci_out = pd.DataFrame(ci_rows)
    ci_out.to_csv("pipeline/table_model_cis.csv", index=False)
    print("\nSaved to pipeline/table_model_cis.csv")

    # --- pairwise significance ---
    sig_rows = []
    for eval_name, y_eval, preds in [("internal-test", y_test, preds_internal), ("external", y_ext, preds_external)]:
        names = list(preds.keys())
        for a, b in itertools.combinations(names, 2):
            diff, lo, hi, p = paired_bootstrap_auc_diff(y_eval, preds[a], preds[b])
            sig_rows.append({
                "eval_set": eval_name, "model_a": a, "model_b": b,
                "auc_a": round(roc_auc_score(y_eval, preds[a]), 4),
                "auc_b": round(roc_auc_score(y_eval, preds[b]), 4),
                "auc_diff": round(diff, 4), "diff_95CI": f"[{lo:.3f}, {hi:.3f}]",
                "p_value": round(p, 4), "significant_at_0.05": bool(p < 0.05),
            })
    sig_out = pd.DataFrame(sig_rows)
    sig_out.to_csv("pipeline/table_pairwise_significance.csv", index=False)
    print("\n=== Pairwise AUC significance (paired bootstrap, n=1000) ===")
    print(sig_out.to_string(index=False))

    n_sig_internal = sig_out[(sig_out["eval_set"] == "internal-test") & sig_out["significant_at_0.05"]].shape[0]
    n_total_internal = sig_out[sig_out["eval_set"] == "internal-test"].shape[0]
    n_sig_external = sig_out[(sig_out["eval_set"] == "external") & sig_out["significant_at_0.05"]].shape[0]
    n_total_external = sig_out[sig_out["eval_set"] == "external"].shape[0]
    print(f"\nSignificant pairs (p<0.05): internal-test {n_sig_internal}/{n_total_internal}, "
          f"external {n_sig_external}/{n_total_external}")
    print("Saved to pipeline/table_pairwise_significance.csv")


if __name__ == "__main__":
    main()
