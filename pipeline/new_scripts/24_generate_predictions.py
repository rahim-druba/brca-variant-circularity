"""
Foundation for every figure in the "figures we need" list (2026-07-29 planning
pass): ROC/PR curves, confusion matrices, calibration plots, and the AUC
forest plot all need per-variant predicted probabilities, not just the
summary AUC/CI numbers already in table_model_cis.csv. Nobody saved those
probability arrays when the models were trained, so this script reloads the
12 already-trained, already-saved models (no retraining) and re-scores the
same internal test set and external set used everywhere else in this
project, writing one wide CSV that every later figure script can read.

Model loading and scoring logic is intentionally identical to
21_statistical_significance.py (same saved model files, same preprocessing,
same train_test_split random_state=42) so the probabilities here are
guaranteed consistent with the numbers already reported in
table_model_cis.csv and table_pairwise_significance.csv -- this script does
not recompute anything, only exposes the per-variant values behind those
already-published summary numbers.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
ID_COLS = ["gene_symbol", "chr", "pos", "ref", "alt", "aachange_1letter"]

RESULTS = "pipeline/final_results_2026-07-26"


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


def get_all_predictions(X_raw):
    """Returns {model_name: proba array} for every tabular + Hybrid-B model.
    Identical logic to 21_statistical_significance.py's get_all_predictions."""
    preds = {}
    tabular_models = {
        "RandomForest": "model_randomforest.pkl", "SVM": "model_svm.pkl",
        "GradientBoosting": "model_gradientboosting.pkl", "XGBoost": "model_xgboost.pkl",
        "ML_VotingEnsemble": "model_voting.pkl", "StackingEnsemble": "model_stacking.pkl",
    }
    for name, fname in tabular_models.items():
        model = joblib.load(f"pipeline/models/{fname}")
        preds[name] = model.predict_proba(X_raw)[:, 1]

    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))
    X_arr = preproc["scaler"].transform(preproc["imputer"].transform(X_raw))
    p_xgb = xgb_final.predict_proba(X_raw)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_arr)
    preds["Hybrid-B"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]
    return preds


def build_rows(eval_name, df_full, X_raw, y):
    preds = get_all_predictions(X_raw)
    out = df_full.loc[X_raw.index, ID_COLS].copy()
    out.insert(0, "eval_set", eval_name)
    out["y_true"] = y.values
    for model_name, proba in preds.items():
        out[f"proba_{model_name}"] = proba
    return out


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

    rows_internal = build_rows("internal-test", df, X_test, y_test)
    rows_external = build_rows("external", ext, X_ext, y_ext)

    out = pd.concat([rows_internal, rows_external], ignore_index=True)
    out_path = f"{RESULTS}/predictions_per_variant.csv"
    out.to_csv(out_path, index=False)

    print(f"internal-test: {len(rows_internal)} variants, external: {len(rows_external)} variants")
    print(f"columns: {list(out.columns)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
