"""
Stage 8: new benchmarking, per Rastogi et al. 2025's methodology
(articles/47 Rastogi.pdf, CAGI6 Annotate-All-Missense benchmark).

Two of the three originally-scoped checks are done here:
  1. High-specificity (FPR<=5%) / high-sensitivity (TPR>=95%) regime
     reporting -- headline AUC hides how a model behaves in the clinically
     relevant low-FPR region specifically; Rastogi's benchmark showed
     predictor rankings can shift substantially between the full-ROC and
     high-specificity regimes.
  2. Gene-label-balanced subset check -- our internal pool is NOT
     gene-balanced (BRCA1: 171 path/2679 benign = 6.0% prior; BRCA2: 143
     path/5616 benign = 2.5% prior -- BRCA1's baseline pathogenic rate is
     more than double BRCA2's). A model could exploit this gene-level base
     rate difference as a shortcut even without gene_symbol as an explicit
     feature, if score distributions happen to correlate with gene identity.
     Checked by evaluating per-gene (BRCA1-only, BRCA2-only) AND on a
     downsampled subset with equal pathogenic:benign ratio in both genes.

The third (BRCA1 deep mutational scanning validation, e.g. Findlay et al.
2018 SGE data) is NOT done here -- no DMS data file exists anywhere in this
repo (checked via `find . -iname "*findlay*" -o -iname "*starita*" -o -iname
"*dms*"` -- zero real hits), so it would need a new data acquisition step,
scoped out of this pass.

Models covered: RandomForest, SVM, GradientBoosting, XGBoost,
ML_VotingEnsemble, StackingEnsemble, Hybrid-B (all tabular-score-only,
already-saved models from pipeline/models/). The BiLSTM-hybrid and standalone
CNN/BiLSTM are excluded -- they need reconstructing raw sequence windows for
every new subset, and were already shown this session to add negligible real
signal beyond the tabular scores (external AUC 0.53/0.54 standalone), so
they're not worth the added complexity for this specific check.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"

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


def load_models():
    models = {
        "RandomForest": joblib.load("pipeline/models/model_randomforest.pkl"),
        "SVM": joblib.load("pipeline/models/model_svm.pkl"),
        "GradientBoosting": joblib.load("pipeline/models/model_gradientboosting.pkl"),
        "XGBoost": joblib.load("pipeline/models/model_xgboost.pkl"),
        "ML_VotingEnsemble": joblib.load("pipeline/models/model_voting.pkl"),
        "StackingEnsemble": joblib.load("pipeline/models/model_stacking.pkl"),
    }
    hybrid_b = {
        "preproc": joblib.load("pipeline/models/model_hybrid_b_preproc.pkl"),
        "xgb": joblib.load("pipeline/models/model_xgboost.pkl"),
        "meta": joblib.load("pipeline/models/model_hybrid_b_meta.pkl"),
    }
    mlp = MLP(len(SCORE_COLS)).to(device)
    mlp.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))
    hybrid_b["mlp"] = mlp
    return models, hybrid_b


def predict_all(models, hybrid_b, X):
    proba = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    X_arr = hybrid_b["preproc"]["scaler"].transform(hybrid_b["preproc"]["imputer"].transform(X))
    p_xgb = hybrid_b["xgb"].predict_proba(X)[:, 1]
    p_mlp = mlp_predict_proba(hybrid_b["mlp"], X_arr)
    proba["Hybrid-B"] = hybrid_b["meta"].predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]
    return proba


def high_spec_high_sens(y_true, proba):
    if len(set(y_true)) < 2:
        return {"tpr_at_fpr5": float("nan"), "fpr_at_tpr95": float("nan")}
    fpr, tpr, _ = roc_curve(y_true, proba)
    # TPR at the largest FPR still <= 0.05 (high-specificity regime)
    idx_spec = np.where(fpr <= 0.05)[0]
    tpr_at_fpr5 = tpr[idx_spec].max() if len(idx_spec) else 0.0
    # FPR at the smallest TPR still >= 0.95 (high-sensitivity regime)
    idx_sens = np.where(tpr >= 0.95)[0]
    fpr_at_tpr95 = fpr[idx_sens].min() if len(idx_sens) else 1.0
    return {"tpr_at_fpr5": round(tpr_at_fpr5, 4), "fpr_at_tpr95": round(fpr_at_tpr95, 4)}


def gene_balance_downsample(df, seed=42):
    """Equalize BOTH gene count and within-gene pathogenic:benign ratio,
    downsampling to the most restrictive common denominator."""
    rng = np.random.RandomState(seed)
    counts = df.groupby(["gene_symbol", "label"]).size()
    n_per_cell = counts.min()  # smallest (gene, label) cell size -> full balance
    parts = []
    for gene in df["gene_symbol"].unique():
        for label in [0, 1]:
            cell = df[(df["gene_symbol"] == gene) & (df["label"] == label)]
            parts.append(cell.sample(n=n_per_cell, random_state=seed))
    return pd.concat(parts), n_per_cell


def main():
    models, hybrid_b = load_models()

    internal_all = pd.read_csv("pipeline/internal_ml_ready.csv", low_memory=False)
    X_all = internal_all[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = internal_all["label"].astype(int)
    # reconstruct the exact same 80/20 split used everywhere else this session
    idx_train, idx_test = train_test_split(internal_all.index, test_size=0.2, stratify=y_all, random_state=42)
    internal_test = internal_all.loc[idx_test].reset_index(drop=True)

    ext = pd.read_csv("pipeline/external_ml_ready.csv", low_memory=False)

    rows = []

    def run(pool_name, sub_df, gene=None):
        sub = sub_df if gene is None else sub_df[sub_df["gene_symbol"] == gene]
        if len(sub) == 0 or sub["label"].nunique() < 2:
            return
        X = sub[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
        y = sub["label"].astype(int).values
        proba = predict_all(models, hybrid_b, X)
        for model_name, p in proba.items():
            m = high_spec_high_sens(y, p)
            rows.append({"pool": pool_name, "gene": gene or "both", "model": model_name,
                         "n": len(sub), "n_pathogenic": int(y.sum()), **m})

    print("=== High-specificity / high-sensitivity regime, by pool and gene ===")
    for pool_name, df in [("internal_test", internal_test), ("external", ext)]:
        run(pool_name, df, gene=None)
        run(pool_name, df, gene="BRCA1")
        run(pool_name, df, gene="BRCA2")

    print("\n=== Gene-AND-label-balanced subset check ===")
    for pool_name, df in [("internal_all", internal_all), ("external", ext)]:
        balanced, n_per_cell = gene_balance_downsample(df)
        print(f"{pool_name}: balanced subset n_per_(gene,label)_cell={n_per_cell}, total={len(balanced)}")
        run(f"{pool_name}_gene_balanced", balanced, gene=None)

    out = pd.DataFrame(rows)
    out.to_csv("pipeline/table_stage8_benchmarking.csv", index=False)
    print("\n=== Saved to pipeline/table_stage8_benchmarking.csv ===")
    print(out.to_string())


if __name__ == "__main__":
    main()
