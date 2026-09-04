"""
Phase 8: external validation of every trained model against the fully
held-out ENIGMA-labeled external pool. Includes an explicit, code-level
overlap check between internal and external pools (must be zero by
construction, verified rather than assumed) and per-model metrics on
whatever subset of the external pool each model actually needs
(complete-case score subset for tabular/hybrid models, SNV subset
for the sequence models).

2026-07-25 (Stage 4): added BayesDel_noAF_score as a 12th score -- see
new_scripts/04_filter_and_report.py's docstring for the decision rationale.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
REGIONS = {13: ("pipeline/dl/chr13_region.fasta", 32314843), 17: ("pipeline/dl/chr17_region.fasta", 43039371)}
WINDOW = 100
BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}


class CNNModel(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.net(x)).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, in_channels=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        pooled = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(pooled).squeeze(-1)


def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_IDX.get(b)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def load_regions():
    seqs = {}
    for chrom, (path, start) in REGIONS.items():
        rec = next(SeqIO.parse(path, "fasta"))
        seqs[chrom] = (str(rec.seq).upper(), start)
    return seqs


def build_windows_alt(df, seqs):
    X = np.zeros((len(df), WINDOW * 2 + 1, 4), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        seq, region_start = seqs[int(row["chr"])]
        offset = int(row["pos"]) - region_start
        lo, hi = offset - WINDOW, offset + WINDOW + 1
        alt_window = seq[lo:offset] + row["alt"] + seq[offset + 1:hi]
        X[i] = one_hot(alt_window)
    return X


def build_windows_cnn(df, seqs):
    X = np.zeros((len(df), WINDOW * 2 + 1, 8), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        seq, region_start = seqs[int(row["chr"])]
        offset = int(row["pos"]) - region_start
        lo, hi = offset - WINDOW, offset + WINDOW + 1
        ref_window = seq[lo:hi]
        alt_window = ref_window[:WINDOW] + row["alt"] + ref_window[WINDOW + 1:]
        X[i] = np.concatenate([one_hot(ref_window), one_hot(alt_window)], axis=1)
    return X


def metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "n": len(y_true),
        "n_pathogenic": int(y_true.sum()),
        "test_roc_auc": round(roc_auc_score(y_true, proba), 4) if len(set(y_true)) > 1 else float("nan"),
        "test_accuracy": round(accuracy_score(y_true, pred), 4),
        "test_precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "test_recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "test_f1": round(f1_score(y_true, pred, zero_division=0), 4),
    }


def overlap_check():
    internal = pd.read_csv("pipeline/internal_pool.csv")[["chr", "pos", "ref", "alt"]]
    external = pd.read_csv("pipeline/external_pool.csv")[["chr", "pos", "ref", "alt"]]
    overlap = pd.merge(internal, external, on=["chr", "pos", "ref", "alt"], how="inner")
    print(f"Internal/external overlap by Chr/Start/Ref/Alt: {len(overlap)} variants (must be 0)")
    assert len(overlap) == 0, "External validation set is not disjoint from internal -- STOP."
    return len(overlap)


def main():
    n_overlap = overlap_check()
    seqs = load_regions()
    results = {}

    # --- Tabular models + ensembles (external ML-ready, complete-case) ---
    ext_ml = pd.read_csv("pipeline/external_ml_ready.csv")
    X_ext = ext_ml[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_ext = ext_ml["label"].astype(int).values

    tabular_models = {
        "RandomForest": "model_randomforest.pkl", "SVM": "model_svm.pkl",
        "GradientBoosting": "model_gradientboosting.pkl", "XGBoost": "model_xgboost.pkl",
        "ML_VotingEnsemble": "model_voting.pkl", "StackingEnsemble": "model_stacking.pkl",
    }
    for name, fname in tabular_models.items():
        model = joblib.load(f"pipeline/models/{fname}")
        proba = model.predict_proba(X_ext)[:, 1]
        results[name] = metrics(y_ext, proba)
        print(name, results[name])

    # --- Hybrid_XGB (external ML-ready + BiLSTM(hybrid) probability) ---
    bilstm_hybrid = BiLSTMModel().to(device)
    bilstm_hybrid.load_state_dict(torch.load("pipeline/models/model_hybrid_bilstm.pt", map_location=device))
    bilstm_hybrid.eval()
    X_win = build_windows_alt(ext_ml, seqs)
    with torch.no_grad():
        bilstm_proba = torch.sigmoid(bilstm_hybrid(torch.tensor(X_win).to(device))).cpu().numpy()
    X_hybrid = X_ext.copy()
    X_hybrid["BiLSTM_oof_proba"] = bilstm_proba
    hybrid_xgb = joblib.load("pipeline/models/model_hybrid_xgb.pkl")
    proba = hybrid_xgb.predict_proba(X_hybrid)[:, 1]
    results["Hybrid_XGB"] = metrics(y_ext, proba)
    print("Hybrid_XGB", results["Hybrid_XGB"])

    # --- CNN / BiLSTM standalone (external DL-ready, SNV-only) ---
    ext_dl = pd.read_csv("pipeline/external_dl_ready.csv")
    y_dl = ext_dl["label"].astype(int).values

    cnn = CNNModel().to(device)
    cnn.load_state_dict(torch.load("pipeline/models/model_cnn.pt", map_location=device))
    cnn.eval()
    X_cnn = build_windows_cnn(ext_dl, seqs)
    with torch.no_grad():
        cnn_proba = torch.sigmoid(cnn(torch.tensor(X_cnn).to(device))).cpu().numpy()
    results["1D-CNN"] = metrics(y_dl, cnn_proba)
    print("1D-CNN", results["1D-CNN"])

    bilstm = BiLSTMModel().to(device)
    bilstm.load_state_dict(torch.load("pipeline/models/model_bilstm.pt", map_location=device))
    bilstm.eval()
    X_bilstm = build_windows_alt(ext_dl, seqs)
    with torch.no_grad():
        bilstm_proba2 = torch.sigmoid(bilstm(torch.tensor(X_bilstm).to(device))).cpu().numpy()
    results["BiLSTM"] = metrics(y_dl, bilstm_proba2)
    print("BiLSTM", results["BiLSTM"])

    out = pd.DataFrame(results).T
    out.to_csv("pipeline/table_external_validation.csv")
    print("\n=== External validation summary ===")
    print(out)

    print("\n=== Internal vs external ROC-AUC (leakage sanity check) ===")
    internal_table = pd.read_csv("pipeline/table_internal_comparison.csv", index_col=0)
    dl_table = pd.read_csv("pipeline/table_dl_comparison.csv", index_col=0)
    comparison = []
    for name in out.index:
        int_auc = None
        if name in internal_table.index:
            int_auc = internal_table.loc[name, "test_roc_auc"]
        elif name in dl_table.index:
            int_auc = dl_table.loc[name, "val_roc_auc"]
        comparison.append({"model": name, "internal_roc_auc": int_auc, "external_roc_auc": out.loc[name, "test_roc_auc"]})
    comp_df = pd.DataFrame(comparison)
    comp_df["external_leq_internal"] = comp_df["external_roc_auc"] <= comp_df["internal_roc_auc"]
    comp_df.to_csv("pipeline/table_internal_vs_external.csv", index=False)
    print(comp_df)


if __name__ == "__main__":
    main()
