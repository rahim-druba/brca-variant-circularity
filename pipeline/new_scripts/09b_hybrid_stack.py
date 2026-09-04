"""
Rebuild of "Hybrid-B" -- the project's canonical best model per
docs/memory/project_hybrid_mldl.md and CLAUDE.md's locked-in decision #2 --
whose generating script does not exist anywhere in this repo (only
2/results/table_hybrid_mldl.csv survived; same kind of gap as the missing
SMOTE-sweep script). 2026-07-25.

Architecture, per the surviving memory note (no other documentation of the
original build exists, so this is a faithful reconstruction of the described
architecture, not a byte-for-byt reproduction of the original run -- the
original's exact MLP hyperparameters/random seed were never recorded):

  - Base learner 1 (ML): XGBoost on the tabular score set. Reuses the
    GridSearchCV-tuned model_xgboost.pkl from 05_classical_ml.py directly --
    trained on the *exact* same train/test split (internal_ml_ready.csv,
    train_test_split(..., test_size=0.2, stratify=y, random_state=42)), so
    reusing it is not an approximation, it's the same model.
  - Base learner 2 (DL): a small feedforward MLP over the *same tabular
    score set* (NOT raw sequence -- that's what distinguishes this from the
    09_hybrid.py/BiLSTM hybrid, whose DL half is a near-coin-flip sequence
    model per CLAUDE.md's locked decision #5.  Trained in PyTorch with a
    class-weighted BCE loss (nn.BCEWithLogitsLoss(pos_weight=...)), matching
    the same class-weighting convention already used for the BiLSTM in
    09_hybrid.py and enforced project-wide since SMOTE was tested and
    rejected (docs/memory/project_smote_rejected.md) -- sklearn's
    MLPClassifier has no class_weight/sample_weight support, which is
    presumably why this needed a custom implementation in the first place.
  - Stacking: 5-fold leakage-free OOF. For each fold, XGB and the MLP are
    fit on the fold's training portion only and score the held-out portion;
    concatenating the 5 held-out predictions gives out-of-fold probabilities
    for every training-set row without any model ever scoring data it was
    fit on. A logistic-regression meta-learner (class_weight='balanced',
    matching StackingEnsemble's convention in 06_ensembles.py) is trained on
    [oof_xgb_proba, oof_mlp_proba] -> label.
  - Final test-set/external-set scoring uses base learners refit on the
    *full* training set (not fold-restricted), consistent with the OOF
    convention used in 09_hybrid.py's BiLSTM.

Feature set: the current 12-score set including BayesDel_noAF_score (Stage 4,
2026-07-25 -- see 04_filter_and_report.py's docstring for that decision).
This means these numbers are not comparable to the old table_hybrid_mldl.csv
even architecturally aside -- that table predates the canonical-transcript
fix, the ESM1b sign fix, and BayesDel entirely.

Saves to pipeline/table_hybrid_b.csv (its own file -- NOT
table_internal_comparison.csv, to avoid repeating the overwrite mistake made
earlier this session when 05/06/09 clobbered the prior 11-score run's table
before it was separately saved).
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, precision_score,
                              recall_score, f1_score, precision_recall_curve)
from xgboost import XGBClassifier

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# matches 05_classical_ml.py's GridSearchCV-selected XGBoost hyperparameters,
# so the OOF fold models and the final model are the same architecture.
XGB_PARAMS = dict(n_estimators=200, max_depth=3, eval_metric="logloss", random_state=42)


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, epochs=100, lr=1e-3, batch_size=64):
    model = MLP(X_train.shape[1]).to(device)
    pw = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def mlp_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(device))).cpu().numpy()


def metrics_row(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "n": len(y_true),
        "n_pathogenic": int(y_true.sum()),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if len(set(y_true)) > 1 else float("nan"),
        "auprc": round(average_precision_score(y_true, proba), 4) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
    }


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv")
    X_all = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = df["label"].astype(int)

    # identical split to 05_classical_ml.py -- lets us reuse its saved XGBoost model as-is
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} pathogenic)  Test: {len(X_test)} ({y_test.sum()} pathogenic)")

    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))

    def prep(X):
        return scaler.transform(imputer.transform(X))

    X_train_arr = prep(X_train)
    y_train_arr = y_train.values

    # --- 5-fold leakage-free OOF for both base learners ---
    print("Generating 5-fold OOF predictions (XGBoost + MLP)...")
    oof_xgb = np.zeros(len(X_train))
    oof_mlp = np.zeros(len(X_train))
    for fold, (fit_idx, hold_idx) in enumerate(CV.split(X_train_arr, y_train_arr)):
        print(f"  fold {fold+1}/5")
        pos_weight = (y_train_arr[fit_idx] == 0).sum() / (y_train_arr[fit_idx] == 1).sum()
        sw = np.where(y_train_arr[fit_idx] == 1, pos_weight, 1.0)
        xgb_fold = XGBClassifier(**XGB_PARAMS)
        xgb_fold.fit(X_train_arr[fit_idx], y_train_arr[fit_idx], sample_weight=sw)
        oof_xgb[hold_idx] = xgb_fold.predict_proba(X_train_arr[hold_idx])[:, 1]

        mlp_fold = train_mlp(X_train_arr[fit_idx], y_train_arr[fit_idx])
        oof_mlp[hold_idx] = mlp_predict_proba(mlp_fold, X_train_arr[hold_idx])

    # --- meta-learner on OOF predictions ---
    meta = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta.fit(np.column_stack([oof_xgb, oof_mlp]), y_train_arr)

    # F1-optimal threshold, chosen on the OOF (training-set) stacked predictions only --
    # never on the held-out test/external sets, to avoid tuning the threshold on eval data.
    oof_stacked_proba = meta.predict_proba(np.column_stack([oof_xgb, oof_mlp]))[:, 1]
    prec, rec, thresh = precision_recall_curve(y_train_arr, oof_stacked_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_threshold = thresh[np.argmax(f1s[:-1])]
    print(f"F1-optimal threshold (from OOF predictions): {best_threshold:.4f}")

    # --- final base learners, refit on the FULL training set ---
    print("Training final base learners on the full training set...")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")  # same split, GridSearchCV-tuned
    mlp_final = train_mlp(X_train_arr, y_train_arr)

    def stacked_proba(X_raw):
        X_arr = prep(X_raw)
        p_xgb = xgb_final.predict_proba(X_raw)[:, 1]  # xgb_final is a full sklearn Pipeline incl. its own imputer
        p_mlp = mlp_predict_proba(mlp_final, X_arr)
        return meta.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]

    ext = pd.read_csv("pipeline/external_ml_ready.csv")
    X_ext = ext[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_ext = ext["label"].astype(int).values

    test_proba = stacked_proba(X_test)
    ext_proba = stacked_proba(X_ext)

    results = {}
    results["internal_test@0.5"] = metrics_row(y_test.values, test_proba, threshold=0.5)
    results["external@0.5"] = metrics_row(y_ext, ext_proba, threshold=0.5)
    results[f"internal_test@{best_threshold:.3f}_f1opt"] = metrics_row(y_test.values, test_proba, threshold=best_threshold)
    results[f"external@{best_threshold:.3f}_f1opt"] = metrics_row(y_ext, ext_proba, threshold=best_threshold)
    for k, v in results.items():
        print(k, v)

    out = pd.DataFrame(results).T
    out.to_csv("pipeline/table_hybrid_b.csv")
    print("\n=== Hybrid-B (XGB+MLP stack) results (saved to pipeline/table_hybrid_b.csv) ===")
    print(out)

    import os
    os.makedirs("pipeline/models", exist_ok=True)
    torch.save(mlp_final.state_dict(), "pipeline/models/model_hybrid_b_mlp.pt")
    joblib.dump(meta, "pipeline/models/model_hybrid_b_meta.pkl")
    joblib.dump({"imputer": imputer, "scaler": scaler}, "pipeline/models/model_hybrid_b_preproc.pkl")


if __name__ == "__main__":
    main()
