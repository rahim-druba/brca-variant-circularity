"""
Stage 7: ACMG PP3/BP4 calibration, rebuilt + extended to gene-specific
(BRCA1 vs BRCA2), per Chen et al. 2026 ("PredictMD", articles/16 chen y.pdf).

Like the Hybrid-B script and the SMOTE sweep, the ORIGINAL calibration
script no longer exists anywhere in this repo -- only its results
(2/results/table_acmg_thresholds.csv, table_acmg_external_validation.csv)
survived, with a one-paragraph memory note (docs/memory/project_acmg_calibration.md)
describing the method: "local likelihood ratio LR+ = f_path(score)/f_benign(score)
via sliding window on leakage-free internal OOF predictions; mapped to OddsPath
points (Supporting 2.08, Moderate 4.33, Strong 18.7)" -- the standard ClinGen
SVI / Pejaver et al. 2022 framework. Reconstructed here as a class-conditional
Gaussian KDE density ratio (the direct, standard implementation of
"f_path(score)/f_benign(score) via sliding window").

Chen et al.'s finding (Fig. 6a-d, verified by reading the actual paper --
NOT taken from the earlier paraphrase alone) is that gene-specific
calibration measurably beats domain-aggregate/genome-wide calibration FOR
BRCA1 specifically (REVEL >=+3 points: OR=2.05 gene-specific vs 1.96
domain-aggregate; AlphaMissense >=+4: OR=6.50 vs 2.81 at >=+3), but for
BRCA2 "no consistent advantage for either calibration strategy was
observed." Chen's own exact BRCA1 score thresholds are published only via
an interactive portal (https://igvf.mavedb.org/, a client-side Nuxt SPA with
no accessible API found -- confirmed by inspecting the JS bundle directly,
not assumed) that couldn't be scraped with the tools available this
session, so rather than guess at numbers that can't be verified, this
script ADOPTS THE FINDING (calibrate BRCA1 separately; don't bother for
BRCA2) and re-derives the actual thresholds from our own labeled data using
our own already-established calibration methodology. This mirrors how the
Hybrid-B rebuild handled the same kind of gap.

Target score: the current canonical model's (Hybrid-B, XGB+MLP stack)
probability output, matching what the original calibration was built on
("score" in table_acmg_thresholds.csv), not individual predictor scores --
our project calibrates the FINAL model's output, unlike Chen et al. who
calibrate individual VEPs (REVEL, AlphaMissense) one at a time. Uses a
fresh 5-fold OOF loop over the FULL internal_ml_ready.csv (not just the
80% train split) so every internal variant gets a leakage-free probability,
maximizing usable calibration data -- especially important for the
per-gene splits, which have much smaller N than the pooled set.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
XGB_PARAMS = dict(n_estimators=200, max_depth=3, eval_metric="logloss", random_state=42)

ODDSPATH = {"Supporting": 2.08, "Moderate": 4.33, "Strong": 18.7, "Very_Strong": 350.0}


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


def full_dataset_oof_hybrid_b(df):
    """5-fold OOF Hybrid-B-style stacked probability for every row of df
    (not just an 80% train split) -- maximizes usable data for the
    per-gene calibration splits."""
    X = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int).values
    imputer = SimpleImputer(strategy="median").fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))
    X_arr = scaler.transform(imputer.transform(X))

    oof_xgb = np.zeros(len(df))
    oof_mlp = np.zeros(len(df))
    for fold, (fit_idx, hold_idx) in enumerate(CV.split(X_arr, y)):
        print(f"  fold {fold+1}/5")
        pos_weight = (y[fit_idx] == 0).sum() / (y[fit_idx] == 1).sum()
        sw = np.where(y[fit_idx] == 1, pos_weight, 1.0)
        xgb_fold = XGBClassifier(**XGB_PARAMS)
        xgb_fold.fit(X_arr[fit_idx], y[fit_idx], sample_weight=sw)
        oof_xgb[hold_idx] = xgb_fold.predict_proba(X_arr[hold_idx])[:, 1]

        mlp_fold = train_mlp(X_arr[fit_idx], y[fit_idx])
        oof_mlp[hold_idx] = mlp_predict_proba(mlp_fold, X_arr[hold_idx])

    meta = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta.fit(np.column_stack([oof_xgb, oof_mlp]), y)
    oof_stacked = meta.predict_proba(np.column_stack([oof_xgb, oof_mlp]))[:, 1]
    return oof_stacked


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lr_thresholds(scores, labels, grid=None):
    """Class-conditional Gaussian KDE density ratio f_path(s)/f_benign(s),
    the direct reading of the documented "local LR via sliding window"
    method. Works in LOGIT space, not raw [0,1] probability -- the Hybrid-B
    stacked probabilities pile up heavily near 0 and near 1 (median
    pathogenic score 0.998, benign max also 0.998 -- a saturated stacked
    classifier, not a smooth score), which breaks a raw-probability KDE
    (both classes' density spikes collapse onto nearly the same point,
    producing degenerate/near-zero thresholds). Logit-transforming spreads
    the saturated mass out into a well-behaved real-valued range; results
    are mapped back to probability space at the end via the sigmoid.

    Returns the lowest score threshold at which LR+ >= each OddsPath value
    (pathogenic side) and the highest at which LR+ <= 1/OddsPath (benign
    side), enforcing monotonicity by taking a running max/min so thresholds
    don't flicker in noisy low-density regions.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    path_logit = logit(scores[labels == 1])
    benign_logit = logit(scores[labels == 0])
    if len(path_logit) < 10 or len(benign_logit) < 10:
        return None, None  # too little data for a stable KDE, don't fabricate a number

    if grid is None:
        # restrict to the actual data's support (0.5th-99.5th percentile of the pooled
        # logit-scores) -- extending even 1 unit past the true min/max let the KDE
        # extrapolate into unsupported territory, where both densities are near-zero
        # and their ratio is numerically wild (billions-scale spikes, not real signal;
        # caught by inspecting the raw LR curve before trusting the result).
        pooled = np.concatenate([path_logit, benign_logit])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        grid = np.linspace(lo, hi, 2001)
    # shared, manually-fixed bandwidth (bw_method=0.3) for both classes -- scipy's default
    # per-class Scott's-rule bandwidth gave the pathogenic class (n=314) a much wider
    # bandwidth than benign (n=8295), and independently-tuned bandwidths for the two
    # densities made their ratio behave erratically; a shared bandwidth is standard
    # practice for LR-via-KDE and gives a much smoother, monotonic-in-trend curve.
    kde_path = gaussian_kde(path_logit, bw_method=0.3)
    kde_benign = gaussian_kde(benign_logit, bw_method=0.3)
    lr = kde_path(grid) / np.clip(kde_benign(grid), 1e-12, None)

    # enforce monotonic non-decreasing LR as score increases (running max from the left)
    lr_mono = np.maximum.accumulate(lr)

    path_thresh = {}
    for name, odds in ODDSPATH.items():
        idx = np.searchsorted(lr_mono, odds)
        path_thresh[name] = round(sigmoid(grid[idx]), 4) if idx < len(grid) else None

    lr_rev_mono = np.minimum.accumulate(lr[::-1])[::-1]  # running min from the right, for benign side
    benign_thresh = {}
    for name, odds in ODDSPATH.items():
        target = 1.0 / odds
        below = np.where(lr_rev_mono <= target)[0]
        idx = below.max() if len(below) else None
        benign_thresh[name] = round(sigmoid(grid[idx]), 4) if idx is not None else None

    return path_thresh, benign_thresh


def evaluate_split(name, scores, labels):
    print(f"\n=== {name}: n={len(scores)} ({labels.sum()} pathogenic, {(labels==0).sum()} benign) ===")
    path_thresh, benign_thresh = lr_thresholds(scores, labels)
    if path_thresh is None:
        print("  too little data for a stable KDE -- skipped")
        return None
    print(f"  PP3 (pathogenic) thresholds: {path_thresh}")
    print(f"  BP4 (benign) thresholds:     {benign_thresh}")
    return {"PP3": path_thresh, "BP4": benign_thresh}


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv", low_memory=False)
    import os
    cache = "pipeline/internal_hybrid_b_oof_cache.csv"
    if os.path.exists(cache):
        print(f"Reusing cached OOF probabilities from {cache}")
        df["hybrid_b_oof_proba"] = pd.read_csv(cache)["hybrid_b_oof_proba"].values
    else:
        print(f"Generating 5-fold OOF Hybrid-B probabilities for all {len(df)} internal variants...")
        df["hybrid_b_oof_proba"] = full_dataset_oof_hybrid_b(df)
        df[["gene_symbol", "label", "hybrid_b_oof_proba"]].to_csv(cache, index=False)

    print("\nOOF score distribution by label:")
    print(df.groupby("label")["hybrid_b_oof_proba"].describe())

    results = {}
    results["pooled_genome_wide"] = evaluate_split("Pooled (genome-wide, BRCA1+BRCA2)",
                                                     df["hybrid_b_oof_proba"], df["label"])
    for gene in ["BRCA1", "BRCA2"]:
        sub = df[df["gene_symbol"] == gene]
        results[f"gene_specific_{gene}"] = evaluate_split(f"Gene-specific: {gene}",
                                                            sub["hybrid_b_oof_proba"], sub["label"])

    rows = []
    for split_name, r in results.items():
        if r is None:
            continue
        for direction, threshes in r.items():
            for tier, val in threshes.items():
                rows.append({"split": split_name, "evidence": f"{direction}_{tier}", "score_threshold": val})
    out = pd.DataFrame(rows)
    out.to_csv("pipeline/table_acmg_calibration_gene_specific.csv", index=False)
    print("\n=== Saved to pipeline/table_acmg_calibration_gene_specific.csv ===")
    print(out)

    # --- external validation: does gene-specific calibration hold out of sample? ---
    print("\n=== External validation ===")
    import joblib
    ext = pd.read_csv("pipeline/external_ml_ready.csv", low_memory=False)
    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")

    X_ext_raw = ext[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    X_ext_arr = preproc["scaler"].transform(preproc["imputer"].transform(X_ext_raw))
    p_xgb = xgb_final.predict_proba(X_ext_raw)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_ext_arr)
    ext["hybrid_b_proba"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]

    val_rows = []
    for split_name, thresh in [("pooled_genome_wide", results["pooled_genome_wide"]),
                                ("gene_specific_BRCA1", results["gene_specific_BRCA1"]),
                                ("gene_specific_BRCA2", results["gene_specific_BRCA2"])]:
        if thresh is None:
            continue
        gene_filter = None if split_name == "pooled_genome_wide" else split_name.split("_")[-1]
        sub = ext if gene_filter is None else ext[ext["gene_symbol"] == gene_filter]
        strong = thresh["PP3"]["Strong"]
        bp4_strong = thresh["BP4"]["Strong"]
        if strong is not None:
            above = sub[sub["hybrid_b_proba"] >= strong]
            pct_path = above["label"].mean() * 100 if len(above) else float("nan")
            val_rows.append({"split": split_name, "check": "PP3_Strong bin", "n": len(above),
                              "pct_pathogenic": round(pct_path, 1)})
        if bp4_strong is not None:
            below = sub[sub["hybrid_b_proba"] <= bp4_strong]
            pct_path = below["label"].mean() * 100 if len(below) else float("nan")
            val_rows.append({"split": split_name, "check": "BP4_Strong bin", "n": len(below),
                              "pct_pathogenic": round(pct_path, 1)})
    val_out = pd.DataFrame(val_rows)
    val_out.to_csv("pipeline/table_acmg_calibration_external_validation.csv", index=False)
    print(val_out)


if __name__ == "__main__":
    main()
