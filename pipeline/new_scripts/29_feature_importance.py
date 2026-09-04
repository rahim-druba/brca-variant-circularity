"""
Fifth figure from the 2026-07-29 figures-planning list: feature importance,
compared across the three tree-based models that expose a native, per-feature
importance (RandomForest, GradientBoosting, XGBoost). Every closely-comparable
paper in articles/ that shows a feature-importance figure uses exactly this
kind of native gain/impurity importance (Hart et al. 2020's variable
importance list, Khandakji et al. 2022/2023's xgb.plot.importance) rather
than a game-theoretic explainer, so this is well precedented on its own.

Scope decision, disclosed rather than silent: SHAP was considered (Khandakji
2023 additionally reports Shapley values via xgb.plot.shap as a secondary
check) but the `shap` package is not installed in this project's venv, and
installing it would have pulled in a numpy downgrade (2.5.1 -> 2.4.6, via
numba's pinned upper bound) in the one shared environment every other frozen
script in this project also depends on. Given this project's own standing
discipline about not risking already-verified results, that trade was not
worth it for a plot that native feature importance already covers adequately
-- if SHAP is wanted later, it should go in its own isolated environment, not
this one.

StackingEnsemble and ML_VotingEnsemble are excluded here on purpose: they are
heterogeneous meta-combinations (soft voting / stacked meta-learner) with no
single well-defined per-original-feature importance the way a single tree
ensemble has -- forcing one would misrepresent what those models actually do.

Showing all three tree models together (rather than just one) is itself the
finding worth plotting: if the same handful of features rank highest
regardless of which algorithm is asked, that is stronger evidence the model
is leaning on real signal rather than one algorithm's idiosyncrasy.

Outputs:
  2/figures/fig_feature_importance.png
  pipeline/final_results_2026-07-26/table_feature_importance.csv
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "2/figures"
RESULTS = "pipeline/final_results_2026-07-26"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]

MODEL_FILES = {
    "RandomForest": "model_randomforest.pkl",
    "GradientBoosting": "model_gradientboosting.pkl",
    "XGBoost": "model_xgboost.pkl",
}
MODEL_COLORS = {"RandomForest": "#4C72B0", "GradientBoosting": "#55A868", "XGBoost": "#C44E52"}


def main():
    importances = {}
    for name, fname in MODEL_FILES.items():
        pipeline = joblib.load(f"pipeline/models/{fname}")
        clf = pipeline.steps[-1][1]
        importances[name] = clf.feature_importances_

    df = pd.DataFrame(importances, index=SCORE_COLS)
    df["mean_importance"] = df.mean(axis=1)
    df = df.sort_values("mean_importance", ascending=True)
    df.round(5).to_csv(f"{RESULTS}/table_feature_importance.csv", index_label="feature")

    fig, ax = plt.subplots(figsize=(9, 7))
    n_features = len(df)
    bar_h = 0.25
    y_base = np.arange(n_features)
    for i, model in enumerate(MODEL_FILES.keys()):
        offset = (i - 1) * bar_h
        ax.barh(y_base + offset, df[model].values, height=bar_h, color=MODEL_COLORS[model], label=model)

    ax.set_yticks(y_base)
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlabel("Feature importance (native gain / impurity, per model)")
    ax.set_title("Feature Importance Across Tree-Based Models", fontsize=13)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.sort_values("mean_importance", ascending=False).round(4).to_string())
    print(f"\nSaved {FIG_DIR}/fig_feature_importance.png")
    print(f"Saved {RESULTS}/table_feature_importance.csv")


if __name__ == "__main__":
    main()
