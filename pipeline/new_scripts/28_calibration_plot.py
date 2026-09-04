"""
Fourth figure from the 2026-07-29 figures-planning list: calibration
(reliability) curves, all 7 models, both eval sets, plus the Brier score
that goes with each curve. This one matters more than a routine "extra"
metric plot for this specific project: CLAUDE.md's locked-in decision #4
states the ACMG PP3/BP4 calibration work depends on model probabilities
carrying real clinical meaning, not just ranking variants correctly, which
is exactly what a reliability diagram checks and an AUC/PR curve cannot.
Molotkov et al. 2023 (SNPred, articles/64 Molotkov.pdf) makes the same
argument explicitly and reports Brier scores across predictors for this
reason -- direct literature precedent for including this figure.

No existing frozen table has Brier scores for these 7 models on these 2 eval
sets (table_smote_comparison.csv has Brier scores but only for the 4 SMOTE-
sweep base models under different oversampling ratios, a different
question), so this script computes them fresh from
predictions_per_variant.csv and writes them alongside the figure.

Layout follows sklearn's own CalibrationDisplay gallery convention: a
reliability curve on top (mean predicted probability vs. observed fraction
of positives, diagonal = perfect calibration) with a histogram of predicted
probabilities underneath, so a reader can see not just how calibrated a
model is but how many predictions actually fall in the region where that
matters.

Outputs:
  2/figures/fig_calibration_curves.png
  pipeline/final_results_2026-07-26/table_calibration_metrics.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

FIG_DIR = "2/figures"
RESULTS = "pipeline/final_results_2026-07-26"
PRED_PATH = f"{RESULTS}/predictions_per_variant.csv"
EVAL_LABELS = {"internal-test": "Internal test (n=1,722)", "external": "External ENIGMA (n=344)"}
N_BINS = 10

MODEL_COLORS = {
    "RandomForest":      "#4C72B0",
    "SVM":               "#DD8452",
    "GradientBoosting":  "#55A868",
    "XGBoost":           "#C44E52",
    "ML_VotingEnsemble": "#8172B3",
    "StackingEnsemble":  "#937860",
    "Hybrid-B":          "#1A1A1A",
}


def main():
    df = pd.read_csv(PRED_PATH)
    model_cols = [c for c in df.columns if c.startswith("proba_")]
    models = [c.replace("proba_", "") for c in model_cols]
    eval_sets = ["internal-test", "external"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
    brier_rows = []

    for col, eval_set in enumerate(eval_sets):
        df_eval = df[df["eval_set"] == eval_set]
        y_true = df_eval["y_true"].values
        ax_curve, ax_hist = axes[0, col], axes[1, col]

        ax_curve.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfectly calibrated")
        for model in models:
            proba = df_eval[f"proba_{model}"].values
            brier = brier_score_loss(y_true, proba)
            brier_rows.append({"eval_set": eval_set, "model": model, "brier_score": round(brier, 4)})
            prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=N_BINS, strategy="uniform")
            lw = 2.4 if model == "Hybrid-B" else 1.4
            ax_curve.plot(prob_pred, prob_true, marker="o", markersize=4, linewidth=lw,
                          color=MODEL_COLORS[model], label=f"{model} (Brier={brier:.3f})")
            ax_hist.hist(proba, bins=np.linspace(0, 1, 21), histtype="step", linewidth=lw,
                         color=MODEL_COLORS[model])

        ax_curve.set_xlabel("Mean predicted probability")
        ax_curve.set_ylabel("Observed fraction pathogenic")
        ax_curve.set_title(EVAL_LABELS[eval_set], fontsize=11)
        ax_curve.set_xlim(-0.02, 1.02)
        ax_curve.set_ylim(-0.02, 1.02)
        ax_curve.spines[["top", "right"]].set_visible(False)

        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_yscale("log")
        ax_hist.set_xlim(-0.02, 1.02)
        ax_hist.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)
    fig.suptitle("Calibration (Reliability) Curves, All Models", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(f"{FIG_DIR}/fig_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(brier_rows).to_csv(f"{RESULTS}/table_calibration_metrics.csv", index=False)

    print("Brier scores (lower = better calibrated):")
    print(pd.DataFrame(brier_rows).pivot(index="model", columns="eval_set", values="brier_score"))
    print(f"\nSaved {FIG_DIR}/fig_calibration_curves.png")
    print(f"Saved {RESULTS}/table_calibration_metrics.csv")


if __name__ == "__main__":
    main()
