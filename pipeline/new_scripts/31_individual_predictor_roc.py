"""
Requested directly by the user, who supplied a screenshot from another
paper's "Fig 1. ROC curves of all models" -- a multi-line ROC overlay where
each line is one individual predictor score treated as a standalone
classifier (its raw score used directly in place of a predicted
probability), not a trained model. This is a genuinely different comparison
from 25_roc_pr_curves.py, which plots our 7 trained models -- this script
plots the 13 individual dbNSFP scores that make up this project's own locked
feature set (see pipeline/final_results_2026-07-26/MANIFEST.md's "Feature
set (locked)" section), answering "how good is any single input feature on
its own, before any model combines them."

The reference screenshot's paper used 17 scores, several of which (SIFT,
SIFT4G, LRT, fathmm-MKL, fathmm-XF, MutationTaster, DANN, PROVEAN, M-CAP) are
NOT part of this project's feature set -- they were considered and not
selected during this project's own Stage 4 feature-set decision. Plotting
only our own locked 13 here is deliberate, not an oversight: adding scores
this project doesn't actually use would blur the "why these 13" story
Section 3 of the report already tells, rather than support it.

Unlike the trained-model ROC curves, an individual score's own AUC can be
computed on however many variants that score is defined for -- the internal
pool here is complete-case (verified: n=8,609 for every one of the 13
scores, zero missingness), so there is no per-predictor denominator
difference to report, unlike some literature examples (e.g. Khandakji et
al.'s Table 1) where different predictors cover different variant counts.

Uses the SAME train/test split as every other frozen result in this project
(test_size=0.2, stratify=y, random_state=42) so the internal-test AUCs here
are on the identical 1,722 variants as every other internal-test figure.

Outputs:
  2/figures/fig_individual_predictor_roc.png
  pipeline/final_results_2026-07-26/table_individual_predictor_auc.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

FIG_DIR = "2/figures"
RESULTS = "pipeline/final_results_2026-07-26"
EVAL_LABELS = {"internal-test": "Internal test (n=1,722)", "external": "External ENIGMA (n=344)"}

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]

# A distinct, print-legible 13-color qualitative palette (not a colormap
# sample, which tends to produce adjacent lines that are hard to tell apart
# in a legend this long -- the exact problem the reference screenshot has).
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3",
    "#8C8C8C", "#CCB974", "#64B5CD", "#1A1A1A", "#2F6F62", "#9C6A2F",
]


def main():
    df = pd.read_csv("pipeline/internal_ml_ready.csv", low_memory=False)
    X_all = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = df["label"].astype(int)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    ext = pd.read_csv("pipeline/external_ml_ready.csv", low_memory=False)
    X_ext = ext[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    y_ext = ext["label"].astype(int)

    datasets = {"internal-test": (X_test, y_test), "external": (X_ext, y_ext)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    auc_rows = []
    for ax, eval_set in zip(axes, ["internal-test", "external"]):
        X_eval, y_eval = datasets[eval_set]
        # Sort legend by AUC descending, matching the reference screenshot's
        # convention of listing the best predictor first.
        results = []
        for score in SCORE_COLS:
            s = X_eval[score]
            mask = s.notna()
            auc = roc_auc_score(y_eval[mask], s[mask])
            fpr, tpr, _ = roc_curve(y_eval[mask], s[mask])
            results.append((score, auc, fpr, tpr, int(mask.sum())))
            auc_rows.append({"eval_set": eval_set, "score": score, "n": int(mask.sum()), "auc": round(auc, 4)})
        results.sort(key=lambda r: r[1], reverse=True)

        ax.plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1)
        for (score, auc, fpr, tpr, n), color in zip(results, COLORS):
            ax.plot(fpr, tpr, color=color, linewidth=1.3, label=f"{score} (AUC={auc:.2f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(EVAL_LABELS[eval_set], fontsize=11)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0)

    fig.suptitle("ROC Curves of Individual Predictor Scores (This Project's 13 Locked Features)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG_DIR}/fig_individual_predictor_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(f"{RESULTS}/table_individual_predictor_auc.csv", index=False)

    print(auc_df.pivot(index="score", columns="eval_set", values="auc").sort_values("internal-test", ascending=False))
    print(f"\nSaved {FIG_DIR}/fig_individual_predictor_roc.png")
    print(f"Saved {RESULTS}/table_individual_predictor_auc.csv")


if __name__ == "__main__":
    main()
