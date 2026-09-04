"""
First of the "must-have" figures identified in the 2026-07-29 figures-planning
pass: ROC and precision-recall curves, multi-model overlay, for the internal
test set and the external ENIGMA set. Precedent for this exact figure style:
Hart et al. 2020 (BRCA-ML, articles/56 Hart.pdf) Fig 1 -- ROC top, PR bottom,
one column per gene there, one column per eval set here; Dong et al. 2015
(articles/55 Dong.pdf) Fig 2 does the same pairing. Every value plotted here
comes from pipeline/final_results_2026-07-26/predictions_per_variant.csv,
itself verified (24_generate_predictions.py) to reproduce table_model_cis.csv
exactly -- so the AUC printed in each legend matches the number already
reported in the technical report, not a re-derived one.

Outputs (all to 2/figures/):
  fig_roc_pr_curves.png   -- the combined 2x2 figure (ROC/PR x internal/external)
  fig_roc_internal.png / fig_roc_external.png / fig_pr_internal.png / fig_pr_external.png
                          -- the same four panels saved individually, in case a
                             journal wants single-panel figures instead of a grid
  roc_pr_curve_data.csv  -- the raw fpr/tpr/precision/recall points behind every
                             curve, long format, for a supplementary data file
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

FIG_DIR = "2/figures"
PRED_PATH = "pipeline/final_results_2026-07-26/predictions_per_variant.csv"

MODEL_COLORS = {
    "RandomForest":      "#4C72B0",
    "SVM":               "#DD8452",
    "GradientBoosting":  "#55A868",
    "XGBoost":           "#C44E52",
    "ML_VotingEnsemble": "#8172B3",
    "StackingEnsemble":  "#937860",
    "Hybrid-B":          "#1A1A1A",
}
EVAL_LABELS = {"internal-test": "Internal test (n=1,722)", "external": "External ENIGMA (n=344)"}


def load_predictions():
    df = pd.read_csv(PRED_PATH)
    model_cols = [c for c in df.columns if c.startswith("proba_")]
    models = [c.replace("proba_", "") for c in model_cols]
    return df, models


def plot_roc_panel(ax, df_eval, models, title):
    for model in models:
        y_true = df_eval["y_true"].values
        proba = df_eval[f"proba_{model}"].values
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        lw = 2.4 if model == "Hybrid-B" else 1.4
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})", color=MODEL_COLORS[model], linewidth=lw)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.spines[["top", "right"]].set_visible(False)


def plot_pr_panel(ax, df_eval, models, title):
    prevalence = df_eval["y_true"].mean()
    for model in models:
        y_true = df_eval["y_true"].values
        proba = df_eval[f"proba_{model}"].values
        precision, recall, _ = precision_recall_curve(y_true, proba)
        auprc = average_precision_score(y_true, proba)
        lw = 2.4 if model == "Hybrid-B" else 1.4
        ax.plot(recall, precision, label=f"{model} (AUPRC={auprc:.3f})", color=MODEL_COLORS[model], linewidth=lw)
    ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1, label=f"Baseline (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.spines[["top", "right"]].set_visible(False)


def save_curve_data(df, models):
    rows = []
    for eval_set, df_eval in df.groupby("eval_set"):
        y_true = df_eval["y_true"].values
        for model in models:
            proba = df_eval[f"proba_{model}"].values
            fpr, tpr, roc_thr = roc_curve(y_true, proba)
            for f, t, thr in zip(fpr, tpr, roc_thr):
                rows.append({"eval_set": eval_set, "model": model, "curve": "ROC",
                             "x_fpr_or_recall": f, "y_tpr_or_precision": t, "threshold": thr})
            precision, recall, pr_thr = precision_recall_curve(y_true, proba)
            pr_thr = list(pr_thr) + [None]  # precision_recall_curve returns one fewer threshold
            for r, p, thr in zip(recall, precision, pr_thr):
                rows.append({"eval_set": eval_set, "model": model, "curve": "PR",
                             "x_fpr_or_recall": r, "y_tpr_or_precision": p, "threshold": thr})
    pd.DataFrame(rows).to_csv(f"{FIG_DIR}/roc_pr_curve_data.csv", index=False)


def main():
    df, models = load_predictions()
    eval_sets = ["internal-test", "external"]

    # --- combined 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, eval_set in enumerate(eval_sets):
        df_eval = df[df["eval_set"] == eval_set]
        plot_roc_panel(axes[0, col], df_eval, models, f"ROC - {EVAL_LABELS[eval_set]}")
        plot_pr_panel(axes[1, col], df_eval, models, f"Precision-Recall - {EVAL_LABELS[eval_set]}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle("ROC and Precision-Recall Curves, All Models", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(f"{FIG_DIR}/fig_roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- individual panels ---
    panel_specs = [
        ("fig_roc_internal.png", plot_roc_panel, "internal-test", f"ROC - {EVAL_LABELS['internal-test']}"),
        ("fig_roc_external.png", plot_roc_panel, "external", f"ROC - {EVAL_LABELS['external']}"),
        ("fig_pr_internal.png", plot_pr_panel, "internal-test", f"Precision-Recall - {EVAL_LABELS['internal-test']}"),
        ("fig_pr_external.png", plot_pr_panel, "external", f"Precision-Recall - {EVAL_LABELS['external']}"),
    ]
    for fname, plot_fn, eval_set, title in panel_specs:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plot_fn(ax, df[df["eval_set"] == eval_set], models, title)
        # Legend placed outside the axes (not "best"/corner placement) so it never
        # overlaps the plotted curves -- an earlier version used loc="lower left"
        # on the PR panels and the legend text landed directly on top of the
        # baseline line and curve, which made several entries unreadable.
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0)
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/{fname}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    save_curve_data(df, models)

    print("Saved:")
    for f in ["fig_roc_pr_curves.png", "fig_roc_internal.png", "fig_roc_external.png",
              "fig_pr_internal.png", "fig_pr_external.png", "roc_pr_curve_data.csv"]:
        print(f"  {FIG_DIR}/{f}")


if __name__ == "__main__":
    main()
