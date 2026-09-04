"""
Second figure from the 2026-07-29 figures-planning list: confusion matrices
for every model, both eval sets. Deliberately built as a small-multiples grid
covering all 7 models rather than picking one "canonical" model, because this
project's own finding (table_pairwise_significance.csv, Section 7 of the
report) is that no single model is statistically distinguishable from the
others -- a confusion-matrix figure that only showed Hybrid-B would quietly
contradict that claim by implying it's "the" model to look at.

Threshold: 0.5 throughout, matching the f1@0.5 column already frozen in
table_model_cis.csv, so the accuracy/sensitivity/specificity/F1 numbers
printed under each matrix here are the same numbers already reported
elsewhere, not a new threshold choice invented for this figure.

Outputs (all to 2/figures/):
  fig_confusion_matrices_internal.png  -- 7-model grid, internal test set
  fig_confusion_matrices_external.png  -- 7-model grid, external ENIGMA set
  fig_confusion_matrix_hybridb_internal.png / _external.png
                                        -- Hybrid-B alone, single panel
                                           (the model discussed most in the
                                           report's prose -- DMS validation,
                                           Kazakh evaluation -- so worth
                                           having as a standalone figure too)
  table_confusion_matrix_metrics.csv   -- raw TP/FP/TN/FN + derived metrics
                                           for every model x eval set, to
                                           pipeline/final_results_2026-07-26/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FIG_DIR = "2/figures"
RESULTS = "pipeline/final_results_2026-07-26"
PRED_PATH = f"{RESULTS}/predictions_per_variant.csv"
THRESHOLD = 0.5
EVAL_LABELS = {"internal-test": "Internal test (n=1,722)", "external": "External ENIGMA (n=344)"}
CLASS_LABELS = ["Benign", "Pathogenic"]


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else float("nan")
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, precision=precision, f1=f1)


def draw_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_LABELS, fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    m = compute_metrics(y_true, y_pred)
    ax.set_title(f"{title}\nacc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}",
                 fontsize=8.5)


def main():
    df = pd.read_csv(PRED_PATH)
    model_cols = [c for c in df.columns if c.startswith("proba_")]
    models = [c.replace("proba_", "") for c in model_cols]

    metric_rows = []
    for eval_set in ["internal-test", "external"]:
        df_eval = df[df["eval_set"] == eval_set]
        y_true = df_eval["y_true"].values

        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        for i, model in enumerate(models):
            proba = df_eval[f"proba_{model}"].values
            y_pred = (proba >= THRESHOLD).astype(int)
            draw_matrix(axes[i], y_true, y_pred, model)
            m = compute_metrics(y_true, y_pred)
            m.update({"eval_set": eval_set, "model": model, "threshold": THRESHOLD})
            metric_rows.append(m)
        for j in range(len(models), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Confusion Matrices, Threshold=0.5 - {EVAL_LABELS[eval_set]}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        suffix = "internal" if eval_set == "internal-test" else "external"
        fig.savefig(f"{FIG_DIR}/fig_confusion_matrices_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Hybrid-B standalone
        proba = df_eval["proba_Hybrid-B"].values
        y_pred = (proba >= THRESHOLD).astype(int)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        draw_matrix(ax, y_true, y_pred, "Hybrid-B")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig_confusion_matrix_hybridb_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)[
        ["eval_set", "model", "threshold", "TN", "FP", "FN", "TP", "accuracy", "sensitivity", "specificity",
         "precision", "f1"]
    ]
    metrics_df.round(4).to_csv(f"{RESULTS}/table_confusion_matrix_metrics.csv", index=False)

    print("Saved:")
    for f in ["fig_confusion_matrices_internal.png", "fig_confusion_matrices_external.png",
              "fig_confusion_matrix_hybridb_internal.png", "fig_confusion_matrix_hybridb_external.png"]:
        print(f"  {FIG_DIR}/{f}")
    print(f"  {RESULTS}/table_confusion_matrix_metrics.csv")


if __name__ == "__main__":
    main()
