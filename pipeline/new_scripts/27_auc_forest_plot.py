"""
Third figure from the 2026-07-29 figures-planning list: a forest plot of AUC
with 95% CI per model, both eval sets -- the visual counterpart to
table_0_HEADLINE_SUMMARY.csv, which is itself the file this project's own
MANIFEST.md names as "the single table to show if asked what's the final
result." A table of overlapping CIs is easy to misread as "there must be a
best one somewhere in there"; a forest plot with a reference line at the top
model's point estimate makes the actual finding -- that most CIs cross that
line, i.e. are not statistically distinguishable from the top performer --
visible at a glance rather than requiring the reader to parse p-values row by
row.

Reads table_0_HEADLINE_SUMMARY.csv directly (not table_model_cis.csv +
table_pairwise_significance.csv separately) specifically so this figure is
guaranteed to agree with the already-published headline table -- there is
exactly one source of truth for "which model is the top point estimate on
which eval set," and this script does not re-derive it.

Output (2/figures/): fig_auc_forest_plot.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "2/figures"
RESULTS = "pipeline/final_results_2026-07-26"
EVAL_LABELS = {"internal-test": "Internal test (n=1,722)", "external": "External ENIGMA (n=344)"}

COLOR_TOP = "#1A1A1A"
COLOR_TIED = "#2f6f62"
COLOR_DIFFERS = "#9c6a2f"


def parse_ci(ci_str):
    lo, hi = ci_str.strip("[]").split(",")
    return float(lo), float(hi)


def status_and_color(vs_top):
    if vs_top == "is the top point estimate":
        return "top", COLOR_TOP
    if "not significant" in vs_top:
        return "tied with top", COLOR_TIED
    return "differs from top", COLOR_DIFFERS


def main():
    df = pd.read_csv(f"{RESULTS}/table_0_HEADLINE_SUMMARY.csv")
    df[["ci_lo", "ci_hi"]] = df["roc_auc_CI"].apply(lambda s: pd.Series(parse_ci(s)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=False)
    for ax, eval_set in zip(axes, ["internal-test", "external"]):
        sub = df[df["eval_set"] == eval_set].sort_values("roc_auc", ascending=True).reset_index(drop=True)
        top_auc = df[(df["eval_set"] == eval_set) & (df["vs_top_model"] == "is the top point estimate")]["roc_auc"].iloc[0]

        y_pos = range(len(sub))
        for y, row in zip(y_pos, sub.itertuples()):
            status, color = status_and_color(row.vs_top_model)
            ax.plot([row.ci_lo, row.ci_hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            marker = "D" if status == "top" else "o"
            ax.plot(row.roc_auc, y, marker=marker, color=color, markersize=7, zorder=3)

        ax.axvline(top_auc, color=COLOR_TOP, linestyle="--", linewidth=1, alpha=0.6, zorder=1)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["model"], fontsize=10)
        ax.set_xlabel("ROC AUC (point estimate, 95% bootstrap CI)")
        ax.set_title(EVAL_LABELS[eval_set], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker="D", color=COLOR_TOP, linestyle="", markersize=7, label="Top point estimate"),
        plt.Line2D([0], [0], marker="o", color=COLOR_TIED, linestyle="", markersize=7,
                   label="Not significantly different from top"),
        plt.Line2D([0], [0], marker="o", color=COLOR_DIFFERS, linestyle="", markersize=7,
                   label="Significantly different from top (p<0.05)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Model AUC with 95% Confidence Intervals, vs. Top Performer per Eval Set", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(f"{FIG_DIR}/fig_auc_forest_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIG_DIR}/fig_auc_forest_plot.png")


if __name__ == "__main__":
    main()
