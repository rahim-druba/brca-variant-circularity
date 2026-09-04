"""
Last of the "must-have" figures from the 2026-07-29 figures-planning list:
the model architecture diagram. Closest precedent for this exact figure type
is Li C et al. 2023 (vERnet-B, articles/67 Li C 2023.pdf) Fig 1A -- a
box-and-arrow workflow from data source through training to the final
recognition step -- and Karalidou et al. 2022 (MARGINAL, articles/62
Karalidou.pdf) Fig 1, a two-tier serial-classifier flowchart. This diagram
follows the same convention: boxes for processing steps, a diamond for the
one real branch point (variant consequence), arrows for data flow.

Built entirely with matplotlib patches (FancyBboxPatch, a rotated-square
diamond, FancyArrowPatch) rather than graphviz/pygraphviz, deliberately --
no new dependency needed, and full manual control over layout for a diagram
this specific.

This is a diagram of the actual two-track design, not a generic template:
left branch is the rule-based loss-of-function track (PVS1, with the BRCA2
K3326X C-terminal exception -- see docs/memory/project_two_track_lof.md);
right branch is the missense ML/DL track, expanded to show Hybrid-B's real
internal structure (XGBoost + MLP base learners, 5-fold leakage-free
out-of-fold stacking, logistic-regression meta-learner) since Hybrid-B is
the model discussed most in the report's prose, while the other 6 candidate
architectures are shown as siblings feeding the same final call, matching
this project's own "no single best model" finding (Section 7) rather than
implying Hybrid-B is the only path.

Output: 2/figures/fig_architecture_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

FIG_DIR = "2/figures"

COLOR_INPUT = "#5B6864"
COLOR_LOF = "#3F7D5C"
COLOR_ML = "#4C72B0"
COLOR_HYBRID = "#1A1A1A"
COLOR_MERGE = "#8172B3"
COLOR_DECISION = "#9c6a2f"
TEXT_DARK = "#1A1A1A"


def box(ax, cx, cy, w, h, text, color, text_color="white", fontsize=9.5, lw=1.4, fontweight="normal"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.25,rounding_size=3",
                        linewidth=lw, edgecolor=color, facecolor=color if text_color == "white" else "none",
                        alpha=1.0)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=text_color,
             fontweight=fontweight, wrap=True)


def outline_box(ax, cx, cy, w, h, text, edgecolor, fontsize=9.5, lw=1.8):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.25,rounding_size=3",
                        linewidth=lw, edgecolor=edgecolor, facecolor="white")
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=edgecolor, fontweight="bold")


def diamond(ax, cx, cy, w, h, text, color):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.6, edgecolor=color, facecolor="white"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold")


def arrow(ax, start, end, color="#333333", style="-", lw=1.3):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                         linewidth=lw, color=color, linestyle=style, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 16))
    ax.set_xlim(-5, 103)
    ax.set_ylim(-8, 142)
    ax.axis("off")

    # --- Row 1: input ---
    box(ax, 50, 134, 56, 9, "BRCA1 / BRCA2 variant\n(ClinVar / BRCA Exchange / ENIGMA / Kazakh cohorts)",
        COLOR_INPUT, fontsize=9.5)

    # --- Row 2: annotation ---
    arrow(ax, (50, 129.5), (50, 122.5))
    box(ax, 50, 118, 74, 9, "Annotation: ANNOVAR + local dbNSFP 5.3.1a (MANE-Select transcript)\n"
                            "+ ESM1b (separately computed, sign-corrected)", COLOR_INPUT, fontsize=9)

    arrow(ax, (50, 113.5), (50, 106.5))
    box(ax, 50, 102, 84, 9,
        "13-feature vector: ClinPred, MetaRNN, REVEL, MetaSVM, MetaLR, VEST4, AlphaMissense,\n"
        "PrimateAI, CADD_raw, CADD_phred, ESM1b, BayesDel_noAF, Eigen_raw_coding",
        COLOR_INPUT, fontsize=8.5)

    # --- Decision ---
    arrow(ax, (50, 97.5), (50, 91.5))
    diamond(ax, 50, 87, 26, 12, "Variant\nconsequence?", COLOR_DECISION)

    # --- Left branch: LOF track ---
    arrow(ax, (41, 82.5), (22, 70.5), color=COLOR_LOF)
    box(ax, 20, 66, 34, 9, "Truncating\n(frameshift, nonsense, splice)", COLOR_LOF, fontsize=9)

    arrow(ax, (20, 61.5), (20, 54.5), color=COLOR_LOF)
    box(ax, 20, 48, 36, 12, "Rule-based track\nPVS1 criterion\n(BRCA2 K3326X C-terminal exception)",
        COLOR_LOF, fontsize=8.7)

    arrow(ax, (20, 42), (20, 12), color=COLOR_LOF)
    box(ax, 20, 8, 26, 8, "Pathogenic\n(rule-based call)", COLOR_LOF, fontsize=9)

    # --- Right branch: ML/DL track ---
    arrow(ax, (59, 82.5), (74, 70.5), color=COLOR_ML)
    box(ax, 74, 66, 20, 9, "Missense", COLOR_ML, fontsize=9.5)

    arrow(ax, (74, 61.5), (74, 54.5), color=COLOR_ML)
    box(ax, 74, 48, 44, 12,
        "ML / DL track\n7 candidate architectures: RF · SVM · GradientBoosting ·\n"
        "XGBoost · Voting · Stacking · Hybrid-B",
        COLOR_ML, fontsize=8.5)

    # Hybrid-B expanded (the model discussed most in the report's prose)
    arrow(ax, (74, 42), (74, 35), color=COLOR_HYBRID)
    outline_box(ax, 74, 30, 30, 7, "Hybrid-B (expanded below)", COLOR_HYBRID, fontsize=8.5, lw=1.6)

    arrow(ax, (67, 26.5), (61, 20.5), color=COLOR_HYBRID)
    arrow(ax, (81, 26.5), (87, 20.5), color=COLOR_HYBRID)
    box(ax, 60, 16, 20, 8, "XGBoost\n(base learner)", COLOR_HYBRID, fontsize=8.5)
    box(ax, 88, 16, 20, 8, "MLP\n(base learner)", COLOR_HYBRID, fontsize=8.5)

    arrow(ax, (62, 11.5), (72, 4.5), color=COLOR_HYBRID)
    arrow(ax, (86, 11.5), (76, 4.5), color=COLOR_HYBRID)
    box(ax, 74, 0, 40, 8, "5-fold leakage-free out-of-fold stacking\n→ logistic-regression meta-learner",
        COLOR_HYBRID, fontsize=8.5)

    # Other 6 architectures also feed a final call -- routed as a curved
    # connector that stays clear of the Hybrid-B expansion (x=59-89) by
    # exiting from the left edge of the ML/DL track box, rather than cutting
    # straight through it as an earlier version did.
    other_models = FancyArrowPatch((53, 43), (46, -8.5), connectionstyle="arc3,rad=-0.25",
                                    arrowstyle="-|>", mutation_scale=14, linewidth=1.0,
                                    linestyle="--", color="#9aa8a3")
    ax.add_patch(other_models)
    ax.text(41, 20, "other 6\narchitectures", fontsize=7.5, color="#7a8884", ha="center", style="italic")

    # --- Merge point ---
    arrow(ax, (20, 3.5), (42, -8), color=COLOR_LOF)
    arrow(ax, (74, -4.5), (58, -8), color=COLOR_HYBRID)
    box(ax, 50, -12.5, 66, 9, "Final pathogenicity call / probability\n(evaluated per model, no single model assumed best)",
        COLOR_MERGE, fontsize=9)

    ax.set_ylim(-18, 142)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_architecture_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR}/fig_architecture_diagram.png")


if __name__ == "__main__":
    main()
