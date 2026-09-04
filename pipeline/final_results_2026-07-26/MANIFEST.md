# Frozen final results - 2026-07-26

This directory is the single, versioned source of truth for every reported
number in this project as of this date. If a number in a write-up doesn't
trace back to a file here, don't cite it without regenerating it first.

**Fixes** blocker #2 from the publishability assessment (2026-07-26): "no
frozen, single final run - a paper needs one locked, versioned run that
every reported number traces back to."

## Feature set (locked)

13 scores, all sourced from local dbNSFP 5.3.1a (MANE-Select canonical
transcript resolution) except ESM1b (separately computed,
`pipeline/esm1b/esm1b_scores.csv`, sign-corrected):

```
ClinPred_score, MetaRNN_score, REVEL_score, MetaSVM_score, MetaLR_score,
VEST4_score, AlphaMissense_score, PrimateAI_score, CADD_raw, CADD_phred,
ESM1b_score, BayesDel_noAF_score, Eigen_raw_coding
```

Decision record for this exact feature set: `pipeline/new_scripts/
04_filter_and_report.py`'s docstring (Stage 4, closed 2026-07-26).
Superseded/earlier feature-set versions (11-score, 12-score-BayesDel-only)
remain in `pipeline/` under their `*_12score_bayesdel.csv`/pre-Eigen
filenames for audit-trail purposes - do not cite those, they predate the
locked feature set.

## Data sources (locked)

- Internal pool: ClinVar-derived, BRCA Exchange release `release-11-11-25`
  snapshot. `pipeline/internal_ml_ready.csv` (8,609 variants, complete-case).
- External pool: ENIGMA-derived, same BRCA Exchange release.
  `pipeline/external_ml_ready.csv` (344 variants, complete-case).
- Train/test split: `train_test_split(test_size=0.2, stratify=y,
  random_state=42)` on `internal_ml_ready.csv` - identical across every
  script in this directory (6,887 train / 1,722 test, 251/63 pathogenic).

## Files, in generation order, with the script and one-line result each

| # | File | Generating script | Result summary |
|---|---|---|---|
| 1 | `missingness_report.csv` | `04_filter_and_report.py` | Per-feature missingness, both pools |
| 2 | `table_internal_comparison.csv` | `05_classical_ml.py` + `06_ensembles.py` | RF/SVM/GB/XGBoost + voting/stacking ensembles, internal test AUC 0.970-0.987 |
| 3 | `table_hybrid_b.csv` | `09b_hybrid_stack.py` | XGBoost+MLP stack (canonical "Hybrid-B"); internal AUC 0.9745, external AUC 0.9986 (F1-opt threshold 0.929) |
| 4 | `table_external_validation.csv`, `table_internal_vs_external.csv` | `10_external_validation.py` | All tabular models + BiLSTM-hybrid on external pool; standalone 1D-CNN/BiLSTM AUC 0.53/0.54 (near coin-flip) |
| 5 | `table_dl_comparison.csv` | `08_dl_models.py` (old_scripts, unchanged - no SCORE_COLS dependency) | Standalone CNN/BiLSTM internal validation |
| 6 | `table_circularity_check.csv` | `11_circularity_check.py` | Robust-6 subset (no clinical-label training) internal AUC 0.978 slightly *exceeds* full-13 (0.975) |
| 7 | `table_eigen_primateai_ablation.csv` | `11b_eigen_primateai_ablation.py` | Stage-4 decision support: Eigen added (F1 lift), EigenPC/PrimateAI-removal rejected |
| 8 | `table_training_era_overlap.csv` | `12_training_era_overlap.py` | External pool 71-98% pre-dates predictor training cutoffs (worse than internal's 20-65%) |
| 9 | `table_acmg_calibration_gene_specific.csv`, `table_acmg_calibration_external_validation.csv` | `13_acmg_calibration.py` | Gene-specific (BRCA1 vs BRCA2) PP3/BP4 thresholds; external PP3_Strong 98-100% pathogenic |
| 10 | `table_stage8_benchmarking.csv` | `14_stage8_benchmarking.py` | BRCA2 needs higher FPR than BRCA1 for 95% sensitivity; gene-balanced check holds |
| 11 | `table_brca1_dms_validation.csv` | `15_brca1_dms_validation.py` | Hybrid-B vs Findlay 2018 SGE: Spearman rho=-0.75 (p=1.2e-69), independent of ClinVar |
| 12 | `kazakh_brca_af_table.csv` | `16_kazakh_af_aggregation.py` | 4,160 BRCA1/2 positions, Kazakh population AF (224 samples, hg19->hg38 lifted) |
| 13 | `kazakh_pm2_bs1_evidence.csv` | `17_kazakh_pm2_bs1.py` | CI-aware PM2 (Supporting)/BS1 evidence codes from the AF table |
| 14 | `kazakh_pm2_bs1_case_study.csv` | `18_kazakh_pm2_bs1_case_study.py` | Validated against 25 real published Kazakh clinical variants |
| 15 | `kazakh_variants_final_annotated.csv` | `19_kazakh_variant_evaluation.py` | Full two-track eval: LOF rule 33/33 correct, Hybrid-B AUC 1.0 on 6 labeled missense (descriptive, n small) |
| 16 | `table_smote_comparison.csv` | `20_smote_sweep.py` | SMOTE 1:1 worst by Brier on all 4 base models; class-weighting confirmed as the right default |
| 17 | `table_model_cis.csv`, `table_pairwise_significance.csv` | `21_statistical_significance.py` | Bootstrap CIs + paired-bootstrap pairwise AUC tests: **0/21 model pairs significant on external set; Hybrid-B not distinguishable from any model on either set** |
| 18 | `table_brca1_dms_full_validation.csv` | `22_brca1_dms_full_validation.py` | Full ~3,893-variant Findlay SGE set: 2,361 scored (1,717 genuinely novel VUS), Hybrid-B vs functional score rho=-0.58 overall / -0.50 novel-only, both p<1e-100 - the clean external estimate (item #4 resolved) |

## table_benchmark_comparison_literature.csv - updated 2026-07-29

Expanded from 4 source papers to 15, after the user downloaded a further batch
of BRCA1/2-specific classifier papers (`articles/55`-`69 *.pdf`) and asked for
them to be researched and folded into this table. New rows added: Hart 2020
(BRCA-ML, gene-specific MCC on functional-assay ground truth - the most
directly comparable paper found), Khandakji 2022/2023 (gene-specific XGBoost,
ENIGMA + independent functional VUS accuracy), Kang 2023 (AUPRC-based
gene/disease/genome-wide comparison - independently replicates our own
"several models statistically tied, no clear winner" finding), Aljarf 2022,
Karalidou 2022 (MARGINAL), Molotkov 2023/SNPred (AUC PR on the *same* Findlay
2018 BRCA1-DMS set used for our own functional check - the single most
apples-to-apples number in the whole table), Kondrashova 2026 (orthogonal
tumor-genomic-profiling evidence, not sequence-based - flagged as
complementary rather than competing), Li C 2023/vERnet-B (structure-based),
Hidayat 2023 and KhajePasha 2018 (weaker comparators, included for
completeness). Full per-paper detail in `articles/citation_log.md` Batch 7.

## predictions_per_variant.csv - added 2026-07-29

Per-variant predicted probability for all 7 tabular/Hybrid-B models, both
eval sets (1,722 internal-test + 344 external rows), generated by
`pipeline/new_scripts/24_generate_predictions.py`. Reloads the 12 already-
saved, already-trained models (`pipeline/models/*.pkl`/`*.pt`) - no
retraining - and re-scores the exact same split used everywhere else
(`train_test_split(test_size=0.2, stratify=y, random_state=42)`). Verified
to reproduce `table_model_cis.csv`'s AUCs exactly (0 mismatches across all
14 model/eval-set combinations) before being trusted as a data source.

This is the foundation file for every figure that needs per-variant values
rather than summary AUCs - ROC/PR curves, confusion matrices, calibration
plots, the AUC forest plot. Figures themselves go in `2/figures/` (created
2026-07-29).

## Figures started - 2026-07-29

`pipeline/new_scripts/25_roc_pr_curves.py` generates the first figure from
the figures-planning list: multi-model ROC and precision-recall curves, both
eval sets, reading `predictions_per_variant.csv` above. AUC/AUPRC values in
every legend verified to match `table_model_cis.csv` exactly. Outputs in
`2/figures/`: `fig_roc_pr_curves.png` (combined 2x2, main figure candidate),
four single-panel versions, and `roc_pr_curve_data.csv` (raw curve points,
for a supplementary data file - matches the convention several papers in
`articles/` use, e.g. Hart et al. 2020's Supplementary Data Set 1).

Caught one real bug before finalizing: the first version placed each
single-panel legend at a fixed corner (`loc="lower left"`/`"lower right"`),
which on the PR panels put legend text directly on top of the baseline line
and curve, making entries unreadable. Fixed by anchoring all single-panel
legends outside the axes (`bbox_to_anchor=(1.02, 1.0)`) instead of trusting
an in-plot corner to stay empty.

`pipeline/new_scripts/26_confusion_matrices.py` (confusion matrices, all 7
models, threshold=0.5) and `27_auc_forest_plot.py` (AUC forest plot) added
next, same day. Confusion matrix deliberately shows all 7 models in a small
multiples grid rather than picking one "canonical" model, so the figure
doesn't quietly contradict the project's own no-single-best-model finding;
Hybrid-B also gets its own standalone panel since it's the model discussed
most in the report's prose. Verified: accuracy/F1 computed here at
threshold=0.5 match `table_model_cis.csv`'s `f1@0.5` column exactly (0
mismatches). New file: `table_confusion_matrix_metrics.csv` (raw TN/FP/FN/TP
+ derived metrics per model x eval set - not in any earlier frozen table).
The forest plot reads `table_0_HEADLINE_SUMMARY.csv` directly (not the two
underlying tables separately) so it is guaranteed to agree with the
headline table's "top model" / significance calls rather than risk
re-deriving them differently.

Outputs in `2/figures/`: `fig_confusion_matrices_internal.png`,
`fig_confusion_matrices_external.png`, `fig_confusion_matrix_hybridb_internal.png`,
`fig_confusion_matrix_hybridb_external.png`, `fig_auc_forest_plot.png`.

`pipeline/new_scripts/28_calibration_plot.py` added next, same day: reliability
curves (all 7 models) + predicted-probability histogram underneath, both eval
sets, plus a fresh Brier score per model/eval-set (no earlier frozen table had
this - `table_smote_comparison.csv` has Brier scores but only for the 4
SMOTE-sweep base models under different oversampling ratios, a different
question). Matters more than a routine metric here specifically because
`CLAUDE.md`'s locked-in decision #4 (ACMG calibration) depends on
probabilities carrying real clinical meaning, not just correct ranking -
exactly what this figure checks and ROC/PR cannot.

**Honest caveat, visible directly in the figure, not just asserted**: on the
external set (n=344, only 74 pathogenic), several models' mid-range
reliability curves are noisy zigzags rather than smooth lines - including
Hybrid-B, which sits near 0 until roughly predicted probability 0.85 then
jumps. The histogram panel underneath shows why: almost all predictions
cluster near 0 or 1 (consistent with the near-ceiling AUCs already reported),
leaving very few variants in the mid-probability bins, so those bins are
noisy from small sample size, not necessarily evidence of poor calibration
in that range. State this caveat if the figure is used in the report, same
discipline as every other small-n finding in this project.

Outputs: `2/figures/fig_calibration_curves.png`, `table_calibration_metrics.csv`.

`pipeline/new_scripts/29_feature_importance.py` added next, same day: native
gain/impurity feature importance for the 3 tree-based models (RandomForest,
GradientBoosting, XGBoost), grouped bar chart, all 13 features. Scope
decision, disclosed rather than silent: SHAP was considered (Khandakji 2023
uses it as a secondary check) but installing the `shap` package would have
downgraded numpy (2.5.1 -> 2.4.6, via numba's pinned upper bound) in the one
shared venv every other frozen script in this project depends on - not worth
that risk for a plot native feature importance already covers, and matches
what most of the closely-comparable papers show anyway (Hart 2020, Khandakji
2022/2023 both use native importance as their primary figure). StackingEnsemble
and ML_VotingEnsemble excluded on purpose - heterogeneous meta-combinations
have no single well-defined per-original-feature importance the way one tree
ensemble does.

**Finding worth citing directly**: the same top 3 features
(AlphaMissense_score, VEST4_score, CADD_raw) rank highest across all three
algorithms, in the same order, despite the three models using different
splitting criteria and training procedures - real convergent evidence the
model is leaning on genuine signal rather than one algorithm's quirk.

Outputs: `2/figures/fig_feature_importance.png`, `table_feature_importance.csv`.

`pipeline/new_scripts/30_architecture_diagram.py` added last, same day - the
final item on the figures-planning must-have list. Box-and-arrow diagram of
the actual two-track pipeline (rule-based LOF/PVS1 track + missense ML/DL
track), following the same convention as the closest precedents (Li C et al.
2023 vERnet-B Fig 1A, Karalidou et al. 2022 MARGINAL Fig 1): data source ->
annotation -> 13-feature vector -> a real decision branch (variant
consequence) -> each track's own processing -> a merge point. Built with
matplotlib patches directly (FancyBboxPatch/Polygon/FancyArrowPatch), no
graphviz dependency added. Hybrid-B is expanded to show its actual internal
structure (XGBoost + MLP base learners -> 5-fold leakage-free OOF stacking ->
logistic-regression meta-learner) since it's the model discussed most in the
report's prose, while the other 6 candidate architectures are shown feeding
the same final call via a separate connector - consistent with this
project's own "no single best model" finding (Section 7), not implying
Hybrid-B is the only path evaluated.

Caught and fixed one real layout bug before finalizing: the first draft
placed the MLP sub-box and part of the ML/DL track box beyond the axes'
xlim, and because matplotlib patches added via `ax.add_patch` are clipped to
axes limits by default, they were silently cut off at the canvas edge in the
saved PNG rather than raising any error - only visible by actually opening
the rendered image. Fixed by widening xlim and re-centering the right-hand
column's coordinates so every element sits safely inside the canvas. Also
rerouted the "other 6 architectures" connector, which originally cut
straight through the Hybrid-B sub-diagram as a confusing diagonal line, as a
curved connector along the track's left edge instead.

Output: `2/figures/fig_architecture_diagram.png`.

`pipeline/new_scripts/31_individual_predictor_roc.py` added 2026-07-30, on
direct request after the user shared a screenshot from another paper's
"Fig 1. ROC curves of all models" - a multi-line overlay where each line is
one individual predictor score used directly as if it were a probability,
not a trained model. Different comparison from 25_roc_pr_curves.py (which
plots the 7 trained models): this one answers "how good is any single input
feature on its own." Deliberately restricted to this project's own 13 locked
features rather than matching the reference screenshot's 17 scores exactly -
several of theirs (SIFT, SIFT4G, LRT, fathmm-MKL/XF, MutationTaster, DANN,
PROVEAN, M-CAP) were considered and rejected during this project's own
Stage 4 feature-set decision, so including them here would blur that story
rather than support it. Same train/test split as every other frozen result
(random_state=42), so internal-test AUCs are on the identical 1,722 variants
used everywhere else. Verified zero missingness across all 13 scores on the
internal pool (n=8,609 each) before plotting, so there's no per-predictor
denominator difference to report unlike some literature examples.

Outputs: `2/figures/fig_individual_predictor_roc.png`,
`table_individual_predictor_auc.csv`.

## Figures-planning must-have list: complete as of 2026-07-30

All 7 must-have figures from the 2026-07-29 planning pass are now built:
architecture diagram, ROC curves, PR curves, confusion matrices, AUC forest
plot, calibration curves, feature importance. Remaining items from that same
planning pass are "strongly recommended" tier, not yet started: the
circularity/training-era bar chart, the BRCA1/2 lollipop variant-position
plot, the BRCA1-DMS scatter (predicted probability vs. functional score),
and the Kazakh population figure.

## table_0_HEADLINE_SUMMARY.csv - the single table to show if asked "what's the final result"

Built 2026-07-29, combining `table_model_cis.csv` and `table_pairwise_significance.csv`
into one row-per-model-per-eval-set table: ROC AUC with its 95% CI, F1, and whether that
model's AUC is statistically distinguishable from the top point estimate on that same
eval set. This is more honest than quoting any single model's raw AUC, because the raw
numbers alone (all 0.97-1.00) invite exactly the "our model is the best" overclaim that
`table_pairwise_significance.csv` already disproved. Headline reading: on the external
set, literally none of the 7 models is significantly different from the top performer
(StackingEnsemble, point estimate only). On internal test, the top performer (SVM) is
significantly different from 4 of the other 6, but not from the voting ensemble or
Hybrid-B. Use this table, not `table_internal_comparison.csv` or
`table_external_validation.csv` alone, when someone asks for "the" result.

## Known, disclosed limitations of this frozen run (not fixed by freezing it)

- ~~No bootstrap confidence intervals or significance testing~~ **Fixed
  2026-07-26** via #17 - see `docs/memory/project_statistical_significance.md`.
  Central finding: no model, including Hybrid-B, is statistically
  distinguishable from the others on external validation. Don't claim a
  "best model" from point-estimate AUC alone in any write-up.
- ~~External pool's circularity exposure has no clean alternative
  estimate~~ **Fixed 2026-07-26** via #18 - post-cutoff ENIGMA filtering was
  tried and found genuinely infeasible (verified: 0 benign variants remain
  post-2020), so pivoted to the full BRCA1-DMS validation instead. See
  `docs/memory/project_clean_external_estimate.md`. Cite rho=-0.58
  (n=2,361) / rho=-0.50 (n=1,717, novel-only) as the clean estimate, not the
  smaller 377-variant subset or the ENIGMA external-pool AUC alone.
- Kazakh missense n=6 (#15) is explicitly descriptive, not a statistical claim.

## Regenerating this directory

```
cd "BC project"
.venv/bin/python pipeline/new_scripts/04_filter_and_report.py
.venv/bin/python pipeline/new_scripts/05_classical_ml.py
.venv/bin/python pipeline/new_scripts/06_ensembles.py
.venv/bin/python pipeline/new_scripts/09_hybrid.py      # BiLSTM hybrid -- superseded by Hybrid-B, but
                                                          # 10_external_validation.py needs its saved model
.venv/bin/python pipeline/new_scripts/09b_hybrid_stack.py
.venv/bin/python pipeline/new_scripts/10_external_validation.py
.venv/bin/python pipeline/new_scripts/11_circularity_check.py
.venv/bin/python pipeline/new_scripts/12_training_era_overlap.py
.venv/bin/python pipeline/new_scripts/13_acmg_calibration.py
.venv/bin/python pipeline/new_scripts/14_stage8_benchmarking.py
.venv/bin/python pipeline/new_scripts/15_brca1_dms_validation.py
.venv/bin/python pipeline/new_scripts/16_kazakh_af_aggregation.py
.venv/bin/python pipeline/new_scripts/17_kazakh_pm2_bs1.py
.venv/bin/python pipeline/new_scripts/18_kazakh_pm2_bs1_case_study.py
.venv/bin/python pipeline/new_scripts/19_kazakh_variant_evaluation.py
.venv/bin/python pipeline/new_scripts/20_smote_sweep.py
```
