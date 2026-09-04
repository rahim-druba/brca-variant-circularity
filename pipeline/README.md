# pipeline/ - script organization (reorganized 2026-07-25)

Scripts are split into `new_scripts/` and `old_scripts/`. **Data files (CSVs, VCFs,
the `esm1b/` and `annovar_out/` folders) stay in this top-level `pipeline/` folder,
unmoved** - every script uses plain relative paths like `"pipeline/internal_pool.csv"`,
which only resolve correctly when you run from `BC project/` as the working
directory. Moving the scripts into subfolders does not affect this, since the paths
are hardcoded strings, not resolved relative to the script's own location. Always
run scripts like this, regardless of which subfolder they're in:

```
cd "BC project"
.venv/bin/python pipeline/new_scripts/01_build_pools.py
```

## final_results_2026-07-26/ - the canonical, frozen source of truth

**Every number in any write-up should trace back to a file in
`final_results_2026-07-26/`, not to any other copy elsewhere in `pipeline/`.**
That directory has its own `MANIFEST.md` listing every file, the exact
script that produced it, the locked 13-feature set, and a one-line result
summary. Earlier per-stage snapshots (`*_12score_bayesdel.csv`,
`*_13score_eigen.csv`, etc.) remain in the top-level `pipeline/` folder as an
audit trail of how each Stage-4 decision changed the numbers, but they are
historical, not canonical - don't cite them directly.

## new_scripts/ - written or fixed/verified this session (2026-07-23/25)

- **`01_build_pools.py`** - extended to also pull BayesDel, SpliceAI, and the
  official ACMG population-frequency evidence code from `variants_output.tsv`
  (previously only 7 of ~309 available columns were read).
- **`02_local_dbnsfp_scores.py`** - new. Replaces the myvariant.info API with
  direct local lookups against `downloads/dbNSFP5.3.1a_grch38.gz`. Fixed 2026-07-25
  to correctly resolve dbNSFP's per-transcript semicolon-separated values to the
  MANE-Select canonical transcript (an earlier version of this script had a real
  bug here - see `new/session_log/chathistory.md` for the full story).
- **`03_merge_annotation.py`** - fixed 2026-07-25: was silently missing the ESM1b
  sign correction (`load_esm1b_scores()` returned the raw, inverted
  log-likelihood-ratio instead of negating it). `BRCA_final_dataset.csv` itself
  was already correct (someone fixed it by hand at some point), but this script's
  own output was not - meaning re-running it before this fix would have silently
  reproduced a near-perfectly-backwards ESM1b column (AUC 0.078 instead of 0.922).
- **`04_filter_and_report.py`**, **`05_classical_ml.py`**, **`06_ensembles.py`**,
  **`09_hybrid.py`**, **`10_external_validation.py`** - copied from `old_scripts/`
  2026-07-25 (Stage 4) with one change: `SCORE_COLS` now includes
  `BayesDel_noAF_score` as a 12th (13th for the hybrid) feature. Decision
  rationale (why add it, why this exact column and not BRCA-Exchange's own
  `bayesdel_noaf`) is in `04_filter_and_report.py`'s docstring and
  `articles/citation_log.md`'s Batch 5 notes. These 5 scripts were already
  re-run once earlier this session (2026-07-25) against the corrected
  annotation data (Eigen/EigenPC fix, canonical-transcript fix, ESM1b sign
  fix) using the *old_scripts* 11-score versions - that run's results
  (`table_internal_comparison.csv`, `table_external_validation.csv`, etc.) are
  now stale and need re-running with these 12/13-score versions.

**Stage 4 closed 2026-07-26**: final feature set is 13 scores (original 11 +
BayesDel_noAF_score + Eigen_raw_coding). popEVE_score was evaluated and
rejected - its coverage (14.9%/2.6% internal/external) doesn't align with the
existing complete-case population, so requiring it would cut the internal set
by 67% and the external set by 45% for only a modest gain. Full reasoning for
all four candidates (BayesDel, Eigen vs EigenPC, PrimateAI, popEVE) is in
`04_filter_and_report.py`'s docstring.

**Resolved 2026-07-25**: `09_hybrid.py` (old or new) is the *original*
BiLSTM-sequence hybrid - its DL half is a near-coin-flip standalone sequence
model (external AUC 0.53/0.54 for 1D-CNN/BiLSTM alone, confirmed by
`10_external_validation.py`'s re-run), so it isn't a real demonstration of
combining ML+DL. The project's canonical "Hybrid-B" (XGBoost + MLP base
learners over the *tabular* scores → logistic-regression meta-learner) had no
surviving script (same kind of gap as the missing SMOTE-sweep script, see
root `CLAUDE.md`'s "Known gaps") - only its stale results table
(`2/results/table_hybrid_mldl.csv`) remained. Rebuilt as
**`new_scripts/09b_hybrid_stack.py`**, on the current 12-score (BayesDel-
inclusive) feature set, saving to its own file
(`pipeline/table_hybrid_b_12score_bayesdel.csv`) rather than the shared
`table_internal_comparison.csv`. New numbers are competitive with, but not a
clean standout over, plain XGBoost/the voting ensemble - see
`docs/memory/project_hybrid_mldl.md`'s 2026-07-25 update before citing the
old "beats both base models" claim.

**`new_scripts/11_circularity_check.py`** - Stage 5's first (coarse) check
(2026-07-25): retrains XGBoost on a "robust" (no clinical-label training
exposure) 5-feature subset vs the full 12 vs a "clinical-label-biased"
7-feature subset. The robust-only subset nearly matches the full model's AUC
(0.976 vs 0.977 internal, 0.9951 vs 0.9986 external) and beats the biased-only
subset. Results: `pipeline/table_circularity_check.csv`.

**`new_scripts/23_generate_final_report.py`** - Stage 10 (2026-07-27), the
final piece. Generates `2/BRCA_Final_Technical_Report_2026-07-27.docx`
directly from `pipeline/final_results_2026-07-26/`, pulling numbers
programmatically rather than typing them in by hand. Formatting: Times New
Roman throughout, 12pt body text, 14pt subtitles, all paragraphs justified,
no em dashes or en dashes anywhere (verified directly: zero of either across
the full document text and every table cell). Covers the entire session's
work end to end - data sources, the 13-feature set and why each addition/
rejection was made, the two-track design, all model results, the
statistical-significance finding (no model beats another externally), the
circularity audit, gene-specific calibration, Stage 8 benchmarking, both
BRCA1-DMS validation passes, the SMOTE sweep, and all four pieces of Kazakh
work. Supersedes `2/BRCA_Technical_Report_v2.docx` and root
`final_report.docx` (both reflect pre-session numbers - do not cite them).

**`new_scripts/22_brca1_dms_full_validation.py`** - fixes blocker #4
(2026-07-26): the "clean external estimate" problem. A post-cutoff-only
ENIGMA filter was tried first and found genuinely infeasible (verified
directly: filtering to post-2020 leaves 0 benign variants, can't compute an
AUC at all) -- disclosed as a real dead end, not silently dropped. Pivoted
to extending `15_brca1_dms_validation.py`'s BRCA1-DMS check to the full
~3,893-variant Findlay et al. 2018 SGE set, annotated in one dbNSFP pass via
its own `HGVSc_VEP` field (confirmed to exactly match Findlay's own
c.-notation format). Caught and fixed a real bug along the way (single-
valued dbNSFP fields silently returning "." when indexed by MANE position
without the size-1 short-circuit `02_local_dbnsfp_scores.py` already has).
Result: 2,361 variants scored, 1,717 (72.7%) genuinely novel VUS never in
our ClinVar/ENIGMA pools; Spearman rho=-0.58 overall / -0.50 novel-only,
both p<1e-100 -- weaker than the smaller 377-variant subset's rho=-0.75 but
honestly reported as expected (that subset skewed toward clearer-cut
ClinVar/ENIGMA-curated cases). See
`docs/memory/project_clean_external_estimate.md`. Results:
`pipeline/table_brca1_dms_full_validation.csv`.

**`new_scripts/21_statistical_significance.py`** - fixes blocker #3 from the
publishability assessment (2026-07-26): bootstrap 95% CIs + pairwise
paired-bootstrap AUC significance testing across all 7 tabular/Hybrid-B
models. Central finding: **0/21 model pairs significant on the external set;
Hybrid-B is not distinguishable from any other model on either evaluation
set.** Do not claim a "best model" from point-estimate AUC alone in any
write-up going forward. See
`docs/memory/project_statistical_significance.md`. Results:
`pipeline/table_model_cis.csv`, `pipeline/table_pairwise_significance.csv`.

**`new_scripts/20_smote_sweep.py`** - rebuild of the missing SMOTE sweep
(2026-07-26), closing another `CLAUDE.md`-flagged gap and its documented
scope shortfall (RF+XGBoost only shipped before; this covers all 4 originally
planned base models). Core finding replicates: SMOTE 1:1 worst by Brier on
every model. See `docs/memory/project_smote_rejected.md`. Results:
`pipeline/table_smote_comparison_13score_frozen.csv`.

**`new_scripts/19_kazakh_variant_evaluation.py`** - closes `CLAUDE.md`'s
long-standing "Kazakh two-class two-track run - not done" gap (2026-07-26).
Annotates and evaluates all of `2/data/kazakh_variants_reference.csv`
through both tracks. The LOF-track rule script was also missing (same fate
as Hybrid-B/ACMG calibration/SMOTE) -- rebuilt faithfully from
`docs/memory/project_two_track_lof.md`'s description. Results: LOF rule
33/33 correct; Hybrid-B perfectly separates the 6 labeled missense variants
(AUC 1.0, explicitly descriptive only, n too small); 3 genuine VUS all score
benign-like (0.116-0.117, indistinguishable from the known-benign cluster).
See `docs/memory/project_kazakh_two_track_run.md`. Results (all carry a
`source_paper` column): `2/data/kazakh_lof_track_evaluation.csv` (Samigatova
2026: 19, Zhunussova 2023: 14), `2/data/kazakh_missense_track_evaluation.csv`
(Akilzhanova 2013: 8, Samigatova 2026: 2, Zhunussova 2023: 1), and
**`2/data/kazakh_variants_final_annotated.csv`** (2026-07-26, added on
request) -- a single master table, all 54 original reference variants in
their original order, one row each, with a `track` column (LOF-rule/
Missense-ML/Excluded), `exclusion_reason` where applicable, both tracks'
predictions, and all 13 SCORE_COLS as columns (NaN where a predictor
doesn't apply, e.g. missense scores for LOF-track truncating variants --
expected, not a gap). Full citations in `articles/citation_log.md`.

**`new_scripts/18_kazakh_pm2_bs1_case_study.py`** - usage demonstration
(2026-07-26), prompted by "can it be used to test other variants, if not
how do we show its usage?" Cross-checked against the 54-row real Kazakh
clinical variant reference (`2/data/kazakh_variants_reference.csv`):
25/45 coordinate-ready variants land on the array. All 20 known pathogenic
ones correctly get PM2_Kazakh_Supporting; both known benign polymorphisms
independently reproduce Akilzhanova's own "likely benign" call via
BS1_Kazakh; the 2 genuine unresolved VUS get real, opposite-direction new
evidence. See `docs/memory/project_kazakh_pm2_bs1_case_study.md`. Results:
`2/data/kazakh_pm2_bs1_case_study.csv`.

**`new_scripts/17_kazakh_pm2_bs1.py`** - built on the AF table (2026-07-26)
to close the gap Stage 5 found (zero PM2/BS1/BA1 computation anywhere in
this repo). PM2 capped at Supporting strength, never Moderate (Liu S et
al. 2025's BayesQuantify found standard gnomAD-scale PM2 barely clears
Supporting; our 224-person sample is far smaller). BS1 uses Strande et
al. 2017's MAF<0.01% threshold applied to a Wilson-confidence-interval
lower bound, not the raw AF point estimate (at n=448 alleles, a single
observation already gives a point estimate above that threshold -- pure
noise, not evidence). Result: 4,096/4,160 positions get PM2, only 64 clear
the CI-aware BS1 bar. Cross-check independently re-flags the same 8
ENIGMA "pathogenic" positions (including the 2 likely array-indel
artifacts) found via raw AF in the previous script -- a different,
stricter method landing on the same variants. See
`docs/memory/project_kazakh_pm2_bs1.md`. Results:
`2/data/kazakh_pm2_bs1_evidence.csv`.

**`new_scripts/16_kazakh_af_aggregation.py`** - Stage 9 (2026-07-26): loops
genotype tallying across all 224 local Kazakh sample VCFs to build a
BRCA1/BRCA2 population AF table. Real bug caught before it corrupted
results: the VCFs are GRCh37/hg19, not hg38 like everything else in this
pipeline (confirmed via the VCF header, not assumed) -- an earlier session's
single-position "test fetch" didn't record which coordinates it used, so may
have silently queried the wrong genomic window. Fixed via `pyliftover` +
UCSC's official chain file. Result: 4,160 positions, 803/2,442 overlap with
our internal/external pools respectively; 2 "pathogenic"-labeled positions
show implausibly high AF (~0.28), flagged as likely SNP-array indel
artifacts (arrays are known-unreliable for indels in repetitive sequence),
not reported as real findings. See
`docs/memory/project_kazakh_af_aggregation.md`. Results:
`2/data/kazakh_brca_af_table.csv`.

**`new_scripts/15_brca1_dms_validation.py`** - Stage 8's third check
(2026-07-26), the one flagged as needing new data acquisition. Found and
fetched Findlay et al. 2018's BRCA1 saturation genome editing data from
MaveDB's real, working REST API (`api.mavedb.org` - the PredictMD *portal*
used for Stage 7 is unscrapable, but the underlying MaveDB infrastructure has
a documented API, found via its `/openapi.json`). Result: Hybrid-B's
predicted probability correlates rho=-0.75 (p=1.2e-69) with Findlay's
independent functional score across 377 overlapping variants, and our
pathogenic/benign labels separate strongly on that same independent score
(p=1.1e-56) -- real validation against ClinVar-independent ground truth. Only
covers the subset already in our own labeled pool (481 of ~3893 SGE
variants); extending to the full set (mostly novel VUS) is a natural next
step, not done here. See `docs/memory/project_brca1_dms_validation.md`.
Results: `pipeline/table_brca1_dms_validation.csv`.

**`new_scripts/14_stage8_benchmarking.py`** - Stage 8 (2026-07-26), per
Rastogi et al.'s benchmark methodology: high-specificity (FPR<=5%) /
high-sensitivity (TPR>=95%) regime reporting, and a gene-AND-label-balanced
subset check. Real finding: BRCA2 needs a notably higher FPR than BRCA1 to
catch 95% of pathogenic variants (e.g. Hybrid-B: 0.32 vs 0.18) -- consistent
with the gene-specific calibration finding, but a distinct, sensitivity-
regime-specific gap, not just a threshold-shift difference. Gene-balanced
check confirms performance isn't just exploiting BRCA1's higher baseline
pathogenic prior. BRCA1-DMS functional-assay validation NOT done -- no such
data file exists anywhere in this repo; would need new data acquisition. See
`docs/memory/project_stage8_benchmarking.md`. Results:
`pipeline/table_stage8_benchmarking.csv`.

**`new_scripts/13_acmg_calibration.py`** - Stage 7 (2026-07-25/26): rebuilt
the lost ACMG PP3/BP4 calibration script (same gap as Hybrid-B/SMOTE - only
`2/results/table_acmg_thresholds.csv` survived), extended to gene-specific
(BRCA1 vs BRCA2) thresholds per Chen et al. 2026's finding. Chen's own exact
BRCA1 thresholds are locked behind an unscrapable client-side SPA
(https://igvf.mavedb.org/, confirmed no accessible API) - adopted the finding
instead of guessing at numbers, and re-derived real thresholds from our own
data. Hit and fixed two real bugs along the way (raw-probability KDE breaking
on the Hybrid-B model's saturated output; grid extrapolation past the data's
support producing spurious billions-scale LR spikes) - see
`docs/memory/project_gene_specific_calibration.md`. External validation holds
up (PP3_Strong 98-100% pathogenic, BP4_Strong 0% pathogenic, all splits).
Results: `pipeline/table_acmg_calibration_gene_specific.csv`,
`pipeline/table_acmg_calibration_external_validation.csv`.

**`new_scripts/12_training_era_overlap.py`** - Stage 5's finer, variant-level
check (2026-07-25): joins each pool's variants against BRCA-Exchange's
`datesignificancelastevaluated_clinvar` to see what fraction already had a
ClinVar classification on/before each predictor's training-data cutoff.
Substantial overlap found, and **worse for the external pool than the
internal one** (71-98% of external variants pre-date BayesDel/MetaRNN-era
cutoffs vs 20-65% internal) - reframes the 11_circularity_check.py result
less optimistically: the small residual lift from the clinically-trained
features over the robust-only subset could itself be the circularity effect.
Don't cite external AUC for the 7 clinically-trained features as a clean
circularity-free result. See `docs/memory/project_training_era_overlap.md`.
Results: `pipeline/table_training_era_overlap.csv`. AF double-counting
(PM2/BS1/BA1 combined with AF-informed scores): verified absent from this
repo's own code (`grep -rl "PM2\|BS1\|BA1"` = zero hits anywhere). The
deeper question (whether ENIGMA/ClinVar's original curators used AF-based
evidence for some of our benign labels themselves) is unresolved -- no
per-variant evidence-code breakdown available to check further.

## old_scripts/ - pre-session, not touched or re-verified this session

- **`02_myvariant_scores.py`** - the original API-based annotation method,
  superseded by `new_scripts/02_local_dbnsfp_scores.py`. Kept for reference/diffing,
  not for active use.
- **`04_filter_and_report.py`**, **`05_classical_ml.py`**, **`06_ensembles.py`**,
  **`09_hybrid.py`**, **`10_external_validation.py`** - 11-score versions,
  superseded by the `new_scripts/` copies above (2026-07-25, Stage 4). Kept for
  diffing.
- **`07_dl_windows.py`**, **`08_dl_models.py`** - unchanged, no `SCORE_COLS`
  reference (sequence-only DL track), so nothing to update here for Stage 4.
- **`12_generate_docx_report.py`** - the writeup-generation stage. Still
  hardcodes prose referring to "11 scores" / "12 feature set" (hybrid);
  needs updating once the 12/13-score re-run and Hybrid-B are both settled.
