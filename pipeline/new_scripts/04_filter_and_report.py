"""
Phase 3 of the BRCA1/2 reproduction pipeline.

Two separate filters off the same annotated pools (chat_history.md SS16,
plan Phase 3):
  - ML/ensemble track: complete-case on all scores. No imputation at this
    stage; median imputation (if any) happens inside the sklearn Pipeline
    in Phase 4, on residual missingness only.
  - DL track: SNV-only (ref/alt both length 1). Doesn't need any of the
    scores, just sequence context + label, so this is deliberately
    much less restrictive than the ML filter -- this directly fixes
    issue #5 from chat_history.md SS5 (DL was starved of data because
    the original pipeline filtered DL down to the same missense-only,
    complete-case set as the tabular models).

Also produces the full per-feature missingness report for both pools,
closing issue #3 ("external missing values never resolved" -- never
even measured explicitly before this).

2026-07-25 (Stage 4): added BayesDel_noAF_score as a 12th score. Decision
made after reading Feng 2017 (BayesDel's primary paper, articles/54 Feng.pdf) --
it has a published, ENIGMA-validated, circularity-controlled BRCA1/BRCA2 AUC
of 0.94, beating MetaSVM (0.90)/MetaLR (0.88)/CADD (0.88) on the same two
genes, which is stronger direct gene-specific evidence than exists for any of
our other 11 features individually. Uses dbNSFP 5.3.1a's own BayesDel_noAF_score
(MANE-Select-resolved by 02_local_dbnsfp_scores.py), NOT BRCA-Exchange's own
bayesdel_noaf column -- traced the latter via
release-11-11-25/output/variants_output_field_metadata.tsv to
"BayesDel_nsfp33a_noAF", i.e. sourced from dbNSFP v3.3a (stale, same kind of
provenance problem the Eigen/EigenPC fix addressed earlier this session).
Verified adding this column costs zero additional rows in complete-case
filtering -- it is present in essentially every row where the other 11 scores
already are (both are missense-only, complete-annotation subsets).

2026-07-25 (Stage 4, continued): added Eigen_raw_coding as a 13th score, and
decided NOT to add Eigen_PC_raw_coding alongside it or PrimateAI_score's
removal, per an empirical ablation
(pipeline/new_scripts/11b_eigen_primateai_ablation.py):
  - Eigen and EigenPC are 0.99-correlated with each other (near-duplicates --
    EigenPC is just Eigen's simpler eigendecomposition-only fallback per the
    primary paper, articles/53 IonitaLaza.pdf) and EigenPC is 0.92-correlated
    with CADD_raw (largely redundant with what's already in the set). The
    primary paper's own BRCA1 benchmark also found Eigen more significant
    than EigenPC (p=2.5E-38 vs 6.0E-25). So: Eigen only, not both.
  - Adding Eigen improved F1 at both internal (0.775->0.813) and external
    (0.960->0.966) with AUC roughly flat -- real incremental signal despite
    moderate correlation with CADD, likely because Eigen's unsupervised
    covariance-based weighting differs enough from CADD's supervised-SVM
    construction.
  - PrimateAI_score (Sundaram et al. 2018, articles/46 Sundaram.pdf) is kept
    despite its own authors' explicit request that its scores not be fed into
    other classifiers, to avoid compounding field-wide circularity. Verified
    via MetaRNN's own paper (Li C et al. 2022) that none of our other 12
    features use PrimateAI as an internal training component, so this isn't a
    within-feature-set double-counting problem. Removing it has a real,
    measured cost (internal F1 0.775->0.739), and this project isn't
    publishing a new widely-reused benchmark back into the field -- the
    specific harm the request targets. Documented here rather than silently
    included or silently dropped.

2026-07-26 (Stage 4, closed): evaluated and REJECTED popEVE_score as a 14th
feature -- the one candidate that does NOT get added, unlike BayesDel/Eigen.
Unlike those two (zero-cost additions -- present in essentially every row the
other scores already are), popEVE's raw coverage is only 14.9%/2.6%
(internal/external) and does NOT align with the existing complete-case
population: requiring it would cut the internal ML-ready set from 8609 to
2850 (-67%) and the external set from 344 to 189 (-45%), the latter already a
small, precision-limited validation set. An ablation on the popEVE-available
subset (n=2850, run inline, not saved as a standalone script) showed only a
modest lift from adding it (AUC 0.980->0.985, F1 0.861->0.873) -- real, but
nowhere near enough to justify losing that much data. Also, unlike BayesDel/
Eigen, no popEVE primary paper was actually read in this project's 54-paper
literature review -- the only credibility on record is indirect, via its
predecessor EVE (cited by Rastogi/Cheng as a strong non-circular baseline),
so this decision doesn't get to lean on the same verified-evidence standard
as the other three. Final feature set: 13 scores (the original 11 +
BayesDel_noAF_score + Eigen_raw_coding).
"""
import pandas as pd

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]


def is_snv(row):
    return len(str(row["ref"])) == 1 and len(str(row["alt"])) == 1 and str(row["ref"]) != "-" and str(row["alt"]) != "-"


def process_pool(pool_name, missingness_rows):
    df = pd.read_csv(f"pipeline/{pool_name}_final_annotated.csv")
    n_total = len(df)

    # Missingness report (every score, this pool)
    for col in SCORE_COLS:
        missing_pct = df[col].isna().mean() * 100
        missingness_rows.append({"pool": pool_name, "feature": col, "n": n_total,
                                  "pct_missing": round(missing_pct, 1)})

    # ML/ensemble track: complete-case
    ml_mask = df[SCORE_COLS].notna().all(axis=1)
    ml_df = df[ml_mask].copy()
    ml_df.to_csv(f"pipeline/{pool_name}_ml_ready.csv", index=False)

    # DL track: SNV-only, independent of score completeness
    dl_mask = df.apply(is_snv, axis=1)
    dl_df = df[dl_mask][["gene_symbol", "chr", "pos", "ref", "alt", "label"]].copy()
    dl_df.to_csv(f"pipeline/{pool_name}_dl_ready.csv", index=False)

    print(f"\n{pool_name} pool: {n_total} total")
    print(f"  ML track (complete-case, {len(SCORE_COLS)} scores): {len(ml_df)} ({len(ml_df)/n_total*100:.1f}%)")
    print(f"    label balance: {ml_df['label'].value_counts().to_dict()}")
    print(f"  DL track (SNV-only): {len(dl_df)} ({len(dl_df)/n_total*100:.1f}%)")
    print(f"    label balance: {dl_df['label'].value_counts().to_dict()}")

    return ml_df, dl_df


if __name__ == "__main__":
    missingness_rows = []
    for pool in ["internal", "external"]:
        process_pool(pool, missingness_rows)

    report = pd.DataFrame(missingness_rows)
    report.to_csv("pipeline/missingness_report.csv", index=False)
    print("\n=== Missingness report (also saved to pipeline/missingness_report.csv) ===")
    pivot = report.pivot(index="feature", columns="pool", values="pct_missing")
    print(pivot)
