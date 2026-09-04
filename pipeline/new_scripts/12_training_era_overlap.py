"""
Stage 5, second (finer-grained) circularity check: training-era overlap.

The coarse check (11_circularity_check.py) tested feature GROUPS and found
the "robust" (non-clinically-trained) subset nearly matches the full model's
AUC -- reassuring, but it can't say whether any specific variant in our own
eval set was literally inside a given predictor's training set. This script
does the closest thing to that check that's actually computable from data we
have: BRCA-Exchange's variants_output.tsv carries
`datesignificancelastevaluated_clinvar`, the date ClinVar's classification for
each variant was last evaluated. If that date is ON OR BEFORE a predictor's
published training-data cutoff, the variant almost certainly already existed,
classified, in ClinVar when that predictor's training set was assembled --
i.e. it's a lower-bound estimate of "at risk of being a training-set member"
(a lower bound because a later evaluation date doesn't rule out the variant
having existed earlier too, just not been re-reviewed).

Cutoffs used, each traced to a primary source already in articles/:
  - BayesDel (Feng 2017, `54 Feng.pdf`): ClinVar snapshot 2015-08-04 --
    the ONLY one of our 13 features with an exact, published cutoff date.
  - MetaRNN (Li C et al. 2022, `9 Li C.pdf`): trained on ClinVar-derived
    labels, exact date not given in the paper -- 2020 used as a conservative
    (i.e. generous to MetaRNN) proxy since the paper itself was published
    2022 and cites a "recent" ClinVar pull.
  - General "2015-2018 era" tools (ClinPred 2018, REVEL 2016, MetaSVM/MetaLR
    2015, VEST4 ~2016): no single exact date is published for any of them in
    what we've read: reported as a range, not pretended to be precise.

This is a LOWER BOUND, not a proof of overlap, and can't say anything at all
about the "robust" (unsupervised) features, which by construction have no
clinical-label training set to overlap with.
"""
import pandas as pd

CUTOFFS = {
    "BayesDel (2015-08-04, exact, Feng 2017)": "2015-08-04",
    "MetaRNN (2020-01-01, conservative proxy, Li C 2022)": "2020-01-01",
    "generic 2015-2018-era tools (2018-12-31, conservative proxy)": "2018-12-31",
}


def earliest_date(raw):
    """The field is comma-separated (one date per ClinVar submission/re-evaluation,
    e.g. '2018-01-13,2020-08-18,2022-04-29'), sometimes with a literal 'None' mixed
    in. A first pass using pd.to_datetime directly on the raw string silently
    returned NaT for ~96% of rows here -- caught by checking the match rate against
    a simpler existence check before trusting the "low overlap" result. What matters
    for a training-era question is the EARLIEST date (when the variant first had a
    ClinVar classification), not any single value from the list.
    """
    if raw == "-" or pd.isna(raw):
        return pd.NaT
    parsed = pd.to_datetime([p for p in raw.split(",") if p not in ("-", "None")], errors="coerce")
    return parsed.min() if len(parsed) else pd.NaT


def main():
    dates = pd.read_csv(
        "release-11-11-25/output/variants_output.tsv", sep="\t", low_memory=False,
        usecols=["chr", "pos", "ref", "alt", "datesignificancelastevaluated_clinvar"],
    )
    dates["eval_date"] = dates["datesignificancelastevaluated_clinvar"].apply(earliest_date)

    rows = []
    for pool_name in ["internal", "external"]:
        pool = pd.read_csv(f"pipeline/{pool_name}_ml_ready.csv")
        merged = pd.merge(pool[["chr", "pos", "ref", "alt", "label"]], dates,
                           on=["chr", "pos", "ref", "alt"], how="left")
        n = len(merged)
        has_date = merged["eval_date"].notna().sum()
        print(f"\n=== {pool_name} ML-ready pool ({n} variants, {has_date} with a ClinVar evaluation date) ===")
        for label_name, label_val in [("pathogenic", 1), ("benign", 0)]:
            sub = merged[merged["label"] == label_val]
            n_sub = len(sub)
            for cutoff_name, cutoff in CUTOFFS.items():
                pre_cutoff = (sub["eval_date"] <= pd.Timestamp(cutoff)).sum()
                pct = pre_cutoff / n_sub * 100 if n_sub else float("nan")
                print(f"  {label_name:10s} (n={n_sub:4d}): {pre_cutoff:4d} ({pct:5.1f}%) evaluated on/before {cutoff_name}")
                rows.append({"pool": pool_name, "label": label_name, "n": n_sub, "cutoff": cutoff_name,
                             "n_pre_cutoff": int(pre_cutoff), "pct_pre_cutoff": round(pct, 1)})

    pd.DataFrame(rows).to_csv("pipeline/table_training_era_overlap.csv", index=False)
    print("\nSaved to pipeline/table_training_era_overlap.csv")


if __name__ == "__main__":
    main()
