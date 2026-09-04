"""
Kazakh-population PM2/BS1 ACMG evidence codes, built from
data/kazakh/kazakh_brca_af_table.csv (Stage 9's output). Addresses the gap
confirmed in Stage 5 (grep -rl "PM2\\|BS1\\|BA1" across every script in this
repo = zero hits before this file): this pipeline computed no
population-frequency ACMG evidence at all.

Why NOT just apply the standard ACMG PM2 (Moderate)/BS1 (Strong) codes at
face value:

1. **PM2 strength**: Liu S et al. 2025 ("BayesQuantify",
   articles/citation_log.md) empirically found standard PM2 (rarity in
   gnomAD, a ~800K-allele reference) only reaches LR+=1.497 -- BELOW even
   the Supporting-evidence threshold (~2.08, this project's own OddsPath
   ladder from Stage 7). ClinGen SVI's own newer guidance is that PM2 is
   commonly overweighted at Moderate strength. Our Kazakh reference sample
   is 224 people / 448 alleles -- vastly smaller than gnomAD's. If PM2 is
   already weak evidence at gnomAD's scale, it cannot be stronger than that
   at ours. Assigned as PM2_Kazakh_Supporting only, never Moderate, and only
   when the sample is essentially fully genotyped (>=200/224 samples called
   at that position, to avoid drawing a "rare" conclusion from a mostly-
   missing position).

2. **BS1 strength and threshold**: Strande et al. 2017 (Genet Med
   19:1041-1051, in articles/) found 97.3% of pathogenic variants across a
   broad spectrum of disorders have population MAF<0.01% (0.0001) --
   supporting that as a starting BS1 threshold absent gene-specific data.
   BUT our resolution floor is 1/448=0.00223 -- meaning a SINGLE observed
   allele already produces a point-estimate AF above the 0.0001 threshold,
   which would trigger BS1 on pure sampling noise. Fixed by computing a
   Wilson score confidence interval on each position's true population AF
   and requiring the LOWER bound (not the point estimate) to exceed the
   Strande threshold -- i.e., only assign BS1 when we're statistically
   confident the true AF exceeds 0.01%, not just when we happened to see it
   once. In practice this requires observing the allele in several samples,
   not one.

This produces a Kazakh-specific supplementary evidence table, not a
replacement for gnomAD-based PM2/BS1 -- it's most useful for variants where
Kazakh-ancestry-specific frequency differs from gnomAD's largely
European/other-ancestry-dominated composition (the actual justification for
building a population-matched resource at all, per the earlier "what can we
do with this file" discussion).
"""
import math
import pandas as pd

BS1_THRESHOLD_STRANDE = 0.0001  # Strande et al. 2017's cross-disorder starting BS1 threshold
MIN_GENOTYPED_FOR_PM2 = 200     # out of 224 -- require near-complete genotyping before calling "absent"
Z_95 = 1.959964


def wilson_lower_bound(alt_count, n_alleles, z=Z_95):
    """Wilson score interval lower bound for a binomial proportion --
    more reliable than a normal approximation at small n or extreme
    proportions (both apply here: alt_count is often 0-3, n_alleles~448)."""
    if n_alleles == 0:
        return 0.0
    p = alt_count / n_alleles
    denom = 1 + z**2 / n_alleles
    center = p + z**2 / (2 * n_alleles)
    margin = z * math.sqrt(p * (1 - p) / n_alleles + z**2 / (4 * n_alleles**2))
    return max(0.0, (center - margin) / denom)


def main():
    af = pd.read_csv("data/kazakh/kazakh_brca_af_table.csv")
    af["n_alleles"] = af["n_samples_genotyped"] * 2

    def classify(row):
        n_alleles = row["n_alleles"]
        alt_count = row["alt_allele_count"]
        n_samples = row["n_samples_genotyped"]

        pm2 = (alt_count == 0) and (n_samples >= MIN_GENOTYPED_FOR_PM2)

        af_lower_bound = wilson_lower_bound(alt_count, n_alleles) if n_alleles else 0.0
        bs1 = af_lower_bound > BS1_THRESHOLD_STRANDE

        return pd.Series({"af_wilson_lower_95": round(af_lower_bound, 6),
                           "PM2_Kazakh_Supporting": pm2,
                           "BS1_Kazakh": bs1})

    evidence = af.join(af.apply(classify, axis=1))
    evidence.to_csv("data/kazakh/kazakh_pm2_bs1_evidence.csv", index=False)

    n_pm2 = evidence["PM2_Kazakh_Supporting"].sum()
    n_bs1 = evidence["BS1_Kazakh"].sum()
    n_neither = len(evidence) - n_pm2 - n_bs1
    print(f"Total positions: {len(evidence)}")
    print(f"  PM2_Kazakh_Supporting (absent, >=200/224 genotyped): {n_pm2} ({n_pm2/len(evidence)*100:.1f}%)")
    print(f"  BS1_Kazakh (Wilson-lower-bound AF > {BS1_THRESHOLD_STRANDE}): {n_bs1} ({n_bs1/len(evidence)*100:.1f}%)")
    print(f"  Neither (present but not confidently common enough for BS1): {n_neither} ({n_neither/len(evidence)*100:.1f}%)")

    # cross-check against our own pools' existing labels: does PM2/BS1 agree with ClinVar/ENIGMA?
    for pool_name in ["internal_pool", "external_pool"]:
        pool = pd.read_csv(f"pipeline/{pool_name}.csv", low_memory=False)[["chr", "pos", "ref", "alt", "label"]]
        pool = pool.rename(columns={"chr": "chr_hg38", "pos": "pos_hg38"})
        pool["chr_hg38"] = pool["chr_hg38"].astype(str)
        evidence["chr_hg38"] = evidence["chr_hg38"].astype(str)
        merged = pd.merge(evidence, pool, on=["chr_hg38", "pos_hg38", "ref", "alt"], how="inner")
        print(f"\n=== {pool_name} cross-check (n={len(merged)}) ===")
        print(pd.crosstab(merged["label"].map({0: "benign", 1: "pathogenic"}),
                           merged[["PM2_Kazakh_Supporting", "BS1_Kazakh"]].apply(
                               lambda r: "PM2" if r["PM2_Kazakh_Supporting"] else ("BS1" if r["BS1_Kazakh"] else "neither"),
                               axis=1)))


if __name__ == "__main__":
    main()
