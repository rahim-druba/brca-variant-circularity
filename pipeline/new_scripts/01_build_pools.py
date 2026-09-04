"""
Phase 1 of the BRCA1/2 reproduction pipeline (see chat_history.md and
/home/rahim/.claude/plans/dynamic-moseying-leaf.md for full context).

Builds the internal (ClinVar-only) and external (ENIGMA-only) variant pools
from our own primary source, release-11-11-25/output/variants_output.tsv
(BRCA Exchange release 74, MD5-verified). Zero overlap by construction:
a variant with any ENIGMA call goes to the external pool and is excluded
from internal, regardless of what ClinVar says about it.

Label rule: parse each comma-separated ClinVar/ENIGMA submission term
individually (fixes issue #6 in chat_history.md -- naive whole-string
substring match is fragile against composite strings like
"Likely_benign,Uncertain_significance"). A variant with both a
pathogenic-family and benign-family term present is treated as conflicting
and dropped, matching the existing "exclude conflicting/uncertain/VUS"
label policy from the original methodology (chat_history.md SS3).

2026-07-24 update: variants_output.tsv has ~309 columns; the original version
of this script only read 7. Extended to also carry through columns that were
sitting unused but directly matter for ACMG evidence beyond predictor scores:
BayesDel (already computed by BRCA Exchange, no dbNSFP lookup needed), SpliceAI,
the official pre-computed ACMG population-frequency evidence code, IARC 5-tier
fields, real functional assay results (Findlay/Mesman/Bouwman/Richardson/
Ikegami/Petitalot, including a Starita saturation-genome-editing column), and
gnomAD AF split into East Asian sub-populations (Japanese/Korean/other) --
relevant to the Kazakh-cohort proxy-population work since East Asian is the
best-supported proxy where direct Kazakh AF is unavailable. These are all
carried through as extra columns, not yet used for labels/decisions -- that
remains a separate call once coverage on our actual 26,608 variants is known.
"""
import pandas as pd

SRC = "release-11-11-25/output/variants_output.tsv"

PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic", "likely pathogenic"}
BENIGN_TERMS = {"benign", "likely_benign", "likely benign"}
IGNORE_TERMS = {"none", "not_provided", "not provided", "", "-"}

# Extra columns beyond the original 7 -- carried through, not yet used for any
# decision. Column 265 ("the provisional acmg population frequency evidence
# code, by the vcep rules") and column 11 (its accompanying free-text reason)
# have awkward literal header names in the source file -- kept verbatim here.
EXTRA_COLS = [
    "bayesdel_noaf",
    "ds_ag_spliceai", "ds_al_spliceai", "ds_dg_spliceai", "ds_dl_spliceai",
    "result_spliceai",
    "the provisional acmg population frequency evidence code, by the vcep rules",
    "details on how the provisional acmg popfreq evidence code was determined   "
    "\"this variant is absent from gnomad v2.1 (exomes only, non-cancer subset, "
    "read depth ≥25) and gnomad v3.1 (non-cancer subset, read depth ≥25) "
    "(pm2_supporting met)\"",
    "iarc_class_exlovd", "pathogenicity_prior", "posterior_probability_exlovd",
    "combined_prior_probablility_exlovd", "missense_analysis_prior_probability_exlovd",
    "result_starita_enigma_brca12_functional_assays",
    "result_findlay_enigma_brca12_functional_assays",
    "rna_score_findlay_enigma_brca12_functional_assays",
    "hdr_mesman_enigma_brca12_functional_assays",
    "hdr_richardson_enigma_brca12_functional_assays",
    "result_mesman_enigma_brca12_functional_assays",
    "result_bouwman1_enigma_brca12_functional_assays",
    "result_bouwman2_enigma_brca12_functional_assays",
    "result_ikegami_enigma_brca12_functional_assays",
    "result_petitalot_enigma_brca12_functional_assays",
    "allele_frequency_genome_eas_gnomadv2", "allele_frequency_genome_eas_gnomadv3",
    "allele_frequency_genome_eas_jpn_gnomadv2", "allele_frequency_genome_eas_kor_gnomadv2",
    "allele_frequency_genome_eas_oea_gnomadv2",
    "allele_frequency_exome_eas_gnomadv2", "allele_frequency_exome_eas_jpn_gnomadv2",
    "allele_frequency_exome_eas_kor_gnomadv2", "allele_frequency_exome_eas_oea_gnomadv2",
    "faf95_popmax_exome_gnomadv2", "faf95_popmax_genome_gnomadv2",
    "faf95_popmax_genome_gnomadv3",
]


def classify(raw):
    if pd.isna(raw):
        return None
    terms = [t.strip().lower() for t in str(raw).split(",")]
    terms = [t for t in terms if t not in IGNORE_TERMS]
    if not terms:
        return None
    is_path = any(t in PATHOGENIC_TERMS for t in terms)
    is_ben = any(t in BENIGN_TERMS for t in terms)
    if is_path and is_ben:
        return None  # conflicting calls within the same source
    if is_path:
        return 1
    if is_ben:
        return 0
    return None  # only uncertain/VUS-type terms


def to_vcf(df, path):
    out = pd.DataFrame({
        "#CHROM": df["chr"].astype(str),
        "POS": df["pos"].astype(int),
        "ID": ".",
        "REF": df["ref"],
        "ALT": df["alt"],
        "QUAL": ".",
        "FILTER": ".",
        "INFO": ".",
    })
    out = out.sort_values(["#CHROM", "POS"])
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        out.to_csv(f, sep="\t", index=False)


RENAME_EXTRA = {
    "the provisional acmg population frequency evidence code, by the vcep rules":
        "acmg_popfreq_evidence_code",
    "\"details on how the provisional acmg popfreq evidence code was determined   "
    "\"\"this variant is absent from gnomad v2.1 (exomes only, non-cancer subset, "
    "read depth ≥25) and gnomad v3.1 (non-cancer subset, read depth ≥25) "
    "(pm2_supporting met)\"\"\"":
        "acmg_popfreq_evidence_reason",
}


def main():
    base_cols = ["gene_symbol", "chr", "pos", "ref", "alt",
                 "clinical_significance_enigma", "clinical_significance_clinvar"]
    df = pd.read_csv(
        SRC, sep="\t", low_memory=False,
        usecols=base_cols + EXTRA_COLS,
    )
    df = df.rename(columns=RENAME_EXTRA)
    print(f"Loaded {len(df)} total variants from {SRC}")

    print("\nCoverage of new Stage 1b columns (% non-null across all variants):")
    for col in [RENAME_EXTRA.get(c, c) for c in EXTRA_COLS]:
        cov = df[col].notna().mean() * 100
        print(f"  {col:55s} {cov:5.1f}%")

    df["enigma_label"] = df["clinical_significance_enigma"].apply(classify)
    df["clinvar_label"] = df["clinical_significance_clinvar"].apply(classify)

    has_enigma_call = df["enigma_label"].notna()
    has_clinvar_call = df["clinvar_label"].notna()

    external = df[has_enigma_call].copy()
    external["label"] = external["enigma_label"].astype(int)
    external["label_source"] = "enigma"

    internal = df[has_clinvar_call & ~has_enigma_call].copy()
    internal["label"] = internal["clinvar_label"].astype(int)
    internal["label_source"] = "clinvar"

    # Zero-overlap-by-construction check, not an assumption
    key_cols = ["chr", "pos", "ref", "alt"]
    overlap = pd.merge(internal[key_cols], external[key_cols], on=key_cols, how="inner")
    print(f"\nInternal pool (ClinVar-labeled, no ENIGMA call): {len(internal)}")
    print(internal["label"].value_counts().rename({1: "Pathogenic", 0: "Benign"}))
    print(f"\nExternal pool (ENIGMA-labeled): {len(external)}")
    print(external["label"].value_counts().rename({1: "Pathogenic", 0: "Benign"}))
    print(f"\nOverlap between pools by Chr/Start/Ref/Alt: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "Pools are not disjoint -- construction bug"

    extra_out_cols = [RENAME_EXTRA.get(c, c) for c in EXTRA_COLS]
    keep_cols = ["gene_symbol", "chr", "pos", "ref", "alt", "label", "label_source"] + extra_out_cols
    internal[keep_cols].to_csv("pipeline/internal_pool.csv", index=False)
    external[keep_cols].to_csv("pipeline/external_pool.csv", index=False)
    to_vcf(internal, "pipeline/internal_to_annotate.vcf")
    to_vcf(external, "pipeline/external_to_annotate.vcf")
    print("\nWrote pipeline/internal_pool.csv, pipeline/external_pool.csv, "
          "pipeline/internal_to_annotate.vcf, pipeline/external_to_annotate.vcf")


if __name__ == "__main__":
    main()
