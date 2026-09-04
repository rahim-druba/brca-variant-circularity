"""
Phase 2 (part 2): merge ANNOVAR gene-context columns + local dbNSFP's
10 dbNSFP scores + our own locally-computed ESM1b scores into one final
annotated table per pool. EVE is dropped (evemodel.org's own API is down
-- verified 502/504 from their server directly, not a client-side issue;
see chat_history.md SS17).

2026-07-25 fix: this script previously read scores from
pipeline/{pool}_annotated_myvariant.csv (the old myvariant.info API source),
which meant re-running this merge would silently reproduce the wrong-isoform
bug already fixed elsewhere (MetaRNN/AlphaMissense/VEST4 resolved via
max-across-all-transcripts instead of the canonical/MANE-Select transcript --
see chathistory.md for the full quantification: differed for 61.5%/64.9% of
AlphaMissense/VEST4-scored variants). Switched to
pipeline/{pool}_annotated_local.csv (new_scripts/02_local_dbnsfp_scores.py's
output), which resolves to the canonical transcript correctly. Feature SET is
unchanged here -- still the same 10 baseline scores + ESM1b, not the new
candidate columns (Eigen/EigenPC/BayesDel/popEVE) that file also carries.
Whether to add those is a separate, not-yet-decided choice (Stage 4).

ESM1b matching: ANNOVAR's AAChange.refGeneWithVer already reports amino
acid changes in 1-letter code (e.g. "p.P2L"), which is exactly ESM1b's
own mut_name format, so no code translation is needed -- just extract
the canonical-transcript entry per row and match by (gene, mut_name).
Canonical transcripts confirmed by matching against UniProt sequence
length: BRCA1 = NM_007294 (1863 aa, matches P38398), BRCA2 = NM_000059
(3418 aa, matches P51587, and BRCA2 only has this one transcript in
refGeneWithVer anyway).
"""
import re
import pandas as pd

CANONICAL_TX = {"BRCA1": "NM_007294", "BRCA2": "NM_000059"}


def extract_canonical_aachange(aachange, gene):
    if pd.isna(aachange) or aachange == ".":
        return None
    tx = CANONICAL_TX.get(gene)
    if tx is None:
        return None
    for part in aachange.split(","):
        fields = part.split(":")
        if len(fields) < 5:
            continue
        g, transcript = fields[0], fields[1]
        if g == gene and transcript.startswith(tx):
            aa = fields[4]  # "p.P2L"
            return aa[2:] if aa.startswith("p.") else aa
    return None


def load_esm1b_scores():
    """esm_score is the RAW log-likelihood-ratio from the model -- lower means
    damaging in that convention, the opposite of every other score in this
    project. Negated here so higher=damaging, consistent with the rest.

    2026-07-25 fix: this negation was previously missing here. It had been
    applied manually somewhere between this script's output and the final
    BRCA_final_dataset.csv (undocumented, untraced), producing the correct
    AUC (0.922, verified 2026-07-25) in the final dataset while this script's
    own OWN output (pipeline/*_final_annotated.csv) stayed silently wrong
    (AUC 0.078 -- an almost-perfectly-inverted classifier). Anyone re-running
    this script before this fix would have reproduced the broken version.
    """
    df = pd.read_csv("pipeline/esm1b/esm1b_scores.csv")
    df["gene"] = df["seq_id"].apply(lambda s: "BRCA1" if "BRCA1" in s else "BRCA2")
    df["ESM1b_score"] = -df["esm_score"]
    df = df[["gene", "mut_name", "ESM1b_score"]]
    return df


def merge_pool(pool_name, esm1b):
    anno = pd.read_csv(f"pipeline/annovar_out/{pool_name}.hg38_multianno.txt", sep="\t", low_memory=False)
    anno = anno.rename(columns={"Chr": "chr", "Start": "pos", "Ref": "ref", "Alt": "alt"})
    scores = pd.read_csv(f"pipeline/{pool_name}_annotated_local.csv")

    merged = pd.merge(
        scores,
        anno[["chr", "pos", "ref", "alt", "Func.refGeneWithVer", "Gene.refGeneWithVer",
              "ExonicFunc.refGeneWithVer", "AAChange.refGeneWithVer"]],
        on=["chr", "pos", "ref", "alt"], how="left",
    )

    merged["aachange_1letter"] = merged.apply(
        lambda r: extract_canonical_aachange(r["AAChange.refGeneWithVer"], r["gene_symbol"]), axis=1
    )
    merged = pd.merge(
        merged, esm1b.rename(columns={"gene": "gene_symbol", "mut_name": "aachange_1letter"}),
        on=["gene_symbol", "aachange_1letter"], how="left",
    )

    merged.to_csv(f"pipeline/{pool_name}_final_annotated.csv", index=False)

    score_cols = ["ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
                  "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
                  "ESM1b_score"]
    print(f"\n{pool_name} pool ({len(merged)} variants) -- final 11-score coverage %:")
    for c in score_cols:
        print(f"  {c:22s} {merged[c].notna().mean()*100:5.1f}%")
    return merged


if __name__ == "__main__":
    esm1b = load_esm1b_scores()
    print(f"Loaded {len(esm1b)} ESM1b variant scores (both genes, all possible AA substitutions)")
    for pool in ["internal", "external"]:
        merge_pool(pool, esm1b)
