"""
Phase 2 (part 1) of the BRCA1/2 reproduction pipeline -- LOCAL replacement
for 02_myvariant_scores.py.

Replaces the myvariant.info API calls with direct local lookups against the
downloaded dbNSFP 5.3.1a GRCh38 file (downloads/dbNSFP5.3.1a_grch38.gz),
using pysam/tabix. Both BRCA1 (chr17) and BRCA2 (chr13) gene regions are
pulled once into memory, then every pool variant is matched by exact
(chr, pos, ref, alt) -- far faster than one tabix query per variant for
~26,600 variants.

Output keeps the same 9 baseline score columns as the old myvariant.info
script (for a clean diff against internal_annotated_myvariant.csv /
external_annotated_myvariant.csv), plus new candidate columns pulled from
the local file that were previously either API-unavailable or dropped
entirely: Eigen, Eigen-PC, BayesDel (addAF + noAF), ESM1b, popEVE. Whether
any of these candidates get promoted into the final feature set is a
separate decision (03_merge_annotation.py / model retraining stage), not
made here -- this script only extracts and reports coverage.

Matching is done against the MANE Select transcript position specifically
(not "any transcript in the semicolon list"), since dbNSFP lists multiple
Ensembl transcripts per row and a naive substring/membership match against
the wrong transcript can silently pick up a coincidentally-identical HGVSc
string from a different isoform. See chathistory.md / citation_log.md for
the concrete case this caught during the Kazakh-variant coordinate fix.

2026-07-25 fix: the first version of this script grabbed each score field's
raw string as-is, which is fine for single-valued columns (ClinPred, MetaSVM,
MetaLR, PrimateAI, CADD, Eigen/Eigen-PC, BayesDel) but WRONG for columns
dbNSFP reports per-transcript, semicolon-separated (MetaRNN, REVEL, VEST4,
AlphaMissense, ESM1b, popEVE) -- ~99.7% of their populated rows had an
unresolved multi-value string like "-2.52;-2.52;-2.52" instead of a single
float. Found via the Stage 3 ESM1b sign-convention check (comparing against
the old pipeline's ESM1b_score crashed on exactly this). Fixed by applying
the same MANE-Select-position selection used for the Kazakh HGVSc matching
to every field, not just the ones that happened to need it during testing.
"""
import gzip
import pandas as pd
import pysam

DBNSFP_PATH = "/home/rahim/BC-project-2026-07-17-FULL/downloads/dbNSFP5.3.1a_grch38.gz"

# hg38 gene regions with margin
GENE_REGIONS = {
    "17": (43044000, 43126000),  # BRCA1
    "13": (32315000, 32401000),  # BRCA2
}

BASELINE_FIELDS = {
    "ClinPred_score": "ClinPred_score",
    "MetaRNN_score": "MetaRNN_score",
    "REVEL_score": "REVEL_score",
    "MetaSVM_score": "MetaSVM_score",
    "MetaLR_score": "MetaLR_score",
    "VEST4_score": "VEST4_score",
    "AlphaMissense_score": "AlphaMissense_score",
    "PrimateAI_score": "PrimateAI_score",
    "CADD_raw": "CADD_raw",
    "CADD_phred": "CADD_phred",
}

CANDIDATE_FIELDS = {
    "Eigen_raw_coding": "Eigen-raw_coding",
    "Eigen_PC_raw_coding": "Eigen-PC-raw_coding",
    "BayesDel_addAF_score": "BayesDel_addAF_score",
    "BayesDel_noAF_score": "BayesDel_noAF_score",
    "ESM1b_score_local": "ESM1b_score",
    "popEVE_score": "popEVE_score",
}

ALL_FIELDS = {**BASELINE_FIELDS, **CANDIDATE_FIELDS}


def load_header():
    with gzip.open(DBNSFP_PATH, "rt") as f:
        return f.readline().strip().split("\t")


def build_lookup(header):
    """chr -> gene region -> {(chr,pos,ref,alt): row_fields}"""
    idx = {name: i for i, name in enumerate(header)}
    i_chr, i_pos, i_ref, i_alt = idx["#chr"], idx["pos(1-based)"], idx["ref"], idx["alt"]

    tbx = pysam.TabixFile(DBNSFP_PATH)
    lookup = {}
    for chrom, (start, end) in GENE_REGIONS.items():
        for row in tbx.fetch(chrom, start, end):
            f = row.split("\t")
            key = (f[i_chr], f[i_pos], f[i_ref], f[i_alt])
            lookup[key] = f
    tbx.close()
    return lookup, idx


def mane_select_pos(fields, idx):
    """Position of the MANE-Select transcript in the semicolon-separated lists,
    or None if this row has no MANE-Select entry."""
    mane_list = fields[idx["MANE"]].split(";")
    return mane_list.index("Select") if "Select" in mane_list else None


def pick_value(raw, mane_pos):
    """Resolve a possibly semicolon-separated, per-transcript field to one
    value: use the MANE-Select position if available and in range, else fall
    back to the first non-'.' entry (documented fallback, not silent)."""
    parts = raw.split(";")
    if len(parts) == 1:
        return parts[0]
    if mane_pos is not None and mane_pos < len(parts) and parts[mane_pos] != ".":
        return parts[mane_pos]
    for p in parts:
        if p != ".":
            return p
    return "."


def extract_scores(fields, idx):
    mane_pos = mane_select_pos(fields, idx)
    out = {}
    for out_col, dbnsfp_col in ALL_FIELDS.items():
        raw = fields[idx[dbnsfp_col]]
        val = pick_value(raw, mane_pos) if raw != "." else "."
        out[out_col] = None if val == "." else val
    return out


def annotate_pool(pool_name, lookup, idx):
    df = pd.read_csv(f"pipeline/{pool_name}_pool.csv")
    df["chr"] = df["chr"].astype(str)
    df["pos"] = df["pos"].astype(str)

    records = []
    for _, row in df.iterrows():
        key = (row["chr"], row["pos"], row["ref"], row["alt"])
        fields = lookup.get(key)
        if fields is None:
            records.append({col: None for col in ALL_FIELDS})
        else:
            records.append(extract_scores(fields, idx))

    score_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    out.to_csv(f"pipeline/{pool_name}_annotated_local.csv", index=False)

    print(f"\n{pool_name} pool ({len(out)} variants) -- coverage %:")
    print("  baseline (9 original fields):")
    for col in BASELINE_FIELDS:
        cov = out[col].notna().mean() * 100
        print(f"    {col:22s} {cov:5.1f}%")
    print("  candidate (new fields, not yet in the model):")
    for col in CANDIDATE_FIELDS:
        cov = out[col].notna().mean() * 100
        print(f"    {col:22s} {cov:5.1f}%")
    return out


if __name__ == "__main__":
    header = load_header()
    lookup, idx = build_lookup(header)
    print(f"Loaded {len(lookup)} unique (chr,pos,ref,alt) rows across BRCA1+BRCA2 regions")
    for pool in ["internal", "external"]:
        annotate_pool(pool, lookup, idx)
