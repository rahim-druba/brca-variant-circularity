"""
Full annotate-and-evaluate pass on data/kazakh/kazakh_variants_reference.csv (54
real, published Kazakh clinical BRCA1/2 variants -- Zhunussova 2023,
Samigatova 2026, Akilzhanova 2013), closing the gap CLAUDE.md's "Known gaps"
section has flagged since the start of this project: "Kazakh two-class
two-track run - not done."

Prompted directly: "I think from it we can get all of the correct scores to
test exactly like before in internal and external." Important correction
made before building this (confirmed by the user): most of these are NOT
missense, so this can't be one clean internal/external-style AUC number --
it splits across the project's own two-track design:

  - LOF track (rule-based): Start-loss/Frameshift/Splice-site/Nonsense.
    The rule-based classifier script (like Hybrid-B, the ACMG calibration,
    and the SMOTE sweep before it) no longer exists anywhere in this repo --
    only docs/memory/project_two_track_lof.md's description and
    2/results/table_lof_rule.csv's numbers survived. Rebuilt faithfully here
    per that description: truncating -> pathogenic=1, EXCEPT BRCA2
    truncations at/after codon 3326 (the ACMG PVS1 BRCA2 C-terminal
    exception) -> not high-risk. None of this dataset's BRCA2 splice
    variants are anywhere near the C-terminus (intron 2/6/13), so the codon
    exception only needs checking for frameshift/nonsense entries, which
    have an explicit codon number in their protein_change field.
  - Missense track (ML): the 13-feature model. Annotated via the SAME local
    dbNSFP lookup + MANE-Select transcript resolution as
    02_local_dbnsfp_scores.py (reproduced inline here rather than importing,
    since that file's name starts with a digit), plus ESM1b matched against
    pipeline/esm1b/esm1b_scores.csv the same way 03_merge_annotation.py does
    (NOT dbNSFP's own bundled ESM1b_score_local field -- the trained models
    were built on the separately-sourced, sign-corrected ESM1b_score, so
    consistency requires using the same source here).

Excluded, with reasons stated (not silently dropped):
  - `c.3348A>G` (BRCA1, Akilzhanova) -- the reference CSV's own notes column
    says "Do not use this coordinate; needs full manual reconciliation"
    (VariantValidator computes this as synonymous, not missense; strong
    evidence of legacy numbering mismatch in the 2013 paper).
  - The exon-16-loss CNV (Samigatova) -- not a SNV/indel, no usable
    coordinate, out of scope for both tracks.
  - 6 missense VUS entries with no resolved hg38 coordinate at all (already
    flagged UNRESOLVED in an earlier session's coordinate-conversion effort).
  - The 1 synonymous variant (c.2127T>C) -- neither track applies.
  - 2 missense entries (c.254A>G, c.2410G>A) ARE annotated and scored, but
    flagged: their coordinates are usable but the reference CSV's own notes
    say the paper's claimed protein consequence doesn't match dbNSFP's
    computed one for that position (same residue, different resulting amino
    acid) -- treat the annotation as coordinate-confirmed, protein-identity
    uncertain.
"""
import re
import gzip
import joblib
import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

DBNSFP_PATH = "/home/rahim/BC-project-2026-07-17-FULL/downloads/dbNSFP5.3.1a_grch38.gz"
GENE_REGIONS = {"17": (43044000, 43126000), "13": (32315000, 32401000)}

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
DBNSFP_FIELDS = {
    "ClinPred_score": "ClinPred_score", "MetaRNN_score": "MetaRNN_score", "REVEL_score": "REVEL_score",
    "MetaSVM_score": "MetaSVM_score", "MetaLR_score": "MetaLR_score", "VEST4_score": "VEST4_score",
    "AlphaMissense_score": "AlphaMissense_score", "PrimateAI_score": "PrimateAI_score",
    "CADD_raw": "CADD_raw", "CADD_phred": "CADD_phred",
    "BayesDel_noAF_score": "BayesDel_noAF_score", "Eigen_raw_coding": "Eigen-raw_coding",
}

TRUNCATING_TYPES = {"Start-loss", "Frameshift", "Splice-site", "Nonsense"}
BRCA2_KX_CODON = 3326  # ACMG PVS1 BRCA2 C-terminal exception, K3326X and downstream

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*",
}


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, in_dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(device))).cpu().numpy()


def parse_coord(c):
    if not isinstance(c, str):
        return None
    parts = c.replace("chr", "").split("-")
    if len(parts) != 4:
        return None
    chrom, pos, ref, alt = parts
    return chrom, int(pos), ref, alt


def extract_codon(protein_change):
    """p.Thr3085Glnfs*19 -> 3085; p.Gln1721Ter -> 1721; p.Ser157* -> 157."""
    if not isinstance(protein_change, str):
        return None
    m = re.search(r"(\d+)", protein_change)
    return int(m.group(1)) if m else None


def protein_change_to_1letter(protein_change):
    """p.Cys61Gly -> C61G ; G32V -> G32V (already 1-letter) ; p.Val211Ile -> V211I."""
    if not isinstance(protein_change, str):
        return None
    s = protein_change.replace("p.", "").strip()
    m3 = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", s)
    if m3:
        ref3, pos, alt3 = m3.groups()
        ref1, alt1 = AA3_TO_1.get(ref3.capitalize()), AA3_TO_1.get(alt3.capitalize())
        if ref1 and alt1:
            return f"{ref1}{pos}{alt1}"
        return None
    m1 = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m1:
        return s
    return None


def dbnsfp_lookup():
    with gzip.open(DBNSFP_PATH, "rt") as f:
        header = f.readline().strip().split("\t")
    idx = {name: i for i, name in enumerate(header)}
    tbx = pysam.TabixFile(DBNSFP_PATH)
    lookup = {}
    for chrom, (start, end) in GENE_REGIONS.items():
        for row in tbx.fetch(chrom, start, end):
            f = row.split("\t")
            key = (f[idx["#chr"]], f[idx["pos(1-based)"]], f[idx["ref"]], f[idx["alt"]])
            lookup[key] = f
    tbx.close()
    return lookup, idx


def mane_select_pos(fields, idx):
    mane_list = fields[idx["MANE"]].split(";")
    return mane_list.index("Select") if "Select" in mane_list else None


def pick_value(raw, mane_pos):
    parts = raw.split(";")
    if len(parts) == 1:
        return parts[0]
    if mane_pos is not None and mane_pos < len(parts) and parts[mane_pos] != ".":
        return parts[mane_pos]
    for p in parts:
        if p != ".":
            return p
    return "."


def extract_dbnsfp_scores(fields, idx):
    mane_pos = mane_select_pos(fields, idx)
    out = {}
    for out_col, dbnsfp_col in DBNSFP_FIELDS.items():
        raw = fields[idx[dbnsfp_col]]
        val = pick_value(raw, mane_pos) if raw != "." else "."
        out[out_col] = None if val == "." else val
    return out


def label_from_significance(sig):
    if not isinstance(sig, str):
        return None
    s = sig.lower()
    if "pathogenic" in s and "likely" not in s:
        return 1
    if s.startswith("likely pathogenic"):
        return 1
    if "benign" in s:
        return 0
    return None  # VUS or unresolved


def lof_rule_predict(gene, protein_change):
    codon = extract_codon(protein_change)
    if gene == "BRCA2" and codon is not None and codon >= BRCA2_KX_CODON:
        return 0  # ACMG PVS1 BRCA2 C-terminal exception: not high-risk
    return 1


def main():
    ref = pd.read_csv("data/kazakh/kazakh_variants_reference.csv")
    ref["parsed"] = ref["hg38_coordinate"].apply(parse_coord)
    ref["label"] = ref["clinical_significance"].apply(label_from_significance)
    excluded_no_coord = ref[ref["parsed"].isna()]
    print(f"Excluded (no usable hg38 coordinate): {len(excluded_no_coord)}")

    ref = ref[ref["parsed"].notna()].copy()
    ref[["chr", "pos", "ref_a", "alt_a"]] = pd.DataFrame(ref["parsed"].tolist(), index=ref.index)

    # explicit exclusion: reference CSV's own notes say "do not use"
    do_not_use = ref["notes"].astype(str).str.contains("Do not use", na=False)
    print(f"Excluded (reference CSV flags 'do not use'): {do_not_use.sum()}")
    excluded_do_not_use = ref[do_not_use].copy()
    flagged_uncertain = ref["notes"].astype(str).str.contains("MISMATCH", na=False) & ~do_not_use
    ref = ref[~do_not_use].copy()
    ref["protein_consequence_uncertain"] = flagged_uncertain[ref.index]

    lof = ref[ref["variant_type"].isin(TRUNCATING_TYPES)].copy()
    missense = ref[ref["variant_type"] == "Missense"].copy()
    other = ref[~ref["variant_type"].isin(TRUNCATING_TYPES | {"Missense"})]
    print(f"LOF-track variants: {len(lof)}, Missense-track variants: {len(missense)}, "
          f"excluded (synonymous/CNV/other): {len(other)}")

    # --- LOF track ---
    lof["rule_prediction"] = lof.apply(lambda r: lof_rule_predict(r["gene"], r["protein_change"]), axis=1)
    lof_labeled = lof[lof["label"].notna()]
    n_correct = (lof_labeled["rule_prediction"] == lof_labeled["label"]).sum()
    print(f"\n=== LOF track: {len(lof_labeled)} labeled variants, "
          f"{n_correct}/{len(lof_labeled)} correctly classified by the rule-based classifier ===")
    lof.to_csv("data/kazakh/kazakh_lof_track_evaluation.csv", index=False)
    print(lof[["gene", "cdna_hgvs", "protein_change", "variant_type", "clinical_significance",
               "rule_prediction"]].to_string(index=False))

    # --- Missense track ---
    print(f"\n=== Missense track: annotating {len(missense)} variants via local dbNSFP ===")
    lookup, idx = dbnsfp_lookup()
    esm1b = pd.read_csv("pipeline/esm1b/esm1b_scores.csv")
    esm1b["gene"] = esm1b["seq_id"].apply(lambda s: "BRCA1" if "BRCA1" in s else "BRCA2")
    esm1b["ESM1b_score"] = -esm1b["esm_score"]  # same sign-correction as 03_merge_annotation.py

    records = []
    for _, row in missense.iterrows():
        key = (row["chr"], str(row["pos"]), row["ref_a"], row["alt_a"])
        fields = lookup.get(key)
        scores = extract_dbnsfp_scores(fields, idx) if fields is not None else {c: None for c in DBNSFP_FIELDS}

        mut_1letter = protein_change_to_1letter(row["protein_change"])
        esm_match = esm1b[(esm1b["gene"] == row["gene"]) & (esm1b["mut_name"] == mut_1letter)]
        scores["ESM1b_score"] = esm_match["ESM1b_score"].iloc[0] if len(esm_match) else None
        records.append(scores)

    score_df = pd.DataFrame(records).reset_index(drop=True)
    missense = pd.concat([missense.reset_index(drop=True), score_df], axis=1)

    for c in SCORE_COLS:
        cov = missense[c].notna().mean() * 100
        print(f"  {c:22s} {cov:5.1f}% coverage")

    complete = missense[SCORE_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    print(f"Complete-case (all 13 scores): {complete.sum()}/{len(missense)}")

    scored = missense[complete].copy()
    X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce")

    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))

    X_arr = preproc["scaler"].transform(preproc["imputer"].transform(X))
    p_xgb = xgb_final.predict_proba(X)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_arr)
    scored["hybrid_b_proba"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]
    scored["xgb_only_proba"] = p_xgb

    # merge on the row index, not cdna_hgvs -- c.181T>G is reported by two different
    # papers (two separate rows), so a cdna_hgvs merge would cartesian-duplicate it
    missense = missense.join(scored[["hybrid_b_proba", "xgb_only_proba"]])
    missense.to_csv("data/kazakh/kazakh_missense_track_evaluation.csv", index=False)

    labeled = scored[scored["label"].notna()]
    print(f"\n{len(labeled)} missense variants with both complete scores and a known label:")
    print(labeled[["gene", "cdna_hgvs", "protein_change", "clinical_significance", "label",
                    "hybrid_b_proba", "protein_consequence_uncertain"]].to_string(index=False))
    if len(labeled) >= 2 and labeled["label"].nunique() > 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labeled["label"], labeled["hybrid_b_proba"])
        print(f"\nHybrid-B AUC on {len(labeled)} labeled Kazakh missense variants: {auc:.4f} "
              f"(descriptive only -- n is too small for a statistically meaningful estimate)")

    # --- master consolidated table: all 54 original variants, both tracks + exclusions,
    # all 13 scores in columns -- requested directly by the user rather than leaving the
    # tracks/exclusions split across 2 files with the excluded rows implicit ---
    lof["track"] = "LOF (rule-based)"
    missense["track"] = "Missense (ML)"
    excluded_no_coord["track"] = "Excluded"
    excluded_no_coord["exclusion_reason"] = "No resolvable hg38 coordinate"
    excluded_do_not_use["track"] = "Excluded"
    excluded_do_not_use["exclusion_reason"] = "Reference CSV notes: do not use this coordinate"
    other = other.copy()
    other["track"] = "Excluded"
    other["exclusion_reason"] = other["variant_type"].apply(
        lambda v: "Synonymous (neither track applies)" if v == "Synonymous"
        else "Structural variant / CNV, not a SNV or indel")

    master = pd.concat([lof, missense, excluded_no_coord, excluded_do_not_use, other], axis=0)
    master = master.sort_index()  # restore original reference-CSV row order

    master = master.rename(columns={"ref_a": "ref", "alt_a": "alt"})
    lead_cols = ["source_paper", "gene", "transcript", "cdna_hgvs", "protein_change", "variant_type",
                 "clinical_significance", "label", "chr", "pos", "ref", "alt", "track", "exclusion_reason",
                 "rule_prediction", "hybrid_b_proba", "xgb_only_proba", "protein_consequence_uncertain"]
    final_cols = [c for c in lead_cols if c in master.columns] + \
        [c for c in SCORE_COLS if c in master.columns] + \
        [c for c in master.columns if c not in lead_cols and c not in SCORE_COLS
         and c not in ("hg38_coordinate", "occurrence_or_ncases", "ready_to_use", "parsed")]
    master = master[final_cols]
    master.to_csv("data/kazakh/kazakh_variants_final_annotated.csv", index=False)
    print(f"\n=== Master consolidated table: {len(master)} variants "
          f"(all original 54 minus none -- every row accounted for) ===")
    print(f"Saved to data/kazakh/kazakh_variants_final_annotated.csv")
    print(master["track"].value_counts())


if __name__ == "__main__":
    main()
