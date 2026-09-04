"""
Blocker #4 from the 2026-07-26 publishability assessment: "the external
validation set's circularity problem is real and currently unresolved...
a reviewer will ask 'so what's your actual clean estimate,' and there isn't
one yet."

Two paths were considered. The first -- restricting the external (ENIGMA)
pool to only variants evaluated in ClinVar AFTER predictors' training
cutoffs, per Rastogi et al.'s benchmark methodology -- was tried and found
GENUINELY INFEASIBLE with current data, not just underpowered: filtering to
post-2020 leaves 4 pathogenic and ZERO benign variants (can't compute an
AUC at all); post-2018 leaves 11 total. Verified directly before writing
this docstring, not assumed. This is a real, disclosed dead end, not a
silent one -- ENIGMA's benign pool is almost entirely old data.

The path that DOES work: extend 15_brca1_dms_validation.py's BRCA1-DMS
check from the 481-variant subset already in our ClinVar-derived pool to
the FULL ~3,893-variant Findlay et al. 2018 SGE dataset. This sidesteps the
ENIGMA-recency problem entirely -- Findlay's functional score has zero
dependence on ClinVar/ENIGMA classification, and most of these variants are
novel VUS from a clinical-database perspective (the majority never had a
chance to leak into any predictor's clinical-label training set at all).

Annotation approach, more efficient than 15_brca1_dms_validation.py's:
dbNSFP itself carries HGVSc_VEP at the MANE-Select transcript position, in
EXACTLY Findlay's own c.-notation format (e.g. "c.5592A>T") -- confirmed by
direct inspection before relying on it. This means the full ~3,893-variant
set can be annotated in a single dbNSFP region scan (reverse-indexed by its
own computed HGVSc), without needing per-variant coordinate conversion.
ESM1b needs separate handling: Findlay's own MaveDB export has hgvs_pro
100% empty (checked directly, not assumed) -- used dbNSFP's own HGVSp_VEP
at the MANE position instead, converted to 1-letter format, matched against
pipeline/esm1b/esm1b_scores.csv (the same sign-corrected source the trained
model actually saw, NOT dbNSFP's own bundled ESM1b_score field).

Unlike 15_brca1_dms_validation.py's complete-case restriction, this script
lets the model's own median imputer (already part of the saved Hybrid-B
preprocessing pipeline) handle missing features rather than dropping rows --
appropriate at this scale, and coverage is still reported transparently so
the imputation rate is visible, not hidden.
"""
import re
import gzip
import joblib
import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn
from scipy.stats import spearmanr

device = "cuda" if torch.cuda.is_available() else "cpu"
DBNSFP_PATH = "/home/rahim/BC-project-2026-07-17-FULL/downloads/dbNSFP5.3.1a_grch38.gz"
BRCA1_REGION = ("17", 43044000, 43126000)

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]
DBNSFP_FIELDS_NO_ESM1B = {
    "ClinPred_score": "ClinPred_score", "MetaRNN_score": "MetaRNN_score", "REVEL_score": "REVEL_score",
    "MetaSVM_score": "MetaSVM_score", "MetaLR_score": "MetaLR_score", "VEST4_score": "VEST4_score",
    "AlphaMissense_score": "AlphaMissense_score", "PrimateAI_score": "PrimateAI_score",
    "CADD_raw": "CADD_raw", "CADD_phred": "CADD_phred",
    "BayesDel_noAF_score": "BayesDel_noAF_score", "Eigen_raw_coding": "Eigen-raw_coding",
}

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
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


def hgvsp_to_1letter(hgvsp):
    """p.Cys61Gly -> C61G. Returns None for anything non-standard
    (stop-loss extensions, synonymous 'p.Xxx=', frameshift, etc.)."""
    if not isinstance(hgvsp, str):
        return None
    s = hgvsp.replace("p.", "")
    m = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", s)
    if not m:
        return None
    ref3, pos, alt3 = m.groups()
    ref1, alt1 = AA3_TO_1.get(ref3.capitalize()), AA3_TO_1.get(alt3.capitalize())
    return f"{ref1}{pos}{alt1}" if ref1 and alt1 else None


def build_dbnsfp_reverse_map():
    with gzip.open(DBNSFP_PATH, "rt") as f:
        header = f.readline().strip().split("\t")
    idx = {n: i for i, n in enumerate(header)}
    tbx = pysam.TabixFile(DBNSFP_PATH)
    chrom, start, end = BRCA1_REGION
    reverse_map = {}
    n_rows = 0
    for row in tbx.fetch(chrom, start, end):
        f = row.split("\t")
        n_rows += 1
        mane_list = f[idx["MANE"]].split(";")
        if "Select" not in mane_list:
            continue
        mp = mane_list.index("Select")
        hgvsc_list = f[idx["HGVSc_VEP"]].split(";")
        if mp >= len(hgvsc_list) or hgvsc_list[mp] in (".", ""):
            continue
        c_notation = hgvsc_list[mp]

        def pick(col):
            parts = f[idx[col]].split(";")
            if len(parts) == 1:
                val = parts[0]  # single-valued (gene-level) field -- no per-transcript indexing needed.
                # BUG FIXED HERE: an earlier version of this function skipped this case and
                # indexed straight by mp, silently returning "." for every single-valued field
                # (ClinPred/MetaSVM/MetaLR/PrimateAI/CADD/BayesDel all showed 0.0% coverage on
                # the first run -- caught by treating that as suspicious rather than real, since
                # dbNSFP normally has near-universal CADD coverage) -- matches the same
                # single-value short-circuit 02_local_dbnsfp_scores.py's pick_value() already has.
            else:
                val = parts[mp] if mp < len(parts) and parts[mp] != "." else next((p for p in parts if p != "."), ".")
            return None if val == "." else val

        rec = {out_col: pick(dbnsfp_col) for out_col, dbnsfp_col in DBNSFP_FIELDS_NO_ESM1B.items()}
        hgvsp_list = f[idx["HGVSp_VEP"]].split(";")
        rec["hgvsp_at_mane"] = hgvsp_list[mp] if mp < len(hgvsp_list) else None
        rec["chr"] = f[idx["#chr"]]
        rec["pos"] = f[idx["pos(1-based)"]]
        rec["ref"] = f[idx["ref"]]
        rec["alt"] = f[idx["alt"]]
        reverse_map[c_notation] = rec
    tbx.close()
    print(f"Scanned {n_rows} dbNSFP rows in the BRCA1 region, "
          f"{len(reverse_map)} unique MANE-Select HGVSc keys indexed")
    return reverse_map


def main():
    sge = pd.read_csv("/tmp/brca1_sge_scores.csv")
    sge = sge[sge["hgvs_nt"].str.startswith("NM_007294.3:", na=False)].copy()
    sge["c_notation"] = sge["hgvs_nt"].str.replace("NM_007294.3:", "", regex=False)
    print(f"Findlay SGE dataset: {len(sge)} variants")

    reverse_map = build_dbnsfp_reverse_map()
    esm1b = pd.read_csv("pipeline/esm1b/esm1b_scores.csv")
    esm1b_lookup = dict(zip(esm1b["mut_name"], -esm1b["esm_score"]))  # sign-corrected, same as elsewhere

    records = []
    matched = 0
    for _, row in sge.iterrows():
        rec = reverse_map.get(row["c_notation"])
        if rec is None:
            records.append({c: None for c in SCORE_COLS} | {"matched": False})
            continue
        matched += 1
        out = {c: rec.get(c) for c in DBNSFP_FIELDS_NO_ESM1B}
        mut_1letter = hgvsp_to_1letter(rec.get("hgvsp_at_mane"))
        out["ESM1b_score"] = esm1b_lookup.get(mut_1letter)
        out["chr"] = rec["chr"]
        out["pos"] = rec["pos"]
        out["ref"] = rec["ref"]
        out["alt"] = rec["alt"]
        out["matched"] = True
        records.append(out)

    ann = pd.concat([sge.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    print(f"Matched to a genomic position via dbNSFP HGVSc: {matched}/{len(sge)}")
    for c in SCORE_COLS:
        cov = pd.to_numeric(ann[c], errors="coerce").notna().mean() * 100
        print(f"  {c:22s} {cov:5.1f}% coverage (of full {len(ann)}-variant set)")

    scoreable = ann[ann["matched"]].copy()
    X = scoreable[SCORE_COLS].apply(pd.to_numeric, errors="coerce")

    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))

    X_arr = preproc["scaler"].transform(preproc["imputer"].transform(X))
    p_xgb = xgb_final.predict_proba(X)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_arr)
    scoreable["hybrid_b_proba"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]

    rho, pval = spearmanr(scoreable["hybrid_b_proba"], scoreable["score"])
    print(f"\n=== Full-set validation: {len(scoreable)} variants scored ===")
    print(f"Spearman correlation (Hybrid-B probability vs Findlay functional score): "
          f"rho={rho:.4f}, p={pval:.2e}")

    # how much of this is genuinely novel (not already in our ClinVar/ENIGMA-derived pools)?
    internal = pd.read_csv("pipeline/internal_pool.csv", low_memory=False)[["chr", "pos", "ref", "alt"]]
    external = pd.read_csv("pipeline/external_pool.csv", low_memory=False)[["chr", "pos", "ref", "alt"]]
    known = pd.concat([internal, external]).assign(chr=lambda d: d["chr"].astype(str))
    scoreable["chr"] = scoreable["chr"].astype(str)
    scoreable["pos"] = scoreable["pos"].astype(int)
    known["pos"] = known["pos"].astype(int)
    merged_known = pd.merge(scoreable, known.drop_duplicates(), on=["chr", "pos", "ref", "alt"], how="left", indicator=True)
    n_novel = (merged_known["_merge"] == "left_only").sum()
    print(f"Of these, {n_novel}/{len(scoreable)} ({n_novel/len(scoreable)*100:.1f}%) are NOT in our "
          f"ClinVar/ENIGMA-derived pools at all -- genuinely novel VUS from a clinical-database "
          f"perspective, the strongest form of this validation.")

    novel_only = scoreable[merged_known["_merge"].values == "left_only"]
    if len(novel_only) > 10:
        rho_novel, p_novel = spearmanr(novel_only["hybrid_b_proba"], novel_only["score"])
        print(f"Spearman correlation on the novel-only subset (n={len(novel_only)}): "
              f"rho={rho_novel:.4f}, p={p_novel:.2e}")

    scoreable.to_csv("pipeline/table_brca1_dms_full_validation.csv", index=False)
    print("\nSaved to pipeline/table_brca1_dms_full_validation.csv")


if __name__ == "__main__":
    main()
