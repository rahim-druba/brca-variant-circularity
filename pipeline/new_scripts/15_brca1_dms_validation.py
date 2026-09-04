"""
Stage 8 (continued): BRCA1 deep mutational scanning validation, per Mahmood
et al. 2017's recommendation (articles/, Tier 5) that a functional-assay-
grounded, ClinVar-independent check is more meaningful than another
clinically-curated external set.

Data: Findlay GM, Daza RM, Martin B, Zhang MD, Leith AP, Gasperini M,
Janizek JD, Huang X, Starita LM, Shendure J. Accurate classification of
BRCA1 variants with saturation genome editing. Nature. 2018;562(7726):
217-222. DOI 10.1038/s41586-018-0461-z. PMID 30209399. -- saturation genome
editing at the endogenous BRCA1 locus, ~3893 SNVs across critical exons,
continuous functional fitness score (more negative = more damaging to HDR
function/cell survival). Fetched directly from MaveDB's public REST API
(urn:mavedb:00000097-0-2) -- confirmed via its own metadata this is exactly
the Findlay et al. 2018 dataset, not assumed from the URN alone.

Data platform (cite alongside the paper, per standard practice for any
database/repository use): Esposito D, Weile J, Shendure J, Starita LM,
Papenfuss AT, Roth FP, Fowler DM, Rubin AF. MaveDB: an open-source platform
to distribute and interpret data from multiplexed assays of variant effect.
Genome Biol. 2019;20:223. DOI 10.1186/s13059-019-1845-6. See
articles/citation_log.md's Batch 6 for full documentation, including why
neither PDF could be saved locally (PMC's JS proof-of-work challenge blocks
non-browser downloads; outbound FTP is blocked by this sandbox) despite a
genuine attempt, and how the API access itself was found.

NOT re-downloaded/cached locally by this script; re-run to refetch, or see
/tmp/brca1_sge_scores.csv for the raw pull used this session (not
version-controlled -- a /tmp path, not part of the repo).

Join: Findlay's hgvs_nt is HGVS-standard c. notation on NM_007294.3 (e.g.
"c.5565A>T"); our own ANNOVAR-derived AAChange.refGeneWithVer uses a
different c. notation ORDER (e.g. "c.A5565T") for the same NM_007294
transcript (a different version, .4 vs .3 -- not assumed identical without
checking; spot-checked several matches by position and both ref/alt bases
agree, so the coding-sequence numbering is unaffected by the version bump).
Restricted to simple substitutions only (both regex patterns require a
single ref/alt base) -- indels/complex delins variants excluded from this
join, a real coverage limitation, not a silent gap.

Scope note: this only validates against the subset of Findlay's ~3893
variants that ALSO already exist in our own ClinVar-derived internal pool
(481 variants) -- NOT the full SGE set, most of which are novel VUS from a
ClinVar perspective (exactly the more valuable use case Mahmood highlighted,
but running our full dbNSFP annotation pipeline on ~3400 additional variants
not currently in our pool is a bigger task, scoped out of this pass).
"""
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, mannwhitneyu

device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]


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


def parse_std(c):
    m = re.match(r"^c\.(\d+)([ACGT])>([ACGT])$", c)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def parse_annovar(c):
    m = re.match(r"^c\.([ACGT])(\d+)([ACGT])$", c)
    return (int(m.group(2)), m.group(1), m.group(3)) if m else None


def extract_nm7294_c(aachange):
    if not isinstance(aachange, str) or aachange in (".", "nan"):
        return None
    for part in aachange.split(","):
        fields = part.split(":")
        if len(fields) < 3:
            continue
        if fields[1].startswith("NM_007294"):
            for f in fields:
                if f.startswith("c."):
                    return f
    return None


def main():
    sge = pd.read_csv("/tmp/brca1_sge_scores.csv")
    sge = sge[sge["hgvs_nt"].str.startswith("NM_007294.3:", na=False)].copy()
    sge["c_notation"] = sge["hgvs_nt"].str.replace("NM_007294.3:", "", regex=False)
    sge["key"] = sge["c_notation"].apply(lambda c: parse_std(c) if isinstance(c, str) else None)
    sge = sge[sge["key"].notna()]
    print(f"Findlay SGE simple substitutions: {len(sge)}")

    df = pd.read_csv("pipeline/internal_final_annotated.csv", low_memory=False)
    brca1 = df[df["gene_symbol"] == "BRCA1"].copy()
    brca1["c_notation"] = brca1["AAChange.refGeneWithVer"].apply(extract_nm7294_c)
    brca1["key"] = brca1["c_notation"].apply(lambda c: parse_annovar(c) if isinstance(c, str) else None)
    brca1 = brca1[brca1["key"].notna()]

    overlap = pd.merge(brca1, sge, on="key", how="inner", suffixes=("_ours", "_sge"))
    print(f"Overlap with our ClinVar-derived internal pool: {len(overlap)}")
    print(f"  label balance in overlap: {overlap['label'].value_counts().to_dict()}")

    X = overlap[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    complete = X.notna().all(axis=1)
    print(f"  complete-case (all 13 scores present): {complete.sum()}")
    overlap = overlap[complete].reset_index(drop=True)
    X = X[complete].reset_index(drop=True)

    preproc = joblib.load("pipeline/models/model_hybrid_b_preproc.pkl")
    xgb_final = joblib.load("pipeline/models/model_xgboost.pkl")
    meta_final = joblib.load("pipeline/models/model_hybrid_b_meta.pkl")
    mlp_final = MLP(len(SCORE_COLS)).to(device)
    mlp_final.load_state_dict(torch.load("pipeline/models/model_hybrid_b_mlp.pt", map_location=device))

    X_arr = preproc["scaler"].transform(preproc["imputer"].transform(X))
    p_xgb = xgb_final.predict_proba(X)[:, 1]
    p_mlp = mlp_predict_proba(mlp_final, X_arr)
    overlap["hybrid_b_proba"] = meta_final.predict_proba(np.column_stack([p_xgb, p_mlp]))[:, 1]

    rho, pval = spearmanr(overlap["hybrid_b_proba"], overlap["score"])
    print(f"\nSpearman correlation (Hybrid-B probability vs Findlay functional score): "
          f"rho={rho:.4f}, p={pval:.2e}")
    print("(negative rho expected: higher predicted pathogenicity <-> more negative/damaging functional score)")

    path = overlap[overlap["label"] == 1]["score"]
    benign = overlap[overlap["label"] == 0]["score"]
    if len(path) >= 5 and len(benign) >= 5:
        u, p = mannwhitneyu(path, benign, alternative="less")
        print(f"\nOur pathogenic-labeled variants have LOWER (more damaging) Findlay scores "
              f"than our benign-labeled variants: Mann-Whitney U p={p:.2e}")
        print(f"  pathogenic (n={len(path)}): median Findlay score = {path.median():.3f}")
        print(f"  benign     (n={len(benign)}): median Findlay score = {benign.median():.3f}")

    overlap[["chr", "pos", "ref", "alt", "label", "hybrid_b_proba", "score"]].to_csv(
        "pipeline/table_brca1_dms_validation.csv", index=False)
    print("\nSaved to pipeline/table_brca1_dms_validation.csv")


if __name__ == "__main__":
    main()
