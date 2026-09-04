"""
Usage demonstration for the Kazakh PM2/BS1 evidence codes (17_kazakh_pm2_bs1.py):
cross-check against data/kazakh/kazakh_variants_reference.csv, the 54-row
compiled reference of REAL Kazakh clinical BRCA1/2 variants from three
published cohorts (Zhunussova 2023, Samigatova 2026, Akilzhanova 2013 --
see docs/memory/project_kazakh_reference.md).

Prompted directly by "can it be used to do some test? like other variants,
if not then how are we gonna show the usage of it?" -- this answers that
concretely: the evidence table can only speak to variants at its exact
~4,160 genotyped array positions (a structural limit of fixed-content-array
data, unlike a model score which generalizes to any variant), but for
variants that DO land on the array, this is a real, checkable test against
independently-published expert classifications, not just an internal
consistency check.

Result (2026-07-26): 25/46 coordinate-ready reference variants overlap the
array. All 20 known pathogenic/likely-pathogenic variants (Zhunussova/
Samigatova) get PM2_Kazakh_Supporting (AF=0, never BS1) -- correctly absent
from a healthy population sample. Both of Akilzhanova's "likely benign,
>30% freq" polymorphisms independently get BS1_Kazakh=True, reproducing the
original paper's own expert classification without having been designed
around it. The 2 genuine VUS entries split (one BS1-leaning, one
PM2-leaning) -- the actually useful case: new evidence on unresolved
variants, not just re-confirmation of already-known ones.
"""
import pandas as pd


def parse_coord(c):
    if not isinstance(c, str):
        return None
    parts = c.replace("chr", "").split("-")
    if len(parts) != 4:
        return None
    return parts[0], int(parts[1]), parts[2], parts[3]


def main():
    ref = pd.read_csv("data/kazakh/kazakh_variants_reference.csv")
    af = pd.read_csv("data/kazakh/kazakh_pm2_bs1_evidence.csv")
    af["chr_hg38"] = af["chr_hg38"].astype(str)

    ref["parsed"] = ref["hg38_coordinate"].apply(parse_coord)
    n_with_coord = ref["parsed"].notna().sum()
    ref = ref[ref["parsed"].notna()].copy()
    ref[["chr_hg38", "pos_hg38", "ref_a", "alt_a"]] = pd.DataFrame(ref["parsed"].tolist(), index=ref.index)

    merged = pd.merge(
        ref, af, left_on=["chr_hg38", "pos_hg38", "ref_a", "alt_a"],
        right_on=["chr_hg38", "pos_hg38", "ref", "alt"], how="inner", suffixes=("_ref", "_af"),
    )
    print(f"Coordinate-ready reference variants: {n_with_coord}/{len(pd.read_csv('data/kazakh/kazakh_variants_reference.csv'))}")
    print(f"Overlap with the Kazakh array's genotyped positions: {len(merged)}/{n_with_coord}")

    def evidence_code(row):
        if row["PM2_Kazakh_Supporting"]:
            return "PM2_Kazakh_Supporting"
        if row["BS1_Kazakh"]:
            return "BS1_Kazakh"
        return "neither"

    merged["evidence"] = merged.apply(evidence_code, axis=1)
    cols = ["source_paper", "cdna_hgvs", "clinical_significance", "kazakh_af", "evidence"]
    out = merged[[c for c in cols if c in merged.columns]].sort_values("clinical_significance")
    out.to_csv("data/kazakh/kazakh_pm2_bs1_case_study.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\nSaved to data/kazakh/kazakh_pm2_bs1_case_study.csv")


if __name__ == "__main__":
    main()
