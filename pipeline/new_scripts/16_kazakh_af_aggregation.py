"""
Stage 9: aggregate a Kazakh population allele-frequency table for BRCA1/BRCA2
from the 224 per-sample genotyping-array VCFs (Serikzhan et al., accessed via
ENA/EVA -- see articles/citation_log.md's Kazakh-data section for how these
were located and verified, and docs/memory/project_kazakh_reference.md for
the broader Kazakh-reference context). Files already downloaded locally to
2/data/serikzhan_kazakh_vcfs/ (224 files, ~3GB, integrity-verified earlier
this session) -- no network access needed for this script.

2026-07-26 real finding, not previously caught: these VCFs are GRCh37/hg19
(##reference=file://GRCh37_genome.fa in the header; chr1 contig length
249250621 is the exact GRCh37 primary-assembly length, not hg38's
248956422), NOT hg38 like the rest of this pipeline. An earlier session
claim ("Test fetch of one sample's BRCA1 region returned 22 genotyped
positions") did not record which coordinates were used for that fetch, and
if hg38 BRCA1/BRCA2 coordinates were used against this hg19 data, that
fetch queried the wrong genomic window entirely by ~1.8Mb (chr17) / ~575kb
(chr13). Caught here by checking the header explicitly rather than assuming
build continuity with the rest of the project. Handled by lifting every
matched position from hg19 to hg38 via pyliftover + UCSC's official
hg19ToHg38.over.chain.gz (fetched from
https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/, cached at
/tmp/hg19ToHg38.over.chain.gz -- a standard, citable UCSC reference resource,
not a data source requiring separate literature citation). Sanity-checked:
lifting hg19 BRCA1/BRCA2 gene-start coordinates lands within a few kb of this
pipeline's own known hg38 region starts (chr17:43039371, chr13:32314843, from
09_hybrid.py's REGIONS dict) -- consistent with a correct liftover, not
coincidence.

No .csi/.tbi index files exist for these VCFs, so region-restricted pysam
.fetch() isn't available; instead shells out to `zcat | awk` per file to
cheaply pre-filter to chr13/chr17 rows (avoids parsing ~650K lines/file x 224
files = ~145M lines in pure Python) before doing exact position-window
filtering and genotype tallying in Python.

Scope reminder (already documented, repeated here): this is a fixed-content
SNP genotyping array (Illumina GSA), not WGS -- only a limited, pre-selected
set of common positions per gene are genotyped. Good for a general Kazakh
population background AF feature; very unlikely to cover the specific rare
pathogenic variants in the Zhunussova/Samigatova/Akilzhanova cohorts.
"""
import subprocess
import pandas as pd
from pyliftover import LiftOver

VCF_DIR = "2/data/serikzhan_kazakh_vcfs"
CHAIN_FILE = "/tmp/hg19ToHg38.over.chain.gz"

# hg19 RefSeq gene boundaries, generous +/-5kb padding for regulatory/UTR probes
REGIONS_HG19 = {
    "BRCA1": ("17", 41_191_312, 41_282_500),
    "BRCA2": ("13", 32_884_617, 32_978_809),
}


def scan_sample(vcf_path, chroms):
    """zcat|awk pre-filter to the needed chromosomes, then parse genotypes
    in Python. Returns list of (chrom, pos, ref, alt, gt) tuples."""
    awk_prog = " || ".join(f'$1=="{c}"' for c in chroms)
    cmd = f'zcat "{vcf_path}" | awk \'!/^#/ && ({awk_prog})\''
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) < 10:
            continue
        chrom, pos, _id, ref, alt, _qual, filt, _info, fmt, sample = fields[:10]
        if filt not in ("PASS", "."):
            continue
        if "," in alt:  # multi-allelic array probes are rare/unreliable here, skip
            continue
        gt_idx = fmt.split(":").index("GT")
        gt = sample.split(":")[gt_idx]
        rows.append((chrom, int(pos), ref, alt, gt))
    return rows


def gt_to_alt_count(gt):
    gt = gt.replace("|", "/")
    if gt in ("./.", ".", ""):
        return None
    alleles = gt.split("/")
    try:
        return sum(int(a) for a in alleles)
    except ValueError:
        return None


def main():
    import glob
    vcf_files = sorted(glob.glob(f"{VCF_DIR}/*.vcf.gz"))
    print(f"Found {len(vcf_files)} sample VCFs")

    chroms_needed = sorted({r[0] for r in REGIONS_HG19.values()})
    tally = {}  # (chrom, pos, ref, alt) -> [n_called, alt_allele_count]

    for i, path in enumerate(vcf_files):
        if i % 25 == 0:
            print(f"  scanning sample {i+1}/{len(vcf_files)}...")
        rows = scan_sample(path, chroms_needed)
        for chrom, pos, ref, alt, gt in rows:
            in_region = any(
                chrom == c and lo <= pos <= hi for c, lo, hi in REGIONS_HG19.values()
            )
            if not in_region:
                continue
            key = (chrom, pos, ref, alt)
            alt_count = gt_to_alt_count(gt)
            if key not in tally:
                tally[key] = [0, 0]
            if alt_count is not None:
                tally[key][0] += 1
                tally[key][1] += alt_count

    print(f"\n{len(tally)} unique BRCA1/BRCA2-region positions genotyped across all samples")

    lo = LiftOver(CHAIN_FILE)
    rows_out = []
    for (chrom, pos, ref, alt), (n_called, alt_count) in tally.items():
        gene = next(g for g, (c, start, end) in REGIONS_HG19.items() if c == chrom and start <= pos <= end)
        lifted = lo.convert_coordinate(f"chr{chrom}", pos - 1)  # pyliftover uses 0-based input
        pos_hg38 = lifted[0][1] + 1 if lifted else None
        af = alt_count / (2 * n_called) if n_called else float("nan")
        rows_out.append({
            "gene": gene, "chr_hg19": chrom, "pos_hg19": pos, "chr_hg38": "17" if gene == "BRCA1" else "13",
            "pos_hg38": pos_hg38, "ref": ref, "alt": alt,
            "n_samples_genotyped": n_called, "alt_allele_count": alt_count,
            "kazakh_af": round(af, 5),
        })

    out = pd.DataFrame(rows_out).sort_values(["gene", "pos_hg19"])
    out.to_csv("data/kazakh/kazakh_brca_af_table.csv", index=False)
    print(f"\nSaved {len(out)} rows to data/kazakh/kazakh_brca_af_table.csv")
    print(f"  liftover failures (pos_hg38 is null): {out['pos_hg38'].isna().sum()}")
    print(out.groupby("gene").size())
    print("\nAF distribution:")
    print(out["kazakh_af"].describe())


if __name__ == "__main__":
    main()
