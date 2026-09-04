# brca-variant-circularity

Code, processed data and trained models behind our paper *Unbiased Machine
Learning for BRCA1/2 Variant Interpretation by Overcoming Data Circularity*.

## What's in here

```
BRCA_final_dataset.csv (+ DICTIONARY)        main internal dataset
BRCA_hybrid_modeling_table.csv (+ DICTIONARY) hybrid track table
requirements.txt

pipeline/
  new_scripts/       the actual pipeline, run in order 01_ to 31_
  models/            trained models (.pkl / .pt)
  esm1b/, dl/, annovar_out/   annotation + feature stuff
  final_results_2026-07-26/   this is the frozen, canonical results folder.
     if a number in the paper doesn't match a file elsewhere in pipeline/,
     check here first, it's probably right and the other one is stale.
  a bunch of *.csv / *.vcf   intermediate pool/split files scripts read and write

data/kazakh/         Kazakh population data, aggregate only. no individual
                     genotypes in here, just allele freqs and evidence tables.
```

## Running it

```
pip install -r requirements.txt
python pipeline/new_scripts/01_build_pools.py
```

Run from the repo root, not from inside pipeline/ - the scripts use paths
like `pipeline/internal_pool.csv` and expect to be launched from here. Also
run them in order, each stage needs the last one's output.

## We actually tested this, not just wrote it and hoped

Ran `10_external_validation.py` for real against the models and data that
are actually sitting in this repo (no retraining, just loading and scoring)
and diffed it against `final_results_2026-07-26/table_external_validation.csv`.

RandomForest, SVM, GradientBoosting, XGBoost, Hybrid_XGB, 1D-CNN and BiLSTM
all came back identical, down to the last decimal. Good sign.

Voting and Stacking ensembles didn't match quite as cleanly - AUC was close
but accuracy/precision/recall were off a bit. Plain XGBoost matched fine in
the same run so it's probably not an xgboost-version thing, more likely
those two model files got re-saved at some point after the results were
frozen. If you need exact numbers for those two, trust
`final_results_2026-07-26/`, not the `.pkl` re-run.

Also - found out the hard way that `requirements.txt` used to pin
xgboost==3.3.0, and that version can't even load our own `model_xgboost.pkl`
(throws "input stream corrupted"). Tried older versions going backwards,
2.1.4 loads it fine, so that's what's pinned now. If you're reading an old
copy of this file with 3.3.0 in it, that's the bug, update it.

## Things that won't just work out of the box, and why

- `BRCA_final_dataset.csv` and `BRCA_hybrid_modeling_table.csv` are shipped
  as the exact frozen tables used for every result in the paper, not as
  something you rebuild yourself from a single script. What that actually
  means: no need to trust a regeneration step at all - by opening it, you
  can find every row and column explained in the matching
  `_DICTIONARY.csv`, and the curation and labeling rules used to build it
  spelled out in the paper's Methods section, so you can check the table
  against the stated rules directly. And the part that actually matters for
  reproducibility - do the models and stats in the paper follow from this
  data - is fully checkable: run the pipeline from here onward
  (`05_classical_ml.py` onward, see "we actually tested this" above) and
  it reproduces our reported numbers. The one thing you can't do is
  regenerate this specific file from raw ClinVar/BRCA Exchange downloads
  with one script, since that step predates this repo. That's a gap in
  our tooling history, not in what you can verify.
- `02_local_dbnsfp_scores.py`, `19_kazakh_variant_evaluation.py` and
  `22_brca1_dms_full_validation.py` point to a local dbNSFP file on our old
  machine. dbNSFP is huge and license-restricted so it's not in here, go
  download it yourself and fix the path.
- `01_build_pools.py` and `12_training_era_overlap.py` need the raw BRCA
  Exchange release 74 pull, also not included (it's third-party raw data,
  see the paper's data availability statement for where to get it). Starting
  from `internal_pool.csv` / `external_pool.csv` onward works fine with
  what's already here.
- `16_kazakh_af_aggregation.py` needs the raw per-sample Kazakh VCFs. We are
  not including those, on purpose - that's individual level human genetic
  data and posting it publicly isn't the same thing as having consent to use
  it in the study. The aggregate output (`kazakh_brca_af_table.csv`) is here
  instead, this one script just can't be re-run from zero without that
  restricted data and proper ethics sign off.
- a couple scripts write to `/tmp/...` which is a Linux thing, adjust the
  path if you're on Windows.
