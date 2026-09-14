# R2M-Model

**R2M: cross-modal translation of pan-cancer DNA methylation landscapes from transcriptomes**

R2M is a deep-learning model that predicts genome-wide, gene-level DNA methylation (beta values for 9,835 genes) from RNA-seq expression across 32 TCGA cancer types. It combines learnable gene identity embeddings, a cancer-type embedding, a CLIP-style contrastively trained sample encoder, and a FiLM-conditioned bilinear global decoder.

> **Manuscript:** "R2M: cross-modal translation of pan-cancer DNA methylation landscapes from transcriptomes via contrastive sample encoding and learnable gene representations" — *under review*. See [Citation](#citation).

## Key performance (held-out TCGA test set, n = 1,364)

| Metric | Value |
|---|---|
| Mean per-gene Pearson r (9,835 genes) | **0.521 ± 0.003** (3 seeds) |
| Top-100 most predictable genes | 0.621 |
| Centered per-sample r (sample-specific signal) | 0.561 |
| Genes with r > 0.3 / r > 0.5 | 9,363 / 6,031 |

## ⚠️ Applicability boundary — read before use

R2M is an **in-distribution imputation tool for its 32 trained cancer types**. Leave-one-cancer-out analysis and external validation on two independent meningioma cohorts showed that for **unseen cancer types**, predictions reproduce the cohort-mean methylation profile but carry **no sample- or gene-specific signal** (centered per-sample r ≈ 0). Do not use R2M predictions for de-novo cancer subtype discovery, and do not interpret zero-shot predictions for cancer types outside the training set as sample-specific. See the manuscript (Sections 3.7–3.8) for details.

## Installation

```bash
git clone https://github.com/Bxz-Eric/R2M-Model.git
cd R2M-Model
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 (CPU is sufficient for inference).

## Quick start: predict methylation from your own RNA-seq data

1. Download the trained weights `r2m_main_model_seed42.pth` from the
   [Releases](https://github.com/Bxz-Eric/R2M-Model/releases) page and place it in `model/`
   (the small files `preprocessing_params.npz` and `cancer_types.json` are already in `model/`).

2. Prepare your expression matrix as a CSV: **genes × samples**, first column = gene ID
   (Ensembl gene ID or gene symbol), values = **TPM** (not log-transformed).

3. Run:

```bash
# for one of the 32 trained cancer types (e.g. BRCA)
python predict.py --input my_rna_tpm.csv --cancer BRCA --output predictions.csv

# for any other cancer type (zero-shot, mean cancer embedding)
python predict.py --input my_rna_tpm.csv --cancer mean --output predictions.csv
```

Output: `predictions.csv`, **samples × 9,835 genes**, predicted methylation beta values in [0, 1].

The 32 trained cancer codes (see `model/cancer_types.json`):
ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM.

**Notes on inputs**
- Missing genes are imputed to the training-set mean; genes outside the 4,000 high-variance model inputs are ignored. For best results, your assay should cover as many of the 9,835 model genes as possible.
- Gene symbols are mapped to Ensembl IDs automatically via [mygene.info](https://mygene.info) (requires internet).

## Training from scratch

Training requires the preprocessed TCGA pan-cancer dataset (see [Data availability](#data-availability)):

```bash
python v52_rna2meth.py --preprocessed preprocessed_v52.npz \
    --out_dir results_main --seed 42 --epochs 60 --patience 10 --no_gene_prior
```

Key flags: `--no_gene_prior` (learnable gene embeddings; main model), `--no_clip` / `--no_cancer` / `--no_expert` (ablations), `--loco_cancer LUAD` (leave-one-cancer-out). See `code/run_train.sh` and `code/run_job.sh` for examples.

## Reproducing the paper analyses

- `code/external_validation.py` — external validation on the two meningioma cohorts (GEO: GSE189672/GSE189673, GSE183653/GSE183656).
- `code/aggregate_results.py` — multi-seed / ablation / LOCO aggregation and Figures 4 and 6.

## Repository structure

```
R2M-Model/
├── README.md
├── LICENSE
├── requirements.txt
├── predict.py                     # inference: RNA TPM -> methylation beta values
├── v52_rna2meth.py                # model definition + training / evaluation
├── model/
│   ├── preprocessing_params.npz   # gene list, HVG indices, scaler & logit statistics
│   ├── cancer_types.json          # 32 trained cancer codes -> embedding indices
│   └── r2m_main_model_seed42.pth  # trained weights (download from Releases)
└── code/
    ├── external_validation.py     # external-cohort validation pipeline
    ├── aggregate_results.py       # results aggregation + figures
    ├── run_train.sh               # training launcher example
    └── run_job.sh                 # cluster job launcher example
```

## Data availability

- **TCGA** RNA-seq (TPM) and DNA methylation (450K) data: [GDC Data Portal](https://portal.gdc.cancer.gov).
- **External meningioma cohorts**: GEO accessions [GSE189672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189672)/[GSE189673](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189673) and [GSE183653](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183653)/[GSE183656](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183656).

## Citation

If you use R2M in your research, please cite:

> Bao X, Tian N. R2M: cross-modal translation of pan-cancer DNA methylation landscapes from transcriptomes via contrastive sample encoding and learnable gene representations. *Translational Cancer Research* (under review), 2026.

## License

This project is released under the [MIT License](LICENSE).

## Contact

Nan Tian (corresponding author): 20111003@zcmu.edu.cn
College of Life Science, Zhejiang Chinese Medical University, Hangzhou, China
