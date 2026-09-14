#!/usr/bin/env python3
"""R2M inference: predict gene-level DNA methylation from RNA expression.

Usage
-----
python predict.py --input my_rna_tpm.csv --cancer BRCA --output predictions.csv
python predict.py --input my_rna_tpm.csv --cancer mean --output predictions.csv

Input
-----
CSV file, genes x samples: first column = gene identifier (Ensembl gene ID or
gene symbol), remaining columns = samples, values = TPM (not log-transformed).

Cancer type
-----------
One of the 32 trained TCGA cancer codes (see model/cancer_types.json), e.g.
BRCA, LUAD, GBM. Use "mean" for any cancer type NOT in the training set
(zero-shot; note that out-of-distribution predictions recapitulate the
cohort-mean methylation profile only — see the manuscript for the
applicability boundary).

Output
------
CSV file, samples x 9,835 genes: predicted methylation beta values in [0, 1].
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v52_rna2meth import RNA2MethV52  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_params(path):
    pp = np.load(path)
    genes = [g.decode() if isinstance(g, bytes) else str(g) for g in pp['gene_ids']]
    return {
        'genes': genes,
        'g2i': {g: i for i, g in enumerate(genes)},
        'hvg_mask': np.isin(np.arange(len(genes)), pp['hvg_indices']),
        'scaler_mean': pp['scaler_mean'].astype(np.float32),
        'scaler_scale': pp['scaler_scale'].astype(np.float32),
        'mean_logit': pp['mean_logit'].astype(np.float32),
        'std_logit': pp['std_logit'].astype(np.float32),
        'top500_idx': pp['top500_idx'],
        'top100_idx': pp['top100_idx'],
    }


def symbols_to_ensembl(symbols):
    """Map human gene symbols to Ensembl gene IDs via mygene.info."""
    import requests
    mapping = {}
    B = 1000
    for i in range(0, len(symbols), B):
        chunk = symbols[i:i + B]
        r = requests.post('https://mygene.info/v3/query', data={
            'q': ','.join(chunk), 'scopes': 'symbol', 'fields': 'ensembl.gene',
            'species': 'human'}, timeout=60)
        for d in r.json():
            ens = d.get('ensembl')
            if isinstance(ens, list):
                ens = ens[0]['gene'] if ens else None
            elif isinstance(ens, dict):
                ens = ens.get('gene')
            if ens and 'query' in d:
                mapping[d['query']] = ens
    return mapping


def build_rna_matrix(df, params):
    """df: genes x samples (index = gene id). Returns samples x 9835 standardized matrix."""
    genes, g2i = params['genes'], params['g2i']
    idx = df.index.astype(str)
    if not idx.str.startswith('ENSG').any():
        print('[predict] gene IDs do not look like Ensembl; mapping symbols via mygene.info ...')
        mapping = symbols_to_ensembl(idx.tolist())
        df = df.copy()
        df['ensg'] = [mapping.get(s) for s in idx]
        df = df.dropna(subset=['ensg']).groupby('ensg').mean()
        print(f'[predict] symbol->Ensembl: {df.shape[0]} genes mapped')
    present = df.index.intersection(genes)
    if len(present) < 1000:
        print(f'[predict] WARNING: only {len(present)} of 9,835 model genes found in input.')
    mat = np.zeros((df.shape[1], len(genes)), dtype=np.float64)
    sub = np.log1p(df.loc[present].values.T.clip(min=0))
    mat[:, [g2i[g] for g in present]] = sub
    # genes absent from the assay are set to the training-set mean (log1p space)
    absent = np.array([g not in set(present) for g in genes])
    mat[:, absent] = params['scaler_mean'][absent]
    # zero non-HVG inputs (as in training), then standardize
    mat[:, ~params['hvg_mask']] = 0.0
    mat = (mat - params['scaler_mean']) / params['scaler_scale']
    return np.nan_to_num(mat.astype(np.float32), nan=0.0)


def load_model(weights, params, device):
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    model = RNA2MethV52(
        gene_feat=torch.zeros(len(params['genes']), 1281), num_cancers=32,
        rna_input_dim=len(params['genes']),
        top500_idx=params['top500_idx'], top100_idx=params['top100_idx'],
        latent_dim=256, hidden_dim=256, clip_hidden_dim=512, dropout=0.10,
        chunk_size=1024, no_gene_prior=True, no_cancer=False, no_expert=False,
        sample_emb_dim=0)
    model.load_state_dict(ck['model_state_dict'])
    # evaluation uses the EMA weights
    for name, param in model.named_parameters():
        if name in ck['ema_shadow']:
            param.data.copy_(ck['ema_shadow'][name].data)
    model.eval().to(device)
    return model


@torch.no_grad()
def predict(model, rna_norm, cancer, c2i, params, device, batch=64):
    if cancer.lower() == 'mean':
        # unseen cancer type: substitute the mean cancer embedding (slot 0)
        w = model.cancer_emb.weight
        w.data[0] = w.mean(dim=0)
        cb = model.global_decoder.cancer_bias.weight
        cb.data[0] = cb.mean(dim=0)
        ctype = torch.zeros(len(rna_norm), dtype=torch.long)
    else:
        if cancer not in c2i:
            raise ValueError(f"unknown cancer code '{cancer}'; use one of "
                             f"{sorted(c2i)} or 'mean'")
        ctype = torch.full((len(rna_norm),), c2i[cancer], dtype=torch.long)
    ml = torch.tensor(params['mean_logit'], device=device)
    sd = torch.tensor(params['std_logit'], device=device)
    preds = []
    for i in range(0, len(rna_norm), batch):
        r = torch.tensor(rna_norm[i:i + batch], device=device)
        out = model.forward_regression(r, ctype[i:i + batch].to(device), None)
        preds.append(torch.sigmoid(out * sd + ml).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser(description='R2M: RNA -> DNA methylation prediction')
    ap.add_argument('--input', required=True,
                    help='CSV, genes x samples, TPM (Ensembl IDs or symbols)')
    ap.add_argument('--cancer', required=True,
                    help="TCGA cancer code (e.g. BRCA) or 'mean' for unseen types")
    ap.add_argument('--output', required=True, help='output CSV, samples x 9835 genes')
    ap.add_argument('--weights', default=str(HERE / 'model' / 'r2m_main_model_seed42.pth'))
    ap.add_argument('--params', default=str(HERE / 'model' / 'preprocessing_params.npz'))
    ap.add_argument('--cancer-map', default=str(HERE / 'model' / 'cancer_types.json'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = load_params(args.params)
    c2i = json.load(open(args.cancer_map))['cancer_to_index']

    df = pd.read_csv(args.input, index_col=0)
    print(f'[predict] input: {df.shape[0]} genes x {df.shape[1]} samples')
    X = build_rna_matrix(df, params)
    model = load_model(args.weights, params, device)
    P = predict(model, X, args.cancer, c2i, params, device)
    out = pd.DataFrame(P, index=df.columns, columns=params['genes'])
    out.index.name = 'Sample_ID'
    out.to_csv(args.output)
    print(f'[predict] wrote {args.output}  ({out.shape[0]} samples x {out.shape[1]} genes)')


if __name__ == '__main__':
    main()
