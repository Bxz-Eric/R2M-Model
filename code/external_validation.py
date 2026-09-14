#!/usr/bin/env python3
"""External validation of R2M on two independent meningioma cohorts.

Cohort 1: GSE189672 (RNA TPM, Ensembl) + GSE189673 (gene-level beta), n=110
Cohort 2: GSE183653 (RNA TPM, symbols) + GSE183656 (gene-level beta), n=185

Settings: mean cancer embedding (primary zero-shot) and GBM embedding (sensitivity).
Ensemble over 3 seeds (42/43/44) of the main learnable-embedding model.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/mnt/shared-workspace/r2m')
from v52_rna2meth import RNA2MethV52  # noqa: E402

SHARED = '/mnt/shared-workspace/r2m'
UP = '/mnt/user-uploads'
OUT = '/workspace/r2m/external'
DEVICE = torch.device('cpu')

pp = np.load(f'{SHARED}/preprocessed_v52.npz')
GENES = [g.decode() if isinstance(g, bytes) else g for g in pp['gene_ids']]
G2I = {g: i for i, g in enumerate(GENES)}
HVG_MASK = np.zeros(len(GENES), bool)
HVG_MASK[pp['hvg_indices']] = True
SCALER_MU = pp['scaler_mean'].astype(np.float32)
SCALER_SD = pp['scaler_scale'].astype(np.float32)
MEAN_LOGIT = pp['mean_logit'].astype(np.float32)
STD_LOGIT = pp['std_logit'].astype(np.float32)
IDX_TR = pp['idx_tr']
TRAIN_MEAN_BETA = pp['meth_beta'][IDX_TR].mean(axis=0)  # null predictor

C2I = json.load(open(f'{SHARED}/cancer2idx.json'))
GBM_IDX = C2I['GBM']


def build_rna_matrix(df, id_is_symbol):
    """df: genes x samples (index = gene id). Return samples x 9835 standardized matrix."""
    if id_is_symbol:
        # symbol -> Ensembl mapping via mygene
        import requests
        syms = df.index.astype(str).tolist()
        mapping = {}
        B = 1000
        for i in range(0, len(syms), B):
            chunk = syms[i:i + B]
            r = requests.post('https://mygene.info/v3/query', data={
                'q': ','.join(chunk), 'scopes': 'symbol', 'fields': 'ensembl.gene',
                'species': 'human'})
            for d in r.json():
                ens = d.get('ensembl')
                if isinstance(ens, list):
                    ens = ens[0]['gene'] if ens else None
                elif isinstance(ens, dict):
                    ens = ens.get('gene')
                if ens and 'query' in d:
                    mapping[d['query']] = ens
        df = df.copy()
        df['ensg'] = [mapping.get(s) for s in df.index]
        df = df.dropna(subset=['ensg']).groupby('ensg').mean()
        print(f'  symbol->Ensembl: {len(df)} mapped genes')
    # align to model genes
    present = df.index.intersection(GENES)
    mat = np.zeros((df.shape[1], len(GENES)), dtype=np.float64)  # log1p space, 0 = no expr
    sub = np.log1p(df.loc[present].values.T.clip(min=0))
    cols = [G2I[g] for g in present]
    mat[:, cols] = sub
    # genes absent from external assay -> training mean in log1p space (= scaler mean pre-zeroing)
    absent = np.array([g not in set(present) for g in GENES])
    # training mean in log1p space for present genes equals scaler mean (zeroing only affects
    # non-HVG columns, whose standardized value is 0 anyway); use scaler_mu as train mean.
    mat[:, absent] = SCALER_MU[absent]
    # zero non-HVG (as in training), then standardize
    mat[:, ~HVG_MASK] = 0.0
    mat = (mat - SCALER_MU) / SCALER_SD
    return np.nan_to_num(mat.astype(np.float32), nan=0.0), present


def load_model(seed_tag):
    ck = torch.load(f'{SHARED}/results_{seed_tag}/best_model.pth', map_location='cpu',
                    weights_only=False)
    model = RNA2MethV52(
        gene_feat=torch.zeros(len(GENES), 1281), num_cancers=32, rna_input_dim=len(GENES),
        top500_idx=pp['top500_idx'], top100_idx=pp['top100_idx'],
        latent_dim=256, hidden_dim=256, clip_hidden_dim=512, dropout=0.10, chunk_size=1024,
        no_gene_prior=True, no_cancer=False, no_expert=False, sample_emb_dim=0)
    model.load_state_dict(ck['model_state_dict'])
    for name, param in model.named_parameters():
        if name in ck['ema_shadow']:
            param.data.copy_(ck['ema_shadow'][name].data)
    model.eval()
    return model


def predict(model, rna_norm, mode):
    """mode: 'mean' (mean cancer embedding) or 'gbm'."""
    with torch.no_grad():
        if mode == 'mean':
            w = model.cancer_emb.weight
            model.cancer_emb.weight.data[0] = w.mean(dim=0)
            cb = model.global_decoder.cancer_bias.weight
            model.global_decoder.cancer_bias.weight.data[0] = cb.mean(dim=0)
            ctype = torch.zeros(len(rna_norm), dtype=torch.long)
        else:
            ctype = torch.full((len(rna_norm),), GBM_IDX, dtype=torch.long)
        r = torch.tensor(rna_norm)
        preds = []
        ml = torch.tensor(MEAN_LOGIT); sd = torch.tensor(STD_LOGIT)
        for i in range(0, len(r), 64):
            out = model.forward_regression(r[i:i + 64], ctype[i:i + 64], None)
            preds.append(torch.sigmoid(out * sd + ml).numpy())
    return np.concatenate(preds)


def per_sample_r(Yt, Yp):
    out = []
    for i in range(Yt.shape[0]):
        m = ~np.isnan(Yt[i])
        out.append(np.corrcoef(Yt[i][m], Yp[i][m])[0, 1])
    return np.array(out)


def per_gene_r(Yt, Yp):
    cors = []
    for j in range(Yt.shape[1]):
        m = ~np.isnan(Yt[:, j])
        if m.sum() < 10 or np.nanstd(Yt[:, j][m]) == 0 or Yp[:, j].std() == 0:
            cors.append(np.nan)
        else:
            cors.append(np.corrcoef(Yt[:, j][m], Yp[:, j][m])[0, 1])
    return np.array(cors)


def per_sample_r_centered(Yt, Yp):
    Yc = Yt - np.nanmean(Yt, axis=0, keepdims=True)
    Pc = Yp - Yp.mean(axis=0, keepdims=True)
    out = []
    for i in range(Yt.shape[0]):
        m = ~np.isnan(Yc[i])
        if m.sum() < 10 or Yc[i][m].std() == 0 or Pc[i][m].std() == 0:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(Yc[i][m], Pc[i][m])[0, 1])
    return np.array(out)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # ---- load cohorts ----
    print('== cohort 1: GSE189672 + GSE189673 ==')
    rna1 = pd.read_csv(f'{UP}/GSE189672_RNAseq_TPM.csv', index_col=0)
    m1 = pd.read_csv(f'{UP}/GSE189673_methy.csv')
    X1, present1 = build_rna_matrix(rna1, id_is_symbol=False)
    Y1 = m1.set_index('Sample_ID').loc[rna1.columns][GENES].values.astype(np.float64)
    print('  X1', X1.shape, 'Y1', Y1.shape, 'HVG coverage:',
          int(HVG_MASK[[G2I[g] for g in present1]].sum()), '/ 4000')

    print('== cohort 2: GSE183653 + GSE183656 ==')
    rna2 = pd.read_csv(f'{UP}/GSE183653_meningioma_tpm.csv', index_col=0)
    m2 = pd.read_csv(f'{UP}/GSE183656_methy.csv')
    X2, present2 = build_rna_matrix(rna2, id_is_symbol=True)
    Y2 = m2.set_index('Sample_ID').loc[rna2.columns][GENES].values.astype(np.float64)
    print('  X2', X2.shape, 'Y2', Y2.shape, 'HVG coverage:',
          int(HVG_MASK[[G2I[g] for g in present2]].sum()), '/ 4000')

    # ---- inference: 3 seeds x 2 settings ----
    results = {}
    for mode in ['mean', 'gbm']:
        preds_seeds = {1: [], 2: []}
        for tag in ['abl_noprior', 's43np', 's44np']:
            model = load_model(tag)
            preds_seeds[1].append(predict(model, X1, mode))
            preds_seeds[2].append(predict(model, X2, mode))
            del model
        results[mode] = {1: np.mean(preds_seeds[1], axis=0), 2: np.mean(preds_seeds[2], axis=0)}

    # ---- metrics ----
    rows = []
    store = {}
    for mode in ['mean', 'gbm']:
        for cid, (X, Y) in {1: (X1, Y1), 2: (X2, Y2)}.items():
            P = results[mode][cid]
            psr = per_sample_r(Y, P)
            psrc = per_sample_r_centered(Y, P)
            pgr = per_gene_r(Y, P)
            # cohort-mean profile correlation (genes with >=10 observed samples)
            valid = (~np.isnan(Y)).sum(axis=0) >= 10
            obs_mean = np.nanmean(Y[:, valid], axis=0)
            pred_mean = P[:, valid].mean(axis=0)
            cmp_r = np.corrcoef(obs_mean, pred_mean)[0, 1]
            # null: train mean beta
            psr_null = per_sample_r(Y, np.tile(TRAIN_MEAN_BETA, (len(Y), 1)))
            cmp_r_null = np.corrcoef(obs_mean, TRAIN_MEAN_BETA[valid])[0, 1]
            rows.append({'cohort': f'GSE189672/73' if cid == 1 else 'GSE183653/56',
                         'setting': f'R2M ({mode} cancer emb)' if mode == 'mean'
                                    else 'R2M (GBM cancer emb)',
                         'n_samples': len(Y),
                         'per_sample_r_mean': psr.mean(), 'per_sample_r_median': np.median(psr),
                         'per_sample_r_centered': np.nanmean(psrc),
                         'cohort_mean_profile_r': cmp_r,
                         'per_gene_r_mean': np.nanmean(pgr),
                         'per_gene_r_median': np.nanmedian(pgr)})
            if mode == 'mean':
                rows.append({'cohort': f'GSE189672/73' if cid == 1 else 'GSE183653/56',
                             'setting': 'Train-mean null', 'n_samples': len(Y),
                             'per_sample_r_mean': psr_null.mean(),
                             'per_sample_r_median': np.median(psr_null),
                             'per_sample_r_centered': 0.0,
                             'cohort_mean_profile_r': cmp_r_null,
                             'per_gene_r_mean': np.nan, 'per_gene_r_median': np.nan})
                store[cid] = {'Y': Y, 'P': P, 'psr': psr, 'psrc': psrc, 'pgr': pgr,
                              'psr_null': psr_null, 'obs_mean': obs_mean, 'pred_mean': pred_mean,
                              'valid': valid}
            print(mode, cid, 'psr=%.3f' % psr.mean(), 'psr_centered=%.3f' % np.nanmean(psrc),
                  'profile_r=%.3f' % cmp_r, 'pgr=%.3f' % np.nanmean(pgr))

    tab = pd.DataFrame(rows)
    tab.to_csv(f'{OUT}/table6_external_validation.csv', index=False)
    np.savez(f'{OUT}/external_predictions.npz',
             **{f'c{cid}_{k}': v for cid, d in store.items() for k, v in d.items()})
    print(tab.round(4).to_string())


if __name__ == '__main__':
    main()
