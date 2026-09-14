# -*- coding: utf-8 -*-
"""
RNA-to-Methylation Full Genome Predictor (v52)
=================================================
Changes vs v51 (global_best):
  1. Three-way split: train / val / test (patient-level, cancer-stratified).
     The TEST set is never used for training, early stopping, or model selection.
  2. Leakage-free checkpoint selection: v51 selected the best checkpoint by the
     mean Pearson of the top-100 genes *re-ranked on the validation set itself*
     (validation cherry-picking). v52 selects by the mean validation Pearson on
     the FIXED train-selected Top500 gene set.
  3. FiLM conditioning of the global decoder on the sample latent.
  4. Residual gene encoder (2 blocks).
  5. Optional sample-embedding branch (--sample_emb_path), ablation flags
     (--no_clip / --no_gene_prior / --no_cancer / --no_expert).
  6. Multi-seed support, per-epoch resume checkpoint, training log CSV,
     test-set evaluation + export from the best checkpoint.

Two data modes:
  A) --preprocessed path/to/preprocessed_v52.npz   (fast, identical splits)
  B) raw CSVs (--rna_path/--meth_path/--nt_emb_path/--cancer_map_path/--gc_file)
     reproduces the full preprocessing (patient-level split) standalone.
"""

import argparse
import contextlib
import gc
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
torch.backends.cudnn.benchmark = True


# ==============================================================================
# 0. Utils
# ==============================================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_stripped_gene_ids(cols):
    return [str(g).split(".")[0] for g in cols]


def autocast_ctx(device):
    return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))


def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def add_subset_delta(base, subset_idx, subset_delta):
    full_delta = torch.zeros_like(base)
    full_delta.index_copy_(1, subset_idx, subset_delta)
    return base + full_delta


# ==============================================================================
# 1. EMA
# ==============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model):
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model, backup):
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data.copy_(backup[name].data)

    @contextlib.contextmanager
    def scope(self, model):
        backup = self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model, backup)


# ==============================================================================
# 2. Scheduler
# ==============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps=500, min_lr_ratio=0.08):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            scale = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]


# ==============================================================================
# 3. Data loading / preprocessing
# ==============================================================================
def load_preprocessed(path):
    print(f"[data] loading preprocessed artifact: {path}", flush=True)
    d = np.load(path, allow_pickle=True)
    out = dict(
        rna_norm=d["rna_norm"].astype(np.float32),
        meth_logit_norm=d["meth_logit_norm"].astype(np.float32),
        cancer_indices=d["cancer_indices"].astype(np.int64),
        gene_features=d["gene_features"].astype(np.float32),
        idx_tr=d["idx_tr"].astype(np.int64),
        idx_val=d["idx_val"].astype(np.int64),
        idx_te=d["idx_te"].astype(np.int64),
        mean_logit=d["mean_logit"].astype(np.float32),
        std_logit=d["std_logit"].astype(np.float32),
        gene_weight=d["gene_weight"].astype(np.float32),
        top500_idx=d["top500_idx"].astype(np.int64),
        top100_idx=d["top100_idx"].astype(np.int64),
        top1000_idx=d["top1000_idx"].astype(np.int64),
        gene_ids=[str(g) for g in d["gene_ids"]],
        sample_ids=[str(s) for s in d["sample_ids"]],
        cancer_codes=[str(c) for c in d["cancer_codes"]],
    )
    out["num_cancers"] = int(out["cancer_indices"].max() + 1)
    print(f"[data] {len(out['sample_ids'])} samples x {out['rna_norm'].shape[1]} genes | "
          f"train {len(out['idx_tr'])} val {len(out['idx_val'])} test {len(out['idx_te'])}", flush=True)
    return out


def load_and_prepare_datasets(args):
    """Standalone preprocessing from raw CSVs (patient-level three-way split)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(f"[data] loading raw RNA ({args.rna_path}) and METH ({args.meth_path}) ...", flush=True)
    df_rna = pd.read_csv(args.rna_path, index_col=0)
    df_meth = pd.read_csv(args.meth_path, index_col=0)
    if df_rna.shape[0] == 9835 and df_rna.shape[1] != 9835:
        df_rna = df_rna.T
    if df_meth.shape[0] == 9835 and df_meth.shape[1] != 9835:
        df_meth = df_meth.T
    common = df_rna.index.intersection(df_meth.index)
    df_rna = df_rna.loc[common]; df_meth = df_meth.loc[common]

    # cancer labels: prefer GDC-corrected map if provided
    if args.gdc_map_path and os.path.exists(args.gdc_map_path):
        gdc = json.load(open(args.gdc_map_path))
        cancer_codes = [gdc.get(str(s)[:16], "OTHER").replace("TCGA-", "") for s in common]
    else:
        df_cancer = pd.read_csv(args.cancer_map_path)
        cmap = dict(zip(df_cancer["sample_name"], df_cancer["cancer_type"]))
        p15 = {k[:15]: v for k, v in cmap.items()}
        p12 = {k[:12]: v for k, v in cmap.items()}
        cancer_codes = [cmap.get(str(s)) or p15.get(str(s)[:15]) or p12.get(str(s)[:12]) or "OTHER" for s in common]

    unique_cancers = sorted(set(cancer_codes))
    cancer2idx = {c: i for i, c in enumerate(unique_cancers)}
    cancer_indices = np.array([cancer2idx[c] for c in cancer_codes], dtype=np.int64)

    rna_mat = np.nan_to_num(df_rna.values.astype(np.float32), nan=0.0)
    meth_mat = np.clip(np.nan_to_num(df_meth.values.astype(np.float32), nan=0.0), 0.0, 1.0)
    rna_mat = np.log1p(np.maximum(rna_mat, 0.0))
    gene_ids = safe_stripped_gene_ids(df_rna.columns)
    num_samples, num_genes = rna_mat.shape

    # patient-level stratified three-way split
    samples = [str(s) for s in common]
    patients = np.array([s[:12] for s in samples])
    upat, first = np.unique(patients, return_index=True)
    pat_cancer = cancer_indices[first]
    cnt = pd.Series(pat_cancer).value_counts()
    rare = cnt[cnt < 10].index.values
    pstrat = pat_cancer.copy(); pstrat[np.isin(pstrat, rare)] = -1
    seed = args.split_seed
    p_tr, p_tmp = train_test_split(np.arange(len(upat)), test_size=args.val_ratio + args.test_ratio,
                                   random_state=seed, shuffle=True, stratify=pstrat)
    p_val, p_te = train_test_split(p_tmp, test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
                                   random_state=seed, shuffle=True, stratify=pstrat[p_tmp])
    pat2part = {}
    for p in p_tr: pat2part[upat[p]] = 0
    for p in p_val: pat2part[upat[p]] = 1
    for p in p_te: pat2part[upat[p]] = 2
    part = np.array([pat2part[p] for p in patients])
    idx_tr = np.where(part == 0)[0]; idx_val = np.where(part == 1)[0]; idx_te = np.where(part == 2)[0]
    print(f"[data] patient-level split: train {len(idx_tr)} val {len(idx_val)} test {len(idx_te)}", flush=True)

    # train-only HVG
    rna_var = np.var(rna_mat[idx_tr], axis=0)
    hvg_indices = np.argsort(rna_var)[-args.hvg_num:]
    hvg_mask = np.zeros(num_genes, bool); hvg_mask[hvg_indices] = True
    rna_hvg = rna_mat.copy(); rna_hvg[:, ~hvg_mask] = 0.0
    scaler = StandardScaler().fit(rna_hvg[idx_tr])
    rna_norm = np.nan_to_num(scaler.transform(rna_hvg).astype(np.float32), nan=0.0)

    # train-only methylation logit stats
    meth_safe = np.clip(meth_mat, 1e-5, 1 - 1e-5)
    meth_logit = (np.log(meth_safe) - np.log(1 - meth_safe)).astype(np.float32)
    mean_logit = meth_logit[idx_tr].mean(axis=0).astype(np.float32)
    std_logit = (meth_logit[idx_tr].std(axis=0) + 1e-6).astype(np.float32)
    meth_logit_norm = np.nan_to_num((meth_logit - mean_logit) / std_logit, nan=0.0).astype(np.float32)

    # train-only predictability score / weights / top sets
    gene_var_meth = np.var(meth_mat[idx_tr], axis=0).astype(np.float32)
    x = rna_hvg[idx_tr]; y = meth_logit[idx_tr]
    xc = x - x.mean(0, keepdims=True); yc = y - y.mean(0, keepdims=True)
    num = np.sum(xc * yc, 0); den = np.sqrt(np.sum(xc ** 2, 0) * np.sum(yc ** 2, 0)) + 1e-8
    rna_meth_corr = np.nan_to_num(np.abs(num / den), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rank_norm = lambda v: np.argsort(np.argsort(v)).astype(np.float32) / (len(v) - 1 + 1e-8)
    score = 0.55 * rank_norm(gene_var_meth) + 0.45 * rank_norm(rna_meth_corr)
    gene_weight = np.clip(0.20 + 5.80 * score ** 3, 0.20, 6.00).astype(np.float32)
    gene_weight /= (gene_weight.mean() + 1e-8)
    top500_idx = np.argsort(score)[-500:]
    top100_idx = top500_idx[np.argsort(score[top500_idx])[-100:]]
    top1000_idx = np.argsort(score)[-1000:]

    # gene features: NT embeddings + GC
    npz = np.load(args.nt_emb_path)
    emb_dict = {str(g).split(".")[0]: e for g, e in zip(npz["gene_ids"], npz["embeddings"])}
    emb_dim = npz["embeddings"].shape[1]
    nt = np.stack([emb_dict.get(g, np.zeros(emb_dim, np.float32)) for g in gene_ids]).astype(np.float32)
    if args.gc_file and os.path.exists(args.gc_file):
        df_gc = pd.read_csv(args.gc_file)
        gc_dict = {str(g).split(".")[0]: v for g, v in zip(df_gc["gene_id"], df_gc["GC"])}
        gc_values = np.array([gc_dict.get(g, 0.5) for g in gene_ids], np.float32).reshape(-1, 1)
        gene_features = np.concatenate([nt, gc_values], axis=1)
    else:
        gene_features = nt

    return dict(
        rna_norm=rna_norm, meth_logit_norm=meth_logit_norm, cancer_indices=cancer_indices,
        gene_features=gene_features.astype(np.float32), idx_tr=idx_tr, idx_val=idx_val, idx_te=idx_te,
        mean_logit=mean_logit, std_logit=std_logit, gene_weight=gene_weight,
        top500_idx=top500_idx, top100_idx=top100_idx, top1000_idx=top1000_idx,
        gene_ids=gene_ids, sample_ids=samples, cancer_codes=cancer_codes,
        num_cancers=len(unique_cancers),
    )


# ==============================================================================
# 4. Model components
# ==============================================================================
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPSampleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.rna_encoder = MLPEncoder(input_dim, hidden_dim, output_dim, dropout)
        self.meth_encoder = MLPEncoder(input_dim, hidden_dim, output_dim, dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_rna(self, x):
        return self.rna_encoder(x)

    def encode_meth(self, x):
        return self.meth_encoder(x)

    def forward(self, rna, meth=None):
        rna_emb = self.encode_rna(rna)
        loss_contrast = None
        if meth is not None:
            meth_emb = self.encode_meth(meth)
            r = F.normalize(rna_emb, dim=-1)
            m = F.normalize(meth_emb, dim=-1)
            logits = self.logit_scale.exp() * (r @ m.T)
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_contrast = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return rna_emb, loss_contrast


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class GeneEncoder(nn.Module):
    """v52: input projection + 2 residual blocks (v51 used a plain MLP)."""
    def __init__(self, gene_feat_dim, latent_dim=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(gene_feat_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(ResBlock(latent_dim, dropout), ResBlock(latent_dim, dropout))
        self.out = nn.LayerNorm(latent_dim)

    def forward(self, g):
        return self.out(self.blocks(self.inp(g)))


class GlobalChunkDecoder(nn.Module):
    """v52: adds FiLM conditioning of gene keys on the sample latent."""
    def __init__(self, latent_dim, cancer_dim, hidden_dim, num_cancers, num_genes,
                 dropout=0.1, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_genes = num_genes

        self.q_proj = nn.Sequential(
            nn.Linear(latent_dim + cancer_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # FiLM: sample -> per-sample scale/shift of gene keys
        self.film = nn.Linear(latent_dim + cancer_dim, 2 * latent_dim)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

        # projected sample context for the residual correction (memory-efficient;
        # the full sample x gene interaction is carried by the FiLM bilinear term)
        self.ctx_dim = 48
        self.sample_ctx = nn.Linear(latent_dim + cancer_dim, self.ctx_dim)

        self.res_mlp = nn.Sequential(
            nn.Linear(latent_dim + cancer_dim + 1 + self.ctx_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.sample_bias = nn.Sequential(
            nn.Linear(latent_dim + cancer_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.cancer_bias = nn.Embedding(num_cancers, 1)
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))
        self.gene_gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(self, sample_latent, cancer_type, cancer_emb, gene_latent, rna_in):
        B, G = rna_in.shape
        L = self.q_proj[-1].out_features if hasattr(self.q_proj[-1], 'out_features') else sample_latent.shape[-1]
        q = self.q_proj(torch.cat([sample_latent, cancer_emb], dim=-1))          # B x L
        k = self.k_proj(gene_latent)                                             # G x L
        film = self.film(torch.cat([sample_latent, cancer_emb], dim=-1))         # B x 2L
        scale, shift = film.chunk(2, dim=-1)                                     # B x L each
        sample_scalar = self.sample_bias(torch.cat([sample_latent, cancer_emb], dim=-1))
        cancer_scalar = self.cancer_bias(cancer_type)

        # FiLM via matmul identity (avoids materializing B x G x L):
        #   <k_g*(1+scale_b) + shift_b, q_b> = (q_b*(1+scale_b)) @ k_g  +  shift_b . q_b
        inv_sqrt = 1.0 / math.sqrt(q.shape[-1])
        q_mod = q * (1.0 + scale)                                                # B x L
        shift_dot = (shift * q).sum(dim=1, keepdim=True) * inv_sqrt              # B x 1
        ctx = self.sample_ctx(torch.cat([sample_latent, cancer_emb], dim=-1))    # B x ctx_dim

        outputs = []
        for start in range(0, G, self.chunk_size):
            end = min(start + self.chunk_size, G)
            g = end - start
            bilinear = (q_mod @ k[start:end].T) * inv_sqrt + shift_dot           # B x g

            ctx_exp = ctx.unsqueeze(1).expand(B, g, -1)
            gene_exp = gene_latent[start:end].unsqueeze(0).expand(B, -1, -1)
            cancer_exp = cancer_emb.unsqueeze(1).expand(B, g, -1)
            rna_exp = rna_in[:, start:end].unsqueeze(-1)
            res = self.res_mlp(torch.cat([gene_exp, cancer_exp, rna_exp, ctx_exp], dim=-1)).squeeze(-1)

            gate = self.gene_gate(gene_latent[start:end]).T.expand(B, -1)
            bias = (self.gene_bias[start:end].unsqueeze(0).expand(B, -1)
                    + sample_scalar.expand(B, g) + cancer_scalar.expand(B, g))
            outputs.append(bilinear + gate * res + bias)
        return torch.cat(outputs, dim=1)


class SubsetResidualHead(nn.Module):
    def __init__(self, latent_dim, cancer_dim, hidden_dim, subset_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + cancer_dim + 1, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gene_gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        self.subset_bias = nn.Parameter(torch.zeros(subset_size))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, sample_latent, cancer_emb, subset_gene_latent, subset_rna):
        B, K = subset_rna.shape
        x = torch.cat([
            sample_latent.unsqueeze(1).expand(B, K, -1),
            subset_gene_latent.unsqueeze(0).expand(B, -1, -1),
            cancer_emb.unsqueeze(1).expand(B, K, -1),
            subset_rna.unsqueeze(-1),
        ], dim=-1)
        delta = self.net(x).squeeze(-1)
        gate = self.gene_gate(subset_gene_latent).T.expand(B, -1)
        return gate * delta + self.subset_bias.unsqueeze(0).expand(B, -1)


class RNA2MethV52(nn.Module):
    def __init__(self, gene_feat, num_cancers, rna_input_dim, top500_idx, top100_idx,
                 latent_dim=256, hidden_dim=256, clip_hidden_dim=512, dropout=0.1,
                 chunk_size=512, no_gene_prior=False, no_cancer=False, no_expert=False,
                 sample_emb_dim=0):
        super().__init__()
        self.num_genes = gene_feat.size(0)
        self.no_gene_prior = no_gene_prior
        self.no_cancer = no_cancer
        self.no_expert = no_expert
        self.sample_emb_dim = sample_emb_dim

        if no_gene_prior:
            self.gene_table = nn.Parameter(torch.randn(self.num_genes, 128) * 0.02)
            gene_feat_dim = 128
            self.register_buffer("gene_feat", torch.zeros(1))  # placeholder
        else:
            self.register_buffer("gene_feat", gene_feat)
            gene_feat_dim = gene_feat.size(1)

        self.register_buffer("top500_idx", torch.tensor(top500_idx, dtype=torch.long))
        self.register_buffer("top100_idx", torch.tensor(top100_idx, dtype=torch.long))

        self.cancer_emb = nn.Embedding(num_cancers, 32)
        self.clip_encoder = CLIPSampleEncoder(rna_input_dim, clip_hidden_dim, latent_dim, dropout)

        fusion_in = latent_dim + 32 + (64 if sample_emb_dim > 0 else 0)
        self.sample_fusion = nn.Sequential(
            nn.Linear(fusion_in, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim),
        )
        if sample_emb_dim > 0:
            self.se_proj = nn.Sequential(nn.Linear(sample_emb_dim, 64), nn.LayerNorm(64), nn.GELU())

        self.gene_encoder = GeneEncoder(gene_feat_dim, latent_dim, dropout)
        self.global_decoder = GlobalChunkDecoder(latent_dim, 32, hidden_dim, num_cancers,
                                                 self.num_genes, dropout, chunk_size)
        if not no_expert:
            top_hidden = hidden_dim * 2
            self.top500_head = SubsetResidualHead(latent_dim, 32, top_hidden, len(top500_idx), dropout)
            self.top100_head = SubsetResidualHead(latent_dim, 32, top_hidden, len(top100_idx), dropout)

    def get_gene_feat(self):
        return self.gene_table if self.no_gene_prior else self.gene_feat

    def forward_contrastive(self, rna, meth=None):
        return self.clip_encoder(rna, meth)

    def forward_regression(self, rna_in, cancer_type, sample_emb=None):
        rna_latent = self.clip_encoder.encode_rna(rna_in)
        if self.no_cancer:
            cancer_type = torch.zeros_like(cancer_type)
        cancer_emb = self.cancer_emb(cancer_type)
        fusion_in = [rna_latent, cancer_emb]
        if self.sample_emb_dim > 0:
            fusion_in.append(self.se_proj(sample_emb))
        sample_latent = self.sample_fusion(torch.cat(fusion_in, dim=-1))
        gene_latent = self.gene_encoder(self.get_gene_feat())

        out = self.global_decoder(sample_latent, cancer_type, cancer_emb, gene_latent, rna_in)
        if not self.no_expert:
            out = add_subset_delta(out, self.top500_idx,
                                   self.top500_head(sample_latent, cancer_emb,
                                                    gene_latent[self.top500_idx], rna_in[:, self.top500_idx]))
            out = add_subset_delta(out, self.top100_idx,
                                   self.top100_head(sample_latent, cancer_emb,
                                                    gene_latent[self.top100_idx], rna_in[:, self.top100_idx]))
        return out


# ==============================================================================
# 5. Losses
# ==============================================================================
def per_gene_pearson(pred, target, eps=1e-8):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    pred_std = torch.sqrt(torch.mean(pred_c ** 2, dim=0) + eps)
    target_std = torch.sqrt(torch.mean(target_c ** 2, dim=0) + eps)
    corr = torch.mean(pred_c * target_c, dim=0) / (pred_std * target_std + eps)
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_pearson_loss(pred, target, gene_weight=None):
    corr = per_gene_pearson(pred, target)
    if gene_weight is None:
        return 1.0 - corr.mean()
    w = gene_weight / (gene_weight.mean() + 1e-8)
    return ((1.0 - corr) * w).sum() / (w.sum() + 1e-8)


def weighted_mse_loss(pred, target, gene_weight=None):
    per_gene = torch.mean((pred - target) ** 2, dim=0)
    if gene_weight is None:
        return per_gene.mean()
    w = gene_weight / (gene_weight.mean() + 1e-8)
    return (per_gene * w).sum() / (w.sum() + 1e-8)


def weighted_huber_loss(pred, target, gene_weight=None, beta=1.0):
    per_gene = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=0)
    if gene_weight is None:
        return per_gene.mean()
    w = gene_weight / (gene_weight.mean() + 1e-8)
    return (per_gene * w).sum() / (w.sum() + 1e-8)


def subset_aux_loss(pred_beta, true_beta, subset_idx):
    pred = pred_beta[:, subset_idx]; true = true_beta[:, subset_idx]
    return 0.75 * weighted_pearson_loss(pred, true, None) + 0.25 * weighted_mse_loss(pred, true, None)


# ==============================================================================
# 6. Metrics (leakage-free: fixed train-selected gene sets)
# ==============================================================================
def gene_metrics_np(y_true, y_pred):
    """Per-gene MSE/MAE/Pearson across samples (rows)."""
    y_true = np.nan_to_num(y_true, nan=0.0).astype(np.float32)
    y_pred = np.nan_to_num(y_pred, nan=0.5).astype(np.float32)
    diff = y_true - y_pred
    mses = np.mean(diff ** 2, axis=0)
    maes = np.mean(np.abs(diff), axis=0)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = np.sum(yt * yp, axis=0)
    den = np.sqrt(np.sum(yt ** 2, axis=0) * np.sum(yp ** 2, axis=0)) + 1e-8
    cors = np.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)
    return cors, mses, maes


def sample_metrics_np(y_true, y_pred):
    """Per-sample Pearson across genes (rows = samples)."""
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = np.sum(yt * yp, axis=1)
    den = np.sqrt(np.sum(yt ** 2, axis=1) * np.sum(yp ** 2, axis=1)) + 1e-8
    return np.nan_to_num(num / den, nan=0.0)


def summarize_split(y_true, y_pred, top500_idx, top100_idx, top1000_idx):
    cors, mses, maes = gene_metrics_np(y_true, y_pred)
    scor = sample_metrics_np(y_true, y_pred)
    return dict(
        gene_mse=float(np.mean(mses)), gene_mae=float(np.mean(maes)),
        gene_cor_all=float(np.mean(cors)),
        gene_cor_top500=float(np.mean(cors[top500_idx])),
        gene_cor_top100=float(np.mean(cors[top100_idx])),
        gene_cor_top1000=float(np.mean(cors[top1000_idx])),
        n_cor_gt_03=int(np.sum(cors > 0.3)), n_cor_gt_05=int(np.sum(cors > 0.5)),
        sample_cor_mean=float(np.mean(scor)), sample_cor_median=float(np.median(scor)),
        sample_mse=float(np.mean((y_true - y_pred) ** 2)), sample_mae=float(np.mean(np.abs(y_true - y_pred))),
    ), cors, mses, maes, scor


# ==============================================================================
# 7. Training control
# ==============================================================================
def set_phase_trainable(model, phase):
    for p in model.parameters():
        p.requires_grad = False
    if phase == "pretrain":
        for p in model.clip_encoder.parameters():
            p.requires_grad = True
    elif phase == "global":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(phase)


def build_optimizer(model, phase, lr, weight_decay):
    if phase == "pretrain":
        return torch.optim.AdamW(trainable_params(model.clip_encoder), lr=lr, weight_decay=weight_decay)
    groups = [
        {"params": trainable_params(model.clip_encoder), "lr": lr * 0.10},
        {"params": trainable_params(model.cancer_emb), "lr": lr * 0.30},
        {"params": trainable_params(model.sample_fusion), "lr": lr * 0.30},
        {"params": trainable_params(model.gene_encoder), "lr": lr * 1.00},
        {"params": trainable_params(model.global_decoder), "lr": lr * 1.00},
    ]
    if not model.no_expert:
        groups += [{"params": trainable_params(model.top500_head), "lr": lr * 1.50},
                   {"params": trainable_params(model.top100_head), "lr": lr * 2.00}]
    if model.no_gene_prior:
        groups.append({"params": [model.gene_table], "lr": lr * 1.00})
    if model.sample_emb_dim > 0:
        groups.append({"params": trainable_params(model.se_proj), "lr": lr * 0.30})
    return torch.optim.AdamW([g for g in groups if len(g["params"]) > 0], weight_decay=weight_decay)


def get_regression_loss_weights(epoch_in_phase, total_phase_epochs):
    prog = epoch_in_phase / max(1, total_phase_epochs)
    early = dict(corr=0.55, beta=0.25, logit=0.10, top500=0.07, top100=0.03)
    late = dict(corr=0.25, beta=0.08, logit=0.04, top500=0.28, top100=0.35)
    blend = min(1.0, max(0.0, (prog - 0.50) / 0.50))
    return tuple(early[k] * (1 - blend) + late[k] * blend for k in ("corr", "beta", "logit", "top500", "top100"))


def run_clip_pretrain_epoch(model, loader, optimizer, scaler, device, accumulation_steps=1):
    model.train()
    total, n = 0.0, len(loader)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader, 1):
        r_b, m_b = batch[0].to(device), batch[1].to(device)
        with autocast_ctx(device):
            _, loss_c = model.forward_contrastive(r_b, m_b)
            loss = loss_c / accumulation_steps
        scaler.scale(loss).backward()
        if i % accumulation_steps == 0 or i == n:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * accumulation_steps
    return total / max(1, n)


def run_regression_train_epoch(model, loader, optimizer, scheduler, scaler, device,
                               mean_logit_t, std_logit_t, gene_weight_t,
                               epoch_in_phase, total_phase_epochs, args, ema=None):
    model.train()
    total, n = 0.0, len(loader)
    optimizer.zero_grad(set_to_none=True)
    corr_w, beta_w, logit_w, top500_w, top100_w = get_regression_loss_weights(epoch_in_phase, total_phase_epochs)
    corr_scale = min(1.0, epoch_in_phase / max(1, args.corr_warmup_epochs))
    if model.no_expert:
        top500_w = top100_w = 0.0

    for i, batch in enumerate(loader, 1):
        r_b = batch[0].to(device); m_b = batch[1].to(device); c_b = batch[2].to(device)
        se_b = batch[3].to(device) if len(batch) > 3 else None
        with autocast_ctx(device):
            pred_norm = model.forward_regression(r_b, c_b, se_b)
            pred_logit = pred_norm * std_logit_t + mean_logit_t
            true_logit = m_b * std_logit_t + mean_logit_t
            pred_beta = torch.sigmoid(pred_logit); true_beta = torch.sigmoid(true_logit)

            loss = (corr_w * corr_scale * weighted_pearson_loss(pred_beta, true_beta, gene_weight_t)
                    + beta_w * weighted_mse_loss(pred_beta, true_beta, gene_weight_t)
                    + logit_w * weighted_huber_loss(pred_norm, m_b, gene_weight_t, beta=1.0))
            if not model.no_expert:
                loss = loss + top500_w * subset_aux_loss(pred_beta, true_beta, model.top500_idx)
                loss = loss + top100_w * subset_aux_loss(pred_beta, true_beta, model.top100_idx)
            if args.joint_contrast_weight > 0:
                _, lc = model.forward_contrastive(r_b, m_b)
                loss = loss + args.joint_contrast_weight * lc
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        if i % args.accumulation_steps == 0 or i == n:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
        total += loss.item() * args.accumulation_steps
    return total / max(1, n)


@torch.no_grad()
def evaluate_regression(model, loader, device, mean_logit_t, std_logit_t):
    model.eval()
    trues, preds = [], []
    for batch in loader:
        r_b = batch[0].to(device); m_b = batch[1].to(device); c_b = batch[2].to(device)
        se_b = batch[3].to(device) if len(batch) > 3 else None
        with autocast_ctx(device):
            pred_norm = model.forward_regression(r_b, c_b, se_b)
            pred_beta = torch.sigmoid(pred_norm * std_logit_t + mean_logit_t)
            true_beta = torch.sigmoid(m_b * std_logit_t + mean_logit_t)
        trues.append(true_beta.float().cpu().numpy())
        preds.append(pred_beta.float().cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ==============================================================================
# 8. Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="RNA-to-Methylation Predictor (v52)")
    ap.add_argument("--preprocessed", type=str, default="")
    ap.add_argument("--rna_path", type=str, default="/share/pub/baoxz/tcga/rna_clean.csv")
    ap.add_argument("--meth_path", type=str, default="/share/pub/baoxz/tcga/meth_clean.csv")
    ap.add_argument("--nt_emb_path", type=str, default="/share/pub/baoxz/tcga/gene_embeddings.npz")
    ap.add_argument("--cancer_map_path", type=str, default="/share/pub/baoxz/tcga/sample_cancer_map.csv")
    ap.add_argument("--gdc_map_path", type=str, default="")
    ap.add_argument("--gc_file", type=str, default="/share/pub/baoxz/tcga/gene_gc_content.csv")
    ap.add_argument("--sample_emb_path", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./result_r2m_v52")

    ap.add_argument("--hvg_num", type=int, default=4000)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--clip_hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--chunk_size", type=int, default=1024)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--pretrain_epochs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--pretrain_lr", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--joint_contrast_weight", type=float, default=0.0)
    ap.add_argument("--corr_warmup_epochs", type=int, default=18)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)

    # ablation flags
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--no_gene_prior", action="store_true")
    ap.add_argument("--no_cancer", action="store_true")
    ap.add_argument("--no_expert", action="store_true")

    # LOCO: hold out one cancer type entirely (train on the rest)
    ap.add_argument("--loco_cancer", type=str, default="")

    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device: {device} | seed {args.seed}", flush=True)

    # ---------------- data ----------------
    if args.preprocessed and os.path.exists(args.preprocessed):
        D = load_preprocessed(args.preprocessed)
    else:
        D = load_and_prepare_datasets(args)

    rna_norm = D["rna_norm"]; meth_logit_norm = D["meth_logit_norm"]
    cancer_indices = D["cancer_indices"]; num_cancers = D["num_cancers"]
    idx_tr, idx_val, idx_te = D["idx_tr"], D["idx_val"], D["idx_te"]

    # LOCO: remove held-out cancer from train/val; test = that cancer only
    if args.loco_cancer:
        held = np.array([c == args.loco_cancer for c in D["cancer_codes"]])
        idx_te = np.where(held)[0]
        idx_tr = idx_tr[~held[idx_tr]]
        idx_val = idx_val[~held[idx_val]]
        print(f"[LOCO] holding out {args.loco_cancer}: train {len(idx_tr)} val {len(idx_val)} test {len(idx_te)}", flush=True)

    num_genes = rna_norm.shape[1]

    # optional sample embeddings
    se_tensor = None
    if args.sample_emb_path and os.path.exists(args.sample_emb_path):
        se = np.load(args.sample_emb_path)
        if se.ndim == 2 and se.shape[0] == rna_norm.shape[0]:
            se_tensor = torch.tensor(np.nan_to_num(se, nan=0.0), dtype=torch.float32)
            print(f"[data] sample embeddings enabled: {se_tensor.shape}", flush=True)
        else:
            print("[data] sample embedding shape mismatch - ignored", flush=True)

    rna_tensor = torch.tensor(rna_norm, dtype=torch.float32)
    meth_tensor = torch.tensor(meth_logit_norm, dtype=torch.float32)
    cancer_tensor = torch.tensor(cancer_indices, dtype=torch.long)
    if se_tensor is not None:
        dataset = data.TensorDataset(rna_tensor, meth_tensor, cancer_tensor, se_tensor)
    else:
        dataset = data.TensorDataset(rna_tensor, meth_tensor, cancer_tensor)

    train_loader = data.DataLoader(data.Subset(dataset, idx_tr), batch_size=args.batch_size,
                                   shuffle=True, drop_last=True, num_workers=args.num_workers,
                                   pin_memory=(device.type == "cuda"))
    val_loader = data.DataLoader(data.Subset(dataset, idx_val), batch_size=64, shuffle=False,
                                 num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = data.DataLoader(data.Subset(dataset, idx_te), batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=(device.type == "cuda"))

    mean_logit_t = torch.tensor(D["mean_logit"], dtype=torch.float32, device=device)
    std_logit_t = torch.tensor(D["std_logit"], dtype=torch.float32, device=device)
    gene_weight_t = torch.tensor(D["gene_weight"], dtype=torch.float32, device=device)
    top500_idx, top100_idx, top1000_idx = D["top500_idx"], D["top100_idx"], D["top1000_idx"]

    # ---------------- model ----------------
    model = RNA2MethV52(
        gene_feat=torch.tensor(D["gene_features"], dtype=torch.float32),
        num_cancers=num_cancers, rna_input_dim=num_genes,
        top500_idx=top500_idx, top100_idx=top100_idx,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        clip_hidden_dim=args.clip_hidden_dim, dropout=args.dropout, chunk_size=args.chunk_size,
        no_gene_prior=args.no_gene_prior, no_cancer=args.no_cancer, no_expert=args.no_expert,
        sample_emb_dim=(se_tensor.shape[1] if se_tensor is not None else 0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)

    best_metric = -1.0; best_epoch = -1; not_improved = 0
    log_rows = []
    start_epoch = 1
    optimizer = scheduler = scaler = ema = None

    # ---------------- Stage 1: CLIP pretrain ----------------
    if args.pretrain_epochs > 0 and not args.no_clip:
        print(f"[Stage 1] CLIP pretrain {args.pretrain_epochs} epochs", flush=True)
        set_phase_trainable(model, "pretrain")
        clip_opt = build_optimizer(model, "pretrain", args.pretrain_lr, args.weight_decay)
        clip_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for epoch in range(1, args.pretrain_epochs + 1):
            t0 = time.time()
            cl = run_clip_pretrain_epoch(model, train_loader, clip_opt, clip_scaler, device,
                                         args.accumulation_steps)
            print(f"[Pretrain {epoch}/{args.pretrain_epochs}] {time.time()-t0:.1f}s contrast_loss {cl:.4f}", flush=True)
        torch.save({"clip_encoder_state_dict": model.clip_encoder.state_dict(), "args": vars(args)},
                   os.path.join(args.out_dir, "clip_pretrain.pth"))

    # ---------------- Stage 2: global regression ----------------
    print(f"[Stage 2] global regression up to {args.epochs} epochs", flush=True)
    set_phase_trainable(model, "global")
    optimizer = build_optimizer(model, "global", args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.accumulation_steps))
    scheduler = WarmupCosineScheduler(optimizer, total_steps=max(1, args.epochs * steps_per_epoch),
                                      warmup_steps=max(20, int(0.01 * args.epochs * steps_per_epoch)),
                                      min_lr_ratio=0.08)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=0.999)

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        ema.shadow = ck["ema_shadow"]
        start_epoch = ck["epoch"] + 1
        best_metric = ck["best_metric"]; best_epoch = ck["best_epoch"]
        not_improved = ck["not_improved"]
        log_rows = ck.get("log_rows", [])
        print(f"[resume] from epoch {ck['epoch']} best {best_metric:.4f}@{best_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = run_regression_train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            mean_logit_t, std_logit_t, gene_weight_t, epoch, args.epochs, args, ema)

        with ema.scope(model):
            val_true, val_pred = evaluate_regression(model, val_loader, device, mean_logit_t, std_logit_t)
        vsum, _, _, _, _ = summarize_split(val_true, val_pred, top500_idx, top100_idx, top1000_idx)
        # leakage-free selection metric: fixed train-selected Top500 val correlation
        sel_metric = vsum["gene_cor_top500"]

        el = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}/{args.epochs}] {el:.1f}s lr {lr_now:.2e} loss {train_loss:.4f} | "
              f"val corAll {vsum['gene_cor_all']:.4f} corT500 {vsum['gene_cor_top500']:.4f} "
              f"corT1000 {vsum['gene_cor_top1000']:.4f} sampCor {vsum['sample_cor_mean']:.4f} "
              f"mse {vsum['gene_mse']:.4f}", flush=True)
        log_rows.append(dict(epoch=epoch, lr=lr_now, train_loss=train_loss, epoch_sec=el, **vsum))
        pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)

        if sel_metric > best_metric:
            best_metric = sel_metric; best_epoch = epoch; not_improved = 0
            torch.save({
                "model_state_dict": model.state_dict(), "ema_shadow": ema.shadow, "args": vars(args),
                "mean_logit": D["mean_logit"], "std_logit": D["std_logit"],
                "gene_ids": D["gene_ids"], "cancer_codes": D["cancer_codes"],
                "top500_idx": top500_idx, "top100_idx": top100_idx, "top1000_idx": top1000_idx,
                "best_epoch": best_epoch, "best_metric": best_metric,
            }, os.path.join(args.out_dir, "best_model.pth"))
            np.savez_compressed(os.path.join(args.out_dir, "val_predictions.npz"),
                                true_methy=val_true, pred_methy=val_pred,
                                sample_ids=np.array([D["sample_ids"][i] for i in idx_val]))
            print(f"  -> new best (val Top500 cor {best_metric:.4f})", flush=True)
        else:
            not_improved += 1

        # per-epoch resume checkpoint
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "ema_shadow": ema.shadow,
            "best_metric": best_metric, "best_epoch": best_epoch,
            "not_improved": not_improved, "log_rows": log_rows,
        }, os.path.join(args.out_dir, "resume_checkpoint.pth"))

        if not_improved >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break
        gc.collect()

    # ---------------- test evaluation from best checkpoint ----------------
    print("[eval] loading best checkpoint (EMA weights) for test evaluation ...", flush=True)
    ck = torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    for name, param in model.named_parameters():
        if param.requires_grad and name in ck["ema_shadow"]:
            param.data.copy_(ck["ema_shadow"][name].data)

    # LOCO zero-shot: replace the held-out cancer's (untrained) embedding with the
    # mean of the trained cancer embeddings, so evaluation does not use random weights.
    if args.loco_cancer:
        c2i = {c: i for i, c in enumerate(sorted(set(D["cancer_codes"])))}
        held_idx = c2i.get(args.loco_cancer)
        if held_idx is not None:
            with torch.no_grad():
                w = model.cancer_emb.weight  # num_cancers x 32
                trained = [i for i in range(w.shape[0]) if i != held_idx]
                mean_emb = w[trained].mean(dim=0, keepdim=True)
                w[held_idx] = mean_emb.squeeze(0)
            print(f"[LOCO] set held-out cancer embedding (idx {held_idx}) to mean of trained embeddings", flush=True)

    test_true, test_pred = evaluate_regression(model, test_loader, device, mean_logit_t, std_logit_t)
    tsum, cors, mses, maes, scor = summarize_split(test_true, test_pred, top500_idx, top100_idx, top1000_idx)
    print("[TEST]", json.dumps(tsum, indent=2), flush=True)

    test_sample_ids = [D["sample_ids"][i] for i in idx_te]
    np.savez_compressed(os.path.join(args.out_dir, "test_predictions.npz"),
                        true_methy=test_true, pred_methy=test_pred,
                        sample_ids=np.array(test_sample_ids))
    pd.DataFrame({"Gene_ID": D["gene_ids"], "MSE": mses, "MAE": maes, "Pearson": cors}
                 ).to_csv(os.path.join(args.out_dir, "test_gene_metrics.csv"), index=False)
    pd.DataFrame({"Sample_ID": test_sample_ids, "Pearson": scor}
                 ).to_csv(os.path.join(args.out_dir, "test_sample_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "test_summary.json"), "w") as f:
        json.dump(dict(seed=args.seed, best_epoch=best_epoch, best_val_top500=best_metric,
                       loco_cancer=args.loco_cancer,
                       ablations=dict(no_clip=args.no_clip, no_gene_prior=args.no_gene_prior,
                                      no_cancer=args.no_cancer, no_expert=args.no_expert,
                                      sample_emb=bool(se_tensor is not None)),
                       **tsum), f, indent=2)
    print(f"done. best epoch {best_epoch}, outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
