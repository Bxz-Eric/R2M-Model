#!/usr/bin/env python3
"""Aggregate multi-seed, ablation, and LOCO results; build Fig 4 and Fig 6.

Main model = learnable gene embedding (tags: abl_noprior [seed42], s43np, s44np).
Static-prior variant = NT sequence prior (tags: s44 [seed44], s43n [seed43]).

Writes:
  - /mnt/results/r2m/tables/table2_multiseed.csv
  - /mnt/results/r2m/tables/table_ablation.csv
  - /mnt/results/r2m/tables/table_loco.csv
  - /mnt/results/r2m/figures/fig4_ablation.{png,svg}
  - /mnt/results/r2m/figures/fig6_per_cancer_loco.{png,svg}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHARED = Path('/mnt/shared-workspace/r2m')
FIGDIR = Path('/mnt/results/r2m/figures')
TABDIR = Path('/mnt/results/r2m/tables')

PAL = {'R2M (main)': '#0279EE', 'static NT prior': '#75A025', 'no CLIP': '#FF9400',
       'no cancer emb': '#FD9BED', 'no expert heads': '#7E57C2'}


def load_summary(tag):
    p = SHARED / f'results_{tag}' / 'test_summary.json'
    return json.load(open(p)) if p.exists() else None


def main():
    # ---------- multi-seed (main model) ----------
    main_seeds = {}
    for s, tag in [(42, 'abl_noprior'), (43, 's43np'), (44, 's44np')]:
        d = load_summary(tag)
        if d:
            main_seeds[s] = d
    metrics = ['gene_cor_all', 'gene_cor_top1000', 'gene_cor_top500', 'gene_cor_top100',
               'sample_cor_mean', 'gene_mse', 'gene_mae', 'n_cor_gt_03', 'n_cor_gt_05']
    rows = []
    for m in metrics:
        vals = [main_seeds[s][m] for s in main_seeds]
        rows.append({'metric': m, 'mean': np.mean(vals), 'sd': np.std(vals),
                     'n_seeds': len(vals),
                     **{f'seed_{s}': main_seeds[s][m] for s in main_seeds}})
    ms = pd.DataFrame(rows)
    ms.to_csv(TABDIR / 'table2_multiseed.csv', index=False)
    print('multi-seed (main):\n', ms[['metric', 'mean', 'sd', 'n_seeds']].to_string())

    # ---------- ablation ----------
    ref = main_seeds.get(42) or list(main_seeds.values())[0]
    abl_rows = [{'variant': 'R2M (main)', 'gene_cor_all': ref['gene_cor_all'],
                 'gene_cor_top500': ref['gene_cor_top500'],
                 'sample_cor_mean': ref['sample_cor_mean'], 'note': 'learnable gene embedding'}]
    # static NT prior variant (multi-seed if available)
    sp = [d for d in [load_summary('s44'), load_summary('s43n')] if d]
    if sp:
        abl_rows.append({'variant': 'static NT prior',
                         'gene_cor_all': float(np.mean([d['gene_cor_all'] for d in sp])),
                         'gene_cor_top500': float(np.mean([d['gene_cor_top500'] for d in sp])),
                         'sample_cor_mean': float(np.mean([d['sample_cor_mean'] for d in sp])),
                         'note': f'n={len(sp)} seeds (43,44), 80 epochs'})
    for t, name in [('abl_noclip_np', 'no CLIP'), ('abl_nocancer_np', 'no cancer emb'),
                    ('abl_noexpert_np', 'no expert heads')]:
        d = load_summary(t)
        if d:
            abl_rows.append({'variant': name, 'gene_cor_all': d['gene_cor_all'],
                             'gene_cor_top500': d['gene_cor_top500'],
                             'sample_cor_mean': d['sample_cor_mean'], 'note': 'seed 42'})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(TABDIR / 'table_ablation.csv', index=False)
    print('\nablation:\n', abl.to_string())

    # Fig 4
    if len(abl) > 1:
        fig, ax = plt.subplots(figsize=(6.8, 3.7))
        x = np.arange(len(abl))
        w = 0.38
        b1 = ax.bar(x - w / 2, abl['gene_cor_all'], w, color='#0279EE', label='All genes')
        b2 = ax.bar(x + w / 2, abl['gene_cor_top500'], w, color='#0279EE', alpha=0.45,
                    hatch='//', edgecolor='white', label='Top-500 genes')
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                        f'{b.get_height():.3f}', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(abl['variant'], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Mean per-gene Pearson r (test)', fontsize=10)
        ymax = max(abl['gene_cor_top500'].max(), abl['gene_cor_all'].max()) * 1.18
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=8.5, frameon=False, loc='lower center',
                  bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGDIR / 'fig4_ablation.png', dpi=300)
        fig.savefig(FIGDIR / 'fig4_ablation.svg')
        plt.close(fig)
        print('fig4 saved')

    # ---------- LOCO ----------
    loco_rows = []
    for t in ['loco_LUAD_np', 'loco_LUAD', 'loco_GBM']:
        d = load_summary(t)
        if d:
            loco_rows.append({'run': t, 'held_out_cancer': d['loco_cancer'],
                              'base': 'learnable emb' if t.endswith('_np') else 'static NT prior',
                              'gene_cor_all': d['gene_cor_all'],
                              'gene_cor_top500': d['gene_cor_top500'],
                              'sample_cor_mean': d['sample_cor_mean']})
    loco = pd.DataFrame(loco_rows)
    loco.to_csv(TABDIR / 'table_loco.csv', index=False)
    print('\nLOCO:\n', loco.to_string())

    # ---------- Fig 6 ----------
    pc = pd.read_csv(TABDIR / 'table3_per_cancer.csv').sort_values('sample_cor', ascending=False)
    loco_np = loco[loco['base'] == 'learnable emb'] if len(loco) else loco
    n_panels = 2 if len(loco_np) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 3.8),
                             gridspec_kw={'width_ratios': [2.2, 1][:n_panels]})
    if n_panels == 1:
        axes = [axes]
    ax = axes[0]
    ax.bar(np.arange(len(pc)), pc['sample_cor'], color='#0279EE', width=0.75)
    ax.set_xticks(np.arange(len(pc)))
    ax.set_xticklabels(pc['cancer'], rotation=90, fontsize=6.5)
    ax.set_ylabel('Per-sample Pearson r (test)', fontsize=10)
    ax.set_ylim(0.85, 1.0)
    ax.set_title('Per-cancer landscape reconstruction', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.04, 1.03, 'A', transform=ax.transAxes, fontsize=13, fontweight='bold')
    if len(loco_np):
        ax2 = axes[1]
        ref_ca = json.load(open('/workspace/r2m/loco_indist_ref.json'))
        labels, vals, gvals = [], [], []
        for _, r in loco_np.iterrows():
            ca = r['held_out_cancer']
            labels += [f'{ca}\nin-dist', f'{ca}\nLOCO']
            vals += [ref_ca[ca]['sample_cor_mean'], r['sample_cor_mean']]
            gvals += [ref_ca[ca]['gene_cor_all'], r['gene_cor_all']]
        x = np.arange(len(labels))
        ax2.bar(x - 0.2, vals, 0.4, color='#0279EE', label='Per-sample r')
        ax2.bar(x + 0.2, gvals, 0.4, color='#FF9400', label='Per-gene r (all)')
        for i, (v, g) in enumerate(zip(vals, gvals)):
            ax2.text(i - 0.2, v + 0.01, f'{v:.2f}', ha='center', fontsize=7)
            ax2.text(i + 0.2, g + 0.01, f'{g:.2f}', ha='center', fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_title('Cross-cancer generalization', fontsize=11)
        ax2.legend(fontsize=8, frameon=False)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.text(-0.06, 1.03, 'B', transform=ax2.transAxes, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig6_per_cancer_loco.png', dpi=300)
    fig.savefig(FIGDIR / 'fig6_per_cancer_loco.svg')
    plt.close(fig)
    print('fig6 saved')


if __name__ == '__main__':
    main()
