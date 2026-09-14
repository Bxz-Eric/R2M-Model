#!/usr/bin/env python3
"""Transform manuscript_draft.md into Translational Cancer Research (AME) format.

TCR Original Article requirements applied:
- Title page (running title, authors placeholder, word count, fig/table counts, contributions)
- Structured abstract Background/Methods/Results/Conclusions, 200-450 words
- 3-5 keywords
- Highlight Box (<=250 words)
- Main text: Introduction / Methods / Results / Discussion / Conclusions (unnumbered)
- Reporting-checklist statement at end of Introduction (TRIPOD)
- Acknowledgments + Footnote (Reporting Checklist, Data Sharing, Funding, COI, Ethical Statement)
- Vancouver references: in-text numbers in round brackets, numbered by first appearance;
  reference list AME style (first 3 authors + et al., abbreviated journal, no issue, no PMID)
"""
import re
from pathlib import Path

MD = Path('/workspace/r2m/manuscript_draft.md')
OUT = Path('/workspace/r2m/manuscript_tcr.md')

md = MD.read_text()
body, _refs = md.split('## References')

# ---------------- citation renumbering ----------------
ORDER = ['22641018', '22781841', '10411935', '24071849', '21839163', '21593595',
         '32127627', '33509261', '33452255', '40200144', '32183722', '38171264',
         'LEVYJURGENSON', '39975295', 'SALMAN', '39609566', 'RADFORD', '26704973',
         'LOSHCHILOV', 'ZOU', '35108039', '35534562', '31066453']
NUM = {k: i + 1 for i, k in enumerate(ORDER)}


def pmid_group_repl(m):
    nums = [str(NUM[p]) for p in re.findall(r'PMID:(\d+)', m.group(0))]
    return '(' + ','.join(nums) + ')'


body = re.sub(r'\[PMID:[\d, :PMID]+\]', pmid_group_repl, body)

# named non-PMID citations -> numbers
body = body.replace('(Levy-Jurgenson et al., bioRxiv 2018, doi:10.1101/491357)', '(13)')
body = body.replace('(Radford et al., arXiv:2103.00020)', '(17)')
body = body.replace('AdamW (Loshchilov and Hutter, arXiv:1711.05101; initial learning rate 3×10⁻⁴, weight decay 1×10⁻²)',
                    'AdamW (19) (initial learning rate 3×10⁻⁴, weight decay 1×10⁻²)')
body = body.replace('(Zou and Hastie, doi:10.1111/j.1467-9868.2005.00503.x)', '(20)')
body = body.replace('(Salman, Authorea 2025, doi:10.22541/au.175942409.95172063/v1)', '(15)')

# ---------------- structural edits on body ----------------
# un-number section headings
body = re.sub(r'^## \d\. ', '## ', body, flags=re.M)
body = re.sub(r'^### \d\.\d+ ', '### ', body, flags=re.M)
body = body.replace('## Materials and Methods', '## Methods')

# fix cross-references to removed section numbers
body = body.replace('the same gene encoder (Section 2.4)',
                    'the same gene encoder (see “Model architecture”)')
body = body.replace('consistent with the within-cancer analysis (Section 3.4)',
                    'consistent with the within-cancer analysis (see the per-cancer analysis below)')
body = body.replace('fully consistent with the LOCO analysis (Section 3.7)',
                    'fully consistent with the LOCO analysis (see “Cross-cancer generalization” above)')

# TRIPOD statement at end of Introduction
intro_end_anchor = 'Here we present R2M, a cross-modal translation model'
assert intro_end_anchor in body
# append checklist sentence to the end of the last intro paragraph
body = body.replace(
    'dissect component contributions by ablation, characterize the biological determinants of '
    'gene-level predictability, and assess cross-cancer generalization by leave-one-cancer-out '
    'zero-shot evaluation.',
    'dissect component contributions by ablation, characterize the biological determinants of '
    'gene-level predictability, and assess cross-cancer generalization by leave-one-cancer-out '
    'zero-shot evaluation and external validation on two independent meningioma cohorts. '
    'We present this article in accordance with the TRIPOD reporting checklist.')

# split Conclusions out of Discussion
concl_anchor = 'In summary, R2M establishes that transcriptome-to-methylome translation'
assert concl_anchor in body
body = body.replace(concl_anchor, '## Conclusions\n\n' + concl_anchor)

# drop old title + abstract + keywords (replaced below)
body = body.split('## 1. Introduction')[-1] if '## 1. Introduction' in body else body
body = '## Introduction' + body.split('## Introduction', 1)[1]

# ---------------- new front matter ----------------
title = ('R2M: cross-modal translation of pan-cancer DNA methylation landscapes from '
         'transcriptomes via contrastive sample encoding and learnable gene representations')

abstract = """## Abstract

**Background:** DNA methylation is a central epigenetic mechanism regulating gene expression, cell-fate determination, and tumorigenesis. Although large cohorts such as The Cancer Genome Atlas (TCGA) profile both transcriptomes and DNA methylomes, methylation data are frequently missing, costly to generate, or incomplete, motivating computational inference of methylation from the more readily available transcriptome. The RNA–methylation relationship is complex, nonlinear, and context-dependent, and remains difficult to model at genome scale.

**Methods:** R2M was trained on 9,127 paired TCGA samples spanning 32 cancer types under a strict patient-level train/validation/test split with leakage-free model selection. The model couples learnable gene identity embeddings and a cancer-type embedding with a contrastively aligned sample encoder and a FiLM-conditioned bilinear decoder. We benchmarked R2M against per-gene ridge, elastic-net, own-expression, and multi-output multi-layer perceptron (MLP) baselines, dissected components by ablation, and probed generalization by leave-one-cancer-out analysis and external validation on two independent meningioma cohorts (n = 110 and 185).

**Results:** R2M achieved a mean per-gene Pearson correlation of 0.521 ± 0.003 (three seeds) across 9,835 genes on the held-out test set (0.62 for the Top-100 most predictable genes), significantly outperforming all baselines (P < 10⁻³⁰⁰, paired Wilcoxon). Static gene sequence priors from a pretrained nucleotide foundation model underperformed simple learnable embeddings; the cancer-type embedding carried most of the per-gene predictive signal; and contrastive pretraining and expert heads were dispensable. The most predictable genes were enriched for extracellular, immune-response, and cell-adhesion functions. The commonly used per-sample correlation was dominated by the shared global methylation shape (a constant train-mean predictor attains r = 0.94); after centering out this shape, R2M captured substantial sample-specific signal in-distribution (centered r = 0.56) but none in unseen cancer types. External validation confirmed this boundary: cohort-mean methylation profiles were reproduced (r = 0.88 and 0.75), yet sample- and gene-specific predictions did not exceed the train-mean null.

**Conclusions:** R2M enables accurate reconstruction of DNA methylation landscapes from transcriptomes within its trained cancer types and delineates the boundary between predictable between-cancer methylation structure and harder within-cancer and cross-cancer heterogeneity.

**Keywords:** DNA methylation; RNA-seq; cross-modal prediction; deep learning; pan-cancer"""

highlight = """## Highlight Box

**Key findings**
- R2M predicts genome-wide, gene-level DNA methylation from RNA expression across 32 cancer types (mean per-gene r = 0.521) and captures genuine sample-specific signal in-distribution (centered per-sample r = 0.56).
- Learnable gene identity embeddings outperform static Nucleotide Transformer sequence priors; the cancer-type embedding carries most per-gene signal; contrastive pretraining and expert heads are dispensable.
- Out-of-distribution (leave-one-cancer-out and two external meningioma cohorts), R2M reproduces only the cohort-mean methylation profile.

**What is known and what is new?**
- RNA–methylation coupling is established, but genome-scale cross-modal prediction has lacked rigorous, null-referenced evaluation.
- We show that the widely used per-sample correlation is dominated by the shared global methylation shape (a constant predictor attains r = 0.94), introduce centered null-referenced metrics, and map precisely where transcriptome-to-methylome translation stops being sample-informative.

**What is the implication, and what should change now?**
- R2M is suitable for in-distribution methylation imputation within its 32 trained cancer types—not for unseen cancer types or de-novo subtype discovery.
- Methylation-prediction benchmarks should adopt per-gene and centered per-sample correlations with explicit null references."""

ack_foot = """## Acknowledgments

The results shown here are in part based upon data generated by The Cancer Genome Atlas (TCGA) Research Network (https://www.cancer.gov/tcga) and upon data deposited in the Gene Expression Omnibus under accessions GSE189672, GSE189673, GSE183653, and GSE183656. We thank the patients and investigators who contributed these data.

## Footnote

**Reporting Checklist:** The authors have completed the TRIPOD reporting checklist.

**Data Sharing Statement:** All data analyzed in this study are publicly available: TCGA RNA-seq and DNA methylation data via the GDC Data Portal (https://portal.gdc.cancer.gov), and meningioma cohort data from the Gene Expression Omnibus (GSE189672/GSE189673; GSE183653/GSE183656). Trained model weights and analysis code are available from the corresponding author upon reasonable request.

**Funding:** None. [Authors to update if applicable.]

**Conflicts of Interest:** All authors have completed the ICMJE uniform disclosure form. The authors have no conflicts of interest to declare. [To be confirmed by all authors at submission.]

**Ethical Statement:** The authors are accountable for all aspects of the work in ensuring that questions related to the accuracy or integrity of any part of the work are appropriately investigated and resolved. This study analyzed only publicly available, de-identified transcriptomic and DNA methylation data (TCGA and GEO); institutional ethical approval and informed consent were therefore not required."""

# ---------------- AME-style reference list (new numbering) ----------------
references = """## References

1. Jones PA. Functions of DNA methylation: islands, start sites, gene bodies and beyond. Nat Rev Genet 2012;13:484-92.

2. Moore LD, Le T, Fan G. DNA methylation and its basic function. Neuropsychopharmacology 2013;38:23-38.

3. Toyota M, Ahuja N, Ohe-Toyota M, et al. CpG island methylator phenotype in colorectal cancer. Proc Natl Acad Sci U S A 1999;96:8681-6.

4. Cancer Genome Atlas Research Network, Weinstein JN, Collisson EA, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nat Genet 2013;45:1113-20.

5. Bibikova M, Barnes B, Tsan C, et al. High density DNA methylation array with single CpG site resolution. Genomics 2011;98:288-95.

6. Sandoval J, Heyn H, Moran S, et al. Validation of a DNA methylation microarray for 450,000 CpG sites in the human genome. Epigenetics 2011;6:692-702.

7. Kim S, Park HJ, Cui X, et al. Collective effects of long-range DNA methylations predict gene expressions and estimate phenotypes in cancer. Sci Rep 2020;10:3920.

8. Mattesen TB, Andersen CL, Bramsen JB. MethCORR infers gene expression from DNA methylation and allows molecular analysis of ten common cancer types using fresh-frozen and formalin-fixed paraffin-embedded tumor samples. Clin Epigenetics 2021;13:20.

9. Xu J, Shi J, Cui X, et al. Cellular Heterogeneity-Adjusted cLonal Methylation (CHALM) improves prediction of gene expression. Nat Commun 2021;12:400.

10. Yan Y, Chai X, Liu J, et al. DeepMethyGene: a deep-learning model to predict gene expression using DNA methylations. BMC Bioinformatics 2025;26:99.

11. Levy JJ, Titus AJ, Petersen CL, et al. MethylNet: an automated and modular deep learning approach for DNA methylation analysis. BMC Bioinformatics 2020;21:108.

12. Jiang J, Song B, Meng J, et al. Tissue-specific RNA methylation prediction from gene expression data using sparse regression models. Comput Biol Med 2024;169:107892.

13. Levy-Jurgenson A, Tekpli X, Kristensen VN, et al. Predicting methylation from sequence and gene expression using deep learning with attention. bioRxiv 2018:491357.

14. Huang X, Liu Q, Zhao Y, et al. MethylProphet: a generalized gene-contextual model for inferring whole-genome DNA methylation landscape. bioRxiv 2025.

15. Salman M. Inferring DNA methylation from RNA-seq in renal papillary carcinoma using paired multi omics data - a GenAI model. Authorea 2025. doi:10.22541/au.175942409.95172063/v1.

16. Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nat Methods 2025;22:287-97.

17. Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.

18. Colaprico A, Silva TC, Olsen C, et al. TCGAbiolinks: an R/Bioconductor package for integrative analysis of TCGA data. Nucleic Acids Res 2016;44:e71.

19. Loshchilov I, Hutter F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

20. Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc Series B Stat Methodol 2005;67:301-20.

21. Bayley JC 5th, Hadley CC, Harmanci AO, et al. Multiple approaches converge on three biological subtypes of meningioma and extract new insights from published studies. Sci Adv 2022;8:eabm6247.

22. Choudhury A, Magill ST, Eaton CD, et al. Meningioma DNA methylation groups identify biological drivers and therapeutic vulnerabilities. Nat Genet 2022;54:649-59.

23. Raudvere U, Kolberg L, Kuzmin I, et al. g:Profiler: a web server for functional enrichment analysis and conversions of gene lists (2019 update). Nucleic Acids Res 2019;47:W191-8."""

# ---------------- word count of main text ----------------
main_text = body.split('## Introduction', 1)[1]
main_text = main_text.split('## Conclusions')[0] + main_text.split('## Conclusions')[1] if '## Conclusions' in main_text else main_text
wc = len(re.sub(r'[#*]', '', main_text).split())

title_page = f"""# {title}

**Running title:** R2M: transcriptome-to-methylome translation

**Authors:** XiaoZhang Bao¹, [Co-Author Name(s)²]

**Affiliations:** ¹ [Department, Institution, City, Country]; ² [Department, Institution, City, Country]

**Correspondence to:** XiaoZhang Bao, [Degree]. [Full postal address]. Email: [email@institution.edu]; Tel: [telephone number]. ORCID: [0000-0000-0000-0000].

**Word count (main text):** {wc}. **Figures:** 8. **Tables:** 6 main + 2 supplementary.

**Contributions:** (I) Conception and design: X Bao; (II) Administrative support: [to be completed]; (III) Provision of study materials or patients: [to be completed]; (IV) Collection and assembly of data: X Bao; (V) Data analysis and interpretation: X Bao; (VI) Manuscript writing: All authors; (VII) Final approval of manuscript: All authors.
"""

out = '\n\n'.join([title_page, abstract, highlight, body.strip(), ack_foot, references]) + '\n'
OUT.write_text(out)
print('written', OUT)
print('main-text word count:', wc)

# ---------------- verification ----------------
cited = set()
for m in re.finditer(r'\((\d+(?:,\d+)*)\)', out.split('## References')[0]):
    for n in m.group(1).split(','):
        cited.add(int(n))
print('citation numbers used:', sorted(cited))
print('all 1-23 used:', cited == set(range(1, 24)))
print('PMID remaining:', out.count('PMID'))
ab = out.split('## Abstract')[1].split('## Highlight Box')[0]
abwc = len(re.sub(r'[#*]|Keywords.*', '', ab, flags=re.S).split())
print('abstract word count (excl keywords):', abwc)
