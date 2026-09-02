# Experimental Setup and Pretraining Plan

## 1. Experimental objective

The experiments should answer a narrow architectural question:

> Does a factorization-inspired, sketch-based population encoder learn biologically useful and subsampling-stable representations of single-cell populations, while scaling favorably with population size?

The benchmark should therefore separate three properties rather than collapse them into one downstream score:

1. biological representation quality;
2. stability to finite sampling of cells;
3. computational scaling and architectural efficiency.

## 2. Initial corpus: Xenium 5K Prime

A practical first corpus is **Xenium 5K Prime** data because it provides a relatively large and consistent gene panel and permits experiments in which gene integration is not the primary difficulty.

The working setup discussed so far is:

- use approximately **4,000 genes as observed model inputs**;
- reserve approximately **1,000 genes as held-out molecular targets**;
- construct training samples as cell populations rather than isolated cells;
- allow the number of cells per sample to vary during training and evaluation.

The exact 4k/1k split should eventually be made deterministic and fixed across methods. The held-out genes should span a range of expression levels and biological functions rather than being chosen only by abundance.

## 3. Population construction

The core input sample should be a biologically coherent collection of cells. Depending on dataset metadata, candidate population definitions include:

- tissue regions;
- fields of view;
- spatial neighborhoods or larger tissue windows;
- condition/donor/sample-level cell collections;
- controlled subsamples from these parent populations.

For the architecture paper, it is important that one parent population can generate multiple stochastic observations with different cell counts. This allows direct measurement of whether the representation captures the underlying population rather than the exact sampled cells.

For a parent population `P`, generate views such as:

- full or high-coverage population;
- 75% cell subsample;
- 50% cell subsample;
- 25% cell subsample;
- optionally lower coverage for stress testing.

These views should preserve the same biological label/metadata.

## 4. Two complementary pretraining regimes

The pretraining study should deliberately separate **supervised** and **self-supervised** settings.

This prevents a strong biological label signal from obscuring whether the proposed operator itself provides useful inductive bias.

### 4.1 Supervised population pretraining

Use available population-level biological metadata to train the encoder directly.

Potential targets include:

- tissue identity;
- donor/sample identity where biologically appropriate;
- disease/control state;
- perturbation or treatment condition;
- anatomical region;
- coarse composition-defined state.

The purpose is not simply classification. Supervision creates a straightforward test of whether SketchNet can aggregate heterogeneous cells into an informative population representation.

A standard objective is

`L_sup = CE(g(f(X)), y)`

for categorical population metadata, or an appropriate regression loss for continuous phenotypes.

A multi-task version may be preferable when several orthogonal biological labels are available.

### 4.2 Self-supervised population pretraining

The self-supervised branch should target properties intrinsic to a population matrix rather than copying masked-language modeling from gene-token models.

The strongest candidate objectives discussed so far are:

#### A. Population-view consistency

Take two random cell subsamples `X_a` and `X_b` from the same parent population and encourage their embeddings to agree.

`L_view = d(f(X_a), f(X_b))`

Negatives can be introduced through contrastive learning, but a non-contrastive consistency objective may be sufficient and avoids making unrelated biological populations artificially repel one another.

This objective directly matches the desired model property.

#### B. Held-out gene prediction

Provide only the 4k observed genes and ask the population representation to predict statistics of the remaining 1k genes.

The target can be defined at several levels:

- population mean expression for each held-out gene;
- detection fraction;
- variance or other distribution summaries;
- cell-type-conditional held-out expression when annotations are available;
- a low-dimensional representation of the held-out gene matrix.

This is attractive because success requires the encoder to preserve biological state that generalizes beyond the molecular dimensions directly observed by the model.

It should be viewed as a **complementary objective**, not the only pretraining task.

#### C. Cross-view held-out reconstruction

A stronger variant combines the two ideas: infer held-out gene summaries for the parent population from a partial cell subsample. The model must then be robust both to missing cells and missing molecular variables.

#### D. Population-statistic reconstruction

Predict stable low-order summaries of the original population from its sketch representation, for example selected moments, covariance summaries, gene-module activities, or cell-type composition.

This explicitly tests whether the sketch preserves useful population geometry.

## 5. Recommended initial self-supervised objective

A simple first objective is

`L_pretrain = lambda_view L_view + lambda_gene L_gene`,

where:

- `L_view` enforces consistency between independently subsampled views of the same population;
- `L_gene` predicts held-out-gene population summaries.

This combination has a clean interpretation:

> The representation should be invariant to which cells happened to be sampled, while retaining enough biological information to predict molecular variables that were never provided as inputs.

It aligns unusually well with the architectural hypothesis and is therefore preferable to adopting conventional random gene masking as the main objective.

## 6. What should be predicted for held-out genes?

Predicting every held-out gene at every cell would turn the task back into a cell-level reconstruction problem and may be inconsistent with the population-level architecture.

The primary target should therefore be **population-level gene statistics**.

A recommended hierarchy is:

### Level 1: mean expression

Predict normalized mean expression of each held-out gene across the population.

This is the simplest task and establishes whether global molecular state is retained.

### Level 2: detection fraction and variance

Predict multiple summaries per gene:

- mean;
- variance;
- fraction of cells with detected expression.

This makes the objective sensitive to population heterogeneity rather than only average abundance.

### Level 3: stratified expression

If stable cell-type labels are available, predict held-out expression summaries conditional on broad cell classes. This tests whether the representation preserves compositional and state information simultaneously.

The manuscript can begin with Levels 1-2 and reserve Level 3 as a biological analysis.

## 7. Supervised downstream evaluation

After pretraining, freeze or fine-tune the population encoder for tasks such as:

- tissue/region classification;
- perturbation prediction;
- phenotype prediction;
- cell-composition regression;
- biological condition retrieval;
- held-out-gene prediction.

Both **linear probing** and **full fine-tuning** should be reported. Linear probing is especially useful because it isolates the quality of the learned representation from the capacity of the downstream head.

## 8. Biological metrics

Accuracy alone is not enough. The representation should be evaluated using biologically meaningful quantitative tests.

### A. Held-out gene predictability

For the 1k unseen genes, report:

- per-gene Pearson/Spearman correlation across populations;
- mean squared or absolute error on normalized population expression;
- performance stratified by gene abundance;
- performance stratified by variability;
- module/pathway-level recovery.

### B. Cell-type composition preservation

Train a simple decoder from the population embedding to broad cell-type proportions and report correlation/error. A good population representation should preserve composition without requiring one embedding per cell.

### C. Biological neighborhood consistency

Populations sharing tissue state, condition, or perturbation should be close in embedding space. This can be quantified with retrieval metrics, kNN label consistency, or rank correlations to known biological similarity.

### D. Gene-program preservation

Compute established gene-program/module scores from the full genes and test whether they can be predicted from the SketchNet population representation.

### E. Differential-signal preservation

For known population contrasts, compare differential signals computed from full data with those inferred from the learned representation or reconstructed held-out genes. Rank correlation of effect sizes is more informative than only classification performance.

## 9. Population-specific architectural metrics

These metrics are especially important because they test what distinguishes SketchNet from generic encoders.

### A. Subsampling stability curve

For each parent population, compute embeddings from progressively smaller cell subsets and compare them with a high-coverage reference embedding.

Plot representation similarity versus retained cell fraction.

This should become a headline metric.

### B. Independent-view variance

Repeatedly subsample the same population at a fixed fraction and measure the variance of the resulting embeddings. Lower variance indicates a more stable estimator of population state.

### C. Between-population / within-population separation

Compare embedding distances between stochastic views of the same population against distances between distinct biological populations. A useful representation should have low within-population sampling variance while retaining high biologically meaningful between-population variance.

### D. Population-size extrapolation

Train using a restricted range of cell counts and evaluate on larger/smaller populations. This tests whether the model genuinely supports variable `N`.

## 10. Computational metrics

Report scaling with number of cells `N` for:

- runtime per forward pass;
- peak GPU memory;
- training throughput;
- inference throughput.

Compare these empirically with Transformer-based population encoders over increasing `N`.

The theoretical motivation predicts that sketch-based operations should avoid the quadratic cell-cell interaction matrix of full attention.

## 11. Baselines

The baseline suite should isolate architectural differences rather than only compare against large foundation models.

Recommended baseline categories:

1. **mean pooling + MLP** — minimal permutation-invariant baseline;
2. **DeepSets-style encoder** — learned cell encoder followed by symmetric pooling;
3. **Set Transformer / attention pooling** — generic set architecture;
4. **Transformer over cells** — attention-based population encoder when computationally feasible;
5. **PCA/SVD-derived population representation** — classical factorization baseline;
6. **random projection + pooling** — tests whether sketching alone explains performance;
7. **SketchNet without Cholesky/factorization** — critical internal ablation.

If more domain-specific single-cell foundation models are added later, they should supplement rather than replace these mechanistic baselines.

## 12. Core ablation matrix

The paper should include at least the following:

- fixed random Rademacher sketch vs Gaussian sketch;
- random sketch vs random cell subsampling;
- random sketch vs learned linear projection;
- no factorization vs Cholesky normalization;
- Cholesky vs QR/SVD variant where computationally feasible;
- pre-sketch MLP vs post-sketch MLP vs both;
- one block vs multiple stacked blocks;
- different population compression schedules;
- with vs without subsampling-consistency loss;
- with vs without held-out-gene prediction;
- supervised-only vs self-supervised-only vs combined pretraining.

## 13. Minimal first manuscript experiment

A deterministic initial experiment can be kept small:

1. select one Xenium 5K Prime corpus with a consistent gene panel;
2. define a fixed 4k observed / 1k held-out gene split;
3. construct biologically coherent parent populations;
4. generate variable-size random cell views from each parent population;
5. train a 2-3 block SketchNet with fixed Rademacher sketches and Cholesky normalization;
6. optimize view-consistency plus held-out-gene-summary prediction;
7. compare with mean pooling, DeepSets, PCA/SVD, and a Set Transformer;
8. evaluate held-out-gene recovery and population metadata prediction;
9. report subsampling stability from 100% down to 25% cells;
10. benchmark runtime and memory as population size increases.

If this experiment succeeds, it directly supports the central paper premise without requiring a very large foundation-model-scale training run.

## 14. Manuscript-level experimental narrative

A coherent Results progression would be:

1. **SketchNet learns stable population representations.** Show subsampling stability and variable-cell-count robustness.
2. **SketchNet preserves biological information.** Show population phenotype/composition prediction and held-out-gene recovery.
3. **Factorization-inspired components matter.** Present sketch/factorization/MLP ablations.
4. **SketchNet scales favorably with population size.** Compare memory and runtime with attention-based set encoders.
5. **Self-supervised population pretraining transfers.** Compare supervised-only, self-supervised, and combined regimes on downstream tasks.

This structure keeps the manuscript centered on the proposed building block rather than on a single biological application.
