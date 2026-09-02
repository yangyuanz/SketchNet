# Project Background and Manuscript Positioning

## 1. Research question

The project starts from a structural question rather than from a particular downstream benchmark:

> **Can we build reusable neural-network blocks for single-cell population data, different from Transformers, whose operations are composable and mimic the useful structure of matrix factorization and randomized numerical linear algebra?**

The intended contribution is not simply another encoder for cells. The goal is to identify a small set of reusable operations that can play, for single-cell modeling, a role analogous to convolutional blocks in vision.

A useful block should therefore be:

- **population-aware**: it should treat an unordered set of cells as the sample;
- **size-flexible**: the representation should not require the population axis to remain fixed at `N` throughout the network;
- **subsampling-stable**: randomly observing fewer cells from the same underlying population should not radically alter the learned representation;
- **composable**: blocks should stack naturally to form deeper models;
- **computationally scalable**: operations should avoid quadratic attention over all cells when possible;
- **structurally interpretable**: the block should have a clear connection to familiar operations such as sketching, covariance estimation, whitening, low-rank approximation, QR, Cholesky, or SVD-like factorization.

## 2. Why not treat cells simply as tokens?

Transformers are powerful because self-attention is an extremely flexible interaction mechanism. For single-cell data, however, that flexibility comes with relatively weak domain-specific inductive bias. A Transformer naturally treats cells as a sequence or set of tokens and learns pairwise interactions through attention. This is useful, but it does not explicitly encode the fact that many classical analyses of cell populations are based on **low-rank structure, covariance structure, subspace recovery, matrix factorization, and randomized summaries of large matrices**.

The motivating hypothesis is that these classical operations reveal something important about the geometry of cell-population data that can be converted into neural building blocks.

The proposed direction therefore does not argue that attention is ineffective. Instead, it asks whether an alternative family of blocks can provide:

- stronger population-level inductive bias;
- better scaling with the number of cells;
- natural invariance to cell ordering;
- explicit compression of the population axis;
- more interpretable intermediate operations.

## 3. Analogy to convolutional neural networks

The CNN analogy is central.

Images are not naturally modeled by a generic fully connected network over all pixels. Convolution succeeds because it introduces a reusable primitive with strong structure: **local filtering with shared weights**, followed by channel mixing and spatial reduction. This primitive can be stacked repeatedly.

The corresponding question for cell populations is:

> What is the equivalent of a convolutional filter bank for an unordered population matrix?

The current proposal is that **randomized sketching and factorization-inspired operators** can act as population filters.

For an input matrix `X` with cells along one axis and molecular features/channels along the other, a randomized projection can probe directions in the population, while second-order summaries and factorization operations can convert these probes into stable subspace information. Learnable channel transformations can then refine the representation before the next block.

The analogy can be summarized as:

| Vision | SketchNet concept |
| --- | --- |
| Pixel grid | Cell population |
| Convolutional kernel | Random/sketched population filter |
| Local receptive field | Randomized population projection / sampled subspace |
| Feature maps | Sketch / factor channels |
| Pooling / striding | Population-axis compression |
| Channel MLP / 1x1 convolution | Learnable feature mixing |
| Hierarchical spatial representation | Hierarchical population representation |

The analogy is conceptual rather than literal: cell populations do not have a Euclidean grid, so the reusable operator must derive its structure from permutation-invariant linear algebra rather than locality.

## 4. Population-level sample definition

A key design choice is that the **population itself is the input sample**. The number of cells `N` may vary across samples and should be allowed to shrink across layers. The network should therefore not be designed around maintaining an `N x K` representation at every stage.

This is important for two reasons.

First, many biological tasks are naturally population-level: tissue state, perturbation condition, donor phenotype, disease state, treatment response, and distributional differences are properties of a collection of cells rather than of one cell in isolation.

Second, forcing every layer to retain one embedding per original cell prevents the architecture from expressing progressive population compression in the same way that CNNs progressively reduce spatial resolution.

The desired architecture should therefore permit a sequence such as

`large population -> sketched population representation -> lower-rank population representation -> compact population embedding`.

## 5. Why randomized numerical linear algebra?

Randomized numerical linear algebra (RandNLA) provides a natural source of primitives because it is explicitly designed to summarize large matrices while preserving important subspace structure.

Random Rademacher or Gaussian projections can be used to create low-dimensional sketches that approximately preserve norms, pairwise geometry, second moments, or dominant subspaces. These operations have several properties that are attractive for single-cell modeling:

- their cost can scale approximately linearly in the number of cells;
- they naturally compress the population axis;
- they provide many parallel random probes, analogous to a filter bank;
- repeated sketches can approximate the dominant structure of the original population matrix;
- the random map itself need not be learned, reducing parameter count and supplying a fixed inductive bias.

The neural component can then learn how to transform, combine, gate, or reweight the resulting sketch channels.

## 6. Why factorization-inspired operations?

Classical methods such as PCA, SVD, NMF, covariance analysis, and low-rank matrix factorization have historically been effective in single-cell analysis because expression matrices often contain structured low-dimensional variation embedded in high-dimensional noisy measurements.

The proposal is not to insert an exact SVD into every layer. Instead, it is to extract the **computational motifs** behind factorization:

- probe a subspace;
- estimate second-order structure;
- normalize or whiten correlated directions;
- separate or rotate latent channels;
- reduce dimensionality;
- repeat.

A neural block constructed from these motifs may retain the interpretability of factorization while gaining the flexibility of deep learning.

## 7. Working architectural principle

The current architecture can be summarized at a high level as:

1. transform molecular channels with an optional MLP;
2. apply a randomized population sketch/filter bank;
3. construct a low-dimensional second-order statistic from the sketched representation;
4. factorize or whiten that statistic, with Cholesky currently the most attractive deterministic option;
5. apply learnable channel mixing/gating;
6. optionally compress the population/sketch dimension further;
7. stack multiple blocks.

Two design branches remain deliberately open:

- **pre-filter MLP vs post-filter MLP**: whether nonlinear channel mixing should occur before randomized population filtering, after it, or both;
- **factorization implementation**: whether Cholesky should be used directly, whether solves should replace explicit inverses, and how gradients should be stabilized.

These should be treated as controlled architectural ablations rather than unresolved implementation ambiguity.

## 8. Working name

The repository uses **SketchNet** as the working project name.

The name captures the central mechanism: the model repeatedly constructs informative, low-dimensional sketches of a cell population and transforms them through learnable factorization-inspired blocks.

A manuscript-style subtitle could be:

> **SketchNet: Factorization-Inspired Neural Building Blocks for Single-Cell Populations**

Alternative emphasis:

> **SketchNet: Composable Neural Operators for Population-Level Single-Cell Learning**

## 9. Core manuscript claim to test

The strongest manuscript claim should not be that SketchNet universally outperforms Transformers. A more defensible and scientifically interesting claim is:

> **Population-level neural blocks derived from randomized sketching and factorization can provide a reusable inductive bias for single-cell learning, producing compact and subsampling-stable representations while remaining competitive with generic attention-based architectures.**

The experiments should therefore explicitly test three axes:

1. **representation quality** — does the model learn useful biological information?
2. **population stability** — does the representation remain consistent under cell subsampling and varying population size?
3. **architectural efficiency and interpretability** — can the model achieve useful performance with favorable scaling and understandable intermediate operators?
