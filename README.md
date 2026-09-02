# SketchNet

SketchNet is an exploratory research project on **factorization-inspired neural building blocks for single-cell population modeling**.

The central question is:

> Can we build reusable neural-network blocks for single-cell data that are composable like convolutional blocks, but whose inductive bias is derived from randomized numerical linear algebra and matrix factorization rather than token-wise attention?

The project treats a **cell population**, rather than an individual cell, as the basic sample. The goal is to learn population-level representations that remain stable when the number of observed cells changes and that can be progressively compressed into fixed-size representations for downstream prediction.

## Current concept

A SketchNet block is intended to combine:

1. **Randomized sketching / filtering** across cells, inspired by RandNLA.
2. **Second-order population summaries** that capture covariance-like structure.
3. **Stable factorization or whitening operations**, such as Cholesky-based transforms.
4. **Learnable channel mixing**, typically through MLPs.
5. **Progressive reduction of the population axis**, so intermediate representations need not remain `N x K`.

The analogy is to CNNs: convolution gives image models a reusable local filtering primitive; SketchNet seeks an analogous reusable primitive for unordered cell populations, with matrix factorization and randomized sketching supplying the inductive bias.

## Manuscript reference notes

The current discussion has been consolidated into:

- [`references/project_background.md`](references/project_background.md) — motivation, research question, CNN/Transformer comparison, and manuscript positioning.
- [`references/model_formulation.md`](references/model_formulation.md) — mathematical design principles, candidate block operations, sketching, second-order statistics, and Cholesky handling.
- [`references/experimental_setup.md`](references/experimental_setup.md) — proposed Xenium 5K Prime setup, gene split, supervised/self-supervised objectives, ablations, and evaluation metrics.

These are working research notes rather than finalized manuscript text. They are intended to preserve the design rationale and make later drafting of the Introduction, Methods, and Experiments sections deterministic.
