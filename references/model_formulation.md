# Model Formulation and Design Notes

## 1. Input representation

Let a biological sample be represented by a cell-by-feature matrix

`X in R^{N x G}`,

where `N` is the number of observed cells and `G` is the number of molecular features or learned channels.

The model should be permutation-invariant with respect to rows of `X`, should accept variable `N`, and should not require the population axis to remain unchanged throughout the network.

The goal of a block is therefore not simply to map `N x G -> N x G`. A useful block may map

`N x G -> M x K`,

with `M << N`, where `M` denotes a sketched or compressed population dimension and `K` denotes learned channels.

## 2. Randomized population filtering

A central primitive is a fixed random sketching matrix

`S in R^{M x N}`,

with entries drawn from a simple distribution such as a normalized Rademacher distribution (`+1/-1`) or a Gaussian distribution.

A basic sketch is

`Y = S X`.

This maps the full population into `M` randomized population probes. Each row of `Y` is a weighted aggregate over cells and can be interpreted as one randomized population filter response.

With normalized Rademacher entries, for example,

`S_ij in {+1/sqrt(M), -1/sqrt(M)}`.

Rademacher sketches are attractive because they are cheap to generate and multiply, have zero mean and controlled second moment, and are standard random embeddings in RandNLA.

The precise distribution is not the scientific contribution and should be treated as an ablation against Gaussian and learned projections.

## 3. Learnable channel transformation

A learnable feature transformation can be placed before or after the sketch.

### Variant A: pre-sketch channel mixing

`H = phi(X W_1 + b_1)`

`Y = S H`

This allows the model to construct nonlinear molecular features before population aggregation.

### Variant B: post-sketch channel mixing

`Y = S X`

`H = phi(Y W_1 + b_1)`

This first applies a fixed population operator and then learns how to interpret the resulting sketch channels.

### Variant C: sandwich block

`H_0 = phi_pre(X)`

`Y = S H_0`

`H_1 = phi_post(Y)`

The manuscript should avoid deciding this by argument alone. These are natural controlled ablations. The minimal architecture should probably begin with a post-sketch MLP because it keeps the population operator structurally clean; a pre-sketch MLP can then test whether learned nonlinear feature construction materially improves performance.

## 4. Second-order population structure

The sketch can be converted into a low-dimensional second-order statistic. For a sketched matrix `Y in R^{M x K}`, one candidate is

`C = (1/M) Y^T Y + eps I`.

Here `C in R^{K x K}` summarizes correlations between learned channels and is independent of the original population size `N` after sketching.

An alternative orientation is

`C_pop = (1/K) Y Y^T + eps I`,

which summarizes relations among sketch filters. The appropriate orientation depends on which axis is intended to be factorized/compressed in the next stage.

The key idea is that second-order statistics transform a variable-size population into a fixed-size positive-semidefinite object. This provides a natural bridge from population data to factorization-inspired neural computation.

## 5. Cholesky factorization

For a symmetric positive-definite matrix `C`, Cholesky factorization computes

`C = L L^T`,

where `L` is lower triangular.

This can be interpreted as constructing a coordinate system for the second-order geometry of the population representation.

A stabilized implementation should use

`C_eps = C + eps I`

before the Cholesky operation.

The model should **not explicitly compute `L^{-1}`**. If a whitening-like operation is required, solve triangular systems instead:

`L Z = Y^T`

or equivalently

`Z = solve_triangular(L, Y^T)`.

This is computationally and numerically preferable to forming an explicit matrix inverse and remains differentiable in modern autodiff frameworks.

## 6. Whitening-inspired transform

A possible factorization-inspired normalized representation is

`C = (1/M) Y^T Y + eps I`

`L = chol(C)`

`Z^T = L^{-1} Y^T`

implemented through a triangular solve.

Then `Z` contains channels transformed relative to the population covariance structure. Conceptually, the operation removes redundant correlation and presents the next layer with normalized latent directions.

The exact normalization convention may be modified, but the important computational motif is:

`sketch -> estimate second-order geometry -> factorize -> solve/normalize`.

## 7. Why Cholesky rather than full SVD or QR everywhere?

Full SVD is conceptually appealing but may be unnecessarily expensive and can introduce difficult gradient behavior near repeated singular values.

QR is also natural for subspace construction, particularly when sketching is used to approximate a range. However, differentiating through repeated QR factorizations can be comparatively heavy for a basic neural block.

Cholesky offers a pragmatic alternative when the block is formulated through a regularized Gram/covariance matrix:

- it is substantially cheaper than a full SVD;
- it operates on a smaller fixed-size matrix after sketching;
- it is deterministic;
- triangular solves avoid explicit inverse computation;
- gradients are well supported when the matrix remains sufficiently positive definite.

This makes Cholesky an attractive initial implementation, while QR/SVD variants can remain ablations.

## 8. Generic SketchNet block

A generic block can therefore be written as follows.

Given

`X_l in R^{N_l x K_l}`,

first apply an optional channel transformation:

`H_l = phi_pre,l(X_l)`.

Apply a population sketch:

`Y_l = S_l H_l`,

where

`S_l in R^{N_{l+1} x N_l}`

and typically `N_{l+1} < N_l`.

Compute a regularized channel Gram matrix:

`C_l = (1/N_{l+1}) Y_l^T Y_l + eps_l I`.

Factorize:

`L_l = chol(C_l)`.

Normalize with a triangular solve:

`Z_l^T = solve(L_l, Y_l^T)`.

Then apply learnable channel mixing:

`X_{l+1} = phi_post,l(Z_l)`.

This gives the structural flow

`X_l -> learnable channels -> random population filters -> second-order geometry -> Cholesky normalization -> learnable channels -> X_{l+1}`.

The population dimension is now explicitly allowed to shrink from `N_l` to `N_{l+1}`.

## 9. Residual and multi-channel variants

Several extensions can be introduced without changing the primitive:

### Multiple sketches

Use several independent sketches `S^(1), ..., S^(B)` and concatenate their outputs. This is analogous to using multiple filter banks and reduces dependence on one random realization.

### Residual feature path

When input and output channel dimensions are compatible, combine transformed statistics with a residual projection of the sketch:

`X_{l+1} = F(Z_l) + R(Y_l)`.

### Learned gating

A small MLP can learn channel-wise gates from the second-order statistic `C_l`, providing data-dependent modulation while keeping the sketch matrix fixed.

### Progressive compression

Use a schedule

`N_0 > N_1 > ... > N_L`,

ending with a compact population representation. This is analogous to progressive spatial reduction in CNNs.

## 10. Subsampling stability as an architectural target

Suppose `X_a` and `X_b` are two random subsamples from the same underlying cell population. The desired population embeddings `f(X_a)` and `f(X_b)` should be close even when the numbers of sampled cells differ substantially.

This suggests an explicit consistency objective:

`L_subsample = d(f(X_a), f(X_b))`,

where `d` may be cosine distance, squared Euclidean distance after normalization, or a distributional alignment loss.

This is more than data augmentation. It directly operationalizes the central architectural claim that the learned representation describes an underlying cell population rather than a particular finite draw of cells.

## 11. Deterministic initial implementation

The first implementation should minimize degrees of freedom:

1. normalize cell-level input features;
2. use fixed normalized Rademacher sketches;
3. use a post-sketch two-layer MLP;
4. compute the channel Gram matrix;
5. add a fixed numerical jitter `eps I`;
6. use differentiable Cholesky factorization;
7. use triangular solves, never an explicit inverse;
8. progressively reduce the sketch/population dimension across blocks;
9. pool the final representation into a fixed-dimensional sample embedding;
10. train with downstream/pretraining objectives plus population-subsampling consistency.

Only after this baseline works should learned sketches, Gaussian sketches, QR/SVD variants, pre-sketch MLPs, gates, or more complicated factorization paths be introduced.

## 12. Key ablations

The model paper should isolate which components matter:

- Rademacher vs Gaussian vs learned projection;
- random sketch vs simple random cell subsampling;
- Cholesky normalization vs no factorization;
- Cholesky vs QR vs SVD-derived normalization;
- pre-sketch MLP vs post-sketch MLP vs both;
- one sketch vs multiple parallel sketches;
- fixed population size vs variable population size during training;
- with vs without subsampling-consistency objective;
- different compression schedules across layers.

The architecture should be judged not only on downstream accuracy but on whether the individual primitives produce the desired inductive properties.
