# TensorLogic SklearRS Kernels — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Kernel methods bridging tensor logic with sklearRS kernel approximations.

## Completed

- [x] Basic crate structure
- [x] **Logic-derived similarity kernels**
  - [x] Rule-based similarity (RuleSimilarityKernel)
  - [x] Predicate overlap kernel (PredicateOverlapKernel)
- [x] **Tensor-based kernels**
  - [x] Linear kernel
  - [x] RBF (Gaussian) kernel
  - [x] Polynomial kernel
  - [x] Cosine similarity kernel
  - [x] Laplacian kernel
  - [x] Sigmoid (Tanh) kernel
  - [x] Chi-squared kernel
  - [x] Histogram Intersection kernel
  - [x] Matern kernel (nu=0.5, 1.5, 2.5)
  - [x] Rational Quadratic kernel
  - [x] Periodic kernel
- [x] **Kernel transformation utilities**
  - [x] Kernel matrix normalization
  - [x] Kernel matrix centering (for kernel PCA)
  - [x] Kernel matrix standardization
  - [x] NormalizedKernel wrapper
- [x] **Kernel utilities for ML workflows**
  - [x] Kernel-target alignment (KTA) for kernel selection
  - [x] Median heuristic bandwidth selection
  - [x] Kernel matrix validation
  - [x] Gram matrix computation utilities
  - [x] Row normalization
- [x] Implement SkleaRS-compatible kernel trait
- [x] Efficient kernel matrix computation
- [x] Comprehensive test suite (408 tests)
- [x] Extensive documentation and examples
- [x] Zero warnings (clippy clean)

## Advanced Kernel Types - COMPLETE

- [x] **Graph kernels from TLExpr**
  - [x] Subgraph matching kernel (SubgraphMatchingKernel)
  - [x] Walk-based kernels (RandomWalkKernel)
  - [x] Weisfeiler-Lehman kernel (WeisfeilerLehmanKernel)
- [x] **Tree kernels for structured data**
  - [x] Subtree kernel (SubtreeKernel)
  - [x] Subset tree kernel (SubsetTreeKernel)
  - [x] Partial tree kernel (PartialTreeKernel)
- [x] **Composite kernels**
  - [x] Weighted sum of kernels (WeightedSumKernel)
  - [x] Product kernels (ProductKernel)
  - [x] Kernel alignment (KernelAlignment)

## Performance Optimizations - COMPLETE

- [x] Sparse kernel matrix support (SparseKernelMatrix, CSR format, builder pattern)
- [x] Kernel caching (CachedKernel, KernelMatrixCache)
- [x] **Low-rank approximations (Nystrom method)**
  - [x] Three sampling methods (Uniform, First, K-means++)
  - [x] Configurable regularization
  - [x] Compression ratio tracking
- [x] **Performance benchmarks** (5 benchmark suites, 47 groups)
  - [x] Kernel computation benchmarks (10 groups)
  - [x] Matrix operations benchmarks (10 groups)
  - [x] Caching performance benchmarks (8 groups)
  - [x] Composite kernels benchmarks (10 groups)
  - [x] Graph kernels benchmarks (9 groups)
- [x] **Online kernel updates**
  - [x] OnlineKernelMatrix - Incremental O(n) updates
  - [x] WindowedKernelMatrix - Sliding window for time series
  - [x] ForgetfulKernelMatrix - Exponential decay for concept drift
  - [x] AdaptiveKernelMatrix - Automatic bandwidth adjustment
  - [x] Comprehensive tests (25 tests)

## Advanced Kernel Methods - COMPLETE

- [x] **String kernels for text data** (NGramKernel, SubsequenceKernel, EditDistanceKernel)
- [x] **Tree kernels for structured data** (Subtree, Subset, Partial)
- [x] **Multi-task kernel learning**
  - [x] IndexKernel - Task-based similarity
  - [x] ICMKernel - Intrinsic Coregionalization Model (B tensor K)
  - [x] LMCKernel - Linear Model of Coregionalization (Sigma B_q tensor K_q)
  - [x] HadamardTaskKernel - Element-wise product
  - [x] MultiTaskKernelBuilder - Builder pattern
  - [x] Comprehensive tests (30 tests)
- [x] **Automatic feature extraction** from TLExpr (FeatureExtractor)
- [x] **Provenance tracking for kernel computations**
  - [x] ProvenanceRecord with rich metadata
  - [x] ProvenanceTracker with query interface
  - [x] ProvenanceKernel wrapper
  - [x] JSON export/import
  - [x] Performance statistics
  - [x] Tagged experiments
  - [x] Comprehensive tests (15 tests)
- [x] **Symbolic kernel composition**
  - [x] KernelExpr with algebraic operations (scale, add, multiply, power)
  - [x] SymbolicKernel for expression evaluation
  - [x] KernelBuilder for declarative construction
  - [x] Expression simplification
  - [x] PSD property checking
  - [x] Comprehensive tests (14 tests)

## Beta.1 Enhancements - COMPLETE

### ARD (Automatic Relevance Determination) Kernels
- [x] **ArdRbfKernel** - ARD version of RBF/Gaussian kernel
  - [x] Per-dimension length scales
  - [x] Signal variance parameter
  - [x] Gradient computation for hyperparameter optimization
- [x] **ArdMaternKernel** - ARD Matern kernel (nu=0.5, 1.5, 2.5)
  - [x] Exponential, nu_3_2, nu_5_2 convenience constructors
- [x] **ArdRationalQuadraticKernel** - ARD Rational Quadratic
- [x] Comprehensive tests (35+ tests)

### GP Utility Kernels
- [x] **WhiteNoiseKernel** - i.i.d. observation noise
- [x] **ConstantKernel** - Constant covariance
- [x] **DotProductKernel** - Linear kernel with variance and bias
- [x] **ScaledKernel** - Generic wrapper to scale any kernel

### Spectral Kernels
- [x] **SpectralMixtureKernel** - Mixture of spectral components
  - [x] SpectralComponent with weight, mean frequency, variance
  - [x] Multi-dimensional support
  - [x] Multiple component composition
- [x] **ExpSineSquaredKernel** - Periodic kernel (scikit-learn compatible)
- [x] **LocallyPeriodicKernel** - RBF x Periodic for decaying periodicity
- [x] **RbfLinearKernel** - RBF x Linear product kernel
- [x] Comprehensive tests (25+ tests)

### Kernel Selection and Cross-Validation
- [x] **KernelSelector** - Comprehensive kernel selection utilities
  - [x] kernel_target_alignment() - KTA metric
  - [x] centered_kernel_target_alignment() - Centered KTA
  - [x] compare_kernels_kta() - Compare multiple kernels
  - [x] loo_error_estimate() - Leave-one-out error
  - [x] k_fold_cv() - K-fold cross-validation
  - [x] grid_search_rbf_gamma() - RBF gamma optimization
- [x] **KFoldConfig** - K-fold CV configuration with shuffle
- [x] **CrossValidationResult** - Fold scores with statistics
- [x] **KernelComparison** - Multi-kernel comparison results
- [x] **GammaSearchResult** - Grid search results
- [x] Comprehensive tests (20+ tests)

### Random Fourier Features (RFF)
- [x] **RandomFourierFeatures** - O(nd) approximate kernel computation
  - [x] Support for RBF, Laplacian, Matern kernels
  - [x] Configurable number of components
  - [x] Transform and approximate_kernel methods
- [x] **OrthogonalRandomFeatures** - Improved variance via orthogonal projection
- [x] **NystroemFeatures** - Nystrom-based feature approximation
- [x] **RffConfig** - Configuration with seed support
- [x] **KernelType** (RffKernelType) - Enum for supported kernel types
- [x] Comprehensive tests (10+ tests)

### Kernel Gradient Computation
- [x] **Element-wise gradients** for standard kernels
  - [x] RbfKernel: compute_with_gradient(), compute_with_length_scale_gradient()
  - [x] PolynomialKernel: compute_with_constant_gradient(), compute_with_all_gradients()
  - [x] MaternKernel: compute_with_length_scale_gradient() (nu=0.5, 1.5, 2.5)
  - [x] LaplacianKernel: compute_with_gradient(), compute_with_sigma_gradient()
  - [x] RationalQuadraticKernel: compute_with_length_scale_gradient(), compute_with_alpha_gradient()
- [x] **Matrix-level gradient computation** (gradient module)
  - [x] compute_rbf_gradient_matrix() - Full NxN gradient matrices
  - [x] compute_polynomial_gradient_matrix()
  - [x] compute_matern_gradient_matrix()
  - [x] compute_laplacian_gradient_matrix()
  - [x] compute_rational_quadratic_gradient_matrix()
  - [x] KernelGradientMatrix, GradientComponent structs
  - [x] trace_product(), frobenius_norm() utilities
- [x] Comprehensive tests (30+ tests)

### Kernel PCA (KPCA)
- [x] **KernelPCA** - Full KPCA implementation
  - [x] fit() - Fit model to training data
  - [x] transform() - Project new data
  - [x] transform_training() - Project training data
  - [x] eigenvalues() - Access eigenvalues
  - [x] explained_variance_ratio() - Variance explained per component
  - [x] cumulative_variance_explained() - Cumulative variance
- [x] **KernelPCAConfig** - Configuration with centering option
- [x] **center_kernel_matrix()** - Utility function
- [x] **select_n_components()** - Automatic component selection
- [x] **reconstruction_error()** - Error analysis
- [x] Comprehensive tests (11 tests)

## Documentation - COMPLETE

- [x] Add README.md with architecture overview
- [x] Kernel design guide
- [x] **Performance benchmarks** (5 benchmark suites, 47 groups)
- [x] Case studies (SVM) — `src/svm/`: `SvcModel`/`SvcFitted` (C-SVM binary+multiclass OvR via SMO), `SvrModel`/`SvrFitted` (ε-SVR via augmented SMO dual), `SmoConfig`; Platt-1998 SMO with Keerthi two-loop heuristics, error cache, KKT convergence; 24 tests

---

**Total Items:** 54 tasks (all complete)
**Completion:** 100% (54/54) - ALL TASKS COMPLETE

**Test Count:** 451 tests (100% passing, zero warnings)

**Status:** Production-ready (v0.1.0 Stable)
**Release Date:** 2026-03-06 (stable: 2026-04-06)

## v0.1.6 Enhancements (2026-03-30)

- [x] **Kernel Matrix Cache + Batch Compute** (`batch.rs`): `KernelCache` (LRU with symmetric key normalization), `BatchKernelComputer` (O(n²/2) with optional caching), `GramMatrix` (symmetry check, trace, Frobenius norm, PSD diagonal check), `KernelMatrixStats`. 18 new tests.

## v0.1.18 (2026-04-06)

- [x] **Kernel Alignment** (`kernel_alignment.rs`): `KernelMatrix` with centering (`center()`), Frobenius inner product (`frobenius_inner()`), Frobenius norm, and trace operations; `KernelTargetAlignment` (KTA) — uncentered alignment between a kernel matrix and an ideal label kernel; `CenteredKernelAlignment` (CKA) — HSIC-normalized centered KTA robust to different-scale kernels; `HilbertSchmidtIndependenceCriterion` (HSIC) — statistical independence criterion between two kernel matrices; `AlignmentResult` carrying raw score, centered score, and HSIC value; `KernelAlignmentGridSearch` exhausts a parameter grid scoring each candidate by CKA; `KernelAlignmentGradientAscent` performs gradient-free gradient-ascent over a scalar parameter using finite differences to maximize alignment; `AlignmentError` typed error enum for dimension mismatches and degenerate (zero-norm) matrices.

## Future Enhancements

- [ ] GPU acceleration
- [x] Case studies (SVM) — see above (`src/svm/`)
- [ ] SkleaRS integration (feature-gated, currently behind `sklears` feature)

## v0.2.0 Research Preview (2026-04-15)

- [x] **Learned kernel composition** (`learned_composition/`):
  `LearnedMixtureKernel` computes `K_mix = sum_i softmax(w)_i * K_i` over
  any library of `Arc<dyn Kernel>` (including `SymbolicKernel` from
  `KernelBuilder`). Gradient with respect to logits uses the clean
  softmax identity `dK_mix/dw_i = p_i * (K_i - K_mix)`, validated against
  central-difference finite differences to `< 1e-4`. `TrainableKernelMixture`
  adapter exposes `evaluate_with_gradient`, `step(grad, lr)` for
  `tensorlogic-train`-style outer loops; `LearnedMixtureBuilder` provides
  fluent assembly. 12 unit tests + 1 end-to-end integration test
  (mixture over RBF(γ=0.5) + RBF(γ=2.0) converges toward the target
  bandwidth over 400 gradient-descent steps).

- [x] **Deep Kernel Learning** (`deep_kernel/`): `DeepKernel<F, K>`
  composes a base `Kernel` with a differentiable `NeuralFeatureMap`
  feature extractor, evaluating `K_DKL(x, y) = K_base(g_θ(x), g_θ(y))`
  (Wilson et al., 2016). Reference extractor is `MLPFeatureExtractor`:
  stacked `DenseLayer`s with ReLU / Tanh / Identity activations,
  Xavier/Glorot-normal init via `scirs2_core::random::StdRng`, biases
  initialised to zero, and a flat-parameter / per-layer-weights
  double view (`parameters_mut` + `sync_from_flat`) usable by outer
  optimisers. Analytical gradient `∂K_DKL/∂θ` is provided for the
  RBF-base case (`rbf_dkl_gradient`) via MLP backprop; every extractor
  supports `finite_difference_gradient` as a reference / fallback.
  `DeepKernelBuilder` offers fluent topology assembly. 28 unit tests
  (MLP forward, ReLU, Xavier bounds, identity-MLP equals base, finite-
  difference gradient check `< 1e-3`, PSD propagation, empty-extractor
  errors, sync round-trip) + 1 integration test (2-layer MLP + RBF on a
  6-point dataset — Gram matrix symmetric, diagonal = 1, Cholesky
  succeeds after a `1e-9` ridge).

- [x] **Kernel PCA** (`kernel_pca/`): Scholkopf-Smola-Muller (1998)
  implementation. `KernelPCA<K: Kernel + Clone + 'static>` with
  `fit` / `fit_transform` / `transform`. Double-centering via
  `centering::double_center`, symmetric eigendecomp via
  `scirs2_linalg::eigh` selecting top-k eigenpairs, and Scholkopf-style
  scaling `alpha_k = v_k / sqrt(lambda_k)` for out-of-sample projection.
  `FittedKernelPCA` stores the kernel as `Box<dyn Kernel>`, training
  data, centering stats, and provides `explained_variance_ratio()`.
  8 unit tests (linear KPCA recovers PCA, RBF two-cluster separation,
  double-center sums, transform-vs-fit_transform consistency, error paths,
  explained-variance-ratio summation) + 2 integration tests (Swiss-roll
  RBF 80-point embedding, collinear linear-KPCA dominance).

## v0.2.0 / Future Work

- [x] **Multi-output / vector-valued kernels** (`multi_output/`): `MultiOutputKernel`
  trait returns `p×p` `Array2<f64>` blocks; `KroneckerICMKernel` and
  `KroneckerLMCKernel` wrap the existing scalar `ICMKernel`/`LMCKernel` to
  implement the trait; `VvgpModel`/`VvgpFitted` provide full vector-valued GP
  fit+predict via Cholesky with O(N·p) posterior mean and covariance.
  10 unit tests + 10 integration tests (all passing, zero warnings).
