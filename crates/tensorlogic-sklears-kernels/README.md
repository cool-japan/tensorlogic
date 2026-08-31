# tensorlogic-sklears-kernels
[![Crate](https://img.shields.io/badge/crates.io-tensorlogic-sklears-kernels-orange)](https://crates.io/crates/tensorlogic-sklears-kernels)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-sklears-kernels)
[![Tests](https://img.shields.io/badge/tests-564%2F564-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

**Logic-derived similarity kernels for machine learning integration**

This crate provides kernel functions that measure similarity based on logical rule satisfaction patterns, enabling TensorLogic to integrate with traditional machine learning algorithms (SVMs, kernel PCA, kernel ridge regression, etc.).

## Features

### Logic, Graph & Tree Kernels
- **Rule Similarity Kernels** - Measure similarity by rule satisfaction agreement
- **Predicate Overlap Kernels** - Similarity based on shared true predicates
- **Graph Kernels** - Subgraph matching, random walk, Weisfeiler-Lehman kernels
- **Tree Kernels** - Subtree, subset tree, and partial tree kernels for hierarchical data
- **TLExpr Conversion** - Automatic graph and tree extraction from logical expressions

### Classical Kernels
- **Linear Kernel** - Inner product in feature space
- **RBF (Gaussian) Kernel** - Infinite-dimensional feature mapping
- **Polynomial Kernel** - Polynomial feature relationships
- **Cosine Similarity** - Angle-based similarity
- **Laplacian Kernel** - L1 distance, robust to outliers
- **Sigmoid Kernel** - Neural network inspired (tanh)
- **Chi-Squared Kernel** - For histogram data
- **Histogram Intersection** - Direct histogram overlap

### Advanced Gaussian Process Kernels
- **Matérn Kernel** - Generalized RBF with smoothness control (nu=0.5, 1.5, 2.5)
- **Rational Quadratic Kernel** - Scale mixture of RBF kernels
- **Periodic Kernel** - For seasonal and cyclic patterns

### ARD (Automatic Relevance Determination) Kernels
- **ArdRbfKernel** - ARD version of RBF/Gaussian kernel with per-dimension length scales
- **ArdMaternKernel** - ARD Matern kernel (nu=0.5, 1.5, 2.5)
- **ArdRationalQuadraticKernel** - ARD Rational Quadratic kernel
- **KernelGradient** - Gradient computation for hyperparameter optimization

### GP Utility Kernels
- **WhiteNoiseKernel** - i.i.d. observation noise (K(x,y) = sigma^2 if x==y, else 0)
- **ConstantKernel** - Constant covariance (K(x,y) = sigma^2)
- **DotProductKernel** - Linear kernel with variance and bias
- **ScaledKernel** - Generic wrapper to scale any kernel

### Spectral Kernels
- **SpectralMixtureKernel** - Mixture of spectral components for pattern discovery
- **ExpSineSquaredKernel** - Periodic kernel (scikit-learn compatible)
- **LocallyPeriodicKernel** - RBF x Periodic for decaying periodicity
- **RbfLinearKernel** - RBF x Linear product kernel

### Kernel Selection and Cross-Validation
- **KernelSelector** - Kernel target alignment (KTA), leave-one-out error, K-fold CV
- **KFoldConfig** - K-fold CV configuration with shuffle support
- **CrossValidationResult** - Fold scores with statistics
- **KernelComparison** - Multi-kernel comparison results
- **GammaSearchResult** - Grid search results for RBF gamma

### Random Fourier Features (Scalable Kernel Approximation)
- **RandomFourierFeatures** - O(nd) approximate kernel computation
- **OrthogonalRandomFeatures** - Improved variance via orthogonal projection
- **NystroemFeatures** - Nystrom-based feature approximation
- Supports RBF, Laplacian, Matern kernels

### Kernel Gradient Computation
- Element-wise gradients for RBF, Polynomial, Matern, Laplacian, RationalQuadratic
- Matrix-level gradient computation (dK/dTheta)
- KernelGradientMatrix and GradientComponent structures
- Utilities for Gaussian Process hyperparameter optimization

### Kernel PCA (KPCA)
- **KernelPCA** - Full KPCA implementation with fit/transform interface
- Eigenvalue-based variance analysis with explained_variance_ratio
- Automatic component selection via select_n_components
- Reconstruction error analysis

### Online Kernel Updates
- **OnlineKernelMatrix** - Incremental O(n) updates
- **WindowedKernelMatrix** - Sliding window for time series
- **ForgetfulKernelMatrix** - Exponential decay for concept drift
- **AdaptiveKernelMatrix** - Automatic bandwidth adjustment

### Multi-Task Kernel Learning
- **IndexKernel** - Task-based similarity
- **ICMKernel** - Intrinsic Coregionalization Model (B tensor K)
- **LMCKernel** - Linear Model of Coregionalization
- **HadamardTaskKernel** - Element-wise product
- **MultiTaskKernelBuilder** - Builder pattern for multi-task kernels

### Batch Kernel Computation and Gram Matrix (v0.1.6)
- **KernelCache** - LRU cache with symmetric key normalization and hit rate statistics
- **BatchKernelComputer** - O(n²/2) Gram matrix computation with optional caching
- **GramMatrix** - Symmetry check, trace, Frobenius norm, PSD diagonal check
- **KernelMatrixStats** - Aggregate statistics for computed kernel matrices

### Kernel Alignment (v0.1.18)
- **KernelMatrix** - Kernel matrix with centering, Frobenius inner product, Frobenius norm, and trace operations
- **KernelTargetAlignment (KTA)** - Uncentered alignment between a kernel matrix and an ideal label kernel
- **CenteredKernelAlignment (CKA)** - HSIC-normalized centered KTA robust to different-scale kernels
- **HilbertSchmidtIndependenceCriterion (HSIC)** - Statistical independence criterion between two kernel matrices
- **AlignmentResult** - Raw score, centered score, and HSIC value
- **KernelAlignmentGridSearch** - Exhaustive parameter grid search scored by CKA
- **KernelAlignmentGradientAscent** - Finite-difference gradient-free ascent to maximize alignment
- **AlignmentError** - Typed error enum for dimension mismatches and degenerate matrices

### Composite and Performance Features
- **Weighted Sum Kernels** - Combine multiple kernels with weights
- **Product Kernels** - Multiplicative kernel combinations
- **Kernel Caching** - LRU cache with hit rate statistics (CachedKernel, KernelMatrixCache)
- **Sparse Matrices** - CSR format for memory-efficient storage (SparseKernelMatrix)
- **Low-Rank Approximations** - Nystrom method with three sampling strategies
- **Performance Benchmarks** - 5 benchmark suites with 47 benchmark groups

### Text and Feature Processing
- **String Kernels** - N-gram, subsequence, edit distance kernels
- **Feature Extraction** - Automatic TLExpr to vector conversion (FeatureExtractor)
- **Vocabulary Building** - Predicate-based feature encoding

### Kernel Transformations
- **Matrix Normalization** - Normalize to unit diagonal
- **Matrix Centering** - Center for kernel PCA
- **Matrix Standardization** - Combined normalization + centering
- **NormalizedKernel** - Auto-normalizing wrapper

### Provenance Tracking
- **Automatic Tracking** - Track all kernel computations transparently
- **Rich Metadata** - Timestamps, computation time, input/output dimensions
- **Query Interface** - Filter by kernel type, tags, or time range
- **JSON Export/Import** - Serialize provenance for analysis and archival
- **Performance Analysis** - Aggregate statistics and profiling
- **Tagged Experiments** - Organize computations with custom tags

### Symbolic Kernel Composition
- **KernelExpr** - Build kernels using algebraic operations (scale, add, multiply, power)
- **SymbolicKernel** - Evaluate symbolic expressions
- **KernelBuilder** - Declarative builder pattern for readability
- Expression simplification and PSD property checking

### Quality Assurance
- **564 Tests** - Comprehensive test coverage (100% passing)
- **Zero Warnings** - Strict code quality enforcement (clippy clean)
- **Type-Safe API** - Builder pattern with validation
- **Production Ready** - Battle-tested implementations
- **Pure Rust RNG** - `rand_09`/`rand_distr_05` removed from `sklears` feature; all random number generation uses `scirs2_core::random`

## Quick Start

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, RbfKernel, RbfKernelConfig,
    RuleSimilarityKernel, RuleSimilarityConfig,
    Kernel,
};
use tensorlogic_ir::TLExpr;

// Linear kernel for baseline
let linear = LinearKernel::new();
let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let sim = linear.compute(&x, &y).unwrap();

// RBF (Gaussian) kernel
let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
let sim = rbf.compute(&x, &y).unwrap();

// Logic-based similarity
let rules = vec![
    TLExpr::pred("rule1", vec![]),
    TLExpr::pred("rule2", vec![]),
];
let config = RuleSimilarityConfig::new();
let logic_kernel = RuleSimilarityKernel::new(rules, config).unwrap();
let sim = logic_kernel.compute(&x, &y).unwrap();
```

## Kernel Matrix Computation

All kernels support efficient matrix computation:

```rust
use tensorlogic_sklears_kernels::{LinearKernel, Kernel};

let kernel = LinearKernel::new();
let inputs = vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
    vec![5.0, 6.0],
];

let matrix = kernel.compute_matrix(&inputs).unwrap();
// matrix[i][j] = kernel(inputs[i], inputs[j])
// Symmetric positive semi-definite matrix
```

## ARD Kernels

Automatic Relevance Determination kernels learn per-dimension length scales:

```rust
use tensorlogic_sklears_kernels::{ArdRbfKernel, Kernel};

// Per-dimension length scales: [1.0, 2.0, 0.5]
let kernel = ArdRbfKernel::new(vec![1.0, 2.0, 0.5], 1.0);
let x = vec![1.0, 2.0, 3.0];
let y = vec![1.5, 2.5, 3.5];
let sim = kernel.compute(&x, &y).unwrap();
```

## Kernel PCA

Nonlinear dimensionality reduction:

```rust
use tensorlogic_sklears_kernels::{KernelPCA, KernelPCAConfig, RbfKernel, RbfKernelConfig};

let kernel = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());
let config = KernelPCAConfig { n_components: 2, center: true };
let mut kpca = KernelPCA::new(config, kernel);

// Fit and transform training data
kpca.fit(&training_data).unwrap();
let projected = kpca.transform_training().unwrap();

// Access eigenvalues and explained variance
let ratios = kpca.explained_variance_ratio();
println!("Variance explained by 2 components: {:.1}%", ratios.iter().sum::<f64>() * 100.0);
```

## Random Fourier Features

Scalable kernel approximation for large datasets:

```rust
use tensorlogic_sklears_kernels::{RandomFourierFeatures, RffConfig, RffKernelType};

let config = RffConfig { n_components: 100, kernel: RffKernelType::Rbf, gamma: 0.5, seed: None };
let rff = RandomFourierFeatures::new(config, 3).unwrap();

// Transform inputs to approximate feature space
let features = rff.transform(&x).unwrap();

// Approximate kernel via inner product
let approx = rff.approximate_kernel(&x, &y).unwrap();
```

## Kernel Selection

Tools for hyperparameter tuning and model selection:

```rust
use tensorlogic_sklears_kernels::{KernelSelector, KFoldConfig};

// K-fold cross-validation error
let config = KFoldConfig { n_folds: 5, shuffle: true, seed: None };
let cv_result = KernelSelector::k_fold_cv(&kernel_matrix, &labels, &config).unwrap();
println!("Mean error: {:.4}", cv_result.mean_score);

// Grid search for RBF gamma
let result = KernelSelector::grid_search_rbf_gamma(&data, &labels, &[0.01, 0.1, 1.0]).unwrap();
println!("Best gamma: {}", result.best_gamma);
```

## Integration with TensorLogic

Kernels integrate with compiled TensorLogic expressions:

```rust,ignore
use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{RuleSimilarityKernel, FeatureExtractor};

// Extract features from TLExpr automatically
let mut extractor = FeatureExtractor::new();
extractor.fit(&expressions).unwrap();
let features = extractor.transform(&new_expr).unwrap();

// Build kernel from extracted features
let kernel = RuleSimilarityKernel::new(rules, config).unwrap();
let kernel_matrix = kernel.compute_matrix(&features).unwrap();
```

## Design Philosophy

1. **Backend Independence**: Kernels work with any feature representation
2. **Composability**: Mix logical and tensor-based similarities
3. **Type Safety**: Compile-time validation where possible
4. **Performance**: Efficient matrix operations with caching and approximation
5. **Interpretability**: Clear mapping from logic to similarity
6. **Extensibility**: Symbolic composition and builder patterns

## Testing

```bash
cargo nextest run -p tensorlogic-sklears-kernels
# 564 tests, all passing, zero warnings
```

## Benchmarking

```bash
cargo bench -p tensorlogic-sklears-kernels
# 5 suites, 47 groups
```

Benchmark groups:
- Kernel computation (10 groups)
- Matrix operations (10 groups)
- Caching performance (8 groups)
- Composite kernels (10 groups)
- Graph kernels (9 groups)

## License

Apache-2.0

---

**Status**: Production Ready (v0.1.2 Stable)
**Last Updated**: 2026-08-30
**Tests**: 564/564 passing (100%)
**Benchmarks**: 5 suites, 47 benchmark groups
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
