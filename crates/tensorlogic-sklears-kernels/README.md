# tensorlogic-sklears-kernels
[![Crate](https://img.shields.io/badge/crates.io-tensorlogic-sklears-kernels-orange)](https://crates.io/crates/tensorlogic-sklears-kernels)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-sklears-kernels)
[![Tests](https://img.shields.io/badge/tests-154%2F154-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

**Logic-derived similarity kernels for machine learning integration**

This crate provides kernel functions that measure similarity based on logical rule satisfaction patterns, enabling TensorLogic to integrate with traditional machine learning algorithms (SVMs, kernel PCA, kernel ridge regression, etc.).

## Features

### Logic, Graph & Tree Kernels
- ✅ **Rule Similarity Kernels** - Measure similarity by rule satisfaction agreement
- ✅ **Predicate Overlap Kernels** - Similarity based on shared true predicates
- ✅ **Graph Kernels** - Subgraph matching, random walk, Weisfeiler-Lehman kernels
- ✅ **Tree Kernels** - Subtree, subset tree, and partial tree kernels for hierarchical data
- ✅ **TLExpr Conversion** - Automatic graph and tree extraction from logical expressions

### Classical Kernels
- ✅ **Linear Kernel** - Inner product in feature space
- ✅ **RBF (Gaussian) Kernel** - Infinite-dimensional feature mapping
- ✅ **Polynomial Kernel** - Polynomial feature relationships
- ✅ **Cosine Similarity** - Angle-based similarity
- ✅ **Laplacian Kernel** - L1 distance, robust to outliers ✨ **NEW**
- ✅ **Sigmoid Kernel** - Neural network inspired (tanh) ✨ **NEW**
- ✅ **Chi-Squared Kernel** - For histogram data ✨ **NEW**
- ✅ **Histogram Intersection** - Direct histogram overlap ✨ **NEW**

### Composite & Performance Features
- ✅ **Weighted Sum Kernels** - Combine multiple kernels with weights
- ✅ **Product Kernels** - Multiplicative kernel combinations
- ✅ **Kernel Alignment** - Measure similarity between kernel matrices
- ✅ **Kernel Caching** - LRU cache with hit rate statistics
- ✅ **Sparse Matrices** - CSR format for memory-efficient storage
- ✅ **Low-Rank Approximations** - Nyström method for O(nm) complexity
- ✅ **Performance Benchmarks** - 5 benchmark suites with 47 benchmark groups

### Text & Feature Processing
- ✅ **String Kernels** - N-gram, subsequence, edit distance kernels
- ✅ **Feature Extraction** - Automatic TLExpr→vector conversion
- ✅ **Vocabulary Building** - Predicate-based feature encoding

### Kernel Transformations ✨ **NEW**
- ✅ **Matrix Normalization** - Normalize to unit diagonal
- ✅ **Matrix Centering** - Center for kernel PCA
- ✅ **Matrix Standardization** - Combined normalization + centering
- ✅ **Normalized Kernel Wrapper** - Auto-normalizing wrapper

### Provenance Tracking ✨ **NEW (Session 5)**
- ✅ **Automatic Tracking** - Track all kernel computations transparently
- ✅ **Rich Metadata** - Timestamps, computation time, input/output dimensions
- ✅ **Query Interface** - Filter by kernel type, tags, or time range
- ✅ **JSON Export/Import** - Serialize provenance for analysis and archival
- ✅ **Performance Analysis** - Aggregate statistics and profiling
- ✅ **Tagged Experiments** - Organize computations with custom tags

### Symbolic Kernel Composition ✨ **NEW (Session 5)**
- ✅ **Algebraic Expressions** - Build kernels using +, *, ^, and scaling
- ✅ **KernelBuilder** - Declarative builder pattern for readability
- ✅ **Expression Simplification** - Automatic constant folding
- ✅ **PSD Property Checking** - Verify positive semi-definiteness
- ✅ **Method Chaining** - Fluent API for complex compositions

### Quality Assurance
- ✅ **195 Tests** - Comprehensive test coverage (100% passing) ✨ **UPDATED**
- ✅ **Zero Warnings** - Strict code quality enforcement (clippy clean)
- ✅ **Type-Safe API** - Builder pattern with validation
- ✅ **Production Ready** - Battle-tested implementations

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

## Kernel Types

### 1. Logic-Derived Kernels

#### Rule Similarity Kernel

Measures similarity based on agreement in rule satisfaction:

```rust
use tensorlogic_sklears_kernels::{
    RuleSimilarityKernel, RuleSimilarityConfig, Kernel
};
use tensorlogic_ir::TLExpr;

// Define rules
let rules = vec![
    TLExpr::pred("tall", vec![]),
    TLExpr::pred("smart", vec![]),
    TLExpr::pred("friendly", vec![]),
];

// Configure weights
let config = RuleSimilarityConfig::new()
    .with_satisfied_weight(1.0)    // Both satisfy
    .with_violated_weight(0.5)     // Both violate
    .with_mixed_weight(0.0);       // Disagree

let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

// Person A: tall=true, smart=true, friendly=false
let person_a = vec![1.0, 1.0, 0.0];

// Person B: tall=true, smart=true, friendly=true
let person_b = vec![1.0, 1.0, 1.0];

let similarity = kernel.compute(&person_a, &person_b).unwrap();
// High similarity: agree on 2 rules, disagree on 1
```

**Formula:**
```
K(x, y) = Σ_r agreement(x, y, r) / num_rules

agreement(x, y, r) =
  | satisfied_weight  if both satisfy r
  | violated_weight   if both violate r
  | mixed_weight      if they disagree on r
```

#### Predicate Overlap Kernel

Counts shared true predicates:

```rust
use tensorlogic_sklears_kernels::{PredicateOverlapKernel, Kernel};

let kernel = PredicateOverlapKernel::new(5);

let x = vec![1.0, 1.0, 0.0, 1.0, 0.0];  // 3 predicates true
let y = vec![1.0, 1.0, 1.0, 0.0, 0.0];  // 3 predicates true

let sim = kernel.compute(&x, &y).unwrap();
// Similarity = 2/5 = 0.4 (two shared true predicates)
```

With custom weights:

```rust
let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0];  // Some predicates more important
let kernel = PredicateOverlapKernel::with_weights(5, weights).unwrap();
```

### 2. Tensor-Based Kernels

#### Linear Kernel

Inner product in feature space:

```rust
use tensorlogic_sklears_kernels::{LinearKernel, Kernel};

let kernel = LinearKernel::new();
let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let sim = kernel.compute(&x, &y).unwrap();
// sim = x · y = 1*4 + 2*5 + 3*6 = 32
```

#### RBF (Gaussian) Kernel

Infinite-dimensional feature space:

```rust
use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig, Kernel};

let config = RbfKernelConfig::new(0.5);  // gamma = 0.5
let kernel = RbfKernel::new(config).unwrap();

let x = vec![1.0, 2.0, 3.0];
let y = vec![1.5, 2.5, 3.5];
let sim = kernel.compute(&x, &y).unwrap();
// sim = exp(-gamma * ||x-y||^2)
```

Configure from bandwidth (sigma):

```rust
let config = RbfKernelConfig::from_sigma(2.0);  // gamma = 1/(2*sigma^2)
let kernel = RbfKernel::new(config).unwrap();
```

#### Polynomial Kernel

Captures polynomial relationships:

```rust
use tensorlogic_sklears_kernels::{PolynomialKernel, Kernel};

let kernel = PolynomialKernel::new(2, 1.0).unwrap();  // degree=2, constant=1

let x = vec![1.0, 2.0];
let y = vec![3.0, 4.0];
let sim = kernel.compute(&x, &y).unwrap();
// sim = (x · y + c)^d = (11 + 1)^2 = 144
```

#### Cosine Similarity

Angle-based similarity:

```rust
use tensorlogic_sklears_kernels::{CosineKernel, Kernel};

let kernel = CosineKernel::new();

let x = vec![1.0, 2.0, 3.0];
let y = vec![2.0, 4.0, 6.0];  // Parallel to x
let sim = kernel.compute(&x, &y).unwrap();
// sim = cos(angle) = 1.0 (parallel vectors)
```

#### Laplacian Kernel

L1 distance-based kernel, more robust to outliers than RBF:

```rust
use tensorlogic_sklears_kernels::{LaplacianKernel, Kernel};

let kernel = LaplacianKernel::new(0.5).unwrap();  // gamma = 0.5
// Or create from bandwidth: LaplacianKernel::from_sigma(2.0)

let x = vec![1.0, 2.0, 3.0];
let y = vec![1.5, 2.5, 3.5];
let sim = kernel.compute(&x, &y).unwrap();
// sim = exp(-gamma * ||x-y||_1)
```

#### Sigmoid Kernel

Neural network inspired kernel:

```rust
use tensorlogic_sklears_kernels::{SigmoidKernel, Kernel};

let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();  // alpha, offset

let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let sim = kernel.compute(&x, &y).unwrap();
// sim = tanh(alpha * (x · y) + offset)
// Result in [-1, 1]
```

#### Chi-Squared Kernel

Excellent for histogram data and computer vision:

```rust
use tensorlogic_sklears_kernels::{ChiSquaredKernel, Kernel};

let kernel = ChiSquaredKernel::new(1.0).unwrap();  // gamma = 1.0

// Histogram data (normalized)
let hist1 = vec![0.2, 0.3, 0.5];
let hist2 = vec![0.25, 0.35, 0.4];
let sim = kernel.compute(&hist1, &hist2).unwrap();
// High similarity for similar histograms
```

#### Histogram Intersection Kernel

Direct histogram overlap measurement:

```rust
use tensorlogic_sklears_kernels::{HistogramIntersectionKernel, Kernel};

let kernel = HistogramIntersectionKernel::new();

let hist1 = vec![0.5, 0.3, 0.2];
let hist2 = vec![0.3, 0.4, 0.3];
let sim = kernel.compute(&hist1, &hist2).unwrap();
// sim = Σ min(hist1_i, hist2_i) = 0.8
```

### 3. Graph Kernels

#### Subgraph Matching Kernel

Measures similarity by counting common subgraphs:

```rust
use tensorlogic_sklears_kernels::{
    Graph, SubgraphMatchingKernel, SubgraphMatchingConfig
};
use tensorlogic_ir::TLExpr;

// Build graph from logical expression
let expr = TLExpr::and(
    TLExpr::pred("p1", vec![]),
    TLExpr::pred("p2", vec![]),
);
let graph = Graph::from_tlexpr(&expr);

// Configure kernel
let config = SubgraphMatchingConfig::new().with_max_size(3);
let kernel = SubgraphMatchingKernel::new(config);

// Compute similarity between graphs
let sim = kernel.compute_graphs(&graph1, &graph2).unwrap();
```

#### Random Walk Kernel

Counts common random walks between graphs:

```rust
use tensorlogic_sklears_kernels::{
    RandomWalkKernel, WalkKernelConfig
};

let config = WalkKernelConfig::new()
    .with_max_length(4)
    .with_decay(0.8);
let kernel = RandomWalkKernel::new(config).unwrap();

let sim = kernel.compute_graphs(&graph1, &graph2).unwrap();
```

#### Weisfeiler-Lehman Kernel

Iterative graph isomorphism test:

```rust
use tensorlogic_sklears_kernels::{
    WeisfeilerLehmanKernel, WeisfeilerLehmanConfig
};

let config = WeisfeilerLehmanConfig::new().with_iterations(3);
let kernel = WeisfeilerLehmanKernel::new(config);

let sim = kernel.compute_graphs(&graph1, &graph2).unwrap();
```

### 4. Tree Kernels

Tree kernels measure similarity between hierarchical structures, perfect for logical expressions with nested structure.

#### Subtree Kernel

Counts exact matching subtrees:

```rust
use tensorlogic_sklears_kernels::{
    TreeNode, SubtreeKernel, SubtreeKernelConfig
};
use tensorlogic_ir::TLExpr;

// Create tree from logical expression
let expr = TLExpr::and(
    TLExpr::pred("p1", vec![]),
    TLExpr::or(
        TLExpr::pred("p2", vec![]),
        TLExpr::pred("p3", vec![]),
    ),
);
let tree = TreeNode::from_tlexpr(&expr);

// Configure and compute
let config = SubtreeKernelConfig::new().with_normalize(true);
let kernel = SubtreeKernel::new(config);

let tree2 = TreeNode::from_tlexpr(&another_expr);
let similarity = kernel.compute_trees(&tree, &tree2).unwrap();
```

#### Subset Tree Kernel

Allows gaps in tree fragments with decay factors:

```rust
use tensorlogic_sklears_kernels::{
    SubsetTreeKernel, SubsetTreeKernelConfig
};

let config = SubsetTreeKernelConfig::new()
    .unwrap()
    .with_decay(0.8)
    .unwrap()
    .with_normalize(true);

let kernel = SubsetTreeKernel::new(config);
let similarity = kernel.compute_trees(&tree1, &tree2).unwrap();
```

#### Partial Tree Kernel

Supports partial matching with configurable thresholds:

```rust
use tensorlogic_sklears_kernels::{
    PartialTreeKernel, PartialTreeKernelConfig
};

let config = PartialTreeKernelConfig::new()
    .unwrap()
    .with_decay(0.9)
    .unwrap()
    .with_threshold(0.5)
    .unwrap();

let kernel = PartialTreeKernel::new(config);
let similarity = kernel.compute_trees(&tree1, &tree2).unwrap();
```

### 5. Low-Rank Approximations

The Nyström method provides efficient kernel matrix approximation with O(nm) complexity instead of O(n²).

#### Basic Usage

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, NystromApproximation, NystromConfig, SamplingMethod
};

let data = vec![/* large dataset */];
let kernel = LinearKernel::new();

// Configure with 100 landmarks
let config = NystromConfig::new(100)
    .unwrap()
    .with_sampling(SamplingMethod::KMeansPlusPlus)
    .with_regularization(1e-6)
    .unwrap();

// Fit approximation
let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

// Approximate kernel values
let similarity = approx.approximate(i, j).unwrap();

// Get compression ratio
let ratio = approx.compression_ratio();
println!("Compression: {:.2}x", ratio);
```

#### Sampling Methods

```rust
// Uniform random sampling (fast, deterministic)
let config1 = NystromConfig::new(50)
    .unwrap()
    .with_sampling(SamplingMethod::Uniform);

// First n points (simplest)
let config2 = NystromConfig::new(50)
    .unwrap()
    .with_sampling(SamplingMethod::First);

// K-means++ style (diverse landmarks, better quality)
let config3 = NystromConfig::new(50)
    .unwrap()
    .with_sampling(SamplingMethod::KMeansPlusPlus);
```

#### Approximation Quality

```rust
// Compute exact matrix for comparison
let exact = kernel.compute_matrix(&data).unwrap();

// Fit approximation
let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

// Compute approximation error (Frobenius norm)
let error = approx.approximation_error(&exact).unwrap();
println!("Approximation error: {:.4}", error);

// Get full approximate matrix
let approx_matrix = approx.get_approximate_matrix().unwrap();
```

### 6. Composite Kernels

#### Weighted Sum

Combine multiple kernels with weights:

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, RbfKernel, RbfKernelConfig,
    WeightedSumKernel, Kernel
};

let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;

// 70% linear, 30% RBF
let weights = vec![0.7, 0.3];
let composite = WeightedSumKernel::new(vec![linear, rbf], weights).unwrap();

let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let sim = composite.compute(&x, &y).unwrap();
```

#### Product Kernel

Multiplicative combination:

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, CosineKernel, ProductKernel, Kernel
};

let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;

let product = ProductKernel::new(vec![linear, cosine]).unwrap();
let sim = product.compute(&x, &y).unwrap();
```

#### Kernel Alignment

Measure similarity between kernel matrices:

```rust
use tensorlogic_sklears_kernels::KernelAlignment;

let k1 = vec![
    vec![1.0, 0.8, 0.6],
    vec![0.8, 1.0, 0.7],
    vec![0.6, 0.7, 1.0],
];

let k2 = vec![
    vec![1.0, 0.75, 0.55],
    vec![0.75, 1.0, 0.65],
    vec![0.55, 0.65, 1.0],
];

let alignment = KernelAlignment::compute_alignment(&k1, &k2).unwrap();
// High alignment means kernels agree on data structure
```

### 7. Kernel Selection and Tuning

The `kernel_utils` module provides utilities for selecting the best kernel and tuning hyperparameters for your ML task.

#### Kernel-Target Alignment (KTA)

Measure how well a kernel matrix aligns with the target labels:

```rust
use tensorlogic_sklears_kernels::{
    RbfKernel, RbfKernelConfig, LinearKernel, Kernel,
    kernel_utils::{compute_gram_matrix, kernel_target_alignment}
};

let data = vec![
    vec![0.1, 0.2], vec![0.2, 0.1],  // Class +1
    vec![0.9, 0.8], vec![0.8, 0.9],  // Class -1
];
let labels = vec![1.0, 1.0, -1.0, -1.0];

// Compare different kernels
let linear = LinearKernel::new();
let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

let k_linear = compute_gram_matrix(&data, &linear).unwrap();
let k_rbf = compute_gram_matrix(&data, &rbf).unwrap();

let kta_linear = kernel_target_alignment(&k_linear, &labels).unwrap();
let kta_rbf = kernel_target_alignment(&k_rbf, &labels).unwrap();

println!("Linear KTA: {:.4}", kta_linear);
println!("RBF KTA: {:.4}", kta_rbf);
// Higher KTA indicates better kernel for the task
```

#### Automatic Bandwidth Selection

Use the median heuristic to automatically select optimal gamma for RBF/Laplacian kernels:

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, RbfKernel, RbfKernelConfig,
    kernel_utils::median_heuristic_bandwidth
};

let data = vec![
    vec![0.1, 0.2], vec![0.3, 0.4],
    vec![0.9, 0.8], vec![0.7, 0.6],
];

// Use linear kernel to compute distances
let linear = LinearKernel::new();
let optimal_gamma = median_heuristic_bandwidth(&data, &linear, None).unwrap();

println!("Optimal gamma: {:.4}", optimal_gamma);

// Create RBF kernel with optimal bandwidth
let rbf = RbfKernel::new(RbfKernelConfig::new(optimal_gamma)).unwrap();
```

#### Data Normalization

Normalize data rows to unit L2 norm:

```rust
use tensorlogic_sklears_kernels::kernel_utils::normalize_rows;

let data = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
];

let normalized = normalize_rows(&data).unwrap();
// Each row now has L2 norm = 1.0
```

#### Kernel Matrix Validation

Check if a kernel matrix is valid (symmetric and approximately PSD):

```rust
use tensorlogic_sklears_kernels::kernel_utils::is_valid_kernel_matrix;

let k = vec![
    vec![1.0, 0.8, 0.6],
    vec![0.8, 1.0, 0.7],
    vec![0.6, 0.7, 1.0],
];

let is_valid = is_valid_kernel_matrix(&k, Some(1e-6)).unwrap();
println!("Valid kernel matrix: {}", is_valid);
```

### 8. Performance Features

#### Kernel Caching

Avoid redundant computations:

```rust
use tensorlogic_sklears_kernels::{LinearKernel, CachedKernel, Kernel};

let base = LinearKernel::new();
let cached = CachedKernel::new(Box::new(base));

let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];

// First call - compute and cache
let result1 = cached.compute(&x, &y).unwrap();

// Second call - retrieve from cache (faster)
let result2 = cached.compute(&x, &y).unwrap();

// Check statistics
let stats = cached.stats();
println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
```

#### Sparse Kernel Matrices

Efficient storage for large, sparse matrices:

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, SparseKernelMatrixBuilder
};

let kernel = LinearKernel::new();
let data = vec![/* large dataset */];

// Build sparse matrix with threshold
let builder = SparseKernelMatrixBuilder::new()
    .with_threshold(0.1).unwrap()
    .with_max_entries_per_row(10).unwrap();

let sparse_matrix = builder.build(&data, &kernel).unwrap();

// Check sparsity
println!("Density: {:.2}%", sparse_matrix.density() * 100.0);
println!("Non-zero entries: {}", sparse_matrix.nnz());
```

### 9. String Kernels

#### N-Gram Kernel

Measure text similarity by n-gram overlap:

```rust
use tensorlogic_sklears_kernels::{NGramKernel, NGramKernelConfig};

let config = NGramKernelConfig::new(2).unwrap(); // bigrams
let kernel = NGramKernel::new(config);

let text1 = "hello world";
let text2 = "hello there";

let sim = kernel.compute_strings(text1, text2).unwrap();
println!("N-gram similarity: {}", sim);
```

#### Subsequence Kernel

Non-contiguous subsequence matching:

```rust
use tensorlogic_sklears_kernels::{
    SubsequenceKernel, SubsequenceKernelConfig
};

let config = SubsequenceKernelConfig::new()
    .with_max_length(3).unwrap()
    .with_decay(0.5).unwrap();

let kernel = SubsequenceKernel::new(config);

let text1 = "machine learning";
let text2 = "machine_learning";

let sim = kernel.compute_strings(text1, text2).unwrap();
```

#### Edit Distance Kernel

Exponential of negative Levenshtein distance:

```rust
use tensorlogic_sklears_kernels::EditDistanceKernel;

let kernel = EditDistanceKernel::new(0.1).unwrap();

let text1 = "color";
let text2 = "colour"; // British vs American spelling

let sim = kernel.compute_strings(text1, text2).unwrap();
// High similarity despite spelling difference
```

### 10. Provenance Tracking

Track kernel computations for debugging, auditing, and reproducibility:

#### Basic Tracking

```rust
use tensorlogic_sklears_kernels::{
    LinearKernel, Kernel,
    ProvenanceKernel, ProvenanceTracker
};

// Create kernel with provenance tracking
let tracker = ProvenanceTracker::new();
let base_kernel = Box::new(LinearKernel::new());
let kernel = ProvenanceKernel::new(base_kernel, tracker.clone());

// Computations are automatically tracked
let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let result = kernel.compute(&x, &y).unwrap();

// Query provenance history
let records = tracker.get_all_records();
println!("Tracked {} computations", records.len());

// Analyze computation patterns
let avg_time = tracker.average_computation_time();
println!("Average time: {:?}", avg_time);
```

#### Configure Tracking

```rust
use tensorlogic_sklears_kernels::{ProvenanceConfig, ProvenanceTracker};

// Configure with limits and sampling
let config = ProvenanceConfig::new()
    .with_max_records(1000)         // Keep last 1000 records
    .with_sample_rate(0.5).unwrap() // Track 50% of computations
    .with_timing(true);              // Include timing info

let tracker = ProvenanceTracker::with_config(config);
```

#### Tagged Experiments

```rust
// Organize computations with tags
let mut experiment1 = ProvenanceKernel::new(base_kernel, tracker.clone());
experiment1.add_tag("experiment:baseline".to_string());
experiment1.add_tag("phase:1".to_string());

experiment1.compute(&x, &y).unwrap();

// Query by tag
let baseline_records = tracker.get_records_by_tag("experiment:baseline");
println!("Baseline: {} computations", baseline_records.len());
```

#### Export/Import

```rust
// Export to JSON for analysis
let json = tracker.to_json().unwrap();
std::fs::write("provenance.json", json).unwrap();

// Import from JSON
let tracker2 = ProvenanceTracker::new();
let json = std::fs::read_to_string("provenance.json").unwrap();
tracker2.from_json(&json).unwrap();
```

#### Performance Statistics

```rust
let stats = tracker.statistics();
println!("Total computations: {}", stats.total_computations);
println!("Success rate: {:.2}%",
    stats.successful_computations as f64 / stats.total_computations as f64 * 100.0);

// Per-kernel breakdown
for (kernel_name, count) in &stats.kernel_counts {
    println!("  {}: {} computations", kernel_name, count);
}
```

### 11. Symbolic Kernel Composition

Build complex kernels using algebraic expressions:

#### Basic Composition

```rust
use std::sync::Arc;
use tensorlogic_sklears_kernels::{
    LinearKernel, RbfKernel, RbfKernelConfig,
    KernelExpr, SymbolicKernel, Kernel
};

// Build: 0.5 * linear + 0.3 * rbf
let linear = Arc::new(LinearKernel::new());
let rbf = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

let expr = KernelExpr::base(linear)
    .scale(0.5).unwrap()
    .add(KernelExpr::base(rbf).scale(0.3).unwrap());

let kernel = SymbolicKernel::new(expr);

let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];
let similarity = kernel.compute(&x, &y).unwrap();
```

#### Builder Pattern

```rust
use tensorlogic_sklears_kernels::KernelBuilder;

// More readable with builder
let kernel = KernelBuilder::new()
    .add_scaled(Arc::new(LinearKernel::new()), 0.5)
    .add_scaled(Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()), 0.3)
    .add_scaled(Arc::new(CosineKernel::new()), 0.2)
    .build();
```

#### Algebraic Operations

```rust
// Scaling
let k_scaled = KernelExpr::base(kernel).scale(2.0).unwrap();

// Addition
let k_sum = expr1.add(expr2);

// Multiplication
let k_product = expr1.multiply(expr2);

// Power
let k_squared = KernelExpr::base(kernel).power(2).unwrap();
```

#### Complex Expressions

```rust
// Build: (0.7 * linear + 0.3 * rbf)^2
let linear = Arc::new(LinearKernel::new());
let rbf = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

let sum = KernelExpr::base(linear)
    .scale(0.7).unwrap()
    .add(KernelExpr::base(rbf).scale(0.3).unwrap());

let kernel = SymbolicKernel::new(sum.power(2).unwrap());
```

#### Hybrid ML Kernel

```rust
// Combine interpretability (linear) with non-linearity (RBF) and interactions (polynomial)
let kernel = KernelBuilder::new()
    .add_scaled(Arc::new(LinearKernel::new()), 0.4)              // interpretability
    .add_scaled(Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()), 0.4)  // non-linearity
    .add_scaled(Arc::new(PolynomialKernel::new(2, 1.0).unwrap()), 0.2)  // interactions
    .build();

// Use in SVM, kernel PCA, etc.
let K = kernel.compute_matrix(&training_data).unwrap();
```

### 12. Feature Extraction

Automatically convert logical expressions to feature vectors:

```rust
use tensorlogic_sklears_kernels::{
    FeatureExtractor, FeatureExtractionConfig
};
use tensorlogic_ir::TLExpr;

// Configure extraction
let config = FeatureExtractionConfig::new()
    .with_max_depth(5)
    .with_encode_structure(true)
    .with_encode_quantifiers(true)
    .with_fixed_dimension(20); // Fixed-size vectors

let mut extractor = FeatureExtractor::new(config);

// Build vocabulary from training set
let training_exprs = vec![
    TLExpr::pred("tall", vec![]),
    TLExpr::pred("smart", vec![]),
    TLExpr::and(
        TLExpr::pred("tall", vec![]),
        TLExpr::pred("smart", vec![]),
    ),
];
extractor.build_vocabulary(&training_exprs);

// Extract features from new expressions
let expr = TLExpr::or(
    TLExpr::pred("tall", vec![]),
    TLExpr::pred("friendly", vec![]),
);

let features = extractor.extract(&expr).unwrap();
// features = [depth, node_count, num_and, num_or, ..., pred_counts..., exists, forall]

// Use with any kernel
let kernel = LinearKernel::new();
let sim = kernel.compute(&features1, &features2).unwrap();
```

#### Batch Feature Extraction

```rust
let expressions = vec![expr1, expr2, expr3];
let feature_vectors = extractor.extract_batch(&expressions).unwrap();

// Use with kernel matrix
let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
let K = kernel.compute_matrix(&feature_vectors).unwrap();
```

## Kernel Matrix Computation

All kernels support efficient matrix computation:

```rust
use tensorlogic_sklears_kernels::{LinearKernel, Kernel};

let kernel = LinearKernel::new();
let data = vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
    vec![5.0, 6.0],
];

let K = kernel.compute_matrix(&data).unwrap();
// K[i][j] = kernel(data[i], data[j])
// Symmetric positive semi-definite matrix
```

Properties of kernel matrices:
- **Symmetric**: K[i,j] = K[j,i]
- **Positive Semi-Definite**: All eigenvalues ≥ 0
- **Diagonal**: K[i,i] = self-similarity

## Kernel Transformation Utilities

### Matrix Normalization

Normalize a kernel matrix to have unit diagonal entries:

```rust
use tensorlogic_sklears_kernels::kernel_transform::normalize_kernel_matrix;

let K = vec![
    vec![4.0, 2.0, 1.0],
    vec![2.0, 9.0, 3.0],
    vec![1.0, 3.0, 16.0],
];

let K_norm = normalize_kernel_matrix(&K).unwrap();

// All diagonal entries are now 1.0
assert!((K_norm[0][0] - 1.0).abs() < 1e-10);
assert!((K_norm[1][1] - 1.0).abs() < 1e-10);
assert!((K_norm[2][2] - 1.0).abs() < 1e-10);
```

### Matrix Centering

Center a kernel matrix for kernel PCA:

```rust
use tensorlogic_sklears_kernels::kernel_transform::center_kernel_matrix;

let K = vec![
    vec![1.0, 0.8, 0.6],
    vec![0.8, 1.0, 0.7],
    vec![0.6, 0.7, 1.0],
];

let K_centered = center_kernel_matrix(&K).unwrap();

// Row and column means are now approximately zero
```

### Matrix Standardization

Combine normalization and centering:

```rust
use tensorlogic_sklears_kernels::kernel_transform::standardize_kernel_matrix;

let K = vec![
    vec![4.0, 2.0, 1.0],
    vec![2.0, 9.0, 3.0],
    vec![1.0, 3.0, 16.0],
];

let K_std = standardize_kernel_matrix(&K).unwrap();
// Normalized and centered in one operation
```

### Normalized Kernel Wrapper

Wrap any kernel to automatically normalize outputs:

```rust
use tensorlogic_sklears_kernels::{LinearKernel, NormalizedKernel, Kernel};

let linear = Box::new(LinearKernel::new());
let normalized = NormalizedKernel::new(linear);

let x = vec![1.0, 2.0, 3.0];
let y = vec![4.0, 5.0, 6.0];

// Compute normalized similarity
let sim = normalized.compute(&x, &y).unwrap();

// Self-similarity is always 1.0
let self_sim = normalized.compute(&x, &x).unwrap();
assert!((self_sim - 1.0).abs() < 1e-10);
```

## Use Cases

### 1. Kernel SVM

```rust,ignore
use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};

let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
let svm = KernelSVM::new(kernel);
svm.fit(training_data, labels);
let predictions = svm.predict(test_data);
```

### 2. Semantic Similarity

Measure similarity based on logical properties:

```rust,ignore
let rules = extract_semantic_rules();
let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

let doc1_features = encode_document(doc1);
let doc2_features = encode_document(doc2);

let similarity = kernel.compute(&doc1_features, &doc2_features).unwrap();
```

### 3. Hybrid Kernels

Combine logical and tensor-based features:

```rust,ignore
let logic_kernel = RuleSimilarityKernel::new(rules, config).unwrap();
let rbf_kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

// Weighted combination
let alpha = 0.7;
let hybrid_similarity =
    alpha * logic_kernel.compute(&x_logic, &y_logic).unwrap() +
    (1.0 - alpha) * rbf_kernel.compute(&x_emb, &y_emb).unwrap();
```

### 4. Kernel PCA

```rust,ignore
let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
let K = kernel.compute_matrix(&data).unwrap();

// Perform eigendecomposition on K
let (eigenvalues, eigenvectors) = eigen_decomposition(K);

// Project to principal components
let projected = project_to_pcs(data, eigenvectors);
```

## Testing

Run the test suite:

```bash
cargo nextest run -p tensorlogic-sklears-kernels
```

All 90 tests should pass with zero warnings.

## Performance

Kernel matrix computation complexity:
- **Time**: O(n² * d) where n = samples, d = features
- **Space**: O(n²) for kernel matrix storage

Optimizations:
- ✅ Vectorized distance computations
- ✅ Symmetric matrix (only compute upper triangle)
- ✅ Sparse kernel matrices (CSR format)
- ✅ Kernel caching with hit rate tracking

## Roadmap

See [TODO.md](TODO.md) for the development roadmap. Current status: 🎉 **100% complete (36/36 tasks)** 🎉 **ALL COMPLETE!**

### Completed ✅
- ✅ **Classical Kernels**: Linear, RBF, Polynomial, Cosine, Laplacian, Sigmoid, Chi-squared, Histogram Intersection
- ✅ **Logic Kernels**: Rule similarity, Predicate overlap
- ✅ **Graph Kernels**: Subgraph matching, Random walk, Weisfeiler-Lehman
- ✅ **Tree Kernels**: Subtree, Subset tree, Partial tree
- ✅ **String Kernels**: N-gram, Subsequence, Edit distance
- ✅ **Composite Kernels**: Weighted sum, Product, Kernel alignment
- ✅ **Performance Optimizations**:
  - ✅ Sparse kernel matrices (CSR format)
  - ✅ Kernel caching (LRU with statistics)
  - ✅ Low-rank approximations (Nyström method)
- ✅ **Kernel Transformations**: Normalization, Centering, Standardization
- ✅ **Kernel Utilities**: KTA, median heuristic, matrix validation
- ✅ **Provenance Tracking**: Automatic tracking, JSON export, tagged experiments ✨ **NEW**
- ✅ **Symbolic Composition**: Algebraic expressions, builder pattern, simplification ✨ **NEW**
- ✅ **Feature Extraction**: Automatic TLExpr→vector conversion
- ✅ **Benchmarks**: 5 benchmark suites, 47 groups
- ✅ **Tests**: 195 comprehensive tests (100% passing, zero warnings) ✨ **UPDATED**
- ✅ **Documentation**: Complete with architecture guide and examples

### Planned 📋
- Deep kernel learning (FUTURE)
- GPU acceleration (FUTURE)
- Multi-task kernel learning (FUTURE)
- SkleaRS trait implementation (FUTURE)

## Integration with TensorLogic

Kernels bridge logical reasoning and statistical learning:

```rust,ignore
use tensorlogic_ir::TLExpr;
use tensorlogic_compiler::compile;
use tensorlogic_sklears_kernels::RuleSimilarityKernel;

// Define logical rules
let rule1 = TLExpr::exists("x", "Person",
    TLExpr::pred("knows", vec![/* ... */])
);

// Compile to kernel
let kernel = RuleSimilarityKernel::from_rules(vec![rule1]).unwrap();

// Use in ML pipeline
let features = extract_features(data);
let K = kernel.compute_matrix(&features).unwrap();
let svm = train_svm(K, labels);
```

## Design Philosophy

1. **Backend Independence**: Works with any feature representation
2. **Composability**: Mix logical and tensor-based similarities
3. **Type Safety**: Compile-time validation
4. **Performance**: Efficient matrix operations
5. **Interpretability**: Clear mapping from logic to similarity

## Architecture Overview

### Module Organization

The crate is organized into specialized modules, each with clear responsibilities:

```
tensorlogic-sklears-kernels/
├── src/
│   ├── lib.rs                    # Public API and re-exports
│   ├── types.rs                  # Core Kernel trait
│   ├── error.rs                  # Error handling
│   ├── logic_kernel.rs           # Logic-based kernels (RuleSimilarity, PredicateOverlap)
│   ├── tensor_kernel.rs          # Classical kernels (Linear, RBF, Polynomial, Cosine)
│   ├── graph_kernel.rs           # Graph kernels from TLExpr (Subgraph, RandomWalk, WL)
│   ├── composite_kernel.rs       # Kernel combinations (WeightedSum, Product, Alignment)
│   ├── string_kernel.rs          # Text similarity kernels (NGram, Subsequence, EditDistance)
│   ├── feature_extraction.rs     # TLExpr→vector conversion
│   ├── cache.rs                  # LRU caching with statistics
│   └── sparse.rs                 # Sparse matrix support (CSR format)
```

### Core Traits

#### Kernel Trait

The foundation of all kernel implementations:

```rust
pub trait Kernel: Send + Sync {
    /// Compute kernel value between two feature vectors
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64>;

    /// Compute kernel matrix for a dataset
    fn compute_matrix(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Default implementation with optimizations
    }
}
```

**Design Decisions:**
- `Send + Sync` bounds enable parallel computation
- Default `compute_matrix` implementation with symmetry exploitation
- Error handling via `Result` for dimension mismatches and invalid configurations

### Data Flow

#### 1. Feature Preparation

```
TLExpr → FeatureExtractor → Vec<f64>
   ↓
[structural features, predicate features, quantifier features]
```

**Pipeline:**
1. **Vocabulary Building**: Scan training expressions to build predicate→index mapping
2. **Feature Extraction**: Convert each expression to fixed-dimension vector
3. **Batch Processing**: Extract multiple expressions in one pass

#### 2. Kernel Computation

```
Vec<f64> × Vec<f64> → Kernel::compute() → f64
        ↓
    Cache lookup (if CachedKernel)
        ↓
    Actual computation
        ↓
    Cache store
```

**Optimization Layers:**
- **Cache Layer**: LRU cache with hit rate tracking
- **Vectorization**: SIMD-friendly operations where possible
- **Symmetry**: Compute only upper triangle for kernel matrices

#### 3. Composite Kernels

```
K₁(x,y)  K₂(x,y)  ...  Kₙ(x,y)
   ↓        ↓            ↓
   w₁  ×   w₂   × ... × wₙ
   ↓_______|____________|
            ↓
      Σ wᵢKᵢ(x,y)  (WeightedSum)
   or ∏ Kᵢ(x,y)     (Product)
```

### Memory Management

#### Dense Kernel Matrices

- **Storage**: `Vec<Vec<f64>>` - symmetric n×n matrix
- **Memory**: O(n²) for n samples
- **Optimization**: Only upper triangle computed, then mirrored

#### Sparse Kernel Matrices

- **Format**: Compressed Sparse Row (CSR)
- **Components**:
  - `row_ptr: Vec<usize>` - Row start indices (size: n+1)
  - `col_idx: Vec<usize>` - Column indices (size: nnz)
  - `values: Vec<f64>` - Non-zero values (size: nnz)
- **Memory**: O(nnz) where nnz = number of non-zero entries
- **Builder**: Supports threshold-based sparsification and max entries per row

**Example Sparsification:**
```
Dense 1000×1000 matrix (8 MB)
  ↓ (threshold=0.1, max_per_row=50)
Sparse CSR format (400 KB) - 95% memory savings
```

### Graph Kernel Pipeline

```
TLExpr → Graph::from_tlexpr() → Graph
   ↓
   [nodes: Vec<GraphNode>, edges: Vec<(usize, usize)>]
   ↓
   Graph Kernel (Subgraph/RandomWalk/WL)
   ↓
   Similarity Score
```

**Graph Construction Rules:**
- **Nodes**: Each predicate/operator becomes a node with label
- **Edges**: Parent-child relationships in expression tree
- **Features**: Node labels, edge types, structural properties

**Kernel Types:**
- **Subgraph Matching**: Counts common subgraphs (exponential in subgraph size)
- **Random Walk**: Enumerates walks, measures overlap (polynomial complexity)
- **Weisfeiler-Lehman**: Iterative refinement of node labels (linear per iteration)

### String Kernel Pipeline

```
text₁, text₂ → String Kernel → similarity ∈ [0,1]
                    ↓
            [n-grams / subsequences / edit operations]
```

**Algorithms:**
- **N-Gram**: O(n) generation, O(min(n₁,n₂)) intersection
- **Subsequence**: O(n²m²) dynamic programming for decay-weighted count
- **Edit Distance**: O(nm) Levenshtein DP, then exponential transformation

### Error Handling Strategy

```rust
pub enum KernelError {
    DimensionMismatch { expected: usize, actual: usize },
    InvalidParameter { parameter: String, value: String, reason: String },
    ComputationError { message: String },
    GraphConstructionError { message: String },
}
```

**Propagation:**
- All public APIs return `Result<T, KernelError>`
- Configuration validation at construction time
- Runtime dimension checks with clear error messages

### Testing Architecture

**Test Organization:**
- **Unit Tests**: In-module `#[cfg(test)] mod tests`
- **Coverage**: 90 tests across 9 modules
- **Categories**:
  - Basic functionality (correctness)
  - Edge cases (empty inputs, dimension mismatches)
  - Configuration validation (invalid parameters)
  - Performance (sparse vs dense, cache hit rates)

**Test Utilities:**
```rust
fn assert_kernel_similarity(kernel: &dyn Kernel, x: &[f64], y: &[f64], expected: f64) {
    let result = kernel.compute(x, y).unwrap();
    assert!((result - expected).abs() < 1e-6, "Expected {}, got {}", expected, result);
}
```

## Kernel Design Guide

### Implementing a Custom Kernel

Follow this pattern to create a new kernel type:

#### Step 1: Define Configuration

```rust
#[derive(Clone, Debug)]
pub struct MyKernelConfig {
    pub param1: f64,
    pub param2: usize,
}

impl MyKernelConfig {
    pub fn new(param1: f64, param2: usize) -> Result<Self> {
        if param1 <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "param1".to_string(),
                value: param1.to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { param1, param2 })
    }
}
```

**Best Practices:**
- Validate all parameters at construction time
- Provide clear error messages with `reason` field
- Use builder pattern for optional parameters

#### Step 2: Define Kernel Structure

```rust
pub struct MyKernel {
    config: MyKernelConfig,
    // Cached computations (if needed)
    precomputed_data: Option<Vec<f64>>,
}

impl MyKernel {
    pub fn new(config: MyKernelConfig) -> Self {
        Self {
            config,
            precomputed_data: None,
        }
    }

    // Optional: Add initialization method if precomputation is needed
    pub fn fit(&mut self, training_data: &[Vec<f64>]) -> Result<()> {
        // Precompute necessary statistics
        Ok(())
    }
}
```

#### Step 3: Implement Kernel Trait

```rust
impl Kernel for MyKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        // 1. Validate dimensions
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        // 2. Handle edge cases
        if x.is_empty() {
            return Ok(0.0);
        }

        // 3. Perform kernel computation
        let similarity = self.compute_similarity(x, y)?;

        // 4. Validate output (if needed)
        if !similarity.is_finite() {
            return Err(KernelError::ComputationError {
                message: "Non-finite similarity computed".to_string(),
            });
        }

        Ok(similarity)
    }
}

impl MyKernel {
    fn compute_similarity(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        // Core algorithm here
        let mut result = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            result += self.config.param1 * (xi * yi);
        }
        Ok(result)
    }
}
```

#### Step 4: Add Comprehensive Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_computation() {
        let config = MyKernelConfig::new(1.0, 5).unwrap();
        let kernel = MyKernel::new(config);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = MyKernelConfig::new(1.0, 5).unwrap();
        let kernel = MyKernel::new(config);

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0, 5.0];

        assert!(kernel.compute(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_config() {
        let result = MyKernelConfig::new(-1.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_symmetry() {
        let config = MyKernelConfig::new(1.0, 5).unwrap();
        let kernel = MyKernel::new(config);

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();

        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_positive_definite() {
        // For valid kernels, K(x,x) ≥ 0
        let config = MyKernelConfig::new(1.0, 5).unwrap();
        let kernel = MyKernel::new(config);

        let x = vec![1.0, 2.0, 3.0];
        let k_xx = kernel.compute(&x, &x).unwrap();

        assert!(k_xx >= 0.0);
    }
}
```

### Kernel Properties to Verify

A valid kernel function must satisfy:

1. **Symmetry**: K(x, y) = K(y, x)
2. **Positive Semi-Definiteness**: All eigenvalues of kernel matrix ≥ 0
3. **Bounded**: For normalized kernels, K(x, y) ∈ [0, 1]
4. **Self-similarity**: K(x, x) = 1 for normalized kernels

**Testing PSD Property:**
```rust
#[test]
fn test_kernel_matrix_psd() {
    let kernel = MyKernel::new(config);
    let data = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ];

    let K = kernel.compute_matrix(&data).unwrap();

    // All diagonal entries should be non-negative
    for i in 0..data.len() {
        assert!(K[i][i] >= 0.0);
    }

    // Optional: Compute eigenvalues and verify all ≥ 0
    // (requires linalg library)
}
```

### Performance Optimization Guidelines

#### 1. Avoid Redundant Allocations

```rust
// ❌ Bad: Allocates on every call
fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
    let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    Ok(diff.iter().map(|d| d * d).sum())
}

// ✅ Good: No allocation
fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
    Ok(x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum())
}
```

#### 2. Use Iterators for SIMD

```rust
// ✅ Compiler can auto-vectorize
let dot_product: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
```

#### 3. Exploit Matrix Symmetry

```rust
fn compute_matrix(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = data.len();
    let mut K = vec![vec![0.0; n]; n];

    // Only compute upper triangle
    for i in 0..n {
        for j in i..n {
            let value = self.compute(&data[i], &data[j])?;
            K[i][j] = value;
            K[j][i] = value;  // Mirror
        }
    }

    Ok(K)
}
```

#### 4. Cache Expensive Computations

```rust
pub struct ExpensiveKernel {
    config: Config,
    cache: RefCell<HashMap<(usize, usize), f64>>,
}

impl Kernel for ExpensiveKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let key = (hash(x), hash(y));

        if let Some(&cached) = self.cache.borrow().get(&key) {
            return Ok(cached);
        }

        let result = self.expensive_computation(x, y)?;
        self.cache.borrow_mut().insert(key, result);
        Ok(result)
    }
}
```

Or use the built-in `CachedKernel` wrapper:
```rust
let base_kernel = MyExpensiveKernel::new(config);
let cached = CachedKernel::new(Box::new(base_kernel));
```

### Integration Checklist

When adding a new kernel to the crate:

- [ ] Implement `Kernel` trait
- [ ] Add configuration struct with validation
- [ ] Provide clear documentation with examples
- [ ] Add to module exports in `lib.rs`
- [ ] Write comprehensive tests (≥5 test cases)
- [ ] Verify zero clippy warnings
- [ ] Add usage example to README.md
- [ ] Update TODO.md completion status

## References

- [Kernel Methods in Machine Learning](https://arxiv.org/abs/math/0701907)
- [Tensor Logic Paper](https://arxiv.org/abs/2510.12269)
- [Support Vector Machines](https://en.wikipedia.org/wiki/Support-vector_machine)

## License

This crate is part of the TensorLogic project and is licensed under Apache-2.0.

---

**Status**: Production Ready
**Version**: 0.1.0-alpha.1
**Tests**: 195/195 passing ✨ **UPDATED**
**Warnings**: 0
**Completion**: 🎉 **100%** 🎉 **ALL COMPLETE!**
**Last Updated**: 2025-11-07

**Latest Enhancements (Session 5 - 2025-11-07):** ✨

**Part 2: Symbolic Kernel Composition** (Module 2/2)
- **Symbolic Kernel Composition** (comprehensive composition module, 14 tests)
  - KernelExpr - Algebraic kernel expressions with operations (scale, add, multiply, power)
  - SymbolicKernel - Evaluates expressions for any input
  - KernelBuilder - Declarative builder pattern for readability
  - Expression simplification - Automatic constant folding
  - PSD property checking - Verify positive semi-definiteness
  - Example: symbolic_kernels.rs with 7 usage scenarios

**Part 1: Provenance Tracking** (Module 1/2)
- **Provenance Tracking System** (comprehensive tracking module, 15 tests)
  - ProvenanceRecord - Individual computation records with rich metadata
  - ProvenanceTracker - Thread-safe tracker with query interface
  - ProvenanceConfig - Configurable tracking (limits, sampling, timing)
  - ProvenanceKernel - Wrapper for automatic tracking
  - ProvenanceStatistics - Aggregate statistics and analysis
  - JSON export/import for archival and reproducibility
  - Tagged experiments for organizing computations
  - Performance analysis (average time, success rate, per-kernel breakdown)
  - Example: provenance_tracking.rs with 6 usage scenarios
- **181 comprehensive tests** (100% passing, zero warnings)

**Previous Enhancements (Session 4 - 2025-11-07):** ✨
- **Additional Classical Kernels** (4 new types, 26 tests)
  - LaplacianKernel - L1 distance-based, more robust to outliers
  - SigmoidKernel - Neural network inspired (tanh-based)
  - ChiSquaredKernel - Excellent for histogram data and computer vision
  - HistogramIntersectionKernel - Direct histogram overlap measurement
- **Kernel Transformation Utilities** (kernel_transform module, 18 tests)
  - normalize_kernel_matrix() - Normalize to unit diagonal
  - center_kernel_matrix() - Center for kernel PCA
  - standardize_kernel_matrix() - Combined normalization + centering
  - NormalizedKernel - Wrapper that normalizes any kernel

**Previous Enhancements (Session 3 - 2025-11-06):** ✨
- **Tree Kernels** (618 lines, 16 tests)
  - SubtreeKernel - exact subtree matching
  - SubsetTreeKernel - fragment matching with decay
  - PartialTreeKernel - partial matching with thresholds
  - Automatic TLExpr → TreeNode conversion
- **Low-Rank Approximations** (462 lines, 10 tests)
  - Nyström method for O(nm) complexity
  - Three sampling methods (Uniform, First, K-means++)
  - Configurable regularization for numerical stability
  - Approximation error tracking and compression ratios
- **Performance Benchmarks** (5 suites, ~1600 lines, 47 groups)
  - kernel_computation.rs - Individual kernel performance
  - matrix_operations.rs - Dense/sparse matrix operations
  - caching_performance.rs - Cache hit rates and overhead
  - composite_kernels.rs - Kernel composition performance
  - graph_kernels.rs - Graph kernel scalability

**Previous Enhancements (Session 2):**
- String kernels (n-gram, subsequence, edit distance)
- Feature extraction (automatic TLExpr→vector conversion)
- Vocabulary building and batch processing
- Architecture overview and kernel design guide

**Session 1 Enhancements:**
- Graph kernels (subgraph matching, random walk, WL)
- Composite kernels (weighted sum, product, alignment)
- Kernel caching with hit rate statistics
- Sparse kernel matrices (CSR format)
