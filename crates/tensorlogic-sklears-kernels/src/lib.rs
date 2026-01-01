//! # Tensorlogic-SkleaRS-Kernels
//!
//! **Version**: 0.1.0-alpha.2 | **Status**: Production Ready
//!
//! Logic-derived similarity kernels for machine learning integration.
//!
//! This crate provides kernel functions that measure similarity based on logical
//! rule satisfaction patterns, enabling TensorLogic to integrate with traditional
//! machine learning algorithms (SVMs, kernel methods, etc.).
//!
//! ## Features
//!
//! - ✅ **Rule Similarity Kernels** - Measure similarity by rule satisfaction agreement
//! - ✅ **Predicate Overlap Kernels** - Similarity based on shared true predicates
//! - ✅ **Tensor Kernels** - Classical kernels (Linear, RBF, Polynomial, Cosine)
//! - ✅ **Kernel Composition** - Combine multiple kernels
//! - ✅ **Type-Safe API** - Builder pattern with validation
//! - ✅ **Comprehensive Tests** - 25+ tests covering all components
//!
//! ## Architecture
//!
//! ### Kernel Trait
//!
//! All kernels implement the `Kernel` trait:
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{Kernel, LinearKernel};
//!
//! let kernel = LinearKernel::new();
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let similarity = kernel.compute(&x, &y).unwrap();
//! ```
//!
//! ### Logic-Derived Kernels
//!
//! #### Rule Similarity
//!
//! Measures similarity based on which rules are satisfied:
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{
//!     RuleSimilarityKernel, RuleSimilarityConfig, Kernel
//! };
//! use tensorlogic_ir::TLExpr;
//!
//! // Define logical rules
//! let rules = vec![
//!     TLExpr::pred("rule1", vec![]),
//!     TLExpr::pred("rule2", vec![]),
//!     TLExpr::pred("rule3", vec![]),
//! ];
//!
//! // Configure similarity weights
//! let config = RuleSimilarityConfig::new()
//!     .with_satisfied_weight(1.0)    // Both satisfy
//!     .with_violated_weight(0.5)     // Both violate
//!     .with_mixed_weight(0.0);       // Disagree
//!
//! let kernel = RuleSimilarityKernel::new(rules, config).unwrap();
//!
//! // Compute similarity
//! let x = vec![1.0, 1.0, 0.0];  // Satisfies first two rules
//! let y = vec![1.0, 1.0, 1.0];  // Satisfies all three rules
//! let sim = kernel.compute(&x, &y).unwrap();
//! ```
//!
//! #### Predicate Overlap
//!
//! Measures similarity by counting shared true predicates:
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{PredicateOverlapKernel, Kernel};
//!
//! let kernel = PredicateOverlapKernel::new(4);
//!
//! let x = vec![1.0, 1.0, 0.0, 0.0];  // First two predicates true
//! let y = vec![1.0, 1.0, 1.0, 0.0];  // First three predicates true
//! let sim = kernel.compute(&x, &y).unwrap();
//! // Similarity = 2/4 = 0.5 (two shared true predicates)
//! ```
//!
//! ### Tensor-Based Kernels
//!
//! #### Linear Kernel
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{LinearKernel, Kernel};
//!
//! let kernel = LinearKernel::new();
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let sim = kernel.compute(&x, &y).unwrap();
//! // Similarity = dot(x, y) = 1*4 + 2*5 + 3*6 = 32
//! ```
//!
//! #### RBF (Gaussian) Kernel
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig, Kernel};
//!
//! let config = RbfKernelConfig::new(0.5);  // gamma = 0.5
//! let kernel = RbfKernel::new(config).unwrap();
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![1.0, 2.0, 3.0];
//! let sim = kernel.compute(&x, &y).unwrap();
//! // Similarity = exp(-gamma * ||x-y||^2) = 1.0 (same vectors)
//! ```
//!
//! #### Polynomial Kernel
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{PolynomialKernel, Kernel};
//!
//! let kernel = PolynomialKernel::new(2, 1.0).unwrap();  // degree=2, constant=1
//!
//! let x = vec![1.0, 2.0];
//! let y = vec![3.0, 4.0];
//! let sim = kernel.compute(&x, &y).unwrap();
//! // Similarity = (dot(x,y) + c)^d = (11 + 1)^2 = 144
//! ```
//!
//! #### Cosine Similarity
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{CosineKernel, Kernel};
//!
//! let kernel = CosineKernel::new();
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![2.0, 4.0, 6.0];  // Parallel to x
//! let sim = kernel.compute(&x, &y).unwrap();
//! // Similarity = cos(angle) = 1.0 (parallel vectors)
//! ```
//!
//! ## Kernel Matrix Computation
//!
//! All kernels support efficient matrix computation:
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{LinearKernel, Kernel};
//!
//! let kernel = LinearKernel::new();
//! let inputs = vec![
//!     vec![1.0, 2.0],
//!     vec![3.0, 4.0],
//!     vec![5.0, 6.0],
//! ];
//!
//! let matrix = kernel.compute_matrix(&inputs).unwrap();
//! // matrix[i][j] = kernel(inputs[i], inputs[j])
//! // Symmetric positive semi-definite matrix
//! ```
//!
//! ## Integration with TensorLogic
//!
//! Kernels can be built from compiled TensorLogic expressions:
//!
//! ```rust,ignore
//! use tensorlogic_ir::TLExpr;
//! use tensorlogic_compiler::CompilerContext;
//! use tensorlogic_sklears_kernels::RuleSimilarityKernel;
//!
//! // Define logical rules
//! let rule = TLExpr::exists(
//!     "x",
//!     "Person",
//!     TLExpr::pred("likes", vec![/* ... */]),
//! );
//!
//! // Compile to kernel
//! let kernel = RuleSimilarityKernel::from_rules(vec![rule]).unwrap();
//!
//! // Use in machine learning
//! let data = vec![/* feature vectors */];
//! let kernel_matrix = kernel.compute_matrix(&data).unwrap();
//! ```
//!
//! ## Use Cases
//!
//! ### 1. Kernel SVM with Logical Features
//!
//! Use rule satisfaction as features for SVM:
//!
//! ```rust,ignore
//! let rules = extract_rules_from_knowledge_base();
//! let kernel = RuleSimilarityKernel::new(rules, config).unwrap();
//! let svm = KernelSVM::new(kernel);
//! svm.fit(training_data, labels);
//! ```
//!
//! ### 2. Semantic Similarity
//!
//! Measure semantic similarity between documents/entities:
//!
//! ```rust,ignore
//! let predicates = extract_predicates_from_text(document);
//! let kernel = PredicateOverlapKernel::new(predicates.len());
//! let similarity = kernel.compute(&doc1_features, &doc2_features).unwrap();
//! ```
//!
//! ### 3. Hybrid Kernels
//!
//! Combine logical and tensor-based features:
//!
//! ```rust,ignore
//! let logic_kernel = RuleSimilarityKernel::new(rules, config).unwrap();
//! let tensor_kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
//!
//! // Weighted combination
//! let alpha = 0.7;
//! let similarity = alpha * logic_kernel.compute(x, y).unwrap()
//!                + (1.0 - alpha) * tensor_kernel.compute(x_emb, y_emb).unwrap();
//! ```
//!
//! ## Design Philosophy
//!
//! 1. **Backend Independence**: Kernels work with any feature representation
//! 2. **Composability**: Mix logical and tensor-based similarities
//! 3. **Type Safety**: Compile-time validation where possible
//! 4. **Performance**: Efficient matrix operations
//! 5. **Interpretability**: Clear mapping from logic to similarity

pub mod ard_kernel;
pub mod cache;
pub mod composite_kernel;
pub mod error;
pub mod feature_extraction;
pub mod gradient;
pub mod graph_kernel;
pub mod kernel_selection;
pub mod kernel_transform;
pub mod kernel_utils;
pub mod kpca;
pub mod logic_kernel;
pub mod low_rank;
pub mod multitask;
pub mod online;
pub mod provenance;
pub mod random_features;
#[cfg(feature = "sklears")]
pub mod sklears_integration;
pub mod sparse;
pub mod spectral_kernel;
pub mod string_kernel;
pub mod symbolic;
pub mod tensor_kernels;
pub mod tree_kernel;
pub mod types;

// Re-export main types for convenience
pub use ard_kernel::{
    ArdMaternKernel, ArdRationalQuadraticKernel, ArdRbfKernel, ConstantKernel, DotProductKernel,
    KernelGradient, ScaledKernel, WhiteNoiseKernel,
};
pub use cache::{CacheStats, CachedKernel, KernelMatrixCache};
pub use composite_kernel::{KernelAlignment, ProductKernel, WeightedSumKernel};
pub use error::{KernelError, Result};
pub use feature_extraction::{FeatureExtractionConfig, FeatureExtractor};
pub use graph_kernel::{
    Graph, RandomWalkKernel, SubgraphMatchingConfig, SubgraphMatchingKernel, WalkKernelConfig,
    WeisfeilerLehmanConfig, WeisfeilerLehmanKernel,
};
pub use kernel_selection::{
    CrossValidationResult, GammaSearchResult, KFoldConfig, KernelComparison, KernelSelector,
};
pub use kernel_transform::NormalizedKernel;
pub use logic_kernel::{PredicateOverlapKernel, RuleSimilarityKernel};
pub use low_rank::{NystromApproximation, NystromConfig, SamplingMethod};
pub use multitask::{
    HadamardTaskKernel, ICMKernel, ICMKernelWrapper, IndexKernel, LMCKernel, LMCKernelWrapper,
    MultiTaskConfig, MultiTaskKernelBuilder, TaskInput,
};
pub use online::{
    AdaptiveKernelMatrix, ForgetfulConfig, ForgetfulKernelMatrix, OnlineConfig, OnlineKernelMatrix,
    OnlineStats, WindowedKernelMatrix,
};
pub use provenance::{
    ComputationResult, ProvenanceConfig, ProvenanceId, ProvenanceKernel, ProvenanceRecord,
    ProvenanceStatistics, ProvenanceTracker,
};
pub use random_features::{
    KernelType as RffKernelType, NystroemFeatures, OrthogonalRandomFeatures, RandomFourierFeatures,
    RffConfig,
};
#[cfg(feature = "sklears")]
pub use sklears_integration::SklearsKernelAdapter;
pub use sparse::{SparseKernelMatrix, SparseKernelMatrixBuilder};
pub use spectral_kernel::{
    ExpSineSquaredKernel, LocallyPeriodicKernel, RbfLinearKernel, SpectralComponent,
    SpectralMixtureKernel,
};
pub use string_kernel::{
    EditDistanceKernel, NGramKernel, NGramKernelConfig, SubsequenceKernel, SubsequenceKernelConfig,
};
pub use symbolic::{KernelBuilder, KernelExpr, SymbolicKernel};
pub use tensor_kernels::{
    ChiSquaredKernel, CosineKernel, HistogramIntersectionKernel, LaplacianKernel, LinearKernel,
    MaternKernel, PeriodicKernel, PolynomialKernel, RationalQuadraticKernel, RbfKernel,
    SigmoidKernel,
};
pub use tree_kernel::{
    PartialTreeKernel, PartialTreeKernelConfig, SubsetTreeKernel, SubsetTreeKernelConfig,
    SubtreeKernel, SubtreeKernelConfig, TreeNode,
};
pub use types::{Kernel, PredicateOverlapConfig, RbfKernelConfig, RuleSimilarityConfig};

// Legacy compatibility
#[deprecated(since = "0.1.0", note = "Use RuleSimilarityKernel instead")]
pub fn logic_kernel_similarity() {
    // Legacy function - use RuleSimilarityKernel instead
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_trait_usage() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(LinearKernel::new()),
            Box::new(CosineKernel::new()),
            Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()),
            Box::new(PolynomialKernel::new(2, 1.0).unwrap()),
        ];

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        for kernel in kernels {
            let result = kernel.compute(&x, &y);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(LinearKernel::new().name(), "Linear");
        assert_eq!(CosineKernel::new().name(), "Cosine");
        assert_eq!(
            RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap().name(),
            "RBF"
        );
        assert_eq!(PolynomialKernel::new(2, 1.0).unwrap().name(), "Polynomial");
    }

    #[test]
    fn test_psd_property() {
        let kernel = LinearKernel::new();
        assert!(kernel.is_psd());

        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        assert!(kernel.is_psd());
    }
}
