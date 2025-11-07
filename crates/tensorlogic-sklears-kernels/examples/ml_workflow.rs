//! Complete Machine Learning Workflow
//!
//! This example demonstrates a complete ML workflow using various
//! kernel types, caching, and matrix operations.
//!
//! Run with: cargo run --example ml_workflow

use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{
    CachedKernel, CosineKernel, Graph, Kernel, LinearKernel, PolynomialKernel, ProductKernel,
    RbfKernel, RbfKernelConfig, RuleSimilarityConfig, RuleSimilarityKernel,
    SparseKernelMatrixBuilder, WeightedSumKernel, WeisfeilerLehmanConfig, WeisfeilerLehmanKernel,
};

fn main() -> anyhow::Result<()> {
    println!("=== Complete Machine Learning Workflow ===\n");

    // Scenario: Document classification using multiple kernel types
    println!("Scenario: Document Classification System");
    println!("─────────────────────────────────────────");
    println!("Goal: Classify documents using logic rules and text features");
    println!();

    // Step 1: Feature extraction
    println!("STEP 1: Feature Extraction");
    println!("──────────────────────────");

    // Generate document features
    let num_documents = 50;
    let text_feature_dim = 100;

    let text_features: Vec<Vec<f64>> = (0..num_documents)
        .map(|i| {
            (0..text_feature_dim)
                .map(|j| ((i * text_feature_dim + j) as f64).sin())
                .collect()
        })
        .collect();

    println!(
        "✓ Extracted {} text features per document",
        text_feature_dim
    );

    // Generate logic rule features
    let num_rules = 10;
    let rules: Vec<TLExpr> = (0..num_rules)
        .map(|i| TLExpr::pred(format!("rule_{}", i), vec![]))
        .collect();

    let _rule_activations: Vec<Vec<f64>> = (0..num_documents)
        .map(|i| vec![if i % 3 == 0 { 1.0 } else { 0.5 }; num_rules])
        .collect();

    println!("✓ Generated {} logic rule activations", num_rules);
    println!();

    // Step 2: Kernel selection and configuration
    println!("STEP 2: Kernel Selection");
    println!("────────────────────────");

    // Text similarity kernel (cosine for normalized text)
    let text_kernel = CosineKernel::new();
    println!("✓ Text kernel: Cosine similarity");

    // Rule similarity kernel
    let rule_config = RuleSimilarityConfig::new();
    let rule_kernel = RuleSimilarityKernel::new(rules.clone(), rule_config)?;
    println!("✓ Rule kernel: Logic-based similarity");

    // Combine kernels using weighted sum
    let combined_kernel = WeightedSumKernel::new(
        vec![
            Box::new(text_kernel) as Box<dyn Kernel>,
            Box::new(rule_kernel) as Box<dyn Kernel>,
        ],
        vec![0.7, 0.3], // Text is more important
    )?;
    println!("✓ Combined kernel: 0.7×text + 0.3×rules");
    println!();

    // Step 3: Kernel matrix computation with caching
    println!("STEP 3: Kernel Matrix Computation");
    println!("──────────────────────────────────");

    // Wrap in cache for repeated computations
    let cached_kernel = CachedKernel::new(Box::new(combined_kernel));

    // Compute kernel matrix
    let start = std::time::Instant::now();
    let _kernel_matrix = cached_kernel.compute_matrix(&text_features)?;
    let compute_time = start.elapsed();

    println!(
        "✓ Computed {}×{} kernel matrix in {:?}",
        num_documents, num_documents, compute_time
    );

    // Check cache performance
    let stats = cached_kernel.stats();
    println!("✓ Cache hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!();

    // Step 4: Sparse kernel matrix for efficiency
    println!("STEP 4: Sparse Kernel Matrix");
    println!("────────────────────────────");

    // Create new text kernel for sparse computation
    let text_kernel2 = CosineKernel::new();

    let sparse_builder = SparseKernelMatrixBuilder::new()
        .with_threshold(0.5) // Only keep similarities > 0.5
        .unwrap()
        .with_max_entries_per_row(10) // Max 10 neighbors per document
        .unwrap();

    let sparse_matrix = sparse_builder.build(&text_features, &text_kernel2)?;

    println!("✓ Sparse matrix format: CSR (Compressed Sparse Row)");
    println!("  Size: {}×{}", sparse_matrix.size(), sparse_matrix.size());
    println!("  Non-zero entries: {}", sparse_matrix.nnz());
    println!(
        "  Sparsity: {:.1}%",
        (sparse_matrix.nnz() as f64 / (num_documents * num_documents) as f64) * 100.0
    );
    println!();

    // Step 5: Graph-based similarity for structured data
    println!("STEP 5: Graph Kernel for Document Structure");
    println!("────────────────────────────────────────────");

    // Create sample document structure graphs
    let doc1_expr = TLExpr::and(
        TLExpr::pred("title", vec![]),
        TLExpr::and(
            TLExpr::pred("abstract", vec![]),
            TLExpr::pred("body", vec![]),
        ),
    );

    let doc2_expr = TLExpr::and(
        TLExpr::pred("title", vec![]),
        TLExpr::and(
            TLExpr::pred("abstract", vec![]),
            TLExpr::pred("references", vec![]),
        ),
    );

    let graph1 = Graph::from_tlexpr(&doc1_expr);
    let graph2 = Graph::from_tlexpr(&doc2_expr);

    let wl_config = WeisfeilerLehmanConfig::new().with_iterations(3);
    let wl_kernel = WeisfeilerLehmanKernel::new(wl_config);

    let structure_similarity = wl_kernel.compute_graphs(&graph1, &graph2)?;

    println!("✓ Document 1 structure: title → (abstract, body)");
    println!("✓ Document 2 structure: title → (abstract, references)");
    println!("✓ Structure similarity: {:.4}", structure_similarity);
    println!();

    // Step 6: Kernel composition for multi-view learning
    println!("STEP 6: Multi-View Learning");
    println!("───────────────────────────");

    // Different views of the data
    let view1_kernel = LinearKernel::new();
    let view2_kernel = RbfKernel::new(RbfKernelConfig::new(0.5))?;
    let view3_kernel = PolynomialKernel::new(2, 1.0)?;

    // Product kernel combines all views
    let multi_view_kernel = ProductKernel::new(vec![
        Box::new(view1_kernel) as Box<dyn Kernel>,
        Box::new(view2_kernel) as Box<dyn Kernel>,
        Box::new(view3_kernel) as Box<dyn Kernel>,
    ])?;

    let sample1 = vec![0.5; text_feature_dim];
    let sample2 = vec![0.7; text_feature_dim];
    let multi_view_sim = multi_view_kernel.compute(&sample1, &sample2)?;

    println!("✓ View 1: Linear kernel (raw features)");
    println!("✓ View 2: RBF kernel (local similarity)");
    println!("✓ View 3: Polynomial kernel (feature interactions)");
    println!("✓ Combined similarity: {:.4}", multi_view_sim);
    println!();

    // Step 7: Performance summary
    println!("STEP 7: Performance Summary");
    println!("───────────────────────────");
    println!("Dataset size: {} documents", num_documents);
    println!("Feature dimensions: {}", text_feature_dim);
    println!("Kernel matrix size: {}×{}", num_documents, num_documents);
    println!("Computation time: {:?}", compute_time);
    println!(
        "Throughput: {:.0} kernel evaluations/second",
        (num_documents * num_documents) as f64 / compute_time.as_secs_f64()
    );
    println!();

    // Step 8: Practical recommendations
    println!("STEP 8: Best Practices for Production");
    println!("──────────────────────────────────────");
    println!("✓ Use CachedKernel for repeated computations");
    println!("✓ Use SparseKernelMatrix for large datasets");
    println!("✓ Combine kernels with WeightedSumKernel or ProductKernel");
    println!("✓ Use graph kernels for structured/hierarchical data");
    println!("✓ Monitor cache hit rates and adjust cache size");
    println!("✓ Profile different kernel types for your data");
    println!();

    // Step 9: Advanced techniques
    println!("STEP 9: Advanced Techniques");
    println!("───────────────────────────");
    println!("1. Kernel Alignment");
    println!("   → Measure alignment between kernels and target labels");
    println!("   → Use KernelAlignment for kernel selection");
    println!();
    println!("2. Low-Rank Approximation");
    println!("   → Use NystromApproximation for very large datasets");
    println!("   → Reduces O(n²) to O(nm) complexity");
    println!();
    println!("3. Multi-Task Learning");
    println!("   → Share kernels across related tasks");
    println!("   → Use composite kernels with task-specific weights");
    println!();
    println!("4. Online Learning");
    println!("   → Update kernel cache incrementally");
    println!("   → Recompute sparse matrices periodically");
    println!();

    println!("=== Workflow Complete ===");
    println!("\nNext Steps:");
    println!("  • Integrate with SkleaRS for SVM/GP training");
    println!("  • Add cross-validation for kernel parameter tuning");
    println!("  • Profile memory usage for your dataset size");
    println!("  • Experiment with different kernel combinations");

    Ok(())
}
