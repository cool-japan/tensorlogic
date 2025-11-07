//! Rule-Based Attention Example
//!
//! This example demonstrates how to use rule-based attention patterns for
//! interpretable and controllable transformers.
//!
//! Run with: `cargo run --example 03_rule_based_attention`

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    rule_attention::patterns, AttentionConfig, Result, RuleAttentionConfig, RuleBasedAttention,
    StructuredAttention,
};

fn main() -> Result<()> {
    println!("=== Rule-Based Attention Example ===\n");

    // Example 1: Hard Rule-Based Attention
    println!("1. Hard constraint: only attend where rule is satisfied...");

    let base_config = AttentionConfig::new(512, 8)?;
    let hard_config = RuleAttentionConfig::hard(base_config.clone());

    let rule = patterns::syntactic_dependency("head", "dep");
    let hard_attn = RuleBasedAttention::new(hard_config)?.with_rule(rule);

    println!("   ✓ Created hard rule-based attention");
    println!("   ✓ Rule: SyntacticDep(head, dep)");
    println!("   ✓ Effect: Attention is ONLY allowed on syntactic dependencies");
    println!("   ✓ Use case: Syntax-aware language models\n");

    // Build graph
    let mut graph = EinsumGraph::new();
    graph.add_tensor("Q");
    graph.add_tensor("K");
    graph.add_tensor("V");
    graph.add_tensor("rule_mask"); // Binary mask from rule evaluation

    let outputs = hard_attn.build_rule_attention_graph(&mut graph)?;
    println!("   ✓ Built graph with {} outputs\n", outputs.len());

    // Example 2: Soft Rule-Based Attention
    println!("2. Soft constraint: bias attention towards rule-satisfying positions...");

    let soft_config = RuleAttentionConfig::soft(base_config.clone(), 0.7);
    let coref_rule = patterns::coreference("mention1", "mention2");
    let _soft_attn = RuleBasedAttention::new(soft_config)?.with_rule(coref_rule);

    println!("   ✓ Created soft rule-based attention with strength=0.7");
    println!("   ✓ Rule: Coref(mention1, mention2)");
    println!("   ✓ Effect: Attention is BIASED towards coreferent mentions");
    println!("   ✓ Use case: Coreference-aware reading comprehension\n");

    // Example 3: Gated Rule-Based Attention
    println!("3. Gated: interpolate between content and rule attention...");

    let gated_config = RuleAttentionConfig::gated(base_config.clone(), 0.5);
    let hier_rule = patterns::hierarchical("sentence", "paragraph");
    let _gated_attn = RuleBasedAttention::new(gated_config)?.with_rule(hier_rule);

    println!("   ✓ Created gated attention with rule_weight=0.5");
    println!("   ✓ Rule: ContainedIn(sentence, paragraph)");
    println!("   ✓ Effect: 50% content-based + 50% rule-based attention");
    println!("   ✓ Use case: Hierarchical document modeling\n");

    // Example 4: Structured Attention (Pure Rule-Based)
    println!("4. Structured attention: attention defined purely by predicates...");

    let struct_config = AttentionConfig::new(512, 8)?;
    let struct_predicate = patterns::syntactic_dependency("token_i", "token_j");
    let struct_attn = StructuredAttention::new(struct_config)?.with_predicate(struct_predicate);

    println!("   ✓ Created structured attention");
    println!("   ✓ Predicate: SyntacticDep(token_i, token_j)");
    println!("   ✓ Effect: Attention weights come ONLY from predicate");
    println!("   ✓ Use case: Fully interpretable attention patterns\n");

    let mut struct_graph = EinsumGraph::new();
    struct_graph.add_tensor("V"); // Values
    struct_graph.add_tensor("structure_matrix"); // From predicate evaluation

    let struct_outputs = struct_attn.build_structured_attention_graph(&mut struct_graph)?;
    println!(
        "   ✓ Built structured attention graph with {} outputs\n",
        struct_outputs.len()
    );

    // Example 5: Common Rule Patterns
    println!("5. Available pre-defined rule patterns...");

    println!("\n   a) Syntactic Dependency:");
    let _syn_rule = patterns::syntactic_dependency("governor", "dependent");
    println!("      Pattern: SyntacticDep(governor, dependent)");
    println!("      Use: Parse-tree aware attention");

    println!("\n   b) Coreference:");
    let _coref = patterns::coreference("entity1", "entity2");
    println!("      Pattern: Coref(entity1, entity2)");
    println!("      Use: Entity-centric models");

    println!("\n   c) Semantic Similarity:");
    let _sem_rule = patterns::semantic_similarity("word1", "word2", 0.8);
    println!("      Pattern: Similarity(word1, word2) > 0.8");
    println!("      Use: Semantic-aware attention");

    println!("\n   d) Hierarchical Structure:");
    let _hier = patterns::hierarchical("child_span", "parent_span");
    println!("      Pattern: ContainedIn(child_span, parent_span)");
    println!("      Use: Hierarchical text modeling\n");

    // Example 6: Combining Rules with Different Strengths
    println!("6. Practical application: Multi-rule attention...");

    println!("\n   Scenario: Question Answering with multiple constraints");
    println!("   Rule 1: Soft syntax guidance (strength=0.3)");
    println!("   Rule 2: Hard coreference constraint");
    println!("   Rule 3: Gated semantic similarity (weight=0.4)");

    let _qa_soft = RuleAttentionConfig::soft(base_config.clone(), 0.3);
    let _qa_hard = RuleAttentionConfig::hard(base_config.clone());
    let _qa_gated = RuleAttentionConfig::gated(base_config, 0.4);

    println!("\n   ✓ Can create multiple attention layers with different rules");
    println!("   ✓ Stack in encoder to combine multiple constraints");
    println!("   ✓ Interpretable: know why model attends to specific positions\n");

    // Example 7: Rule Types Comparison
    println!("7. Rule types comparison...");
    println!("\n   ╔══════════════════╦══════════════════════════════════════════╗");
    println!("   ║ Type             ║ Behavior                                 ║");
    println!("   ╠══════════════════╬══════════════════════════════════════════╣");
    println!("   ║ Hard             ║ Mask out non-rule positions              ║");
    println!("   ║                  ║ (zero attention weight)                  ║");
    println!("   ╠══════════════════╬══════════════════════════════════════════╣");
    println!("   ║ Soft (strength)  ║ Add bias to scores proportional to       ║");
    println!("   ║                  ║ rule satisfaction × strength             ║");
    println!("   ╠══════════════════╬══════════════════════════════════════════╣");
    println!("   ║ Gated (weight)   ║ Interpolate: weight×rule + (1-w)×content║");
    println!("   ║                  ║ Balances interpretability & performance  ║");
    println!("   ╠══════════════════╬══════════════════════════════════════════╣");
    println!("   ║ Structured       ║ Pure rule-based, no content attention    ║");
    println!("   ║                  ║ Fully deterministic & interpretable      ║");
    println!("   ╚══════════════════╩══════════════════════════════════════════╝\n");

    // Example 8: Validation
    println!("8. Configuration validation...");

    // Valid configurations
    let valid_soft = RuleAttentionConfig::soft(AttentionConfig::new(512, 8)?, 0.5);
    println!(
        "   ✓ Soft with strength=0.5: {}",
        valid_soft.validate().is_ok()
    );

    let valid_gated = RuleAttentionConfig::gated(AttentionConfig::new(512, 8)?, 0.7);
    println!(
        "   ✓ Gated with weight=0.7: {}",
        valid_gated.validate().is_ok()
    );

    // Invalid configurations
    let invalid_soft = RuleAttentionConfig::soft(AttentionConfig::new(512, 8)?, 1.5);
    println!(
        "   ✗ Soft with strength=1.5: {}",
        invalid_soft.validate().is_err()
    );

    let invalid_gated = RuleAttentionConfig::gated(AttentionConfig::new(512, 8)?, -0.1);
    println!(
        "   ✗ Gated with weight=-0.1: {}\n",
        invalid_gated.validate().is_err()
    );

    println!("=== Rule-based attention example completed successfully! ===");
    Ok(())
}
