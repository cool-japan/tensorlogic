//! Example: Metadata propagation for debugging and provenance tracking
//!
//! This example demonstrates how to attach metadata to compiled tensor graphs,
//! enabling better debugging, provenance tracking, and understanding of the
//! compilation process.

use tensorlogic_compiler::passes::{
    propagate_metadata, MetadataBuilder, MetadataCompilationResult,
};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{Metadata, Provenance, SourceLocation, SourceSpan, TLExpr, Term};

fn main() {
    println!("=== Metadata Propagation Example ===\n");

    // Example 1: Basic metadata builder usage
    println!("1. Basic MetadataBuilder:");
    let mut builder = MetadataBuilder::new()
        .with_source_file("rules.tl")
        .with_rule_id("social_network");

    let pred_meta = builder.predicate_metadata("knows", &["x".to_string(), "y".to_string()]);
    println!("   Predicate metadata name: {:?}", pred_meta.name);
    println!(
        "   Source file: {:?}",
        pred_meta.get_attribute("source_file")
    );
    println!("   Rule ID: {:?}\n", pred_meta.get_attribute("rule_id"));

    // Example 2: Metadata from expression
    println!("2. Metadata from TLExpr:");
    let expr = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    );
    let expr_meta = builder.from_expr(&expr);
    println!("   Expression metadata name: {:?}", expr_meta.name);
    println!(
        "   Quantifier type: {:?}",
        expr_meta.get_attribute("quantifier")
    );
    println!("   Domain: {:?}\n", expr_meta.get_attribute("domain"));

    // Example 3: Compile with metadata propagation
    println!("3. Compiling with metadata propagation:");
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    let rule = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    );

    let mut graph = compile_to_einsum_with_context(&rule, &mut ctx).unwrap();

    // Add metadata to the graph
    let mut metadata_builder = MetadataBuilder::new().with_source_file("social_network.tl");
    propagate_metadata(&mut graph, &ctx, &mut metadata_builder);

    println!("   Graph has {} tensors", graph.tensors.len());
    println!(
        "   Metadata attached to {} tensors",
        graph.tensor_metadata.len()
    );

    for (idx, meta) in &graph.tensor_metadata {
        println!("\n   Tensor {}: {}", idx, graph.tensors[*idx]);
        if let Some(name) = &meta.name {
            println!("     Name: {}", name);
        }
        if let Some(domain) = meta.get_attribute("domain") {
            println!("     Domain: {}", domain);
        }
        if let Some(tensor_type) = meta.get_attribute("tensor_type") {
            println!("     Type: {}", tensor_type);
        }
    }

    // Example 4: Node metadata
    println!("\n4. Node metadata:");
    let node = graph.nodes.first();
    if let Some(n) = node {
        println!("   First node: {}", n.operation_description());
        if let Some(meta) = n.get_metadata() {
            println!("   Node has metadata: {:?}", meta.name);
        } else {
            println!("   Node has no metadata (can be added during compilation)");
        }
    }

    // Example 5: Advanced metadata with provenance
    println!("\n5. Advanced metadata with provenance:");
    let provenance = Provenance::new()
        .with_rule_id("transitivity_rule")
        .with_source_file("rules.tl")
        .with_attribute("author", "user")
        .with_attribute("version", "1.0");

    let span = SourceSpan::single(SourceLocation::new("rules.tl", 42, 5));

    let advanced_meta = Metadata::new()
        .with_name("transitivity")
        .with_span(span.clone())
        .with_provenance(provenance)
        .with_attribute("complexity", "high");

    println!("   Metadata name: {:?}", advanced_meta.name);
    println!("   Source span: {}", span);
    if let Some(prov) = &advanced_meta.provenance {
        println!("   Rule ID: {:?}", prov.rule_id);
        println!("   Author: {:?}", prov.get_attribute("author"));
    }

    // Example 6: Using MetadataCompilationResult
    println!("\n6. MetadataCompilationResult:");
    let mut result = MetadataCompilationResult::new(graph.clone(), metadata_builder);
    result.record_expression("exists_knows", vec![0, 1]);
    result.record_expression("predicate_knows", vec![2]);

    println!("   Tracked expressions: {}", result.expr_to_nodes.len());
    for (expr_id, nodes) in &result.expr_to_nodes {
        println!("   Expression '{}' -> nodes: {:?}", expr_id, nodes);
    }

    println!("\n=== Complete! ===");
    println!("\nMetadata propagation enables:");
    println!("  • Tracking which source rules generated which tensor operations");
    println!("  • Debugging compilation by inspecting intermediate metadata");
    println!("  • Building provenance chains from input to output");
    println!("  • Understanding the relationship between logic and tensors");
}
