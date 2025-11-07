//! Serialization and Deserialization
//!
//! This example demonstrates how to serialize and deserialize TensorLogic IR
//! expressions and graphs using JSON and binary formats.

use tensorlogic_ir::{
    serialization::{VersionedExpr, VersionedGraph},
    EinsumGraph, EinsumNode, IrError, TLExpr, Term, FORMAT_VERSION,
};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Serialization ===\n");

    // 1. Expression Serialization to JSON
    println!("1. Expression Serialization to JSON:");

    // Create a logical expression: Person(x) → Mortal(x)
    let person = TLExpr::pred("Person", vec![Term::var("x")]);
    let mortal = TLExpr::pred("Mortal", vec![Term::var("x")]);
    let rule = TLExpr::imply(person, mortal);

    // Serialize to JSON
    let json = serde_json::to_string(&rule).expect("Failed to serialize to JSON");
    println!("   Compact JSON:");
    println!("   {}", json);

    // Pretty JSON
    let pretty_json =
        serde_json::to_string_pretty(&rule).expect("Failed to serialize to pretty JSON");
    println!("\n   Pretty JSON:");
    println!("{}", pretty_json);

    // 2. Expression Deserialization from JSON
    println!("\n2. Expression Deserialization from JSON:");

    let restored: TLExpr = serde_json::from_str(&json).expect("Failed to deserialize from JSON");
    println!("   Original: {:?}", rule);
    println!("   Restored: {:?}", restored);

    // Verify they're equivalent
    let original_json = serde_json::to_string(&rule).expect("Failed to serialize to JSON");
    let restored_json = serde_json::to_string(&restored).expect("Failed to serialize to JSON");
    if original_json == restored_json {
        println!("   ✓ Roundtrip successful!");
    }

    // 3. Versioned Expression Serialization
    println!("\n3. Versioned Expression Serialization:");

    // Create a complex expression
    let expr = TLExpr::forall(
        "x",
        "Person",
        TLExpr::imply(
            TLExpr::pred("Person", vec![Term::var("x")]),
            TLExpr::exists(
                "y",
                "City",
                TLExpr::pred("livesIn", vec![Term::var("x"), Term::var("y")]),
            ),
        ),
    );

    // Wrap in versioned container with metadata
    let mut metadata = serde_json::Map::new();
    metadata.insert("author".to_string(), serde_json::json!("alice"));
    metadata.insert("purpose".to_string(), serde_json::json!("residency_rule"));
    let versioned = VersionedExpr::with_metadata(expr.clone(), metadata);

    let versioned_json =
        serde_json::to_string_pretty(&versioned).expect("Failed to serialize to JSON");
    println!("   Versioned Expression JSON:");
    println!("{}", versioned_json);

    println!("\n   Metadata:");
    println!("   - Format version: {}", FORMAT_VERSION);
    println!(
        "   - Created at: {}",
        versioned.created_at.as_ref().unwrap_or(&"N/A".to_string())
    );
    println!("   - Custom metadata: {:?}", versioned.metadata);

    // 4. Graph Serialization
    println!("\n4. Graph Serialization:");

    let mut graph = EinsumGraph::new();

    // Build a simple computation graph
    let input_a = graph.add_tensor("input_a");
    let input_b = graph.add_tensor("input_b");
    let intermediate = graph.add_tensor("intermediate");
    let output = graph.add_tensor("output");

    graph.add_node(EinsumNode::einsum(
        "ik,kj->ij",
        vec![input_a, input_b],
        vec![intermediate],
    ))?;

    graph.add_node(EinsumNode::elem_unary("relu", intermediate, output))?;

    graph.add_output(output)?;

    // Serialize graph to JSON
    let graph_json =
        serde_json::to_string_pretty(&graph).expect("Failed to serialize graph to JSON");
    println!("   Graph JSON:");
    println!("{}", graph_json);

    // 5. Graph Deserialization
    println!("\n5. Graph Deserialization:");

    let restored_graph: EinsumGraph =
        serde_json::from_str(&graph_json).expect("Failed to deserialize graph from JSON");
    println!("   Original graph:");
    println!("   - Tensors: {}", graph.tensors.len());
    println!("   - Nodes: {}", graph.nodes.len());
    println!("   - Outputs: {}", graph.outputs.len());

    println!("   Restored graph:");
    println!("   - Tensors: {}", restored_graph.tensors.len());
    println!("   - Nodes: {}", restored_graph.nodes.len());
    println!("   - Outputs: {}", restored_graph.outputs.len());

    match restored_graph.validate() {
        Ok(_) => println!("   ✓ Restored graph is valid"),
        Err(e) => println!("   ✗ Validation error: {:?}", e),
    }

    // 6. Versioned Graph Serialization
    println!("\n6. Versioned Graph Serialization:");

    let mut graph_metadata = serde_json::Map::new();
    graph_metadata.insert("model_name".to_string(), serde_json::json!("simple_mlp"));
    graph_metadata.insert(
        "architecture".to_string(),
        serde_json::json!("2_layer_relu"),
    );
    let versioned_graph = VersionedGraph::with_metadata(graph.clone(), graph_metadata);

    let versioned_graph_json = serde_json::to_string_pretty(&versioned_graph)
        .expect("Failed to serialize versioned graph");
    println!("   Versioned Graph JSON (truncated):");
    let lines: Vec<&str> = versioned_graph_json.lines().take(20).collect();
    for line in lines {
        println!("{}", line);
    }
    println!("   ...");

    // 7. Binary Serialization (Bincode)
    println!("\n7. Binary Serialization:");

    // Serialize expression to binary
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("y")]),
    );

    let binary_data = bincode::serde::encode_to_vec(&expr, bincode::config::standard())
        .expect("Failed to serialize to binary");
    println!("   Expression binary size: {} bytes", binary_data.len());

    // Deserialize from binary
    let (_restored_expr, _): (TLExpr, usize) =
        bincode::serde::decode_from_slice(&binary_data, bincode::config::standard())
            .expect("Failed to deserialize from binary");
    println!("   ✓ Binary roundtrip successful");

    // Compare sizes
    let json_size = serde_json::to_string(&expr)
        .expect("Failed to serialize to JSON")
        .len();
    println!("   JSON size: {} bytes", json_size);
    println!("   Binary size: {} bytes", binary_data.len());
    println!(
        "   Binary is {:.1}% of JSON size",
        (binary_data.len() as f64 / json_size as f64) * 100.0
    );

    // 8. Graph Binary Serialization
    println!("\n8. Graph Binary Serialization:");

    let mut large_graph = EinsumGraph::new();

    // Build a larger graph
    let mut current_tensor = large_graph.add_tensor("input");

    for i in 0..10 {
        let next_tensor = large_graph.add_tensor(format!("layer_{}", i));
        large_graph.add_node(EinsumNode::elem_unary("relu", current_tensor, next_tensor))?;
        current_tensor = next_tensor;
    }

    large_graph.add_output(current_tensor)?;

    let graph_binary = bincode::serde::encode_to_vec(&large_graph, bincode::config::standard())
        .expect("Failed to serialize graph to binary");
    let graph_json =
        serde_json::to_string(&large_graph).expect("Failed to serialize graph to JSON");

    println!("   Large graph (10 layers):");
    println!("   - JSON size: {} bytes", graph_json.len());
    println!("   - Binary size: {} bytes", graph_binary.len());
    println!(
        "   - Compression: {:.1}%",
        (graph_binary.len() as f64 / graph_json.len() as f64) * 100.0
    );

    // 9. Saving to Files
    println!("\n9. Saving to Files:");

    use std::env;
    use std::fs;

    let temp_dir = env::temp_dir();

    // Save expression as JSON
    let expr_json_path = temp_dir.join("tensorlogic_expr.json");
    let versioned_expr = VersionedExpr::new(expr.clone());
    let json_content =
        serde_json::to_string_pretty(&versioned_expr).expect("Failed to serialize to JSON");
    fs::write(&expr_json_path, json_content).expect("Failed to write JSON file");
    println!("   ✓ Saved expression to: {:?}", expr_json_path);

    // Save graph as binary
    let graph_bin_path = temp_dir.join("tensorlogic_graph.bin");
    let versioned_graph = VersionedGraph::new(large_graph.clone());
    let binary_content =
        bincode::serde::encode_to_vec(&versioned_graph, bincode::config::standard())
            .expect("Failed to serialize to binary");
    fs::write(&graph_bin_path, binary_content).expect("Failed to write binary file");
    println!("   ✓ Saved graph to: {:?}", graph_bin_path);

    // 10. Loading from Files
    println!("\n10. Loading from Files:");

    // Load expression from JSON
    let loaded_json = fs::read_to_string(&expr_json_path).expect("Failed to read JSON file");
    let loaded_expr: VersionedExpr =
        serde_json::from_str(&loaded_json).expect("Failed to deserialize JSON");
    println!("   ✓ Loaded expression from JSON");
    println!("   - Format version: {}", loaded_expr.version);
    println!(
        "   - Created at: {}",
        loaded_expr
            .created_at
            .as_ref()
            .unwrap_or(&"N/A".to_string())
    );

    // Load graph from binary
    let loaded_binary = fs::read(&graph_bin_path).expect("Failed to read binary file");
    let (loaded_graph, _): (VersionedGraph, usize) =
        bincode::serde::decode_from_slice(&loaded_binary, bincode::config::standard())
            .expect("Failed to deserialize binary");
    println!("   ✓ Loaded graph from binary");
    println!(
        "   - Graph has {} tensors",
        loaded_graph.graph.tensors.len()
    );

    // Clean up
    fs::remove_file(&expr_json_path).ok();
    fs::remove_file(&graph_bin_path).ok();
    println!("   ✓ Cleaned up temporary files");

    // 11. Version Compatibility
    println!("\n11. Version Compatibility:");

    let v1_expr = VersionedExpr::new(TLExpr::pred("P", vec![Term::var("x")]));
    println!("   Current format version: {}", v1_expr.version);

    // In production, you would check:
    // if v1_expr.is_compatible_with("1.0.0") { ... }
    println!("   ✓ Version information preserved for compatibility checks");

    // 12. Metadata Preservation
    println!("\n12. Metadata Preservation:");

    let mut metadata_map = serde_json::Map::new();
    metadata_map.insert("rule_id".to_string(), serde_json::json!("mortality_axiom"));
    metadata_map.insert("confidence".to_string(), serde_json::json!("1.0"));
    metadata_map.insert("source".to_string(), serde_json::json!("knowledge_base.kb"));

    let metadata_expr = VersionedExpr::with_metadata(
        TLExpr::forall("x", "Person", TLExpr::pred("Mortal", vec![Term::var("x")])),
        metadata_map,
    );

    let meta_json =
        serde_json::to_string_pretty(&metadata_expr).expect("Failed to serialize to JSON");
    println!("   Expression with metadata:");
    println!("{}", meta_json);

    println!("\n=== Example Complete ===");

    Ok(())
}
