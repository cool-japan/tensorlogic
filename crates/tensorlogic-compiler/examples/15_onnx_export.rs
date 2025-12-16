//! Demonstrates ONNX export functionality for compiled TensorLogic graphs.
//!
//! This example shows how to export compiled logical expressions to ONNX format,
//! enabling execution on any ONNX-compatible runtime (ONNX Runtime, PyTorch, TensorFlow, etc.).
//!
//! Run with:
//! ```sh
//! cargo run --example 15_onnx_export --features onnx
//! ```

#[cfg(feature = "onnx")]
use tensorlogic_compiler::{
    compile_to_einsum, compile_to_einsum_with_context,
    export::onnx::{export_to_onnx, export_to_onnx_with_config, DataType, OnnxExportConfig},
    CompilerContext,
};

#[cfg(feature = "onnx")]
use tensorlogic_ir::{TLExpr, Term};

#[cfg(feature = "onnx")]
fn main() -> anyhow::Result<()> {
    println!("=== ONNX Export Example ===\n");

    // Example 1: Simple predicate conjunction
    println!("1. Simple Conjunction (P ∧ Q)");
    let expr1 = TLExpr::And(
        Box::new(TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        }),
        Box::new(TLExpr::Pred {
            name: "Q".to_string(),
            args: vec![Term::Var("x".to_string())],
        }),
    );

    let graph1 = compile_to_einsum(&expr1)?;
    println!("   Compiled graph:");
    println!("     Tensors: {:?}", graph1.tensors);
    println!("     Nodes: {}", graph1.nodes.len());

    let onnx_bytes1 = export_to_onnx(&graph1, "conjunction_model")?;
    println!("   ONNX model size: {} bytes\n", onnx_bytes1.len());

    // Example 2: Existential quantification
    println!("2. Existential Quantification (∃y. knows(x,y))");
    let expr2 = TLExpr::Exists {
        var: "y".to_string(),
        domain: "Person".to_string(),
        body: Box::new(TLExpr::Pred {
            name: "knows".to_string(),
            args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
        }),
    };

    // Create context with domain definitions for quantified expressions
    let mut ctx2 = CompilerContext::new();
    ctx2.add_domain("Person", 100); // Domain size of 100 persons

    let graph2 = compile_to_einsum_with_context(&expr2, &mut ctx2)?;
    println!("   Compiled graph:");
    println!("     Tensors: {:?}", graph2.tensors);
    println!("     Nodes: {}", graph2.nodes.len());

    let onnx_bytes2 = export_to_onnx(&graph2, "exists_model")?;
    println!("   ONNX model size: {} bytes\n", onnx_bytes2.len());

    // Example 3: Implication rule
    println!("3. Implication Rule (P(x) → Q(x))");
    let expr3 = TLExpr::Imply(
        Box::new(TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        }),
        Box::new(TLExpr::Pred {
            name: "Q".to_string(),
            args: vec![Term::Var("x".to_string())],
        }),
    );

    let graph3 = compile_to_einsum(&expr3)?;
    println!("   Compiled graph:");
    println!("     Tensors: {:?}", graph3.tensors);
    println!("     Nodes: {}", graph3.nodes.len());

    let onnx_bytes3 = export_to_onnx(&graph3, "implication_model")?;
    println!("   ONNX model size: {} bytes\n", onnx_bytes3.len());

    // Example 4: Complex transitivity rule with custom configuration
    println!("4. Transitivity Rule with Custom Config");
    println!("   Rule: knows(x,y) ∧ knows(y,z) → knows(x,z)");

    let knows_xy = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
    };

    let knows_yz = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("y".to_string()), Term::Var("z".to_string())],
    };

    let knows_xz = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("z".to_string())],
    };

    let premise = TLExpr::And(Box::new(knows_xy), Box::new(knows_yz));
    let expr4 = TLExpr::Imply(Box::new(premise), Box::new(knows_xz));

    let graph4 = compile_to_einsum(&expr4)?;
    println!("   Compiled graph:");
    println!("     Tensors: {:?}", graph4.tensors);
    println!("     Nodes: {}", graph4.nodes.len());

    // Export with custom configuration
    let custom_config = OnnxExportConfig {
        model_name: "transitivity_rule".to_string(),
        opset_version: 14,                // Use ONNX opset version 14
        default_dtype: DataType::Float64, // Use double precision
    };

    let onnx_bytes4 = export_to_onnx_with_config(&graph4, custom_config)?;
    println!("   ONNX model size: {} bytes", onnx_bytes4.len());
    println!("   Configuration: opset=14, dtype=Float64\n");

    // Example 5: Save models to files
    println!("5. Saving Models to Files");

    // Save conjunction model
    std::fs::write("/tmp/conjunction.onnx", &onnx_bytes1)?;
    println!(
        "   ✓ Saved: /tmp/conjunction.onnx ({} bytes)",
        onnx_bytes1.len()
    );

    // Save existential quantification model
    std::fs::write("/tmp/exists.onnx", &onnx_bytes2)?;
    println!("   ✓ Saved: /tmp/exists.onnx ({} bytes)", onnx_bytes2.len());

    // Save implication model
    std::fs::write("/tmp/implication.onnx", &onnx_bytes3)?;
    println!(
        "   ✓ Saved: /tmp/implication.onnx ({} bytes)",
        onnx_bytes3.len()
    );

    // Save transitivity model
    std::fs::write("/tmp/transitivity.onnx", &onnx_bytes4)?;
    println!(
        "   ✓ Saved: /tmp/transitivity.onnx ({} bytes)",
        onnx_bytes4.len()
    );

    println!("\n=== Usage Notes ===");
    println!("These ONNX models can be loaded and executed using:");
    println!("  • ONNX Runtime (C++, Python, C#, Java, JavaScript)");
    println!("  • PyTorch (torch.onnx.load)");
    println!("  • TensorFlow (tf.import_graph_def)");
    println!("  • Any ONNX-compatible inference engine");

    println!("\nExample Python usage:");
    println!("```python");
    println!("import onnxruntime as ort");
    println!("session = ort.InferenceSession('/tmp/conjunction.onnx')");
    println!("result = session.run(None, {{'P_x_': P_data, 'Q_x_': Q_data}})");
    println!("```");

    println!("\n=== ONNX Export Complete ===");

    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("This example requires the 'onnx' feature.");
    eprintln!("Run with: cargo run --example 15_onnx_export --features onnx");
    std::process::exit(1);
}
