//! # TensorFlow GraphDef Export Example
//!
//! This example demonstrates how to compile logic expressions to einsum graphs
//! and export them to TensorFlow GraphDef format for execution in TensorFlow.
//!
//! ## Features Demonstrated
//!
//! 1. **Basic Logic to TensorFlow**: Compile predicate logic to TensorFlow ops
//! 2. **Complex Expressions**: Export arithmetic and logical operations
//! 3. **Quantifiers**: Export quantified expressions with reductions
//! 4. **Configuration**: Use custom export settings (dtype, output formatting)
//!
//! ## TensorFlow Integration
//!
//! The exported GraphDef can be:
//! - Loaded in TensorFlow using `tf.import_graph_def()`
//! - Saved as a TensorFlow SavedModel
//! - Executed on CPU, GPU, or TPU via TensorFlow runtime
//! - Integrated into TensorFlow Serving for production deployment
//!
//! ## Running this Example
//!
//! ```bash
//! cargo run --example 16_tensorflow_export --features tensorflow
//! ```

use anyhow::Result;

#[cfg(feature = "tensorflow")]
use tensorlogic_compiler::{
    compile_to_einsum,
    export::tensorflow::{
        export_to_tensorflow, export_to_tensorflow_with_config, TensorFlowExportConfig, TfDataType,
    },
    CompilerContext,
};

#[cfg(feature = "tensorflow")]
use tensorlogic_ir::{TLExpr, Term};

fn main() -> Result<()> {
    #[cfg(not(feature = "tensorflow"))]
    println!("This example requires the 'tensorflow' feature to be enabled.");
    #[cfg(not(feature = "tensorflow"))]
    println!("Run with: cargo run --example 16_tensorflow_export --features tensorflow");

    #[cfg(feature = "tensorflow")]
    {
        println!("=== TensorFlow GraphDef Export Example ===\n");

        example_1_simple_predicate()?;
        example_2_logical_operations()?;
        example_3_quantified_expression()?;
        example_4_arithmetic_operations()?;
        example_5_custom_configuration()?;
        example_6_complex_rule()?;

        println!("\n=== Summary ===");
        println!("✓ All examples completed successfully");
        println!("✓ TensorFlow GraphDef files created");
        println!("\nNext steps:");
        println!("1. Load the .pb files in TensorFlow:");
        println!("   ```python");
        println!("   import tensorflow as tf");
        println!("   with tf.io.gfile.GFile('model.pb', 'rb') as f:");
        println!("       graph_def = tf.compat.v1.GraphDef()");
        println!("       graph_def.ParseFromString(f.read())");
        println!("   ```");
        println!("2. Import into TensorFlow graph:");
        println!("   ```python");
        println!("   with tf.Graph().as_default() as graph:");
        println!("       tf.import_graph_def(graph_def, name='')");
        println!("   ```");
        println!("3. Execute operations in TensorFlow session");
    }

    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_1_simple_predicate() -> Result<()> {
    println!("## Example 1: Simple Predicate Export");
    println!("Compiling: knows(x, y)");

    // Compile a simple predicate
    let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let graph = compile_to_einsum(&expr)?;

    println!("  Graph nodes: {}", graph.nodes.len());
    println!("  Graph tensors: {}", graph.tensors.len());

    // Export to TensorFlow
    let tf_bytes = export_to_tensorflow(&graph, "simple_predicate")?;
    println!("  TensorFlow GraphDef size: {} bytes", tf_bytes.len());

    // Optionally write to file
    #[cfg(not(test))]
    {
        std::fs::write("/tmp/simple_predicate.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/simple_predicate.pb");
    }

    println!();
    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_2_logical_operations() -> Result<()> {
    println!("## Example 2: Logical Operations Export");
    println!("Compiling: P(x) ∧ Q(x)");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let graph = compile_to_einsum(&expr)?;
    let tf_bytes = export_to_tensorflow(&graph, "logical_and")?;

    println!("  Operators: AND (element-wise multiplication)");
    println!("  TensorFlow op: Mul");
    println!("  GraphDef size: {} bytes", tf_bytes.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/logical_and.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/logical_and.pb");
    }

    println!();
    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_3_quantified_expression() -> Result<()> {
    println!("## Example 3: Quantified Expression Export");
    println!("Compiling: ∃y. knows(x, y)");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    let expr = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    );

    let graph = tensorlogic_compiler::compile_to_einsum_with_context(&expr, &mut ctx)?;
    let tf_bytes = export_to_tensorflow(&graph, "existential_query")?;

    println!("  Quantifier: EXISTS (reduction over y-axis)");
    println!("  TensorFlow op: Sum");
    println!("  Domain: Person[100]");
    println!("  GraphDef size: {} bytes", tf_bytes.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/existential_query.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/existential_query.pb");
    }

    println!();
    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_4_arithmetic_operations() -> Result<()> {
    println!("## Example 4: Arithmetic Operations Export");
    println!("Compiling: (a + b) * c");

    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::pred("c", vec![Term::var("x")]);

    let sum = TLExpr::add(a, b);
    let expr = TLExpr::mul(sum, c);

    let graph = compile_to_einsum(&expr)?;
    let tf_bytes = export_to_tensorflow(&graph, "arithmetic_expr")?;

    println!("  Operations: Add, Mul");
    println!("  TensorFlow ops: Add, Mul");
    println!("  GraphDef size: {} bytes", tf_bytes.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/arithmetic_expr.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/arithmetic_expr.pb");
    }

    println!();
    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_5_custom_configuration() -> Result<()> {
    println!("## Example 5: Custom Export Configuration");
    println!("Using Float64 dtype and no identity outputs");

    let expr = TLExpr::pred("score", vec![Term::var("x")]);
    let graph = compile_to_einsum(&expr)?;

    let config = TensorFlowExportConfig {
        model_name: "custom_model".to_string(),
        default_dtype: TfDataType::Float64,
        add_identity_outputs: false,
    };

    let tf_bytes = export_to_tensorflow_with_config(&graph, config)?;

    println!("  Data type: Float64 (double precision)");
    println!("  Identity outputs: disabled");
    println!("  GraphDef size: {} bytes", tf_bytes.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/custom_model.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/custom_model.pb");
    }

    println!();
    Ok(())
}

#[cfg(feature = "tensorflow")]
fn example_6_complex_rule() -> Result<()> {
    println!("## Example 6: Complex Logical Rule Export");
    println!("Compiling: ∀x. (Person(x) → Mortal(x))");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Entity", 50);

    let person = TLExpr::pred("Person", vec![Term::var("x")]);
    let mortal = TLExpr::pred("Mortal", vec![Term::var("x")]);
    let implication = TLExpr::imply(person, mortal);
    let rule = TLExpr::forall("x", "Entity", implication);

    let graph = tensorlogic_compiler::compile_to_einsum_with_context(&rule, &mut ctx)?;
    let tf_bytes = export_to_tensorflow(&graph, "complex_rule")?;

    println!("  Structure: Universal quantifier + Implication");
    println!("  TensorFlow ops: Sub, Relu, Min (or Product)");
    println!("  Domain: Entity[50]");
    println!("  GraphDef size: {} bytes", tf_bytes.len());
    println!("  Interpretation: All persons are mortal");

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/complex_rule.pb", &tf_bytes)?;
        println!("  ✓ Written to /tmp/complex_rule.pb");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_1() {
        assert!(example_1_simple_predicate().is_ok());
    }

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_2() {
        assert!(example_2_logical_operations().is_ok());
    }

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_3() {
        assert!(example_3_quantified_expression().is_ok());
    }

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_4() {
        assert!(example_4_arithmetic_operations().is_ok());
    }

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_5() {
        assert!(example_5_custom_configuration().is_ok());
    }

    #[test]
    #[cfg(feature = "tensorflow")]
    fn test_example_6() {
        assert!(example_6_complex_rule().is_ok());
    }
}
