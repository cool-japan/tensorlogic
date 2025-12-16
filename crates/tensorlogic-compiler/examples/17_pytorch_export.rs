//! # PyTorch Code Generation Example
//!
//! This example demonstrates how to compile logic expressions to einsum graphs
//! and generate PyTorch nn.Module Python code for integration with PyTorch workflows.
//!
//! ## Features Demonstrated
//!
//! 1. **Basic Logic to PyTorch**: Generate PyTorch code from predicate logic
//! 2. **Complex Expressions**: Export arithmetic and logical operations
//! 3. **Quantifiers**: Export quantified expressions with reductions
//! 4. **Configuration**: Use custom export settings (class name, dtype, decorators)
//!
//! ## PyTorch Integration
//!
//! The generated Python code can be:
//! - Executed directly in PyTorch eager mode
//! - Traced with `torch.jit.trace()` for optimization
//! - Scripted with `torch.jit.script()` for deployment
//! - Exported to ONNX via PyTorch's export functionality
//! - Used in training loops or inference pipelines
//!
//! ## Running this Example
//!
//! ```bash
//! cargo run --example 17_pytorch_export --features pytorch
//! ```

use anyhow::Result;

#[cfg(feature = "pytorch")]
use tensorlogic_compiler::{
    compile_to_einsum,
    export::pytorch::{
        export_to_pytorch, export_to_pytorch_with_config, PyTorchDtype, PyTorchExportConfig,
    },
    CompilerContext,
};

#[cfg(feature = "pytorch")]
use tensorlogic_ir::{TLExpr, Term};

fn main() -> Result<()> {
    #[cfg(not(feature = "pytorch"))]
    println!("This example requires the 'pytorch' feature to be enabled.");
    #[cfg(not(feature = "pytorch"))]
    println!("Run with: cargo run --example 17_pytorch_export --features pytorch");

    #[cfg(feature = "pytorch")]
    {
        println!("=== PyTorch Code Generation Example ===\n");

        example_1_simple_predicate()?;
        example_2_logical_operations()?;
        example_3_quantified_expression()?;
        example_4_arithmetic_operations()?;
        example_5_custom_configuration()?;
        example_6_complex_rule()?;

        println!("\n=== Summary ===");
        println!("✓ All examples completed successfully");
        println!("✓ PyTorch Python modules created");
        println!("\nNext steps:");
        println!("1. Load the .py files in PyTorch:");
        println!("   ```python");
        println!("   import torch");
        println!("   from model import TensorLogicModel");
        println!("   ");
        println!("   model = TensorLogicModel()");
        println!("   inputs = {{\"tensor_0\": torch.rand(10)}}");
        println!("   output = model(inputs)");
        println!("   ```");
        println!("2. Trace for TorchScript:");
        println!("   ```python");
        println!("   traced = torch.jit.trace(model, inputs)");
        println!("   traced.save('model.pt')");
        println!("   ```");
        println!("3. Use in training or inference");
    }

    Ok(())
}

#[cfg(feature = "pytorch")]
fn example_1_simple_predicate() -> Result<()> {
    println!("## Example 1: Simple Predicate Export");
    println!("Compiling: knows(x, y)");

    // Compile a simple predicate
    let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let graph = compile_to_einsum(&expr)?;

    println!("  Graph nodes: {}", graph.nodes.len());
    println!("  Graph tensors: {}", graph.tensors.len());

    // Export to PyTorch
    let pytorch_code = export_to_pytorch(&graph, "SimplePredicate")?;
    println!("  PyTorch code length: {} bytes", pytorch_code.len());
    println!("  Generated class: SimplePredicate");

    // Optionally write to file
    #[cfg(not(test))]
    {
        std::fs::write("/tmp/simple_predicate.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/simple_predicate.py");
    }

    // Show snippet of generated code
    let lines: Vec<&str> = pytorch_code.lines().take(15).collect();
    println!("\n  Generated code preview:");
    for line in lines {
        println!("  | {}", line);
    }

    println!();
    Ok(())
}

#[cfg(feature = "pytorch")]
fn example_2_logical_operations() -> Result<()> {
    println!("## Example 2: Logical Operations Export");
    println!("Compiling: P(x) ∧ Q(x)");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let graph = compile_to_einsum(&expr)?;
    let pytorch_code = export_to_pytorch(&graph, "LogicalAnd")?;

    println!("  Operators: AND (element-wise multiplication)");
    println!("  PyTorch op: * (multiplication)");
    println!("  Code length: {} bytes", pytorch_code.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/logical_and.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/logical_and.py");
    }

    println!();
    Ok(())
}

#[cfg(feature = "pytorch")]
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
    let pytorch_code = export_to_pytorch(&graph, "ExistentialQuery")?;

    println!("  Quantifier: EXISTS (reduction over y-axis)");
    println!("  PyTorch op: torch.sum");
    println!("  Domain: Person[100]");
    println!("  Code length: {} bytes", pytorch_code.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/existential_query.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/existential_query.py");
    }

    println!();
    Ok(())
}

#[cfg(feature = "pytorch")]
fn example_4_arithmetic_operations() -> Result<()> {
    println!("## Example 4: Arithmetic Operations Export");
    println!("Compiling: (a + b) * c");

    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::pred("c", vec![Term::var("x")]);

    let sum = TLExpr::add(a, b);
    let expr = TLExpr::mul(sum, c);

    let graph = compile_to_einsum(&expr)?;
    let pytorch_code = export_to_pytorch(&graph, "ArithmeticExpr")?;

    println!("  Operations: Add, Mul");
    println!("  PyTorch ops: +, *");
    println!("  Code length: {} bytes", pytorch_code.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/arithmetic_expr.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/arithmetic_expr.py");
    }

    println!();
    Ok(())
}

#[cfg(feature = "pytorch")]
fn example_5_custom_configuration() -> Result<()> {
    println!("## Example 5: Custom Export Configuration");
    println!("Using Float64 dtype and TorchScript decorators");

    let expr = TLExpr::pred("score", vec![Term::var("x")]);
    let graph = compile_to_einsum(&expr)?;

    let config = PyTorchExportConfig {
        class_name: "CustomModel".to_string(),
        default_dtype: PyTorchDtype::Float64,
        add_jit_decorators: true,
        indent: "  ".to_string(), // 2-space indentation
    };

    let pytorch_code = export_to_pytorch_with_config(&graph, config)?;

    println!("  Data type: torch.float64 (double precision)");
    println!("  TorchScript: enabled (@torch.jit.export)");
    println!("  Indentation: 2 spaces");
    println!("  Code length: {} bytes", pytorch_code.len());

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/custom_model.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/custom_model.py");
    }

    // Show that decorators are present
    if pytorch_code.contains("@torch.jit.export") {
        println!("  ✓ TorchScript decorators added");
    }

    println!();
    Ok(())
}

#[cfg(feature = "pytorch")]
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
    let pytorch_code = export_to_pytorch(&graph, "MortalityRule")?;

    println!("  Structure: Universal quantifier + Implication");
    println!("  PyTorch ops: -, torch.nn.functional.relu, torch.min");
    println!("  Domain: Entity[50]");
    println!("  Code length: {} bytes", pytorch_code.len());
    println!("  Interpretation: All persons are mortal");

    #[cfg(not(test))]
    {
        std::fs::write("/tmp/mortality_rule.py", &pytorch_code)?;
        println!("  ✓ Written to /tmp/mortality_rule.py");

        // Also show the generated code
        println!("\n  Generated Python module:");
        println!("  {}", "-".repeat(60));
        for (i, line) in pytorch_code.lines().enumerate() {
            println!("  {:3} | {}", i + 1, line);
        }
        println!("  {}", "-".repeat(60));
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_1() {
        assert!(example_1_simple_predicate().is_ok());
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_2() {
        assert!(example_2_logical_operations().is_ok());
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_3() {
        assert!(example_3_quantified_expression().is_ok());
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_4() {
        assert!(example_4_arithmetic_operations().is_ok());
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_5() {
        assert!(example_5_custom_configuration().is_ok());
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_example_6() {
        assert!(example_6_complex_rule().is_ok());
    }
}
