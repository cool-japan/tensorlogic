use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Tensorlogic Example: Minimal Rule ===\n");

    // Define a simple implication rule:
    // Parent(a, b) → Ancestor(a, b)
    // "If a is a parent of b, then a is an ancestor of b"

    let parent = TLExpr::pred("Parent", vec![Term::var("a"), Term::var("b")]);

    let ancestor = TLExpr::pred("Ancestor", vec![Term::var("a"), Term::var("b")]);

    let rule = TLExpr::imply(parent, ancestor);

    println!("Rule: Parent(a, b) → Ancestor(a, b)");
    println!("Compiling logic rule to tensor graph...\n");

    match compile_to_einsum(&rule) {
        Ok(plan) => {
            println!("Compilation successful!");
            println!("  Tensors: {}", plan.tensors.len());
            println!("  Nodes: {}", plan.nodes.len());
            println!("\nGenerated tensors:");
            for (i, tensor) in plan.tensors.iter().enumerate() {
                println!("  {}: {}", i, tensor);
            }
            println!("\nGenerated nodes:");
            for (i, node) in plan.nodes.iter().enumerate() {
                println!("  {}: {:?} (inputs: {:?})", i, node.op, node.inputs);
            }
        }
        Err(e) => {
            eprintln!("Compilation error: {}", e);
        }
    }

    println!("\n=== Compilation Complete ===");
}
