use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Tensorlogic Example: Exists Quantifier with Reduction ===\n");

    // Define domains
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 10);
    ctx.add_domain("City", 5);

    println!("Domains defined:");
    println!("  Person: cardinality = 10");
    println!("  City: cardinality = 5\n");

    // Example 1: Basic exists quantifier
    // ∃z ∈ Person. Friend(x, z)
    // "There exists a person z who is a friend of x"
    println!("Example 1: ∃z ∈ Person. Friend(x, z)");
    println!("  'For each person x, check if there exists a friend z'");

    let friend_pred = TLExpr::pred("Friend", vec![Term::var("x"), Term::var("z")]);
    let exists_friend = TLExpr::exists("z", "Person", friend_pred);

    match compile_to_einsum_with_context(&exists_friend, &mut ctx.clone()) {
        Ok(graph) => {
            println!("  Compiled successfully!");
            println!("  Tensors: {}", graph.tensors.len());
            println!("  Nodes: {}", graph.nodes.len());
            println!("  Graph tensors: {:?}\n", graph.tensors);
        }
        Err(e) => println!("  Compilation error: {}\n", e),
    }

    // Example 2: Exists with conjunction
    // ∃z ∈ Person. (Friend(x, z) ∧ LivesIn(z, "Tokyo"))
    // "There exists a person z who is a friend of x AND lives in Tokyo"
    println!("Example 2: ∃z ∈ Person. (Friend(x, z) ∧ LivesIn(z, Tokyo))");
    println!("  'For each person x, check if they have a friend in Tokyo'");

    let friend_pred2 = TLExpr::pred("Friend", vec![Term::var("x"), Term::var("z")]);
    let lives_in_tokyo = TLExpr::pred("LivesIn", vec![Term::var("z"), Term::constant("Tokyo")]);
    let and_expr = TLExpr::and(friend_pred2, lives_in_tokyo);
    let exists_friend_tokyo = TLExpr::exists("z", "Person", and_expr);

    match compile_to_einsum_with_context(&exists_friend_tokyo, &mut ctx.clone()) {
        Ok(graph) => {
            println!("  Compiled successfully!");
            println!("  Tensors: {}", graph.tensors.len());
            println!("  Nodes: {}", graph.nodes.len());
            for (i, tensor) in graph.tensors.iter().enumerate() {
                println!("    Tensor {}: {}", i, tensor);
            }
            println!();
        }
        Err(e) => println!("  Compilation error: {}\n", e),
    }

    // Example 3: Nested quantifiers
    // ∃y ∈ City. ∃z ∈ Person. (LivesIn(z, y) ∧ Friend(x, z))
    // "There exists a city y and a person z such that z lives in y and is a friend of x"
    println!("Example 3: ∃y ∈ City. ∃z ∈ Person. (LivesIn(z, y) ∧ Friend(x, z))");
    println!("  'For each person x, check if they have a friend living somewhere'");

    let lives_in = TLExpr::pred("LivesIn", vec![Term::var("z"), Term::var("y")]);
    let friend = TLExpr::pred("Friend", vec![Term::var("x"), Term::var("z")]);
    let inner_and = TLExpr::and(lives_in, friend);
    let exists_z = TLExpr::exists("z", "Person", inner_and);
    let exists_y_z = TLExpr::exists("y", "City", exists_z);

    match compile_to_einsum_with_context(&exists_y_z, &mut ctx.clone()) {
        Ok(graph) => {
            println!("  Compiled successfully!");
            println!("  Tensors: {}", graph.tensors.len());
            println!("  Nodes: {}", graph.nodes.len());
            println!("  This demonstrates reduction over multiple axes:");
            for (i, node) in graph.nodes.iter().enumerate() {
                println!("    Node {}: op = {:?}", i, node.op);
            }
            println!();
        }
        Err(e) => println!("  Compilation error: {}\n", e),
    }

    println!("=== Compilation Complete ===");
    println!("\nNote: These examples demonstrate how existential quantifiers");
    println!("compile to tensor reductions. The actual tensor execution would");
    println!("happen with a backend like SciRS2.");
}
