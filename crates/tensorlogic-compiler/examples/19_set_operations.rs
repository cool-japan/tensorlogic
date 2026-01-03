//! Set Theory Operations Example
//!
//! This example demonstrates the compilation of set-theoretic operations
//! into tensor computations. Sets are represented as characteristic functions
//! (indicator tensors) where 1 indicates membership and 0 indicates non-membership.
//!
//! Run with:
//! ```bash
//! cargo run --example 19_set_operations
//! ```

use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() -> anyhow::Result<()> {
    println!("=== Set Theory Operations in TensorLogic ===\n");

    // Set up domain
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    println!("Domain: Person (size 100)\n");

    // ========== Example 1: Set Comprehension ==========
    println!("Example 1: Set Comprehension");
    println!("------------------------------");
    println!("Set S = {{ x : Person | Adult(x) }}");
    println!("\"Set of all adults\"\n");

    let adult_set = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(TLExpr::pred("Adult", vec![Term::var("x")])),
    };

    let graph = compile_to_einsum_with_context(&adult_set, &mut ctx)?;
    println!("Compiled graph:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  This creates a characteristic function over the Person domain\n");

    // ========== Example 2: Set Union ==========
    println!("Example 2: Set Union");
    println!("--------------------");
    println!("A = {{ x : Person | Student(x) }}");
    println!("B = {{ x : Person | Teacher(x) }}");
    println!("A ∪ B = \"Students or Teachers\"\n");

    let students = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(TLExpr::pred("Student", vec![Term::var("x")])),
    };

    let teachers = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(TLExpr::pred("Teacher", vec![Term::var("x")])),
    };

    let union = TLExpr::SetUnion {
        left: Box::new(students.clone()),
        right: Box::new(teachers.clone()),
    };

    let mut ctx2 = CompilerContext::new();
    ctx2.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&union, &mut ctx2)?;
    println!("Union compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Uses element-wise max(χ_A, χ_B)\n");

    // ========== Example 3: Set Intersection ==========
    println!("Example 3: Set Intersection");
    println!("----------------------------");
    println!("A ∩ B = \"People who are both students and teachers\"\n");

    let intersection = TLExpr::SetIntersection {
        left: Box::new(students.clone()),
        right: Box::new(teachers.clone()),
    };

    let mut ctx3 = CompilerContext::new();
    ctx3.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&intersection, &mut ctx3)?;
    println!("Intersection compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Uses element-wise min(χ_A, χ_B)\n");

    // ========== Example 4: Set Difference ==========
    println!("Example 4: Set Difference");
    println!("-------------------------");
    println!("A \\ B = \"Students who are not teachers\"\n");

    let difference = TLExpr::SetDifference {
        left: Box::new(students.clone()),
        right: Box::new(teachers.clone()),
    };

    let mut ctx4 = CompilerContext::new();
    ctx4.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&difference, &mut ctx4)?;
    println!("Difference compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Uses χ_A * (1 - χ_B)\n");

    // ========== Example 5: Set Cardinality ==========
    println!("Example 5: Set Cardinality");
    println!("--------------------------");
    println!("|A| = \"Number of students\"\n");

    let cardinality = TLExpr::SetCardinality {
        set: Box::new(students.clone()),
    };

    let mut ctx5 = CompilerContext::new();
    ctx5.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&cardinality, &mut ctx5)?;
    println!("Cardinality compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Reduces to a scalar: sum(χ_A)\n");

    // ========== Example 6: Empty Set ==========
    println!("Example 6: Empty Set");
    println!("--------------------");
    println!("∅ = Empty set (constant zero)\n");

    let empty = TLExpr::EmptySet;

    let mut ctx6 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&empty, &mut ctx6)?;
    println!("Empty set compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Represented as a constant zero tensor\n");

    // ========== Example 7: Set Membership ==========
    println!("Example 7: Set Membership");
    println!("-------------------------");
    println!("alice ∈ A = \"Is Alice a student?\"\n");

    let alice_is_student = TLExpr::pred("IsAlice", vec![Term::var("y")]);
    let membership = TLExpr::SetMembership {
        element: Box::new(alice_is_student),
        set: Box::new(students.clone()),
    };

    let mut ctx7 = CompilerContext::new();
    ctx7.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&membership, &mut ctx7)?;
    println!("Membership test compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Checks if element indicator AND set characteristic function both hold\n");

    // ========== Example 8: Nested Set Operations ==========
    println!("Example 8: Nested Set Operations");
    println!("---------------------------------");
    println!("(A ∪ B) ∩ C where:");
    println!("  A = Students");
    println!("  B = Teachers");
    println!("  C = {{ x : Person | Adult(x) }}");
    println!("= \"Adult students or teachers\"\n");

    let adults = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(TLExpr::pred("Adult", vec![Term::var("x")])),
    };

    let union_students_teachers = TLExpr::SetUnion {
        left: Box::new(students),
        right: Box::new(teachers),
    };

    let nested = TLExpr::SetIntersection {
        left: Box::new(union_students_teachers),
        right: Box::new(adults),
    };

    let mut ctx8 = CompilerContext::new();
    ctx8.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&nested, &mut ctx8)?;
    println!("Nested operation compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Combines union and intersection operations\n");

    // ========== Example 9: Set Comprehension with Complex Condition ==========
    println!("Example 9: Set Comprehension with Complex Condition");
    println!("----------------------------------------------------");
    println!("S = {{ x : Person | Adult(x) ∧ ¬Student(x) }}");
    println!("= \"Adults who are not students\"\n");

    let complex_condition = TLExpr::and(
        TLExpr::pred("Adult", vec![Term::var("x")]),
        TLExpr::negate(TLExpr::pred("Student", vec![Term::var("x")])),
    );

    let complex_set = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(complex_condition),
    };

    let mut ctx9 = CompilerContext::new();
    ctx9.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&complex_set, &mut ctx9)?;
    println!("Complex comprehension compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Combines logical operations within set comprehension\n");

    // ========== Example 10: Set Operations with Quantifiers ==========
    println!("Example 10: Set Operations with Quantifiers");
    println!("--------------------------------------------");
    println!("|{{ x : Person | ∃y. Knows(x,y) }}|");
    println!("= \"Number of people who know at least one person\"\n");

    let knows_someone = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("Knows", vec![Term::var("x"), Term::var("y")]),
    );

    let people_who_know_someone = TLExpr::SetComprehension {
        var: "x".to_string(),
        domain: "Person".to_string(),
        condition: Box::new(knows_someone),
    };

    let count = TLExpr::SetCardinality {
        set: Box::new(people_who_know_someone),
    };

    let mut ctx10 = CompilerContext::new();
    ctx10.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&count, &mut ctx10)?;
    println!("Set with quantifier + cardinality compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Combines existential quantification with set comprehension and cardinality\n");

    println!("=== Summary ===");
    println!("\nSet operations compiled successfully!");
    println!("All operations use tensor representations:");
    println!("  • Sets as characteristic functions (0/1 tensors)");
    println!("  • Union as element-wise max");
    println!("  • Intersection as element-wise min");
    println!("  • Difference as masked multiplication");
    println!("  • Cardinality as sum reduction");
    println!("  • Membership as element-wise product");
    println!("\nThese operations can be efficiently executed on tensor backends (CPU/GPU).");

    Ok(())
}
