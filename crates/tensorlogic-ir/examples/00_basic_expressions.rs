//! Basic Expression Construction
//!
//! This example demonstrates how to construct basic logical expressions
//! using the TensorLogic IR.

use tensorlogic_ir::{IrError, TLExpr, Term};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Basic Expressions ===\n");

    // 1. Simple Predicates
    println!("1. Simple Predicates:");
    let person_x = TLExpr::pred("Person", vec![Term::var("x")]);
    println!("   Person(x) = {:?}", person_x);

    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    println!("   knows(x, y) = {:?}", knows_xy);

    // 2. Logical Connectives
    println!("\n2. Logical Connectives:");

    // AND: Person(x) ∧ Mortal(x)
    let person = TLExpr::pred("Person", vec![Term::var("x")]);
    let mortal = TLExpr::pred("Mortal", vec![Term::var("x")]);
    let _and_expr = TLExpr::and(person.clone(), mortal.clone());
    println!("   AND: Person(x) ∧ Mortal(x)");

    // OR: Student(x) ∨ Teacher(x)
    let student = TLExpr::pred("Student", vec![Term::var("x")]);
    let teacher = TLExpr::pred("Teacher", vec![Term::var("x")]);
    let _or_expr = TLExpr::or(student, teacher);
    println!("   OR: Student(x) ∨ Teacher(x)");

    // NOT: ¬Mortal(x)
    let _not_expr = TLExpr::negate(mortal.clone());
    println!("   NOT: ¬Mortal(x)");

    // 3. Implication
    println!("\n3. Implication:");

    // Person(x) → Mortal(x)
    let _implication = TLExpr::imply(person.clone(), mortal.clone());
    println!("   Person(x) → Mortal(x)");

    // 4. Nested Expressions
    println!("\n4. Nested Expressions:");

    // (Person(x) ∧ Wise(x)) → Respected(x)
    let wise = TLExpr::pred("Wise", vec![Term::var("x")]);
    let respected = TLExpr::pred("Respected", vec![Term::var("x")]);
    let _person_and_wise = TLExpr::and(person.clone(), wise);
    let _complex_rule = TLExpr::imply(_person_and_wise.clone(), respected);
    println!("   (Person(x) ∧ Wise(x)) → Respected(x)");

    // 5. Constants vs Variables
    println!("\n5. Constants vs Variables:");

    let alice = Term::constant("alice");
    let bob = Term::constant("bob");
    let x_var = Term::var("x");

    println!("   Variable: {:?}", x_var);
    println!("   Constant: {:?}", alice);

    // knows(alice, bob)
    let _knows_alice_bob = TLExpr::pred("knows", vec![alice.clone(), bob.clone()]);
    println!("   knows(alice, bob) - concrete predicate");

    // knows(alice, x)
    let _knows_alice_x = TLExpr::pred("knows", vec![alice.clone(), x_var.clone()]);
    println!("   knows(alice, x) - partially grounded predicate");

    // 6. Free Variable Analysis
    println!("\n6. Free Variable Analysis:");

    let expr1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let free_vars = expr1.free_vars();
    println!("   Expression: P(x, y)");
    println!("   Free variables: {:?}", free_vars);

    let expr2 = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x"), Term::var("z")]),
    );
    let free_vars2 = expr2.free_vars();
    println!("   Expression: P(x) ∧ Q(x, z)");
    println!("   Free variables: {:?}", free_vars2);

    // 7. Arity Validation
    println!("\n7. Arity Validation:");

    // Valid: P(x, y) ∧ P(a, b) - same arity
    let p1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let p2 = TLExpr::pred("P", vec![Term::var("a"), Term::var("b")]);
    let valid_expr = TLExpr::and(p1, p2);

    match valid_expr.validate_arity() {
        Ok(_) => println!("   ✓ P(x,y) ∧ P(a,b) - valid (same arity)"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // Invalid: P(x) ∧ P(a, b) - different arity
    let p3 = TLExpr::pred("P", vec![Term::var("x")]);
    let p4 = TLExpr::pred("P", vec![Term::var("a"), Term::var("b")]);
    let invalid_expr = TLExpr::and(p3, p4);

    match invalid_expr.validate_arity() {
        Ok(_) => println!("   ✓ Valid"),
        Err(e) => println!("   ✗ P(x) ∧ P(a,b) - invalid: {:?}", e),
    }

    // 8. Extracting Predicates
    println!("\n8. Extracting Predicates:");

    let complex = TLExpr::and(
        TLExpr::pred("Person", vec![Term::var("x")]),
        TLExpr::or(
            TLExpr::pred("Teacher", vec![Term::var("x")]),
            TLExpr::pred("Student", vec![Term::var("x")]),
        ),
    );

    let predicates = complex.all_predicates();
    println!("   Expression: Person(x) ∧ (Teacher(x) ∨ Student(x))");
    println!("   All predicates: {:?}", predicates);

    println!("\n=== Example Complete ===");

    Ok(())
}
