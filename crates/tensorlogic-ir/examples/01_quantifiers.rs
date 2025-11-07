//! Quantifiers: Exists and ForAll
//!
//! This example demonstrates how to work with existential (∃) and universal (∀)
//! quantifiers in TensorLogic IR.

use tensorlogic_ir::{DomainInfo, DomainRegistry, IrError, TLExpr, Term};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Quantifiers ===\n");

    // Create a domain registry with built-in domains
    let mut registry = DomainRegistry::with_builtins();

    // Add custom domains
    registry.register(DomainInfo::finite("Person", 100))?;
    registry.register(DomainInfo::finite("City", 50))?;

    // 1. Existential Quantifier (∃)
    println!("1. Existential Quantifier (∃):");

    // ∃x. Person(x)
    // "There exists some x such that x is a Person"
    let _exists_person =
        TLExpr::exists("x", "Person", TLExpr::pred("Person", vec![Term::var("x")]));
    println!("   ∃x. Person(x) - 'There exists a person'");

    // ∃x. (Person(x) ∧ Wise(x))
    // "There exists someone who is both a person and wise"
    let person = TLExpr::pred("Person", vec![Term::var("x")]);
    let wise = TLExpr::pred("Wise", vec![Term::var("x")]);
    let _exists_wise_person = TLExpr::exists("x", "Person", TLExpr::and(person, wise));
    println!("   ∃x. (Person(x) ∧ Wise(x)) - 'There exists a wise person'");

    // 2. Universal Quantifier (∀)
    println!("\n2. Universal Quantifier (∀):");

    // ∀x. Person(x) → Mortal(x)
    // "For all x, if x is a person, then x is mortal"
    let person = TLExpr::pred("Person", vec![Term::var("x")]);
    let mortal = TLExpr::pred("Mortal", vec![Term::var("x")]);
    let _all_persons_mortal =
        TLExpr::forall("x", "Person", TLExpr::imply(person.clone(), mortal.clone()));
    println!("   ∀x. Person(x) → Mortal(x) - 'All persons are mortal'");

    // ∀x. (Student(x) → StudiesHard(x))
    let student = TLExpr::pred("Student", vec![Term::var("x")]);
    let studies = TLExpr::pred("StudiesHard", vec![Term::var("x")]);
    let _all_students_study = TLExpr::forall("x", "Person", TLExpr::imply(student, studies));
    println!("   ∀x. Student(x) → StudiesHard(x) - 'All students study hard'");

    // 3. Nested Quantifiers
    println!("\n3. Nested Quantifiers:");

    // ∀x. ∃y. knows(x, y)
    // "For every person x, there exists someone y that x knows"
    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let exists_y_knows = TLExpr::exists("y", "Person", knows_xy.clone());
    let _everyone_knows_someone = TLExpr::forall("x", "Person", exists_y_knows);
    println!("   ∀x. ∃y. knows(x, y) - 'Everyone knows someone'");

    // ∃x. ∀y. (Person(y) → knows(x, y))
    // "There exists someone who knows all persons"
    let person_y = TLExpr::pred("Person", vec![Term::var("y")]);
    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let knows_all = TLExpr::imply(person_y, knows_xy);
    let forall_y_knows = TLExpr::forall("y", "Person", knows_all);
    let _someone_knows_all = TLExpr::exists("x", "Person", forall_y_knows);
    println!("   ∃x. ∀y. (Person(y) → knows(x, y)) - 'Someone knows everyone'");

    // 4. Multiple Variable Quantification
    println!("\n4. Multiple Variable Quantification:");

    // ∃x. ∃y. (Person(x) ∧ Person(y) ∧ friends(x, y))
    // "There exist two persons who are friends"
    let person_x = TLExpr::pred("Person", vec![Term::var("x")]);
    let person_y = TLExpr::pred("Person", vec![Term::var("y")]);
    let friends = TLExpr::pred("friends", vec![Term::var("x"), Term::var("y")]);
    let both_persons_and_friends = TLExpr::and(TLExpr::and(person_x, person_y), friends);
    let exists_y_friends = TLExpr::exists("y", "Person", both_persons_and_friends);
    let _exists_friends_pair = TLExpr::exists("x", "Person", exists_y_friends);
    println!("   ∃x. ∃y. (Person(x) ∧ Person(y) ∧ friends(x, y)) - 'Two people are friends'");

    // 5. Free Variables After Quantification
    println!("\n5. Free Variables After Quantification:");

    // Before quantification: P(x, y)
    let expr = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    println!("   Expression: P(x, y)");
    println!("   Free variables: {:?}", expr.free_vars());

    // After binding x: ∃x. P(x, y)
    let exists_x = TLExpr::exists("x", "Person", expr.clone());
    println!("   Expression: ∃x. P(x, y)");
    println!("   Free variables: {:?}", exists_x.free_vars());

    // After binding both: ∃x. ∃y. P(x, y)
    let exists_xy = TLExpr::exists("y", "Person", exists_x);
    println!("   Expression: ∃x. ∃y. P(x, y)");
    println!("   Free variables: {:?}", exists_xy.free_vars());

    // 6. Domain Validation
    println!("\n6. Domain Validation:");

    // Valid: quantifier over registered domain
    let valid_expr = TLExpr::exists("x", "Person", TLExpr::pred("Person", vec![Term::var("x")]));
    match valid_expr.validate_domains(&registry) {
        Ok(_) => println!("   ✓ Domain 'Person' is valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // Invalid: quantifier over unregistered domain
    let invalid_expr = TLExpr::exists("x", "Alien", TLExpr::pred("Alien", vec![Term::var("x")]));
    match invalid_expr.validate_domains(&registry) {
        Ok(_) => println!("   ✓ Valid"),
        Err(e) => println!("   ✗ Domain 'Alien' not registered: {:?}", e),
    }

    // 7. Complex Quantified Expressions
    println!("\n7. Complex Quantified Expressions:");

    // ∀x. (Person(x) → ∃y. (City(y) ∧ livesIn(x, y)))
    // "Every person lives in some city"
    let person_x = TLExpr::pred("Person", vec![Term::var("x")]);
    let city_y = TLExpr::pred("City", vec![Term::var("y")]);
    let lives_in = TLExpr::pred("livesIn", vec![Term::var("x"), Term::var("y")]);
    let city_and_lives = TLExpr::and(city_y, lives_in);
    let exists_city = TLExpr::exists("y", "City", city_and_lives);
    let person_lives_somewhere = TLExpr::imply(person_x, exists_city);
    let everyone_lives_somewhere = TLExpr::forall("x", "Person", person_lives_somewhere);

    println!("   ∀x. (Person(x) → ∃y. (City(y) ∧ livesIn(x, y)))");
    println!("   'Every person lives in some city'");
    println!(
        "   Free variables: {:?}",
        everyone_lives_somewhere.free_vars()
    );

    // Validate all domains are registered
    match everyone_lives_somewhere.validate_domains(&registry) {
        Ok(_) => println!("   ✓ All domains valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 8. Quantifier Domain Information
    println!("\n8. Quantifier Domain Information:");

    let expr = TLExpr::exists("x", "Person", TLExpr::pred("Person", vec![Term::var("x")]));

    let domains = expr.referenced_domains();
    println!("   Expression: ∃x:Person. Person(x)");
    println!("   Referenced domains: {:?}", domains);

    println!("\n=== Example Complete ===");

    Ok(())
}
