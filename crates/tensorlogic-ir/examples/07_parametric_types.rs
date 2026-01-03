//! Example demonstrating parametric type system with type constructors and unification.
//!
//! This example shows how to use the parametric type system for generic predicates
//! and type-safe reasoning with polymorphic types.

use std::collections::HashMap;
use tensorlogic_ir::parametric_types::{
    compose_substitutions, generalize, instantiate, unify, ParametricType, TypeConstructor,
};
use tensorlogic_ir::PredicateSignature;

fn main() {
    println!("=== Parametric Types Example ===\n");

    // 1. Basic Type Constructors
    println!("1. Type Constructors and Kinds");
    demonstrate_type_constructors();
    println!();

    // 2. Parametric Types
    println!("2. Parametric Types (Generics)");
    demonstrate_parametric_types();
    println!();

    // 3. Type Unification
    println!("3. Type Unification");
    demonstrate_unification();
    println!();

    // 4. Parametric Predicate Signatures
    println!("4. Parametric Predicate Signatures");
    demonstrate_parametric_signatures();
    println!();

    // 5. Complex Type Unification
    println!("5. Complex Type Unification");
    demonstrate_complex_unification();
    println!();

    // 6. Type Generalization and Instantiation
    println!("6. Type Generalization and Instantiation");
    demonstrate_generalization();
    println!();
}

fn demonstrate_type_constructors() {
    // Type constructors have kinds that describe their arity
    println!("  List has kind: {}", TypeConstructor::List.kind());
    println!("  Option has kind: {}", TypeConstructor::Option.kind());
    println!("  Tuple has kind: {}", TypeConstructor::Tuple.kind());
    println!("  Function has kind: {}", TypeConstructor::Function.kind());

    // Create custom type constructors
    let tree = TypeConstructor::custom("Tree", 1);
    println!("  Custom Tree has kind: {}", tree.kind());
}

fn demonstrate_parametric_types() {
    // Concrete types
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let person_type = ParametricType::concrete("Person");

    println!("  Concrete types:");
    println!("    Int: {}", int_type);
    println!("    String: {}", string_type);
    println!("    Person: {}", person_type);

    // Type variables
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");

    println!("\n  Type variables:");
    println!("    T: {}", t);
    println!("    U: {}", u);

    // Parametric types (type constructor applications)
    let list_int = ParametricType::list(int_type.clone());
    let option_string = ParametricType::option(string_type.clone());
    let list_t = ParametricType::list(t.clone());

    println!("\n  Parametric types:");
    println!("    List<Int>: {}", list_int);
    println!("    Option<String>: {}", option_string);
    println!("    List<T>: {}", list_t);

    // Nested parametric types
    let list_list_int = ParametricType::list(list_int.clone());
    let option_list_string = ParametricType::option(ParametricType::list(string_type.clone()));

    println!("\n  Nested parametric types:");
    println!("    List<List<Int>>: {}", list_list_int);
    println!("    Option<List<String>>: {}", option_list_string);

    // Function types
    let int_to_string = ParametricType::function(int_type.clone(), string_type.clone());
    println!("\n  Function types:");
    println!("    Int -> String: {}", int_to_string);

    // Map types
    let map_string_person = ParametricType::map(string_type.clone(), person_type.clone());
    println!("\n  Map types:");
    println!("    Map<String, Person>: {}", map_string_person);

    // Free variables
    let complex_type = ParametricType::tuple(vec![list_t, ParametricType::option(u.clone())]);
    println!("\n  Free variables in Tuple<List<T>, Option<U>>:");
    println!("    {:?}", complex_type.free_variables());
}

fn demonstrate_unification() {
    // Unify concrete types
    let int1 = ParametricType::concrete("Int");
    let int2 = ParametricType::concrete("Int");

    match unify(&int1, &int2) {
        Ok(subst) => {
            println!("  Unify Int with Int:");
            println!("    Success! Substitution: {:?}", subst);
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Unify variable with concrete type
    let t = ParametricType::variable("T");
    let int_type = ParametricType::concrete("Int");

    match unify(&t, &int_type) {
        Ok(subst) => {
            println!("\n  Unify T with Int:");
            println!("    Success! T = {}", subst.get("T").unwrap());
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Unify parametric types
    let list_t = ParametricType::list(ParametricType::variable("T"));
    let list_int = ParametricType::list(ParametricType::concrete("Int"));

    match unify(&list_t, &list_int) {
        Ok(subst) => {
            println!("\n  Unify List<T> with List<Int>:");
            println!("    Success! T = {}", subst.get("T").unwrap());
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Occurs check (should fail)
    let t = ParametricType::variable("T");
    let list_t = ParametricType::list(t.clone());

    match unify(&t, &list_t) {
        Ok(_) => println!("\n  Unify T with List<T>: Unexpected success!"),
        Err(e) => {
            println!("\n  Unify T with List<T>:");
            println!("    Failed (expected): {}", e);
        }
    }
}

fn demonstrate_parametric_signatures() {
    // Signature for a polymorphic contains predicate: List<T> x T -> Bool
    let t = ParametricType::variable("T");
    let contains_sig = PredicateSignature::parametric(
        "contains",
        vec![ParametricType::list(t.clone()), t.clone()],
    );

    println!("  Polymorphic signature:");
    println!("    contains: List<T> x T -> Bool");
    println!("    Arity: {}", contains_sig.arity);
    println!("    Is parametric: {}", contains_sig.is_parametric());

    // Unify with concrete types
    let int_type = ParametricType::concrete("Int");
    let list_int = ParametricType::list(int_type.clone());

    match contains_sig.unify_parametric(&[list_int, int_type.clone()]) {
        Ok(subst) => {
            println!("\n  Unifying contains signature with (List<Int>, Int):");
            println!("    Success! T = {}", subst.get("T").unwrap());

            // Instantiate the signature
            let instantiated = contains_sig.instantiate(&subst);
            println!("    Instantiated signature:");
            if let Some(param_types) = instantiated.get_parametric_types() {
                for (i, ty) in param_types.iter().enumerate() {
                    println!("      Arg {}: {}", i, ty);
                }
            }
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Signature for map_over: (T -> U) x List<T> -> List<U>
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let map_sig = PredicateSignature::parametric(
        "map_over",
        vec![
            ParametricType::function(t.clone(), u.clone()),
            ParametricType::list(t.clone()),
            ParametricType::list(u.clone()),
        ],
    );

    println!("\n  Higher-order polymorphic signature:");
    println!("    map_over: (T -> U) x List<T> -> List<U>");

    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let int_to_string = ParametricType::function(int_type.clone(), string_type.clone());
    let list_int = ParametricType::list(int_type.clone());
    let list_string = ParametricType::list(string_type.clone());

    match map_sig.unify_parametric(&[int_to_string, list_int, list_string]) {
        Ok(subst) => {
            println!("\n  Unifying map_over with ((Int->String), List<Int>, List<String>):");
            println!(
                "    Success! T = {}, U = {}",
                subst.get("T").unwrap(),
                subst.get("U").unwrap()
            );
        }
        Err(e) => println!("    Failed: {}", e),
    }
}

fn demonstrate_complex_unification() {
    // Unify multiple variables simultaneously
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let int_type = ParametricType::concrete("Int");

    // Tuple<T, U> with Tuple<Int, Int>
    let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);
    let tuple_int_int = ParametricType::tuple(vec![int_type.clone(), int_type.clone()]);

    match unify(&tuple_tu, &tuple_int_int) {
        Ok(subst) => {
            println!("  Unify Tuple<T, U> with Tuple<Int, Int>:");
            println!(
                "    Success! T = {}, U = {}",
                subst.get("T").unwrap(),
                subst.get("U").unwrap()
            );
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Nested unification
    let list_t = ParametricType::list(t.clone());
    let list_list_t = ParametricType::list(list_t.clone());
    let list_list_int = ParametricType::list(ParametricType::list(int_type.clone()));

    match unify(&list_list_t, &list_list_int) {
        Ok(subst) => {
            println!("\n  Unify List<List<T>> with List<List<Int>>:");
            println!("    Success! T = {}", subst.get("T").unwrap());
        }
        Err(e) => println!("    Failed: {}", e),
    }

    // Compose substitutions
    let mut subst1 = HashMap::new();
    subst1.insert("T".to_string(), ParametricType::variable("U"));

    let mut subst2 = HashMap::new();
    subst2.insert("U".to_string(), int_type.clone());

    let composed = compose_substitutions(&subst1, &subst2);
    println!("\n  Compose substitutions:");
    println!("    subst1: T -> U");
    println!("    subst2: U -> Int");
    println!("    composed: T -> {}", composed.get("T").unwrap());
}

fn demonstrate_generalization() {
    // Generalize a type with free variables
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);

    println!("  Original type: {}", tuple_tu);
    println!("  Free variables: {:?}", tuple_tu.free_variables());

    // Generalize with empty environment
    let generalized = generalize(&tuple_tu, &[]);
    println!("\n  Generalized type: {}", generalized);
    println!("  Free variables: {:?}", generalized.free_variables());

    // Instantiate to get fresh type variables
    let inst1 = instantiate(&tuple_tu);
    let inst2 = instantiate(&tuple_tu);

    println!("\n  Instance 1: {}", inst1);
    println!("  Free variables: {:?}", inst1.free_variables());

    println!("\n  Instance 2: {}", inst2);
    println!("  Free variables: {:?}", inst2.free_variables());

    println!("\n  (Note: Each instance has fresh, unique type variables)");

    // Apply substitution
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), ParametricType::concrete("Int"));
    subst.insert("U".to_string(), ParametricType::concrete("String"));

    let substituted = tuple_tu.substitute(&subst);
    println!(
        "\n  After substitution [T->Int, U->String]: {}",
        substituted
    );
}
