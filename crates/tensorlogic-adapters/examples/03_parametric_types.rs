//! Parametric types example.
//!
//! This example demonstrates how to define and use parametric (generic) types
//! for domains.

use tensorlogic_adapters::{ParametricType, TypeBound, TypeParameter};

fn main() {
    println!("=== Parametric Types Example ===\n");

    // Simple parametric types
    println!("Simple parametric types:");

    let list_int = ParametricType::list(TypeParameter::concrete("Int"));
    println!("  - {}", list_int);

    let option_str = ParametricType::option(TypeParameter::concrete("String"));
    println!("  - {}", option_str);

    let pair = ParametricType::pair(
        TypeParameter::concrete("Person"),
        TypeParameter::concrete("City"),
    );
    println!("  - {}", pair);

    let map = ParametricType::map(
        TypeParameter::concrete("String"),
        TypeParameter::concrete("Int"),
    );
    println!("  - {}\n", map);

    // Nested parametric types
    println!("Nested parametric types:");

    let list_option_person = ParametricType::list(TypeParameter::parametric(
        ParametricType::option(TypeParameter::concrete("Person")),
    ));
    println!("  - {}", list_option_person);

    let map_of_lists = ParametricType::map(
        TypeParameter::concrete("String"),
        TypeParameter::parametric(ParametricType::list(TypeParameter::concrete("Person"))),
    );
    println!("  - {}\n", map_of_lists);

    // Type validation
    println!("Type validation:");

    let valid = ParametricType::list(TypeParameter::concrete("Person"));
    match valid.validate() {
        Ok(_) => println!("  ✓ {} is valid", valid),
        Err(e) => println!("  ✗ Validation error: {}", e),
    }

    let invalid = ParametricType::new(
        "List",
        vec![TypeParameter::concrete("A"), TypeParameter::concrete("B")],
    );
    match invalid.validate() {
        Ok(_) => println!("  ✓ {} is valid", invalid),
        Err(e) => println!("  ✗ {} is invalid: {}", invalid, e),
    }
    println!();

    // Type substitution
    println!("Type substitution:");

    let generic_list = ParametricType::list(TypeParameter::concrete("T"));
    println!("  Original: {}", generic_list);

    let concrete_list = generic_list.substitute("T", &TypeParameter::concrete("Person"));
    println!("  After substituting T with Person: {}\n", concrete_list);

    // Type bounds
    println!("Type bounds:");

    let comparable_bound = TypeBound::comparable("T");
    println!("  - {}", comparable_bound);

    let numeric_bound = TypeBound::numeric("N");
    println!("  - {}", numeric_bound);

    let subtype_bound = TypeBound::subtype("T", "Agent");
    println!("  - {}", subtype_bound);

    let trait_bound = TypeBound::trait_bound("T", "Serializable");
    println!("  - {}", trait_bound);
    println!();

    // Complex type with bounds
    println!("Complex parametric type example:");
    println!("  Map<K: Comparable, V: List<T>>");
    println!("  Where:");
    println!("    - K must be Comparable");
    println!("    - V is a List of T");
    println!("    - T is any type");
}
