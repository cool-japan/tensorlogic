//! Example 19: Schema Auto-completion System
//!
//! This example demonstrates the intelligent auto-completion system
//! that provides context-aware suggestions for building schemas.

use tensorlogic_adapters::{AutoCompleter, DomainInfo, SymbolTable};

fn main() {
    println!("=== Schema Auto-completion System ===\n");

    // Create an auto-completer
    let mut autocompleter = AutoCompleter::new().with_max_suggestions(5);

    println!("Auto-completer initialized");
    let stats = autocompleter.stats();
    println!(
        "  - Pattern database loaded with {} patterns\n",
        stats.num_patterns
    );

    // 1. Domain name suggestions
    println!("=== 1. Domain Name Suggestions ===\n");

    let partial = "per";
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_domain_names(partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2}, estimated cardinality: {})",
            i + 1,
            suggestion.name,
            suggestion.confidence,
            suggestion.estimated_cardinality
        );
        if let Some(ref desc) = suggestion.description {
            println!("     Description: {}", desc);
        }
    }
    println!();

    let partial = "stud";
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_domain_names(partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2})",
            i + 1,
            suggestion.name,
            suggestion.confidence
        );
    }
    println!();

    // 2. Predicate suggestions based on domain context
    println!("=== 2. Predicate Suggestions ===\n");

    let domains = vec!["Person".to_string()];
    let partial = "know";
    println!("Context: domains = {:?}", domains);
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_predicates(&domains, partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {}({}) - confidence: {:.2}",
            i + 1,
            suggestion.name,
            suggestion.arg_domains.join(", "),
            suggestion.confidence
        );
    }
    println!();

    let domains = vec!["Student".to_string()];
    let partial = "enroll";
    println!("Context: domains = {:?}", domains);
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_predicates(&domains, partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {}({}) - confidence: {:.2}",
            i + 1,
            suggestion.name,
            suggestion.arg_domains.join(", "),
            suggestion.confidence
        );
    }
    println!();

    // 3. Variable name suggestions
    println!("=== 3. Variable Name Suggestions ===\n");

    let domain = "Person";
    let partial = "p";
    println!("Context: variable of type '{}'", domain);
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_variable_names(domain, partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2})",
            i + 1,
            suggestion.name,
            suggestion.confidence
        );
    }
    println!();

    let domain = "Course";
    let partial = "c";
    println!("Context: variable of type '{}'", domain);
    println!("User types: \"{}\"", partial);
    let suggestions = autocompleter.suggest_variable_names(domain, partial);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2})",
            i + 1,
            suggestion.name,
            suggestion.confidence
        );
    }
    println!();

    // 4. Predicate argument completion
    println!("=== 4. Predicate Argument Domain Suggestions ===\n");

    let predicate_name = "teaches";
    let existing_args = vec!["Teacher".to_string()];
    println!(
        "Building predicate: {}({}; ?)",
        predicate_name, existing_args[0]
    );
    println!("Suggesting domain for second argument...");

    let suggestions =
        autocompleter.suggest_domain_for_predicate_arg(predicate_name, &existing_args, 1);
    println!("Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2})",
            i + 1,
            suggestion.name,
            suggestion.confidence
        );
    }
    println!();

    // 5. Index existing schema for better suggestions
    println!("=== 5. Learning from Existing Schema ===\n");

    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Employee", 1000)).unwrap();
    table.add_domain(DomainInfo::new("Department", 50)).unwrap();
    table.add_domain(DomainInfo::new("Project", 200)).unwrap();

    println!("Indexing existing schema...");
    autocompleter.index_table(&table);

    let stats = autocompleter.stats();
    println!("Updated statistics:");
    println!("  - Indexed domains: {}", stats.num_indexed_domains);
    println!("  - Indexed predicates: {}", stats.num_indexed_predicates);
    println!("  - Total patterns: {}\n", stats.num_patterns);

    // 6. Demonstration of suggestion sources
    println!("=== 6. Suggestion Sources ===\n");

    println!("Suggestions can come from multiple sources:");
    let suggestions = autocompleter.suggest_domain_names("emp");
    for suggestion in suggestions.iter() {
        println!(
            "  - {} (source: {:?}, confidence: {:.2})",
            suggestion.name, suggestion.source, suggestion.confidence
        );
    }
    println!();

    // Summary
    println!("=== Summary ===");
    println!("✓ Domain name auto-completion with confidence scoring");
    println!("✓ Context-aware predicate suggestions based on domains");
    println!("✓ Variable name suggestions following conventions");
    println!("✓ Intelligent argument domain completion");
    println!("✓ Learning from existing schemas");
    println!("\nThe auto-completion system helps users build schemas faster");
    println!("with intelligent, context-aware suggestions based on common");
    println!("domain modeling patterns and existing schema knowledge!");
}
