//! Example 26: Schema Merging Strategies
//!
//! This example demonstrates advanced schema merging capabilities with multiple
//! conflict resolution strategies.
//! Features:
//! - Multiple merge strategies (KeepFirst, KeepSecond, FailOnConflict, Union, Intersection)
//! - Conflict detection and resolution
//! - Detailed merge reports
//! - Compatible predicate signature detection
//! - Domain cardinality-based resolution

use tensorlogic_adapters::{DomainInfo, MergeStrategy, PredicateInfo, SchemaMerger, SymbolTable};

fn main() {
    println!("=== Example 26: Schema Merging Strategies ===\n");

    // Scenario 1: Merge without conflicts
    println!("✨ Scenario 1: Merge Without Conflicts");
    println!("─────────────────────────────────────");
    scenario_no_conflicts();
    println!();

    // Scenario 2: KeepFirst strategy
    println!("🥇 Scenario 2: KeepFirst Strategy");
    println!("──────────────────────────────────");
    scenario_keep_first();
    println!();

    // Scenario 3: KeepSecond strategy
    println!("🥈 Scenario 3: KeepSecond Strategy");
    println!("───────────────────────────────────");
    scenario_keep_second();
    println!();

    // Scenario 4: FailOnConflict strategy
    println!("❌ Scenario 4: FailOnConflict Strategy");
    println!("───────────────────────────────────────");
    scenario_fail_on_conflict();
    println!();

    // Scenario 5: Union strategy
    println!("🔗 Scenario 5: Union Strategy");
    println!("─────────────────────────────");
    scenario_union();
    println!();

    // Scenario 6: Intersection strategy
    println!("⚡ Scenario 6: Intersection Strategy");
    println!("────────────────────────────────────");
    scenario_intersection();
    println!();

    // Scenario 7: Complex merge scenario
    println!("🌟 Scenario 7: Complex Merge Scenario");
    println!("──────────────────────────────────────");
    scenario_complex_merge();
    println!();

    println!("✅ All schema merging scenarios completed!");
}

fn scenario_no_conflicts() {
    // Create two non-overlapping schemas
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100)).unwrap();
    base.add_predicate(PredicateInfo::new(
        "knows",
        vec!["Person".to_string(), "Person".to_string()],
    ))
    .unwrap();

    let mut incoming = SymbolTable::new();
    incoming.add_domain(DomainInfo::new("Person", 100)).unwrap(); // Add Person domain for works_at predicate
    incoming
        .add_domain(DomainInfo::new("Organization", 50))
        .unwrap();
    incoming
        .add_predicate(PredicateInfo::new(
            "works_at",
            vec!["Person".to_string(), "Organization".to_string()],
        ))
        .unwrap();

    println!("Base schema:");
    println!("  - Domains: Person");
    println!("  - Predicates: knows");
    println!("\nIncoming schema:");
    println!("  - Domains: Organization");
    println!("  - Predicates: works_at");
    println!();

    // Merge with KeepFirst strategy
    let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merged schema:");
    println!("  - Total domains: {}", result.merged.domains.len());
    println!("  - Total predicates: {}", result.merged.predicates.len());
    println!();

    println!("Merge report:");
    println!("  - Base domains: {}", result.report.base_domains.len());
    println!(
        "  - Incoming domains: {}",
        result.report.incoming_domains.len()
    );
    println!("  - Conflicts: {}", result.report.conflict_count());
    println!("✓ No conflicts detected (disjoint schemas)");
}

fn scenario_keep_first() {
    // Create schemas with overlapping domains
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100).with_description("Base version"))
        .unwrap();
    base.add_domain(DomainInfo::new("Organization", 30))
        .unwrap();

    let mut incoming = SymbolTable::new();
    incoming
        .add_domain(DomainInfo::new("Person", 200).with_description("Incoming version"))
        .unwrap();
    incoming.add_domain(DomainInfo::new("Event", 50)).unwrap();

    println!("Base schema:");
    println!("  - Person (cardinality: 100, description: 'Base version')");
    println!("  - Organization (cardinality: 30)");
    println!("\nIncoming schema:");
    println!("  - Person (cardinality: 200, description: 'Incoming version')");
    println!("  - Event (cardinality: 50)");
    println!();

    // Merge with KeepFirst strategy
    let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merge strategy: KeepFirst");
    println!("Result:");

    // Check that base version of Person is kept
    let person = result.merged.get_domain("Person").unwrap();
    println!("  - Person kept from base:");
    println!("    • Cardinality: {} (from base)", person.cardinality);
    println!(
        "    • Description: '{}' (from base)",
        person.description.as_deref().unwrap_or("none")
    );

    println!("  - Total domains: {}", result.merged.domains.len());
    println!(
        "  - Conflicts resolved: {}",
        result.report.conflicting_domains.len()
    );
    println!("\n✓ KeepFirst strategy preserved base schema on conflicts");
}

fn scenario_keep_second() {
    // Create schemas with overlapping predicates
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100)).unwrap();
    base.add_predicate(
        PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Base version"),
    )
    .unwrap();

    let mut incoming = SymbolTable::new();
    incoming.add_domain(DomainInfo::new("Person", 100)).unwrap();
    incoming
        .add_predicate(
            PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
                .with_description("Incoming version with improvements"),
        )
        .unwrap();

    println!("Base schema:");
    println!("  - Predicate 'knows' (description: 'Base version')");
    println!("\nIncoming schema:");
    println!("  - Predicate 'knows' (description: 'Incoming version with improvements')");
    println!();

    // Merge with KeepSecond strategy
    let merger = SchemaMerger::new(MergeStrategy::KeepSecond);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merge strategy: KeepSecond");
    println!("Result:");

    // Check that incoming version is kept
    let knows = result.merged.get_predicate("knows").unwrap();
    println!("  - Predicate 'knows' kept from incoming:");
    println!(
        "    • Description: '{}'",
        knows.description.as_deref().unwrap_or("none")
    );

    println!(
        "  - Conflicts resolved: {}",
        result.report.conflicting_predicates.len()
    );
    println!("\n✓ KeepSecond strategy preferred incoming schema on conflicts");
}

fn scenario_fail_on_conflict() {
    // Create schemas with conflicts
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100)).unwrap();

    let mut incoming = SymbolTable::new();
    incoming.add_domain(DomainInfo::new("Person", 200)).unwrap();

    println!("Base schema:");
    println!("  - Person (cardinality: 100)");
    println!("\nIncoming schema:");
    println!("  - Person (cardinality: 200)");
    println!();

    // Merge with FailOnConflict strategy
    let merger = SchemaMerger::new(MergeStrategy::FailOnConflict);
    let result = merger.merge(&base, &incoming);

    println!("Merge strategy: FailOnConflict");

    match result {
        Ok(_) => println!("❌ Unexpected: merge should have failed"),
        Err(e) => {
            println!("✓ Merge failed as expected:");
            println!("  Error: {}", e);
        }
    }
}

fn scenario_union() {
    // Create schemas with some overlap
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100)).unwrap();
    base.add_domain(DomainInfo::new("Organization", 50))
        .unwrap();
    base.add_predicate(PredicateInfo::new(
        "knows",
        vec!["Person".to_string(), "Person".to_string()],
    ))
    .unwrap();

    let mut incoming = SymbolTable::new();
    incoming.add_domain(DomainInfo::new("Person", 100)).unwrap(); // Same cardinality - compatible
    incoming.add_domain(DomainInfo::new("Event", 75)).unwrap();
    incoming
        .add_predicate(PredicateInfo::new(
            "attends",
            vec!["Person".to_string(), "Event".to_string()],
        ))
        .unwrap();

    println!("Base schema:");
    println!("  - Domains: Person (100), Organization (50)");
    println!("  - Predicates: knows");
    println!("\nIncoming schema:");
    println!("  - Domains: Person (100), Event (75)");
    println!("  - Predicates: attends");
    println!();

    // Merge with Union strategy
    let merger = SchemaMerger::new(MergeStrategy::Union);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merge strategy: Union");
    println!("Result:");
    println!(
        "  - Total domains: {} (Person, Organization, Event)",
        result.merged.domains.len()
    );
    println!(
        "  - Total predicates: {} (knows, attends)",
        result.merged.predicates.len()
    );
    println!("  - Conflicts: {}", result.report.conflict_count());
    println!("\n✓ Union strategy combined all compatible items");
}

fn scenario_intersection() {
    // Create schemas with overlap
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Person", 100)).unwrap();
    base.add_domain(DomainInfo::new("Organization", 50))
        .unwrap();
    base.add_predicate(PredicateInfo::new(
        "knows",
        vec!["Person".to_string(), "Person".to_string()],
    ))
    .unwrap();

    let mut incoming = SymbolTable::new();
    incoming.add_domain(DomainInfo::new("Person", 100)).unwrap();
    incoming.add_domain(DomainInfo::new("Event", 75)).unwrap();
    incoming
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    println!("Base schema:");
    println!("  - Domains: Person (100), Organization (50)");
    println!("  - Predicates: knows");
    println!("\nIncoming schema:");
    println!("  - Domains: Person (100), Event (75)");
    println!("  - Predicates: knows");
    println!();

    // Merge with Intersection strategy
    let merger = SchemaMerger::new(MergeStrategy::Intersection);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merge strategy: Intersection");
    println!("Result:");
    println!(
        "  - Total domains: {} (only Person appears in both)",
        result.merged.domains.len()
    );
    println!(
        "  - Total predicates: {} (only knows appears in both)",
        result.merged.predicates.len()
    );
    println!(
        "  - Items from base only: {}",
        result.report.base_domains.len() + result.report.base_predicates.len()
    );
    println!(
        "  - Items from incoming only: {}",
        result.report.incoming_domains.len() + result.report.incoming_predicates.len()
    );
    println!("\n✓ Intersection strategy kept only common items");
}

fn scenario_complex_merge() {
    // Create a complex real-world scenario
    let mut base = SymbolTable::new();

    // Base schema: Academic system
    base.add_domain(DomainInfo::new("Person", 1000).with_description("Base academic system"))
        .unwrap();
    base.add_domain(DomainInfo::new("Course", 200)).unwrap();
    base.add_domain(DomainInfo::new("Department", 20)).unwrap();

    base.add_predicate(PredicateInfo::new(
        "teaches",
        vec!["Person".to_string(), "Course".to_string()],
    ))
    .unwrap();
    base.add_predicate(PredicateInfo::new(
        "enrolled",
        vec!["Person".to_string(), "Course".to_string()],
    ))
    .unwrap();
    base.add_predicate(PredicateInfo::new(
        "member_of",
        vec!["Person".to_string(), "Department".to_string()],
    ))
    .unwrap();

    base.bind_variable("professor", "Person").unwrap();
    base.bind_variable("student", "Person").unwrap();

    let mut incoming = SymbolTable::new();

    // Incoming schema: Extended academic system with research
    incoming
        .add_domain(DomainInfo::new("Person", 1500).with_description("Extended with researchers"))
        .unwrap();
    incoming.add_domain(DomainInfo::new("Course", 200)).unwrap(); // Same
    incoming
        .add_domain(DomainInfo::new("ResearchProject", 100))
        .unwrap();
    incoming
        .add_domain(DomainInfo::new("Publication", 500))
        .unwrap();

    incoming
        .add_predicate(PredicateInfo::new(
            "teaches",
            vec!["Person".to_string(), "Course".to_string()],
        ))
        .unwrap(); // Same
    incoming
        .add_predicate(PredicateInfo::new(
            "works_on",
            vec!["Person".to_string(), "ResearchProject".to_string()],
        ))
        .unwrap();
    incoming
        .add_predicate(PredicateInfo::new(
            "authored",
            vec!["Person".to_string(), "Publication".to_string()],
        ))
        .unwrap();

    incoming.bind_variable("researcher", "Person").unwrap();
    incoming.bind_variable("student", "Person").unwrap(); // Same

    println!("Base schema (Academic System):");
    println!("  - Domains: Person (1000), Course (200), Department (20)");
    println!("  - Predicates: teaches, enrolled, member_of");
    println!("  - Variables: professor, student");
    println!();

    println!("Incoming schema (Research Extension):");
    println!("  - Domains: Person (1500), Course (200), ResearchProject (100), Publication (500)");
    println!("  - Predicates: teaches, works_on, authored");
    println!("  - Variables: researcher, student");
    println!();

    // Merge with Union strategy (prefer larger cardinality)
    let merger = SchemaMerger::new(MergeStrategy::Union);
    let result = merger.merge(&base, &incoming).unwrap();

    println!("Merge strategy: Union");
    println!();

    println!("📊 Merge Results:");
    println!("─────────────────");
    println!("Final schema:");
    println!("  - Total domains: {}", result.merged.domains.len());
    println!(
        "    • Person (cardinality: {})",
        result.merged.get_domain("Person").unwrap().cardinality
    );
    println!(
        "    • Course (cardinality: {})",
        result.merged.get_domain("Course").unwrap().cardinality
    );
    println!("    • Department, ResearchProject, Publication");
    println!();

    println!("  - Total predicates: {}", result.merged.predicates.len());
    println!("    • teaches (from both)");
    println!("    • enrolled (from base)");
    println!("    • member_of (from base)");
    println!("    • works_on (from incoming)");
    println!("    • authored (from incoming)");
    println!();

    println!("  - Total variables: {}", result.merged.variables.len());
    println!("    • professor, student, researcher");
    println!();

    println!("📋 Merge Report:");
    println!("────────────────");
    println!("  - Base-only domains: {:?}", result.report.base_domains);
    println!(
        "  - Incoming-only domains: {:?}",
        result.report.incoming_domains
    );
    println!(
        "  - Domain conflicts: {}",
        result.report.conflicting_domains.len()
    );
    if !result.report.conflicting_domains.is_empty() {
        for conflict in &result.report.conflicting_domains {
            println!(
                "    • {} (base: {}, incoming: {})",
                conflict.name, conflict.base.cardinality, conflict.incoming.cardinality
            );
        }
    }
    println!();

    println!(
        "  - Base-only predicates: {:?}",
        result.report.base_predicates
    );
    println!(
        "  - Incoming-only predicates: {:?}",
        result.report.incoming_predicates
    );
    println!(
        "  - Predicate conflicts: {}",
        result.report.conflicting_predicates.len()
    );
    println!();

    println!("  - Merged variables: {:?}", result.report.merged_variables);
    println!(
        "  - Variable conflicts: {}",
        result.report.conflicting_variables.len()
    );
    println!();

    println!("  - Total items merged: {}", result.report.merged_count());
    println!("  - Total conflicts: {}", result.report.conflict_count());

    println!("\n✓ Complex merge completed successfully!");
    println!("  Combined academic and research schemas into unified system");
}
