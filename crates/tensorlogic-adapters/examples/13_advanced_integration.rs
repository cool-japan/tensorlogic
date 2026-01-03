//! Comprehensive integration example demonstrating all advanced features.
//!
//! This example shows how to use:
//! - Incremental validation for efficient schema updates
//! - Query planner for optimized predicate lookups
//! - Schema evolution for version management
//!
//! Together, these features enable production-grade schema management with
//! performance, flexibility, and safety.

use anyhow::Result;
use tensorlogic_adapters::{
    ChangeTracker, DomainInfo, EvolutionAnalyzer, IncrementalValidator, PredicateInfo,
    PredicatePattern, PredicateQuery, QueryPlanner, SchemaBuilder, SymbolTable, VersionBump,
};

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Advanced Integration Example: TensorLogic Adapters       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Part 1: Build initial schema with incremental validation
    println!("📋 Part 1: Building Schema with Incremental Validation");
    println!("─────────────────────────────────────────────────────────────");

    let mut table = SymbolTable::new();
    let mut tracker = ChangeTracker::new();

    // Build schema incrementally
    println!("  → Adding initial domains...");
    table.add_domain(DomainInfo::new("Person", 1000))?;
    tracker.record_domain_addition("Person");

    table.add_domain(DomainInfo::new("Location", 500))?;
    tracker.record_domain_addition("Location");

    table.add_domain(DomainInfo::new("Organization", 200))?;
    tracker.record_domain_addition("Organization");

    // Validate incrementally
    let report1 = {
        let mut validator = IncrementalValidator::new(&table, &tracker);
        validator.validate_incremental()?
    };

    println!(
        "  ✓ Initial validation: {} components validated, {} cached",
        report1.components_validated, report1.components_cached
    );
    println!("  ✓ Duration: {:?}\n", report1.duration);

    // Add predicates in a batch
    println!("  → Adding predicates in batch...");
    let _batch_id = tracker.begin_batch();

    table.add_predicate(PredicateInfo::new(
        "worksAt",
        vec!["Person".to_string(), "Organization".to_string()],
    ))?;
    tracker.record_predicate_addition("worksAt");

    table.add_predicate(PredicateInfo::new(
        "locatedIn",
        vec!["Organization".to_string(), "Location".to_string()],
    ))?;
    tracker.record_predicate_addition("locatedIn");

    table.add_predicate(PredicateInfo::new(
        "livesIn",
        vec!["Person".to_string(), "Location".to_string()],
    ))?;
    tracker.record_predicate_addition("livesIn");

    tracker.end_batch();

    // Incremental validation with cache
    let report2 = {
        let mut validator = IncrementalValidator::new(&table, &tracker);
        validator.validate_incremental()?
    };

    println!(
        "  ✓ Batch validation: {} components validated, {} cached",
        report2.components_validated, report2.components_cached
    );
    println!(
        "  ✓ Cache hit rate: {:.1}%",
        report2.cache_hit_rate() * 100.0
    );
    println!("  ✓ Duration: {:?}\n", report2.duration);

    // Part 2: Query planning for efficient lookups
    println!("📊 Part 2: Query Planning & Optimization");
    println!("─────────────────────────────────────────────────────────────");

    let mut planner = QueryPlanner::new(&table);

    // Query 1: Find all binary predicates
    println!("  → Query 1: All binary predicates");
    let query1 = PredicateQuery::by_arity(2);
    let results1 = planner.execute(&query1)?;
    println!("    Found {} binary predicates:", results1.len());
    for (name, _) in &results1 {
        println!("      • {}", name);
    }

    // Query 2: Find predicates involving Location
    println!("\n  → Query 2: Predicates involving Location domain");
    let query2 = PredicateQuery::by_domain("Location");
    let results2 = planner.execute(&query2)?;
    println!("    Found {} predicates:", results2.len());
    for (name, pred) in &results2 {
        println!("      • {}: {:?}", name, pred.arg_domains);
    }

    // Query 3: Complex query with pattern matching
    println!("\n  → Query 3: Complex pattern-based query");
    let pattern = PredicatePattern::new()
        .with_name_pattern("*At")
        .with_arity_range(2, 2)
        .with_required_domain("Person");

    let query3 = PredicateQuery::by_pattern(pattern);
    let results3 = planner.execute(&query3)?;
    println!("    Found {} matching predicates:", results3.len());
    for (name, _) in &results3 {
        println!("      • {}", name);
    }

    // Query statistics
    println!("\n  → Query Statistics:");
    let stats = planner.statistics();
    let top_queries = stats.top_queries(5);
    for (query_type, count) in &top_queries {
        println!("    • {}: {} executions", query_type, count);
    }
    println!("    • Cache size: {} plans\n", planner.cache_size());

    // Part 3: Schema evolution and version management
    println!("🔄 Part 3: Schema Evolution & Version Management");
    println!("─────────────────────────────────────────────────────────────");

    // Create version 2 of the schema with changes
    let table_v2 = SchemaBuilder::new()
        .domain("Person", 1000)
        .domain("Location", 500)
        .domain("Organization", 200)
        .domain("Department", 100) // NEW: Added department
        .predicate("worksAt", vec!["Person", "Organization"])
        .predicate("locatedIn", vec!["Organization", "Location"])
        .predicate("livesIn", vec!["Person", "Location"])
        .predicate("manages", vec!["Person", "Department"]) // NEW: Management relation
        .predicate("partOf", vec!["Department", "Organization"]) // NEW: Department structure
        .build()?;

    // Analyze evolution from v1 to v2
    println!("  → Analyzing changes from v1 to v2...");
    let analyzer = EvolutionAnalyzer::new(&table, &table_v2);
    let evo_report = analyzer.analyze()?;

    println!("\n  ✓ Evolution Analysis:");
    println!(
        "    • Breaking changes: {}",
        evo_report.breaking_changes.len()
    );
    println!(
        "    • Compatible changes: {}",
        evo_report.backward_compatible_changes.len()
    );
    println!(
        "    • Suggested version: {}",
        evo_report.suggested_version_bump()
    );
    println!(
        "    • Is backward compatible: {}",
        evo_report.is_backward_compatible()
    );

    if !evo_report.backward_compatible_changes.is_empty() {
        println!("\n  → Backward Compatible Changes:");
        for change in &evo_report.backward_compatible_changes {
            println!("    ✓ {}", change);
        }
    }

    // Migration plan
    println!("\n  → Migration Plan:");
    println!("    • Steps: {}", evo_report.migration_plan.steps.len());
    println!(
        "    • Complexity: {}",
        evo_report.migration_plan.estimated_complexity
    );
    println!(
        "    • Automatic: {}",
        evo_report.migration_plan.is_automatic()
    );

    if !evo_report.migration_plan.steps.is_empty() {
        println!("\n  → Migration Steps:");
        for (i, step) in evo_report.migration_plan.steps.iter().enumerate() {
            println!("    {}. {}", i + 1, step.description());
        }
    }

    // Now create a breaking change scenario
    println!("\n  → Creating v3 with breaking changes...");
    let table_v3 = SchemaBuilder::new()
        .domain("Person", 500) // BREAKING: Reduced cardinality
        .domain("Location", 500)
        // BREAKING: Removed Organization domain
        .predicate("livesIn", vec!["Person", "Location"])
        .build()?;

    let analyzer_v3 = EvolutionAnalyzer::new(&table_v2, &table_v3);
    let evo_report_v3 = analyzer_v3.analyze()?;

    println!("\n  ✓ v2 → v3 Evolution Analysis:");
    println!(
        "    • Breaking changes: {} ⚠️",
        evo_report_v3.breaking_changes.len()
    );
    println!(
        "    • Suggested version: {} (MAJOR)",
        evo_report_v3.suggested_version_bump()
    );

    if !evo_report_v3.breaking_changes.is_empty() {
        println!("\n  ⚠️  Breaking Changes Detected:");
        for change in &evo_report_v3.breaking_changes {
            use tensorlogic_adapters::ChangeImpact;
            println!(
                "    • [{}] {}",
                match change.impact {
                    ChangeImpact::Critical => "CRITICAL",
                    ChangeImpact::Major => "MAJOR",
                    ChangeImpact::Moderate => "MODERATE",
                    ChangeImpact::Minor => "MINOR",
                    ChangeImpact::None => "NONE",
                },
                change.description
            );

            if let Some(hint) = &change.migration_hint {
                println!("      → Migration: {}", hint);
            }

            if !change.affected_components.is_empty() {
                println!("      → Affects: {}", change.affected_components.join(", "));
            }
        }
    }

    // Part 4: Integration workflow
    println!("\n🔗 Part 4: Integrated Development Workflow");
    println!("─────────────────────────────────────────────────────────────");

    println!("  → Simulating iterative development...\n");

    // Start with v2 schema
    let mut dev_schema = table_v2.clone();
    let mut dev_tracker = ChangeTracker::new();

    // Iteration 1: Add a new predicate
    println!("  Iteration 1: Adding 'reportsTo' predicate");
    dev_schema.add_predicate(PredicateInfo::new(
        "reportsTo",
        vec!["Person".to_string(), "Person".to_string()],
    ))?;
    dev_tracker.record_predicate_addition("reportsTo");

    let iter1_report = {
        let mut dev_validator = IncrementalValidator::new(&dev_schema, &dev_tracker);
        dev_validator.validate_incremental()?
    };

    println!(
        "    ✓ Validated in {:?} ({} components, {:.1}% cached)",
        iter1_report.duration,
        iter1_report.components_validated,
        iter1_report.cache_hit_rate() * 100.0
    );

    // Iteration 2: Query the new predicate
    println!("\n  Iteration 2: Querying new predicate structure");
    let mut dev_planner = QueryPlanner::new(&dev_schema);
    let query_new = PredicateQuery::by_name("reportsTo");
    let new_pred = dev_planner.execute(&query_new)?;
    println!(
        "    ✓ Found: {:?}",
        new_pred.first().map(|(n, p)| (n, &p.arg_domains))
    );

    // Iteration 3: Check compatibility with production
    println!("\n  Iteration 3: Checking compatibility with production (v2)");
    let compat_analyzer = EvolutionAnalyzer::new(&table_v2, &dev_schema);
    let compat_report = compat_analyzer.analyze()?;

    match compat_report.suggested_version_bump() {
        VersionBump::Patch => println!("    ✓ Safe to deploy as PATCH update"),
        VersionBump::Minor => println!("    ✓ Safe to deploy as MINOR update"),
        VersionBump::Major => println!("    ⚠️  Requires MAJOR version bump"),
        VersionBump::None => println!("    ✓ No version change needed"),
    }

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Summary: Advanced Features Demonstration                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("✓ Incremental Validation:");
    println!(
        "  • Validated {} → {} components across iterations",
        report1.components_validated, iter1_report.components_validated
    );
    println!(
        "  • Cache hit rate: up to {:.1}%",
        iter1_report.cache_hit_rate() * 100.0
    );
    println!(
        "  • Performance: {:?} → {:?}",
        report1.duration, iter1_report.duration
    );

    println!("\n✓ Query Planning:");
    println!(
        "  • Executed {} different query types",
        stats.top_queries(10).len()
    );
    println!("  • Cached {} execution plans", planner.cache_size());
    println!("  • Supports: name, arity, signature, domain, pattern queries");

    println!("\n✓ Schema Evolution:");
    println!("  • Analyzed v1 → v2 → v3 transitions");
    println!(
        "  • Detected {} breaking changes in v3",
        evo_report_v3.breaking_changes.len()
    );
    println!("  • Generated migration plans automatically");
    println!("  • Provides semantic versioning guidance");

    println!("\n🎉 All advanced features working seamlessly together!\n");

    Ok(())
}
