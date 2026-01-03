//! Advanced SPARQL 1.1 Query Examples
//!
//! This example demonstrates comprehensive SPARQL 1.1 support including:
//! - All query types: SELECT, ASK, DESCRIBE, CONSTRUCT
//! - Advanced patterns: OPTIONAL, UNION
//! - Filter conditions: comparisons, BOUND, isIRI, regex
//! - Solution modifiers: DISTINCT, LIMIT, OFFSET, ORDER BY
//!
//! Run with: cargo run --example 09_sparql_advanced -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::SparqlCompiler;

fn main() -> Result<()> {
    println!("=== Advanced SPARQL 1.1 Examples ===\n");

    let mut compiler = SparqlCompiler::new();

    // Set up predicate mappings
    compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());
    compiler.add_predicate_mapping("http://example.org/name".to_string(), "name".to_string());
    compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());
    compiler.add_predicate_mapping("http://example.org/email".to_string(), "email".to_string());
    compiler.add_predicate_mapping("http://example.org/type".to_string(), "type".to_string());
    compiler.add_predicate_mapping("http://example.org/likes".to_string(), "likes".to_string());

    // ========================================
    // Example 1: Basic SELECT Query
    // ========================================
    println!("1️⃣  Basic SELECT Query");
    println!("────────────────────────");

    let basic_select = r#"
        SELECT ?person ?friend WHERE {
          ?person <http://example.org/knows> ?friend .
        }
    "#;

    let query = compiler.parse_query(basic_select)?;
    let expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", basic_select.trim());
    println!("Compiled to: {:?}", expr);
    println!();

    // ========================================
    // Example 2: SELECT with DISTINCT and LIMIT
    // ========================================
    println!("2️⃣  SELECT with DISTINCT and LIMIT");
    println!("────────────────────────────────");

    let distinct_query = r#"
        SELECT DISTINCT ?person WHERE {
          ?person <http://example.org/knows> ?other .
        } LIMIT 10
    "#;

    let query = compiler.parse_query(distinct_query)?;
    println!("Query: {}", distinct_query.trim());
    println!("LIMIT: {:?}", query.limit);
    println!("DISTINCT: {:?}", query.query_type);
    println!();

    // ========================================
    // Example 3: SELECT with FILTER
    // ========================================
    println!("3️⃣  SELECT with FILTER Conditions");
    println!("────────────────────────────────");

    let filter_query = r#"
        SELECT ?person ?age WHERE {
          ?person <http://example.org/age> ?age .
          FILTER(?age >= 18)
        }
    "#;

    let query = compiler.parse_query(filter_query)?;
    let expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", filter_query.trim());
    println!("Compiled with filter: {:?}", expr);
    println!();

    // ========================================
    // Example 4: ASK Query (Boolean Check)
    // ========================================
    println!("4️⃣  ASK Query (Boolean Existence Check)");
    println!("──────────────────────────────────────");

    let ask_query = r#"
        ASK WHERE {
          ?person <http://example.org/knows> ?friend .
          ?friend <http://example.org/age> ?age .
          FILTER(?age > 21)
        }
    "#;

    let query = compiler.parse_query(ask_query)?;
    let expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", ask_query.trim());
    println!("Type: ASK (returns boolean)");
    println!("Compiled to: {:?}", expr);
    println!();

    // ========================================
    // Example 5: DESCRIBE Query
    // ========================================
    println!("5️⃣  DESCRIBE Query (Resource Description)");
    println!("────────────────────────────────────────");

    let describe_query = r#"
        DESCRIBE ?person WHERE {
          ?person <http://example.org/type> <http://example.org/Person> .
          ?person <http://example.org/name> ?name .
        }
    "#;

    let query = compiler.parse_query(describe_query)?;

    println!("Query: {}", describe_query.trim());
    println!("Type: DESCRIBE (describes resources)");
    if let tensorlogic_oxirs_bridge::sparql::QueryType::Describe { resources } = &query.query_type {
        println!("Resources to describe: {:?}", resources);
    }
    println!();

    // ========================================
    // Example 6: CONSTRUCT Query
    // ========================================
    println!("6️⃣  CONSTRUCT Query (Graph Construction)");
    println!("───────────────────────────────────────");

    let construct_query = r#"
        CONSTRUCT {
          ?person <http://example.org/friend> ?friend
        }
        WHERE {
          ?person <http://example.org/knows> ?friend .
          ?friend <http://example.org/knows> ?person .
        }
    "#;

    let query = compiler.parse_query(construct_query)?;

    println!("Query: {}", construct_query.trim());
    println!("Type: CONSTRUCT (creates new triples)");
    if let tensorlogic_oxirs_bridge::sparql::QueryType::Construct { template } = &query.query_type {
        println!("Template patterns: {} triple(s)", template.len());
    }
    println!();

    // ========================================
    // Example 7: OPTIONAL Pattern
    // ========================================
    println!("7️⃣  OPTIONAL Pattern (Left-Outer Join)");
    println!("─────────────────────────────────────");

    let optional_query = r#"
        SELECT ?person ?name ?email WHERE {
          ?person <http://example.org/name> ?name .
          OPTIONAL { ?person <http://example.org/email> ?email }
        }
    "#;

    let query = compiler.parse_query(optional_query)?;
    let _expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", optional_query.trim());
    println!("Note: Email is optional, results include persons without email");
    println!("Pattern structure: {:?}", query.where_pattern);
    println!();

    // ========================================
    // Example 8: UNION Pattern
    // ========================================
    println!("8️⃣  UNION Pattern (Disjunction)");
    println!("──────────────────────────────");

    let union_query = r#"
        SELECT ?person ?relation ?other WHERE {
          { ?person <http://example.org/knows> ?other }
          UNION
          { ?person <http://example.org/likes> ?other }
        }
    "#;

    let query = compiler.parse_query(union_query)?;
    let expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", union_query.trim());
    println!("Note: Matches either 'knows' or 'likes' relationships");
    println!("Compiled with OR: {:?}", expr);
    println!();

    // ========================================
    // Example 9: Complex Query with Multiple Features
    // ========================================
    println!("9️⃣  Complex Query (Multiple Features Combined)");
    println!("──────────────────────────────────────────────");

    let complex_query = r#"
        SELECT DISTINCT ?person ?name ?age WHERE {
          ?person <http://example.org/name> ?name .
          ?person <http://example.org/age> ?age .
          OPTIONAL {
            ?person <http://example.org/email> ?email .
            FILTER(regex(?email, "@example.com"))
          }
          FILTER(?age >= 18)
          FILTER(?age <= 65)
        } ORDER BY ?age LIMIT 100 OFFSET 0
    "#;

    let query = compiler.parse_query(complex_query)?;
    let _expr = compiler.compile_to_tensorlogic(&query)?;

    println!("Query: {}", complex_query.trim());
    println!("\nFeatures used:");
    println!("  ✓ DISTINCT - Remove duplicates");
    println!("  ✓ OPTIONAL - Email is optional");
    println!("  ✓ FILTER - Age range and email pattern");
    println!("  ✓ regex() - Pattern matching");
    println!("  ✓ ORDER BY - Sort by age");
    println!("  ✓ LIMIT/OFFSET - Pagination");
    println!();

    // ========================================
    // Example 10: Advanced Filter Functions
    // ========================================
    println!("🔟 Advanced Filter Functions");
    println!("───────────────────────────");

    let filter_funcs = r#"
        SELECT ?x ?value WHERE {
          ?x <http://example.org/name> ?value .
          FILTER(BOUND(?value))
          FILTER(isLiteral(?value))
        }
    "#;

    let _query = compiler.parse_query(filter_funcs)?;

    println!("Query: {}", filter_funcs.trim());
    println!("\nFilter functions supported:");
    println!("  ✓ BOUND(?var) - Check if variable is bound");
    println!("  ✓ isIRI(?var) - Check if value is IRI");
    println!("  ✓ isLiteral(?var) - Check if value is literal");
    println!("  ✓ regex(?var, pattern) - Regular expression matching");
    println!();

    // ========================================
    // Summary
    // ========================================
    println!("═══════════════════════════════════════");
    println!("🎉 SPARQL 1.1 Feature Summary");
    println!("═══════════════════════════════════════");
    println!();
    println!("✅ Query Types:");
    println!("   • SELECT - Variable projection queries");
    println!("   • ASK - Boolean existence checks");
    println!("   • DESCRIBE - Resource descriptions");
    println!("   • CONSTRUCT - RDF graph construction");
    println!();
    println!("✅ Graph Patterns:");
    println!("   • Triple patterns with variables");
    println!("   • OPTIONAL - Left-outer join semantics");
    println!("   • UNION - Disjunction of patterns");
    println!("   • Nested patterns with braces");
    println!();
    println!("✅ Filters:");
    println!("   • Comparisons: >, <, >=, <=, =, !=");
    println!("   • BOUND(?var)");
    println!("   • isIRI(?var) / isURI(?var)");
    println!("   • isLiteral(?var)");
    println!("   • regex(?var, pattern)");
    println!();
    println!("✅ Solution Modifiers:");
    println!("   • DISTINCT - Remove duplicates");
    println!("   • LIMIT N - Limit results");
    println!("   • OFFSET N - Skip results");
    println!("   • ORDER BY ?var - Sort results");
    println!();
    println!("All features compile to TensorLogic expressions!");
    println!("═══════════════════════════════════════");

    Ok(())
}
