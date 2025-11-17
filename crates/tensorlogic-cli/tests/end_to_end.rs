//! End-to-end tests for real-world scenarios

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn get_cli_binary() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test executable name
    path.pop(); // Remove 'deps'

    let debug_path = path.join("tensorlogic");
    if debug_path.exists() {
        debug_path
    } else {
        path.pop(); // Remove 'debug'
        path.join("release").join("tensorlogic")
    }
}

fn run_cli(args: &[&str]) -> (String, String, i32) {
    let binary = get_cli_binary();
    let output = Command::new(&binary)
        .args(args)
        .output()
        .expect("Failed to execute CLI");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);

    (stdout, stderr, code)
}

#[test]
fn test_social_network_reasoning() {
    // Real-world scenario: Social network relationship inference
    let expression = "FORALL x IN Person. FORALL y IN Person. FORALL z IN Person. (knows(x, y) AND knows(y, z)) -> might_know(x, z)";

    let (stdout, stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Person:1000",
        "--strategy",
        "fuzzy_godel",
        "--output-format",
        "stats",
        "--analyze",
        "--validate",
    ]);

    if code != 0 {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    assert_eq!(code, 0, "Social network reasoning should compile");
    assert!(stdout.contains("Tensors:"), "Should show statistics");
}

#[test]
fn test_knowledge_base_query() {
    // Real-world scenario: Knowledge base querying
    let expression =
        "EXISTS x IN Entity. (is_person(x) AND has_birthplace(x) AND has_residence(x))";

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Entity:10000",
        "--domains",
        "Person:5000",
        "--domains",
        "City:500",
        "--strategy",
        "probabilistic",
        "--output-format",
        "json",
    ]);

    assert_eq!(code, 0, "Knowledge base query should compile");
    // Verify JSON output
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");
}

#[test]
fn test_recommendation_system() {
    // Real-world scenario: Recommendation system logic
    let expression = concat!(
        "FORALL u IN User. ",
        "FORALL i IN Item. FORALL j IN Item. ",
        "(liked(u, i) AND similar(i, j)) -> recommend(u, j)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "User:100000",
        "--domains",
        "Item:50000",
        "--strategy",
        "soft_differentiable",
        "--output-format",
        "stats",
    ]);

    assert_eq!(code, 0, "Recommendation system logic should compile");
    assert!(
        stdout.contains("FLOPs:"),
        "Should estimate computational cost"
    );
}

#[test]
fn test_access_control_policy() {
    // Real-world scenario: Access control policy
    let expression = concat!(
        "FORALL u IN User. FORALL r IN Resource. ",
        "(has_role(u, admin) OR (owns(u, r) AND NOT locked(r))) -> can_access(u, r)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "User:10000",
        "--domains",
        "Resource:50000",
        "--strategy",
        "hard_boolean",
        "--validate",
        "--output-format",
        "dot",
    ]);

    assert_eq!(code, 0, "Access control policy should compile");
    assert!(stdout.contains("digraph"), "Should generate DOT graph");
}

#[test]
fn test_temporal_reasoning() {
    // Real-world scenario: Temporal reasoning
    let expression = concat!(
        "FORALL e IN Event. ",
        "(happened(e) AND before(e, now)) -> in_past(e)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Event:1000",
        "--strategy",
        "fuzzy_lukasiewicz",
        "--analyze",
    ]);

    assert_eq!(code, 0, "Temporal reasoning should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_scientific_calculation() {
    // Real-world scenario: Scientific calculation with conditionals
    let expression = concat!(
        "IF temperature(x) > 100 THEN ",
        "state(x, gas) ",
        "ELSE IF temperature(x) > 0 THEN ",
        "state(x, liquid) ",
        "ELSE ",
        "state(x, solid)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--strategy",
        "soft_differentiable",
        "--output-format",
        "graph",
    ]);

    assert_eq!(code, 0, "Scientific calculation should compile");
    assert!(!stdout.is_empty());
}

#[test]
fn test_data_validation_rules() {
    // Real-world scenario: Data validation
    let expression = concat!(
        "FORALL x IN Record. ",
        "(age(x) >= 0 AND age(x) <= 150 AND ",
        "length(name(x)) > 0) -> valid(x)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Record:1000000",
        "--strategy",
        "hard_boolean",
        "--output-format",
        "stats",
        "--analyze",
    ]);

    assert_eq!(code, 0, "Data validation rules should compile");
    assert!(
        stdout.contains("Estimated Complexity"),
        "Should show complexity analysis"
    );
}

#[test]
fn test_graph_traversal() {
    // Real-world scenario: Graph traversal/reachability
    let expression = concat!(
        "FORALL x IN Node. FORALL y IN Node. ",
        "(edge(x, y) OR ",
        "(EXISTS z IN Node. edge(x, z) AND edge(z, y))) ",
        "-> reachable(x, y)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Node:1000",
        "--strategy",
        "fuzzy_product",
        "--validate",
        "--output-format",
        "json",
    ]);

    assert_eq!(code, 0, "Graph traversal should compile");
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");
}

#[test]
fn test_pipeline_compilation_and_conversion() {
    // Real-world scenario: Full pipeline - expression format conversions then compilation
    let temp_dir = std::env::temp_dir();
    let json_file = temp_dir.join("test_pipeline.json");
    let yaml_file = temp_dir.join("test_pipeline.yaml");
    let expr_file = temp_dir.join("test_pipeline.txt");
    let compiled_file = temp_dir.join("test_pipeline_compiled.json");

    // Clean up if exists
    let _ = fs::remove_file(&json_file);
    let _ = fs::remove_file(&yaml_file);
    let _ = fs::remove_file(&expr_file);
    let _ = fs::remove_file(&compiled_file);

    // Step 1: Convert expression to JSON (TLExpr format)
    let expression = "knows(x, y) AND likes(y, z)";
    let (_stdout, _stderr, code) = run_cli(&[
        "convert",
        expression,
        "--from",
        "expr",
        "--to",
        "json",
        "--output",
        json_file.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "Expression to JSON conversion should succeed");
    assert!(json_file.exists(), "JSON file should be created");

    // Step 2: Convert JSON to YAML
    let (_stdout, _stderr, code) = run_cli(&[
        "convert",
        json_file.to_str().unwrap(),
        "--from",
        "json",
        "--to",
        "yaml",
        "--output",
        yaml_file.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "JSON to YAML conversion should succeed");
    assert!(yaml_file.exists(), "YAML file should be created");

    // Step 3: Convert YAML back to expression
    let (_stdout, _stderr, code) = run_cli(&[
        "convert",
        yaml_file.to_str().unwrap(),
        "--from",
        "yaml",
        "--to",
        "expr",
        "--output",
        expr_file.to_str().unwrap(),
        "--pretty",
    ]);
    assert_eq!(code, 0, "YAML to expression conversion should succeed");
    assert!(expr_file.exists(), "Expression file should be created");

    // Step 4: Compile the expression to graph
    let (_stdout, _stderr, code) = run_cli(&[
        expression,
        "--output",
        compiled_file.to_str().unwrap(),
        "--output-format",
        "json",
        "--quiet",
    ]);
    assert_eq!(code, 0, "Compilation to graph JSON should succeed");
    assert!(
        compiled_file.exists(),
        "Compiled JSON file should be created"
    );

    // Verify content
    let expr_content = fs::read_to_string(&expr_file).unwrap();
    assert!(
        expr_content.contains("knows") && expr_content.contains("likes"),
        "Expression should contain predicates"
    );

    // Clean up
    let _ = fs::remove_file(&json_file);
    let _ = fs::remove_file(&yaml_file);
    let _ = fs::remove_file(&expr_file);
    let _ = fs::remove_file(&compiled_file);
}

#[test]
fn test_multi_strategy_comparison() {
    // Real-world scenario: Compare different compilation strategies
    let expression = "p AND q AND r";
    let strategies = vec![
        "soft_differentiable",
        "hard_boolean",
        "fuzzy_godel",
        "fuzzy_product",
        "fuzzy_lukasiewicz",
        "probabilistic",
    ];

    for strategy in strategies {
        let (stdout, _stderr, code) = run_cli(&[
            expression,
            "--strategy",
            strategy,
            "--output-format",
            "stats",
            "--quiet",
        ]);

        assert_eq!(code, 0, "Strategy {} should work", strategy);
        assert!(
            stdout.contains("Tensors:"),
            "Strategy {} should produce stats",
            strategy
        );
    }
}

#[test]
fn test_complex_nested_expression() {
    // Real-world scenario: Deeply nested logical expression
    let expression = concat!(
        "FORALL x IN A. ",
        "(EXISTS y IN B. ",
        "  (p(x, y) AND ",
        "   (FORALL z IN C. ",
        "     (q(y, z) AND r(x, z))))) ",
        "-> result(x)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "A:100",
        "--domains",
        "B:200",
        "--domains",
        "C:300",
        "--strategy",
        "soft_differentiable",
        "--analyze",
        "--output-format",
        "stats",
    ]);

    assert_eq!(code, 0, "Complex nested expression should compile");
    assert!(
        stdout.contains("depth") || stdout.contains("Depth"),
        "Should show graph depth"
    );
}

#[test]
fn test_batch_file_processing() {
    // Real-world scenario: Batch processing expressions from file
    let temp_dir = std::env::temp_dir();
    let batch_file = temp_dir.join("test_batch.txt");

    // Create batch file with multiple expressions
    let expressions = [
        "knows(x, y)",
        "likes(y, z)",
        "knows(x, y) AND likes(y, z)",
        "EXISTS x IN Person. knows(x, alice)",
        "FORALL x IN Person. mortal(x)",
    ];

    fs::write(&batch_file, expressions.join("\n")).unwrap();

    let (stdout, _stderr, code) = run_cli(&["batch", batch_file.to_str().unwrap()]);

    // Batch command should succeed if most expressions are valid
    assert!(code == 0 || code == 1, "Batch processing should run");
    assert!(
        stdout.contains("success") || stdout.contains("processed") || stdout.contains("Success"),
        "Should show processing results"
    );

    // Clean up
    let _ = fs::remove_file(&batch_file);
}

#[test]
fn test_visualization_workflow() {
    // Real-world scenario: Generate visualization
    let temp_dir = std::env::temp_dir();
    let dot_file = temp_dir.join("test_graph.dot");

    // Clean up if exists
    let _ = fs::remove_file(&dot_file);

    let expression = concat!(
        "FORALL x IN Node. FORALL y IN Node. ",
        "edge(x, y) -> connected(x, y)"
    );

    let (_stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "Node:50",
        "--output",
        dot_file.to_str().unwrap(),
        "--output-format",
        "dot",
        "--quiet",
    ]);

    assert_eq!(code, 0, "DOT generation should succeed");
    assert!(dot_file.exists(), "DOT file should be created");

    // Verify DOT content
    let dot_content = fs::read_to_string(&dot_file).unwrap();
    assert!(dot_content.contains("digraph"), "Should be valid DOT");
    assert!(dot_content.contains("->"), "Should contain edges");

    // Clean up
    let _ = fs::remove_file(&dot_file);
}

#[test]
fn test_error_handling_invalid_expression() {
    // Real-world scenario: Handle parsing errors gracefully
    let invalid_expressions = vec![
        "",          // Empty expression
        "   ",       // Whitespace only
        "EXISTS x.", // Missing body
        "FORALL y.", // Missing body
    ];

    for expr in invalid_expressions {
        let (_stdout, _stderr, code) = run_cli(&[expr, "--quiet"]);
        assert_ne!(code, 0, "Invalid expression '{}' should fail", expr);
    }
}

#[test]
fn test_performance_large_domain() {
    // Real-world scenario: Handle large domains
    let expression = concat!(
        "FORALL x IN LargeDomain. ",
        "EXISTS y IN LargeDomain. ",
        "related(x, y)"
    );

    let (stdout, _stderr, code) = run_cli(&[
        expression,
        "--domains",
        "LargeDomain:1000000",
        "--output-format",
        "stats",
        "--analyze",
        "--quiet",
    ]);

    assert_eq!(code, 0, "Large domain should compile");
    assert!(
        stdout.contains("FLOPs:") || stdout.contains("Memory:"),
        "Should provide performance estimates"
    );
}
