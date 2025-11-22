//! Schema Learning from Data Example
//!
//! This example demonstrates automatic schema inference from sample datasets,
//! including JSON and CSV data sources.

use tensorlogic_adapters::{DataSample, InferenceConfig, SchemaLearner};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Adapters: Schema Learning from Data ===\n");

    // Example 1: Learn schema from JSON user data
    example_json_learning()?;

    // Example 2: Learn schema from CSV product data
    example_csv_learning()?;

    // Example 3: Advanced inference with custom configuration
    example_advanced_inference()?;

    // Example 4: Analyze confidence scores
    example_confidence_analysis()?;

    Ok(())
}

fn example_json_learning() -> anyhow::Result<()> {
    println!("--- Example 1: JSON Schema Learning ---");

    let json_data = r#"[
        {
            "id": 1,
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 30,
            "active": true,
            "city": "New York"
        },
        {
            "id": 2,
            "name": "Bob Smith",
            "email": "bob@example.com",
            "age": 25,
            "active": true,
            "city": "Los Angeles"
        },
        {
            "id": 3,
            "name": "Charlie Brown",
            "email": "charlie@example.com",
            "age": 35,
            "active": false,
            "city": "Chicago"
        },
        {
            "id": 4,
            "name": "Diana Prince",
            "email": "diana@example.com",
            "age": 28,
            "active": true,
            "city": "New York"
        }
    ]"#;

    let sample = DataSample::from_json(json_data)?;
    println!("Sample size: {} records", sample.len());
    println!("Fields: {:?}", sample.field_names());

    let config = InferenceConfig::default();
    let mut learner = SchemaLearner::new(config);

    let schema = learner.learn_from_sample(&sample)?;
    let stats = learner.statistics();

    println!("\nLearning Results:");
    println!("  Domains inferred: {}", stats.domains_inferred);
    println!("  Predicates inferred: {}", stats.predicates_inferred);
    println!("  Constraints inferred: {}", stats.constraints_inferred);
    println!("  Samples analyzed: {}", stats.total_samples_analyzed);
    println!("  Inference time: {} ms", stats.inference_time_ms);

    println!("\nInferred Domains:");
    for domain in schema.domains.values() {
        println!("  - {} (cardinality: {})", domain.name, domain.cardinality);
    }

    println!("\nInferred Predicates:");
    for predicate in schema.predicates.values() {
        println!(
            "  - {} (arity: {}, domains: {:?})",
            predicate.name,
            predicate.arg_domains.len(),
            predicate.arg_domains
        );
    }

    println!();
    Ok(())
}

fn example_csv_learning() -> anyhow::Result<()> {
    println!("--- Example 2: CSV Schema Learning ---");

    let csv_data = "product_id,name,price,stock,category,available\n\
                    1,Laptop,999.99,15,Electronics,true\n\
                    2,Mouse,25.50,100,Electronics,true\n\
                    3,Desk,299.00,8,Furniture,true\n\
                    4,Chair,149.99,20,Furniture,true\n\
                    5,Monitor,399.00,12,Electronics,false\n\
                    6,Keyboard,79.99,50,Electronics,true";

    let sample = DataSample::from_csv(csv_data)?;
    println!("Sample size: {} records", sample.len());
    println!("Fields: {:?}", sample.field_names());

    let config = InferenceConfig {
        min_confidence: 0.8,
        infer_hierarchies: true,
        infer_constraints: true,
        infer_dependencies: true,
        cardinality_multiplier: 3.0, // Higher multiplier for product catalog
        max_nesting_depth: 5,
    };

    let mut learner = SchemaLearner::new(config);
    let schema = learner.learn_from_sample(&sample)?;
    let stats = learner.statistics();

    println!("\nLearning Results:");
    println!("  Domains inferred: {}", stats.domains_inferred);
    println!("  Predicates inferred: {}", stats.predicates_inferred);
    println!("  Inference time: {} ms", stats.inference_time_ms);

    println!("\nInferred Schema:");
    for domain in schema.domains.values() {
        println!(
            "  Domain: {} (est. size: {})",
            domain.name, domain.cardinality
        );
    }

    println!();
    Ok(())
}

fn example_advanced_inference() -> anyhow::Result<()> {
    println!("--- Example 3: Advanced Inference Configuration ---");

    let json_data = r#"[
        {"student_id": 101, "name": "Alice", "grade": 95, "class": "Math"},
        {"student_id": 102, "name": "Bob", "grade": 87, "class": "Math"},
        {"student_id": 103, "name": "Charlie", "grade": 92, "class": "Science"},
        {"student_id": 104, "name": "Diana", "grade": 88, "class": "Science"},
        {"student_id": 105, "name": "Eve", "grade": 91, "class": "Math"}
    ]"#;

    let sample = DataSample::from_json(json_data)?;

    // Conservative configuration for high-confidence inference
    let conservative_config = InferenceConfig {
        min_confidence: 0.9,
        infer_hierarchies: false,
        infer_constraints: true,
        infer_dependencies: true,
        cardinality_multiplier: 1.5,
        max_nesting_depth: 3,
    };

    let mut learner = SchemaLearner::new(conservative_config);
    let schema = learner.learn_from_sample(&sample)?;
    let stats = learner.statistics();

    println!("Conservative Inference Results:");
    println!("  Domains: {}", stats.domains_inferred);
    println!("  Predicates: {}", stats.predicates_inferred);
    println!("  Constraints: {}", stats.constraints_inferred);

    println!("\nDomain Details:");
    for domain in schema.domains.values() {
        println!(
            "  {} - estimated size: {} elements",
            domain.name, domain.cardinality
        );
    }

    println!();
    Ok(())
}

fn example_confidence_analysis() -> anyhow::Result<()> {
    println!("--- Example 4: Confidence Score Analysis ---");

    let json_data = r#"[
        {"id": 1, "value": 100},
        {"id": 2, "value": 200},
        {"id": 3, "value": 300}
    ]"#;

    let sample = DataSample::from_json(json_data)?;
    let config = InferenceConfig::default();
    let mut learner = SchemaLearner::new(config);

    learner.learn_from_sample(&sample)?;

    println!("Confidence Scores for Inferred Elements:");
    for (element, confidence) in learner.all_confidences() {
        println!(
            "  {}: {:.2}% (evidence: {} samples, reason: {})",
            element,
            confidence.score * 100.0,
            confidence.evidence_count,
            confidence.reasoning
        );
    }

    // Check specific confidence thresholds
    println!("\nHigh-Confidence Elements (>= 85%):");
    for (element, confidence) in learner.all_confidences() {
        if confidence.is_confident(0.85) {
            println!("  ✓ {}: {:.1}%", element, confidence.score * 100.0);
        }
    }

    println!();
    Ok(())
}
