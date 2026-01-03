//! Example: Structured Logging with Tracing
//!
//! This example demonstrates how to use structured logging with the `tracing`
//! ecosystem for production-grade observability.
//!
//! Run with: cargo run --example 16_structured_logging --features structured-logging

#[cfg(feature = "structured-logging")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use tensorlogic_train::structured_logging::{LogFormat, LogLevel, TracingLogger};

    println!("=== Structured Logging Examples ===\n");

    // Example 1: Development logging (pretty format, debug level)
    println!("1. Development Logging (pretty format):");
    println!("   Initializing logger...\n");

    let _logger = TracingLogger::builder()
        .with_format(LogFormat::Pretty)
        .with_level(LogLevel::Debug)
        .with_file_location(true)
        .build()?;

    // Use tracing macros for structured logging
    tracing::info!("Training started");
    tracing::debug!(model = "linear", features = 4, "Model initialized");

    // Simulate training loop with structured logging
    for epoch in 1..=3 {
        let loss = 1.0 / (epoch as f64 + 1.0);
        let accuracy = 0.6 + 0.1 * epoch as f64;

        tracing::info!(
            epoch = epoch,
            loss = loss,
            accuracy = accuracy,
            "Epoch completed"
        );

        // Log batch-level details
        for batch in 1..=2 {
            let batch_loss = loss * (1.0 + 0.1 * batch as f64);
            tracing::debug!(
                epoch = epoch,
                batch = batch,
                loss = batch_loss,
                "Batch processed"
            );
        }

        // Log gradient statistics
        let gradient_norm = 0.5 + 0.01 * epoch as f64;
        tracing::trace!(
            epoch = epoch,
            gradient_norm = gradient_norm,
            "Gradient statistics"
        );
    }

    // Log warnings for anomalies
    let large_gradient = 100.0;
    tracing::warn!(
        gradient_norm = large_gradient,
        threshold = 10.0,
        "Large gradient detected"
    );

    println!("\n2. Using Spans for Hierarchical Logging:");
    println!("   (Spans provide context for nested operations)\n");

    // Create a span for an operation
    let training_span = tracing::info_span!("training", model = "resnet50", dataset = "imagenet");
    let _enter = training_span.enter();

    tracing::info!("Starting forward pass");
    tracing::debug!(batch_size = 32, "Processing batch");
    tracing::info!("Forward pass completed");

    drop(_enter); // Exit the span

    println!("\n3. Structured Data Examples:");
    println!("   (All fields are structured and searchable)\n");

    // Log with various data types
    tracing::info!(
        epoch = 5,
        loss = 0.123,
        accuracy = 0.876,
        learning_rate = 0.001,
        batch_size = 32,
        samples_processed = 50000,
        "Training metrics"
    );

    // Log with nested context
    tracing::info!(
        optimizer = "adam",
        "beta1" = 0.9,
        "beta2" = 0.999,
        epsilon = 1e-8,
        "Optimizer configuration"
    );

    println!("\n4. Error Logging:");
    println!("   (Errors with full context)\n");

    // Simulate an error
    let error_msg = "Invalid batch size";
    tracing::error!(
        error = error_msg,
        batch_size = 0,
        expected = ">0",
        "Configuration error"
    );

    println!("\n5. Custom Macros for Training:");
    println!("   (Use custom macros for common patterns)\n");

    // Note: The custom macros are defined in the structured_logging module
    // They provide convenient wrappers for common training scenarios

    println!("\n=== JSON Format Example ===");
    println!(
        "To use JSON format (for production/log aggregation), initialize with:\n\
         TracingLogger::builder()\n\
         .with_format(LogFormat::Json)\n\
         .with_level(LogLevel::Info)\n\
         .build()?;\n"
    );

    println!("JSON output is machine-parseable and can be sent to:");
    println!("  - Elasticsearch/Kibana");
    println!("  - Datadog, New Relic, etc.");
    println!("  - CloudWatch Logs");
    println!("  - Any log aggregation system\n");

    println!("=== Environment Filter Example ===");
    println!(
        "Use env filter for fine-grained control:\n\
         TracingLogger::builder()\n\
         .with_env_filter(\"tensorlogic=debug,scirs2=info\")\n\
         .build()?;\n"
    );

    println!("Or set via environment variable:");
    println!("  RUST_LOG=tensorlogic=debug cargo run --example 16_structured_logging\n");

    println!("=== Best Practices ===");
    println!("1. Use appropriate log levels:");
    println!("   - ERROR: Application errors that need attention");
    println!("   - WARN:  Anomalies or potential issues");
    println!("   - INFO:  High-level progress and metrics (default for production)");
    println!("   - DEBUG: Detailed diagnostic information");
    println!("   - TRACE: Very detailed (gradient norms, etc.)");
    println!();
    println!("2. Include relevant context fields:");
    println!("   - epoch, batch, step numbers");
    println!("   - metric values (loss, accuracy, etc.)");
    println!("   - configuration parameters");
    println!("   - identifiers (model_id, run_id, etc.)");
    println!();
    println!("3. Use spans for hierarchical operations:");
    println!("   - Training loop");
    println!("   - Validation phase");
    println!("   - Model saving/loading");
    println!();
    println!("4. For production:");
    println!("   - Use JSON format for machine parsing");
    println!("   - Set level to INFO or WARN");
    println!("   - Disable file locations (reduces overhead)");
    println!("   - Send logs to centralized logging service");

    Ok(())
}

#[cfg(not(feature = "structured-logging"))]
fn main() {
    eprintln!("This example requires the 'structured-logging' feature.");
    eprintln!("Run with: cargo run --example 16_structured_logging --features structured-logging");
    std::process::exit(1);
}
