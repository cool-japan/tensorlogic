//! Example 14: Advanced Type System Features
//!
//! This example demonstrates the four advanced type system modules:
//! - Refinement Types: Value constraints beyond simple types
//! - Dependent Types: Types that depend on values
//! - Linear Types: Resource tracking with usage guarantees
//! - Effect System: Tracking computational side effects
//!
//! Run with: cargo run --example 14_advanced_type_system

use tensorlogic_adapters::{
    dependent_patterns,
    infer_effects,
    // Dependent Types
    DependentType,
    DependentTypeContext,
    DimConstraint,
    DimExpr,
    DimRelation,
    // Effect System
    Effect,
    EffectContext,
    EffectHandler,
    EffectRegistry,
    EffectSet,
    LinearContext,
    // Linear Types
    LinearType,
    LinearTypeRegistry,
    RefinementContext,
    RefinementPredicate,
    RefinementRegistry,
    // Refinement Types
    RefinementType,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          TensorLogic Advanced Type System Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Part 1: Refinement Types
    demo_refinement_types();

    // Part 2: Dependent Types
    demo_dependent_types();

    // Part 3: Linear Types
    demo_linear_types();

    // Part 4: Effect System
    demo_effect_system();

    // Part 5: Combined Example
    demo_combined_usage();

    println!("\n✓ All advanced type system demos completed successfully!");
}

fn demo_refinement_types() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Part 1: Refinement Types                                    │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Create custom refinement types
    let learning_rate = RefinementType::new("Float")
        .with_name("LearningRate")
        .with_predicate(RefinementPredicate::range(0.0, 1.0))
        .with_predicate(RefinementPredicate::GreaterThan(0.0))
        .with_description("Learning rate must be in (0, 1]");

    let batch_size = RefinementType::new("Int")
        .with_name("BatchSize")
        .with_predicate(RefinementPredicate::GreaterThan(0.0))
        .with_predicate(RefinementPredicate::modulo(2, 0)) // Must be even
        .with_description("Batch size must be positive and even");

    // Validate values
    println!("  LearningRate validation:");
    println!(
        "    0.001 -> {}",
        if learning_rate.check(0.001) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );
    println!(
        "    0.0   -> {}",
        if learning_rate.check(0.0) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );
    println!(
        "    1.5   -> {}",
        if learning_rate.check(1.5) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );

    println!("\n  BatchSize validation:");
    println!(
        "    32    -> {}",
        if batch_size.check(32.0) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );
    println!(
        "    33    -> {}",
        if batch_size.check(33.0) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );
    println!(
        "    -4    -> {}",
        if batch_size.check(-4.0) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );

    // Use built-in registry
    let registry = RefinementRegistry::with_builtins();
    println!("\n  Built-in types:");
    for name in registry.type_names() {
        print!("    {}", name);
        if let Some(ty) = registry.get(name) {
            if let Some(desc) = &ty.description {
                print!(" - {}", desc);
            }
        }
        println!();
    }

    // Dependent refinement with context
    let bounded = RefinementType::new("Int")
        .with_name("BoundedIndex")
        .with_predicate(RefinementPredicate::GreaterThanOrEqual(0.0))
        .with_predicate(RefinementPredicate::dependent(
            "array_len",
            tensorlogic_adapters::DependentRelation::LessThan,
        ));

    let mut ctx = RefinementContext::new();
    ctx.set_value("array_len", 10.0);

    println!("\n  Dependent refinement (index < array_len where array_len=10):");
    println!(
        "    5  -> {}",
        if bounded.check_with_context(5.0, &ctx) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );
    println!(
        "    10 -> {}",
        if bounded.check_with_context(10.0, &ctx) {
            "✓ valid"
        } else {
            "✗ invalid"
        }
    );

    println!();
}

fn demo_dependent_types() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Part 2: Dependent Types                                     │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Create dimension context
    let mut ctx = DependentTypeContext::new();
    ctx.set_dim("batch", 32);
    ctx.set_dim("seq_len", 512);
    ctx.set_dim("hidden", 768);
    ctx.set_dim("heads", 12);
    ctx.set_dim("head_dim", 64);

    // Define transformer tensor types
    let input_tensor = DependentType::tensor(
        "Float",
        vec![
            DimExpr::var("batch"),
            DimExpr::var("seq_len"),
            DimExpr::var("hidden"),
        ],
    )
    .with_name("TransformerInput");

    let attention = dependent_patterns::attention_tensor(
        DimExpr::var("batch"),
        DimExpr::var("heads"),
        DimExpr::var("seq_len"),
        DimExpr::var("head_dim"),
    );

    // Compute shapes
    println!("  Dimension context:");
    println!(
        "    batch={}, seq_len={}, hidden={}, heads={}, head_dim={}",
        ctx.get_dim("batch").unwrap(),
        ctx.get_dim("seq_len").unwrap(),
        ctx.get_dim("hidden").unwrap(),
        ctx.get_dim("heads").unwrap(),
        ctx.get_dim("head_dim").unwrap(),
    );

    println!("\n  Tensor shapes:");
    if let Some(shape) = input_tensor.eval_shape(&ctx) {
        println!("    {} -> {:?}", input_tensor.type_name(), shape);
    }
    if let Some(shape) = attention.eval_shape(&ctx) {
        println!("    {} -> {:?}", attention.type_name(), shape);
    }

    // Dimension expressions
    let total_params = DimExpr::var("hidden")
        .mul(DimExpr::var("hidden"))
        .mul(DimExpr::constant(4)); // 4 matrices in attention

    println!("\n  Dimension expressions:");
    println!("    hidden * hidden * 4 = {:?}", total_params.eval(&ctx));

    // Type with constraints
    let square_matrix = DependentType::matrix("Float", DimExpr::var("n"), DimExpr::var("n"))
        .with_constraint(
            DimConstraint::new(
                DimExpr::var("n"),
                DimRelation::GreaterThan,
                DimExpr::constant(0),
            )
            .with_message("Matrix dimension must be positive"),
        );

    ctx.set_dim("n", 64);
    match square_matrix.check_constraints(&ctx) {
        Ok(()) => println!("\n  SquareMatrix<64,64> constraints: ✓ satisfied"),
        Err(e) => println!("\n  SquareMatrix constraints: ✗ {}", e),
    }

    println!();
}

fn demo_linear_types() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Part 3: Linear Types                                        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Create a linear context
    let mut ctx = LinearContext::new();

    // Show linearity kinds
    println!("  Linearity kinds:");
    println!("    Linear:       must be used exactly once");
    println!("    Affine:       can be used at most once (droppable)");
    println!("    Relevant:     must be used at least once (copyable)");
    println!("    Unrestricted: no constraints (standard types)");

    // Create resources with different linearity
    let gpu_tensor = LinearType::linear("Tensor")
        .with_name("GpuTensor")
        .with_tag("gpu")
        .with_description("GPU tensor that must be freed");

    let file_handle = LinearType::affine("Handle")
        .with_name("FileHandle")
        .with_tag("io");

    let config = LinearType::relevant("Config").with_name("SharedConfig");

    // Track resources
    ctx.enter_scope();
    ctx.create_resource("weights", gpu_tensor.clone(), "model.rs:10")
        .unwrap();
    ctx.create_resource("data_file", file_handle, "loader.rs:20")
        .unwrap();
    ctx.create_resource("training_config", config, "main.rs:5")
        .unwrap();

    println!("\n  Created resources:");
    println!("    weights (linear), data_file (affine), training_config (relevant)");

    // Use resources
    ctx.use_resource("weights", "forward.rs:50").unwrap();
    ctx.use_resource("training_config", "train.rs:100").unwrap();
    ctx.use_resource("training_config", "eval.rs:200").unwrap(); // relevant can be used multiple times

    // Check statistics
    let stats = ctx.statistics();
    println!("\n  Resource statistics:");
    println!("    Total: {}", stats.total);
    println!("    Used:  {}", stats.used);
    println!("    Unused: {}", stats.unused);

    // Validate scope
    match ctx.exit_scope() {
        Ok(()) => println!("\n  Scope validation: ✓ all linear resources properly used"),
        Err(errors) => {
            println!("\n  Scope validation errors:");
            for e in errors {
                println!("    ✗ {}", e);
            }
        }
    }

    // Show built-in types
    let registry = LinearTypeRegistry::with_builtins();
    println!("\n  Built-in linear types:");
    for name in registry.type_names() {
        if let Some(ty) = registry.get(name) {
            println!(
                "    {} ({}) - {:?}",
                name,
                ty.kind,
                ty.description.as_deref().unwrap_or("")
            );
        }
    }

    println!();
}

fn demo_effect_system() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Part 4: Effect System                                       │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Create effect sets
    let pure = EffectSet::pure();
    let io_effects = EffectSet::new().with(Effect::IO).with(Effect::FileSystem);
    let gpu_effects = EffectSet::new().with(Effect::GPU).with(Effect::Alloc);
    let random_effects = EffectSet::singleton(Effect::NonDet);

    println!("  Effect sets:");
    println!("    Pure math: {}", pure);
    println!("    IO ops:    {}", io_effects);
    println!("    GPU ops:   {}", gpu_effects);
    println!("    Random:    {}", random_effects);

    // Effect operations
    let combined = io_effects.union(&gpu_effects);
    println!("\n  Combined effects (IO ∪ GPU): {}", combined);

    // Effect properties
    println!("\n  Effect properties:");
    println!("    Pure is_pure: {}", pure.is_pure());
    println!("    Combined is_total: {}", combined.is_total());
    println!(
        "    Random is_deterministic: {}",
        random_effects.is_deterministic()
    );

    // Effect handlers
    let mut ctx = EffectContext::new();
    ctx.install_handler(
        EffectHandler::new("io_handler")
            .with_effect(Effect::IO)
            .with_effect(Effect::FileSystem),
    );

    let unhandled = ctx.unhandled(&combined);
    println!("\n  After installing IO handler:");
    println!("    Unhandled effects: {}", unhandled);
    println!("    All handled: {}", ctx.all_handled(&combined));

    // Use built-in registry
    let registry = EffectRegistry::with_builtins();
    println!("\n  Function effect signatures:");
    for name in ["sin", "print", "random", "gpu_matmul"].iter() {
        if let Some(sig) = registry.get(name) {
            println!("    {}: {}", name, sig.effects);
        }
    }

    // Infer effects from operations
    let ops = vec!["sin", "cos", "exp"];
    let inferred = infer_effects(&registry, &ops);
    println!("\n  Effect inference:");
    println!(
        "    [sin, cos, exp] -> {} (pure: {})",
        inferred,
        inferred.is_pure()
    );

    let ops = vec!["sin", "print", "random"];
    let inferred = infer_effects(&registry, &ops);
    println!("    [sin, print, random] -> {}", inferred);

    println!();
}

fn demo_combined_usage() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Part 5: Combined Usage - Neural Network Layer               │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Define a fully-typed neural network layer

    // 1. Refinement: Layer parameters
    let dropout_rate = RefinementType::new("Float")
        .with_name("DropoutRate")
        .with_predicate(RefinementPredicate::range(0.0, 1.0));

    // 2. Dependent: Tensor shapes
    let mut dim_ctx = DependentTypeContext::new();
    dim_ctx.set_dim("batch", 32);
    dim_ctx.set_dim("in_features", 512);
    dim_ctx.set_dim("out_features", 256);

    let weight_type = DependentType::matrix(
        "Float",
        DimExpr::var("out_features"),
        DimExpr::var("in_features"),
    )
    .with_constraint(DimConstraint::new(
        DimExpr::var("in_features"),
        DimRelation::GreaterThan,
        DimExpr::constant(0),
    ));

    let input_type =
        DependentType::matrix("Float", DimExpr::var("batch"), DimExpr::var("in_features"));

    // 3. Linear: Resource management
    let mut linear_ctx = LinearContext::new();
    let gpu_tensor = LinearType::linear("Tensor").with_tag("gpu");

    linear_ctx.enter_scope();
    linear_ctx
        .create_resource("weights", gpu_tensor.clone(), "layer.rs:1")
        .unwrap();
    linear_ctx
        .create_resource("input", gpu_tensor.clone(), "layer.rs:2")
        .unwrap();

    // 4. Effects: Operation tracking
    let layer_effects = EffectSet::new()
        .with(Effect::GPU)
        .with(Effect::Alloc)
        .with(Effect::NonDet); // Dropout introduces non-determinism

    // Validate layer configuration
    println!("  Neural Network Layer Configuration:");
    println!("  ────────────────────────────────────");

    // Check dropout rate
    let rate = 0.3;
    println!("\n  Dropout rate: {}", rate);
    if dropout_rate.check(rate) {
        println!("    Refinement check: ✓ valid");
    }

    // Check tensor shapes
    if let (Some(w_shape), Some(i_shape)) = (
        weight_type.eval_shape(&dim_ctx),
        input_type.eval_shape(&dim_ctx),
    ) {
        println!("\n  Tensor shapes:");
        println!("    Weights: {:?}", w_shape);
        println!("    Input:   {:?}", i_shape);

        // Output shape would be [batch, out_features]
        println!(
            "    Output:  [{}, {}]",
            dim_ctx.get_dim("batch").unwrap(),
            dim_ctx.get_dim("out_features").unwrap()
        );
    }

    // Use resources
    linear_ctx.use_resource("weights", "forward:1").unwrap();
    linear_ctx.use_resource("input", "forward:2").unwrap();

    println!("\n  Resource usage:");
    let stats = linear_ctx.statistics();
    println!("    {} resources, {} used", stats.total, stats.used);

    // Check effects
    println!("\n  Layer effects: {}", layer_effects);
    println!("    GPU ops:     {}", layer_effects.has(Effect::GPU));
    println!("    Deterministic: {}", layer_effects.is_deterministic());

    // Cleanup
    match linear_ctx.exit_scope() {
        Ok(()) => println!("\n  Resource cleanup: ✓ all GPU tensors freed"),
        Err(e) => println!("\n  Resource cleanup errors: {:?}", e),
    }

    println!("\n  ═══════════════════════════════════════════════════════════");
    println!("  Layer fully validated with all 4 type system features!");
}
