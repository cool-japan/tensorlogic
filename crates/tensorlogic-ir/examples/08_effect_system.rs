//! Example demonstrating the effect system for tracking computational effects.
//!
//! This example shows how to use the effect system to track various kinds of
//! computational effects in logical expressions and tensor operations.

use tensorlogic_ir::effect_system::{
    infer_operation_effects, ComputationalEffect, Effect, EffectAnnotation, EffectScheme,
    EffectSet, EffectSubstitution, EffectVar, MemoryEffect, ProbabilisticEffect,
};

fn main() {
    println!("=== Effect System Example ===\n");

    // 1. Basic Effects
    println!("1. Basic Effect Types");
    demonstrate_basic_effects();
    println!();

    // 2. Effect Sets
    println!("2. Effect Sets and Combinations");
    demonstrate_effect_sets();
    println!();

    // 3. Effect Inference
    println!("3. Effect Inference for Operations");
    demonstrate_effect_inference();
    println!();

    // 4. Effect Polymorphism
    println!("4. Effect Polymorphism");
    demonstrate_effect_polymorphism();
    println!();

    // 5. Effect Checking
    println!("5. Effect Compatibility Checking");
    demonstrate_effect_checking();
    println!();

    // 6. Effect Annotations
    println!("6. Effect Annotations");
    demonstrate_effect_annotations();
    println!();
}

fn demonstrate_basic_effects() {
    // Computational effects
    let pure = Effect::Computational(ComputationalEffect::Pure);
    let impure = Effect::Computational(ComputationalEffect::Impure);
    let io = Effect::Computational(ComputationalEffect::IO);

    println!("  Computational Effects:");
    println!("    Pure: {}", pure);
    println!("    Impure: {}", impure);
    println!("    IO: {}", io);

    // Memory effects
    let read_only = Effect::Memory(MemoryEffect::ReadOnly);
    let read_write = Effect::Memory(MemoryEffect::ReadWrite);
    let allocating = Effect::Memory(MemoryEffect::Allocating);

    println!("\n  Memory Effects:");
    println!("    ReadOnly: {}", read_only);
    println!("    ReadWrite: {}", read_write);
    println!("    Allocating: {}", allocating);

    // Probabilistic effects
    let deterministic = Effect::Probabilistic(ProbabilisticEffect::Deterministic);
    let stochastic = Effect::Probabilistic(ProbabilisticEffect::Stochastic);

    println!("\n  Probabilistic Effects:");
    println!("    Deterministic: {}", deterministic);
    println!("    Stochastic: {}", stochastic);

    // Other effects
    let differentiable = Effect::Differentiable;
    let non_differentiable = Effect::NonDifferentiable;
    let async_effect = Effect::Async;
    let parallel = Effect::Parallel;
    let custom = Effect::Custom("GPUCompute".to_string());

    println!("\n  Other Effects:");
    println!("    Differentiable: {}", differentiable);
    println!("    NonDifferentiable: {}", non_differentiable);
    println!("    Async: {}", async_effect);
    println!("    Parallel: {}", parallel);
    println!("    Custom: {}", custom);
}

fn demonstrate_effect_sets() {
    // Create pure effect set
    let pure_set = EffectSet::pure();
    println!("  Pure effect set: {}", pure_set);
    println!("    Is pure? {}", pure_set.is_pure());
    println!("    Is impure? {}", pure_set.is_impure());

    // Create differentiable effect set
    let diff_set = EffectSet::differentiable();
    println!("\n  Differentiable effect set: {}", diff_set);
    println!("    Is differentiable? {}", diff_set.is_differentiable());

    // Create stochastic effect set
    let stochastic_set = EffectSet::stochastic();
    println!("\n  Stochastic effect set: {}", stochastic_set);
    println!("    Is stochastic? {}", stochastic_set.is_stochastic());

    // Combine effects
    let combined = pure_set.union(&diff_set);
    println!("\n  Combined (pure ∪ differentiable): {}", combined);
    println!("    Is pure? {}", combined.is_pure());
    println!("    Is differentiable? {}", combined.is_differentiable());

    // Build custom effect set
    let custom_set = EffectSet::new()
        .with(Effect::Computational(ComputationalEffect::Pure))
        .with(Effect::Differentiable)
        .with(Effect::Parallel)
        .with(Effect::Custom("TensorOp".to_string()));

    println!("\n  Custom effect set: {}", custom_set);
    println!(
        "    Contains Pure? {}",
        custom_set.contains(&Effect::Computational(ComputationalEffect::Pure))
    );
    println!(
        "    Contains Parallel? {}",
        custom_set.contains(&Effect::Parallel)
    );
}

fn demonstrate_effect_inference() {
    // Infer effects for common operations
    let operations = vec![
        "and",
        "or",
        "not",
        "implies",
        "add",
        "subtract",
        "multiply",
        "divide",
        "exists",
        "forall",
        "equal",
        "less_than",
        "sample",
        "random",
        "read",
        "write",
        "unknown_op",
    ];

    println!("  Operation effect inference:");
    for op in operations {
        let effects = infer_operation_effects(op);
        println!("    {:<12} -> {}", op, effects);
    }
}

fn demonstrate_effect_polymorphism() {
    // Effect variables for polymorphism
    let e1 = EffectVar::new("1");
    let e2 = EffectVar::new("2");

    println!("  Effect variables:");
    println!("    ε1 = {}", e1);
    println!("    ε2 = {}", e2);

    // Concrete effect scheme
    let concrete = EffectScheme::concrete(EffectSet::pure());
    println!("\n  Concrete effect scheme: {}", concrete);

    // Variable effect scheme
    let variable = EffectScheme::variable("f");
    println!("  Variable effect scheme: {}", variable);

    // Union of effect schemes
    let pure_scheme = EffectScheme::concrete(EffectSet::pure());
    let diff_scheme = EffectScheme::concrete(EffectSet::differentiable());
    let union = EffectScheme::union(pure_scheme, diff_scheme);
    println!("  Union effect scheme: {}", union);

    // Effect substitution
    let mut subst = EffectSubstitution::new();
    subst.insert(EffectVar::new("f"), EffectSet::pure());

    let var_scheme = EffectScheme::variable("f");
    let substituted = var_scheme.substitute(&subst);
    println!("\n  Substitution:");
    println!("    Before: {}", var_scheme);
    println!("    After: {}", substituted);

    // Evaluate effect scheme
    match union.evaluate(&EffectSubstitution::new()) {
        Ok(effects) => {
            println!("\n  Evaluated union:");
            println!("    Effects: {}", effects);
            println!("    Is pure? {}", effects.is_pure());
            println!("    Is differentiable? {}", effects.is_differentiable());
        }
        Err(e) => println!("  Error: {}", e),
    }
}

fn demonstrate_effect_checking() {
    // Compatible effects
    let pure1 = EffectSet::pure();
    let pure2 = EffectSet::pure().with(Effect::Differentiable);

    println!("  Effect compatibility:");
    println!("    pure ⊆ (pure + diff)? {}", pure1.is_subset_of(&pure2));
    println!("    (pure + diff) ⊆ pure? {}", pure2.is_subset_of(&pure1));
    println!("    Compatible? {}", pure1.is_compatible_with(&pure2));

    // Conflicting effects
    let pure = EffectSet::pure();
    let impure = EffectSet::impure();

    println!("\n  Conflicting effects:");
    println!("    Pure: {}", pure);
    println!("    Impure: {}", impure);
    println!("    Compatible? {}", pure.is_compatible_with(&impure));

    // Differentiable vs non-differentiable
    let diff = EffectSet::new().with(Effect::Differentiable);
    let non_diff = EffectSet::new().with(Effect::NonDifferentiable);

    println!("\n  Differentiability conflict:");
    println!("    Diff: {}", diff);
    println!("    NonDiff: {}", non_diff);
    println!("    Compatible? {}", diff.is_compatible_with(&non_diff));

    // Intersection
    let set1 = EffectSet::pure().with(Effect::Differentiable);
    let set2 = EffectSet::differentiable().with(Effect::Parallel);
    let intersection = set1.intersection(&set2);

    println!("\n  Effect intersection:");
    println!("    Set1: {}", set1);
    println!("    Set2: {}", set2);
    println!("    Intersection: {}", intersection);
}

fn demonstrate_effect_annotations() {
    // Pure annotation
    let pure_ann = EffectAnnotation::pure().with_description("Pure mathematical computation");

    println!("  Effect annotations:");
    println!("    Pure: {}", pure_ann.scheme);
    if let Some(desc) = &pure_ann.description {
        println!("      Description: {}", desc);
    }

    // Differentiable annotation
    let diff_ann =
        EffectAnnotation::differentiable().with_description("Supports automatic differentiation");

    println!("\n    Differentiable: {}", diff_ann.scheme);
    if let Some(desc) = &diff_ann.description {
        println!("      Description: {}", desc);
    }

    // Custom annotation with effect polymorphism
    let poly_scheme = EffectScheme::union(
        EffectScheme::variable("ε"),
        EffectScheme::concrete(EffectSet::differentiable()),
    );

    let poly_ann = EffectAnnotation::new(poly_scheme)
        .with_description("Polymorphic over effect ε, but always differentiable");

    println!("\n    Polymorphic: {}", poly_ann.scheme);
    if let Some(desc) = &poly_ann.description {
        println!("      Description: {}", desc);
    }

    // Practical example: Neural network layer
    let nn_layer_effects = EffectSet::new()
        .with(Effect::Computational(ComputationalEffect::Pure))
        .with(Effect::Differentiable)
        .with(Effect::Memory(MemoryEffect::Allocating))
        .with(Effect::Parallel);

    let nn_ann = EffectAnnotation::new(EffectScheme::concrete(nn_layer_effects)).with_description(
        "Neural network layer: pure, differentiable, allocates memory, parallelizable",
    );

    println!("\n    Neural Network Layer: {}", nn_ann.scheme);
    if let Some(desc) = &nn_ann.description {
        println!("      Description: {}", desc);
    }

    // Sampling operation
    let sample_effects = EffectSet::new()
        .with(Effect::Probabilistic(ProbabilisticEffect::Stochastic))
        .with(Effect::NonDifferentiable)
        .with(Effect::Memory(MemoryEffect::ReadOnly));

    let sample_ann = EffectAnnotation::new(EffectScheme::concrete(sample_effects))
        .with_description("Random sampling: stochastic, non-differentiable");

    println!("\n    Sampling Operation: {}", sample_ann.scheme);
    if let Some(desc) = &sample_ann.description {
        println!("      Description: {}", desc);
    }
}
