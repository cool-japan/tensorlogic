//! Linear Types Example
//!
//! Demonstrates the linear type system for resource management and safe in-place operations.
//! Linear types ensure values are used exactly once, preventing use-after-free and double-free bugs.

use std::collections::HashSet;
use tensorlogic_ir::linear::{
    Capability, LinearContext, LinearResource, LinearType, LinearityChecker, Multiplicity,
};

fn main() {
    println!("=== Linear Types in TensorLogic ===\n");

    // Example 1: Basic linear types
    example_basic_linear_types();

    // Example 2: Multiplicity system
    example_multiplicity_system();

    // Example 3: Linear context and usage tracking
    example_linear_context();

    // Example 4: Linearity violations
    example_linearity_violations();

    // Example 5: Linear resources with capabilities
    example_linear_resources();

    // Example 6: Context merging (branching control flow)
    example_context_merging();

    // Example 7: Context splitting (parallel use)
    example_context_splitting();

    // Example 8: Linearity checker
    example_linearity_checker();
}

fn example_basic_linear_types() {
    println!("--- Example 1: Basic Linear Types ---");

    // Linear type: must be used exactly once
    let tensor_handle = LinearType::linear("TensorHandle");
    println!("Linear type: {}", tensor_handle);
    println!("Is linear: {}", tensor_handle.is_linear());
    println!("Is unrestricted: {}", tensor_handle.is_unrestricted());

    // Affine type: at most once
    let file_handle = LinearType::affine("FileHandle");
    println!("\nAffine type: {}", file_handle);

    // Relevant type: at least once
    let resource = LinearType::relevant("Resource");
    println!("Relevant type: {}", resource);

    // Unrestricted type: any number of times
    let int_type = LinearType::unrestricted("Int");
    println!("Unrestricted type: {}\n", int_type);
}

fn example_multiplicity_system() {
    println!("--- Example 2: Multiplicity System ---");

    let linear = Multiplicity::Linear;
    let affine = Multiplicity::Affine;
    let relevant = Multiplicity::Relevant;
    let unrestricted = Multiplicity::Unrestricted;

    println!("Linear (1):");
    println!("  Allows 0 uses: {}", linear.allows(0));
    println!("  Allows 1 use: {}", linear.allows(1));
    println!("  Allows 2 uses: {}", linear.allows(2));

    println!("\nAffine (0..1):");
    println!("  Allows 0 uses: {}", affine.allows(0));
    println!("  Allows 1 use: {}", affine.allows(1));
    println!("  Allows 2 uses: {}", affine.allows(2));

    println!("\nRelevant (1..):");
    println!("  Allows 0 uses: {}", relevant.allows(0));
    println!("  Allows 1 use: {}", relevant.allows(1));
    println!("  Allows 2 uses: {}", relevant.allows(2));

    println!("\nUnrestricted (0..):");
    println!("  Allows 0 uses: {}", unrestricted.allows(0));
    println!("  Allows 1 use: {}", unrestricted.allows(1));
    println!("  Allows 100 uses: {}\n", unrestricted.allows(100));

    // Combining multiplicities
    println!("Combining multiplicities:");
    println!("  Linear + Linear = {}", linear.combine(&linear));
    println!(
        "  Linear + Unrestricted = {}",
        linear.combine(&unrestricted)
    );
    println!(
        "  Unrestricted + Unrestricted = {}\n",
        unrestricted.combine(&unrestricted)
    );
}

fn example_linear_context() {
    println!("--- Example 3: Linear Context and Usage Tracking ---");

    let mut ctx = LinearContext::new();

    // Bind a linear tensor
    let tensor = LinearType::linear("Tensor");
    ctx.bind("x", tensor);

    println!("Bound linear variable 'x'");
    println!("Is linear: {}", ctx.is_linear("x"));
    println!("Is consumed: {}", ctx.is_consumed("x"));

    // Use it once - should succeed
    match ctx.use_var("x") {
        Ok(_) => println!("✓ First use of 'x' succeeded"),
        Err(e) => println!("✗ Error: {}", e),
    }

    println!("Is consumed after use: {}", ctx.is_consumed("x"));

    // Try to use again - should fail
    match ctx.use_var("x") {
        Ok(_) => println!("✗ Second use succeeded (should have failed!)"),
        Err(e) => println!("✓ Second use failed as expected: {}", e),
    }

    // Unrestricted variable can be used multiple times
    let mut ctx2 = LinearContext::new();
    ctx2.bind("y", LinearType::unrestricted("Int"));

    for i in 1..=5 {
        ctx2.use_var("y").unwrap();
        println!("Use #{} of unrestricted variable 'y': OK", i);
    }

    println!();
}

fn example_linearity_violations() {
    println!("--- Example 4: Linearity Violations ---");

    // Violation 1: Not using a linear variable
    let mut ctx1 = LinearContext::new();
    ctx1.bind("x", LinearType::linear("Tensor"));

    match ctx1.validate() {
        Ok(_) => println!("✗ Validation passed (should fail - unused linear variable)"),
        Err(errors) => {
            println!("✓ Validation failed for unused linear variable:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Violation 2: Using a relevant variable zero times
    let mut ctx2 = LinearContext::new();
    ctx2.bind("r", LinearType::relevant("Resource"));

    match ctx2.validate() {
        Ok(_) => println!("\n✗ Validation passed (should fail - relevant not used)"),
        Err(errors) => {
            println!("\n✓ Validation failed for unused relevant variable:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Correct usage
    let mut ctx3 = LinearContext::new();
    ctx3.bind("z", LinearType::linear("Tensor"));
    ctx3.use_var("z").unwrap();

    match ctx3.validate() {
        Ok(_) => println!("\n✓ Validation passed for properly used linear variable"),
        Err(_) => println!("\n✗ Validation failed (should pass)"),
    }

    println!();
}

fn example_linear_resources() {
    println!("--- Example 5: Linear Resources with Capabilities ---");

    // Read-only resource
    let read_only = LinearResource::read_only(LinearType::linear("File"));
    println!("Read-only resource:");
    println!(
        "  Has Read capability: {}",
        read_only.has_capability(&Capability::Read)
    );
    println!(
        "  Has Write capability: {}",
        read_only.has_capability(&Capability::Write)
    );
    println!(
        "  Has Own capability: {}",
        read_only.has_capability(&Capability::Own)
    );

    // Read-write resource
    let read_write = LinearResource::read_write(LinearType::linear("File"));
    println!("\nRead-write resource:");
    println!(
        "  Has Read capability: {}",
        read_write.has_capability(&Capability::Read)
    );
    println!(
        "  Has Write capability: {}",
        read_write.has_capability(&Capability::Write)
    );
    println!(
        "  Has Own capability: {}",
        read_write.has_capability(&Capability::Own)
    );

    // Owned resource (full access)
    let owned = LinearResource::owned(LinearType::linear("Tensor"));
    println!("\nOwned resource:");
    println!(
        "  Has Read capability: {}",
        owned.has_capability(&Capability::Read)
    );
    println!(
        "  Has Write capability: {}",
        owned.has_capability(&Capability::Write)
    );
    println!(
        "  Has Own capability: {}",
        owned.has_capability(&Capability::Own)
    );

    // Custom capabilities
    let mut custom_caps = HashSet::new();
    custom_caps.insert(Capability::Read);
    custom_caps.insert(Capability::Execute);
    let custom = LinearResource::new(LinearType::affine("Code"), custom_caps);

    println!("\nCustom resource (Read + Execute):");
    println!(
        "  Has Read capability: {}",
        custom.has_capability(&Capability::Read)
    );
    println!(
        "  Has Execute capability: {}",
        custom.has_capability(&Capability::Execute)
    );
    println!(
        "  Has Write capability: {}",
        custom.has_capability(&Capability::Write)
    );

    println!();
}

fn example_context_merging() {
    println!("--- Example 6: Context Merging (Branching Control Flow) ---");

    // Simulate if-else branching
    let mut then_branch = LinearContext::new();
    let mut else_branch = LinearContext::new();

    // Both branches have the same variable
    then_branch.bind("x", LinearType::unrestricted("Int"));
    else_branch.bind("x", LinearType::unrestricted("Int"));

    // Different usage in each branch
    then_branch.use_var("x").unwrap();
    then_branch.use_var("x").unwrap();

    else_branch.use_var("x").unwrap();

    // Merge contexts
    match then_branch.merge(&else_branch) {
        Ok(merged) => {
            println!("✓ Successfully merged contexts");
            println!("Merged context valid: {}", merged.validate().is_ok());
        }
        Err(e) => println!("✗ Failed to merge: {}", e),
    }

    // Linear variables must be used in both branches
    let mut then_branch2 = LinearContext::new();
    let mut else_branch2 = LinearContext::new();

    then_branch2.bind("y", LinearType::linear("Tensor"));
    else_branch2.bind("y", LinearType::linear("Tensor"));

    // Use in then branch
    then_branch2.use_var("y").unwrap();

    // Don't use in else branch
    // (This should fail when merging)

    match then_branch2.merge(&else_branch2) {
        Ok(_) => {
            println!("\n✗ Merge succeeded (should fail - linear var not used in both branches)")
        }
        Err(e) => println!("\n✓ Merge failed as expected: {}", e),
    }

    println!();
}

fn example_context_splitting() {
    println!("--- Example 7: Context Splitting (Parallel Use) ---");

    let mut ctx = LinearContext::new();

    // Add linear and unrestricted variables
    ctx.bind("x", LinearType::linear("Tensor"));
    ctx.bind("y", LinearType::unrestricted("Int"));

    println!("Original context has: x (linear), y (unrestricted)");

    // Split off x for parallel use
    match ctx.split(&["x".to_string()]) {
        Ok(split_ctx) => {
            println!("✓ Successfully split context");
            println!("Split context has x: {}", split_ctx.get_type("x").is_some());
            println!("Original context - x is consumed: {}", ctx.is_consumed("x"));
            println!(
                "Original context still has y: {}",
                ctx.get_type("y").is_some()
            );
        }
        Err(e) => println!("✗ Split failed: {}", e),
    }

    // Try to split an affine variable
    let mut ctx2 = LinearContext::new();
    ctx2.bind("z", LinearType::affine("File"));

    match ctx2.split(&["z".to_string()]) {
        Ok(_) => println!("\n✗ Split affine succeeded (should fail)"),
        Err(e) => println!("\n✓ Split affine failed as expected: {}", e),
    }

    println!();
}

fn example_linearity_checker() {
    println!("--- Example 8: Linearity Checker ---");

    let mut checker = LinearityChecker::new();

    // Bind variables
    checker.bind("tensor", LinearType::linear("Tensor"));
    checker.bind("counter", LinearType::unrestricted("Int"));
    checker.bind("file", LinearType::affine("File"));

    println!("Bound variables: tensor (linear), counter (unrestricted), file (affine)");

    // Use them
    checker.use_var("tensor");
    checker.use_var("counter");
    checker.use_var("counter");
    checker.use_var("counter");
    // Note: file is affine, so we can choose not to use it

    println!("\nUsage:");
    println!("  tensor: 1 time");
    println!("  counter: 3 times");
    println!("  file: 0 times (affine allows this)");

    // Check linearity
    match checker.check() {
        Ok(_) => println!("\n✓ Linearity check passed"),
        Err(errors) => {
            println!("\n✗ Linearity check failed:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Example with violation
    let mut bad_checker = LinearityChecker::new();
    bad_checker.bind("x", LinearType::linear("Tensor"));
    bad_checker.use_var("x");
    bad_checker.use_var("x"); // Double use!

    match bad_checker.check() {
        Ok(_) => println!("\n✗ Check passed (should fail)"),
        Err(errors) => {
            println!("\n✓ Check failed as expected:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Example with unused linear variable
    let mut unused_checker = LinearityChecker::new();
    unused_checker.bind("y", LinearType::linear("Tensor"));
    // Don't use it!

    match unused_checker.check() {
        Ok(_) => println!("\n✗ Check passed (should fail - unused)"),
        Err(errors) => {
            println!("\n✓ Check failed for unused linear variable:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Get unused required variables
    let unused = unused_checker.context().get_unused_required();
    println!("\nUnused required variables: {:?}", unused);
}
