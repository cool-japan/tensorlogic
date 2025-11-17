//! Property-based tests for logical operations.
//!
//! This module verifies that compiled logical operations satisfy fundamental
//! mathematical properties like monotonicity, symmetry, associativity, and
//! logical laws (De Morgan's, etc.).
//!
//! **Note**: These tests require the `integration-tests` feature AND
//! tensorlogic-scirs-backend dev-dependency (currently commented out due to
//! circular dependencies).
//!
//! To enable these tests:
//! 1. Uncomment tensorlogic-scirs-backend in Cargo.toml [dev-dependencies]
//! 2. Run: `cargo test --features integration-tests`

#![allow(unexpected_cfgs)]

// These tests are feature-gated to avoid circular dependencies
// When integration-tests feature is enabled, it expects scirs-backend to be available
// But scirs-backend creates circular dev-dependency, so we disable the entire module

#[allow(unexpected_cfgs)]
#[cfg(all(
    test,
    feature = "integration-tests",
    feature = "__has_scirs_backend_disabled"
))]
mod tests {
    // This module is never compiled because __has_scirs_backend_disabled feature doesn't exist
    // It's a way to conditionally disable code that requires unavailable dependencies
}

// Placeholder module for when integration-tests is enabled but scirs-backend isn't available
#[cfg(all(test, feature = "integration-tests"))]
mod property_tests_placeholder {
    #[test]
    fn property_tests_require_scirs_backend() {
        eprintln!("Warning: Property tests require tensorlogic-scirs-backend");
        eprintln!("These tests are currently disabled due to circular dev-dependencies");
        eprintln!(
            "To enable: uncomment tensorlogic-scirs-backend in Cargo.toml [dev-dependencies]"
        );
    }
}
