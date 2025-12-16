//! Validation passes for TLExpr.
//!
//! This module provides comprehensive validation for logical expressions before compilation.
//! Validation helps catch errors early and provide helpful error messages.

use anyhow::{anyhow, Result};
use tensorlogic_ir::{PredicateSignature, TLExpr};

use super::diagnostics::{diagnose_expression, DiagnosticLevel};
use super::scope_analysis::{analyze_scopes, suggest_quantifiers};
use super::type_checking::TypeChecker;

/// Validate that all predicates with the same name have the same arity.
///
/// This is a basic validation that checks for consistency in predicate usage.
/// Predicates must have the same number of arguments everywhere they appear.
///
/// # Errors
///
/// Returns an error if any predicate is used with different arities.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::passes::validate_arity;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Valid: knows/2 used consistently
/// let expr = TLExpr::and(
///     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
///     TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]),
/// );
/// assert!(validate_arity(&expr).is_ok());
/// ```
pub fn validate_arity(expr: &TLExpr) -> Result<()> {
    expr.validate_arity().map_err(|e| anyhow!("{}", e))
}

/// Result of pre-compilation validation.
///
/// Contains all validation errors, warnings, and suggestions found during validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed (no errors)
    pub passed: bool,
    /// Number of errors found
    pub error_count: usize,
    /// Number of warnings found
    pub warning_count: usize,
    /// All diagnostic messages (errors, warnings, hints)
    pub diagnostics: Vec<String>,
}

impl ValidationResult {
    /// Returns true if validation passed (no errors)
    pub fn is_ok(&self) -> bool {
        self.passed
    }

    /// Returns true if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Returns a formatted error message with all diagnostics
    pub fn error_message(&self) -> String {
        self.diagnostics.join("\n")
    }
}

/// Performs comprehensive pre-compilation validation.
///
/// This function runs all available validation passes:
/// 1. Arity validation (predicate consistency)
/// 2. Scope analysis (unbound variables)
/// 3. Enhanced diagnostics (unused bindings, type conflicts)
///
/// # Arguments
///
/// * `expr` - The expression to validate
///
/// # Returns
///
/// A `ValidationResult` containing all errors, warnings, and suggestions.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::passes::validate_expression;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Valid expression (fully quantified)
/// let expr = TLExpr::exists(
///     "x",
///     "Person",
///     TLExpr::exists(
///         "y",
///         "Person",
///         TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
///     ),
/// );
///
/// let result = validate_expression(&expr);
/// assert!(result.is_ok());
/// ```
///
/// ```
/// use tensorlogic_compiler::passes::validate_expression;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Expression with unbound variables
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
///
/// let result = validate_expression(&expr);
/// assert!(result.has_errors());
/// assert!(result.error_count >= 2); // x and y unbound
/// ```
pub fn validate_expression(expr: &TLExpr) -> ValidationResult {
    let mut diagnostics = Vec::new();
    let mut error_count = 0;
    let mut warning_count = 0;

    // 1. Arity validation
    if let Err(e) = validate_arity(expr) {
        diagnostics.push(format!("Arity error: {}", e));
        error_count += 1;
    }

    // 2. Scope analysis
    match analyze_scopes(expr) {
        Ok(scope_result) => {
            // Check for unbound variables
            if !scope_result.unbound_variables.is_empty() {
                for var in &scope_result.unbound_variables {
                    diagnostics.push(format!("Unbound variable: '{}'", var));
                    error_count += 1;
                }

                // Provide helpful suggestion
                if let Ok(suggestions) = suggest_quantifiers(expr) {
                    if !suggestions.is_empty() {
                        diagnostics.push(format!("Suggestion: {}", suggestions.join(", ")));
                    }
                }
            }

            // Check for type conflicts
            for conflict in &scope_result.type_conflicts {
                diagnostics.push(format!(
                    "Type conflict: variable '{}' has conflicting types '{}' and '{}'",
                    conflict.variable, conflict.type1, conflict.type2
                ));
                error_count += 1;
            }
        }
        Err(e) => {
            diagnostics.push(format!("Scope analysis error: {}", e));
            error_count += 1;
        }
    }

    // 3. Enhanced diagnostics (warnings and hints)
    let diag_messages = diagnose_expression(expr);
    for diag in diag_messages {
        let formatted = diag.format();
        match diag.level {
            DiagnosticLevel::Error => {
                // Skip if we already reported this error above
                if !diagnostics.iter().any(|d| d.contains(&diag.message)) {
                    diagnostics.push(formatted);
                    error_count += 1;
                }
            }
            DiagnosticLevel::Warning => {
                diagnostics.push(formatted);
                warning_count += 1;
            }
            DiagnosticLevel::Info | DiagnosticLevel::Hint => {
                diagnostics.push(formatted);
            }
        }
    }

    ValidationResult {
        passed: error_count == 0,
        error_count,
        warning_count,
        diagnostics,
    }
}

/// Validates an expression with type signatures.
///
/// This is an extended validation that includes type checking against
/// registered predicate signatures.
///
/// # Arguments
///
/// * `expr` - The expression to validate
/// * `signatures` - Predicate signatures for type checking
///
/// # Returns
///
/// A `ValidationResult` with type checking errors included.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::passes::validate_expression_with_types;
/// use tensorlogic_ir::{PredicateSignature, TLExpr, Term, TypeAnnotation};
///
/// let signatures = vec![
///     PredicateSignature::new(
///         "knows",
///         vec![
///             TypeAnnotation { type_name: "Person".to_string() },
///             TypeAnnotation { type_name: "Person".to_string() },
///         ],
///     )
/// ];
///
/// // Fully quantified expression
/// let expr = TLExpr::exists(
///     "x",
///     "Person",
///     TLExpr::exists(
///         "y",
///         "Person",
///         TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
///     ),
/// );
///
/// let result = validate_expression_with_types(&expr, &signatures);
/// assert!(result.is_ok());
/// ```
pub fn validate_expression_with_types(
    expr: &TLExpr,
    signatures: &[PredicateSignature],
) -> ValidationResult {
    let mut result = validate_expression(expr);

    // Add type checking
    use tensorlogic_ir::SignatureRegistry;
    let mut registry = SignatureRegistry::new();
    for sig in signatures {
        registry.register(sig.clone());
    }

    let checker = TypeChecker::new(registry);
    if let Err(e) = checker.check_expr(expr) {
        result.diagnostics.push(format!("Type error: {}", e));
        result.error_count += 1;
        result.passed = false;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_validate_expression_ok() {
        // Fully quantified expression with no unbound variables
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::exists(
                "y",
                "Person",
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
            ),
        );

        let result = validate_expression(&expr);
        if !result.is_ok() {
            eprintln!("Validation failed with errors:");
            for diag in &result.diagnostics {
                eprintln!("  - {}", diag);
            }
        }
        assert!(result.is_ok());
        assert_eq!(result.error_count, 0);
    }

    #[test]
    fn test_validate_expression_partial_binding() {
        // Expression where y is bound but x is not
        let expr = TLExpr::exists(
            "y",
            "Person",
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        );

        let result = validate_expression(&expr);
        eprintln!(
            "Error count: {}, diagnostics: {:?}",
            result.error_count, result.diagnostics
        );
        assert!(result.has_errors());
        // Expected: 1 error for unbound x, but diagnose_expression also reports it
        // So we get 2 total (1 from scope analysis, 1 from diagnostics module)
        assert!(result.error_count >= 1); // At least 1 for unbound x
    }

    #[test]
    fn test_validate_expression_unbound_vars() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        let result = validate_expression(&expr);
        assert!(result.has_errors());
        // Both x and y are unbound - at least 2 errors
        assert!(result.error_count >= 2);
    }

    #[test]
    fn test_validate_expression_arity_mismatch() {
        let expr = TLExpr::and(
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
            TLExpr::pred("knows", vec![Term::var("z")]),
        );

        let result = validate_expression(&expr);
        assert!(result.has_errors());
        assert!(result.diagnostics.iter().any(|d| d.contains("Arity")));
    }

    #[test]
    fn test_validate_expression_with_warnings() {
        // Expression with unused binding
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::pred("p", vec![Term::var("y")]), // x not used
        );

        let result = validate_expression(&expr);
        assert!(result.warning_count > 0);
    }

    #[test]
    fn test_validate_with_types() {
        use tensorlogic_ir::TypeAnnotation;

        let signatures = vec![PredicateSignature {
            name: "knows".to_string(),
            arity: 2,
            arg_types: vec![
                TypeAnnotation {
                    type_name: "Person".to_string(),
                },
                TypeAnnotation {
                    type_name: "Person".to_string(),
                },
            ],
            parametric_types: None,
        }];

        // Fully quantified expression
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::exists(
                "y",
                "Person",
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
            ),
        );

        let result = validate_expression_with_types(&expr, &signatures);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_result_message() {
        let expr = TLExpr::pred("knows", vec![Term::var("x")]);

        let result = validate_expression(&expr);
        let message = result.error_message();
        assert!(!message.is_empty());
        assert!(message.contains("Unbound"));
    }
}
