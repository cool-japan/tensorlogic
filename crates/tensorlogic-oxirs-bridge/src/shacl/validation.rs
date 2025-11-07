//! SHACL validation report generation
//!
//! This module provides functionality to generate SHACL validation reports
//! that conform to the SHACL specification.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SHACL validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Violation - constraint must be satisfied
    Violation,
    /// Warning - constraint should be satisfied
    Warning,
    /// Info - informational message
    Info,
}

impl ValidationSeverity {
    pub fn to_iri(&self) -> &'static str {
        match self {
            ValidationSeverity::Violation => "http://www.w3.org/ns/shacl#Violation",
            ValidationSeverity::Warning => "http://www.w3.org/ns/shacl#Warning",
            ValidationSeverity::Info => "http://www.w3.org/ns/shacl#Info",
        }
    }
}

/// A single validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// The type of validation result (sh:ValidationResult)
    pub result_type: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// The focus node that was validated
    pub focus_node: String,
    /// The property path that was validated (optional)
    pub result_path: Option<String>,
    /// The value that caused the violation (optional)
    pub value: Option<String>,
    /// The shape that was not satisfied
    pub source_shape: String,
    /// The constraint component that failed
    pub source_constraint_component: String,
    /// Human-readable message
    pub message: String,
}

impl ValidationResult {
    pub fn new(
        focus_node: impl Into<String>,
        source_shape: impl Into<String>,
        constraint_component: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        ValidationResult {
            result_type: "http://www.w3.org/ns/shacl#ValidationResult".to_string(),
            severity: ValidationSeverity::Violation,
            focus_node: focus_node.into(),
            result_path: None,
            value: None,
            source_shape: source_shape.into(),
            source_constraint_component: constraint_component.into(),
            message: message.into(),
        }
    }

    pub fn with_severity(mut self, severity: ValidationSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.result_path = Some(path.into());
        self
    }

    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Convert to RDF Turtle format
    pub fn to_turtle(&self) -> String {
        let mut turtle = String::new();
        turtle.push_str("[\n");
        turtle.push_str("  a sh:ValidationResult ;\n");
        turtle.push_str(&format!(
            "  sh:resultSeverity {} ;\n",
            self.severity.to_iri()
        ));
        turtle.push_str(&format!("  sh:focusNode <{}> ;\n", self.focus_node));

        if let Some(ref path) = self.result_path {
            turtle.push_str(&format!("  sh:resultPath <{}> ;\n", path));
        }

        if let Some(ref value) = self.value {
            turtle.push_str(&format!("  sh:value \"{}\" ;\n", value));
        }

        turtle.push_str(&format!("  sh:sourceShape <{}> ;\n", self.source_shape));
        turtle.push_str(&format!(
            "  sh:sourceConstraintComponent <{}> ;\n",
            self.source_constraint_component
        ));
        turtle.push_str(&format!("  sh:resultMessage \"{}\" .\n", self.message));
        turtle.push(']');

        turtle
    }
}

/// SHACL validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the data conforms to all shapes
    pub conforms: bool,
    /// List of validation results
    pub results: Vec<ValidationResult>,
    /// Statistics about the validation
    pub statistics: ValidationStatistics,
}

/// Statistics about validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub total_shapes: usize,
    pub total_constraints: usize,
    pub violations: usize,
    pub warnings: usize,
    pub infos: usize,
}

impl ValidationReport {
    pub fn new() -> Self {
        ValidationReport {
            conforms: true,
            results: Vec::new(),
            statistics: ValidationStatistics::default(),
        }
    }

    pub fn add_result(&mut self, result: ValidationResult) {
        if result.severity == ValidationSeverity::Violation {
            self.conforms = false;
            self.statistics.violations += 1;
        } else if result.severity == ValidationSeverity::Warning {
            self.statistics.warnings += 1;
        } else {
            self.statistics.infos += 1;
        }

        self.results.push(result);
    }

    pub fn set_statistics(&mut self, total_shapes: usize, total_constraints: usize) {
        self.statistics.total_shapes = total_shapes;
        self.statistics.total_constraints = total_constraints;
    }

    /// Convert to RDF Turtle format
    pub fn to_turtle(&self) -> String {
        let mut turtle = String::new();

        // Prefixes
        turtle.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Validation report
        turtle.push_str("[\n");
        turtle.push_str("  a sh:ValidationReport ;\n");
        turtle.push_str(&format!("  sh:conforms {} ;\n", self.conforms));

        if !self.results.is_empty() {
            turtle.push_str("  sh:result\n");
            for (i, result) in self.results.iter().enumerate() {
                turtle.push_str("    ");
                turtle.push_str(&result.to_turtle());
                if i < self.results.len() - 1 {
                    turtle.push_str(" ,\n");
                } else {
                    turtle.push_str(" .\n");
                }
            }
        }

        turtle.push_str("] .\n");

        turtle
    }

    /// Convert to JSON format
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Summary of the validation report
    pub fn summary(&self) -> String {
        format!(
            "Validation Report: {} - {} violations, {} warnings, {} infos (checked {} shapes with {} constraints)",
            if self.conforms { "CONFORMS" } else { "VIOLATIONS" },
            self.statistics.violations,
            self.statistics.warnings,
            self.statistics.infos,
            self.statistics.total_shapes,
            self.statistics.total_constraints
        )
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Validator that checks tensor outputs against SHACL constraints
pub struct ShaclValidator {
    /// Mapping from constraint predicates to their expected behavior
    #[allow(dead_code, clippy::type_complexity)]
    constraint_handlers: HashMap<String, Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl ShaclValidator {
    pub fn new() -> Self {
        ShaclValidator {
            constraint_handlers: HashMap::new(),
        }
    }

    /// Validate tensor outputs and generate a report
    ///
    /// This is a placeholder that demonstrates the structure.
    /// In a real implementation, this would:
    /// 1. Take tensor computation results
    /// 2. Check each constraint against the results
    /// 3. Generate validation results for failures
    pub fn validate_mock(
        &self,
        shape_name: &str,
        focus_nodes: &[&str],
        constraint_checks: &[(String, bool)],
    ) -> ValidationReport {
        let mut report = ValidationReport::new();

        report.set_statistics(1, constraint_checks.len());

        for (constraint, passed) in constraint_checks {
            if !passed {
                for focus_node in focus_nodes {
                    let result = ValidationResult::new(
                        *focus_node,
                        shape_name,
                        constraint,
                        format!("Constraint '{}' failed for node {}", constraint, focus_node),
                    );
                    report.add_result(result);
                }
            }
        }

        report
    }

    /// Validate a specific constraint type
    pub fn validate_min_count(
        &self,
        focus_node: &str,
        property: &str,
        min_count: usize,
        actual_count: usize,
    ) -> Option<ValidationResult> {
        if actual_count < min_count {
            Some(
                ValidationResult::new(
                    focus_node,
                    format!("Shape_{}", property),
                    "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
                    format!(
                        "Property {} has {} values but requires at least {}",
                        property, actual_count, min_count
                    ),
                )
                .with_path(property),
            )
        } else {
            None
        }
    }

    /// Validate a specific constraint type
    pub fn validate_max_count(
        &self,
        focus_node: &str,
        property: &str,
        max_count: usize,
        actual_count: usize,
    ) -> Option<ValidationResult> {
        if actual_count > max_count {
            Some(
                ValidationResult::new(
                    focus_node,
                    format!("Shape_{}", property),
                    "http://www.w3.org/ns/shacl#MaxCountConstraintComponent",
                    format!(
                        "Property {} has {} values but allows at most {}",
                        property, actual_count, max_count
                    ),
                )
                .with_path(property),
            )
        } else {
            None
        }
    }

    /// Validate datatype constraint
    pub fn validate_datatype(
        &self,
        focus_node: &str,
        property: &str,
        value: &str,
        expected_type: &str,
        actual_type: &str,
    ) -> Option<ValidationResult> {
        if expected_type != actual_type {
            Some(
                ValidationResult::new(
                    focus_node,
                    format!("Shape_{}", property),
                    "http://www.w3.org/ns/shacl#DatatypeConstraintComponent",
                    format!(
                        "Value '{}' has type {} but expected type {}",
                        value, actual_type, expected_type
                    ),
                )
                .with_path(property)
                .with_value(value),
            )
        } else {
            None
        }
    }

    /// Validate pattern constraint
    pub fn validate_pattern(
        &self,
        focus_node: &str,
        property: &str,
        value: &str,
        pattern: &str,
    ) -> Option<ValidationResult> {
        // In a real implementation, use regex matching
        // For now, simple contains check
        if !value.contains('@') && pattern.contains('@') {
            Some(
                ValidationResult::new(
                    focus_node,
                    format!("Shape_{}", property),
                    "http://www.w3.org/ns/shacl#PatternConstraintComponent",
                    format!("Value '{}' does not match pattern '{}'", value, pattern),
                )
                .with_path(property)
                .with_value(value),
            )
        } else {
            None
        }
    }
}

impl Default for ShaclValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Missing required property",
        );

        assert_eq!(result.focus_node, "http://example.org/person/1");
        assert_eq!(result.severity, ValidationSeverity::Violation);
    }

    #[test]
    fn test_validation_report_conforms() {
        let report = ValidationReport::new();
        assert!(report.conforms);
        assert_eq!(report.results.len(), 0);
    }

    #[test]
    fn test_validation_report_with_violation() {
        let mut report = ValidationReport::new();

        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Missing required property",
        );

        report.add_result(result);

        assert!(!report.conforms);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.statistics.violations, 1);
    }

    #[test]
    fn test_validation_report_with_warning() {
        let mut report = ValidationReport::new();

        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#PatternConstraintComponent",
            "Recommended pattern not matched",
        )
        .with_severity(ValidationSeverity::Warning);

        report.add_result(result);

        assert!(report.conforms); // Still conforms with warnings
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.statistics.warnings, 1);
    }

    #[test]
    fn test_validation_result_to_turtle() {
        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Missing required property",
        )
        .with_path("http://example.org/name")
        .with_value("invalid");

        let turtle = result.to_turtle();

        assert!(turtle.contains("sh:ValidationResult"));
        assert!(turtle.contains("sh:focusNode"));
        assert!(turtle.contains("sh:resultPath"));
        assert!(turtle.contains("sh:value"));
    }

    #[test]
    fn test_validation_report_to_turtle() {
        let mut report = ValidationReport::new();

        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Missing required property",
        );

        report.add_result(result);
        report.set_statistics(1, 5);

        let turtle = report.to_turtle();

        assert!(turtle.contains("@prefix sh:"));
        assert!(turtle.contains("sh:ValidationReport"));
        assert!(turtle.contains("sh:conforms false"));
        assert!(turtle.contains("sh:result"));
    }

    #[test]
    fn test_validation_report_to_json() {
        let mut report = ValidationReport::new();

        let result = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Missing required property",
        );

        report.add_result(result);

        let json = report.to_json().unwrap();
        assert!(json.contains("conforms"));
        assert!(json.contains("results"));
    }

    #[test]
    fn test_shacl_validator_min_count() {
        let validator = ShaclValidator::new();

        // Valid case
        let result = validator.validate_min_count("http://example.org/person/1", "name", 1, 2);
        assert!(result.is_none());

        // Invalid case
        let result = validator.validate_min_count("http://example.org/person/1", "name", 1, 0);
        assert!(result.is_some());
    }

    #[test]
    fn test_shacl_validator_max_count() {
        let validator = ShaclValidator::new();

        // Valid case
        let result = validator.validate_max_count("http://example.org/person/1", "name", 2, 1);
        assert!(result.is_none());

        // Invalid case
        let result = validator.validate_max_count("http://example.org/person/1", "name", 1, 2);
        assert!(result.is_some());
    }

    #[test]
    fn test_validation_report_summary() {
        let mut report = ValidationReport::new();
        report.set_statistics(3, 10);

        let result1 = ValidationResult::new(
            "http://example.org/person/1",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            "Violation 1",
        );

        let result2 = ValidationResult::new(
            "http://example.org/person/2",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#PatternConstraintComponent",
            "Warning 1",
        )
        .with_severity(ValidationSeverity::Warning);

        report.add_result(result1);
        report.add_result(result2);

        let summary = report.summary();
        assert!(summary.contains("VIOLATIONS"));
        assert!(summary.contains("1 violations"));
        assert!(summary.contains("1 warnings"));
        assert!(summary.contains("3 shapes"));
        assert!(summary.contains("10 constraints"));
    }
}
