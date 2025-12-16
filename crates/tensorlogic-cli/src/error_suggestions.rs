//! Error handling with helpful suggestions
//!
//! Provides user-friendly error messages with actionable suggestions.

#![allow(dead_code)]

use anyhow::Result;
use std::fmt;

/// Error with suggestions for resolution
#[derive(Debug)]
pub struct ErrorWithSuggestions {
    pub error: String,
    pub suggestions: Vec<String>,
    pub examples: Vec<String>,
}

impl fmt::Display for ErrorWithSuggestions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error: {}", self.error)?;

        if !self.suggestions.is_empty() {
            writeln!(f, "\nSuggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  • {}", suggestion)?;
            }
        }

        if !self.examples.is_empty() {
            writeln!(f, "\nExamples:")?;
            for example in &self.examples {
                writeln!(f, "  {}", example)?;
            }
        }

        Ok(())
    }
}

/// Enhance compilation errors with helpful context
pub fn enhance_compilation_error(error: &str) -> ErrorWithSuggestions {
    let error_lower = error.to_lowercase();

    // Detect common error patterns and provide suggestions
    if error_lower.contains("free variable") || error_lower.contains("unbound") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Add a quantifier (EXISTS or FORALL) for unbound variables".to_string(),
                "Define a domain for the variable using --domains".to_string(),
                "Check variable names for typos".to_string(),
            ],
            examples: vec![
                "EXISTS x IN Person. knows(x, alice)".to_string(),
                "FORALL x IN Person. mortal(x)".to_string(),
                "tensorlogic \"expr\" --domains Person:100".to_string(),
            ],
        }
    } else if error_lower.contains("arity") || error_lower.contains("argument") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Check the number of arguments in predicate calls".to_string(),
                "Verify predicate signatures match their usage".to_string(),
                "Ensure all arguments are properly specified".to_string(),
            ],
            examples: vec![
                "knows(x, y)          # Binary predicate".to_string(),
                "person(x)            # Unary predicate".to_string(),
                "located(x, y, z)     # Ternary predicate".to_string(),
            ],
        }
    } else if error_lower.contains("type") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Check that operations use compatible types".to_string(),
                "Verify arithmetic is only used with numeric expressions".to_string(),
                "Use appropriate comparison operators for the data type".to_string(),
            ],
            examples: vec![
                "age(x) + 10          # Arithmetic on numeric values".to_string(),
                "name(x) = \"alice\"    # String comparison".to_string(),
                "score(x) > threshold # Numeric comparison".to_string(),
            ],
        }
    } else if error_lower.contains("syntax") || error_lower.contains("parse") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Check for unmatched parentheses".to_string(),
                "Verify operator spelling (AND, OR, NOT, IMPLIES)".to_string(),
                "Use quotes for string literals".to_string(),
                "Ensure quantifiers have IN domain clause".to_string(),
            ],
            examples: vec![
                "(p AND q) OR r       # Grouped expression".to_string(),
                "EXISTS x IN D. p(x)  # Quantifier with domain".to_string(),
                "knows(\"alice\", bob)  # String literal in quotes".to_string(),
            ],
        }
    } else if error_lower.contains("domain") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Define the domain using --domains option".to_string(),
                "Check domain name spelling in quantifiers".to_string(),
                "Ensure domain size is positive".to_string(),
            ],
            examples: vec![
                "tensorlogic expr --domains Person:100".to_string(),
                "tensorlogic expr --domains User:1000 --domains Item:5000".to_string(),
                "EXISTS x IN Person. knows(x, y)  # Domain must be defined".to_string(),
            ],
        }
    } else if error_lower.contains("validation") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Check that all inputs and outputs are properly connected".to_string(),
                "Verify that the graph structure is valid".to_string(),
                "Ensure all tensors have producers if required".to_string(),
                "Use --debug to see detailed graph structure".to_string(),
            ],
            examples: vec![
                "tensorlogic expr --debug      # Show detailed compilation info".to_string(),
                "tensorlogic expr --no-validate # Skip validation".to_string(),
            ],
        }
    } else if error_lower.contains("strategy") {
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Use one of the 6 valid compilation strategies".to_string(),
                "Check strategy name spelling".to_string(),
            ],
            examples: vec![
                "--strategy soft_differentiable   # For neural training".to_string(),
                "--strategy hard_boolean          # For discrete logic".to_string(),
                "--strategy fuzzy_godel           # For Gödel fuzzy logic".to_string(),
                "--strategy fuzzy_product         # For product fuzzy logic".to_string(),
                "--strategy fuzzy_lukasiewicz     # For Łukasiewicz logic".to_string(),
                "--strategy probabilistic         # For probabilities".to_string(),
            ],
        }
    } else {
        // Generic error with general suggestions
        ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Use --debug flag to see detailed error information".to_string(),
                "Check the expression syntax and formatting".to_string(),
                "Consult the documentation for correct usage".to_string(),
            ],
            examples: vec![
                "tensorlogic expr --debug".to_string(),
                "tensorlogic --help".to_string(),
            ],
        }
    }
}

/// Enhance file operation errors
pub fn enhance_file_error(path: &str, error: &str) -> ErrorWithSuggestions {
    let error_lower = error.to_lowercase();

    if error_lower.contains("not found") || error_lower.contains("no such") {
        ErrorWithSuggestions {
            error: format!("File not found: {}", path),
            suggestions: vec![
                "Check the file path spelling and location".to_string(),
                "Use absolute path or relative path from current directory".to_string(),
                "Verify the file exists using ls or find command".to_string(),
            ],
            examples: vec![
                format!("ls {}", path),
                format!(
                    "find . -name \"{}\"",
                    path.rsplit('/').next().unwrap_or(path)
                ),
            ],
        }
    } else if error_lower.contains("permission") {
        ErrorWithSuggestions {
            error: format!("Permission denied: {}", path),
            suggestions: vec![
                "Check file permissions".to_string(),
                "Ensure you have read access to the file".to_string(),
                "Try using sudo if appropriate".to_string(),
            ],
            examples: vec![format!("ls -l {}", path), format!("chmod +r {}", path)],
        }
    } else {
        ErrorWithSuggestions {
            error: format!("File error for {}: {}", path, error),
            suggestions: vec![
                "Check file permissions and accessibility".to_string(),
                "Verify disk space is available".to_string(),
            ],
            examples: vec!["df -h".to_string()],
        }
    }
}

/// Provide helpful error context for common CLI mistakes
pub fn suggest_for_cli_args(error: &str) -> Result<()> {
    let error_lower = error.to_lowercase();

    if error_lower.contains("unexpected argument") || error_lower.contains("found argument") {
        let suggestions = ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Check argument spelling and dashes (- vs --)".to_string(),
                "Use --help to see all available arguments".to_string(),
                "Some arguments require values (e.g., --domains Person:100)".to_string(),
            ],
            examples: vec![
                "tensorlogic --help".to_string(),
                "tensorlogic expr --domains Person:100  # Correct".to_string(),
                "tensorlogic expr --domain Person:100   # Wrong (should be --domains)".to_string(),
            ],
        };

        Err(anyhow::anyhow!("{}", suggestions))
    } else if error_lower.contains("required") {
        let suggestions = ErrorWithSuggestions {
            error: error.to_string(),
            suggestions: vec![
                "Provide the required input expression or file".to_string(),
                "Use a subcommand (repl, batch, etc.) or provide input".to_string(),
            ],
            examples: vec![
                "tensorlogic \"knows(x, y)\"        # Direct expression".to_string(),
                "tensorlogic file.tl              # From file".to_string(),
                "tensorlogic repl                 # Interactive mode".to_string(),
            ],
        };

        Err(anyhow::anyhow!("{}", suggestions))
    } else {
        Err(anyhow::anyhow!("{}", error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_free_variable_error() {
        let error = enhance_compilation_error("free variable 'x' found in expression");

        assert!(error.error.contains("free variable"));
        assert!(!error.suggestions.is_empty());
        assert!(error.suggestions[0].contains("quantifier"));
        assert!(!error.examples.is_empty());
    }

    #[test]
    fn test_enhance_arity_error() {
        let error = enhance_compilation_error("arity mismatch: expected 2 arguments");

        assert!(error.error.contains("arity"));
        assert!(!error.suggestions.is_empty());
        assert!(!error.examples.is_empty());
    }

    #[test]
    fn test_enhance_strategy_error() {
        let error = enhance_compilation_error("unknown strategy: foo_bar");

        assert!(error.error.contains("strategy"));
        assert!(!error.suggestions.is_empty());
        assert!(error.examples.len() >= 6); // All 6 strategies
    }

    #[test]
    fn test_enhance_file_error() {
        let error = enhance_file_error("/path/to/file.tl", "No such file or directory");

        assert!(error.error.contains("not found"));
        assert!(!error.suggestions.is_empty());
        assert!(!error.examples.is_empty());
    }
}
