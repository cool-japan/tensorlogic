//! Einsum specification parsing and validation.
//!
//! This module provides utilities for parsing and validating Einstein summation
//! notation specifications (e.g., "ij,jk->ik").

use crate::error::IrError;
use std::collections::HashSet;

/// Parsed einsum specification with input and output subscripts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumSpec {
    /// Input subscripts (e.g., ["ij", "jk"] for "ij,jk->ik")
    pub inputs: Vec<Vec<char>>,
    /// Output subscript (e.g., "ik" for "ij,jk->ik")
    pub output: Vec<char>,
    /// All unique indices appearing in the spec
    pub all_indices: HashSet<char>,
    /// Indices that are summed over (appear in inputs but not output)
    pub summed_indices: HashSet<char>,
}

impl EinsumSpec {
    /// Parse an einsum specification string.
    ///
    /// Supports formats:
    /// - Explicit: "ij,jk->ik"
    /// - Implicit (no arrow): "ij,jk" (all indices in output, sorted)
    pub fn parse(spec: &str) -> Result<Self, IrError> {
        if spec.is_empty() {
            return Err(IrError::EmptyEinsumSpec);
        }

        let spec = spec.trim();

        // Check for explicit vs implicit mode
        let (input_part, output_part) = if spec.contains("->") {
            let parts: Vec<&str> = spec.split("->").collect();
            if parts.len() != 2 {
                return Err(IrError::InvalidEinsumSpec {
                    spec: spec.to_string(),
                    reason: "Multiple '->' found".to_string(),
                });
            }
            (parts[0], Some(parts[1]))
        } else {
            (spec, None)
        };

        // Parse input subscripts
        let input_subscripts: Vec<&str> = input_part.split(',').map(|s| s.trim()).collect();
        if input_subscripts.is_empty() {
            return Err(IrError::InvalidEinsumSpec {
                spec: spec.to_string(),
                reason: "No input subscripts found".to_string(),
            });
        }

        let mut inputs = Vec::new();
        let mut all_indices = HashSet::new();

        for subscript in &input_subscripts {
            if subscript.is_empty() {
                return Err(IrError::InvalidEinsumSpec {
                    spec: spec.to_string(),
                    reason: "Empty input subscript".to_string(),
                });
            }

            // Validate that subscript contains only valid characters
            for ch in subscript.chars() {
                if !ch.is_ascii_lowercase() && !ch.is_ascii_uppercase() {
                    return Err(IrError::InvalidEinsumSpec {
                        spec: spec.to_string(),
                        reason: format!("Invalid character '{}' in subscript", ch),
                    });
                }
                all_indices.insert(ch);
            }

            inputs.push(subscript.chars().collect());
        }

        // Parse or infer output subscript
        let output = if let Some(out) = output_part {
            let out = out.trim();
            if out.is_empty() {
                // Empty output means scalar result
                Vec::new()
            } else {
                // Validate output subscript
                for ch in out.chars() {
                    if !ch.is_ascii_lowercase() && !ch.is_ascii_uppercase() {
                        return Err(IrError::InvalidEinsumSpec {
                            spec: spec.to_string(),
                            reason: format!("Invalid character '{}' in output", ch),
                        });
                    }
                }
                out.chars().collect()
            }
        } else {
            // Implicit mode: output contains all unique indices in sorted order
            let mut indices: Vec<char> = all_indices.iter().copied().collect();
            indices.sort();
            indices
        };

        // Compute summed indices
        let output_set: HashSet<char> = output.iter().copied().collect();
        let summed_indices: HashSet<char> = all_indices
            .iter()
            .copied()
            .filter(|ch| !output_set.contains(ch))
            .collect();

        Ok(EinsumSpec {
            inputs,
            output,
            all_indices,
            summed_indices,
        })
    }

    /// Validate the spec against the number of input tensors.
    pub fn validate_input_count(&self, num_inputs: usize) -> Result<(), IrError> {
        if self.inputs.len() != num_inputs {
            return Err(IrError::NodeValidation {
                node: 0,
                message: format!(
                    "Einsum spec expects {} inputs, but {} provided",
                    self.inputs.len(),
                    num_inputs
                ),
            });
        }
        Ok(())
    }

    /// Get the number of input tensors expected by this spec.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of output dimensions.
    pub fn output_ndim(&self) -> usize {
        self.output.len()
    }

    /// Check if this is a reduction (has summed indices).
    pub fn is_reduction(&self) -> bool {
        !self.summed_indices.is_empty()
    }

    /// Check if this is a scalar output (empty output subscript).
    pub fn is_scalar_output(&self) -> bool {
        self.output.is_empty()
    }

    /// Get the canonical spec string representation.
    pub fn canonical_form(&self) -> String {
        let input_part = self
            .inputs
            .iter()
            .map(|sub| sub.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join(",");

        let output_part = self.output.iter().collect::<String>();

        format!("{}->{}", input_part, output_part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_explicit_spec() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.inputs[0], vec!['i', 'j']);
        assert_eq!(spec.inputs[1], vec!['j', 'k']);
        assert_eq!(spec.output, vec!['i', 'k']);

        assert_eq!(spec.all_indices.len(), 3);
        assert!(spec.all_indices.contains(&'i'));
        assert!(spec.all_indices.contains(&'j'));
        assert!(spec.all_indices.contains(&'k'));

        assert_eq!(spec.summed_indices.len(), 1);
        assert!(spec.summed_indices.contains(&'j'));
    }

    #[test]
    fn test_parse_implicit_spec() {
        let spec = EinsumSpec::parse("ij,jk").unwrap();

        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.output, vec!['i', 'j', 'k']); // Sorted
        assert!(!spec.is_reduction()); // No summed indices in implicit mode
    }

    #[test]
    fn test_parse_scalar_output() {
        let spec = EinsumSpec::parse("i,i->").unwrap();

        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.output.len(), 0);
        assert!(spec.is_scalar_output());
        assert!(spec.is_reduction());
    }

    #[test]
    fn test_parse_trace() {
        let spec = EinsumSpec::parse("ii->i").unwrap();

        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.inputs[0], vec!['i', 'i']);
        assert_eq!(spec.output, vec!['i']);
    }

    #[test]
    fn test_parse_batch_matmul() {
        let spec = EinsumSpec::parse("bij,bjk->bik").unwrap();

        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.output, vec!['b', 'i', 'k']);
        assert_eq!(spec.summed_indices.len(), 1);
        assert!(spec.summed_indices.contains(&'j'));
    }

    #[test]
    fn test_parse_empty_spec() {
        let result = EinsumSpec::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_char() {
        let result = EinsumSpec::parse("i1,jk->ik");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_multiple_arrows() {
        let result = EinsumSpec::parse("ij->jk->ik");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_input_count() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        assert!(spec.validate_input_count(2).is_ok());
        assert!(spec.validate_input_count(1).is_err());
        assert!(spec.validate_input_count(3).is_err());
    }

    #[test]
    fn test_spec_properties() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        assert_eq!(spec.num_inputs(), 2);
        assert_eq!(spec.output_ndim(), 2);
        assert!(spec.is_reduction());
        assert!(!spec.is_scalar_output());
    }

    #[test]
    fn test_canonical_form() {
        let original = "ij,jk->ik";
        let spec = EinsumSpec::parse(original).unwrap();
        assert_eq!(spec.canonical_form(), original);

        // Also test Display trait
        assert_eq!(format!("{}", spec), original);
    }
}
