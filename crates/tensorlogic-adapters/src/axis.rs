//! Axis metadata for variable-to-dimension mappings.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Axis metadata linking variables to tensor dimensions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxisMetadata {
    pub var_to_axis: IndexMap<String, usize>,
    pub axis_to_domain: IndexMap<usize, String>,
    pub axis_to_char: IndexMap<usize, char>,
}

impl AxisMetadata {
    pub fn new() -> Self {
        AxisMetadata {
            var_to_axis: IndexMap::new(),
            axis_to_domain: IndexMap::new(),
            axis_to_char: IndexMap::new(),
        }
    }

    pub fn assign(&mut self, var: impl Into<String>, domain: impl Into<String>) -> usize {
        let var = var.into();
        let domain = domain.into();

        if let Some(&axis) = self.var_to_axis.get(&var) {
            return axis;
        }

        let axis = self.var_to_axis.len();
        let axis_char = (b'a' + axis as u8) as char;

        self.var_to_axis.insert(var, axis);
        self.axis_to_domain.insert(axis, domain);
        self.axis_to_char.insert(axis, axis_char);

        axis
    }

    pub fn get_axis(&self, var: &str) -> Option<usize> {
        self.var_to_axis.get(var).copied()
    }

    pub fn get_domain(&self, axis: usize) -> Option<&str> {
        self.axis_to_domain.get(&axis).map(|s| s.as_str())
    }

    pub fn get_char(&self, axis: usize) -> Option<char> {
        self.axis_to_char.get(&axis).copied()
    }

    pub fn build_spec(&self, vars: &[String]) -> String {
        vars.iter()
            .filter_map(|v| self.var_to_axis.get(v))
            .filter_map(|&axis| self.axis_to_char.get(&axis))
            .collect()
    }
}

impl Default for AxisMetadata {
    fn default() -> Self {
        Self::new()
    }
}
