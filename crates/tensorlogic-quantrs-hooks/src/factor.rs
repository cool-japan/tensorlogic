//! Factor representation and operations.

use scirs2_core::ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::error::{PgmError, Result};

/// A factor in a probabilistic graphical model.
///
/// Represents a function over a subset of variables: φ(X₁, X₂, ..., Xₖ) → ℝ⁺
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Factor {
    /// Variables this factor depends on
    pub variables: Vec<String>,
    /// Probability/potential values
    pub values: ArrayD<f64>,
    /// Factor name for debugging
    pub name: String,
}

impl Factor {
    /// Create a new factor.
    pub fn new(name: String, variables: Vec<String>, values: ArrayD<f64>) -> Result<Self> {
        // Validate dimensions match number of variables
        if values.ndim() != variables.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![variables.len()],
                got: vec![values.ndim()],
            });
        }

        Ok(Self {
            name,
            variables,
            values,
        })
    }

    /// Create a uniform factor.
    pub fn uniform(name: String, variables: Vec<String>, card: usize) -> Self {
        let shape = vec![card; variables.len()];
        let values = ArrayD::from_elem(shape, 1.0 / (card.pow(variables.len() as u32) as f64));
        Self {
            name,
            variables,
            values,
        }
    }

    /// Normalize factor to sum to 1.
    pub fn normalize(&mut self) {
        let sum: f64 = self.values.iter().sum();
        if sum > 0.0 {
            self.values /= sum;
        }
    }

    /// Get cardinality of a variable.
    pub fn get_cardinality(&self, var: &str) -> Option<usize> {
        self.variables
            .iter()
            .position(|v| v == var)
            .map(|idx| self.values.shape()[idx])
    }
}

/// Operations on factors.
pub enum FactorOp {
    /// Product of factors
    Product,
    /// Sum over variables
    Marginalize,
    /// Divide factors
    Divide,
}

impl Factor {
    /// Compute the product of two factors.
    ///
    /// φ₁(X₁) * φ₂(X₂) = φ(X₁ ∪ X₂)
    pub fn product(&self, other: &Factor) -> Result<Factor> {
        // Find union of variables
        let mut all_vars = self.variables.clone();
        for v in &other.variables {
            if !all_vars.contains(v) {
                all_vars.push(v.clone());
            }
        }

        // Build shape and index mappings
        let mut shape = Vec::new();
        let mut self_mapping = Vec::new(); // Maps result dims to self dims
        let mut other_mapping = Vec::new(); // Maps result dims to other dims

        for var in &all_vars {
            // Find the variable in both factors
            let self_idx_opt = self.variables.iter().position(|v| v == var);
            let other_idx_opt = other.variables.iter().position(|v| v == var);

            let cardinality = if let Some(self_idx) = self_idx_opt {
                self_mapping.push(Some(self_idx));
                self.values.shape()[self_idx]
            } else if let Some(other_idx) = other_idx_opt {
                self_mapping.push(None);
                other.values.shape()[other_idx]
            } else {
                unreachable!("Variable must be in at least one factor");
            };

            if let Some(other_idx) = other_idx_opt {
                other_mapping.push(Some(other_idx));
            } else {
                other_mapping.push(None);
            }

            shape.push(cardinality);
        }

        // Compute product
        let mut result_values = ArrayD::zeros(shape.clone());
        let total_size: usize = shape.iter().product();

        for linear_idx in 0..total_size {
            // Convert linear index to multi-dimensional assignment
            let mut assignment = Vec::new();
            let mut temp_idx = linear_idx;
            for &dim in shape.iter().rev() {
                assignment.push(temp_idx % dim);
                temp_idx /= dim;
            }
            assignment.reverse();

            // Map to indices for self and other
            let self_idx: Vec<usize> = self_mapping
                .iter()
                .enumerate()
                .filter_map(|(i, &opt)| opt.map(|_| assignment[i]))
                .collect();

            let other_idx: Vec<usize> = other_mapping
                .iter()
                .enumerate()
                .filter_map(|(i, &opt)| opt.map(|_| assignment[i]))
                .collect();

            // Get values
            let self_val = if self_idx.len() == self.variables.len() {
                self.values[self_idx.as_slice()]
            } else {
                1.0
            };

            let other_val = if other_idx.len() == other.variables.len() {
                other.values[other_idx.as_slice()]
            } else {
                1.0
            };

            result_values[assignment.as_slice()] = self_val * other_val;
        }

        Ok(Factor {
            name: format!("{}*{}", self.name, other.name),
            variables: all_vars,
            values: result_values,
        })
    }

    /// Marginalize out a variable by summing over it.
    ///
    /// ∑ₓ φ(X, Y) = φ(Y)
    pub fn marginalize_out(&self, var: &str) -> Result<Factor> {
        use scirs2_core::ndarray::Axis;

        // Find variable index
        let var_idx = self
            .variables
            .iter()
            .position(|v| v == var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        // Sum over the axis
        let new_values = self.values.sum_axis(Axis(var_idx));

        // Remove variable from list
        let new_vars: Vec<String> = self
            .variables
            .iter()
            .filter(|v| *v != var)
            .cloned()
            .collect();

        Ok(Factor {
            name: format!("{}_marg", self.name),
            variables: new_vars,
            values: new_values,
        })
    }

    /// Marginalize out multiple variables.
    pub fn marginalize_out_vars(&self, vars: &[String]) -> Result<Factor> {
        let mut result = self.clone();
        for var in vars {
            result = result.marginalize_out(var)?;
        }
        Ok(result)
    }

    /// Marginalize out all variables except the specified ones.
    ///
    /// This is useful for extracting marginals: to get P(X), marginalize out all variables except X.
    pub fn marginalize_out_all_except(&self, keep_vars: &[String]) -> Result<Factor> {
        let vars_to_remove: Vec<String> = self
            .variables
            .iter()
            .filter(|v| !keep_vars.contains(v))
            .cloned()
            .collect();

        self.marginalize_out_vars(&vars_to_remove)
    }

    /// Maximize out a variable (for max-product algorithm).
    ///
    /// max_x φ(X, Y) = φ(Y) where φ(Y) = max over X
    pub fn maximize_out(&self, var: &str) -> Result<Factor> {
        use scirs2_core::ndarray::Axis;

        // Find variable index
        let var_idx = self
            .variables
            .iter()
            .position(|v| v == var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        // Take max over the axis
        let new_values = self.values.map_axis(Axis(var_idx), |view| {
            view.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
        });

        // Remove variable from list
        let new_vars: Vec<String> = self
            .variables
            .iter()
            .filter(|v| *v != var)
            .cloned()
            .collect();

        Ok(Factor {
            name: format!("{}_max", self.name),
            variables: new_vars,
            values: new_values,
        })
    }

    /// Maximize out multiple variables.
    pub fn maximize_out_vars(&self, vars: &[String]) -> Result<Factor> {
        let mut result = self.clone();
        for var in vars {
            result = result.maximize_out(var)?;
        }
        Ok(result)
    }

    /// Divide this factor by another factor.
    ///
    /// φ₁(X) / φ₂(X) - used for message division
    pub fn divide(&self, other: &Factor) -> Result<Factor> {
        // Variables must match
        if self.variables != other.variables {
            return Err(PgmError::InvalidDistribution(
                "Cannot divide factors with different variables".to_string(),
            ));
        }

        // Perform element-wise division with safeguard
        let result_values = &self.values
            / &other
                .values
                .mapv(|x| if x.abs() < 1e-10 { 1e-10 } else { x });

        Ok(Factor {
            name: format!("{}/{}", self.name, other.name),
            variables: self.variables.clone(),
            values: result_values,
        })
    }

    /// Reduce factor to specific variable assignment (evidence).
    pub fn reduce(&self, var: &str, value: usize) -> Result<Factor> {
        use scirs2_core::ndarray::Axis;

        let var_idx = self
            .variables
            .iter()
            .position(|v| v == var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        // Check bounds
        if value >= self.values.shape()[var_idx] {
            return Err(PgmError::InvalidDistribution(format!(
                "Value {} out of bounds for variable {} with cardinality {}",
                value,
                var,
                self.values.shape()[var_idx]
            )));
        }

        // Slice at the given value
        let new_values = self.values.index_axis(Axis(var_idx), value).to_owned();

        // Remove variable
        let new_vars: Vec<String> = self
            .variables
            .iter()
            .filter(|v| *v != var)
            .cloned()
            .collect();

        Ok(Factor {
            name: format!("{}_reduced", self.name),
            variables: new_vars,
            values: new_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_factor_creation() {
        let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        assert_eq!(factor.variables.len(), 2);
        assert_eq!(factor.values.ndim(), 2);
    }

    #[test]
    fn test_factor_normalize() {
        let values = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn();
        let mut factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        factor.normalize();
        let sum: f64 = factor.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_factor() {
        let factor = Factor::uniform("f1".to_string(), vec!["x".to_string()], 3);
        assert_eq!(factor.values.len(), 3);
        let sum: f64 = factor.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_factor_product() {
        // φ₁(X) and φ₂(Y) → φ(X,Y)
        let f1_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let f1 = Factor::new("f1".to_string(), vec!["x".to_string()], f1_values).unwrap();

        let f2_values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
            .unwrap()
            .into_dyn();
        let f2 = Factor::new("f2".to_string(), vec!["y".to_string()], f2_values).unwrap();

        let product = f1.product(&f2).unwrap();
        assert_eq!(product.variables.len(), 2);
        assert_eq!(product.values.shape(), &[2, 2]);

        // Check values: [0.6*0.7, 0.6*0.3, 0.4*0.7, 0.4*0.3]
        let expected = 0.6 * 0.7 + 0.6 * 0.3 + 0.4 * 0.7 + 0.4 * 0.3;
        let actual: f64 = product.values.iter().sum();
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn test_factor_marginalize() {
        // φ(X,Y) → φ(X)
        let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        let marginal = factor.marginalize_out("y").unwrap();
        assert_eq!(marginal.variables.len(), 1);
        assert_eq!(marginal.variables[0], "x");
        assert_eq!(marginal.values.shape(), &[2]);

        // Sum over Y: [0.1+0.2, 0.3+0.4] = [0.3, 0.7]
        assert!((marginal.values[[0]] - 0.3).abs() < 1e-10);
        assert!((marginal.values[[1]] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_factor_divide() {
        let values1 = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let f1 = Factor::new("f1".to_string(), vec!["x".to_string()], values1).unwrap();

        let values2 = Array::from_shape_vec(vec![2], vec![0.3, 0.2])
            .unwrap()
            .into_dyn();
        let f2 = Factor::new("f2".to_string(), vec!["x".to_string()], values2).unwrap();

        let result = f1.divide(&f2).unwrap();
        assert_eq!(result.variables.len(), 1);

        // 0.6/0.3 = 2.0, 0.4/0.2 = 2.0
        assert!((result.values[[0]] - 2.0).abs() < 1e-10);
        assert!((result.values[[1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_factor_reduce() {
        // φ(X,Y) with evidence Y=1 → φ(X)
        let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        let reduced = factor.reduce("y", 1).unwrap();
        assert_eq!(reduced.variables.len(), 1);
        assert_eq!(reduced.variables[0], "x");

        // Y=1 slice: [0.2, 0.4]
        assert!((reduced.values[[0]] - 0.2).abs() < 1e-10);
        assert!((reduced.values[[1]] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_factor_product_with_shared_vars() {
        // φ₁(X,Y) and φ₂(Y,Z) → φ(X,Y,Z)
        let f1_values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let f1 = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            f1_values,
        )
        .unwrap();

        let f2_values = Array::from_shape_vec(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5])
            .unwrap()
            .into_dyn();
        let f2 = Factor::new(
            "f2".to_string(),
            vec!["y".to_string(), "z".to_string()],
            f2_values,
        )
        .unwrap();

        let product = f1.product(&f2).unwrap();
        assert_eq!(product.variables.len(), 3);
        assert!(product.variables.contains(&"x".to_string()));
        assert!(product.variables.contains(&"y".to_string()));
        assert!(product.variables.contains(&"z".to_string()));
    }

    #[test]
    fn test_factor_maximize() {
        // φ(X,Y) → max_Y φ(X)
        let values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string()],
            values,
        )
        .unwrap();

        let maximized = factor.maximize_out("y").unwrap();
        assert_eq!(maximized.variables.len(), 1);
        assert_eq!(maximized.variables[0], "x");
        assert_eq!(maximized.values.shape(), &[2]);

        // Max over Y: [max(0.1, 0.2), max(0.3, 0.4)] = [0.2, 0.4]
        assert!((maximized.values[[0]] - 0.2).abs() < 1e-10);
        assert!((maximized.values[[1]] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_factor_maximize_multiple() {
        // φ(X,Y,Z) → max_{Y,Z} φ(X)
        let values =
            Array::from_shape_vec(vec![2, 2, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap()
                .into_dyn();
        let factor = Factor::new(
            "f1".to_string(),
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            values,
        )
        .unwrap();

        let maximized = factor
            .maximize_out_vars(&["y".to_string(), "z".to_string()])
            .unwrap();
        assert_eq!(maximized.variables.len(), 1);
        assert_eq!(maximized.variables[0], "x");
    }
}
