//! Expectation Propagation (EP) for approximate inference.
//!
//! EP is an iterative algorithm that approximates complex posterior distributions
//! using products of simpler "site" approximations via moment matching.
//!
//! # Algorithm
//!
//! 1. Initialize site approximations (factors)
//! 2. For each factor:
//!    a. Compute cavity distribution (remove current site)
//!    b. Compute tilted distribution (include true factor)
//!    c. Moment match to update site approximation
//! 3. Repeat until convergence
//!
//! # References
//!
//! - Minka, "Expectation Propagation for approximate Bayesian inference" (2001)
//! - Bishop, "Pattern Recognition and Machine Learning" (2006), Section 10.7

use crate::{Factor, FactorGraph, PgmError, Result};
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

/// Site approximation for a single factor.
///
/// In EP, each true factor f_i(x) is approximated by a simpler site s_i(x).
/// For discrete distributions, we store the site as a factor.
#[derive(Debug, Clone)]
pub struct Site {
    /// The site approximation (as a factor)
    pub factor: Factor,
    /// Variables this site depends on
    pub variables: Vec<String>,
}

impl Site {
    /// Create a new site initialized to uniform distribution.
    pub fn new_uniform(
        name: String,
        variables: Vec<String>,
        cardinalities: &[usize],
    ) -> Result<Self> {
        let total_size: usize = cardinalities.iter().product();
        let uniform_value = 1.0 / total_size as f64;
        let values = ArrayD::from_elem(cardinalities.to_vec(), uniform_value);

        let factor = Factor::new(name, variables.clone(), values)?;
        Ok(Self { factor, variables })
    }

    /// Create a new site from a factor.
    pub fn from_factor(factor: Factor) -> Self {
        let variables = factor.variables.clone();
        Self { factor, variables }
    }
}

/// Expectation Propagation algorithm for approximate inference.
///
/// EP approximates the posterior distribution by iteratively refining
/// local approximations using moment matching.
pub struct ExpectationPropagation {
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Damping factor (0.0 = no damping, 1.0 = full damping)
    damping: f64,
    /// Minimum value for numerical stability
    min_value: f64,
}

impl Default for ExpectationPropagation {
    fn default() -> Self {
        Self::new(100, 1e-6, 0.0)
    }
}

impl ExpectationPropagation {
    /// Create a new EP algorithm with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64, damping: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping,
            min_value: 1e-10,
        }
    }

    /// Run EP inference on a factor graph.
    ///
    /// Returns the approximate marginal distributions for each variable.
    pub fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize sites
        let mut sites = self.initialize_sites(graph)?;

        // Compute initial approximation
        let mut approx = self.compute_global_approximation(graph, &sites)?;

        // EP iterations
        for iteration in 0..self.max_iterations {
            let mut max_change: f64 = 0.0;

            // Update each site
            for (factor_idx, factor) in graph.factors().enumerate() {
                // Compute cavity distribution (remove current site)
                let cavity = self.compute_cavity(&approx, &sites[factor_idx])?;

                // Compute tilted distribution (include true factor)
                let tilted = self.compute_tilted(&cavity, factor)?;

                // Moment match to update site
                let new_site = self.moment_match(&cavity, &tilted, &sites[factor_idx])?;

                // Apply damping
                let damped_site = self.apply_damping(&sites[factor_idx], &new_site)?;

                // Compute change
                let change = self.compute_site_change(&sites[factor_idx], &damped_site)?;
                max_change = max_change.max(change);

                // Update site
                sites[factor_idx] = damped_site;
            }

            // Recompute global approximation
            approx = self.compute_global_approximation(graph, &sites)?;

            // Check convergence
            if max_change < self.tolerance {
                eprintln!(
                    "EP converged in {} iterations (max change: {:.6})",
                    iteration + 1,
                    max_change
                );
                break;
            }

            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "EP reached maximum iterations ({}) with max change: {:.6}",
                    self.max_iterations, max_change
                );
            }
        }

        // Extract marginals from global approximation
        self.extract_marginals(graph, &approx, &sites)
    }

    /// Initialize sites uniformly.
    fn initialize_sites(&self, graph: &FactorGraph) -> Result<Vec<Site>> {
        let mut sites = Vec::new();

        for (idx, factor) in graph.factors().enumerate() {
            let cardinalities: Vec<usize> = factor
                .variables
                .iter()
                .map(|var| graph.get_variable(var).map(|v| v.cardinality).unwrap_or(2))
                .collect();

            let site = Site::new_uniform(
                format!("site_{}", idx),
                factor.variables.clone(),
                &cardinalities,
            )?;

            sites.push(site);
        }

        Ok(sites)
    }

    /// Compute global approximation as product of all sites.
    fn compute_global_approximation(&self, _graph: &FactorGraph, sites: &[Site]) -> Result<Factor> {
        if sites.is_empty() {
            return Err(PgmError::InvalidGraph(
                "No sites to compute approximation".to_string(),
            ));
        }

        let mut result = sites[0].factor.clone();

        for site in sites.iter().skip(1) {
            result = result.product(&site.factor)?;
        }

        // Normalize
        result.normalize();

        Ok(result)
    }

    /// Compute cavity distribution by removing a site.
    fn compute_cavity(&self, approx: &Factor, site: &Site) -> Result<Factor> {
        // First, marginalize the approximation to the variables in the site
        // to ensure both have the same scope before division
        let approx_marginal = if approx.variables == site.variables {
            approx.clone()
        } else {
            approx.marginalize_out_all_except(&site.variables)?
        };

        // Cavity = approximation / site
        let cavity = approx_marginal.divide(&site.factor)?;
        Ok(cavity)
    }

    /// Compute tilted distribution by including the true factor.
    fn compute_tilted(&self, cavity: &Factor, true_factor: &Factor) -> Result<Factor> {
        // Tilted = cavity × true_factor
        let tilted = cavity.product(true_factor)?;
        Ok(tilted)
    }

    /// Moment match: find site that makes cavity × site ≈ tilted.
    fn moment_match(&self, cavity: &Factor, tilted: &Factor, _old_site: &Site) -> Result<Site> {
        // For discrete distributions, we can compute the new site as:
        // new_site = tilted / cavity

        let new_factor = tilted.divide(cavity)?;

        // Ensure numerical stability
        let mut stabilized = new_factor.clone();
        stabilized.values.mapv_inplace(|v| v.max(self.min_value));

        Ok(Site::from_factor(stabilized))
    }

    /// Apply damping to site update.
    fn apply_damping(&self, old_site: &Site, new_site: &Site) -> Result<Site> {
        if self.damping == 0.0 {
            return Ok(new_site.clone());
        }

        // Damped = (1 - damping) × new + damping × old
        let old_values = &old_site.factor.values;
        let new_values = &new_site.factor.values;

        let damped_values = (1.0 - self.damping) * new_values + self.damping * old_values;

        let damped_factor = Factor::new(
            new_site.factor.name.clone(),
            new_site.factor.variables.clone(),
            damped_values,
        )?;

        Ok(Site::from_factor(damped_factor))
    }

    /// Compute change between two sites (for convergence check).
    fn compute_site_change(&self, old_site: &Site, new_site: &Site) -> Result<f64> {
        // Compute L1 distance between site parameters
        let diff = &new_site.factor.values - &old_site.factor.values;
        let change = diff.mapv(|v| v.abs()).sum();
        Ok(change)
    }

    /// Extract marginals from the global approximation.
    fn extract_marginals(
        &self,
        graph: &FactorGraph,
        approx: &Factor,
        _sites: &[Site],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut marginals = HashMap::new();

        for (var, _) in graph.variables() {
            let marginal = approx.marginalize_out_all_except(std::slice::from_ref(var))?;
            let mut normalized = marginal.clone();
            normalized.normalize();
            marginals.insert(var.clone(), normalized.values);
        }

        Ok(marginals)
    }
}

/// Gaussian site approximation for continuous variables.
///
/// In the Gaussian case, sites are parameterized by natural parameters (precision, precision-weighted mean).
#[derive(Debug, Clone)]
pub struct GaussianSite {
    /// Variable name
    pub variable: String,
    /// Precision (inverse variance)
    pub precision: f64,
    /// Precision-weighted mean (precision × mean)
    pub precision_mean: f64,
}

impl GaussianSite {
    /// Create a new Gaussian site with given parameters.
    pub fn new(variable: String, precision: f64, precision_mean: f64) -> Self {
        Self {
            variable,
            precision,
            precision_mean,
        }
    }

    /// Create a uniform (uninformative) Gaussian site.
    pub fn uniform(variable: String) -> Self {
        Self {
            variable,
            precision: 0.0,
            precision_mean: 0.0,
        }
    }

    /// Compute mean from natural parameters.
    pub fn mean(&self) -> f64 {
        if self.precision > 1e-10 {
            self.precision_mean / self.precision
        } else {
            0.0
        }
    }

    /// Compute variance from precision.
    pub fn variance(&self) -> f64 {
        if self.precision > 1e-10 {
            1.0 / self.precision
        } else {
            f64::INFINITY
        }
    }

    /// Product of two Gaussian sites (in natural parameterization).
    pub fn product(&self, other: &GaussianSite) -> Self {
        Self {
            variable: self.variable.clone(),
            precision: self.precision + other.precision,
            precision_mean: self.precision_mean + other.precision_mean,
        }
    }

    /// Division of two Gaussian sites (in natural parameterization).
    pub fn divide(&self, other: &GaussianSite) -> Self {
        Self {
            variable: self.variable.clone(),
            precision: self.precision - other.precision,
            precision_mean: self.precision_mean - other.precision_mean,
        }
    }
}

/// Gaussian EP for continuous variables with moment matching.
#[allow(dead_code)]
pub struct GaussianEP {
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Damping factor
    damping: f64,
}

impl Default for GaussianEP {
    fn default() -> Self {
        Self::new(100, 1e-6, 0.0)
    }
}

impl GaussianEP {
    /// Create a new Gaussian EP instance.
    pub fn new(max_iterations: usize, tolerance: f64, damping: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping,
        }
    }

    /// Compute Gaussian moments (mean, variance) from a tilted distribution.
    ///
    /// This is a placeholder for moment computation. In practice, you would:
    /// 1. Compute cavity distribution
    /// 2. Multiply by true factor
    /// 3. Compute mean and variance of the result
    pub fn compute_moments(
        &self,
        cavity: &GaussianSite,
        _true_factor_callback: impl Fn(f64) -> f64,
    ) -> (f64, f64) {
        // This is simplified - in practice, you'd integrate to get moments
        let mean = cavity.mean();
        let variance = cavity.variance();
        (mean, variance)
    }

    /// Match moments to update site.
    pub fn match_moments(
        &self,
        cavity: &GaussianSite,
        tilted_mean: f64,
        tilted_var: f64,
    ) -> GaussianSite {
        // Compute new site such that cavity × site has given moments
        let new_precision = 1.0 / tilted_var - cavity.precision;
        let new_precision_mean = tilted_mean / tilted_var - cavity.precision_mean;

        GaussianSite::new(
            cavity.variable.clone(),
            new_precision.max(0.0), // Ensure non-negative
            new_precision_mean,
        )
    }

    /// Apply damping to site update.
    pub fn damp_site(&self, old_site: &GaussianSite, new_site: &GaussianSite) -> GaussianSite {
        if self.damping == 0.0 {
            return new_site.clone();
        }

        let damped_precision =
            (1.0 - self.damping) * new_site.precision + self.damping * old_site.precision;
        let damped_precision_mean =
            (1.0 - self.damping) * new_site.precision_mean + self.damping * old_site.precision_mean;

        GaussianSite::new(
            new_site.variable.clone(),
            damped_precision,
            damped_precision_mean,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_site_creation() {
        let site = Site::new_uniform("test_site".to_string(), vec!["X".to_string()], &[2]).unwrap();

        assert_eq!(site.variables.len(), 1);
        assert_eq!(site.factor.variables[0], "X");

        // Should be uniform
        let sum: f64 = site.factor.values.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_site_moments() {
        let site = GaussianSite::new("X".to_string(), 2.0, 4.0);

        // mean = precision_mean / precision = 4.0 / 2.0 = 2.0
        assert_abs_diff_eq!(site.mean(), 2.0, epsilon = 1e-10);

        // variance = 1 / precision = 1 / 2.0 = 0.5
        assert_abs_diff_eq!(site.variance(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_site_product() {
        let site1 = GaussianSite::new("X".to_string(), 2.0, 4.0);
        let site2 = GaussianSite::new("X".to_string(), 3.0, 6.0);

        let product = site1.product(&site2);

        // Precision adds: 2.0 + 3.0 = 5.0
        assert_abs_diff_eq!(product.precision, 5.0, epsilon = 1e-10);

        // Precision-weighted means add: 4.0 + 6.0 = 10.0
        assert_abs_diff_eq!(product.precision_mean, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_site_divide() {
        let site1 = GaussianSite::new("X".to_string(), 5.0, 10.0);
        let site2 = GaussianSite::new("X".to_string(), 2.0, 4.0);

        let quotient = site1.divide(&site2);

        // Precision subtracts: 5.0 - 2.0 = 3.0
        assert_abs_diff_eq!(quotient.precision, 3.0, epsilon = 1e-10);

        // Precision-weighted means subtract: 10.0 - 4.0 = 6.0
        assert_abs_diff_eq!(quotient.precision_mean, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ep_initialization() {
        let ep = ExpectationPropagation::new(50, 1e-5, 0.5);
        assert_eq!(ep.max_iterations, 50);
        assert_abs_diff_eq!(ep.tolerance, 1e-5, epsilon = 1e-10);
        assert_abs_diff_eq!(ep.damping, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_ep_simple_graph() {
        use crate::FactorGraph;

        // Create a simple factor graph with one binary variable
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("X".to_string(), "Binary".to_string(), 2);

        // Add a simple factor P(X) = [0.7, 0.3]
        let values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
            .unwrap()
            .into_dyn();
        let factor = Factor::new("P(X)".to_string(), vec!["X".to_string()], values).unwrap();
        graph.add_factor(factor).unwrap();

        // Run EP
        let ep = ExpectationPropagation::default();
        let marginals = ep.run(&graph).unwrap();

        // Check that we got a marginal for X
        assert!(marginals.contains_key("X"));

        let marginal = &marginals["X"];
        assert_eq!(marginal.ndim(), 1);
        assert_eq!(marginal.len(), 2);

        // Should be normalized
        let sum: f64 = marginal.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian_ep_moment_matching() {
        let gep = GaussianEP::default();

        // Cavity distribution: N(mean=0, var=1) => precision=1, precision_mean=0
        let cavity = GaussianSite::new("X".to_string(), 1.0, 0.0);

        // Tilted distribution has mean=2, var=0.5
        let tilted_mean = 2.0;
        let tilted_var = 0.5;

        // Match moments
        let new_site = gep.match_moments(&cavity, tilted_mean, tilted_var);

        // Verify product has correct moments
        let product = cavity.product(&new_site);

        assert_abs_diff_eq!(product.mean(), tilted_mean, epsilon = 1e-6);
        assert_abs_diff_eq!(product.variance(), tilted_var, epsilon = 1e-6);
    }

    #[test]
    fn test_ep_two_factor_graph() {
        use crate::FactorGraph;

        // Create a factor graph with two variables and two factors
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("X".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("Y".to_string(), "Binary".to_string(), 2);

        // Factor P(X) = [0.6, 0.4]
        let px_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let px = Factor::new("P(X)".to_string(), vec!["X".to_string()], px_values).unwrap();
        graph.add_factor(px).unwrap();

        // Factor P(Y|X)
        let pyx_values = Array::from_shape_vec(
            vec![2, 2],
            vec![0.8, 0.2, 0.3, 0.7], // P(Y|X=0), P(Y|X=1)
        )
        .unwrap()
        .into_dyn();
        let pyx = Factor::new(
            "P(Y|X)".to_string(),
            vec!["X".to_string(), "Y".to_string()],
            pyx_values,
        )
        .unwrap();
        graph.add_factor(pyx).unwrap();

        // Run EP
        let ep = ExpectationPropagation::new(100, 1e-6, 0.0);
        let marginals = ep.run(&graph).unwrap();

        // Check marginals
        assert!(marginals.contains_key("X"));
        assert!(marginals.contains_key("Y"));

        // Both should be normalized
        let sum_x: f64 = marginals["X"].sum();
        let sum_y: f64 = marginals["Y"].sum();
        assert_abs_diff_eq!(sum_x, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sum_y, 1.0, epsilon = 1e-6);
    }
}
