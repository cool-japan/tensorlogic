//! Quantum circuit integration for probabilistic graphical models.
//!
//! This module provides conversion between TensorLogic expressions and quantum circuits
//! for QAOA (Quantum Approximate Optimization Algorithm) and VQE (Variational Quantum
//! Eigensolver) applications.
//!
//! # Overview
//!
//! The key insight is that many optimization problems can be encoded as:
//! 1. Classical constraints → QUBO (Quadratic Unconstrained Binary Optimization)
//! 2. QUBO → Ising Hamiltonian
//! 3. Ising Hamiltonian → Parameterized quantum circuit
//!
//! # Architecture
//!
//! ```text
//! TLExpr → FactorGraph → QUBO → Ising → QAOA Circuit
//!    ↓          ↓          ↓       ↓          ↓
//! Predicates  Factors   Matrix  Hamiltonian  Gates
//! ```
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_quantrs_hooks::quantum_circuit::{
//!     QuantumCircuitBuilder, QUBOProblem, constraint_to_qubo,
//! };
//! use tensorlogic_ir::TLExpr;
//!
//! // Create a QUBO from constraints
//! let constraints = vec![
//!     TLExpr::pred("edge", vec![tensorlogic_ir::Term::var("x"), tensorlogic_ir::Term::var("y")]),
//! ];
//! let qubo = constraint_to_qubo(&constraints, 4).unwrap();
//!
//! // Build a QAOA circuit
//! let builder = QuantumCircuitBuilder::new(4);
//! let circuit = builder.build_qaoa_circuit(&qubo, 2).unwrap();
//! ```

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;
use quantrs2_sim::circuit_optimizer::{Circuit, Gate, GateType};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use tensorlogic_ir::TLExpr;

/// A Quadratic Unconstrained Binary Optimization (QUBO) problem.
///
/// QUBO represents the optimization problem:
/// minimize x^T Q x + c^T x
///
/// where x ∈ {0, 1}^n is a binary vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QUBOProblem {
    /// Number of binary variables
    pub num_variables: usize,
    /// Variable names for reference
    pub variable_names: Vec<String>,
    /// Quadratic coefficients matrix Q (upper triangular)
    pub quadratic: Array2<f64>,
    /// Linear coefficients c
    pub linear: Array1<f64>,
    /// Constant offset
    pub offset: f64,
}

impl QUBOProblem {
    /// Create a new QUBO problem with given number of variables.
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            variable_names: (0..num_variables).map(|i| format!("x_{}", i)).collect(),
            quadratic: Array2::zeros((num_variables, num_variables)),
            linear: Array1::zeros(num_variables),
            offset: 0.0,
        }
    }

    /// Create a QUBO problem with named variables.
    pub fn with_names(variable_names: Vec<String>) -> Self {
        let n = variable_names.len();
        Self {
            num_variables: n,
            variable_names,
            quadratic: Array2::zeros((n, n)),
            linear: Array1::zeros(n),
            offset: 0.0,
        }
    }

    /// Set a linear coefficient.
    pub fn set_linear(&mut self, i: usize, value: f64) {
        if i < self.num_variables {
            self.linear[i] = value;
        }
    }

    /// Set a quadratic coefficient (ensures upper triangular).
    pub fn set_quadratic(&mut self, i: usize, j: usize, value: f64) {
        if i < self.num_variables && j < self.num_variables {
            if i <= j {
                self.quadratic[[i, j]] = value;
            } else {
                self.quadratic[[j, i]] = value;
            }
        }
    }

    /// Add to a linear coefficient.
    pub fn add_linear(&mut self, i: usize, value: f64) {
        if i < self.num_variables {
            self.linear[i] += value;
        }
    }

    /// Add to a quadratic coefficient.
    pub fn add_quadratic(&mut self, i: usize, j: usize, value: f64) {
        if i < self.num_variables && j < self.num_variables {
            if i <= j {
                self.quadratic[[i, j]] += value;
            } else {
                self.quadratic[[j, i]] += value;
            }
        }
    }

    /// Evaluate the QUBO objective for a binary assignment.
    pub fn evaluate(&self, assignment: &[usize]) -> f64 {
        let mut value = self.offset;

        // Linear terms
        for (i, &xi) in assignment.iter().enumerate() {
            if i < self.num_variables {
                value += self.linear[i] * xi as f64;
            }
        }

        // Quadratic terms
        for i in 0..self.num_variables {
            for j in i..self.num_variables {
                let xi = if i < assignment.len() {
                    assignment[i]
                } else {
                    0
                };
                let xj = if j < assignment.len() {
                    assignment[j]
                } else {
                    0
                };
                value += self.quadratic[[i, j]] * (xi * xj) as f64;
            }
        }

        value
    }

    /// Convert QUBO to Ising model (h, J).
    ///
    /// The Ising model is: H = Σᵢ hᵢ σᵢ + Σᵢⱼ Jᵢⱼ σᵢσⱼ
    /// where σ ∈ {-1, +1}.
    ///
    /// The transformation is: x = (1 + σ) / 2
    pub fn to_ising(&self) -> IsingModel {
        let n = self.num_variables;
        let mut h = Array1::zeros(n);
        let mut j_matrix = Array2::zeros((n, n));
        let mut offset = self.offset;

        // Transform linear terms
        for i in 0..n {
            h[i] = self.linear[i] / 2.0;
            offset += self.linear[i] / 2.0;
        }

        // Transform quadratic terms
        for i in 0..n {
            for j in i..n {
                let qij = self.quadratic[[i, j]];
                if i == j {
                    // Diagonal: x² = x for binary variables
                    h[i] += qij / 2.0;
                    offset += qij / 2.0;
                } else {
                    // Off-diagonal: xᵢxⱼ → (1/4)(1 + σᵢ)(1 + σⱼ)
                    j_matrix[[i, j]] = qij / 4.0;
                    h[i] += qij / 4.0;
                    h[j] += qij / 4.0;
                    offset += qij / 4.0;
                }
            }
        }

        IsingModel {
            num_spins: n,
            h,
            j: j_matrix,
            offset,
        }
    }

    /// Get the index of a variable by name.
    pub fn variable_index(&self, name: &str) -> Option<usize> {
        self.variable_names.iter().position(|n| n == name)
    }
}

/// Ising model representation.
///
/// H = Σᵢ hᵢ σᵢ + Σᵢⱼ Jᵢⱼ σᵢσⱼ + offset
#[derive(Debug, Clone)]
pub struct IsingModel {
    /// Number of spins
    pub num_spins: usize,
    /// Local fields
    pub h: Array1<f64>,
    /// Coupling strengths (upper triangular)
    pub j: Array2<f64>,
    /// Constant offset
    pub offset: f64,
}

impl IsingModel {
    /// Evaluate the Ising Hamiltonian for a spin configuration.
    ///
    /// Spins should be in {-1, +1}.
    pub fn evaluate(&self, spins: &[i32]) -> f64 {
        let mut energy = self.offset;

        // Local field terms
        for i in 0..self.num_spins {
            if i < spins.len() {
                energy += self.h[i] * spins[i] as f64;
            }
        }

        // Coupling terms
        for i in 0..self.num_spins {
            for j in (i + 1)..self.num_spins {
                if i < spins.len() && j < spins.len() {
                    energy += self.j[[i, j]] * (spins[i] * spins[j]) as f64;
                }
            }
        }

        energy
    }
}

/// Convert TensorLogic constraints to a QUBO problem.
///
/// This function interprets logical constraints as penalty terms in an optimization problem.
///
/// # Constraint Interpretation
///
/// - `Pred("edge", [x, y])`: Penalizes configurations where x and y differ
/// - `And(a, b)`: Sum of penalties
/// - `Not(a)`: Inverts the penalty
///
/// # Arguments
///
/// * `constraints` - List of TLExpr constraints
/// * `num_variables` - Number of binary variables
///
/// # Returns
///
/// A QUBO problem encoding the constraints.
pub fn constraint_to_qubo(constraints: &[TLExpr], num_variables: usize) -> Result<QUBOProblem> {
    let mut qubo = QUBOProblem::new(num_variables);
    let mut var_map: HashMap<String, usize> = HashMap::new();
    let mut next_idx = 0;

    // First pass: collect all variables
    for expr in constraints {
        collect_variables(expr, &mut var_map, &mut next_idx, num_variables)?;
    }

    // Update variable names
    qubo.variable_names = vec![String::new(); num_variables];
    for (name, &idx) in &var_map {
        if idx < num_variables {
            qubo.variable_names[idx] = name.clone();
        }
    }

    // Second pass: add constraints as penalties
    for expr in constraints {
        add_constraint_penalty(expr, &var_map, &mut qubo)?;
    }

    Ok(qubo)
}

/// Collect variables from a TLExpr.
fn collect_variables(
    expr: &TLExpr,
    var_map: &mut HashMap<String, usize>,
    next_idx: &mut usize,
    max_vars: usize,
) -> Result<()> {
    match expr {
        TLExpr::Pred { args, .. } => {
            for term in args {
                if let tensorlogic_ir::Term::Var(v) = term {
                    if !var_map.contains_key(v) && *next_idx < max_vars {
                        var_map.insert(v.clone(), *next_idx);
                        *next_idx += 1;
                    }
                }
            }
        }
        TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
            collect_variables(left, var_map, next_idx, max_vars)?;
            collect_variables(right, var_map, next_idx, max_vars)?;
        }
        TLExpr::Not(inner) => {
            collect_variables(inner, var_map, next_idx, max_vars)?;
        }
        TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
            collect_variables(body, var_map, next_idx, max_vars)?;
        }
        _ => {}
    }
    Ok(())
}

/// Add a constraint as a penalty term to the QUBO.
fn add_constraint_penalty(
    expr: &TLExpr,
    var_map: &HashMap<String, usize>,
    qubo: &mut QUBOProblem,
) -> Result<()> {
    match expr {
        TLExpr::Pred { name, args } => {
            // Extract variable indices from arguments
            let mut var_indices = Vec::new();
            for term in args {
                if let tensorlogic_ir::Term::Var(v) = term {
                    if let Some(&idx) = var_map.get(v) {
                        var_indices.push(idx);
                    }
                }
            }

            // Add penalty based on predicate name and arity
            match name.as_str() {
                "edge" | "connected" | "equal" if var_indices.len() >= 2 => {
                    // Penalize when x ≠ y: penalty = (x - y)² = x + y - 2xy
                    let i = var_indices[0];
                    let j = var_indices[1];
                    qubo.add_linear(i, 1.0);
                    qubo.add_linear(j, 1.0);
                    qubo.add_quadratic(i, j, -2.0);
                }
                "conflict" | "different" | "not_equal" if var_indices.len() >= 2 => {
                    // Penalize when x = y: penalty = xy
                    let i = var_indices[0];
                    let j = var_indices[1];
                    qubo.add_quadratic(i, j, 1.0);
                }
                "select" | "chosen" if !var_indices.is_empty() => {
                    // Encourage selection: penalty = -x
                    let i = var_indices[0];
                    qubo.add_linear(i, -1.0);
                }
                "exclude" | "reject" if !var_indices.is_empty() => {
                    // Discourage selection: penalty = x
                    let i = var_indices[0];
                    qubo.add_linear(i, 1.0);
                }
                _ => {
                    // Generic interaction penalty for any pair of variables
                    for i in 0..var_indices.len() {
                        for j in (i + 1)..var_indices.len() {
                            qubo.add_quadratic(var_indices[i], var_indices[j], 0.5);
                        }
                    }
                }
            }
        }
        TLExpr::And(left, right) => {
            add_constraint_penalty(left, var_map, qubo)?;
            add_constraint_penalty(right, var_map, qubo)?;
        }
        TLExpr::Not(inner) => {
            // For NOT, we would invert the penalty sign (simplified approach)
            // In practice, this requires creating auxiliary variables
            add_constraint_penalty(inner, var_map, qubo)?;
        }
        _ => {
            // Other expressions are ignored for now
        }
    }
    Ok(())
}

/// Convert a TensorLogic expression to a QAOA circuit.
///
/// This is a high-level function that:
/// 1. Converts the expression to a QUBO
/// 2. Converts QUBO to Ising model
/// 3. Builds a QAOA circuit
///
/// # Arguments
///
/// * `expr` - The TensorLogic expression
/// * `num_layers` - Number of QAOA layers (p parameter)
/// * `num_qubits` - Number of qubits to use
///
/// # Returns
///
/// A quantum circuit implementing QAOA.
pub fn tlexpr_to_qaoa_circuit(
    expr: &TLExpr,
    num_layers: usize,
    num_qubits: usize,
) -> Result<Circuit> {
    // Convert expression to constraints
    let constraints = vec![expr.clone()];
    let qubo = constraint_to_qubo(&constraints, num_qubits)?;

    // Build QAOA circuit
    let builder = QuantumCircuitBuilder::new(num_qubits);
    builder.build_qaoa_circuit(&qubo, num_layers)
}

/// Convert a factor graph to a QUBO problem.
///
/// Each factor becomes a penalty term in the QUBO.
pub fn factor_graph_to_qubo(graph: &FactorGraph) -> Result<QUBOProblem> {
    let num_vars = graph.num_variables();
    let mut qubo = QUBOProblem::new(num_vars);

    // Map variable names to indices
    let mut var_to_idx: HashMap<String, usize> = HashMap::new();
    for (idx, var_name) in graph.variable_names().enumerate() {
        var_to_idx.insert(var_name.clone(), idx);
        if idx < qubo.variable_names.len() {
            qubo.variable_names[idx] = var_name.clone();
        }
    }

    // Convert factors to QUBO terms
    for factor in graph.factors() {
        let factor_vars: Vec<usize> = factor
            .variables
            .iter()
            .filter_map(|v| var_to_idx.get(v).copied())
            .collect();

        // For binary variables, convert factor values to QUBO coefficients
        if factor_vars.len() == 1 {
            // Unary factor: linear term
            let i = factor_vars[0];
            if factor.values.len() >= 2 {
                // Penalty for value 1 vs value 0
                let penalty = -(factor.values[[1]].ln() - factor.values[[0]].ln());
                if penalty.is_finite() {
                    qubo.add_linear(i, penalty);
                }
            }
        } else if factor_vars.len() == 2 {
            // Binary factor: quadratic term
            let i = factor_vars[0];
            let j = factor_vars[1];
            if factor.values.len() >= 4 {
                // Extract coupling strength from factor values
                // J_ij ≈ log(f(0,0) * f(1,1)) - log(f(0,1) * f(1,0))
                let f00 = factor.values[[0, 0]].max(1e-10);
                let f01 = factor.values[[0, 1]].max(1e-10);
                let f10 = factor.values[[1, 0]].max(1e-10);
                let f11 = factor.values[[1, 1]].max(1e-10);

                let coupling = ((f00 * f11).ln() - (f01 * f10).ln()) / 4.0;
                if coupling.is_finite() {
                    qubo.add_quadratic(i, j, -coupling);
                }
            }
        }
        // Higher-order factors require auxiliary variables (not implemented)
    }

    Ok(qubo)
}

/// Builder for constructing quantum circuits.
///
/// This builder creates parameterized circuits for variational algorithms
/// like QAOA and VQE.
pub struct QuantumCircuitBuilder {
    /// Number of qubits
    num_qubits: usize,
    /// Default parameter values (gamma, beta for QAOA)
    default_params: Vec<f64>,
}

impl QuantumCircuitBuilder {
    /// Create a new circuit builder.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            default_params: vec![PI / 4.0, PI / 4.0], // Default gamma, beta
        }
    }

    /// Set default parameters.
    pub fn with_params(mut self, params: Vec<f64>) -> Self {
        self.default_params = params;
        self
    }

    /// Build an initial state circuit (|+⟩^⊗n).
    pub fn build_initial_state(&self) -> Result<Circuit> {
        let mut circuit = Circuit::new(self.num_qubits);

        // Apply Hadamard to all qubits
        for qubit in 0..self.num_qubits {
            let _ = circuit.add_gate(Gate::new(GateType::H, vec![qubit]));
        }

        Ok(circuit)
    }

    /// Build a QAOA circuit for a QUBO problem.
    ///
    /// QAOA consists of alternating layers:
    /// 1. Problem unitary: U_C(γ) = exp(-iγC) where C is the cost function
    /// 2. Mixer unitary: U_B(β) = exp(-iβB) where B = Σ Xᵢ
    ///
    /// # Arguments
    ///
    /// * `qubo` - The QUBO problem to optimize
    /// * `num_layers` - Number of QAOA layers (p)
    ///
    /// # Returns
    ///
    /// A quantum circuit implementing QAOA.
    pub fn build_qaoa_circuit(&self, qubo: &QUBOProblem, num_layers: usize) -> Result<Circuit> {
        if qubo.num_variables > self.num_qubits {
            return Err(PgmError::InvalidGraph(format!(
                "QUBO has {} variables but circuit has {} qubits",
                qubo.num_variables, self.num_qubits
            )));
        }

        // Convert QUBO to Ising model for circuit construction
        let ising = qubo.to_ising();

        let mut circuit = Circuit::new(self.num_qubits);

        // Initial state: |+⟩^⊗n
        for qubit in 0..self.num_qubits {
            let _ = circuit.add_gate(Gate::new(GateType::H, vec![qubit]));
        }

        // QAOA layers
        for layer in 0..num_layers {
            // Get parameters for this layer
            let gamma = if layer * 2 < self.default_params.len() {
                self.default_params[layer * 2]
            } else {
                PI / (4.0 * (layer + 1) as f64)
            };

            let beta = if layer * 2 + 1 < self.default_params.len() {
                self.default_params[layer * 2 + 1]
            } else {
                PI / (4.0 * (layer + 1) as f64)
            };

            // Problem unitary: exp(-iγC)
            self.add_problem_unitary(&mut circuit, &ising, gamma);

            // Mixer unitary: exp(-iβB)
            self.add_mixer_unitary(&mut circuit, beta);
        }

        Ok(circuit)
    }

    /// Add the problem unitary U_C(γ) to the circuit.
    fn add_problem_unitary(&self, circuit: &mut Circuit, ising: &IsingModel, gamma: f64) {
        let n = ising.num_spins.min(self.num_qubits);

        // Local field terms: exp(-iγhᵢZᵢ) = Rz(2γhᵢ)
        for i in 0..n {
            if ising.h[i].abs() > 1e-10 {
                let angle = 2.0 * gamma * ising.h[i];
                let _ = circuit.add_gate(Gate::with_parameters(GateType::RZ, vec![i], vec![angle]));
            }
        }

        // Coupling terms: exp(-iγJᵢⱼZᵢZⱼ) = Rzz(2γJᵢⱼ)
        for i in 0..n {
            for j in (i + 1)..n {
                let jij = ising.j[[i, j]];
                if jij.abs() > 1e-10 {
                    self.add_rzz_gate(circuit, i, j, 2.0 * gamma * jij);
                }
            }
        }
    }

    /// Add the mixer unitary U_B(β) to the circuit.
    fn add_mixer_unitary(&self, circuit: &mut Circuit, beta: f64) {
        // X-mixer: exp(-iβΣXᵢ) = ⊗ᵢ Rx(2β)
        for qubit in 0..self.num_qubits {
            let _ = circuit.add_gate(Gate::with_parameters(
                GateType::RX,
                vec![qubit],
                vec![2.0 * beta],
            ));
        }
    }

    /// Add an Rzz gate (ZZ rotation) to the circuit.
    ///
    /// Rzz(θ) = exp(-iθZZ/2) decomposed into native gates:
    /// CNOT(i, j) - Rz(θ)@j - CNOT(i, j)
    fn add_rzz_gate(&self, circuit: &mut Circuit, qubit1: usize, qubit2: usize, angle: f64) {
        let _ = circuit.add_gate(Gate::new(GateType::CNOT, vec![qubit1, qubit2]));
        let _ = circuit.add_gate(Gate::with_parameters(
            GateType::RZ,
            vec![qubit2],
            vec![angle],
        ));
        let _ = circuit.add_gate(Gate::new(GateType::CNOT, vec![qubit1, qubit2]));
    }

    /// Build a hardware-efficient ansatz circuit.
    ///
    /// This circuit uses alternating layers of single-qubit rotations
    /// and entangling gates.
    pub fn build_hardware_efficient_ansatz(&self, num_layers: usize) -> Result<Circuit> {
        let mut circuit = Circuit::new(self.num_qubits);

        for _layer in 0..num_layers {
            // Single-qubit rotations: Ry-Rz
            for qubit in 0..self.num_qubits {
                let _ = circuit.add_gate(Gate::with_parameters(
                    GateType::RY,
                    vec![qubit],
                    vec![PI / 4.0],
                ));
                let _ = circuit.add_gate(Gate::with_parameters(
                    GateType::RZ,
                    vec![qubit],
                    vec![PI / 4.0],
                ));
            }

            // Entangling layer: linear connectivity
            for qubit in 0..(self.num_qubits - 1) {
                let _ = circuit.add_gate(Gate::new(GateType::CNOT, vec![qubit, qubit + 1]));
            }
        }

        // Final rotation layer
        for qubit in 0..self.num_qubits {
            let _ = circuit.add_gate(Gate::with_parameters(
                GateType::RY,
                vec![qubit],
                vec![PI / 4.0],
            ));
        }

        Ok(circuit)
    }
}

/// QAOA result containing the optimal parameters and solution.
#[derive(Debug, Clone)]
pub struct QAOAResult {
    /// Optimal gamma parameters
    pub gamma: Vec<f64>,
    /// Optimal beta parameters
    pub beta: Vec<f64>,
    /// Best solution found
    pub best_solution: Vec<usize>,
    /// Best objective value
    pub best_value: f64,
    /// Number of optimization iterations
    pub iterations: usize,
}

/// Configuration for QAOA optimization.
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (p)
    pub num_layers: usize,
    /// Number of measurement shots
    pub num_shots: usize,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            num_shots: 1000,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl QAOAConfig {
    /// Create a new QAOA configuration.
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            ..Default::default()
        }
    }

    /// Set the number of measurement shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.num_shots = shots;
        self
    }

    /// Set the maximum iterations.
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qubo_creation() {
        let qubo = QUBOProblem::new(3);
        assert_eq!(qubo.num_variables, 3);
        assert_eq!(qubo.variable_names.len(), 3);
    }

    #[test]
    fn test_qubo_with_names() {
        let qubo = QUBOProblem::with_names(vec!["x".to_string(), "y".to_string(), "z".to_string()]);
        assert_eq!(qubo.num_variables, 3);
        assert_eq!(qubo.variable_names[0], "x");
        assert_eq!(qubo.variable_index("y"), Some(1));
    }

    #[test]
    fn test_qubo_evaluation() {
        let mut qubo = QUBOProblem::new(2);
        qubo.set_linear(0, 1.0);
        qubo.set_linear(1, 2.0);
        qubo.set_quadratic(0, 1, 3.0);

        // x=0, y=0: 0
        assert_abs_diff_eq!(qubo.evaluate(&[0, 0]), 0.0);

        // x=1, y=0: 1
        assert_abs_diff_eq!(qubo.evaluate(&[1, 0]), 1.0);

        // x=0, y=1: 2
        assert_abs_diff_eq!(qubo.evaluate(&[0, 1]), 2.0);

        // x=1, y=1: 1 + 2 + 3 = 6
        assert_abs_diff_eq!(qubo.evaluate(&[1, 1]), 6.0);
    }

    #[test]
    fn test_qubo_to_ising() {
        let mut qubo = QUBOProblem::new(2);
        qubo.set_linear(0, 1.0);
        qubo.set_quadratic(0, 1, 2.0);

        let ising = qubo.to_ising();
        assert_eq!(ising.num_spins, 2);

        // Verify Ising evaluation matches QUBO for corresponding configurations
        // x=0 → σ=-1, x=1 → σ=+1
        let qubo_00 = qubo.evaluate(&[0, 0]);
        let qubo_11 = qubo.evaluate(&[1, 1]);

        let ising_mm = ising.evaluate(&[-1, -1]);
        let ising_pp = ising.evaluate(&[1, 1]);

        // The values should match (up to offset)
        assert_abs_diff_eq!(qubo_11 - qubo_00, ising_pp - ising_mm, epsilon = 1e-6);
    }

    #[test]
    fn test_constraint_to_qubo() {
        use tensorlogic_ir::Term;

        let constraints = vec![TLExpr::pred("edge", vec![Term::var("x"), Term::var("y")])];

        let qubo = constraint_to_qubo(&constraints, 4).ok();
        assert!(qubo.is_some());

        let qubo = qubo.expect("QUBO creation failed");
        assert_eq!(qubo.num_variables, 4);
    }

    #[test]
    fn test_circuit_builder_initial_state() {
        let builder = QuantumCircuitBuilder::new(3);
        let circuit = builder.build_initial_state();

        assert!(circuit.is_ok());
        let circuit = circuit.expect("Circuit creation failed");
        assert_eq!(circuit.n_qubits, 3);
    }

    #[test]
    fn test_qaoa_circuit_creation() {
        let mut qubo = QUBOProblem::new(2);
        qubo.set_quadratic(0, 1, 1.0);

        let builder = QuantumCircuitBuilder::new(2);
        let circuit = builder.build_qaoa_circuit(&qubo, 1);

        assert!(circuit.is_ok());
        let circuit = circuit.expect("Circuit creation failed");
        assert_eq!(circuit.n_qubits, 2);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let builder = QuantumCircuitBuilder::new(3);
        let circuit = builder.build_hardware_efficient_ansatz(2);

        assert!(circuit.is_ok());
        let circuit = circuit.expect("Circuit creation failed");
        assert_eq!(circuit.n_qubits, 3);
    }

    #[test]
    fn test_qaoa_config() {
        let config = QAOAConfig::new(3).with_shots(2000).with_max_iterations(50);

        assert_eq!(config.num_layers, 3);
        assert_eq!(config.num_shots, 2000);
        assert_eq!(config.max_iterations, 50);
    }

    #[test]
    fn test_tlexpr_to_qaoa() {
        use tensorlogic_ir::Term;

        let expr = TLExpr::pred("conflict", vec![Term::var("a"), Term::var("b")]);

        let circuit = tlexpr_to_qaoa_circuit(&expr, 1, 4);
        assert!(circuit.is_ok());
    }
}
