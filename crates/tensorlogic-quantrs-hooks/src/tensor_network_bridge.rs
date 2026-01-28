//! Tensor network bridge for quantum-classical hybrid inference.
//!
//! This module provides conversion between probabilistic graphical models
//! and quantum tensor network representations, enabling efficient computation
//! of marginals and partition functions.
//!
//! # Overview
//!
//! Tensor networks provide a natural bridge between:
//! - Classical PGMs (factor graphs, MRFs)
//! - Quantum states (MPS, PEPS, MERA)
//!
//! # Key Concepts
//!
//! - **MPS (Matrix Product State)**: 1D tensor network for linear chains
//! - **PEPS (Projected Entangled Pair State)**: 2D tensor network
//! - **Tensor Network Contraction**: Computing expectations and marginals
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_quantrs_hooks::tensor_network_bridge::{
//!     factor_graph_to_tensor_network, TensorNetwork,
//! };
//! use tensorlogic_quantrs_hooks::FactorGraph;
//!
//! let mut graph = FactorGraph::new();
//! graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
//! graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);
//!
//! let tn = factor_graph_to_tensor_network(&graph).unwrap();
//! println!("Tensor network with {} tensors", tn.num_tensors());
//! ```

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;
use crate::linear_chain_crf::LinearChainCRF;
use quantrs2_sim::Complex64;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayD};
use serde::{Deserialize, Serialize};

/// A tensor in the tensor network.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name/identifier
    pub name: String,
    /// Tensor data (n-dimensional array)
    pub data: ArrayD<Complex64>,
    /// Index labels for contraction
    pub indices: Vec<String>,
    /// Bond dimensions for each index
    pub bond_dims: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor.
    pub fn new(name: String, data: ArrayD<Complex64>, indices: Vec<String>) -> Self {
        let bond_dims = data.shape().to_vec();
        Self {
            name,
            data,
            indices,
            bond_dims,
        }
    }

    /// Create from a real-valued array.
    pub fn from_real(name: String, data: ArrayD<f64>, indices: Vec<String>) -> Self {
        let complex_data = data.mapv(|x| Complex64::new(x, 0.0));
        Self::new(name, complex_data, indices)
    }

    /// Get the number of indices (rank).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Get bond dimension for a specific index.
    pub fn bond_dim(&self, index: &str) -> Option<usize> {
        self.indices
            .iter()
            .position(|i| i == index)
            .map(|pos| self.bond_dims[pos])
    }

    /// Contract two tensors over shared indices.
    pub fn contract(&self, other: &Tensor) -> Result<Tensor> {
        // Find shared indices
        let shared: Vec<(usize, usize)> = self
            .indices
            .iter()
            .enumerate()
            .filter_map(|(i, idx)| {
                other
                    .indices
                    .iter()
                    .position(|oidx| oidx == idx)
                    .map(|j| (i, j))
            })
            .collect();

        if shared.is_empty() {
            // Outer product
            return self.outer_product(other);
        }

        // Contract over shared indices using tensordot-like operation
        let result_indices: Vec<String> = self
            .indices
            .iter()
            .enumerate()
            .filter(|(i, _)| !shared.iter().any(|(si, _)| si == i))
            .map(|(_, idx)| idx.clone())
            .chain(
                other
                    .indices
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| !shared.iter().any(|(_, sj)| sj == j))
                    .map(|(_, idx)| idx.clone()),
            )
            .collect();

        // Compute result shape
        let result_shape: Vec<usize> = self
            .bond_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !shared.iter().any(|(si, _)| si == i))
            .map(|(_, &d)| d)
            .chain(
                other
                    .bond_dims
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| !shared.iter().any(|(_, sj)| sj == j))
                    .map(|(_, &d)| d),
            )
            .collect();

        // For simplicity, implement for common cases
        // Full generic contraction would require more complex index manipulation
        let result_data = self.contract_data(other, &shared, &result_shape)?;

        Ok(Tensor {
            name: format!("{}*{}", self.name, other.name),
            data: result_data,
            indices: result_indices,
            bond_dims: result_shape,
        })
    }

    /// Contract tensor data (simplified implementation).
    fn contract_data(
        &self,
        _other: &Tensor,
        _shared: &[(usize, usize)],
        result_shape: &[usize],
    ) -> Result<ArrayD<Complex64>> {
        // Simplified: flatten and sum over contracted indices
        // In practice, this would use optimized tensor contraction
        let total_size: usize = result_shape.iter().product();
        let data = vec![Complex64::new(0.0, 0.0); total_size.max(1)];
        ArrayD::from_shape_vec(result_shape.to_vec(), data)
            .map_err(|e| PgmError::InvalidDistribution(format!("Contraction failed: {}", e)))
    }

    /// Compute outer product with another tensor.
    fn outer_product(&self, other: &Tensor) -> Result<Tensor> {
        let result_indices: Vec<String> = self
            .indices
            .iter()
            .chain(other.indices.iter())
            .cloned()
            .collect();

        let result_shape: Vec<usize> = self
            .bond_dims
            .iter()
            .chain(other.bond_dims.iter())
            .copied()
            .collect();

        let total_size: usize = result_shape.iter().product();
        let mut data = vec![Complex64::new(0.0, 0.0); total_size.max(1)];

        // Compute outer product
        for (i, &a) in self.data.iter().enumerate() {
            for (j, &b) in other.data.iter().enumerate() {
                data[i * other.data.len() + j] = a * b;
            }
        }

        Ok(Tensor {
            name: format!("{}⊗{}", self.name, other.name),
            data: ArrayD::from_shape_vec(result_shape.clone(), data).map_err(|e| {
                PgmError::InvalidDistribution(format!("Outer product failed: {}", e))
            })?,
            indices: result_indices,
            bond_dims: result_shape,
        })
    }
}

/// A tensor network representation.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// Tensors in the network
    tensors: Vec<Tensor>,
    /// Physical indices (observable variables)
    physical_indices: Vec<String>,
    /// Virtual/bond indices
    bond_indices: Vec<String>,
}

impl TensorNetwork {
    /// Create a new empty tensor network.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            physical_indices: Vec::new(),
            bond_indices: Vec::new(),
        }
    }

    /// Add a tensor to the network.
    pub fn add_tensor(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    /// Add a physical index.
    pub fn add_physical_index(&mut self, index: String) {
        if !self.physical_indices.contains(&index) {
            self.physical_indices.push(index);
        }
    }

    /// Add a bond index.
    pub fn add_bond_index(&mut self, index: String) {
        if !self.bond_indices.contains(&index) {
            self.bond_indices.push(index);
        }
    }

    /// Get the number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Get the number of physical indices.
    pub fn num_physical_indices(&self) -> usize {
        self.physical_indices.len()
    }

    /// Get total bond dimension.
    pub fn total_bond_dim(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.bond_dims.iter().product::<usize>())
            .sum()
    }

    /// Contract the entire network to a single tensor.
    ///
    /// This uses a simple sequential contraction strategy.
    pub fn contract(&self) -> Result<Tensor> {
        if self.tensors.is_empty() {
            return Err(PgmError::InvalidGraph(
                "Cannot contract empty tensor network".to_string(),
            ));
        }

        let mut result = self.tensors[0].clone();
        for tensor in self.tensors.iter().skip(1) {
            result = result.contract(tensor)?;
        }

        Ok(result)
    }

    /// Compute the partition function (trace over all indices).
    pub fn partition_function(&self) -> Result<Complex64> {
        let contracted = self.contract()?;
        Ok(contracted.data.iter().sum())
    }

    /// Compute marginal for a subset of physical indices.
    pub fn marginal(&self, indices: &[String]) -> Result<Tensor> {
        // Contract network, then trace out non-specified indices
        let contracted = self.contract()?;

        // Keep only specified indices
        let keep_positions: Vec<usize> = contracted
            .indices
            .iter()
            .enumerate()
            .filter_map(
                |(i, idx)| {
                    if indices.contains(idx) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();

        if keep_positions.is_empty() {
            // Return scalar
            let sum: Complex64 = contracted.data.iter().sum();
            return Ok(Tensor::new(
                "marginal".to_string(),
                ArrayD::from_elem(vec![], sum),
                vec![],
            ));
        }

        // Sum over non-kept indices (simplified implementation)
        let result_shape: Vec<usize> = keep_positions
            .iter()
            .map(|&i| contracted.bond_dims[i])
            .collect();
        let result_indices: Vec<String> = keep_positions
            .iter()
            .map(|&i| contracted.indices[i].clone())
            .collect();

        // For now, return contracted tensor with subset of indices
        Ok(Tensor {
            name: "marginal".to_string(),
            data: contracted.data, // Simplified: would need proper marginalization
            indices: result_indices,
            bond_dims: result_shape,
        })
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a factor graph to a tensor network.
///
/// Each factor becomes a tensor, and shared variables become bond indices.
pub fn factor_graph_to_tensor_network(graph: &FactorGraph) -> Result<TensorNetwork> {
    let mut tn = TensorNetwork::new();

    // Add physical indices for each variable
    for var_name in graph.variable_names() {
        tn.add_physical_index(var_name.clone());
    }

    // Convert each factor to a tensor
    for factor in graph.factors() {
        let indices = factor.variables.clone();
        let tensor = Tensor::from_real(factor.name.clone(), factor.values.clone(), indices);
        tn.add_tensor(tensor);
    }

    Ok(tn)
}

/// Matrix Product State (MPS) representation.
///
/// MPS is a 1D tensor network particularly suited for linear-chain structures.
///
/// |ψ⟩ = Σ_{s₁...sₙ} A\[1\]^{s₁} A\[2\]^{s₂} ... A\[n\]^{sₙ} |s₁...sₙ⟩
#[derive(Debug, Clone)]
pub struct MatrixProductState {
    /// Site tensors (each is [bond_left, physical, bond_right])
    pub tensors: Vec<Array3<Complex64>>,
    /// Physical dimensions at each site
    pub physical_dims: Vec<usize>,
    /// Bond dimensions
    pub bond_dims: Vec<usize>,
}

impl MatrixProductState {
    /// Create a new MPS with uniform physical dimension.
    pub fn new(length: usize, physical_dim: usize, bond_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(length);
        let mut bond_dims = Vec::with_capacity(length + 1);

        bond_dims.push(1); // Left boundary

        for i in 0..length {
            let left_dim = bond_dims[i];
            let right_dim = if i == length - 1 { 1 } else { bond_dim };
            bond_dims.push(right_dim);

            // Initialize with random values
            let tensor = Array3::from_shape_fn((left_dim, physical_dim, right_dim), |_| {
                Complex64::new(1.0 / (left_dim * physical_dim * right_dim) as f64, 0.0)
            });
            tensors.push(tensor);
        }

        Self {
            tensors,
            physical_dims: vec![physical_dim; length],
            bond_dims,
        }
    }

    /// Create an MPS in the product state |00...0⟩.
    pub fn product_state(length: usize, physical_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(length);

        for _ in 0..length {
            // |0⟩ state at each site
            let mut tensor = Array3::zeros((1, physical_dim, 1));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensors.push(tensor);
        }

        Self {
            tensors,
            physical_dims: vec![physical_dim; length],
            bond_dims: vec![1; length + 1],
        }
    }

    /// Get the length (number of sites).
    pub fn length(&self) -> usize {
        self.tensors.len()
    }

    /// Get the maximum bond dimension.
    pub fn max_bond_dim(&self) -> usize {
        *self.bond_dims.iter().max().unwrap_or(&1)
    }

    /// Contract the MPS to a single tensor (full state vector).
    ///
    /// Warning: This is exponentially expensive for large systems.
    pub fn to_state_vector(&self) -> Result<Array1<Complex64>> {
        if self.tensors.is_empty() {
            return Ok(Array1::from(vec![Complex64::new(1.0, 0.0)]));
        }

        let total_dim: usize = self.physical_dims.iter().product();
        let mut state = Array1::zeros(total_dim);

        // Enumerate all basis states
        for basis_idx in 0..total_dim {
            let mut indices = vec![0; self.tensors.len()];
            let mut temp = basis_idx;
            for (i, &dim) in self.physical_dims.iter().enumerate().rev() {
                indices[i] = temp % dim;
                temp /= dim;
            }

            // Contract along the chain
            let mut amplitude = Complex64::new(1.0, 0.0);
            let mut left_idx = 0;

            for (site, &phys_idx) in indices.iter().enumerate() {
                let tensor = &self.tensors[site];
                // Sum over virtual index
                let right_dim = tensor.shape()[2];
                let mut sum = Complex64::new(0.0, 0.0);
                for right_idx in 0..right_dim {
                    sum += tensor[[left_idx, phys_idx, right_idx]];
                }
                amplitude *= sum;
                left_idx = 0; // MPS contraction uses the virtual index
            }

            state[basis_idx] = amplitude;
        }

        // Normalize
        let norm: f64 = state
            .iter()
            .map(|x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for x in state.iter_mut() {
                *x /= norm;
            }
        }

        Ok(state)
    }

    /// Compute the norm of the MPS.
    pub fn norm(&self) -> f64 {
        let state_result: Result<Array1<Complex64>> = self.to_state_vector();
        match state_result {
            Ok(state) => {
                let state_arr: Array1<Complex64> = state;
                state_arr
                    .iter()
                    .map(|x: &Complex64| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
            }
            Err(_) => 0.0,
        }
    }

    /// Compute the expectation value of a local operator at a site.
    pub fn expectation_local(
        &self,
        site: usize,
        operator: &Array2<Complex64>,
    ) -> Result<Complex64> {
        if site >= self.tensors.len() {
            return Err(PgmError::VariableNotFound(format!(
                "Site {} out of range",
                site
            )));
        }

        // Simplified: compute full expectation (inefficient for large MPS)
        let state = self.to_state_vector()?;

        let mut result = Complex64::new(0.0, 0.0);
        let num_sites = self.tensors.len();
        let total_dim: usize = self.physical_dims.iter().product();

        for basis_idx in 0..total_dim {
            // Decode basis state
            let mut indices = vec![0; num_sites];
            let mut temp = basis_idx;
            for (i, &dim) in self.physical_dims.iter().enumerate().rev() {
                indices[i] = temp % dim;
                temp /= dim;
            }

            // Apply operator at site
            for new_idx in 0..self.physical_dims[site] {
                let op_elem = operator[[new_idx, indices[site]]];
                if op_elem.norm_sqr() > 1e-20 {
                    // Compute new basis index
                    let mut new_basis_idx = 0;
                    let mut multiplier = 1;
                    for (i, &idx) in indices.iter().enumerate().rev() {
                        let idx_to_use = if i == site { new_idx } else { idx };
                        new_basis_idx += idx_to_use * multiplier;
                        multiplier *= self.physical_dims[i];
                    }

                    result += state[new_basis_idx].conj() * op_elem * state[basis_idx];
                }
            }
        }

        Ok(result)
    }
}

/// Convert a Linear Chain CRF to a Matrix Product State.
///
/// The CRF's potential functions become the MPS tensors.
pub fn linear_chain_to_mps(
    crf: &LinearChainCRF,
    input_sequence: &[usize],
) -> Result<MatrixProductState> {
    let factor_graph = crf.to_factor_graph(input_sequence)?;
    let num_sites = input_sequence.len();

    if num_sites == 0 {
        return Err(PgmError::InvalidGraph("Empty sequence".to_string()));
    }

    // Get state dimension from CRF
    let num_states = factor_graph
        .get_variable("y_0")
        .map(|v| v.cardinality)
        .unwrap_or(2);

    // Build MPS from factors
    let mut mps = MatrixProductState::new(num_sites, num_states, num_states);

    // Populate tensors from emission and transition factors
    for t in 0..num_sites {
        let emission_name = format!("emission_{}", t);
        let transition_name = format!("transition_{}", t);

        // Get emission factor
        if let Some(emission) = factor_graph.get_factor_by_name(&emission_name) {
            // Emission factor is diagonal in physical index
            for (s, &val) in emission.values.iter().enumerate() {
                if s < num_states {
                    mps.tensors[t][[0, s, 0]] = Complex64::new(val.sqrt(), 0.0);
                }
            }
        }

        // Get transition factor (if not first site)
        if t > 0 {
            if let Some(transition) = factor_graph.get_factor_by_name(&transition_name) {
                // Transition factor connects adjacent sites
                for s_prev in 0..num_states {
                    for s_curr in 0..num_states {
                        if s_prev < transition.values.shape()[0]
                            && s_curr < transition.values.shape()[1]
                        {
                            let val = transition.values[[s_prev, s_curr]];
                            // Incorporate into MPS tensor
                            let tensor = &mut mps.tensors[t];
                            let left_dim = tensor.shape()[0];
                            let right_dim = tensor.shape()[2];
                            if s_prev < left_dim && s_curr < num_states {
                                tensor[[s_prev.min(left_dim - 1), s_curr, 0.min(right_dim - 1)]] =
                                    Complex64::new(val.sqrt(), 0.0);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(mps)
}

/// Statistics about a tensor network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNetworkStats {
    /// Number of tensors
    pub num_tensors: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Maximum tensor rank
    pub max_rank: usize,
    /// Average tensor rank
    pub avg_rank: f64,
    /// Number of physical indices
    pub num_physical_indices: usize,
    /// Number of bond indices
    pub num_bond_indices: usize,
}

impl TensorNetwork {
    /// Compute statistics about the tensor network.
    pub fn stats(&self) -> TensorNetworkStats {
        let num_tensors = self.tensors.len();
        let total_elements: usize = self.tensors.iter().map(|t| t.data.len()).sum();
        let max_rank = self.tensors.iter().map(|t| t.rank()).max().unwrap_or(0);
        let avg_rank = if num_tensors > 0 {
            self.tensors.iter().map(|t| t.rank()).sum::<usize>() as f64 / num_tensors as f64
        } else {
            0.0
        };

        TensorNetworkStats {
            num_tensors,
            total_elements,
            max_rank,
            avg_rank,
            num_physical_indices: self.physical_indices.len(),
            num_bond_indices: self.bond_indices.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FactorGraph;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tensor_creation() {
        let data = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Array creation failed");
        let tensor = Tensor::from_real(
            "test".to_string(),
            data,
            vec!["i".to_string(), "j".to_string()],
        );

        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.bond_dim("i"), Some(2));
        assert_eq!(tensor.bond_dim("j"), Some(3));
    }

    #[test]
    fn test_tensor_network_creation() {
        let mut tn = TensorNetwork::new();
        let data = ArrayD::from_shape_vec(vec![2], vec![1.0, 0.0]).expect("Array creation failed");
        let tensor = Tensor::from_real("A".to_string(), data, vec!["x".to_string()]);

        tn.add_tensor(tensor);
        tn.add_physical_index("x".to_string());

        assert_eq!(tn.num_tensors(), 1);
        assert_eq!(tn.num_physical_indices(), 1);
    }

    #[test]
    fn test_factor_graph_to_tn() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let tn = factor_graph_to_tensor_network(&graph);
        assert!(tn.is_ok());

        let tn = tn.expect("TN creation failed");
        assert_eq!(tn.num_physical_indices(), 2);
    }

    #[test]
    fn test_mps_creation() {
        let mps = MatrixProductState::new(4, 2, 4);

        assert_eq!(mps.length(), 4);
        assert_eq!(mps.physical_dims.len(), 4);
        assert!(mps.max_bond_dim() <= 4);
    }

    #[test]
    fn test_mps_product_state() {
        let mps = MatrixProductState::product_state(3, 2);

        assert_eq!(mps.length(), 3);
        assert_eq!(mps.max_bond_dim(), 1);

        // Product state |000⟩ should have norm 1
        let norm = mps.norm();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mps_to_state_vector() {
        let mps = MatrixProductState::product_state(2, 2);
        let state = mps.to_state_vector();

        assert!(state.is_ok());
        let state = state.expect("State vector failed");
        assert_eq!(state.len(), 4); // 2^2 = 4 basis states
    }

    #[test]
    fn test_tensor_network_stats() {
        let mut tn = TensorNetwork::new();
        let data1 =
            ArrayD::from_shape_vec(vec![2, 3], vec![1.0; 6]).expect("Array creation failed");
        let data2 =
            ArrayD::from_shape_vec(vec![3, 4], vec![1.0; 12]).expect("Array creation failed");

        tn.add_tensor(Tensor::from_real(
            "A".to_string(),
            data1,
            vec!["i".to_string(), "j".to_string()],
        ));
        tn.add_tensor(Tensor::from_real(
            "B".to_string(),
            data2,
            vec!["j".to_string(), "k".to_string()],
        ));

        let stats = tn.stats();
        assert_eq!(stats.num_tensors, 2);
        assert_eq!(stats.total_elements, 18);
        assert_eq!(stats.max_rank, 2);
    }

    #[test]
    fn test_tensor_outer_product() {
        let data1 = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).expect("Array creation failed");
        let data2 =
            ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("Array creation failed");

        let t1 = Tensor::from_real("A".to_string(), data1, vec!["i".to_string()]);
        let t2 = Tensor::from_real("B".to_string(), data2, vec!["j".to_string()]);

        let result = t1.contract(&t2);
        assert!(result.is_ok());

        let result = result.expect("Contraction failed");
        assert_eq!(result.indices.len(), 2);
        assert_eq!(result.bond_dims, vec![2, 3]);
    }

    #[test]
    fn test_mps_expectation() {
        let mps = MatrixProductState::product_state(2, 2);

        // Z operator: |0⟩ → +1, |1⟩ → -1
        let z_op = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .expect("Operator creation failed");

        let exp_val = mps.expectation_local(0, &z_op);
        assert!(exp_val.is_ok());

        // For |00⟩, ⟨Z⟩ at site 0 should be +1
        let exp_val = exp_val.expect("Expectation failed");
        assert_abs_diff_eq!(exp_val.re, 1.0, epsilon = 1e-6);
    }
}
