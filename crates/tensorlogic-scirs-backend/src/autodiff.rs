//! Automatic differentiation support (forward/backward passes).

use tensorlogic_infer::{ExecutorError, TlAutodiff, TlExecutor};
use tensorlogic_ir::EinsumGraph;

use crate::einsum_grad::compute_einsum_gradients;
use crate::ops::{parse_elem_op, parse_reduce_op};
use crate::{Scirs2Exec, Scirs2Tensor};

/// Stores intermediate values from forward pass for gradient computation
#[derive(Clone)]
pub struct ForwardTape {
    /// All computed tensors indexed by their tensor index
    pub tensors: Vec<Option<Scirs2Tensor>>,
    /// Input tensors for each node (for gradient computation)
    pub node_inputs: Vec<Vec<Scirs2Tensor>>,
}

impl ForwardTape {
    /// Check if the tape has any computed gradients
    pub fn is_empty(&self) -> bool {
        self.tensors.iter().all(|t| t.is_none())
    }

    /// Get the number of non-None gradients in the tape
    pub fn len(&self) -> usize {
        self.tensors.iter().filter(|t| t.is_some()).count()
    }
}

impl TlAutodiff for Scirs2Exec {
    type Tape = ForwardTape;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if graph.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "Empty graph provided".to_string(),
            ));
        }

        if graph.outputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "No output tensors specified".to_string(),
            ));
        }

        let mut computed_tensors: Vec<Option<Scirs2Tensor>> = vec![None; graph.tensors.len()];
        let mut node_inputs: Vec<Vec<Scirs2Tensor>> = Vec::with_capacity(graph.nodes.len());

        // Initialize input tensors from our stored tensors
        for (idx, tensor_name) in graph.tensors.iter().enumerate() {
            // Try direct lookup first
            if let Some(tensor) = self.tensors.get(tensor_name) {
                computed_tensors[idx] = Some(tensor.clone());
            } else {
                // Handle tensors with axes notation (e.g., "age[a]" -> "age")
                let base_name = tensor_name.split('[').next().unwrap_or(tensor_name);

                if let Some(tensor) = self.tensors.get(base_name) {
                    computed_tensors[idx] = Some(tensor.clone());
                } else if tensor_name.starts_with("const_") || base_name.starts_with("const_") {
                    // Handle constant tensors: parse value from name like "const_5" or "const_3.14"
                    let const_name = if tensor_name.starts_with("const_") {
                        tensor_name
                    } else {
                        base_name
                    };

                    if let Some(value_str) = const_name.strip_prefix("const_") {
                        if let Ok(value) = value_str.parse::<f64>() {
                            // Create a scalar tensor with the constant value
                            use scirs2_core::ndarray::arr0;
                            computed_tensors[idx] = Some(arr0(value).into_dyn());
                        }
                    }
                }
            }
        }

        // Execute each operation node in the graph
        for node in &graph.nodes {
            let inputs: Result<Vec<_>, _> = node
                .inputs
                .iter()
                .map(|&idx| {
                    computed_tensors
                        .get(idx)
                        .and_then(|t| t.as_ref())
                        .cloned()
                        .ok_or_else(|| {
                            ExecutorError::TensorNotFound(format!(
                                "Tensor at index {} not found for node with op: {:?}",
                                idx, node.op
                            ))
                        })
                })
                .collect();

            let input_tensors = inputs?;

            // Store input tensors for backward pass
            node_inputs.push(input_tensors.clone());

            // Dispatch based on operation type
            let result = match &node.op {
                tensorlogic_ir::OpType::Einsum { spec } => self.einsum(spec, &input_tensors)?,
                tensorlogic_ir::OpType::ElemUnary { op } => {
                    if input_tensors.len() != 1 {
                        return Err(ExecutorError::InvalidEinsumSpec(format!(
                            "Element-wise unary op '{}' requires 1 input, got {}",
                            op,
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    self.elem_op(elem_op, &input_tensors[0])?
                }
                tensorlogic_ir::OpType::ElemBinary { op } => {
                    if input_tensors.len() != 2 {
                        return Err(ExecutorError::InvalidEinsumSpec(format!(
                            "Element-wise binary op '{}' requires 2 inputs, got {}",
                            op,
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    self.elem_op_binary(elem_op, &input_tensors[0], &input_tensors[1])?
                }
                tensorlogic_ir::OpType::Reduce { op, axes } => {
                    if input_tensors.len() != 1 {
                        return Err(ExecutorError::InvalidEinsumSpec(format!(
                            "Reduce op '{}' requires 1 input, got {}",
                            op,
                            input_tensors.len()
                        )));
                    }
                    let reduce_op = parse_reduce_op(op)?;
                    self.reduce(reduce_op, &input_tensors[0], axes)?
                }
            };

            // Store the result at the correct output index specified by the node
            if let Some(&output_idx) = node.outputs.first() {
                computed_tensors[output_idx] = Some(result);
            } else {
                return Err(ExecutorError::InvalidEinsumSpec(
                    "Node has no output index specified".to_string(),
                ));
            }
        }

        // Store tape for potential backward pass
        self.tape = Some(ForwardTape {
            tensors: computed_tensors.clone(),
            node_inputs,
        });

        // Return the output tensor
        let output_idx = graph.outputs[0];
        computed_tensors
            .get(output_idx)
            .and_then(|t| t.clone())
            .ok_or_else(|| ExecutorError::TensorNotFound("Output tensor not computed".to_string()))
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss_grad: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        if graph.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "Empty graph provided".to_string(),
            ));
        }

        // Get the stored forward tape and clone node_inputs to avoid borrow conflicts
        let node_inputs_vec = {
            let forward_tape = self.tape.as_ref().ok_or_else(|| {
                ExecutorError::InvalidEinsumSpec(
                    "Forward pass must be called before backward pass".to_string(),
                )
            })?;
            forward_tape.node_inputs.clone()
        };

        // Initialize gradient storage - one gradient per tensor in the graph
        let mut gradients: Vec<Option<Scirs2Tensor>> = vec![None; graph.tensors.len()];

        // Set the gradient of the output tensor to the provided loss gradient
        if !graph.outputs.is_empty() {
            let output_idx = graph.outputs[0];
            gradients[output_idx] = Some(loss_grad.clone());
        }

        // Backward pass through nodes in reverse order
        for (node_idx, node) in graph.nodes.iter().enumerate().rev() {
            // Get the gradient of this node's output
            let output_idx = if let Some(&idx) = node.outputs.first() {
                idx
            } else {
                continue;
            };

            let output_grad = if let Some(grad) = &gradients[output_idx] {
                grad.clone()
            } else {
                // No gradient for this node's output - skip it
                continue;
            };

            // Get the input tensors that were used in forward pass
            let input_tensors = &node_inputs_vec[node_idx];

            // Compute gradients for inputs based on operation type
            match &node.op {
                tensorlogic_ir::OpType::Einsum { spec } => {
                    // Proper einsum gradient computation
                    match compute_einsum_gradients(spec, input_tensors, &output_grad, self) {
                        Ok(einsum_grads) => {
                            // Accumulate gradients for each input
                            for (i, &input_idx) in node.inputs.iter().enumerate() {
                                if i < einsum_grads.len() {
                                    let grad = &einsum_grads[i];
                                    if gradients[input_idx].is_none() {
                                        gradients[input_idx] = Some(grad.clone());
                                    } else if let Some(existing_grad) = &mut gradients[input_idx] {
                                        *existing_grad = &*existing_grad + grad;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Fallback: pass gradients through (for unsupported einsum patterns)
                            for &input_idx in &node.inputs {
                                if gradients[input_idx].is_none() {
                                    gradients[input_idx] = Some(output_grad.clone());
                                } else if let Some(existing_grad) = &mut gradients[input_idx] {
                                    *existing_grad = &*existing_grad + &output_grad;
                                }
                            }
                        }
                    }
                }
                tensorlogic_ir::OpType::ElemUnary { op } => {
                    // Gradient through unary operations
                    if node.inputs.len() == 1 && !input_tensors.is_empty() {
                        let input_idx = node.inputs[0];
                        let input = &input_tensors[0];

                        let grad = match op.as_str() {
                            "relu" => {
                                // ReLU gradient: grad * (input > 0)
                                use scirs2_core::ndarray::Zip;
                                Zip::from(&output_grad).and(input).map_collect(|&g, &x| {
                                    if x > 0.0 {
                                        g
                                    } else {
                                        0.0
                                    }
                                })
                            }
                            "sigmoid" => {
                                // Sigmoid gradient: grad * sigmoid(x) * (1 - sigmoid(x))
                                use scirs2_core::ndarray::Zip;
                                Zip::from(&output_grad).and(input).map_collect(|&g, &x| {
                                    let s = 1.0 / (1.0 + (-x).exp());
                                    g * s * (1.0 - s)
                                })
                            }
                            "oneminus" => {
                                // OneMinus gradient: d/dx(1 - x) = -1
                                &output_grad * (-1.0)
                            }
                            _ => output_grad.clone(),
                        };

                        if gradients[input_idx].is_none() {
                            gradients[input_idx] = Some(grad);
                        } else if let Some(existing_grad) = &mut gradients[input_idx] {
                            *existing_grad = &*existing_grad + &grad;
                        }
                    }
                }
                tensorlogic_ir::OpType::ElemBinary { op } => {
                    // Gradient through binary operations with access to input values
                    if node.inputs.len() == 2 && input_tensors.len() == 2 {
                        let x = &input_tensors[0];
                        let y = &input_tensors[1];

                        let (grad_x, grad_y) = match op.as_str() {
                            "add" => {
                                // d/dx(x + y) = 1, d/dy(x + y) = 1
                                (output_grad.clone(), output_grad.clone())
                            }
                            "subtract" | "sub" => {
                                // d/dx(x - y) = 1, d/dy(x - y) = -1
                                (output_grad.clone(), &output_grad * (-1.0))
                            }
                            "multiply" | "mul" => {
                                // d/dx(x * y) = y, d/dy(x * y) = x
                                (&output_grad * y, &output_grad * x)
                            }
                            "divide" | "div" => {
                                // d/dx(x / y) = 1/y, d/dy(x / y) = -x/y^2
                                (&output_grad / y, &output_grad * (-x) / (y * y))
                            }
                            // Comparison operations have zero gradients (non-differentiable)
                            "eq" | "lt" | "gt" | "lte" | "gte" => {
                                let zero_grad = Scirs2Tensor::zeros(output_grad.raw_dim());
                                (zero_grad.clone(), zero_grad)
                            }
                            // Extended logical operations with proper gradients
                            "or_max" | "ormax" => {
                                // OR(max): gradient flows to the larger value
                                use scirs2_core::ndarray::Zip;
                                let grad_x = Zip::from(&output_grad)
                                    .and(x)
                                    .and(y)
                                    .map_collect(|&g, &a, &b| if a >= b { g } else { 0.0 });
                                let grad_y = Zip::from(&output_grad)
                                    .and(x)
                                    .and(y)
                                    .map_collect(|&g, &a, &b| if b > a { g } else { 0.0 });
                                (grad_x, grad_y)
                            }
                            "or_prob_sum" | "orprobsum" | "or_probabilistic" => {
                                // OR(prob): a + b - ab, gradient: da = (1-b), db = (1-a)
                                use scirs2_core::ndarray::Zip;
                                let grad_x = Zip::from(&output_grad)
                                    .and(y)
                                    .map_collect(|&g, &b| g * (1.0 - b));
                                let grad_y = Zip::from(&output_grad)
                                    .and(x)
                                    .map_collect(|&g, &a| g * (1.0 - a));
                                (grad_x, grad_y)
                            }
                            "nand" => {
                                // NAND: 1 - ab, gradient: da = -b, db = -a
                                (&output_grad * (-y), &output_grad * (-x))
                            }
                            "nor" => {
                                // NOR: 1 - max(a,b), gradient flows negatively to max
                                use scirs2_core::ndarray::Zip;
                                let grad_x = Zip::from(&output_grad)
                                    .and(x)
                                    .and(y)
                                    .map_collect(|&g, &a, &b| if a >= b { -g } else { 0.0 });
                                let grad_y = Zip::from(&output_grad)
                                    .and(x)
                                    .and(y)
                                    .map_collect(|&g, &a, &b| if b > a { -g } else { 0.0 });
                                (grad_x, grad_y)
                            }
                            "xor" => {
                                // XOR: a + b - 2ab, gradient: da = 1 - 2b, db = 1 - 2a
                                use scirs2_core::ndarray::Zip;
                                let grad_x = Zip::from(&output_grad)
                                    .and(y)
                                    .map_collect(|&g, &b| g * (1.0 - 2.0 * b));
                                let grad_y = Zip::from(&output_grad)
                                    .and(x)
                                    .map_collect(|&g, &a| g * (1.0 - 2.0 * a));
                                (grad_x, grad_y)
                            }
                            _ => (output_grad.clone(), output_grad.clone()),
                        };

                        // Accumulate gradient for first input
                        let input_idx_0 = node.inputs[0];
                        if gradients[input_idx_0].is_none() {
                            gradients[input_idx_0] = Some(grad_x);
                        } else if let Some(existing_grad) = &mut gradients[input_idx_0] {
                            *existing_grad = &*existing_grad + &grad_x;
                        }

                        // Accumulate gradient for second input
                        let input_idx_1 = node.inputs[1];
                        if gradients[input_idx_1].is_none() {
                            gradients[input_idx_1] = Some(grad_y);
                        } else if let Some(existing_grad) = &mut gradients[input_idx_1] {
                            *existing_grad = &*existing_grad + &grad_y;
                        }
                    }
                }
                tensorlogic_ir::OpType::Reduce { op: _, axes } => {
                    // Gradient through reduction: broadcast gradient back to original shape
                    if node.inputs.len() == 1 && !input_tensors.is_empty() {
                        let input_idx = node.inputs[0];
                        let input_shape = input_tensors[0].shape();

                        // For reduction, gradient needs to be broadcast back to input shape
                        let grad = if axes.is_empty() {
                            // Global reduction - broadcast scalar to original shape
                            let mut result = Scirs2Tensor::zeros(input_shape);
                            result.fill(output_grad[[]]);
                            result
                        } else {
                            // Reduction over specific axes - expand dimensions
                            // For sum reduction, gradient is broadcast
                            // For max/min, gradient goes to the locations that were selected
                            use scirs2_core::ndarray::ArrayD;
                            let mut expanded_shape: Vec<usize> = input_shape.to_vec();
                            for &axis in axes {
                                expanded_shape[axis] = 1;
                            }

                            // Reshape output grad to match expanded shape
                            let reshaped = if let Ok(reshaped) = output_grad
                                .clone()
                                .into_shape_with_order(expanded_shape.clone())
                            {
                                reshaped
                            } else {
                                output_grad.clone()
                            };

                            // Broadcast to original shape
                            if let Some(broadcasted) = reshaped.broadcast(input_shape) {
                                broadcasted.to_owned()
                            } else {
                                // Fallback: just replicate the gradient
                                let mut result = ArrayD::zeros(input_shape);
                                // Simple replication for sum (correct for sum, approximate for max/min)
                                result
                                    .iter_mut()
                                    .for_each(|v| *v = output_grad.iter().sum::<f64>());
                                result
                            }
                        };

                        if gradients[input_idx].is_none() {
                            gradients[input_idx] = Some(grad);
                        } else if let Some(existing_grad) = &mut gradients[input_idx] {
                            *existing_grad = &*existing_grad + &grad;
                        }
                    }
                }
            }
        }

        // Return the forward tape with gradients computed
        Ok(ForwardTape {
            tensors: gradients,
            node_inputs: node_inputs_vec,
        })
    }
}
