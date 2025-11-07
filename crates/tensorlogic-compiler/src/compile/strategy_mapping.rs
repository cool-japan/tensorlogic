//! Strategy mapping utilities.
//!
//! Maps compilation strategies to actual tensor operations.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::config::{AndStrategy, NotStrategy, OrStrategy};
use crate::context::CompilerContext;

/// Compile AND operation based on strategy.
pub(crate) fn compile_and_with_strategy(
    left_idx: usize,
    right_idx: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<usize> {
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match ctx.config.and_strategy {
        AndStrategy::Product | AndStrategy::ProductTNorm => {
            // a * b
            let node = EinsumNode::elem_binary("multiply", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        AndStrategy::Min | AndStrategy::Godel => {
            // min(a, b)
            let node = EinsumNode::elem_binary("min", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        AndStrategy::ProbabilisticSum => {
            // a + b - a*b
            // First compute a*b
            let mult_name = ctx.fresh_temp();
            let mult_idx = graph.add_tensor(mult_name);
            let mult_node = EinsumNode::elem_binary("multiply", left_idx, right_idx, mult_idx);
            graph.add_node(mult_node)?;

            // Then compute a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Finally compute (a + b) - (a*b)
            let node = EinsumNode::elem_binary("subtract", sum_idx, mult_idx, result_idx);
            graph.add_node(node)?;
        }
        AndStrategy::Lukasiewicz => {
            // max(0, a + b - 1)
            // First compute a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Create constant 1
            let one_name = "const_1.0".to_string();
            let one_idx = if !graph.tensors.contains(&one_name) {
                graph.add_tensor(one_name)
            } else {
                graph.tensors.iter().position(|t| t == "const_1.0").unwrap()
            };

            // Compute (a + b) - 1
            let sub_name = ctx.fresh_temp();
            let sub_idx = graph.add_tensor(sub_name);
            let sub_node = EinsumNode::elem_binary("subtract", sum_idx, one_idx, sub_idx);
            graph.add_node(sub_node)?;

            // Apply ReLU to get max(0, x)
            let node = EinsumNode::elem_unary("relu", sub_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(result_idx)
}

/// Compile OR operation based on strategy.
pub(crate) fn compile_or_with_strategy(
    left_idx: usize,
    right_idx: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<usize> {
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match ctx.config.or_strategy {
        OrStrategy::Max | OrStrategy::Godel => {
            // max(a, b)
            let node = EinsumNode::elem_binary("max", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        OrStrategy::ProbabilisticSum | OrStrategy::ProbabilisticSNorm => {
            // a + b - a*b (same as Or_ProbSum operation)
            let node = EinsumNode::elem_binary("or_prob_sum", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        OrStrategy::Lukasiewicz => {
            // min(1, a + b)
            // First compute a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Create constant 1
            let one_name = "const_1.0".to_string();
            let one_idx = if !graph.tensors.contains(&one_name) {
                graph.add_tensor(one_name)
            } else {
                graph.tensors.iter().position(|t| t == "const_1.0").unwrap()
            };

            // Compute min(1, a + b)
            let node = EinsumNode::elem_binary("min", one_idx, sum_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(result_idx)
}

/// Compile NOT operation based on strategy.
pub(crate) fn compile_not_with_strategy(
    input_idx: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<usize> {
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match ctx.config.not_strategy {
        NotStrategy::Complement => {
            // 1 - a
            let node = EinsumNode::elem_unary("one_minus", input_idx, result_idx);
            graph.add_node(node)?;
        }
        NotStrategy::Sigmoid { temperature } => {
            // 1 / (1 + exp(T * a))
            // Implemented as: sigmoid(-T * a)
            // Since sigmoid(x) = 1/(1+exp(-x)), we have sigmoid(-T*a) = 1/(1+exp(T*a))

            if temperature == 1 {
                // Optimize for T=1: just negate and apply sigmoid
                let neg_name = ctx.fresh_temp();
                let neg_idx = graph.add_tensor(neg_name);
                let neg_node = EinsumNode::elem_unary("negate", input_idx, neg_idx);
                graph.add_node(neg_node)?;

                let node = EinsumNode::elem_unary("sigmoid", neg_idx, result_idx);
                graph.add_node(node)?;
            } else {
                // General case: multiply by temperature, negate, then sigmoid
                // Create constant for temperature
                let temp_f64 = temperature as f64;
                let temp_name = format!("const_{}", temp_f64);
                let temp_idx = if !graph.tensors.contains(&temp_name) {
                    graph.add_tensor(temp_name.clone())
                } else {
                    graph.tensors.iter().position(|t| t == &temp_name).unwrap()
                };

                // Multiply input by temperature: T * a
                let scaled_name = ctx.fresh_temp();
                let scaled_idx = graph.add_tensor(scaled_name);
                let scale_node =
                    EinsumNode::elem_binary("multiply", temp_idx, input_idx, scaled_idx);
                graph.add_node(scale_node)?;

                // Negate: -(T * a)
                let neg_name = ctx.fresh_temp();
                let neg_idx = graph.add_tensor(neg_name);
                let neg_node = EinsumNode::elem_unary("negate", scaled_idx, neg_idx);
                graph.add_node(neg_node)?;

                // Apply sigmoid: sigmoid(-(T * a)) = 1/(1 + exp(T * a))
                let node = EinsumNode::elem_unary("sigmoid", neg_idx, result_idx);
                graph.add_node(node)?;
            }
        }
    }

    Ok(result_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CompilationConfigBuilder;

    #[test]
    fn test_sigmoid_not_with_temperature_1() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .not_strategy(NotStrategy::Sigmoid { temperature: 1 })
                .build(),
        );
        let mut graph = EinsumGraph::new();

        // Create input tensor
        let input_idx = graph.add_tensor("input");

        // Compile NOT with temperature=1
        let result_idx = compile_not_with_strategy(input_idx, &mut ctx, &mut graph).unwrap();

        // Should create: negate -> sigmoid
        assert!(result_idx > input_idx);
        assert_eq!(graph.nodes.len(), 2); // negate + sigmoid

        // First node should be negate
        assert_eq!(graph.nodes[0].operation_description(), "ElemUnary(negate)");
        // Second node should be sigmoid
        assert_eq!(graph.nodes[1].operation_description(), "ElemUnary(sigmoid)");
    }

    #[test]
    fn test_sigmoid_not_with_temperature_2() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .not_strategy(NotStrategy::Sigmoid { temperature: 2 })
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let input_idx = graph.add_tensor("input");

        // Compile NOT with temperature=2
        let result_idx = compile_not_with_strategy(input_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > input_idx);
        // Should create: multiply (temp) -> negate -> sigmoid
        assert_eq!(graph.nodes.len(), 3);

        // Check operations in order
        assert_eq!(
            graph.nodes[0].operation_description(),
            "ElemBinary(multiply)"
        );
        assert_eq!(graph.nodes[1].operation_description(), "ElemUnary(negate)");
        assert_eq!(graph.nodes[2].operation_description(), "ElemUnary(sigmoid)");

        // Check that temperature constant was created
        assert!(graph.tensors.contains(&"const_2".to_string()));
    }

    #[test]
    fn test_sigmoid_not_with_temperature_10() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .not_strategy(NotStrategy::Sigmoid { temperature: 10 })
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let input_idx = graph.add_tensor("input");

        let result_idx = compile_not_with_strategy(input_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > input_idx);
        assert_eq!(graph.nodes.len(), 3);

        // Check that temperature constant was created
        assert!(graph.tensors.contains(&"const_10".to_string()));
    }

    #[test]
    fn test_complement_not_strategy() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .not_strategy(NotStrategy::Complement)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let input_idx = graph.add_tensor("input");

        let result_idx = compile_not_with_strategy(input_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > input_idx);
        assert_eq!(graph.nodes.len(), 1); // Just one_minus
        assert_eq!(
            graph.nodes[0].operation_description(),
            "ElemUnary(one_minus)"
        );
    }

    #[test]
    fn test_and_strategy_product() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .and_strategy(AndStrategy::Product)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let left_idx = graph.add_tensor("left");
        let right_idx = graph.add_tensor("right");

        let result_idx =
            compile_and_with_strategy(left_idx, right_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > right_idx);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(
            graph.nodes[0].operation_description(),
            "ElemBinary(multiply)"
        );
    }

    #[test]
    fn test_and_strategy_min() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .and_strategy(AndStrategy::Min)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let left_idx = graph.add_tensor("left");
        let right_idx = graph.add_tensor("right");

        let result_idx =
            compile_and_with_strategy(left_idx, right_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > right_idx);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].operation_description(), "ElemBinary(min)");
    }

    #[test]
    fn test_and_strategy_lukasiewicz() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .and_strategy(AndStrategy::Lukasiewicz)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let left_idx = graph.add_tensor("left");
        let right_idx = graph.add_tensor("right");

        let result_idx =
            compile_and_with_strategy(left_idx, right_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > right_idx);
        // add + subtract + relu = 3 operations
        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.tensors.contains(&"const_1.0".to_string()));
    }

    #[test]
    fn test_or_strategy_max() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .or_strategy(OrStrategy::Max)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let left_idx = graph.add_tensor("left");
        let right_idx = graph.add_tensor("right");

        let result_idx =
            compile_or_with_strategy(left_idx, right_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > right_idx);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].operation_description(), "ElemBinary(max)");
    }

    #[test]
    fn test_or_strategy_lukasiewicz() {
        let mut ctx = CompilerContext::with_config(
            CompilationConfigBuilder::default()
                .or_strategy(OrStrategy::Lukasiewicz)
                .build(),
        );
        let mut graph = EinsumGraph::new();

        let left_idx = graph.add_tensor("left");
        let right_idx = graph.add_tensor("right");

        let result_idx =
            compile_or_with_strategy(left_idx, right_idx, &mut ctx, &mut graph).unwrap();

        assert!(result_idx > right_idx);
        // add + min = 2 operations
        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.tensors.contains(&"const_1.0".to_string()));
    }
}
