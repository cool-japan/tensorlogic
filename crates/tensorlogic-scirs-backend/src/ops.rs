//! Operation parsing utilities.

use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp};

/// Parse element-wise operation from string.
pub(crate) fn parse_elem_op(op: &str) -> Result<ElemOp, ExecutorError> {
    match op.to_lowercase().as_str() {
        // Activation functions
        "relu" => Ok(ElemOp::Relu),
        "sigmoid" => Ok(ElemOp::Sigmoid),

        // Unary operations
        "oneminus" | "one_minus" => Ok(ElemOp::OneMinus),

        // Arithmetic binary operations
        "add" => Ok(ElemOp::Add),
        "subtract" | "sub" => Ok(ElemOp::Subtract),
        "multiply" | "mul" => Ok(ElemOp::Multiply),
        "divide" | "div" => Ok(ElemOp::Divide),
        "min" => Ok(ElemOp::Min),
        "max" => Ok(ElemOp::Max),

        // Comparison operations
        "eq" | "equal" => Ok(ElemOp::Eq),
        "lt" | "lessthan" => Ok(ElemOp::Lt),
        "gt" | "greaterthan" => Ok(ElemOp::Gt),
        "lte" | "lessthanorequal" => Ok(ElemOp::Lte),
        "gte" | "greaterthanorequal" => Ok(ElemOp::Gte),

        // Extended logical operations
        "or_max" | "ormax" => Ok(ElemOp::OrMax),
        "or_prob_sum" | "orprobsum" | "or_probabilistic" => Ok(ElemOp::OrProbSum),
        "nand" => Ok(ElemOp::Nand),
        "nor" => Ok(ElemOp::Nor),
        "xor" => Ok(ElemOp::Xor),

        _ => Err(ExecutorError::UnsupportedOperation(format!(
            "Unknown element-wise operation: {}",
            op
        ))),
    }
}

/// Parse reduction operation from string.
pub(crate) fn parse_reduce_op(op: &str) -> Result<ReduceOp, ExecutorError> {
    match op.to_lowercase().as_str() {
        "sum" => Ok(ReduceOp::Sum),
        "max" => Ok(ReduceOp::Max),
        "min" => Ok(ReduceOp::Min),
        "mean" => Ok(ReduceOp::Mean),
        "product" | "prod" => Ok(ReduceOp::Product),
        _ => Err(ExecutorError::UnsupportedOperation(format!(
            "Unknown reduction operation: {}",
            op
        ))),
    }
}
