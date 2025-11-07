//! Python wrappers for TensorLogic IR types

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use tensorlogic_ir::{EinsumGraph, TLExpr, Term};

/// Python wrapper for Term (variables and constants)
#[pyclass(name = "Term")]
#[derive(Clone)]
pub struct PyTerm {
    pub inner: Term,
}

#[pymethods]
impl PyTerm {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match &self.inner {
            Term::Var(name) => name.clone(),
            Term::Const(name) => name.clone(),
            Term::Typed {
                value,
                type_annotation,
            } => {
                format!("{}:{}", value.name(), type_annotation.type_name)
            }
        }
    }

    /// Get the name of the term
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Check if this is a variable
    fn is_var(&self) -> bool {
        self.inner.is_var()
    }

    /// Check if this is a constant
    fn is_const(&self) -> bool {
        self.inner.is_const()
    }
}

/// Python wrapper for TLExpr (logic expressions)
#[pyclass(name = "TLExpr")]
#[derive(Clone)]
pub struct PyTLExpr {
    pub inner: TLExpr,
}

#[pymethods]
impl PyTLExpr {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        tensorlogic_ir::pretty_print_expr(&self.inner)
    }

    /// Combine two expressions with AND
    fn and(&self, other: &PyTLExpr) -> PyTLExpr {
        PyTLExpr {
            inner: TLExpr::and(self.inner.clone(), other.inner.clone()),
        }
    }

    /// Combine two expressions with OR
    fn or(&self, other: &PyTLExpr) -> PyTLExpr {
        PyTLExpr {
            inner: TLExpr::or(self.inner.clone(), other.inner.clone()),
        }
    }

    /// Negate this expression
    fn negate(&self) -> PyTLExpr {
        PyTLExpr {
            inner: TLExpr::negate(self.inner.clone()),
        }
    }

    /// Get the list of free variables in this expression
    fn free_vars(&self) -> Vec<String> {
        self.inner.free_vars().into_iter().collect()
    }

    // Operator overloading for Python-native syntax

    /// Implement & operator (AND)
    ///
    /// Example:
    ///     >>> expr1 & expr2  # Same as and_(expr1, expr2)
    fn __and__(&self, other: &PyTLExpr) -> PyTLExpr {
        self.and(other)
    }

    /// Implement | operator (OR)
    ///
    /// Example:
    ///     >>> expr1 | expr2  # Same as or_(expr1, expr2)
    fn __or__(&self, other: &PyTLExpr) -> PyTLExpr {
        self.or(other)
    }

    /// Implement ~ operator (NOT)
    ///
    /// Example:
    ///     >>> ~expr  # Same as not_(expr)
    fn __invert__(&self) -> PyTLExpr {
        self.negate()
    }

    /// Implement >> operator (IMPLY)
    ///
    /// Example:
    ///     >>> premise >> conclusion  # Same as imply(premise, conclusion)
    fn __rshift__(&self, other: &PyTLExpr) -> PyTLExpr {
        PyTLExpr {
            inner: TLExpr::Imply(Box::new(self.inner.clone()), Box::new(other.inner.clone())),
        }
    }
}

/// Python wrapper for EinsumGraph (compiled tensor computation graph)
#[pyclass(name = "EinsumGraph")]
#[derive(Clone)]
pub struct PyEinsumGraph {
    pub inner: EinsumGraph,
}

#[pymethods]
impl PyEinsumGraph {
    fn __repr__(&self) -> String {
        format!(
            "EinsumGraph(nodes={}, outputs={})",
            self.inner.nodes.len(),
            self.inner.outputs.len()
        )
    }

    fn __str__(&self) -> String {
        tensorlogic_ir::pretty_print_graph(&self.inner)
    }

    /// Get the number of nodes in the graph
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    /// Get the number of outputs
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.outputs.len()
    }

    /// Get list of node indices
    #[getter]
    fn nodes(&self) -> Vec<usize> {
        (0..self.inner.nodes.len()).collect()
    }

    /// Get graph statistics
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = tensorlogic_ir::GraphStats::compute(&self.inner);
        let dict = PyDict::new(py);
        dict.set_item("num_tensors", stats.tensor_count)?;
        dict.set_item("num_nodes", stats.node_count)?;
        dict.set_item("num_outputs", stats.output_count)?;
        dict.set_item("num_einsum", stats.einsum_count)?;
        dict.set_item("num_elem_unary", stats.elem_unary_count)?;
        dict.set_item("num_elem_binary", stats.elem_binary_count)?;
        dict.set_item("num_reduce", stats.reduce_count)?;
        dict.set_item("avg_inputs_per_node", stats.avg_inputs_per_node)?;
        Ok(dict)
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::einsum_graph_html;
        use std::collections::HashMap;

        let stats = tensorlogic_ir::GraphStats::compute(&self.inner);

        // Count node types
        let mut node_types = HashMap::new();
        *node_types.entry("Input".to_string()).or_insert(0) += stats.tensor_count;
        *node_types.entry("Einsum".to_string()).or_insert(0) += stats.einsum_count;
        *node_types
            .entry("ElementwiseUnary".to_string())
            .or_insert(0) += stats.elem_unary_count;
        *node_types
            .entry("ElementwiseBinary".to_string())
            .or_insert(0) += stats.elem_binary_count;
        *node_types.entry("Reduce".to_string()).or_insert(0) += stats.reduce_count;

        einsum_graph_html(
            self.inner.nodes.len(),
            stats.tensor_count,
            self.inner.outputs.len(),
            &node_types,
        )
    }
}

// Helper functions for creating terms and expressions

/// Create a variable term
///
/// Args:
///     name: The variable name
///
/// Returns:
///     A Term representing a variable
///
/// Example:
///     >>> x = var("x")
#[pyfunction(name = "var")]
pub fn py_var(name: String) -> PyTerm {
    PyTerm {
        inner: Term::var(name),
    }
}

/// Create a constant term
///
/// Args:
///     name: The constant name
///
/// Returns:
///     A Term representing a constant
///
/// Example:
///     >>> alice = const("alice")
#[pyfunction(name = "const")]
pub fn py_const(name: String) -> PyTerm {
    PyTerm {
        inner: Term::constant(name),
    }
}

/// Create a predicate expression
///
/// Args:
///     name: The predicate name
///     args: List of Term arguments
///
/// Returns:
///     A TLExpr representing the predicate
///
/// Example:
///     >>> knows = pred("knows", [var("x"), var("y")])
#[pyfunction(name = "pred")]
pub fn py_pred(name: String, args: Vec<PyTerm>) -> PyTLExpr {
    let terms: Vec<Term> = args.into_iter().map(|t| t.inner).collect();
    PyTLExpr {
        inner: TLExpr::pred(name, terms),
    }
}

/// Create an AND expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left AND right)
///
/// Example:
///     >>> and_expr = and_(expr1, expr2)
#[pyfunction(name = "and_")]
pub fn py_and(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::and(left.inner.clone(), right.inner.clone()),
    }
}

/// Create an OR expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left OR right)
///
/// Example:
///     >>> or_expr = or_(expr1, expr2)
#[pyfunction(name = "or_")]
pub fn py_or(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::or(left.inner.clone(), right.inner.clone()),
    }
}

/// Create a NOT expression
///
/// Args:
///     expr: Expression to negate
///
/// Returns:
///     A TLExpr representing (NOT expr)
///
/// Example:
///     >>> not_expr = not_(expr)
#[pyfunction(name = "not_")]
pub fn py_not(expr: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::negate(expr.inner.clone()),
    }
}

/// Create an EXISTS quantifier
///
/// Args:
///     var: Variable name to quantify over
///     domain: Domain name for the variable
///     body: Body expression
///
/// Returns:
///     A TLExpr representing (EXISTS var IN domain . body)
///
/// Example:
///     >>> exists_expr = exists("y", "Person", pred("knows", [var("x"), var("y")]))
#[pyfunction(name = "exists")]
pub fn py_exists(var: String, domain: String, body: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::exists(var, domain, body.inner.clone()),
    }
}

/// Create a FORALL quantifier
///
/// Args:
///     var: Variable name to quantify over
///     domain: Domain name for the variable
///     body: Body expression
///
/// Returns:
///     A TLExpr representing (FORALL var IN domain . body)
///
/// Example:
///     >>> forall_expr = forall("y", "Person", pred("knows", [var("x"), var("y")]))
#[pyfunction(name = "forall")]
pub fn py_forall(var: String, domain: String, body: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::forall(var, domain, body.inner.clone()),
    }
}

/// Create an implication expression
///
/// Args:
///     premise: Premise expression
///     conclusion: Conclusion expression
///
/// Returns:
///     A TLExpr representing (premise => conclusion)
///
/// Example:
///     >>> imply_expr = imply(premise, conclusion)
#[pyfunction(name = "imply")]
pub fn py_imply(premise: &PyTLExpr, conclusion: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::imply(premise.inner.clone(), conclusion.inner.clone()),
    }
}

/// Create a constant numeric expression
///
/// Args:
///     value: The numeric constant value
///
/// Returns:
///     A TLExpr representing the constant
///
/// Example:
///     >>> const_expr = constant(3.14)
#[pyfunction(name = "constant")]
pub fn py_constant(value: f64) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Constant(value),
    }
}

/// Create an addition expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left + right)
///
/// Example:
///     >>> add_expr = add(expr1, expr2)
#[pyfunction(name = "add")]
pub fn py_add(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Add(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a subtraction expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left - right)
///
/// Example:
///     >>> sub_expr = sub(expr1, expr2)
#[pyfunction(name = "sub")]
pub fn py_sub(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Sub(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a multiplication expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left * right)
///
/// Example:
///     >>> mul_expr = mul(expr1, expr2)
#[pyfunction(name = "mul")]
pub fn py_mul(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Mul(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a division expression
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left / right)
///
/// Example:
///     >>> div_expr = div(expr1, expr2)
#[pyfunction(name = "div")]
pub fn py_div(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Div(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create an equality comparison
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left == right)
///
/// Example:
///     >>> eq_expr = eq(expr1, expr2)
#[pyfunction(name = "eq")]
pub fn py_eq(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Eq(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a less-than comparison
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left < right)
///
/// Example:
///     >>> lt_expr = lt(expr1, expr2)
#[pyfunction(name = "lt")]
pub fn py_lt(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Lt(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a greater-than comparison
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left > right)
///
/// Example:
///     >>> gt_expr = gt(expr1, expr2)
#[pyfunction(name = "gt")]
pub fn py_gt(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Gt(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a less-than-or-equal comparison
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left <= right)
///
/// Example:
///     >>> lte_expr = lte(expr1, expr2)
#[pyfunction(name = "lte")]
pub fn py_lte(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Lte(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a greater-than-or-equal comparison
///
/// Args:
///     left: First expression
///     right: Second expression
///
/// Returns:
///     A TLExpr representing (left >= right)
///
/// Example:
///     >>> gte_expr = gte(expr1, expr2)
#[pyfunction(name = "gte")]
pub fn py_gte(left: &PyTLExpr, right: &PyTLExpr) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::Gte(Box::new(left.inner.clone()), Box::new(right.inner.clone())),
    }
}

/// Create a conditional if-then-else expression
///
/// Args:
///     condition: Boolean condition expression
///     then_branch: Expression to evaluate if condition is true
///     else_branch: Expression to evaluate if condition is false
///
/// Returns:
///     A TLExpr representing (if condition then then_branch else else_branch)
///
/// Example:
///     >>> result = if_then_else(condition, then_expr, else_expr)
#[pyfunction(name = "if_then_else")]
pub fn py_if_then_else(
    condition: &PyTLExpr,
    then_branch: &PyTLExpr,
    else_branch: &PyTLExpr,
) -> PyTLExpr {
    PyTLExpr {
        inner: TLExpr::IfThenElse {
            condition: Box::new(condition.inner.clone()),
            then_branch: Box::new(then_branch.inner.clone()),
            else_branch: Box::new(else_branch.inner.clone()),
        },
    }
}
