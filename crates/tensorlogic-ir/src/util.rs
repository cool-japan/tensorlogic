//! Utility functions for the IR.
//!
//! This module provides helper functions for:
//! - Pretty printing expressions and graphs
//! - Computing IR statistics
//! - Formatting and display utilities

use std::fmt::{self, Write};

use crate::{EinsumGraph, TLExpr, Term};

/// Pretty-print a TLExpr to a string.
pub fn pretty_print_expr(expr: &TLExpr) -> String {
    let mut buffer = String::new();
    pretty_print_expr_inner(expr, &mut buffer, 0).unwrap();
    buffer
}

fn pretty_print_expr_inner(expr: &TLExpr, buf: &mut String, indent: usize) -> fmt::Result {
    let spaces = "  ".repeat(indent);

    match expr {
        TLExpr::Pred { name, args } => {
            write!(buf, "{}{}(", spaces, name)?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(buf, ", ")?;
                }
                write!(buf, "{}", term_to_string(arg))?;
            }
            writeln!(buf, ")")?;
        }
        TLExpr::And(l, r) => {
            writeln!(buf, "{}AND(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Or(l, r) => {
            writeln!(buf, "{}OR(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Not(e) => {
            writeln!(buf, "{}NOT(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Exists { var, domain, body } => {
            writeln!(buf, "{}∃{}:{}.(", spaces, var, domain)?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::ForAll { var, domain, body } => {
            writeln!(buf, "{}∀{}:{}.(", spaces, var, domain)?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            write!(buf, "{}AGG_{:?}({}:{}", spaces, op, var, domain)?;
            if let Some(group_vars) = group_by {
                write!(buf, " GROUP BY {:?}", group_vars)?;
            }
            writeln!(buf, ")(")?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Imply(premise, conclusion) => {
            writeln!(buf, "{}IMPLY(", spaces)?;
            pretty_print_expr_inner(premise, buf, indent + 1)?;
            writeln!(buf, "{}⇒", spaces)?;
            pretty_print_expr_inner(conclusion, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Score(e) => {
            writeln!(buf, "{}SCORE(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Add(l, r) => {
            writeln!(buf, "{}ADD(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Sub(l, r) => {
            writeln!(buf, "{}SUB(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Mul(l, r) => {
            writeln!(buf, "{}MUL(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Div(l, r) => {
            writeln!(buf, "{}DIV(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Pow(l, r) => {
            writeln!(buf, "{}POW(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Mod(l, r) => {
            writeln!(buf, "{}MOD(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Min(l, r) => {
            writeln!(buf, "{}MIN(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Max(l, r) => {
            writeln!(buf, "{}MAX(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Abs(e) => {
            writeln!(buf, "{}ABS(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Floor(e) => {
            writeln!(buf, "{}FLOOR(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Ceil(e) => {
            writeln!(buf, "{}CEIL(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Round(e) => {
            writeln!(buf, "{}ROUND(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Sqrt(e) => {
            writeln!(buf, "{}SQRT(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Exp(e) => {
            writeln!(buf, "{}EXP(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Log(e) => {
            writeln!(buf, "{}LOG(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Sin(e) => {
            writeln!(buf, "{}SIN(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Cos(e) => {
            writeln!(buf, "{}COS(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Tan(e) => {
            writeln!(buf, "{}TAN(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Box(e) => {
            writeln!(buf, "{}BOX(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Diamond(e) => {
            writeln!(buf, "{}DIAMOND(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Next(e) => {
            writeln!(buf, "{}NEXT(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Eventually(e) => {
            writeln!(buf, "{}EVENTUALLY(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Always(e) => {
            writeln!(buf, "{}ALWAYS(", spaces)?;
            pretty_print_expr_inner(e, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Until { before, after } => {
            writeln!(buf, "{}UNTIL(", spaces)?;
            pretty_print_expr_inner(before, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(after, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            writeln!(buf, "{}T-NORM_{:?}(", spaces, kind)?;
            pretty_print_expr_inner(left, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(right, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::TCoNorm { kind, left, right } => {
            writeln!(buf, "{}T-CONORM_{:?}(", spaces, kind)?;
            pretty_print_expr_inner(left, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(right, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::FuzzyNot { kind, expr } => {
            writeln!(buf, "{}FUZZY-NOT_{:?}(", spaces, kind)?;
            pretty_print_expr_inner(expr, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            writeln!(buf, "{}FUZZY-IMPLY_{:?}(", spaces, kind)?;
            pretty_print_expr_inner(premise, buf, indent + 1)?;
            writeln!(buf, "{}⇒", spaces)?;
            pretty_print_expr_inner(conclusion, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }

        // Probabilistic operators
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            writeln!(
                buf,
                "{}SOFT-∃{}:{}[T={}](",
                spaces, var, domain, temperature
            )?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            writeln!(
                buf,
                "{}SOFT-∀{}:{}[T={}](",
                spaces, var, domain, temperature
            )?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::WeightedRule { weight, rule } => {
            writeln!(buf, "{}WEIGHTED[{}](", spaces, weight)?;
            pretty_print_expr_inner(rule, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            writeln!(buf, "{}PROB-CHOICE[", spaces)?;
            for (i, (prob, expr)) in alternatives.iter().enumerate() {
                if i > 0 {
                    writeln!(buf, "{},", spaces)?;
                }
                writeln!(buf, "{}  {}: ", spaces, prob)?;
                pretty_print_expr_inner(expr, buf, indent + 2)?;
            }
            writeln!(buf, "{}]", spaces)?;
        }

        // Extended temporal logic
        TLExpr::Release { released, releaser } => {
            writeln!(buf, "{}RELEASE(", spaces)?;
            pretty_print_expr_inner(released, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(releaser, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::WeakUntil { before, after } => {
            writeln!(buf, "{}WEAK-UNTIL(", spaces)?;
            pretty_print_expr_inner(before, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(after, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::StrongRelease { released, releaser } => {
            writeln!(buf, "{}STRONG-RELEASE(", spaces)?;
            pretty_print_expr_inner(released, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(releaser, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }

        TLExpr::Eq(l, r) => {
            writeln!(buf, "{}EQ(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Lt(l, r) => {
            writeln!(buf, "{}LT(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Gt(l, r) => {
            writeln!(buf, "{}GT(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Lte(l, r) => {
            writeln!(buf, "{}LTE(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Gte(l, r) => {
            writeln!(buf, "{}GTE(", spaces)?;
            pretty_print_expr_inner(l, buf, indent + 1)?;
            writeln!(buf, "{},", spaces)?;
            pretty_print_expr_inner(r, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            writeln!(buf, "{}IF(", spaces)?;
            pretty_print_expr_inner(condition, buf, indent + 1)?;
            writeln!(buf, "{}) THEN(", spaces)?;
            pretty_print_expr_inner(then_branch, buf, indent + 1)?;
            writeln!(buf, "{}) ELSE(", spaces)?;
            pretty_print_expr_inner(else_branch, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Let { var, value, body } => {
            writeln!(buf, "{}LET {} =(", spaces, var)?;
            pretty_print_expr_inner(value, buf, indent + 1)?;
            writeln!(buf, "{}) IN(", spaces)?;
            pretty_print_expr_inner(body, buf, indent + 1)?;
            writeln!(buf, "{})", spaces)?;
        }
        TLExpr::Constant(value) => {
            writeln!(buf, "{}{}", spaces, value)?;
        }
    }

    Ok(())
}

fn term_to_string(term: &Term) -> String {
    match term {
        Term::Var(name) => format!("?{}", name),
        Term::Const(name) => name.clone(),
        Term::Typed {
            value,
            type_annotation,
        } => format!("{}:{}", term_to_string(value), type_annotation.type_name),
    }
}

/// Statistics about a TLExpr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprStats {
    /// Total number of nodes in the expression tree
    pub node_count: usize,
    /// Maximum depth of the expression tree
    pub max_depth: usize,
    /// Number of predicates
    pub predicate_count: usize,
    /// Number of quantifiers (exists + forall)
    pub quantifier_count: usize,
    /// Number of logical operators (and, or, not, imply)
    pub logical_op_count: usize,
    /// Number of arithmetic operators
    pub arithmetic_op_count: usize,
    /// Number of comparison operators
    pub comparison_op_count: usize,
    /// Number of free variables
    pub free_var_count: usize,
}

impl ExprStats {
    /// Compute statistics for an expression.
    pub fn compute(expr: &TLExpr) -> Self {
        let mut stats = ExprStats {
            node_count: 0,
            max_depth: 0,
            predicate_count: 0,
            quantifier_count: 0,
            logical_op_count: 0,
            arithmetic_op_count: 0,
            comparison_op_count: 0,
            free_var_count: expr.free_vars().len(),
        };

        stats.max_depth = Self::compute_recursive(expr, &mut stats, 0);
        stats
    }

    fn compute_recursive(expr: &TLExpr, stats: &mut ExprStats, depth: usize) -> usize {
        stats.node_count += 1;
        let mut max_child_depth = depth;

        match expr {
            TLExpr::Pred { .. } => {
                stats.predicate_count += 1;
            }
            TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
                stats.logical_op_count += 1;
                let left_depth = Self::compute_recursive(l, stats, depth + 1);
                let right_depth = Self::compute_recursive(r, stats, depth + 1);
                max_child_depth = left_depth.max(right_depth);
            }
            TLExpr::Not(e) | TLExpr::Score(e) => {
                stats.logical_op_count += 1;
                max_child_depth = Self::compute_recursive(e, stats, depth + 1);
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                stats.quantifier_count += 1;
                max_child_depth = Self::compute_recursive(body, stats, depth + 1);
            }
            TLExpr::Aggregate { body, .. } => {
                stats.quantifier_count += 1; // Aggregates are similar to quantifiers
                max_child_depth = Self::compute_recursive(body, stats, depth + 1);
            }
            TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r) => {
                stats.arithmetic_op_count += 1;
                let left_depth = Self::compute_recursive(l, stats, depth + 1);
                let right_depth = Self::compute_recursive(r, stats, depth + 1);
                max_child_depth = left_depth.max(right_depth);
            }
            TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e)
            | TLExpr::Box(e)
            | TLExpr::Diamond(e)
            | TLExpr::Next(e)
            | TLExpr::Eventually(e)
            | TLExpr::Always(e) => {
                stats.arithmetic_op_count += 1;
                max_child_depth = Self::compute_recursive(e, stats, depth + 1);
            }
            TLExpr::Until { before, after } => {
                stats.logical_op_count += 1;
                let depth_before = Self::compute_recursive(before, stats, depth + 1);
                let depth_after = Self::compute_recursive(after, stats, depth + 1);
                max_child_depth = depth_before.max(depth_after);
            }
            TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                stats.comparison_op_count += 1;
                let left_depth = Self::compute_recursive(l, stats, depth + 1);
                let right_depth = Self::compute_recursive(r, stats, depth + 1);
                max_child_depth = left_depth.max(right_depth);
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_depth = Self::compute_recursive(condition, stats, depth + 1);
                let then_depth = Self::compute_recursive(then_branch, stats, depth + 1);
                let else_depth = Self::compute_recursive(else_branch, stats, depth + 1);
                max_child_depth = cond_depth.max(then_depth).max(else_depth);
            }
            TLExpr::Let { value, body, .. } => {
                let value_depth = Self::compute_recursive(value, stats, depth + 1);
                let body_depth = Self::compute_recursive(body, stats, depth + 1);
                max_child_depth = value_depth.max(body_depth);
            }

            // Fuzzy logic operators
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                stats.logical_op_count += 1;
                let left_depth = Self::compute_recursive(left, stats, depth + 1);
                let right_depth = Self::compute_recursive(right, stats, depth + 1);
                max_child_depth = left_depth.max(right_depth);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                stats.logical_op_count += 1;
                max_child_depth = Self::compute_recursive(expr, stats, depth + 1);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                stats.logical_op_count += 1;
                let prem_depth = Self::compute_recursive(premise, stats, depth + 1);
                let conc_depth = Self::compute_recursive(conclusion, stats, depth + 1);
                max_child_depth = prem_depth.max(conc_depth);
            }

            // Probabilistic operators
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                stats.quantifier_count += 1;
                max_child_depth = Self::compute_recursive(body, stats, depth + 1);
            }
            TLExpr::WeightedRule { rule, .. } => {
                stats.logical_op_count += 1;
                max_child_depth = Self::compute_recursive(rule, stats, depth + 1);
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                stats.logical_op_count += 1;
                let mut max_alt_depth = depth;
                for (_, expr) in alternatives {
                    let alt_depth = Self::compute_recursive(expr, stats, depth + 1);
                    max_alt_depth = max_alt_depth.max(alt_depth);
                }
                max_child_depth = max_alt_depth;
            }

            // Extended temporal logic
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                stats.logical_op_count += 1;
                let rel_depth = Self::compute_recursive(released, stats, depth + 1);
                let reler_depth = Self::compute_recursive(releaser, stats, depth + 1);
                max_child_depth = rel_depth.max(reler_depth);
            }

            TLExpr::Constant(_) => {
                // Leaf node
            }
        }

        max_child_depth
    }
}

/// Statistics about an EinsumGraph.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    /// Number of tensors
    pub tensor_count: usize,
    /// Number of nodes
    pub node_count: usize,
    /// Number of output tensors
    pub output_count: usize,
    /// Number of einsum operations
    pub einsum_count: usize,
    /// Number of element-wise unary operations
    pub elem_unary_count: usize,
    /// Number of element-wise binary operations
    pub elem_binary_count: usize,
    /// Number of reduction operations
    pub reduce_count: usize,
    /// Average inputs per node
    pub avg_inputs_per_node: f64,
}

impl GraphStats {
    /// Compute statistics for a graph.
    pub fn compute(graph: &EinsumGraph) -> Self {
        let mut stats = GraphStats {
            tensor_count: graph.tensors.len(),
            node_count: graph.nodes.len(),
            output_count: graph.outputs.len(),
            einsum_count: 0,
            elem_unary_count: 0,
            elem_binary_count: 0,
            reduce_count: 0,
            avg_inputs_per_node: 0.0,
        };

        let mut total_inputs = 0;

        for node in &graph.nodes {
            total_inputs += node.inputs.len();

            match &node.op {
                crate::graph::OpType::Einsum { .. } => stats.einsum_count += 1,
                crate::graph::OpType::ElemUnary { .. } => stats.elem_unary_count += 1,
                crate::graph::OpType::ElemBinary { .. } => stats.elem_binary_count += 1,
                crate::graph::OpType::Reduce { .. } => stats.reduce_count += 1,
            }
        }

        if stats.node_count > 0 {
            stats.avg_inputs_per_node = total_inputs as f64 / stats.node_count as f64;
        }

        stats
    }
}

/// Pretty-print a graph to a string.
pub fn pretty_print_graph(graph: &EinsumGraph) -> String {
    let mut buffer = String::new();
    writeln!(buffer, "EinsumGraph {{").unwrap();
    writeln!(buffer, "  Tensors: {} total", graph.tensors.len()).unwrap();

    for (idx, name) in graph.tensors.iter().enumerate() {
        writeln!(buffer, "    t{}: {}", idx, name).unwrap();
    }

    writeln!(buffer, "  Nodes: {} total", graph.nodes.len()).unwrap();
    for (idx, node) in graph.nodes.iter().enumerate() {
        write!(buffer, "    n{}: ", idx).unwrap();
        match &node.op {
            crate::graph::OpType::Einsum { spec } => {
                write!(buffer, "Einsum(\"{}\")", spec).unwrap()
            }
            crate::graph::OpType::ElemUnary { op } => write!(buffer, "ElemUnary({})", op).unwrap(),
            crate::graph::OpType::ElemBinary { op } => {
                write!(buffer, "ElemBinary({})", op).unwrap()
            }
            crate::graph::OpType::Reduce { op, axes } => {
                write!(buffer, "Reduce({}, axes={:?})", op, axes).unwrap()
            }
        }
        write!(buffer, " <- [").unwrap();
        for (i, input) in node.inputs.iter().enumerate() {
            if i > 0 {
                write!(buffer, ", ").unwrap();
            }
            write!(buffer, "t{}", input).unwrap();
        }
        writeln!(buffer, "]").unwrap();
    }

    writeln!(buffer, "  Outputs: {:?}", graph.outputs).unwrap();
    writeln!(buffer, "}}").unwrap();

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_stats_simple() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let stats = ExprStats::compute(&expr);

        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.predicate_count, 1);
        assert_eq!(stats.quantifier_count, 0);
        assert_eq!(stats.free_var_count, 1);
    }

    #[test]
    fn test_expr_stats_complex() {
        // ∀x. P(x) ∧ Q(x)
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("x")]);
        let and_expr = TLExpr::and(p, q);
        let expr = TLExpr::forall("x", "Domain", and_expr);

        let stats = ExprStats::compute(&expr);

        assert_eq!(stats.node_count, 4); // forall, and, p, q
        assert_eq!(stats.predicate_count, 2);
        assert_eq!(stats.quantifier_count, 1);
        assert_eq!(stats.logical_op_count, 1);
        assert_eq!(stats.free_var_count, 0); // x is bound
    }

    #[test]
    fn test_expr_stats_arithmetic() {
        // score(x) * 2 + 1
        let score = TLExpr::pred("score", vec![Term::var("x")]);
        let mul = TLExpr::mul(score, TLExpr::constant(2.0));
        let add = TLExpr::add(mul, TLExpr::constant(1.0));

        let stats = ExprStats::compute(&add);

        assert_eq!(stats.arithmetic_op_count, 2); // mul, add
        assert_eq!(stats.predicate_count, 1);
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input");
        let t1 = graph.add_tensor("output");

        graph
            .add_node(crate::EinsumNode {
                inputs: vec![t0],
                outputs: vec![t1],
                op: crate::graph::OpType::Einsum {
                    spec: "i->i".to_string(),
                },
                metadata: None,
            })
            .unwrap();

        graph.add_output(t1).unwrap();

        let stats = GraphStats::compute(&graph);

        assert_eq!(stats.tensor_count, 2);
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.einsum_count, 1);
        assert_eq!(stats.avg_inputs_per_node, 1.0);
    }

    #[test]
    fn test_pretty_print_expr() {
        let expr = TLExpr::pred("Person", vec![Term::var("x")]);
        let output = pretty_print_expr(&expr);
        assert!(output.contains("Person(?x)"));
    }

    #[test]
    fn test_pretty_print_graph() {
        let mut graph = EinsumGraph::new();
        let _t0 = graph.add_tensor("input");

        let output = pretty_print_graph(&graph);
        assert!(output.contains("t0: input"));
        assert!(output.contains("Tensors: 1 total"));
    }
}
