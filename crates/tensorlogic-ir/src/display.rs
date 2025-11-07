//! Display trait implementations for IR types.
//!
//! Provides human-readable string representations for debugging and error messages.

use std::fmt;

use crate::{
    expr::{AggregateOp, TLExpr},
    graph::{EinsumGraph, EinsumNode, OpType},
    term::Term,
};

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(name) => write!(f, "?{}", name),
            Term::Const(name) => write!(f, "{}", name),
            Term::Typed {
                value,
                type_annotation,
            } => write!(f, "{}:{}", value, type_annotation.type_name),
        }
    }
}

impl fmt::Display for AggregateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateOp::Count => write!(f, "COUNT"),
            AggregateOp::Sum => write!(f, "SUM"),
            AggregateOp::Average => write!(f, "AVG"),
            AggregateOp::Max => write!(f, "MAX"),
            AggregateOp::Min => write!(f, "MIN"),
            AggregateOp::Product => write!(f, "PROD"),
            AggregateOp::Any => write!(f, "ANY"),
            AggregateOp::All => write!(f, "ALL"),
        }
    }
}

impl fmt::Display for TLExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TLExpr::Pred { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            TLExpr::And(l, r) => write!(f, "({} ∧ {})", l, r),
            TLExpr::Or(l, r) => write!(f, "({} ∨ {})", l, r),
            TLExpr::Not(e) => write!(f, "¬{}", e),
            TLExpr::Exists { var, domain, body } => {
                write!(f, "∃{}:{}. {}", var, domain, body)
            }
            TLExpr::ForAll { var, domain, body } => {
                write!(f, "∀{}:{}. {}", var, domain, body)
            }
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                write!(f, "{}({}:{}, ", op, var, domain)?;
                if let Some(group_vars) = group_by {
                    write!(f, "GROUP BY [")?;
                    for (i, gv) in group_vars.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", gv)?;
                    }
                    write!(f, "], ")?;
                }
                write!(f, "{})", body)
            }
            TLExpr::Imply(premise, conclusion) => write!(f, "({} → {})", premise, conclusion),
            TLExpr::Score(e) => write!(f, "score({})", e),
            TLExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            TLExpr::Sub(l, r) => write!(f, "({} - {})", l, r),
            TLExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            TLExpr::Div(l, r) => write!(f, "({} / {})", l, r),
            TLExpr::Pow(l, r) => write!(f, "({} ^ {})", l, r),
            TLExpr::Mod(l, r) => write!(f, "({} % {})", l, r),
            TLExpr::Min(l, r) => write!(f, "min({}, {})", l, r),
            TLExpr::Max(l, r) => write!(f, "max({}, {})", l, r),
            TLExpr::Abs(e) => write!(f, "abs({})", e),
            TLExpr::Floor(e) => write!(f, "floor({})", e),
            TLExpr::Ceil(e) => write!(f, "ceil({})", e),
            TLExpr::Round(e) => write!(f, "round({})", e),
            TLExpr::Sqrt(e) => write!(f, "sqrt({})", e),
            TLExpr::Exp(e) => write!(f, "exp({})", e),
            TLExpr::Log(e) => write!(f, "log({})", e),
            TLExpr::Sin(e) => write!(f, "sin({})", e),
            TLExpr::Cos(e) => write!(f, "cos({})", e),
            TLExpr::Tan(e) => write!(f, "tan({})", e),
            TLExpr::Eq(l, r) => write!(f, "({} = {})", l, r),
            TLExpr::Lt(l, r) => write!(f, "({} < {})", l, r),
            TLExpr::Gt(l, r) => write!(f, "({} > {})", l, r),
            TLExpr::Lte(l, r) => write!(f, "({} ≤ {})", l, r),
            TLExpr::Gte(l, r) => write!(f, "({} ≥ {})", l, r),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => write!(
                f,
                "if {} then {} else {}",
                condition, then_branch, else_branch
            ),
            TLExpr::Let { var, value, body } => {
                write!(f, "let {} = {} in {}", var, value, body)
            }
            TLExpr::Box(e) => write!(f, "□{}", e),
            TLExpr::Diamond(e) => write!(f, "◇{}", e),
            TLExpr::Next(e) => write!(f, "X{}", e),
            TLExpr::Eventually(e) => write!(f, "F{}", e),
            TLExpr::Always(e) => write!(f, "G{}", e),
            TLExpr::Until { before, after } => write!(f, "({} U {})", before, after),
            // Fuzzy logic operators
            TLExpr::TNorm { kind, left, right } => {
                write!(f, "({} ⊤_{:?} {})", left, kind, right)
            }
            TLExpr::TCoNorm { kind, left, right } => {
                write!(f, "({} ⊥_{:?} {})", left, kind, right)
            }
            TLExpr::FuzzyNot { kind, expr } => write!(f, "¬_{:?}({})", kind, expr),
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => write!(f, "({} →_{:?} {})", premise, kind, conclusion),
            // Probabilistic operators
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => write!(f, "∃^{{{}}}{}:{}. {}", temperature, var, domain, body),
            TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature,
            } => write!(f, "∀^{{{}}}{}:{}. {}", temperature, var, domain, body),
            TLExpr::WeightedRule { weight, rule } => write!(f, "{}::{}", weight, rule),
            TLExpr::ProbabilisticChoice { alternatives } => {
                write!(f, "choice[")?;
                for (i, (prob, expr)) in alternatives.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", prob, expr)?;
                }
                write!(f, "]")
            }
            // Extended temporal logic
            TLExpr::Release { released, releaser } => write!(f, "({} R {})", released, releaser),
            TLExpr::WeakUntil { before, after } => write!(f, "({} W {})", before, after),
            TLExpr::StrongRelease { released, releaser } => {
                write!(f, "({} M {})", released, releaser)
            }
            TLExpr::Constant(value) => write!(f, "{}", value),
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Einsum { spec } => write!(f, "einsum({})", spec),
            OpType::ElemUnary { op } => write!(f, "{}(·)", op),
            OpType::ElemBinary { op } => write!(f, "{}(·, ·)", op),
            OpType::Reduce { op, axes } => write!(f, "{}(·, axes={:?})", op, axes),
        }
    }
}

impl fmt::Display for EinsumNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ", self.op)?;
        write!(f, "inputs={:?}", self.inputs)?;
        write!(f, " outputs={:?}", self.outputs)
    }
}

impl fmt::Display for EinsumGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EinsumGraph {{")?;
        writeln!(f, "  tensors: {:?}", self.tensors)?;
        writeln!(f, "  nodes: [")?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "    {}: {}", i, node)?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "  outputs: {:?}", self.outputs)?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_term() {
        let var = Term::var("x");
        assert_eq!(format!("{}", var), "?x");

        let const_term = Term::constant("alice");
        assert_eq!(format!("{}", const_term), "alice");

        let typed = Term::typed_var("x", "Int");
        assert_eq!(format!("{}", typed), "?x:Int");
    }

    #[test]
    fn test_display_aggregate_op() {
        assert_eq!(format!("{}", AggregateOp::Count), "COUNT");
        assert_eq!(format!("{}", AggregateOp::Sum), "SUM");
        assert_eq!(format!("{}", AggregateOp::Average), "AVG");
    }

    #[test]
    fn test_display_simple_expr() {
        let pred = TLExpr::pred("Person", vec![Term::var("x")]);
        assert_eq!(format!("{}", pred), "Person(?x)");
    }

    #[test]
    fn test_display_logical_ops() {
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("y")]);

        let and_expr = TLExpr::and(p.clone(), q.clone());
        assert_eq!(format!("{}", and_expr), "(P(?x) ∧ Q(?y))");

        let or_expr = TLExpr::or(p.clone(), q);
        assert_eq!(format!("{}", or_expr), "(P(?x) ∨ Q(?y))");

        let not_expr = TLExpr::negate(p);
        assert_eq!(format!("{}", not_expr), "¬P(?x)");
    }

    #[test]
    fn test_display_quantifiers() {
        let body = TLExpr::pred("P", vec![Term::var("x")]);

        let exists = TLExpr::exists("x", "Domain", body.clone());
        assert_eq!(format!("{}", exists), "∃x:Domain. P(?x)");

        let forall = TLExpr::forall("x", "Domain", body);
        assert_eq!(format!("{}", forall), "∀x:Domain. P(?x)");
    }

    #[test]
    fn test_display_aggregate() {
        let body = TLExpr::pred("Value", vec![Term::var("x")]);

        let sum = TLExpr::sum("x", "Domain", body.clone());
        assert_eq!(format!("{}", sum), "SUM(x:Domain, Value(?x))");

        let count = TLExpr::count("x", "Domain", body);
        assert_eq!(format!("{}", count), "COUNT(x:Domain, Value(?x))");
    }

    #[test]
    fn test_display_aggregate_with_group_by() {
        let body = TLExpr::pred("Value", vec![Term::var("x"), Term::var("y")]);

        let agg = TLExpr::aggregate_with_group_by(
            AggregateOp::Sum,
            "x",
            "Domain",
            body,
            vec!["y".to_string()],
        );

        let display = format!("{}", agg);
        assert!(display.contains("SUM"));
        assert!(display.contains("GROUP BY"));
        assert!(display.contains("y"));
    }

    #[test]
    fn test_display_arithmetic() {
        let x = TLExpr::constant(5.0);
        let y = TLExpr::constant(3.0);

        let add = TLExpr::add(x.clone(), y.clone());
        assert_eq!(format!("{}", add), "(5 + 3)");

        let mul = TLExpr::mul(x, y);
        assert_eq!(format!("{}", mul), "(5 * 3)");
    }

    #[test]
    fn test_display_comparison() {
        let x = TLExpr::pred("X", vec![Term::var("i")]);
        let threshold = TLExpr::constant(0.5);

        let gt = TLExpr::gt(x, threshold);
        let display = format!("{}", gt);
        assert!(display.contains(">"));
        assert!(display.contains("0.5"));
    }

    #[test]
    fn test_display_conditional() {
        let cond = TLExpr::pred("IsAdult", vec![Term::var("x")]);
        let then_br = TLExpr::constant(1.0);
        let else_br = TLExpr::constant(0.0);

        let if_expr = TLExpr::if_then_else(cond, then_br, else_br);
        let display = format!("{}", if_expr);
        assert!(display.contains("if"));
        assert!(display.contains("then"));
        assert!(display.contains("else"));
    }

    #[test]
    fn test_display_einsum_node() {
        let node = EinsumNode::new("ij,jk->ik", vec![0, 1], vec![2]);
        let display = format!("{}", node);
        assert!(display.contains("einsum"));
        assert!(display.contains("ij,jk->ik"));
        assert!(display.contains("inputs=[0, 1]"));
        assert!(display.contains("outputs=[2]"));
    }

    #[test]
    fn test_display_graph() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input");
        let t1 = graph.add_tensor("output");

        graph
            .add_node(EinsumNode::new("i->i", vec![t0], vec![t1]))
            .unwrap();
        graph.add_output(t1).unwrap();

        let display = format!("{}", graph);
        assert!(display.contains("EinsumGraph"));
        assert!(display.contains("tensors"));
        assert!(display.contains("input"));
        assert!(display.contains("output"));
    }
}
