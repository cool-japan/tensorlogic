//! Core compilation functions for TLExpr → EinsumGraph.

mod abduction;
mod arithmetic;
mod comparison;
mod conditional;
mod constraints;
mod counting_quantifiers;
pub mod custom_ops;
mod fixpoint;
mod fuzzy;
mod higher_order;
mod hybrid;
mod implication;
mod let_binding;
mod logic_ops;
mod modal_temporal;
mod predicate;
mod probabilistic;
mod quantifiers;
mod set_operations;
mod strategy_mapping;

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::context::{CompileState, CompilerContext};

pub(crate) use abduction::{compile_abducible, compile_explain};
pub(crate) use arithmetic::{
    compile_abs, compile_add, compile_ceil, compile_cos, compile_div, compile_exp, compile_floor,
    compile_log, compile_max_binary, compile_min_binary, compile_mod, compile_mul, compile_pow,
    compile_round, compile_sin, compile_sqrt, compile_sub, compile_tan,
};
pub(crate) use comparison::{compile_eq, compile_gt, compile_gte, compile_lt, compile_lte};
pub(crate) use conditional::{compile_constant, compile_if_then_else};
pub(crate) use constraints::{compile_all_different, compile_global_cardinality};
pub(crate) use counting_quantifiers::{
    compile_counting_exists, compile_counting_forall, compile_exact_count, compile_majority,
};
pub use custom_ops::{
    CustomOpData, CustomOpHandler, CustomOpMetadata, CustomOpRegistry, ExtendedCompilerContext,
};
pub(crate) use fixpoint::{compile_greatest_fixpoint, compile_least_fixpoint};
pub(crate) use fuzzy::{
    compile_fuzzy_implication, compile_fuzzy_not, compile_tconorm, compile_tnorm,
};
pub(crate) use higher_order::{compile_apply, compile_lambda};
pub(crate) use hybrid::{compile_at, compile_everywhere, compile_nominal, compile_somewhere};
pub(crate) use implication::compile_imply;
pub(crate) use let_binding::compile_let;
pub(crate) use logic_ops::{compile_and, compile_not, compile_or};
pub(crate) use modal_temporal::{
    compile_always, compile_box, compile_diamond, compile_eventually, compile_next,
    compile_release, compile_strong_release, compile_until, compile_weak_until,
};
pub(crate) use predicate::compile_predicate;
pub(crate) use probabilistic::{compile_probabilistic_choice, compile_weighted_rule};
pub(crate) use quantifiers::{
    compile_aggregate, compile_exists, compile_forall, compile_soft_exists, compile_soft_forall,
};
pub(crate) use set_operations::{
    compile_empty_set, compile_set_cardinality, compile_set_comprehension, compile_set_difference,
    compile_set_intersection, compile_set_membership, compile_set_union,
};

/// Infer domain from expression context (if available).
pub(crate) fn infer_domain(expr: &TLExpr, _var: &str) -> Option<String> {
    match expr {
        TLExpr::Exists { domain, .. }
        | TLExpr::ForAll { domain, .. }
        | TLExpr::Aggregate { domain, .. }
        | TLExpr::SoftExists { domain, .. }
        | TLExpr::SoftForAll { domain, .. }
        | TLExpr::SetComprehension { domain, .. }
        | TLExpr::CountingExists { domain, .. }
        | TLExpr::CountingForAll { domain, .. }
        | TLExpr::ExactCount { domain, .. }
        | TLExpr::Majority { domain, .. } => Some(domain.clone()),
        // Modal/temporal logic operators - not yet implemented
        TLExpr::Box(_)
        | TLExpr::Diamond(_)
        | TLExpr::Next(_)
        | TLExpr::Eventually(_)
        | TLExpr::Always(_)
        | TLExpr::Until { .. }
        | TLExpr::Release { .. }
        | TLExpr::WeakUntil { .. }
        | TLExpr::StrongRelease { .. } => None,
        _ => None,
    }
}

/// Dispatch to appropriate compilation function based on expression type.
pub(crate) fn compile_expr(
    expr: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    match expr {
        TLExpr::Pred { name, args } => compile_predicate(name, args, ctx, graph),
        TLExpr::And(left, right) => compile_and(left, right, ctx, graph),
        TLExpr::Or(left, right) => compile_or(left, right, ctx, graph),
        TLExpr::Not(inner) => compile_not(inner, ctx, graph),
        TLExpr::Exists { var, domain, body } => compile_exists(var, domain, body, ctx, graph),
        TLExpr::ForAll { var, domain, body } => compile_forall(var, domain, body, ctx, graph),
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => compile_aggregate(op, var, domain, body, group_by, ctx, graph),
        TLExpr::Imply(premise, conclusion) => compile_imply(premise, conclusion, ctx, graph),
        TLExpr::Score(inner) => compile_expr(inner, ctx, graph),

        // Arithmetic operations
        TLExpr::Add(left, right) => compile_add(left, right, ctx, graph),
        TLExpr::Sub(left, right) => compile_sub(left, right, ctx, graph),
        TLExpr::Mul(left, right) => compile_mul(left, right, ctx, graph),
        TLExpr::Div(left, right) => compile_div(left, right, ctx, graph),

        // Comparison operations
        TLExpr::Eq(left, right) => compile_eq(left, right, ctx, graph),
        TLExpr::Lt(left, right) => compile_lt(left, right, ctx, graph),
        TLExpr::Gt(left, right) => compile_gt(left, right, ctx, graph),
        TLExpr::Lte(left, right) => compile_lte(left, right, ctx, graph),
        TLExpr::Gte(left, right) => compile_gte(left, right, ctx, graph),

        // Additional arithmetic operations
        TLExpr::Pow(left, right) => compile_pow(left, right, ctx, graph),
        TLExpr::Mod(left, right) => compile_mod(left, right, ctx, graph),
        TLExpr::Min(left, right) => compile_min_binary(left, right, ctx, graph),
        TLExpr::Max(left, right) => compile_max_binary(left, right, ctx, graph),

        // Unary mathematical operations
        TLExpr::Abs(inner) => compile_abs(inner, ctx, graph),
        TLExpr::Floor(inner) => compile_floor(inner, ctx, graph),
        TLExpr::Ceil(inner) => compile_ceil(inner, ctx, graph),
        TLExpr::Round(inner) => compile_round(inner, ctx, graph),
        TLExpr::Sqrt(inner) => compile_sqrt(inner, ctx, graph),
        TLExpr::Exp(inner) => compile_exp(inner, ctx, graph),
        TLExpr::Log(inner) => compile_log(inner, ctx, graph),
        TLExpr::Sin(inner) => compile_sin(inner, ctx, graph),
        TLExpr::Cos(inner) => compile_cos(inner, ctx, graph),
        TLExpr::Tan(inner) => compile_tan(inner, ctx, graph),

        // Conditional expression
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => compile_if_then_else(condition, then_branch, else_branch, ctx, graph),

        // Constant
        TLExpr::Constant(value) => compile_constant(*value, ctx, graph),

        // Let binding
        TLExpr::Let { var, value, body } => compile_let(var, value, body, ctx, graph),

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => compile_tnorm(*kind, left, right, ctx, graph),
        TLExpr::TCoNorm { kind, left, right } => compile_tconorm(*kind, left, right, ctx, graph),
        TLExpr::FuzzyNot { kind, expr } => compile_fuzzy_not(*kind, expr, ctx, graph),
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => compile_fuzzy_implication(*kind, premise, conclusion, ctx, graph),
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => compile_soft_exists(var, domain, body, *temperature, ctx, graph),
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => compile_soft_forall(var, domain, body, *temperature, ctx, graph),
        TLExpr::WeightedRule { weight, rule } => compile_weighted_rule(*weight, rule, ctx, graph),
        TLExpr::ProbabilisticChoice { alternatives } => {
            compile_probabilistic_choice(alternatives, ctx, graph)
        }

        // Modal logic operators
        TLExpr::Box(inner) => compile_box(inner, ctx, graph),
        TLExpr::Diamond(inner) => compile_diamond(inner, ctx, graph),

        // Temporal logic operators
        TLExpr::Next(inner) => compile_next(inner, ctx, graph),
        TLExpr::Eventually(inner) => compile_eventually(inner, ctx, graph),
        TLExpr::Always(inner) => compile_always(inner, ctx, graph),
        TLExpr::Until { before, after } => compile_until(before, after, ctx, graph),
        TLExpr::Release { released, releaser } => compile_release(releaser, released, ctx, graph),
        TLExpr::WeakUntil { before, after } => compile_weak_until(before, after, ctx, graph),
        TLExpr::StrongRelease { released, releaser } => {
            compile_strong_release(releaser, released, ctx, graph)
        }

        // Counting quantifiers
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => compile_counting_exists(var, domain, body, *min_count, ctx, graph),
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => compile_counting_forall(var, domain, body, *min_count, ctx, graph),
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => compile_exact_count(var, domain, body, *count, ctx, graph),
        TLExpr::Majority { var, domain, body } => compile_majority(var, domain, body, ctx, graph),

        // Higher-order logic
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => compile_lambda(var, var_type, body, ctx, graph),
        TLExpr::Apply { function, argument } => compile_apply(function, argument, ctx, graph),

        // Set theory operations
        TLExpr::SetMembership { element, set } => compile_set_membership(element, set, ctx, graph),
        TLExpr::SetUnion { left, right } => compile_set_union(left, right, ctx, graph),
        TLExpr::SetIntersection { left, right } => {
            compile_set_intersection(left, right, ctx, graph)
        }
        TLExpr::SetDifference { left, right } => compile_set_difference(left, right, ctx, graph),
        TLExpr::SetCardinality { set } => compile_set_cardinality(set, ctx, graph),
        TLExpr::EmptySet => compile_empty_set(ctx, graph),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => compile_set_comprehension(var, domain, condition, ctx, graph),

        // Fixed-point operators
        TLExpr::LeastFixpoint { var, body } => compile_least_fixpoint(var, body, ctx, graph),
        TLExpr::GreatestFixpoint { var, body } => compile_greatest_fixpoint(var, body, ctx, graph),

        // Hybrid logic
        TLExpr::Nominal { name } => compile_nominal(name, ctx, graph),
        TLExpr::At { nominal, formula } => compile_at(nominal, formula, ctx, graph),
        TLExpr::Somewhere { formula } => compile_somewhere(formula, ctx, graph),
        TLExpr::Everywhere { formula } => compile_everywhere(formula, ctx, graph),

        // Constraint programming
        TLExpr::AllDifferent { variables } => compile_all_different(variables, ctx, graph),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => compile_global_cardinality(
            variables,
            values,
            min_occurrences,
            max_occurrences,
            ctx,
            graph,
        ),

        // Abductive reasoning
        TLExpr::Abducible { name, cost } => compile_abducible(name, *cost, ctx, graph),
        TLExpr::Explain { formula } => compile_explain(formula, ctx, graph),
    }
}
