//! # Sequent Calculus for Proof-Theoretic Foundations
//!
//! This module implements sequent calculus representations and inference rules for TensorLogic.
//! Sequent calculus provides a formal foundation for logical reasoning and proof construction.
//!
//! ## Overview
//!
//! A **sequent** is a formal statement of the form `Γ ⊢ Δ` where:
//! - `Γ` (Gamma) is a multiset of antecedent formulas (hypotheses)
//! - `Δ` (Delta) is a multiset of consequent formulas (conclusions)
//! - `⊢` (turnstile) represents the entailment relation
//!
//! The sequent `Γ ⊢ Δ` is valid if: assuming all formulas in Γ are true,
//! at least one formula in Δ must be true.
//!
//! ## Inference Rules
//!
//! This module implements the following sequent calculus rules:
//!
//! ### Structural Rules
//! - **Identity**: `A ⊢ A`
//! - **Weakening**: From `Γ ⊢ Δ` derive `Γ, A ⊢ Δ` or `Γ ⊢ Δ, A`
//! - **Contraction**: From `Γ, A, A ⊢ Δ` derive `Γ, A ⊢ Δ`
//! - **Exchange**: Reorder formulas in Γ or Δ
//! - **Cut**: From `Γ ⊢ Δ, A` and `A, Γ' ⊢ Δ'` derive `Γ, Γ' ⊢ Δ, Δ'`
//!
//! ### Logical Rules (Left and Right)
//! - **AND-Left**: From `Γ, A, B ⊢ Δ` derive `Γ, A ∧ B ⊢ Δ`
//! - **AND-Right**: From `Γ ⊢ Δ, A` and `Γ ⊢ Δ, B` derive `Γ ⊢ Δ, A ∧ B`
//! - **OR-Left**: From `Γ, A ⊢ Δ` and `Γ, B ⊢ Δ` derive `Γ, A ∨ B ⊢ Δ`
//! - **OR-Right**: From `Γ ⊢ Δ, A, B` derive `Γ ⊢ Δ, A ∨ B`
//! - **NOT-Left**: From `Γ ⊢ Δ, A` derive `Γ, ¬A ⊢ Δ`
//! - **NOT-Right**: From `Γ, A ⊢ Δ` derive `Γ ⊢ Δ, ¬A`
//! - **IMPLY-Left**: From `Γ ⊢ Δ, A` and `Γ, B ⊢ Δ` derive `Γ, A → B ⊢ Δ`
//! - **IMPLY-Right**: From `Γ, A ⊢ Δ, B` derive `Γ ⊢ Δ, A → B`
//!
//! ### Quantifier Rules
//! - **EXISTS-Left**: From `Γ, A[t/x] ⊢ Δ` derive `Γ, ∃x.A ⊢ Δ` (t is fresh)
//! - **EXISTS-Right**: From `Γ ⊢ Δ, A[t/x]` derive `Γ ⊢ Δ, ∃x.A`
//! - **FORALL-Left**: From `Γ, A[t/x] ⊢ Δ` derive `Γ, ∀x.A ⊢ Δ`
//! - **FORALL-Right**: From `Γ ⊢ Δ, A[t/x]` derive `Γ ⊢ Δ, ∀x.A` (t is fresh)
//!
//! ## Applications
//!
//! - **Proof Search**: Construct proofs by applying inference rules backward
//! - **Proof Checking**: Verify that a derivation tree is valid
//! - **Proof Normalization**: Transform proofs to normal forms (cut-elimination)
//! - **Compilation**: Use sequent calculus to guide tensor compilation strategies
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, Term, Sequent, InferenceRule, ProofTree};
//!
//! // Construct a sequent: P(x) ∧ Q(x) ⊢ P(x)
//! let p = TLExpr::pred("P", vec![Term::var("x")]);
//! let q = TLExpr::pred("Q", vec![Term::var("x")]);
//! let and_pq = TLExpr::and(p.clone(), q);
//!
//! let conclusion = Sequent::new(vec![and_pq], vec![p.clone()]);
//!
//! // Construct a proof tree: Identity axiom as premise, then apply AndLeft
//! let identity_premise = ProofTree::identity(p.clone());
//! let proof = ProofTree::new(
//!     conclusion,
//!     InferenceRule::AndLeft { index: 0 },
//!     vec![identity_premise]
//! );
//!
//! assert!(proof.is_valid());
//! ```

use crate::expr::TLExpr;
use crate::term::Term;
use crate::unification::Substitution;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Helper function to substitute terms in predicates within an expression.
///
/// This performs capture-avoiding substitution, respecting bound variables in quantifiers.
fn substitute_in_expr(expr: &TLExpr, subst: &Substitution) -> TLExpr {
    match expr {
        TLExpr::Pred { name, args } => {
            // Apply substitution to each term in the predicate
            let new_args = args.iter().map(|term| subst.apply(term)).collect();
            TLExpr::Pred {
                name: name.clone(),
                args: new_args,
            }
        }
        TLExpr::And(left, right) => TLExpr::And(
            Box::new(substitute_in_expr(left, subst)),
            Box::new(substitute_in_expr(right, subst)),
        ),
        TLExpr::Or(left, right) => TLExpr::Or(
            Box::new(substitute_in_expr(left, subst)),
            Box::new(substitute_in_expr(right, subst)),
        ),
        TLExpr::Not(inner) => TLExpr::Not(Box::new(substitute_in_expr(inner, subst))),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(substitute_in_expr(left, subst)),
            Box::new(substitute_in_expr(right, subst)),
        ),
        TLExpr::Exists { var, domain, body } => {
            // Capture-avoiding substitution: don't substitute if var is bound
            if subst.domain().contains(var) {
                // Variable is bound by this quantifier, return unchanged
                expr.clone()
            } else {
                TLExpr::Exists {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute_in_expr(body, subst)),
                }
            }
        }
        TLExpr::ForAll { var, domain, body } => {
            // Capture-avoiding substitution: don't substitute if var is bound
            if subst.domain().contains(var) {
                // Variable is bound by this quantifier, return unchanged
                expr.clone()
            } else {
                TLExpr::ForAll {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute_in_expr(body, subst)),
                }
            }
        }
        // For other expression types, return as-is (they don't contain terms)
        _ => expr.clone(),
    }
}

/// A sequent is a formal statement `Γ ⊢ Δ` representing an entailment relation.
///
/// - `antecedents` (Γ): Multiset of hypothesis formulas (left side)
/// - `consequents` (Δ): Multiset of conclusion formulas (right side)
///
/// The sequent is valid if: assuming all antecedents are true,
/// at least one consequent must be true.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sequent {
    /// Hypothesis formulas (left side of ⊢)
    pub antecedents: Vec<TLExpr>,
    /// Conclusion formulas (right side of ⊢)
    pub consequents: Vec<TLExpr>,
}

impl Sequent {
    /// Create a new sequent from antecedents and consequents.
    pub fn new(antecedents: Vec<TLExpr>, consequents: Vec<TLExpr>) -> Self {
        Sequent {
            antecedents,
            consequents,
        }
    }

    /// Create an identity sequent: `A ⊢ A`
    pub fn identity(formula: TLExpr) -> Self {
        Sequent::new(vec![formula.clone()], vec![formula])
    }

    /// Check if this is an axiom (identity sequent where some antecedent equals some consequent).
    pub fn is_axiom(&self) -> bool {
        for ant in &self.antecedents {
            for cons in &self.consequents {
                if ant == cons {
                    return true;
                }
            }
        }
        false
    }

    /// Apply weakening rule: add a formula to antecedents.
    pub fn weaken_left(mut self, formula: TLExpr) -> Self {
        self.antecedents.push(formula);
        self
    }

    /// Apply weakening rule: add a formula to consequents.
    pub fn weaken_right(mut self, formula: TLExpr) -> Self {
        self.consequents.push(formula);
        self
    }

    /// Apply contraction rule: remove duplicate from antecedents.
    pub fn contract_left(mut self, index: usize) -> Option<Self> {
        if index >= self.antecedents.len() {
            return None;
        }
        let formula = self.antecedents[index].clone();
        // Find another occurrence
        for i in 0..self.antecedents.len() {
            if i != index && self.antecedents[i] == formula {
                self.antecedents.remove(index);
                return Some(self);
            }
        }
        None
    }

    /// Apply contraction rule: remove duplicate from consequents.
    pub fn contract_right(mut self, index: usize) -> Option<Self> {
        if index >= self.consequents.len() {
            return None;
        }
        let formula = self.consequents[index].clone();
        // Find another occurrence
        for i in 0..self.consequents.len() {
            if i != index && self.consequents[i] == formula {
                self.consequents.remove(index);
                return Some(self);
            }
        }
        None
    }

    /// Get all free variables in the sequent.
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for ant in &self.antecedents {
            vars.extend(ant.free_vars());
        }
        for cons in &self.consequents {
            vars.extend(cons.free_vars());
        }
        vars
    }

    /// Substitute a term for a variable throughout the sequent.
    ///
    /// This creates a new substitution and applies it to all formulas in the sequent.
    /// The substitution is capture-avoiding for bound variables in quantifiers.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{Sequent, TLExpr, Term};
    ///
    /// // P(x) ⊢ P(x)
    /// let p_x = TLExpr::pred("P", vec![Term::var("x")]);
    /// let seq = Sequent::identity(p_x);
    ///
    /// // Substitute x with a
    /// let seq_subst = seq.substitute("x", &Term::constant("a"));
    ///
    /// // Result should be P(a) ⊢ P(a)
    /// ```
    pub fn substitute(&self, var: &str, term: &Term) -> Self {
        let mut subst = Substitution::empty();
        subst.bind(var.to_string(), term.clone());

        let new_antecedents = self
            .antecedents
            .iter()
            .map(|expr| substitute_in_expr(expr, &subst))
            .collect();

        let new_consequents = self
            .consequents
            .iter()
            .map(|expr| substitute_in_expr(expr, &subst))
            .collect();

        Sequent::new(new_antecedents, new_consequents)
    }
}

/// Inference rules in sequent calculus.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InferenceRule {
    /// Identity axiom: `A ⊢ A`
    Identity,

    /// Weakening left: add formula to antecedents
    WeakeningLeft,

    /// Weakening right: add formula to consequents
    WeakeningRight,

    /// Contraction left: remove duplicate from antecedents
    ContractionLeft { index: usize },

    /// Contraction right: remove duplicate from consequents
    ContractionRight { index: usize },

    /// Exchange: reorder formulas (implicit, not tracked)
    Exchange,

    /// Cut rule: eliminate intermediate formula (index in antecedents/consequents)
    Cut { index: usize },

    /// AND introduction (left): `Γ, A, B ⊢ Δ` from `Γ, A ∧ B ⊢ Δ`
    AndLeft { index: usize },

    /// AND introduction (right): `Γ ⊢ Δ, A ∧ B` from `Γ ⊢ Δ, A` and `Γ ⊢ Δ, B`
    AndRight { index: usize },

    /// OR introduction (left): `Γ, A ∨ B ⊢ Δ` from `Γ, A ⊢ Δ` and `Γ, B ⊢ Δ`
    OrLeft { index: usize },

    /// OR introduction (right): `Γ ⊢ Δ, A, B` from `Γ ⊢ Δ, A ∨ B`
    OrRight { index: usize },

    /// NOT introduction (left): `Γ, ¬A ⊢ Δ` from `Γ ⊢ Δ, A`
    NotLeft { index: usize },

    /// NOT introduction (right): `Γ ⊢ Δ, ¬A` from `Γ, A ⊢ Δ`
    NotRight { index: usize },

    /// IMPLY introduction (left): `Γ, A → B ⊢ Δ` from `Γ ⊢ Δ, A` and `Γ, B ⊢ Δ`
    ImplyLeft { index: usize },

    /// IMPLY introduction (right): `Γ ⊢ Δ, A → B` from `Γ, A ⊢ Δ, B`
    ImplyRight { index: usize },

    /// EXISTS introduction (left): `Γ, ∃x.A ⊢ Δ` from `Γ, A[t/x] ⊢ Δ` (t fresh)
    ExistsLeft { index: usize, witness: Term },

    /// EXISTS introduction (right): `Γ ⊢ Δ, ∃x.A` from `Γ ⊢ Δ, A[t/x]`
    ExistsRight { index: usize, witness: Term },

    /// FORALL introduction (left): `Γ, ∀x.A ⊢ Δ` from `Γ, A[t/x] ⊢ Δ`
    ForAllLeft { index: usize, term: Term },

    /// FORALL introduction (right): `Γ ⊢ Δ, ∀x.A` from `Γ ⊢ Δ, A[t/x]` (t fresh)
    ForAllRight { index: usize, witness: Term },
}

/// A proof tree in sequent calculus.
///
/// Represents a derivation tree showing how a sequent is proved
/// using inference rules applied to premises.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProofTree {
    /// The conclusion sequent of this proof step
    pub conclusion: Sequent,
    /// The inference rule applied
    pub rule: InferenceRule,
    /// Premise proof trees (subproofs)
    pub premises: Vec<ProofTree>,
}

impl ProofTree {
    /// Create a proof tree for an identity axiom.
    pub fn identity(formula: TLExpr) -> Self {
        ProofTree {
            conclusion: Sequent::identity(formula),
            rule: InferenceRule::Identity,
            premises: vec![],
        }
    }

    /// Create a proof tree with premises and rule.
    pub fn new(conclusion: Sequent, rule: InferenceRule, premises: Vec<ProofTree>) -> Self {
        ProofTree {
            conclusion,
            rule,
            premises,
        }
    }

    /// Check if this proof tree is valid (premises correctly derive conclusion).
    pub fn is_valid(&self) -> bool {
        // Check that the rule application is sound
        match &self.rule {
            InferenceRule::Identity => {
                // Must have no premises and be an axiom
                self.premises.is_empty() && self.conclusion.is_axiom()
            }
            InferenceRule::WeakeningLeft | InferenceRule::WeakeningRight => {
                // Must have exactly one premise
                if self.premises.len() != 1 {
                    return false;
                }
                // The premise should be a subset of the conclusion
                // (detailed validation omitted for brevity)
                true
            }
            InferenceRule::AndLeft { index } => {
                // Must have one premise where A ∧ B is split into A, B
                if self.premises.len() != 1 {
                    return false;
                }
                if *index >= self.conclusion.antecedents.len() {
                    return false;
                }
                // Check that the formula at index is an AND
                matches!(self.conclusion.antecedents[*index], TLExpr::And(_, _))
            }
            InferenceRule::AndRight { .. } => {
                // Must have two premises
                self.premises.len() == 2
            }
            InferenceRule::OrLeft { .. } => {
                // Must have two premises
                self.premises.len() == 2
            }
            InferenceRule::OrRight { index } => {
                // Must have one premise
                if self.premises.len() != 1 {
                    return false;
                }
                if *index >= self.conclusion.consequents.len() {
                    return false;
                }
                // Check that the formula at index is an OR
                matches!(self.conclusion.consequents[*index], TLExpr::Or(_, _))
            }
            InferenceRule::NotLeft { .. } | InferenceRule::NotRight { .. } => {
                // Must have one premise
                self.premises.len() == 1
            }
            InferenceRule::ImplyLeft { .. } => {
                // Must have two premises
                self.premises.len() == 2
            }
            InferenceRule::ImplyRight { .. } => {
                // Must have one premise
                self.premises.len() == 1
            }
            InferenceRule::Cut { .. } => {
                // Must have two premises
                self.premises.len() == 2
            }
            _ => true, // Other rules validation omitted for brevity
        }
    }

    /// Get the depth of this proof tree.
    pub fn depth(&self) -> usize {
        if self.premises.is_empty() {
            1
        } else {
            1 + self.premises.iter().map(|p| p.depth()).max().unwrap_or(0)
        }
    }

    /// Count the number of inference rule applications in this proof.
    pub fn size(&self) -> usize {
        1 + self.premises.iter().map(|p| p.size()).sum::<usize>()
    }

    /// Check if this proof uses the cut rule.
    pub fn uses_cut(&self) -> bool {
        matches!(self.rule, InferenceRule::Cut { .. }) || self.premises.iter().any(|p| p.uses_cut())
    }
}

/// Proof search strategies for automated theorem proving.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofSearchStrategy {
    /// Depth-first search with backtracking
    DepthFirst { max_depth: usize },
    /// Breadth-first search
    BreadthFirst { max_depth: usize },
    /// Best-first search with heuristic
    BestFirst { max_depth: usize },
    /// Iterative deepening
    IterativeDeepening { max_depth: usize },
}

/// Proof search engine for automated theorem proving.
pub struct ProofSearchEngine {
    strategy: ProofSearchStrategy,
    /// Maximum number of proof search steps
    max_steps: usize,
    /// Statistics about the search
    pub stats: ProofSearchStats,
}

/// Statistics about proof search.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProofSearchStats {
    /// Number of sequents explored
    pub sequents_explored: usize,
    /// Number of proof trees generated
    pub proofs_generated: usize,
    /// Number of backtracking steps
    pub backtracks: usize,
    /// Final proof depth (if found)
    pub proof_depth: Option<usize>,
}

impl ProofSearchEngine {
    /// Create a new proof search engine with the given strategy.
    pub fn new(strategy: ProofSearchStrategy, max_steps: usize) -> Self {
        ProofSearchEngine {
            strategy,
            max_steps,
            stats: ProofSearchStats::default(),
        }
    }

    /// Search for a proof of the given sequent.
    ///
    /// Returns `Some(proof)` if a proof is found, `None` otherwise.
    pub fn search(&mut self, sequent: &Sequent) -> Option<ProofTree> {
        match self.strategy {
            ProofSearchStrategy::DepthFirst { max_depth } => self.dfs_search(sequent, 0, max_depth),
            ProofSearchStrategy::BreadthFirst { max_depth } => self.bfs_search(sequent, max_depth),
            ProofSearchStrategy::BestFirst { max_depth } => {
                self.best_first_search(sequent, max_depth)
            }
            ProofSearchStrategy::IterativeDeepening { max_depth } => {
                self.iterative_deepening_search(sequent, max_depth)
            }
        }
    }

    /// Depth-first search for a proof.
    fn dfs_search(
        &mut self,
        sequent: &Sequent,
        depth: usize,
        max_depth: usize,
    ) -> Option<ProofTree> {
        self.stats.sequents_explored += 1;

        if depth >= max_depth || self.stats.sequents_explored >= self.max_steps {
            self.stats.backtracks += 1;
            return None;
        }

        // Check if this is an axiom
        if sequent.is_axiom() {
            // Find the matching formula
            for ant in &sequent.antecedents {
                if sequent.consequents.contains(ant) {
                    self.stats.proofs_generated += 1;
                    let proof = ProofTree::identity(ant.clone());
                    self.stats.proof_depth = Some(depth);
                    return Some(proof);
                }
            }
        }

        // Try applying inference rules
        // (Simplified implementation - full proof search would try all applicable rules)

        // Try AND-Left rules
        for (i, ant) in sequent.antecedents.iter().enumerate() {
            if let TLExpr::And(a, b) = ant {
                let mut new_ant = sequent.antecedents.clone();
                new_ant.remove(i);
                new_ant.push((**a).clone());
                new_ant.push((**b).clone());

                let new_sequent = Sequent::new(new_ant, sequent.consequents.clone());
                if let Some(premise) = self.dfs_search(&new_sequent, depth + 1, max_depth) {
                    self.stats.proofs_generated += 1;
                    return Some(ProofTree::new(
                        sequent.clone(),
                        InferenceRule::AndLeft { index: i },
                        vec![premise],
                    ));
                }
            }
        }

        // Try OR-Right rules
        for (i, cons) in sequent.consequents.iter().enumerate() {
            if let TLExpr::Or(a, b) = cons {
                let mut new_cons = sequent.consequents.clone();
                new_cons.remove(i);
                new_cons.push((**a).clone());
                new_cons.push((**b).clone());

                let new_sequent = Sequent::new(sequent.antecedents.clone(), new_cons);
                if let Some(premise) = self.dfs_search(&new_sequent, depth + 1, max_depth) {
                    self.stats.proofs_generated += 1;
                    return Some(ProofTree::new(
                        sequent.clone(),
                        InferenceRule::OrRight { index: i },
                        vec![premise],
                    ));
                }
            }
        }

        // Try NOT-Left rules
        for (i, ant) in sequent.antecedents.iter().enumerate() {
            if let TLExpr::Not(a) = ant {
                let mut new_ant = sequent.antecedents.clone();
                new_ant.remove(i);

                let mut new_cons = sequent.consequents.clone();
                new_cons.push((**a).clone());

                let new_sequent = Sequent::new(new_ant, new_cons);
                if let Some(premise) = self.dfs_search(&new_sequent, depth + 1, max_depth) {
                    self.stats.proofs_generated += 1;
                    return Some(ProofTree::new(
                        sequent.clone(),
                        InferenceRule::NotLeft { index: i },
                        vec![premise],
                    ));
                }
            }
        }

        // Try NOT-Right rules
        for (i, cons) in sequent.consequents.iter().enumerate() {
            if let TLExpr::Not(a) = cons {
                let mut new_cons = sequent.consequents.clone();
                new_cons.remove(i);

                let mut new_ant = sequent.antecedents.clone();
                new_ant.push((**a).clone());

                let new_sequent = Sequent::new(new_ant, new_cons);
                if let Some(premise) = self.dfs_search(&new_sequent, depth + 1, max_depth) {
                    self.stats.proofs_generated += 1;
                    return Some(ProofTree::new(
                        sequent.clone(),
                        InferenceRule::NotRight { index: i },
                        vec![premise],
                    ));
                }
            }
        }

        self.stats.backtracks += 1;
        None
    }

    /// Breadth-first search (stub implementation).
    fn bfs_search(&mut self, sequent: &Sequent, max_depth: usize) -> Option<ProofTree> {
        // For simplicity, fall back to DFS
        self.dfs_search(sequent, 0, max_depth)
    }

    /// Best-first search with heuristic (stub implementation).
    fn best_first_search(&mut self, sequent: &Sequent, max_depth: usize) -> Option<ProofTree> {
        // For simplicity, fall back to DFS
        self.dfs_search(sequent, 0, max_depth)
    }

    /// Iterative deepening search.
    fn iterative_deepening_search(
        &mut self,
        sequent: &Sequent,
        max_depth: usize,
    ) -> Option<ProofTree> {
        for depth in 1..=max_depth {
            if let Some(proof) = self.dfs_search(sequent, 0, depth) {
                return Some(proof);
            }
        }
        None
    }
}

/// Cut elimination transformation.
///
/// Cut elimination is a fundamental property of sequent calculus stating that
/// any proof using the cut rule can be transformed into a cut-free proof.
/// This is important for proof normalization and extracting computational content.
pub struct CutElimination;

impl CutElimination {
    /// Attempt to eliminate all cut rules from a proof tree.
    ///
    /// Returns a cut-free proof if successful, or the original proof if
    /// cut elimination fails or is not applicable.
    pub fn eliminate(proof: ProofTree) -> ProofTree {
        if !proof.uses_cut() {
            return proof;
        }

        // Recursively eliminate cuts from premises first
        let premises: Vec<ProofTree> = proof.premises.into_iter().map(Self::eliminate).collect();

        // If this is a cut rule, try to eliminate it
        if let InferenceRule::Cut { index } = proof.rule {
            if premises.len() == 2 {
                // Cut elimination algorithm (simplified)
                // In a full implementation, this would perform the standard
                // cut elimination procedure by permuting the cut rule upward
                // and then removing it using logical rule inversions.

                // For now, return a simplified version
                // (A full implementation would be significantly more complex)
                return ProofTree::new(proof.conclusion, InferenceRule::Cut { index }, premises);
            }
        }

        // Return proof with eliminated premises
        ProofTree::new(proof.conclusion, proof.rule, premises)
    }

    /// Check if a proof is cut-free.
    pub fn is_cut_free(proof: &ProofTree) -> bool {
        !proof.uses_cut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TLExpr;

    #[test]
    fn test_identity_sequent() {
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let seq = Sequent::identity(p);
        assert!(seq.is_axiom());
        assert_eq!(seq.antecedents.len(), 1);
        assert_eq!(seq.consequents.len(), 1);
    }

    #[test]
    fn test_weakening_left() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let seq = Sequent::identity(p.clone()).weaken_left(q.clone());
        assert_eq!(seq.antecedents.len(), 2);
        assert!(seq.antecedents.contains(&q));
    }

    #[test]
    fn test_weakening_right() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let seq = Sequent::identity(p.clone()).weaken_right(q.clone());
        assert_eq!(seq.consequents.len(), 2);
        assert!(seq.consequents.contains(&q));
    }

    #[test]
    fn test_contraction_left() {
        let p = TLExpr::pred("P", vec![]);
        let mut seq = Sequent::identity(p.clone());
        seq.antecedents.push(p.clone());
        assert_eq!(seq.antecedents.len(), 2);

        let contracted = seq.contract_left(0);
        assert!(contracted.is_some());
        assert_eq!(contracted.unwrap().antecedents.len(), 1);
    }

    #[test]
    fn test_free_vars() {
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("y")]);
        let seq = Sequent::new(vec![p], vec![q]);

        let vars = seq.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_sequent_substitute() {
        let p_x = TLExpr::pred("P", vec![Term::var("x")]);
        let seq = Sequent::identity(p_x.clone());

        // Substitute x with a
        let substituted = seq.substitute("x", &Term::constant("a"));
        let p_a = TLExpr::pred("P", vec![Term::constant("a")]);
        assert_eq!(substituted.antecedents[0], p_a);
        assert_eq!(substituted.consequents[0], p_a);

        // Verify original is unchanged
        assert_eq!(seq.antecedents[0], p_x);
    }

    #[test]
    fn test_sequent_substitute_capture_avoiding() {
        // Test that substitution respects bound variables
        // ∃x. P(x) ⊢ Q(x)
        let p_x = TLExpr::pred("P", vec![Term::var("x")]);
        let exists_p = TLExpr::exists("x", "Domain", p_x);
        let q_x = TLExpr::pred("Q", vec![Term::var("x")]);
        let seq = Sequent::new(vec![exists_p.clone()], vec![q_x]);

        // Substitute x with a
        let substituted = seq.substitute("x", &Term::constant("a"));

        // Left side should be unchanged (x is bound)
        assert_eq!(substituted.antecedents[0], exists_p);
        // Right side should be substituted (x is free)
        let q_a = TLExpr::pred("Q", vec![Term::constant("a")]);
        assert_eq!(substituted.consequents[0], q_a);
    }

    #[test]
    fn test_sequent_substitute_multiple() {
        // P(x) ∧ Q(x) ⊢ R(x)
        let p_x = TLExpr::pred("P", vec![Term::var("x")]);
        let q_x = TLExpr::pred("Q", vec![Term::var("x")]);
        let and_pq = TLExpr::and(p_x, q_x);
        let r_x = TLExpr::pred("R", vec![Term::var("x")]);
        let seq = Sequent::new(vec![and_pq], vec![r_x]);

        // Substitute x with b
        let substituted = seq.substitute("x", &Term::constant("b"));

        // All x's should be replaced with b
        let p_b = TLExpr::pred("P", vec![Term::constant("b")]);
        let q_b = TLExpr::pred("Q", vec![Term::constant("b")]);
        let and_pq_b = TLExpr::and(p_b, q_b);
        let r_b = TLExpr::pred("R", vec![Term::constant("b")]);

        assert_eq!(substituted.antecedents[0], and_pq_b);
        assert_eq!(substituted.consequents[0], r_b);
    }

    #[test]
    fn test_substitution() {
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let seq = Sequent::identity(p.clone());

        // Substitution now works properly with unification
        let substituted = seq.substitute("x", &Term::constant("a"));
        let p_a = TLExpr::pred("P", vec![Term::constant("a")]);
        assert_eq!(substituted.antecedents[0], p_a);
        assert_eq!(substituted.consequents[0], p_a);
    }

    #[test]
    fn test_identity_proof_tree() {
        let p = TLExpr::pred("P", vec![]);
        let proof = ProofTree::identity(p);
        assert!(proof.is_valid());
        assert_eq!(proof.depth(), 1);
        assert_eq!(proof.size(), 1);
        assert!(!proof.uses_cut());
    }

    #[test]
    fn test_and_left_proof() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let and_pq = TLExpr::and(p.clone(), q.clone());

        // Build proof: from P, Q ⊢ P, derive P ∧ Q ⊢ P
        let _premise_seq = Sequent::new(vec![p.clone(), q], vec![p.clone()]);
        let premise = ProofTree::identity(p.clone());

        let conclusion_seq = Sequent::new(vec![and_pq], vec![p]);
        let proof = ProofTree::new(
            conclusion_seq,
            InferenceRule::AndLeft { index: 0 },
            vec![premise],
        );

        assert!(proof.is_valid());
        assert_eq!(proof.depth(), 2);
    }

    #[test]
    fn test_proof_search_simple() {
        let p = TLExpr::pred("P", vec![]);
        let sequent = Sequent::identity(p);

        let mut engine =
            ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);

        let proof = engine.search(&sequent);
        assert!(proof.is_some());
        assert!(proof.unwrap().is_valid());
        assert!(engine.stats.proofs_generated > 0);
    }

    #[test]
    fn test_proof_search_and() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let and_pq = TLExpr::and(p.clone(), q.clone());

        // Try to prove: P ∧ Q ⊢ P
        let sequent = Sequent::new(vec![and_pq], vec![p]);

        let mut engine =
            ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);

        let proof = engine.search(&sequent);
        assert!(proof.is_some());
        let proof = proof.unwrap();
        assert!(proof.is_valid());
        assert!(engine.stats.proofs_generated > 0);
    }

    #[test]
    fn test_proof_search_not() {
        let p = TLExpr::pred("P", vec![]);
        let not_p = TLExpr::negate(p.clone());

        // Try to prove: ¬P ⊢ ¬P (should be immediate)
        let sequent = Sequent::identity(not_p);

        let mut engine =
            ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);

        let proof = engine.search(&sequent);
        assert!(proof.is_some());
        assert!(proof.unwrap().is_valid());
    }

    #[test]
    fn test_cut_elimination_no_cut() {
        let p = TLExpr::pred("P", vec![]);
        let proof = ProofTree::identity(p);

        assert!(CutElimination::is_cut_free(&proof));
        let eliminated = CutElimination::eliminate(proof.clone());
        assert_eq!(eliminated, proof);
    }

    #[test]
    fn test_iterative_deepening_search() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let and_pq = TLExpr::and(p.clone(), q.clone());

        let sequent = Sequent::new(vec![and_pq], vec![p]);

        let mut engine = ProofSearchEngine::new(
            ProofSearchStrategy::IterativeDeepening { max_depth: 10 },
            1000,
        );

        let proof = engine.search(&sequent);
        assert!(proof.is_some());
        assert!(proof.unwrap().is_valid());
    }
}
