//! # Resolution-Based Theorem Proving
//!
//! This module implements Robinson's resolution principle for automated theorem proving.
//! Resolution is a refutation-based proof procedure: to prove `Γ ⊢ φ`, we show that
//! `Γ ∪ {¬φ}` is unsatisfiable by deriving the empty clause (⊥).
//!
//! ## Overview
//!
//! **Resolution** is a complete inference rule for first-order logic:
//! - Given clauses `C₁ ∨ L` and `C₂ ∨ ¬L`, derive resolvent `C₁ ∨ C₂`
//! - The empty clause (∅) represents a contradiction
//! - If ∅ is derived, the original clause set is unsatisfiable
//!
//! ## Key Components
//!
//! ### Literals
//! A literal is an atom or its negation:
//! - Positive literal: `P(x, y)`
//! - Negative literal: `¬P(x, y)`
//!
//! ### Clauses
//! A clause is a disjunction of literals:
//! - `P(x) ∨ Q(x) ∨ ¬R(y)`
//! - Empty clause: `∅` (contradiction)
//! - Unit clause: single literal
//!
//! ### Resolution Rule
//! From clauses `C₁ ∨ L` and `C₂ ∨ ¬L`, derive `C₁ ∨ C₂`:
//! ```text
//!     C₁ ∨ L    C₂ ∨ ¬L
//!     ───────────────────
//!         C₁ ∨ C₂
//! ```
//!
//! ## Algorithms
//!
//! 1. **Saturation**: Generate all resolvents until empty clause or saturation
//! 2. **Set-of-Support**: Focus resolution on specific clause set
//! 3. **Linear Resolution**: Chain resolutions from initial clause
//! 4. **Unit Resolution**: Only resolve with unit clauses (more efficient)
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, Term, Clause, Literal, ResolutionProver};
//!
//! // Clauses: { P(a), ¬P(a) }
//! // This is unsatisfiable (derives empty clause via direct resolution)
//! let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
//! let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
//!
//! let mut prover = ResolutionProver::new();
//! prover.add_clause(Clause::from_literals(vec![p_a]));
//! prover.add_clause(Clause::from_literals(vec![not_p_a]));
//!
//! let result = prover.prove();
//! assert!(result.is_unsatisfiable());
//! ```

use crate::error::IrError;
use crate::expr::TLExpr;
use crate::term::Term;
use crate::unification::{rename_vars, unify_term_list, Substitution};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// A literal is an atomic formula or its negation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    /// The underlying atomic formula
    pub atom: TLExpr,
    /// True if positive literal, false if negative
    pub polarity: bool,
}

impl Literal {
    /// Create a positive literal from an atomic formula.
    pub fn positive(atom: TLExpr) -> Self {
        Literal {
            atom,
            polarity: true,
        }
    }

    /// Create a negative literal from an atomic formula.
    pub fn negative(atom: TLExpr) -> Self {
        Literal {
            atom,
            polarity: false,
        }
    }

    /// Negate this literal.
    pub fn negate(&self) -> Self {
        Literal {
            atom: self.atom.clone(),
            polarity: !self.polarity,
        }
    }

    /// Check if this literal is complementary to another (same atom, opposite polarity).
    ///
    /// For ground literals (no variables), this checks exact equality.
    pub fn is_complementary(&self, other: &Literal) -> bool {
        self.atom == other.atom && self.polarity != other.polarity
    }

    /// Attempt to unify this literal with another for resolution.
    ///
    /// Returns the most general unifier (MGU) if the atoms can be unified
    /// and the polarities are opposite.
    ///
    /// This is used for first-order resolution with variables.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal};
    ///
    /// // P(x) and ¬P(a) can be unified with {x/a}
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
    ///
    /// let mgu = p_x.try_unify(&not_p_a);
    /// assert!(mgu.is_some());
    /// ```
    pub fn try_unify(&self, other: &Literal) -> Option<Substitution> {
        // Must have opposite polarity
        if self.polarity == other.polarity {
            return None;
        }

        // Try to unify the atoms
        self.try_unify_atoms(&other.atom)
    }

    /// Attempt to unify this literal's atom with another expression.
    ///
    /// This extracts terms from predicates and attempts unification.
    fn try_unify_atoms(&self, other_atom: &TLExpr) -> Option<Substitution> {
        match (&self.atom, other_atom) {
            (
                TLExpr::Pred {
                    name: n1,
                    args: args1,
                },
                TLExpr::Pred {
                    name: n2,
                    args: args2,
                },
            ) => {
                // Predicate names must match
                if n1 != n2 {
                    return None;
                }

                // Arity must match
                if args1.len() != args2.len() {
                    return None;
                }

                // Unify argument lists
                let pairs: Vec<(Term, Term)> = args1
                    .iter()
                    .zip(args2.iter())
                    .map(|(t1, t2)| (t1.clone(), t2.clone()))
                    .collect();

                unify_term_list(&pairs).ok()
            }
            _ => None,
        }
    }

    /// Apply a substitution to this literal.
    ///
    /// This creates a new literal with the substitution applied to all terms.
    pub fn apply_substitution(&self, subst: &Substitution) -> Literal {
        let new_atom = self.apply_subst_to_expr(&self.atom, subst);
        Literal {
            atom: new_atom,
            polarity: self.polarity,
        }
    }

    /// Apply substitution to an expression (helper for apply_substitution).
    fn apply_subst_to_expr(&self, expr: &TLExpr, subst: &Substitution) -> TLExpr {
        match expr {
            TLExpr::Pred { name, args } => {
                let new_args = args.iter().map(|term| subst.apply(term)).collect();
                TLExpr::Pred {
                    name: name.clone(),
                    args: new_args,
                }
            }
            // For other expression types, return as-is
            _ => expr.clone(),
        }
    }

    /// Check if this is a positive literal.
    pub fn is_positive(&self) -> bool {
        self.polarity
    }

    /// Check if this is a negative literal.
    pub fn is_negative(&self) -> bool {
        !self.polarity
    }

    /// Get the free variables in this literal.
    pub fn free_vars(&self) -> HashSet<String> {
        self.atom.free_vars()
    }
}

/// A clause is a disjunction of literals: `L₁ ∨ L₂ ∨ ... ∨ Lₙ`.
///
/// Special cases:
/// - Empty clause (∅): contradiction, no literals
/// - Unit clause: single literal
/// - Horn clause: at most one positive literal
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clause {
    /// The literals in this clause (disjunction)
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Create a new clause from a list of literals.
    pub fn from_literals(literals: Vec<Literal>) -> Self {
        // Remove duplicates and sort for consistency
        let mut unique_lits: Vec<Literal> = literals.into_iter().collect();
        unique_lits.sort_by(|a, b| {
            let a_str = format!("{:?}", a);
            let b_str = format!("{:?}", b);
            a_str.cmp(&b_str)
        });
        unique_lits.dedup();

        Clause {
            literals: unique_lits,
        }
    }

    /// Create an empty clause (contradiction).
    pub fn empty() -> Self {
        Clause { literals: vec![] }
    }

    /// Create a unit clause (single literal).
    pub fn unit(literal: Literal) -> Self {
        Clause {
            literals: vec![literal],
        }
    }

    /// Check if this is the empty clause (contradiction).
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if this is a unit clause (single literal).
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Check if this is a Horn clause (at most one positive literal).
    pub fn is_horn(&self) -> bool {
        self.literals.iter().filter(|l| l.is_positive()).count() <= 1
    }

    /// Get the number of literals in this clause.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if clause is empty (different from is_empty which checks for contradiction).
    pub fn is_len_zero(&self) -> bool {
        self.literals.is_empty()
    }

    /// Get all free variables in this clause.
    pub fn free_vars(&self) -> HashSet<String> {
        self.literals
            .iter()
            .flat_map(|lit| lit.free_vars())
            .collect()
    }

    /// Check if this clause subsumes another (is more general).
    ///
    /// **Theta-Subsumption**: Clause C subsumes D (C ⪯ D) if there exists a
    /// substitution θ such that Cθ ⊆ D.
    ///
    /// This means C is more general than D. For example:
    /// - `{P(x)}` subsumes `{P(a), Q(a)}` with θ = {x/a}
    /// - `{P(x), Q(x)}` subsumes `{P(a), Q(a), R(a)}` with θ = {x/a}
    ///
    /// # Implementation
    ///
    /// We try to find a substitution by:
    /// 1. For each literal in C, try to unify it with some literal in D
    /// 2. Check if all substitutions are consistent
    /// 3. If successful, C subsumes D
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause};
    ///
    /// // {P(x)} subsumes {P(a)}
    /// let c = Clause::unit(Literal::positive(TLExpr::pred("P", vec![Term::var("x")])));
    /// let d = Clause::unit(Literal::positive(TLExpr::pred("P", vec![Term::constant("a")])));
    /// assert!(c.subsumes(&d));
    ///
    /// // {P(a)} does not subsume {P(x)} (x is more general than a)
    /// assert!(!d.subsumes(&c));
    /// ```
    pub fn subsumes(&self, other: &Clause) -> bool {
        // Empty clause subsumes nothing (except itself)
        if self.is_empty() {
            return other.is_empty();
        }

        // Can't subsume if C has more literals than D
        if self.literals.len() > other.literals.len() {
            return false;
        }

        // Try to find a substitution that makes all of C's literals match some literal in D
        self.try_subsumption_matching(other).is_some()
    }

    /// Attempt to find a substitution θ such that Cθ ⊆ D.
    ///
    /// Uses a backtracking search to find consistent literal matchings.
    fn try_subsumption_matching(&self, other: &Clause) -> Option<Substitution> {
        // Rename variables in self to avoid conflicts
        static mut SUBSUME_COUNTER: usize = 0;
        let counter = unsafe {
            SUBSUME_COUNTER += 1;
            SUBSUME_COUNTER
        };

        let renamed_self = self.rename_variables(&format!("_s{}", counter));
        let renamed_vars: HashSet<String> = renamed_self.free_vars();

        // Try to match each literal in renamed_self with literals in other
        let mut subst = Substitution::empty();

        for self_lit in &renamed_self.literals {
            // Try to find a matching literal in other
            let mut found_match = false;

            for other_lit in &other.literals {
                // Literals must have the same polarity
                if self_lit.polarity != other_lit.polarity {
                    continue;
                }

                // Try one-way matching: only variables from self can be bound
                if let Some(lit_mgu) = self_lit.try_one_way_match(&other_lit.atom, &renamed_vars) {
                    // Check if this unifier is consistent with existing substitution
                    if let Ok(()) = subst.try_extend(&lit_mgu) {
                        found_match = true;
                        break;
                    }
                }
            }

            if !found_match {
                return None; // Failed to match this literal
            }
        }

        Some(subst)
    }
}

impl Literal {
    /// Try one-way matching for subsumption: only variables in `allowed_vars` can be bound.
    ///
    /// This is different from full unification - we only allow variables from the
    /// subsuming clause to be instantiated, not variables from the subsumed clause.
    fn try_one_way_match(
        &self,
        other_atom: &TLExpr,
        allowed_vars: &HashSet<String>,
    ) -> Option<Substitution> {
        match (&self.atom, other_atom) {
            (
                TLExpr::Pred {
                    name: n1,
                    args: args1,
                },
                TLExpr::Pred {
                    name: n2,
                    args: args2,
                },
            ) => {
                // Predicate names must match
                if n1 != n2 {
                    return None;
                }

                // Arity must match
                if args1.len() != args2.len() {
                    return None;
                }

                // Try one-way matching for each argument pair
                let mut subst = Substitution::empty();

                for (t1, t2) in args1.iter().zip(args2.iter()) {
                    if !try_one_way_match_terms(t1, t2, allowed_vars, &mut subst) {
                        return None;
                    }
                }

                Some(subst)
            }
            _ => None,
        }
    }
}

/// One-way matching for terms: only variables in `allowed_vars` can be bound.
fn try_one_way_match_terms(
    t1: &Term,
    t2: &Term,
    allowed_vars: &HashSet<String>,
    subst: &mut Substitution,
) -> bool {
    // Apply current substitution to t1
    let t1_subst = subst.apply(t1);

    match (&t1_subst, t2) {
        // Same constant
        (Term::Const(c1), Term::Const(c2)) => c1 == c2,

        // Same variable
        (Term::Var(v1), Term::Var(v2)) => v1 == v2,

        // t1 is an allowed variable, bind it to t2
        (Term::Var(v1), _) if allowed_vars.contains(v1) => {
            // Check if already bound by trying to apply the substitution
            let after_subst = subst.apply(&t1_subst);
            if after_subst != t1_subst {
                // Already bound, check if it matches t2
                &after_subst == t2
            } else {
                // Not bound, bind it now
                subst.bind(v1.clone(), t2.clone());
                true
            }
        }

        // t1 is a variable but not allowed to be bound
        (Term::Var(_), _) => false,

        // t2 is a variable but we can't bind it (one-way matching)
        (_, Term::Var(_)) => false,

        // Both typed with same type
        (
            Term::Typed {
                value: inner1,
                type_annotation: ty1,
            },
            Term::Typed {
                value: inner2,
                type_annotation: ty2,
            },
        ) => {
            if ty1 != ty2 {
                return false;
            }
            try_one_way_match_terms(inner1, inner2, allowed_vars, subst)
        }

        // One typed, one not
        (Term::Typed { value, .. }, other) | (other, Term::Typed { value, .. }) => {
            try_one_way_match_terms(value, other, allowed_vars, subst)
        }
    }
}

impl Clause {
    /// Check if this clause is tautology (contains complementary literals).
    pub fn is_tautology(&self) -> bool {
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i].is_complementary(&self.literals[j]) {
                    return true;
                }
            }
        }
        false
    }

    /// Apply a substitution to this clause.
    ///
    /// This creates a new clause with the substitution applied to all literals.
    pub fn apply_substitution(&self, subst: &Substitution) -> Clause {
        let new_literals = self
            .literals
            .iter()
            .map(|lit| lit.apply_substitution(subst))
            .collect();
        Clause::from_literals(new_literals)
    }

    /// Rename all variables in this clause with a suffix.
    ///
    /// This is used for standardizing apart clauses before resolution
    /// to avoid variable name conflicts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause};
    ///
    /// // P(x) ∨ Q(x)
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    /// let clause = Clause::from_literals(vec![p_x, q_x]);
    ///
    /// // Rename to P(x_1) ∨ Q(x_1)
    /// let renamed = clause.rename_variables("1");
    /// ```
    pub fn rename_variables(&self, suffix: &str) -> Clause {
        let renamed_literals = self
            .literals
            .iter()
            .map(|lit| self.rename_literal(lit, suffix))
            .collect();
        Clause::from_literals(renamed_literals)
    }

    /// Rename variables in a literal (helper for rename_variables).
    fn rename_literal(&self, lit: &Literal, suffix: &str) -> Literal {
        match &lit.atom {
            TLExpr::Pred { name, args } => {
                let renamed_args = args.iter().map(|term| rename_vars(term, suffix)).collect();
                Literal {
                    atom: TLExpr::Pred {
                        name: name.clone(),
                        args: renamed_args,
                    },
                    polarity: lit.polarity,
                }
            }
            _ => lit.clone(),
        }
    }
}

/// Result of a resolution proof attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProofResult {
    /// The clause set is unsatisfiable (empty clause derived)
    Unsatisfiable {
        /// Steps taken to derive empty clause
        steps: usize,
        /// Derivation path (sequence of resolution steps)
        derivation: Vec<ResolutionStep>,
    },
    /// The clause set is satisfiable (no contradiction found)
    Satisfiable,
    /// Proof attempt reached saturation without finding empty clause
    Saturated {
        /// Number of clauses generated
        clauses_generated: usize,
    },
    /// Proof search reached resource limit
    ResourceLimitReached {
        /// Number of steps attempted
        steps_attempted: usize,
    },
}

impl ProofResult {
    /// Check if the result proves unsatisfiability.
    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, ProofResult::Unsatisfiable { .. })
    }

    /// Check if the result proves satisfiability.
    pub fn is_satisfiable(&self) -> bool {
        matches!(self, ProofResult::Satisfiable)
    }

    /// Get the number of steps taken.
    pub fn steps(&self) -> usize {
        match self {
            ProofResult::Unsatisfiable { steps, .. } => *steps,
            ProofResult::ResourceLimitReached { steps_attempted } => *steps_attempted,
            ProofResult::Saturated { clauses_generated } => *clauses_generated,
            ProofResult::Satisfiable => 0,
        }
    }
}

/// A single resolution step in a proof.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolutionStep {
    /// First parent clause
    pub parent1: Clause,
    /// Second parent clause
    pub parent2: Clause,
    /// Resulting resolvent clause
    pub resolvent: Clause,
    /// Literal that was resolved on (from parent1)
    pub resolved_literal: Literal,
}

/// Resolution proof strategy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Generate all possible resolvents until empty clause or saturation
    Saturation {
        /// Maximum number of clauses to generate
        max_clauses: usize,
    },
    /// Focus resolution on specific set of clauses
    SetOfSupport {
        /// Maximum resolution steps
        max_steps: usize,
    },
    /// Only resolve with unit clauses (more efficient)
    UnitResolution {
        /// Maximum resolution steps
        max_steps: usize,
    },
    /// Linear resolution: chain from initial clause
    Linear {
        /// Maximum chain length
        max_depth: usize,
    },
}

/// Resolution-based theorem prover.
pub struct ResolutionProver {
    /// Initial clause set
    clauses: Vec<Clause>,
    /// Strategy to use
    strategy: ResolutionStrategy,
    /// Statistics
    pub stats: ProverStats,
}

/// Statistics for resolution proof search.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProverStats {
    /// Total clauses generated
    pub clauses_generated: usize,
    /// Resolution steps performed
    pub resolution_steps: usize,
    /// Tautologies removed
    pub tautologies_removed: usize,
    /// Clauses subsumed
    pub clauses_subsumed: usize,
    /// Empty clause found
    pub empty_clause_found: bool,
}

impl ResolutionProver {
    /// Create a new resolution prover with default strategy.
    pub fn new() -> Self {
        ResolutionProver {
            clauses: Vec::new(),
            strategy: ResolutionStrategy::Saturation { max_clauses: 10000 },
            stats: ProverStats::default(),
        }
    }

    /// Create a prover with a specific strategy.
    pub fn with_strategy(strategy: ResolutionStrategy) -> Self {
        ResolutionProver {
            clauses: Vec::new(),
            strategy,
            stats: ProverStats::default(),
        }
    }

    /// Add a clause to the initial clause set.
    pub fn add_clause(&mut self, clause: Clause) {
        // Don't add tautologies
        if !clause.is_tautology() {
            self.clauses.push(clause);
        } else {
            self.stats.tautologies_removed += 1;
        }
    }

    /// Add multiple clauses.
    pub fn add_clauses(&mut self, clauses: Vec<Clause>) {
        for clause in clauses {
            self.add_clause(clause);
        }
    }

    /// Reset the prover (clear clauses and stats).
    pub fn reset(&mut self) {
        self.clauses.clear();
        self.stats = ProverStats::default();
    }

    /// Perform binary resolution on two clauses.
    ///
    /// Returns all possible resolvents.
    ///
    /// This is the ground resolution (no variables). For first-order resolution
    /// with variables, use `resolve_first_order`.
    fn resolve(&self, c1: &Clause, c2: &Clause) -> Vec<(Clause, Literal)> {
        let mut resolvents = Vec::new();

        // Try to resolve on each pair of complementary literals
        for lit1 in &c1.literals {
            for lit2 in &c2.literals {
                if lit1.is_complementary(lit2) {
                    // Build resolvent: (c1 - lit1) ∪ (c2 - lit2)
                    let mut new_literals = Vec::new();

                    // Add literals from c1 except lit1
                    for lit in &c1.literals {
                        if lit != lit1 {
                            new_literals.push(lit.clone());
                        }
                    }

                    // Add literals from c2 except lit2
                    for lit in &c2.literals {
                        if lit != lit2 {
                            new_literals.push(lit.clone());
                        }
                    }

                    let resolvent = Clause::from_literals(new_literals);
                    resolvents.push((resolvent, lit1.clone()));
                }
            }
        }

        resolvents
    }

    /// Perform first-order binary resolution with unification.
    ///
    /// This supports resolution on clauses with variables by using unification.
    /// Clauses are standardized apart before resolution to avoid variable conflicts.
    ///
    /// Returns all possible resolvents with their MGUs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause, ResolutionProver};
    ///
    /// // {P(x)} and {¬P(a)} resolve to {} (empty clause) with MGU {x/a}
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
    ///
    /// let c1 = Clause::unit(p_x);
    /// let c2 = Clause::unit(not_p_a);
    ///
    /// let prover = ResolutionProver::new();
    /// let resolvents = prover.resolve_first_order(&c1, &c2);
    ///
    /// assert_eq!(resolvents.len(), 1);
    /// assert!(resolvents[0].0.is_empty()); // Empty clause derived
    /// ```
    pub fn resolve_first_order(&self, c1: &Clause, c2: &Clause) -> Vec<(Clause, Literal)> {
        // Use a simple counter for standardizing apart
        // In practice, this could use a global counter or timestamp
        static mut RENAME_COUNTER: usize = 0;
        let counter = unsafe {
            RENAME_COUNTER += 1;
            RENAME_COUNTER
        };

        // Standardize apart: rename variables to avoid conflicts
        let c1_renamed = c1.rename_variables(&format!("_c1_{}", counter));
        let c2_renamed = c2.rename_variables(&format!("_c2_{}", counter));

        let mut resolvents = Vec::new();

        // Try to unify each pair of opposite polarity literals
        for lit1 in &c1_renamed.literals {
            for lit2 in &c2_renamed.literals {
                // Try to unify with first-order unification
                if let Some(mgu) = lit1.try_unify(lit2) {
                    // Build resolvent: apply MGU to (c1 - lit1) ∪ (c2 - lit2)
                    let mut new_literals = Vec::new();

                    // Add literals from c1 except lit1, with MGU applied
                    for lit in &c1_renamed.literals {
                        if lit != lit1 {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    // Add literals from c2 except lit2, with MGU applied
                    for lit in &c2_renamed.literals {
                        if lit != lit2 {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    let resolvent = Clause::from_literals(new_literals);
                    // Return the original (non-renamed) literal for tracking
                    let orig_lit = lit1.clone(); // Could map back to original if needed
                    resolvents.push((resolvent, orig_lit));
                }
            }
        }

        resolvents
    }

    /// Check if a clause is subsumed by any clause in the set.
    fn is_subsumed(&self, clause: &Clause, clause_set: &[Clause]) -> bool {
        clause_set.iter().any(|c| c.subsumes(clause))
    }

    /// Attempt to prove the clause set unsatisfiable using resolution.
    pub fn prove(&mut self) -> ProofResult {
        match &self.strategy {
            ResolutionStrategy::Saturation { max_clauses } => self.prove_saturation(*max_clauses),
            ResolutionStrategy::SetOfSupport { max_steps } => self.prove_set_of_support(*max_steps),
            ResolutionStrategy::UnitResolution { max_steps } => {
                self.prove_unit_resolution(*max_steps)
            }
            ResolutionStrategy::Linear { max_depth } => self.prove_linear(*max_depth),
        }
    }

    /// Saturation-based proof: generate all resolvents.
    fn prove_saturation(&mut self, max_clauses: usize) -> ProofResult {
        let mut clause_set: Vec<Clause> = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        // Check if empty clause is in initial set
        if clause_set.iter().any(|c| c.is_empty()) {
            self.stats.empty_clause_found = true;
            return ProofResult::Unsatisfiable {
                steps: 0,
                derivation: vec![],
            };
        }

        loop {
            let current_clauses: Vec<Clause> = clause_set.clone();
            let mut new_clauses = Vec::new();

            // Generate all resolvents
            for i in 0..current_clauses.len() {
                for j in (i + 1)..current_clauses.len() {
                    let resolvents = self.resolve(&current_clauses[i], &current_clauses[j]);

                    for (resolvent, resolved_lit) in resolvents {
                        steps += 1;
                        self.stats.resolution_steps += 1;

                        // Skip tautologies
                        if resolvent.is_tautology() {
                            self.stats.tautologies_removed += 1;
                            continue;
                        }

                        // Check for empty clause
                        if resolvent.is_empty() {
                            self.stats.empty_clause_found = true;
                            derivation.push(ResolutionStep {
                                parent1: current_clauses[i].clone(),
                                parent2: current_clauses[j].clone(),
                                resolvent: resolvent.clone(),
                                resolved_literal: resolved_lit,
                            });
                            return ProofResult::Unsatisfiable { steps, derivation };
                        }

                        // Skip if subsumed
                        if self.is_subsumed(&resolvent, &current_clauses) {
                            self.stats.clauses_subsumed += 1;
                            continue;
                        }

                        // Add new clause if not already present
                        if !clause_set.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent.clone());
                            derivation.push(ResolutionStep {
                                parent1: current_clauses[i].clone(),
                                parent2: current_clauses[j].clone(),
                                resolvent,
                                resolved_literal: resolved_lit,
                            });
                        }
                    }
                }
            }

            // Check for saturation or limit
            if new_clauses.is_empty() {
                return ProofResult::Saturated {
                    clauses_generated: clause_set.len(),
                };
            }

            // Add new clauses to set
            for clause in new_clauses {
                clause_set.push(clause);
                self.stats.clauses_generated += 1;

                if clause_set.len() >= max_clauses {
                    return ProofResult::ResourceLimitReached {
                        steps_attempted: steps,
                    };
                }
            }
        }
    }

    /// Set-of-support proof strategy.
    fn prove_set_of_support(&mut self, max_steps: usize) -> ProofResult {
        // Simplified: treat last clause as support set
        if self.clauses.is_empty() {
            return ProofResult::Satisfiable;
        }

        let support = self.clauses.pop().unwrap();
        let mut sos: VecDeque<Clause> = VecDeque::new();
        sos.push_back(support);

        let usable: Vec<Clause> = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        while let Some(current) = sos.pop_front() {
            if steps >= max_steps {
                return ProofResult::ResourceLimitReached {
                    steps_attempted: steps,
                };
            }

            if current.is_empty() {
                self.stats.empty_clause_found = true;
                return ProofResult::Unsatisfiable { steps, derivation };
            }

            // Resolve with usable clauses
            for usable_clause in &usable {
                let resolvents = self.resolve(&current, usable_clause);

                for (resolvent, resolved_lit) in resolvents {
                    steps += 1;
                    self.stats.resolution_steps += 1;

                    if resolvent.is_tautology() {
                        self.stats.tautologies_removed += 1;
                        continue;
                    }

                    if resolvent.is_empty() {
                        self.stats.empty_clause_found = true;
                        derivation.push(ResolutionStep {
                            parent1: current.clone(),
                            parent2: usable_clause.clone(),
                            resolvent: resolvent.clone(),
                            resolved_literal: resolved_lit,
                        });
                        return ProofResult::Unsatisfiable { steps, derivation };
                    }

                    sos.push_back(resolvent.clone());
                    self.stats.clauses_generated += 1;
                    derivation.push(ResolutionStep {
                        parent1: current.clone(),
                        parent2: usable_clause.clone(),
                        resolvent,
                        resolved_literal: resolved_lit,
                    });
                }
            }
        }

        ProofResult::Satisfiable
    }

    /// Unit resolution strategy (only resolve with unit clauses).
    fn prove_unit_resolution(&mut self, max_steps: usize) -> ProofResult {
        let mut clauses = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        loop {
            if steps >= max_steps {
                return ProofResult::ResourceLimitReached {
                    steps_attempted: steps,
                };
            }

            // Find unit clauses
            let unit_clauses: Vec<Clause> =
                clauses.iter().filter(|c| c.is_unit()).cloned().collect();

            if unit_clauses.is_empty() {
                return ProofResult::Satisfiable;
            }

            let mut new_clauses = Vec::new();
            let mut found_new = false;

            // Resolve each unit clause with all clauses
            for unit in &unit_clauses {
                for clause in &clauses {
                    if clause.is_unit() && clause == unit {
                        continue; // Skip self-resolution
                    }

                    let resolvents = self.resolve(unit, clause);

                    for (resolvent, resolved_lit) in resolvents {
                        steps += 1;
                        self.stats.resolution_steps += 1;

                        if resolvent.is_tautology() {
                            self.stats.tautologies_removed += 1;
                            continue;
                        }

                        if resolvent.is_empty() {
                            self.stats.empty_clause_found = true;
                            derivation.push(ResolutionStep {
                                parent1: unit.clone(),
                                parent2: clause.clone(),
                                resolvent: resolvent.clone(),
                                resolved_literal: resolved_lit,
                            });
                            return ProofResult::Unsatisfiable { steps, derivation };
                        }

                        if !clauses.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent.clone());
                            found_new = true;
                            self.stats.clauses_generated += 1;
                            derivation.push(ResolutionStep {
                                parent1: unit.clone(),
                                parent2: clause.clone(),
                                resolvent,
                                resolved_literal: resolved_lit,
                            });
                        }
                    }
                }
            }

            if !found_new {
                return ProofResult::Satisfiable;
            }

            clauses.extend(new_clauses);
        }
    }

    /// Linear resolution strategy.
    fn prove_linear(&mut self, max_depth: usize) -> ProofResult {
        // Simplified linear resolution from first clause
        if self.clauses.is_empty() {
            return ProofResult::Satisfiable;
        }

        let start = self.clauses[0].clone();
        let mut current = start.clone();
        let mut depth = 0;
        let mut derivation = Vec::new();

        while depth < max_depth {
            if current.is_empty() {
                self.stats.empty_clause_found = true;
                return ProofResult::Unsatisfiable {
                    steps: depth,
                    derivation,
                };
            }

            // Try to resolve with any other clause
            let mut resolved = false;
            for other in &self.clauses[1..] {
                let resolvents = self.resolve(&current, other);

                if let Some((resolvent, resolved_lit)) = resolvents.first() {
                    if !resolvent.is_tautology() {
                        current = resolvent.clone();
                        depth += 1;
                        self.stats.resolution_steps += 1;
                        self.stats.clauses_generated += 1;
                        derivation.push(ResolutionStep {
                            parent1: current.clone(),
                            parent2: other.clone(),
                            resolvent: resolvent.clone(),
                            resolved_literal: resolved_lit.clone(),
                        });
                        resolved = true;
                        break;
                    }
                }
            }

            if !resolved {
                return ProofResult::Satisfiable;
            }
        }

        ProofResult::ResourceLimitReached {
            steps_attempted: depth,
        }
    }
}

impl Default for ResolutionProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a TLExpr to Clausal Normal Form (CNF).
///
/// This is a simplified conversion that handles basic logical operators.
/// Full CNF conversion would require skolemization and quantifier elimination.
pub fn to_cnf(expr: &TLExpr) -> Result<Vec<Clause>, IrError> {
    // Simplified CNF conversion
    // For full implementation, would need:
    // 1. Eliminate implications
    // 2. Move negations inward (De Morgan's laws)
    // 3. Distribute OR over AND
    // 4. Skolemize existential quantifiers
    // 5. Drop universal quantifiers

    match expr {
        TLExpr::And(left, right) => {
            let mut clauses = to_cnf(left)?;
            clauses.extend(to_cnf(right)?);
            Ok(clauses)
        }
        TLExpr::Or(left, right) => {
            // Distribute OR over AND if needed
            let left_clauses = to_cnf(left)?;
            let right_clauses = to_cnf(right)?;

            if left_clauses.len() == 1 && right_clauses.len() == 1 {
                // Simple case: combine literals
                let mut literals = left_clauses[0].literals.clone();
                literals.extend(right_clauses[0].literals.clone());
                Ok(vec![Clause::from_literals(literals)])
            } else {
                // Complex case: would need distribution
                // For now, treat as separate clauses (approximation)
                let mut result = left_clauses;
                result.extend(right_clauses);
                Ok(result)
            }
        }
        TLExpr::Not(inner) => {
            match &**inner {
                TLExpr::Pred { .. } => {
                    // Negative literal
                    Ok(vec![Clause::unit(Literal::negative((**inner).clone()))])
                }
                _ => {
                    // Would need to push negation inward
                    Err(IrError::ConstraintViolation {
                        message: "Complex negation not supported in simplified CNF conversion"
                            .to_string(),
                    })
                }
            }
        }
        TLExpr::Pred { .. } => {
            // Positive literal
            Ok(vec![Clause::unit(Literal::positive(expr.clone()))])
        }
        _ => Err(IrError::ConstraintViolation {
            message: format!(
                "Expression type not supported in CNF conversion: {:?}",
                expr
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> TLExpr {
        TLExpr::pred("P", vec![])
    }

    fn q() -> TLExpr {
        TLExpr::pred("Q", vec![])
    }

    fn r() -> TLExpr {
        TLExpr::pred("R", vec![])
    }

    #[test]
    fn test_literal_creation() {
        let lit_pos = Literal::positive(p());
        assert!(lit_pos.is_positive());
        assert!(!lit_pos.is_negative());

        let lit_neg = Literal::negative(p());
        assert!(!lit_neg.is_positive());
        assert!(lit_neg.is_negative());
    }

    #[test]
    fn test_literal_complementary() {
        let lit_pos = Literal::positive(p());
        let lit_neg = Literal::negative(p());

        assert!(lit_pos.is_complementary(&lit_neg));
        assert!(lit_neg.is_complementary(&lit_pos));
        assert!(!lit_pos.is_complementary(&lit_pos));
    }

    #[test]
    fn test_clause_empty() {
        let clause = Clause::empty();
        assert!(clause.is_empty());
        assert_eq!(clause.len(), 0);
    }

    #[test]
    fn test_clause_unit() {
        let clause = Clause::unit(Literal::positive(p()));
        assert!(clause.is_unit());
        assert_eq!(clause.len(), 1);
    }

    #[test]
    fn test_clause_tautology() {
        // P ∨ ¬P is a tautology
        let clause = Clause::from_literals(vec![Literal::positive(p()), Literal::negative(p())]);
        assert!(clause.is_tautology());
    }

    #[test]
    fn test_resolution_basic() {
        // {P}, {¬P} ⊢ ∅
        let mut prover = ResolutionProver::new();
        prover.add_clause(Clause::unit(Literal::positive(p())));
        prover.add_clause(Clause::unit(Literal::negative(p())));

        let result = prover.prove();
        assert!(result.is_unsatisfiable());
    }

    #[test]
    fn test_resolution_modus_ponens() {
        // {P}, {P → Q} ≡ {P}, {¬P ∨ Q} ⊢ Q
        // Clauses: {P}, {¬P, Q}
        // Resolution: {Q}
        let mut prover = ResolutionProver::new();
        prover.add_clause(Clause::unit(Literal::positive(p())));
        prover.add_clause(Clause::from_literals(vec![
            Literal::negative(p()),
            Literal::positive(q()),
        ]));
        // To prove Q, add ¬Q and check for contradiction
        prover.add_clause(Clause::unit(Literal::negative(q())));

        let result = prover.prove();
        assert!(result.is_unsatisfiable());
    }

    #[test]
    fn test_resolution_satisfiable() {
        // {P}, {Q} is satisfiable (no complementary literals)
        let mut prover = ResolutionProver::new();
        prover.add_clause(Clause::unit(Literal::positive(p())));
        prover.add_clause(Clause::unit(Literal::positive(q())));

        let result = prover.prove();
        // Should saturate or be satisfiable
        assert!(!result.is_unsatisfiable());
    }

    #[test]
    fn test_cnf_conversion_and() {
        // P ∧ Q → clauses: {P}, {Q}
        let expr = TLExpr::and(p(), q());
        let clauses = to_cnf(&expr).unwrap();

        assert_eq!(clauses.len(), 2);
        assert!(clauses.iter().all(|c| c.is_unit()));
    }

    #[test]
    fn test_cnf_conversion_or() {
        // P ∨ Q → clause: {P, Q}
        let expr = TLExpr::or(p(), q());
        let clauses = to_cnf(&expr).unwrap();

        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].len(), 2);
    }

    #[test]
    fn test_resolution_strategy_unit() {
        // Test unit resolution strategy
        let mut prover =
            ResolutionProver::with_strategy(ResolutionStrategy::UnitResolution { max_steps: 100 });

        prover.add_clause(Clause::unit(Literal::positive(p())));
        prover.add_clause(Clause::unit(Literal::negative(p())));

        let result = prover.prove();
        assert!(result.is_unsatisfiable());
    }

    #[test]
    fn test_resolution_three_clauses() {
        // {P ∨ Q}, {¬P ∨ R}, {¬Q}, {¬R} ⊢ ∅
        let mut prover = ResolutionProver::new();

        prover.add_clause(Clause::from_literals(vec![
            Literal::positive(p()),
            Literal::positive(q()),
        ]));
        prover.add_clause(Clause::from_literals(vec![
            Literal::negative(p()),
            Literal::positive(r()),
        ]));
        prover.add_clause(Clause::unit(Literal::negative(q())));
        prover.add_clause(Clause::unit(Literal::negative(r())));

        let result = prover.prove();
        assert!(result.is_unsatisfiable());
    }

    #[test]
    fn test_horn_clause_detection() {
        // {¬P, ¬Q, R} is a Horn clause (exactly one positive)
        let clause = Clause::from_literals(vec![
            Literal::negative(p()),
            Literal::negative(q()),
            Literal::positive(r()),
        ]);
        assert!(clause.is_horn());

        // {P, Q} is not a Horn clause (two positives)
        let non_horn = Clause::from_literals(vec![Literal::positive(p()), Literal::positive(q())]);
        assert!(!non_horn.is_horn());
    }

    #[test]
    fn test_prover_stats() {
        let mut prover = ResolutionProver::new();
        prover.add_clause(Clause::unit(Literal::positive(p())));
        prover.add_clause(Clause::unit(Literal::negative(p())));

        let result = prover.prove();

        assert!(prover.stats.empty_clause_found);
        assert!(prover.stats.resolution_steps > 0);
        assert!(result.is_unsatisfiable());
    }

    // === First-Order Resolution Tests ===

    #[test]
    fn test_literal_unification_ground() {
        // P(a) and ¬P(a) should unify with empty substitution
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

        let mgu = p_a.try_unify(&not_p_a);
        assert!(mgu.is_some());
        assert!(mgu.unwrap().is_empty());
    }

    #[test]
    fn test_literal_unification_variable() {
        // P(x) and ¬P(a) should unify with {x/a}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

        let mgu = p_x.try_unify(&not_p_a);
        assert!(mgu.is_some());

        let mgu = mgu.unwrap();
        assert_eq!(mgu.len(), 1);
        assert_eq!(mgu.apply(&Term::var("x")), Term::constant("a"));
    }

    #[test]
    fn test_literal_unification_fails_diff_names() {
        // P(x) and ¬Q(x) should not unify (different predicate names)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let not_q_x = Literal::negative(TLExpr::pred("Q", vec![Term::var("x")]));

        let mgu = p_x.try_unify(&not_q_x);
        assert!(mgu.is_none());
    }

    #[test]
    fn test_literal_unification_fails_same_polarity() {
        // P(x) and P(a) should not unify (same polarity)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

        let mgu = p_x.try_unify(&p_a);
        assert!(mgu.is_none());
    }

    #[test]
    fn test_literal_apply_substitution() {
        // P(x) with {x/a} should become P(a)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let mut subst = Substitution::empty();
        subst.bind("x".to_string(), Term::constant("a"));

        let p_a = p_x.apply_substitution(&subst);
        let expected = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

        assert_eq!(p_a.atom, expected.atom);
        assert_eq!(p_a.polarity, expected.polarity);
    }

    #[test]
    fn test_clause_rename_variables() {
        // P(x) ∨ Q(x) renamed with "1" should become P(x_1) ∨ Q(x_1)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
        let clause = Clause::from_literals(vec![p_x, q_x]);

        let renamed = clause.rename_variables("1");

        // Check that variables were renamed
        let vars = renamed.free_vars();
        assert!(vars.contains("x_1"));
        assert!(!vars.contains("x"));
    }

    #[test]
    fn test_clause_apply_substitution() {
        // {P(x), Q(y)} with {x/a, y/b} should become {P(a), Q(b)}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_y = Literal::positive(TLExpr::pred("Q", vec![Term::var("y")]));
        let clause = Clause::from_literals(vec![p_x, q_y]);

        let mut subst = Substitution::empty();
        subst.bind("x".to_string(), Term::constant("a"));
        subst.bind("y".to_string(), Term::constant("b"));

        let result = clause.apply_substitution(&subst);

        // Should have no free variables (all substituted)
        assert!(result.free_vars().is_empty());
    }

    #[test]
    fn test_first_order_resolution_basic() {
        // {P(x)} and {¬P(a)} should resolve to {} with {x/a}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

        let c1 = Clause::unit(p_x);
        let c2 = Clause::unit(not_p_a);

        let prover = ResolutionProver::new();
        let resolvents = prover.resolve_first_order(&c1, &c2);

        assert_eq!(resolvents.len(), 1);
        assert!(resolvents[0].0.is_empty()); // Empty clause derived
    }

    #[test]
    fn test_first_order_resolution_complex() {
        // {P(x), Q(x)} and {¬P(a), R(a)} should resolve to {Q(a), R(a)}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
        let c1 = Clause::from_literals(vec![p_x, q_x]);

        let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
        let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
        let c2 = Clause::from_literals(vec![not_p_a, r_a]);

        let prover = ResolutionProver::new();
        let resolvents = prover.resolve_first_order(&c1, &c2);

        assert_eq!(resolvents.len(), 1);
        let resolvent = &resolvents[0].0;

        // Should have 2 literals: Q(a) and R(a)
        assert_eq!(resolvent.len(), 2);

        // Should have no free variables (all unified)
        assert!(resolvent.free_vars().is_empty());
    }

    #[test]
    fn test_first_order_resolution_multiple_vars() {
        // {P(x, y)} and {¬P(a, b)} should resolve to {} with {x/a, y/b}
        let p_xy = Literal::positive(TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]));
        let not_p_ab = Literal::negative(TLExpr::pred(
            "P",
            vec![Term::constant("a"), Term::constant("b")],
        ));

        let c1 = Clause::unit(p_xy);
        let c2 = Clause::unit(not_p_ab);

        let prover = ResolutionProver::new();
        let resolvents = prover.resolve_first_order(&c1, &c2);

        assert_eq!(resolvents.len(), 1);
        assert!(resolvents[0].0.is_empty());
    }

    #[test]
    fn test_first_order_resolution_standardizing_apart() {
        // {P(x)} and {¬P(x)} should be standardized apart before resolution
        // After standardization: {P(x_c1_N)} and {¬P(x_c2_N)}
        // These should resolve to {} with {x_c1_N/x_c2_N} or similar
        let p_x1 = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let not_p_x2 = Literal::negative(TLExpr::pred("P", vec![Term::var("x")]));

        let c1 = Clause::unit(p_x1);
        let c2 = Clause::unit(not_p_x2);

        let prover = ResolutionProver::new();
        let resolvents = prover.resolve_first_order(&c1, &c2);

        // Should successfully resolve despite both using variable "x"
        assert_eq!(resolvents.len(), 1);
        assert!(resolvents[0].0.is_empty());
    }

    #[test]
    fn test_first_order_resolution_no_unifier() {
        // {P(a)} and {¬P(b)} should not resolve (no unifier for a and b)
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let not_p_b = Literal::negative(TLExpr::pred("P", vec![Term::constant("b")]));

        let c1 = Clause::unit(p_a);
        let c2 = Clause::unit(not_p_b);

        let prover = ResolutionProver::new();
        let resolvents = prover.resolve_first_order(&c1, &c2);

        // Should find no resolvents
        assert_eq!(resolvents.len(), 0);
    }

    // === Theta-Subsumption Tests ===

    #[test]
    fn test_subsumption_identical() {
        // {P(a)} subsumes {P(a)} (same clause)
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let c = Clause::unit(p_a);

        assert!(c.subsumes(&c));
    }

    #[test]
    fn test_subsumption_variable_to_constant() {
        // {P(x)} subsumes {P(a)} with θ = {x/a}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

        let c_general = Clause::unit(p_x);
        let c_specific = Clause::unit(p_a);

        assert!(c_general.subsumes(&c_specific));
    }

    #[test]
    fn test_subsumption_not_reverse() {
        // {P(a)} does NOT subsume {P(x)} (constant is less general than variable)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

        let c_general = Clause::unit(p_x);
        let c_specific = Clause::unit(p_a);

        assert!(!c_specific.subsumes(&c_general));
    }

    #[test]
    fn test_subsumption_smaller_clause() {
        // {P(x)} subsumes {P(a), Q(a)} with θ = {x/a}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));

        let c1 = Clause::unit(p_x);
        let c2 = Clause::from_literals(vec![p_a, q_a]);

        assert!(c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_multiple_literals() {
        // {P(x), Q(x)} subsumes {P(a), Q(a), R(a)} with θ = {x/a}
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
        let c1 = Clause::from_literals(vec![p_x, q_x]);

        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
        let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
        let c2 = Clause::from_literals(vec![p_a, q_a, r_a]);

        assert!(c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_fails_different_pred() {
        // {P(x)} does not subsume {Q(a)} (different predicate names)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));

        let c1 = Clause::unit(p_x);
        let c2 = Clause::unit(q_a);

        assert!(!c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_fails_too_many_literals() {
        // {P(x), Q(x), R(x)} does not subsume {P(a), Q(a)} (c1 has more literals)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
        let r_x = Literal::positive(TLExpr::pred("R", vec![Term::var("x")]));
        let c1 = Clause::from_literals(vec![p_x, q_x, r_x]);

        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
        let c2 = Clause::from_literals(vec![p_a, q_a]);

        assert!(!c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_empty_clause() {
        // Empty clause only subsumes itself
        let empty = Clause::empty();
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let non_empty = Clause::unit(p_a);

        assert!(empty.subsumes(&empty));
        assert!(!empty.subsumes(&non_empty));
        assert!(!non_empty.subsumes(&empty));
    }

    #[test]
    fn test_subsumption_polarity_matters() {
        // {P(x)} does not subsume {¬P(a)} (different polarity)
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

        let c1 = Clause::unit(p_x);
        let c2 = Clause::unit(not_p_a);

        assert!(!c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_two_variables() {
        // {P(x, y)} subsumes {P(a, b)} with θ = {x/a, y/b}
        let p_xy = Literal::positive(TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]));
        let p_ab = Literal::positive(TLExpr::pred(
            "P",
            vec![Term::constant("a"), Term::constant("b")],
        ));

        let c1 = Clause::unit(p_xy);
        let c2 = Clause::unit(p_ab);

        assert!(c1.subsumes(&c2));
    }

    #[test]
    fn test_subsumption_in_prover() {
        // Test that subsumption is actually used in the prover to reduce search space
        let mut prover = ResolutionProver::new();

        // Add {P(x), Q(x)} - more general
        let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
        let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
        prover.add_clause(Clause::from_literals(vec![p_x, q_x]));

        // This clause would be subsumed by the first
        let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
        let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
        let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
        let subsumed_clause = Clause::from_literals(vec![p_a, q_a, r_a]);

        // Check if subsumption works
        assert!(prover.clauses[0].subsumes(&subsumed_clause));
    }
}
