//! # First-Order Unification
//!
//! This module implements Robinson's unification algorithm for first-order terms.
//! Unification is a fundamental operation in automated theorem proving, logic programming,
//! and type inference.
//!
//! ## Overview
//!
//! **Unification** finds a most general unifier (MGU) that makes two terms syntactically equal:
//! - Given terms `s` and `t`, find substitution `θ` such that `sθ = tθ`
//! - The MGU is the most general such substitution (no more specific than necessary)
//!
//! ## Examples
//!
//! ```rust
//! use tensorlogic_ir::{Term, Substitution, unify_terms};
//!
//! // Unify P(x, f(y)) with P(a, f(b))
//! let term1 = Term::var("x");
//! let term2 = Term::constant("a");
//!
//! let result = unify_terms(&term1, &term2);
//! assert!(result.is_ok());
//! let subst = result.unwrap();
//! assert_eq!(subst.apply(&term1), term2);
//! ```
//!
//! ## Algorithm
//!
//! We use Robinson's unification algorithm with occur-check:
//! 1. If both terms are identical, return empty substitution
//! 2. If one is a variable, bind it to the other (with occur-check)
//! 3. If both are compound terms, recursively unify arguments
//! 4. Otherwise, unification fails
//!
//! ## Applications
//!
//! - **Theorem Proving**: Resolution requires unification of complementary literals
//! - **Logic Programming**: Pattern matching in Prolog-style languages
//! - **Type Inference**: Hindley-Milner type inference uses unification
//! - **Sequent Calculus**: Term substitution in quantifier rules

use crate::error::IrError;
use crate::term::Term;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A substitution maps variables to terms.
///
/// A substitution θ = {x₁/t₁, x₂/t₂, ...} represents the simultaneous
/// replacement of each variable xᵢ with term tᵢ.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Substitution {
    /// Mapping from variable names to their replacement terms
    bindings: HashMap<String, Term>,
}

impl Substitution {
    /// Create an empty substitution (identity).
    pub fn empty() -> Self {
        Substitution {
            bindings: HashMap::new(),
        }
    }

    /// Create a substitution with a single binding.
    pub fn singleton(var: String, term: Term) -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(var, term);
        Substitution { bindings }
    }

    /// Create a substitution from a map of bindings.
    pub fn from_map(bindings: HashMap<String, Term>) -> Self {
        Substitution { bindings }
    }

    /// Check if this is the empty substitution.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Get the number of bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Get the binding for a variable, if it exists.
    pub fn get(&self, var: &str) -> Option<&Term> {
        self.bindings.get(var)
    }

    /// Add a binding to the substitution.
    pub fn bind(&mut self, var: String, term: Term) {
        self.bindings.insert(var, term);
    }

    /// Apply this substitution to a term.
    ///
    /// This recursively replaces all occurrences of bound variables
    /// with their substituted values.
    pub fn apply(&self, term: &Term) -> Term {
        match term {
            Term::Var(name) => {
                // If variable is bound, return its substitution
                // Otherwise return the variable unchanged
                self.bindings
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| term.clone())
            }
            Term::Const(_) => term.clone(),
            Term::Typed {
                value,
                type_annotation,
            } => Term::Typed {
                value: Box::new(self.apply(value)),
                type_annotation: type_annotation.clone(),
            },
        }
    }

    /// Compose two substitutions: (σ ∘ θ)(x) = σ(θ(x))
    ///
    /// The composition applies θ first, then σ.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = HashMap::new();

        // Apply self to all bindings in other
        for (var, term) in &other.bindings {
            result.insert(var.clone(), self.apply(term));
        }

        // Add bindings from self that aren't in other
        for (var, term) in &self.bindings {
            if !result.contains_key(var) {
                result.insert(var.clone(), term.clone());
            }
        }

        Substitution::from_map(result)
    }

    /// Get all bound variables.
    pub fn domain(&self) -> Vec<String> {
        self.bindings.keys().cloned().collect()
    }

    /// Get all terms that variables are bound to.
    pub fn range(&self) -> Vec<Term> {
        self.bindings.values().cloned().collect()
    }

    /// Extend this substitution with a new binding.
    ///
    /// Returns an error if the binding conflicts with existing bindings.
    pub fn extend(&mut self, var: String, term: Term) -> Result<(), IrError> {
        if let Some(existing) = self.bindings.get(&var) {
            if existing != &term {
                return Err(IrError::UnificationFailure {
                    type1: format!("{:?}", existing),
                    type2: format!("{:?}", term),
                });
            }
        }
        self.bindings.insert(var, term);
        Ok(())
    }

    /// Try to extend this substitution with all bindings from another substitution.
    ///
    /// This is used for subsumption checking where we need to merge substitutions.
    /// Returns an error if any binding conflicts with existing bindings.
    pub fn try_extend(&mut self, other: &Substitution) -> Result<(), IrError> {
        for (var, term) in &other.bindings {
            if let Some(existing) = self.bindings.get(var) {
                if existing != term {
                    return Err(IrError::UnificationFailure {
                        type1: format!("{:?}", existing),
                        type2: format!("{:?}", term),
                    });
                }
            } else {
                self.bindings.insert(var.clone(), term.clone());
            }
        }
        Ok(())
    }
}

/// Check if a variable occurs in a term (occur-check).
///
/// This is essential for preventing infinite structures in unification.
/// For example, unifying `x` with `f(x)` would create an infinite term.
fn occurs_check(var: &str, term: &Term) -> bool {
    match term {
        Term::Var(name) => name == var,
        Term::Const(_) => false,
        Term::Typed { value, .. } => occurs_check(var, value),
    }
}

/// Unify two terms, returning the most general unifier (MGU).
///
/// The MGU is a substitution θ such that `θ(term1) = θ(term2)`.
///
/// # Errors
///
/// Returns `IrError::UnificationFailure` if the terms cannot be unified.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_ir::{Term, unify_terms};
///
/// let x = Term::var("x");
/// let a = Term::constant("a");
///
/// let mgu = unify_terms(&x, &a).unwrap();
/// assert_eq!(mgu.apply(&x), a);
/// ```
pub fn unify_terms(term1: &Term, term2: &Term) -> Result<Substitution, IrError> {
    unify_impl(term1, term2, &mut Substitution::empty())
}

/// Internal unification implementation with accumulating substitution.
fn unify_impl(
    term1: &Term,
    term2: &Term,
    subst: &mut Substitution,
) -> Result<Substitution, IrError> {
    // Apply current substitution to both terms
    let t1 = subst.apply(term1);
    let t2 = subst.apply(term2);

    match (&t1, &t2) {
        // Both are the same variable
        (Term::Var(n1), Term::Var(n2)) if n1 == n2 => Ok(subst.clone()),

        // t1 is a variable, bind it to t2
        (Term::Var(name), _) => {
            if occurs_check(name, &t2) {
                return Err(IrError::UnificationFailure {
                    type1: format!("{:?}", t1),
                    type2: format!("{:?}", t2),
                });
            }
            subst.bind(name.clone(), t2.clone());
            Ok(subst.clone())
        }

        // t2 is a variable, bind it to t1
        (_, Term::Var(name)) => {
            if occurs_check(name, &t1) {
                return Err(IrError::UnificationFailure {
                    type1: format!("{:?}", t1),
                    type2: format!("{:?}", t2),
                });
            }
            subst.bind(name.clone(), t1.clone());
            Ok(subst.clone())
        }

        // Both are constants
        (Term::Const(v1), Term::Const(v2)) => {
            if v1 == v2 {
                Ok(subst.clone())
            } else {
                Err(IrError::UnificationFailure {
                    type1: format!("{:?}", t1),
                    type2: format!("{:?}", t2),
                })
            }
        }

        // Both are typed terms - unify the inner terms
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
            // Type annotations must match
            if ty1 != ty2 {
                return Err(IrError::UnificationFailure {
                    type1: format!("{:?}", t1),
                    type2: format!("{:?}", t2),
                });
            }
            unify_impl(inner1, inner2, subst)
        }

        // One typed, one not - unify inner term with the other
        (Term::Typed { value, .. }, other) | (other, Term::Typed { value, .. }) => {
            unify_impl(value, other, subst)
        }
    }
}

/// Attempt to unify a list of term pairs.
///
/// This is useful for unifying predicate arguments in resolution.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_ir::{Term, unify_term_list};
///
/// let pairs = vec![
///     (Term::var("x"), Term::constant("a")),
///     (Term::var("y"), Term::constant("b")),
/// ];
///
/// let mgu = unify_term_list(&pairs).unwrap();
/// assert_eq!(mgu.len(), 2);
/// ```
pub fn unify_term_list(pairs: &[(Term, Term)]) -> Result<Substitution, IrError> {
    let mut subst = Substitution::empty();
    for (t1, t2) in pairs {
        subst = unify_impl(t1, t2, &mut subst)?;
    }
    Ok(subst)
}

/// Check if two terms are unifiable (without computing the unifier).
pub fn are_unifiable(term1: &Term, term2: &Term) -> bool {
    unify_terms(term1, term2).is_ok()
}

/// Rename variables in a term to avoid conflicts.
///
/// This is useful when applying quantifier rules in sequent calculus.
pub fn rename_vars(term: &Term, suffix: &str) -> Term {
    match term {
        Term::Var(name) => Term::Var(format!("{}_{}", name, suffix)),
        Term::Const(_) => term.clone(),
        Term::Typed {
            value,
            type_annotation,
        } => Term::Typed {
            value: Box::new(rename_vars(value, suffix)),
            type_annotation: type_annotation.clone(),
        },
    }
}

// ============================================================================
// Anti-Unification (Most Specific Generalization)
// ============================================================================

/// Anti-unification finds the most specific generalization (MSG) of two terms.
///
/// While unification finds a substitution that makes two terms equal,
/// anti-unification finds a term that generalizes both input terms.
///
/// ## Examples
///
/// ```rust
/// use tensorlogic_ir::{Term, anti_unify_terms};
///
/// // anti_unify(f(a, b), f(c, d)) = f(X, Y)
/// // where X and Y are fresh variables
/// let t1 = Term::constant("a");
/// let t2 = Term::constant("b");
///
/// let (gen, _subst1, _subst2) = anti_unify_terms(&t1, &t2);
/// // gen is a fresh variable that generalizes both a and b
/// ```
///
/// ## Applications
///
/// - **Inductive Logic Programming**: Learn patterns from examples
/// - **Program Synthesis**: Generalize from concrete cases
/// - **Code Clone Detection**: Find common structure in code
/// - **Proof Generalization**: Abstract from specific proofs
pub fn anti_unify_terms(term1: &Term, term2: &Term) -> (Term, Substitution, Substitution) {
    let mut var_counter = 0;
    let mut subst1 = Substitution::empty();
    let mut subst2 = Substitution::empty();

    let gen = anti_unify_impl(term1, term2, &mut var_counter, &mut subst1, &mut subst2);
    (gen, subst1, subst2)
}

/// Internal implementation of anti-unification with fresh variable generation.
fn anti_unify_impl(
    term1: &Term,
    term2: &Term,
    var_counter: &mut usize,
    subst1: &mut Substitution,
    subst2: &mut Substitution,
) -> Term {
    match (term1, term2) {
        // If both are the same constant, return the constant
        (Term::Const(c1), Term::Const(c2)) if c1 == c2 => term1.clone(),

        // If both are the same variable, return the variable
        (Term::Var(v1), Term::Var(v2)) if v1 == v2 => term1.clone(),

        // For typed terms with same type annotation, anti-unify the inner values
        (
            Term::Typed {
                value: inner1,
                type_annotation: ty1,
            },
            Term::Typed {
                value: inner2,
                type_annotation: ty2,
            },
        ) if ty1 == ty2 => {
            let inner_gen = anti_unify_impl(inner1, inner2, var_counter, subst1, subst2);
            Term::Typed {
                value: Box::new(inner_gen),
                type_annotation: ty1.clone(),
            }
        }

        // Otherwise, introduce a fresh variable
        _ => {
            *var_counter += 1;
            let fresh_var = Term::Var(format!("_G{}", var_counter));

            // Record the substitutions: fresh_var maps to term1 in subst1, term2 in subst2
            subst1.bind(format!("_G{}", var_counter), term1.clone());
            subst2.bind(format!("_G{}", var_counter), term2.clone());

            fresh_var
        }
    }
}

/// Compute the least general generalization (LGG) of a list of terms.
///
/// This repeatedly applies anti-unification to find a term that generalizes all inputs.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_ir::{Term, lgg_terms};
///
/// let terms = vec![
///     Term::constant("a"),
///     Term::constant("b"),
///     Term::constant("c"),
/// ];
///
/// let (gen, substs) = lgg_terms(&terms);
/// // gen is a fresh variable that generalizes all three constants
/// assert_eq!(substs.len(), 3);
/// ```
pub fn lgg_terms(terms: &[Term]) -> (Term, Vec<Substitution>) {
    if terms.is_empty() {
        return (Term::Var("_Empty".to_string()), vec![]);
    }

    if terms.len() == 1 {
        return (terms[0].clone(), vec![Substitution::empty()]);
    }

    // Start with first two terms
    let (mut gen, subst1, subst2) = anti_unify_terms(&terms[0], &terms[1]);
    let mut substs = vec![subst1, subst2];

    // Generalize with remaining terms
    for term in &terms[2..] {
        let (new_gen, gen_subst, term_subst) = anti_unify_terms(&gen, term);
        gen = new_gen;

        // Update existing substitutions by composing with gen_subst
        for s in &mut substs {
            *s = gen_subst.compose(s);
        }

        // Add new term's substitution
        substs.push(term_subst);
    }

    (gen, substs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_substitution() {
        let subst = Substitution::empty();
        assert!(subst.is_empty());
        assert_eq!(subst.len(), 0);

        let term = Term::var("x");
        assert_eq!(subst.apply(&term), term);
    }

    #[test]
    fn test_singleton_substitution() {
        let subst = Substitution::singleton("x".to_string(), Term::constant("a"));
        assert_eq!(subst.len(), 1);

        let x = Term::var("x");
        let a = Term::constant("a");
        assert_eq!(subst.apply(&x), a);
    }

    #[test]
    fn test_substitution_application() {
        let mut subst = Substitution::empty();
        subst.bind("x".to_string(), Term::constant("a"));
        subst.bind("y".to_string(), Term::constant("b"));

        let x = Term::var("x");
        let y = Term::var("y");
        let z = Term::var("z");

        assert_eq!(subst.apply(&x), Term::constant("a"));
        assert_eq!(subst.apply(&y), Term::constant("b"));
        assert_eq!(subst.apply(&z), z); // Unbound variable unchanged
    }

    #[test]
    fn test_unify_var_constant() {
        let x = Term::var("x");
        let a = Term::constant("a");

        let mgu = unify_terms(&x, &a).unwrap();
        assert_eq!(mgu.apply(&x), a);
    }

    #[test]
    fn test_unify_same_variable() {
        let x = Term::var("x");
        let mgu = unify_terms(&x, &x).unwrap();
        assert!(mgu.is_empty());
    }

    #[test]
    fn test_unify_different_constants() {
        let a = Term::constant("a");
        let b = Term::constant("b");

        let result = unify_terms(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_same_constant() {
        let a = Term::constant("a");
        let mgu = unify_terms(&a, &a).unwrap();
        assert!(mgu.is_empty());
    }

    #[test]
    fn test_occur_check() {
        let x = Term::var("x");
        assert!(occurs_check("x", &x));
        assert!(!occurs_check("y", &x));

        let a = Term::constant("a");
        assert!(!occurs_check("x", &a));
    }

    #[test]
    fn test_substitution_composition() {
        // σ = {x/a}
        let sigma = Substitution::singleton("x".to_string(), Term::constant("a"));
        // θ = {y/x}
        let theta = Substitution::singleton("y".to_string(), Term::var("x"));

        // σ ∘ θ = {x/a, y/a}
        let composed = sigma.compose(&theta);
        assert_eq!(composed.len(), 2);
        assert_eq!(composed.apply(&Term::var("x")), Term::constant("a"));
        assert_eq!(composed.apply(&Term::var("y")), Term::constant("a"));
    }

    #[test]
    fn test_unify_term_list() {
        let pairs = vec![
            (Term::var("x"), Term::constant("a")),
            (Term::var("y"), Term::constant("b")),
            (Term::var("z"), Term::var("x")),
        ];

        let mgu = unify_term_list(&pairs).unwrap();
        assert_eq!(mgu.len(), 3);
        assert_eq!(mgu.apply(&Term::var("x")), Term::constant("a"));
        assert_eq!(mgu.apply(&Term::var("y")), Term::constant("b"));
        assert_eq!(mgu.apply(&Term::var("z")), Term::constant("a"));
    }

    #[test]
    fn test_are_unifiable() {
        let x = Term::var("x");
        let a = Term::constant("a");
        let b = Term::constant("b");

        assert!(are_unifiable(&x, &a));
        assert!(are_unifiable(&a, &a));
        assert!(!are_unifiable(&a, &b));
    }

    #[test]
    fn test_rename_vars() {
        let x = Term::var("x");
        let renamed = rename_vars(&x, "1");
        assert_eq!(renamed, Term::var("x_1"));

        let a = Term::constant("a");
        let renamed_const = rename_vars(&a, "1");
        assert_eq!(renamed_const, a); // Constants unchanged
    }

    #[test]
    fn test_extend_substitution() {
        let mut subst = Substitution::empty();
        assert!(subst.extend("x".to_string(), Term::constant("a")).is_ok());
        assert!(subst.extend("y".to_string(), Term::constant("b")).is_ok());

        // Extending with same binding is OK
        assert!(subst.extend("x".to_string(), Term::constant("a")).is_ok());

        // Extending with conflicting binding fails
        assert!(subst.extend("x".to_string(), Term::constant("b")).is_err());
    }

    #[test]
    fn test_typed_term_unification() {
        use crate::term::TypeAnnotation;

        let x = Term::Typed {
            value: Box::new(Term::var("x")),
            type_annotation: TypeAnnotation::new("Int"),
        };
        let a = Term::Typed {
            value: Box::new(Term::constant("5")),
            type_annotation: TypeAnnotation::new("Int"),
        };

        let mgu = unify_terms(&x, &a).unwrap();
        assert_eq!(mgu.len(), 1);
    }

    // === Anti-Unification Tests ===

    #[test]
    fn test_anti_unify_same_constant() {
        // anti_unify(a, a) = a
        let a1 = Term::constant("a");
        let a2 = Term::constant("a");

        let (gen, subst1, subst2) = anti_unify_terms(&a1, &a2);

        // Should return the constant itself
        assert_eq!(gen, a1);
        assert!(subst1.is_empty());
        assert!(subst2.is_empty());
    }

    #[test]
    fn test_anti_unify_different_constants() {
        // anti_unify(a, b) = X (fresh variable)
        let a = Term::constant("a");
        let b = Term::constant("b");

        let (gen, subst1, subst2) = anti_unify_terms(&a, &b);

        // Should return a fresh variable
        match gen {
            Term::Var(name) => assert!(name.starts_with("_G")),
            _ => panic!("Expected fresh variable"),
        }

        // Substitutions should map the fresh variable to a and b respectively
        assert_eq!(subst1.len(), 1);
        assert_eq!(subst2.len(), 1);
    }

    #[test]
    fn test_anti_unify_variable_constant() {
        // anti_unify(x, a) = X (fresh variable)
        let x = Term::var("x");
        let a = Term::constant("a");

        let (gen, _subst1, _subst2) = anti_unify_terms(&x, &a);

        // Should return a fresh variable (since x ≠ a)
        if let Term::Var(name) = gen {
            // Could be x if they're the same, or _GN if different
            assert!(name == "x" || name.starts_with("_G"));
        }

        // Should have recorded the generalization
        // (substitutions may be empty or non-empty depending on the terms)
    }

    #[test]
    fn test_anti_unify_same_variable() {
        // anti_unify(x, x) = x
        let x1 = Term::var("x");
        let x2 = Term::var("x");

        let (gen, subst1, subst2) = anti_unify_terms(&x1, &x2);

        // Should return the variable itself
        assert_eq!(gen, x1);
        assert!(subst1.is_empty());
        assert!(subst2.is_empty());
    }

    #[test]
    fn test_anti_unify_typed_terms() {
        use crate::term::TypeAnnotation;

        // anti_unify(Int(5), Int(10)) = Int(X)
        let t1 = Term::Typed {
            value: Box::new(Term::constant("5")),
            type_annotation: TypeAnnotation::new("Int"),
        };
        let t2 = Term::Typed {
            value: Box::new(Term::constant("10")),
            type_annotation: TypeAnnotation::new("Int"),
        };

        let (gen, _subst1, _subst2) = anti_unify_terms(&t1, &t2);

        // Should return Int(X) where X is fresh
        match gen {
            Term::Typed {
                value,
                type_annotation,
            } => {
                assert_eq!(type_annotation.type_name, "Int");
                match *value {
                    Term::Var(name) => assert!(name.starts_with("_G")),
                    _ => panic!("Expected fresh variable inside typed term"),
                }
            }
            _ => panic!("Expected typed term"),
        }
    }

    #[test]
    fn test_lgg_single_term() {
        // LGG([a]) = a
        let terms = vec![Term::constant("a")];
        let (gen, substs) = lgg_terms(&terms);

        assert_eq!(gen, Term::constant("a"));
        assert_eq!(substs.len(), 1);
        assert!(substs[0].is_empty());
    }

    #[test]
    fn test_lgg_two_same_terms() {
        // LGG([a, a]) = a
        let terms = vec![Term::constant("a"), Term::constant("a")];
        let (gen, substs) = lgg_terms(&terms);

        assert_eq!(gen, Term::constant("a"));
        assert_eq!(substs.len(), 2);
    }

    #[test]
    fn test_lgg_two_different_terms() {
        // LGG([a, b]) = X
        let terms = vec![Term::constant("a"), Term::constant("b")];
        let (gen, substs) = lgg_terms(&terms);

        // Should be a fresh variable
        match gen {
            Term::Var(name) => assert!(name.starts_with("_G")),
            _ => panic!("Expected fresh variable"),
        }

        assert_eq!(substs.len(), 2);
    }

    #[test]
    fn test_lgg_three_terms() {
        // LGG([a, b, c]) = X (all different)
        let terms = vec![
            Term::constant("a"),
            Term::constant("b"),
            Term::constant("c"),
        ];
        let (gen, substs) = lgg_terms(&terms);

        // Should be a fresh variable
        match gen {
            Term::Var(name) => assert!(name.starts_with("_G")),
            _ => panic!("Expected fresh variable"),
        }

        assert_eq!(substs.len(), 3);
    }

    #[test]
    fn test_lgg_empty() {
        // LGG([]) = special empty variable
        let terms: Vec<Term> = vec![];
        let (gen, substs) = lgg_terms(&terms);

        match gen {
            Term::Var(name) => assert_eq!(name, "_Empty"),
            _ => panic!("Expected _Empty variable"),
        }

        assert_eq!(substs.len(), 0);
    }

    #[test]
    fn test_anti_unify_preserves_structure() {
        // When generalizing identical structures, structure should be preserved
        let a1 = Term::constant("a");
        let a2 = Term::constant("a");

        let (gen, _, _) = anti_unify_terms(&a1, &a2);

        // Should preserve the constant
        assert_eq!(gen, Term::constant("a"));
    }
}
