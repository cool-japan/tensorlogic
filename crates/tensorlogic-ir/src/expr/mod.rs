//! TensorLogic expressions (TLExpr).

pub mod ac_matching;
pub mod advanced_analysis;
pub mod advanced_rewriting;
mod analysis;
pub mod confluence;
pub mod defuzzification;
pub mod distributive_laws;
mod domain_validation;
pub mod ltl_ctl_utilities;
pub mod modal_axioms;
pub mod modal_equivalences;
pub mod normal_forms;
pub mod optimization;
pub mod optimization_pipeline;
pub mod probabilistic_reasoning;
pub mod rewriting;
pub mod strategy_selector;
pub mod temporal_equivalences;
mod validation;

use serde::{Deserialize, Serialize};

use crate::term::Term;

/// Aggregation operation type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateOp {
    /// Count the number of elements
    Count,
    /// Sum of all elements
    Sum,
    /// Average (mean) of elements
    Average,
    /// Maximum element
    Max,
    /// Minimum element
    Min,
    /// Product of all elements
    Product,
    /// Any (existential - true if any element is true)
    Any,
    /// All (universal - true if all elements are true)
    All,
}

/// T-norm (triangular norm) kinds for fuzzy AND operations.
/// A t-norm is a binary operation T: \[0,1\] × \[0,1\] → \[0,1\] that is:
/// - Commutative: T(a,b) = T(b,a)
/// - Associative: T(a,T(b,c)) = T(T(a,b),c)
/// - Monotonic: If a ≤ b then T(a,c) ≤ T(b,c)
/// - Has 1 as identity: T(a,1) = a
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TNormKind {
    /// Minimum t-norm (Gödel): T(a,b) = min(a,b)
    /// Standard fuzzy AND, also known as Zadeh t-norm
    Minimum,

    /// Product t-norm: T(a,b) = a * b
    /// Probabilistic interpretation of independence
    Product,

    /// Łukasiewicz t-norm: T(a,b) = max(0, a + b - 1)
    /// Strong conjunction in Łukasiewicz logic
    Lukasiewicz,

    /// Drastic t-norm: T(a,b) = { b if a=1, a if b=1, 0 otherwise }
    /// Most restrictive t-norm
    Drastic,

    /// Nilpotent minimum: T(a,b) = { min(a,b) if a+b>1, 0 otherwise }
    NilpotentMinimum,

    /// Hamacher product: T(a,b) = ab/(a+b-ab) for a,b > 0
    /// Generalizes product t-norm
    Hamacher,
}

/// T-conorm (triangular conorm) kinds for fuzzy OR operations.
/// A t-conorm is the dual of a t-norm: S(a,b) = 1 - T(1-a, 1-b)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TCoNormKind {
    /// Maximum t-conorm (Gödel): S(a,b) = max(a,b)
    /// Standard fuzzy OR, dual of minimum t-norm
    Maximum,

    /// Probabilistic sum: S(a,b) = a + b - a*b
    /// Dual of product t-norm
    ProbabilisticSum,

    /// Łukasiewicz t-conorm: S(a,b) = min(1, a + b)
    /// Bounded sum, dual of Łukasiewicz t-norm
    BoundedSum,

    /// Drastic t-conorm: S(a,b) = { b if a=0, a if b=0, 1 otherwise }
    /// Most permissive t-conorm, dual of drastic t-norm
    Drastic,

    /// Nilpotent maximum: S(a,b) = { max(a,b) if a+b<1, 1 otherwise }
    /// Dual of nilpotent minimum
    NilpotentMaximum,

    /// Hamacher sum: dual of Hamacher product
    Hamacher,
}

/// Fuzzy negation kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuzzyNegationKind {
    /// Standard fuzzy negation: N(a) = 1 - a
    Standard,

    /// Sugeno negation: N(a) = (1-a)/(1+λa) for λ > -1
    /// Parameterized family of negations
    Sugeno {
        /// Lambda parameter, must be > -1
        lambda: i32, // Using i32 to maintain Eq trait; actual value is lambda/100
    },

    /// Yager negation: N(a) = (1 - a^w)^(1/w) for w > 0
    Yager {
        /// w parameter stored as integer (actual = w/10)
        w: u32,
    },
}

/// Fuzzy implication operator kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuzzyImplicationKind {
    /// Gödel implication: I(a,b) = { 1 if a≤b, b otherwise }
    Godel,

    /// Łukasiewicz implication: I(a,b) = min(1, 1-a+b)
    Lukasiewicz,

    /// Reichenbach implication: I(a,b) = 1 - a + ab
    Reichenbach,

    /// Kleene-Dienes implication: I(a,b) = max(1-a, b)
    KleeneDienes,

    /// Rescher implication: I(a,b) = { 1 if a≤b, 0 otherwise }
    Rescher,

    /// Goguen implication: I(a,b) = { 1 if a≤b, b/a otherwise }
    Goguen,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TLExpr {
    Pred {
        name: String,
        args: Vec<Term>,
    },
    And(Box<TLExpr>, Box<TLExpr>),
    Or(Box<TLExpr>, Box<TLExpr>),
    Not(Box<TLExpr>),
    Exists {
        var: String,
        domain: String,
        body: Box<TLExpr>,
    },
    ForAll {
        var: String,
        domain: String,
        body: Box<TLExpr>,
    },
    Imply(Box<TLExpr>, Box<TLExpr>),
    Score(Box<TLExpr>),

    // Arithmetic operations
    Add(Box<TLExpr>, Box<TLExpr>),
    Sub(Box<TLExpr>, Box<TLExpr>),
    Mul(Box<TLExpr>, Box<TLExpr>),
    Div(Box<TLExpr>, Box<TLExpr>),
    Pow(Box<TLExpr>, Box<TLExpr>),
    Mod(Box<TLExpr>, Box<TLExpr>),
    Min(Box<TLExpr>, Box<TLExpr>),
    Max(Box<TLExpr>, Box<TLExpr>),

    // Unary mathematical operations
    Abs(Box<TLExpr>),
    Floor(Box<TLExpr>),
    Ceil(Box<TLExpr>),
    Round(Box<TLExpr>),
    Sqrt(Box<TLExpr>),
    Exp(Box<TLExpr>),
    Log(Box<TLExpr>),
    Sin(Box<TLExpr>),
    Cos(Box<TLExpr>),
    Tan(Box<TLExpr>),

    // Comparison operations
    Eq(Box<TLExpr>, Box<TLExpr>),
    Lt(Box<TLExpr>, Box<TLExpr>),
    Gt(Box<TLExpr>, Box<TLExpr>),
    Lte(Box<TLExpr>, Box<TLExpr>),
    Gte(Box<TLExpr>, Box<TLExpr>),

    // Conditional expression
    IfThenElse {
        condition: Box<TLExpr>,
        then_branch: Box<TLExpr>,
        else_branch: Box<TLExpr>,
    },

    // Numeric literal
    Constant(f64),

    // Aggregation operations (re-enabled with explicit output tracking support)
    Aggregate {
        op: AggregateOp,
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Optional group-by variables
        group_by: Option<Vec<String>>,
    },

    // Let binding for local variable definitions
    Let {
        var: String,
        value: Box<TLExpr>,
        body: Box<TLExpr>,
    },

    // Modal logic operators
    /// Necessity operator (□, "box"): something is necessarily true
    /// In all possible worlds or states, the expression holds
    Box(Box<TLExpr>),

    /// Possibility operator (◇, "diamond"): something is possibly true
    /// In at least one possible world or state, the expression holds
    /// Related to Box by: ◇P = ¬□¬P
    Diamond(Box<TLExpr>),

    // Temporal logic operators
    /// Next operator (X): true in the next state
    Next(Box<TLExpr>),

    /// Eventually operator (F): true in some future state
    Eventually(Box<TLExpr>),

    /// Always/Globally operator (G): true in all future states
    Always(Box<TLExpr>),

    /// Until operator (U): first expression holds until second becomes true
    Until {
        before: Box<TLExpr>,
        after: Box<TLExpr>,
    },

    // Fuzzy logic operators
    /// T-norm (fuzzy AND) with specified semantics
    TNorm {
        kind: TNormKind,
        left: Box<TLExpr>,
        right: Box<TLExpr>,
    },

    /// T-conorm (fuzzy OR) with specified semantics
    TCoNorm {
        kind: TCoNormKind,
        left: Box<TLExpr>,
        right: Box<TLExpr>,
    },

    /// Fuzzy negation with specified semantics
    FuzzyNot {
        kind: FuzzyNegationKind,
        expr: Box<TLExpr>,
    },

    /// Fuzzy implication operator
    FuzzyImplication {
        kind: FuzzyImplicationKind,
        premise: Box<TLExpr>,
        conclusion: Box<TLExpr>,
    },

    // Probabilistic operators
    /// Soft existential quantifier with temperature parameter
    /// Temperature controls how "soft" the quantifier is:
    /// - Low temperature (→0): approaches hard max (standard exists)
    /// - High temperature: smoother aggregation (log-sum-exp)
    SoftExists {
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Temperature parameter (default: 1.0)
        temperature: f64,
    },

    /// Soft universal quantifier with temperature parameter
    /// Temperature controls how "soft" the quantifier is:
    /// - Low temperature (→0): approaches hard min (standard forall)
    /// - High temperature: smoother aggregation
    SoftForAll {
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Temperature parameter (default: 1.0)
        temperature: f64,
    },

    /// Weighted rule with confidence/probability
    /// Used in probabilistic logic programming
    WeightedRule {
        weight: f64,
        rule: Box<TLExpr>,
    },

    /// Probabilistic choice between alternatives with given probabilities
    /// Probabilities should sum to 1.0
    ProbabilisticChoice {
        alternatives: Vec<(f64, TLExpr)>, // (probability, expression) pairs
    },

    // Extended temporal logic (LTL properties)
    /// Release operator (R): dual of Until
    /// P R Q means Q holds until and including when P becomes true
    Release {
        released: Box<TLExpr>,
        releaser: Box<TLExpr>,
    },

    /// Weak Until (W): P W Q means P holds until Q, but Q may never hold
    WeakUntil {
        before: Box<TLExpr>,
        after: Box<TLExpr>,
    },

    /// Strong Release (M): dual of weak until
    StrongRelease {
        released: Box<TLExpr>,
        releaser: Box<TLExpr>,
    },

    // ====== ALPHA.3 ENHANCEMENTS ======

    // Higher-order logic
    /// Lambda abstraction: λvar. body
    /// Creates a function that binds var in body
    Lambda {
        var: String,
        /// Optional type annotation for the parameter
        var_type: Option<String>,
        body: Box<TLExpr>,
    },

    /// Function application: Apply(f, arg) represents f(arg)
    Apply {
        function: Box<TLExpr>,
        argument: Box<TLExpr>,
    },

    // Set theory operations
    /// Set membership: elem ∈ set
    SetMembership {
        element: Box<TLExpr>,
        set: Box<TLExpr>,
    },

    /// Set union: A ∪ B
    SetUnion {
        left: Box<TLExpr>,
        right: Box<TLExpr>,
    },

    /// Set intersection: A ∩ B
    SetIntersection {
        left: Box<TLExpr>,
        right: Box<TLExpr>,
    },

    /// Set difference: A \ B
    SetDifference {
        left: Box<TLExpr>,
        right: Box<TLExpr>,
    },

    /// Set cardinality: |S|
    SetCardinality {
        set: Box<TLExpr>,
    },

    /// Empty set: ∅
    EmptySet,

    /// Set comprehension: { var : domain | condition }
    SetComprehension {
        var: String,
        domain: String,
        condition: Box<TLExpr>,
    },

    // Counting quantifiers
    /// Counting existential quantifier: ∃≥k x. P(x)
    /// "There exist at least k elements satisfying P"
    CountingExists {
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Minimum count threshold
        min_count: usize,
    },

    /// Counting universal quantifier: ∀≥k x. P(x)
    /// "At least k elements satisfy P"
    CountingForAll {
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Minimum count threshold
        min_count: usize,
    },

    /// Exact count quantifier: ∃=k x. P(x)
    /// "Exactly k elements satisfy P"
    ExactCount {
        var: String,
        domain: String,
        body: Box<TLExpr>,
        /// Exact count required
        count: usize,
    },

    /// Majority quantifier: Majority x. P(x)
    /// "More than half of the elements satisfy P"
    Majority {
        var: String,
        domain: String,
        body: Box<TLExpr>,
    },

    // Fixed-point operators
    /// Least fixed point (mu): μX. F(X)
    /// Used for inductive definitions
    LeastFixpoint {
        /// Variable representing the fixed point
        var: String,
        /// Function body that references var
        body: Box<TLExpr>,
    },

    /// Greatest fixed point (nu): νX. F(X)
    /// Used for coinductive definitions
    GreatestFixpoint {
        /// Variable representing the fixed point
        var: String,
        /// Function body that references var
        body: Box<TLExpr>,
    },

    // Hybrid logic
    /// Nominal (named state/world): @i
    /// Represents a specific named state in a model
    Nominal {
        name: String,
    },

    /// Satisfaction operator: @i φ
    /// "Formula φ is true at the nominal state i"
    At {
        nominal: String,
        formula: Box<TLExpr>,
    },

    /// Universal modality: E φ
    /// "φ is true in some state reachable from here"
    Somewhere {
        formula: Box<TLExpr>,
    },

    /// Universal modality (dual): A φ
    /// "φ is true in all reachable states"
    Everywhere {
        formula: Box<TLExpr>,
    },

    // Constraint programming
    /// All-different constraint: all variables must have different values
    AllDifferent {
        variables: Vec<String>,
    },

    /// Global cardinality constraint
    /// Each value in values must occur at least min and at most max times in variables
    GlobalCardinality {
        variables: Vec<String>,
        values: Vec<TLExpr>,
        min_occurrences: Vec<usize>,
        max_occurrences: Vec<usize>,
    },

    // Abductive reasoning
    /// Abducible literal: can be assumed true for explanation
    Abducible {
        name: String,
        cost: f64, // Cost of assuming this literal
    },

    /// Explanation marker: indicates this part needs explanation
    Explain {
        formula: Box<TLExpr>,
    },
}

impl TLExpr {
    pub fn pred(name: impl Into<String>, args: Vec<Term>) -> Self {
        TLExpr::Pred {
            name: name.into(),
            args,
        }
    }

    pub fn and(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Or(Box::new(left), Box::new(right))
    }

    pub fn negate(expr: TLExpr) -> Self {
        TLExpr::Not(Box::new(expr))
    }

    pub fn exists(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        TLExpr::Exists {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
        }
    }

    pub fn forall(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        TLExpr::ForAll {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
        }
    }

    pub fn imply(premise: TLExpr, conclusion: TLExpr) -> Self {
        TLExpr::Imply(Box::new(premise), Box::new(conclusion))
    }

    pub fn score(expr: TLExpr) -> Self {
        TLExpr::Score(Box::new(expr))
    }

    // Arithmetic operations
    #[allow(clippy::should_implement_trait)]
    pub fn add(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Add(Box::new(left), Box::new(right))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Sub(Box::new(left), Box::new(right))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Mul(Box::new(left), Box::new(right))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn div(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Div(Box::new(left), Box::new(right))
    }

    pub fn pow(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Pow(Box::new(left), Box::new(right))
    }

    pub fn modulo(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Mod(Box::new(left), Box::new(right))
    }

    pub fn min(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Min(Box::new(left), Box::new(right))
    }

    pub fn max(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Max(Box::new(left), Box::new(right))
    }

    // Unary mathematical operations
    pub fn abs(expr: TLExpr) -> Self {
        TLExpr::Abs(Box::new(expr))
    }

    pub fn floor(expr: TLExpr) -> Self {
        TLExpr::Floor(Box::new(expr))
    }

    pub fn ceil(expr: TLExpr) -> Self {
        TLExpr::Ceil(Box::new(expr))
    }

    pub fn round(expr: TLExpr) -> Self {
        TLExpr::Round(Box::new(expr))
    }

    pub fn sqrt(expr: TLExpr) -> Self {
        TLExpr::Sqrt(Box::new(expr))
    }

    pub fn exp(expr: TLExpr) -> Self {
        TLExpr::Exp(Box::new(expr))
    }

    pub fn log(expr: TLExpr) -> Self {
        TLExpr::Log(Box::new(expr))
    }

    pub fn sin(expr: TLExpr) -> Self {
        TLExpr::Sin(Box::new(expr))
    }

    pub fn cos(expr: TLExpr) -> Self {
        TLExpr::Cos(Box::new(expr))
    }

    pub fn tan(expr: TLExpr) -> Self {
        TLExpr::Tan(Box::new(expr))
    }

    // Comparison operations
    pub fn eq(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Eq(Box::new(left), Box::new(right))
    }

    pub fn lt(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Lt(Box::new(left), Box::new(right))
    }

    pub fn gt(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Gt(Box::new(left), Box::new(right))
    }

    pub fn lte(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Lte(Box::new(left), Box::new(right))
    }

    pub fn gte(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::Gte(Box::new(left), Box::new(right))
    }

    // Conditional
    pub fn if_then_else(condition: TLExpr, then_branch: TLExpr, else_branch: TLExpr) -> Self {
        TLExpr::IfThenElse {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }

    // Constant
    pub fn constant(value: f64) -> Self {
        TLExpr::Constant(value)
    }

    // Aggregation operations
    pub fn aggregate(
        op: AggregateOp,
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
    ) -> Self {
        TLExpr::Aggregate {
            op,
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            group_by: None,
        }
    }

    pub fn aggregate_with_group_by(
        op: AggregateOp,
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        group_by: Vec<String>,
    ) -> Self {
        TLExpr::Aggregate {
            op,
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            group_by: Some(group_by),
        }
    }

    pub fn count(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Count, var, domain, body)
    }

    pub fn sum(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Sum, var, domain, body)
    }

    pub fn average(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Average, var, domain, body)
    }

    pub fn max_agg(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Max, var, domain, body)
    }

    pub fn min_agg(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Min, var, domain, body)
    }

    pub fn product(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        Self::aggregate(AggregateOp::Product, var, domain, body)
    }

    // Let binding
    pub fn let_binding(var: impl Into<String>, value: TLExpr, body: TLExpr) -> Self {
        TLExpr::Let {
            var: var.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    // Modal logic operators
    /// Create a Box (necessity) operator.
    ///
    /// □P: "P is necessarily true" - holds in all possible worlds/states
    pub fn modal_box(expr: TLExpr) -> Self {
        TLExpr::Box(Box::new(expr))
    }

    /// Create a Diamond (possibility) operator.
    ///
    /// ◇P: "P is possibly true" - holds in at least one possible world/state
    pub fn modal_diamond(expr: TLExpr) -> Self {
        TLExpr::Diamond(Box::new(expr))
    }

    // Temporal logic operators
    /// Create a Next operator.
    ///
    /// XP: "P is true in the next state"
    pub fn next(expr: TLExpr) -> Self {
        TLExpr::Next(Box::new(expr))
    }

    /// Create an Eventually operator.
    ///
    /// FP: "P will eventually be true" - true in some future state
    pub fn eventually(expr: TLExpr) -> Self {
        TLExpr::Eventually(Box::new(expr))
    }

    /// Create an Always operator.
    ///
    /// GP: "P is always true" - true in all future states
    pub fn always(expr: TLExpr) -> Self {
        TLExpr::Always(Box::new(expr))
    }

    /// Create an Until operator.
    ///
    /// P U Q: "P holds until Q becomes true"
    pub fn until(before: TLExpr, after: TLExpr) -> Self {
        TLExpr::Until {
            before: Box::new(before),
            after: Box::new(after),
        }
    }

    // Fuzzy logic builders

    /// Create a T-norm (fuzzy AND) operation with specified semantics.
    pub fn tnorm(kind: TNormKind, left: TLExpr, right: TLExpr) -> Self {
        TLExpr::TNorm {
            kind,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a minimum T-norm (standard fuzzy AND).
    pub fn fuzzy_and(left: TLExpr, right: TLExpr) -> Self {
        Self::tnorm(TNormKind::Minimum, left, right)
    }

    /// Create a product T-norm (probabilistic AND).
    pub fn probabilistic_and(left: TLExpr, right: TLExpr) -> Self {
        Self::tnorm(TNormKind::Product, left, right)
    }

    /// Create a T-conorm (fuzzy OR) operation with specified semantics.
    pub fn tconorm(kind: TCoNormKind, left: TLExpr, right: TLExpr) -> Self {
        TLExpr::TCoNorm {
            kind,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a maximum T-conorm (standard fuzzy OR).
    pub fn fuzzy_or(left: TLExpr, right: TLExpr) -> Self {
        Self::tconorm(TCoNormKind::Maximum, left, right)
    }

    /// Create a probabilistic sum T-conorm.
    pub fn probabilistic_or(left: TLExpr, right: TLExpr) -> Self {
        Self::tconorm(TCoNormKind::ProbabilisticSum, left, right)
    }

    /// Create a fuzzy negation with specified semantics.
    pub fn fuzzy_not(kind: FuzzyNegationKind, expr: TLExpr) -> Self {
        TLExpr::FuzzyNot {
            kind,
            expr: Box::new(expr),
        }
    }

    /// Create a standard fuzzy negation (1 - x).
    pub fn standard_fuzzy_not(expr: TLExpr) -> Self {
        Self::fuzzy_not(FuzzyNegationKind::Standard, expr)
    }

    /// Create a fuzzy implication with specified semantics.
    pub fn fuzzy_imply(kind: FuzzyImplicationKind, premise: TLExpr, conclusion: TLExpr) -> Self {
        TLExpr::FuzzyImplication {
            kind,
            premise: Box::new(premise),
            conclusion: Box::new(conclusion),
        }
    }

    // Probabilistic operators builders

    /// Create a soft existential quantifier with temperature parameter.
    ///
    /// # Arguments
    /// * `var` - Variable name
    /// * `domain` - Domain name
    /// * `body` - Expression body
    /// * `temperature` - Temperature parameter (default 1.0). Lower = harder max.
    pub fn soft_exists(
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        temperature: f64,
    ) -> Self {
        TLExpr::SoftExists {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            temperature,
        }
    }

    /// Create a soft universal quantifier with temperature parameter.
    ///
    /// # Arguments
    /// * `var` - Variable name
    /// * `domain` - Domain name
    /// * `body` - Expression body
    /// * `temperature` - Temperature parameter (default 1.0). Lower = harder min.
    pub fn soft_forall(
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        temperature: f64,
    ) -> Self {
        TLExpr::SoftForAll {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            temperature,
        }
    }

    /// Create a weighted rule with confidence/probability.
    ///
    /// # Arguments
    /// * `weight` - Weight/confidence (typically in \[0,1\] for probabilities)
    /// * `rule` - The rule expression
    pub fn weighted_rule(weight: f64, rule: TLExpr) -> Self {
        TLExpr::WeightedRule {
            weight,
            rule: Box::new(rule),
        }
    }

    /// Create a probabilistic choice between alternatives.
    ///
    /// # Arguments
    /// * `alternatives` - Vector of (probability, expression) pairs. Should sum to 1.0.
    pub fn probabilistic_choice(alternatives: Vec<(f64, TLExpr)>) -> Self {
        TLExpr::ProbabilisticChoice { alternatives }
    }

    // Extended temporal logic builders

    /// Create a Release operator (R).
    ///
    /// P R Q: "Q holds until and including when P becomes true"
    pub fn release(released: TLExpr, releaser: TLExpr) -> Self {
        TLExpr::Release {
            released: Box::new(released),
            releaser: Box::new(releaser),
        }
    }

    /// Create a Weak Until operator (W).
    ///
    /// P W Q: "P holds until Q, but Q may never hold"
    pub fn weak_until(before: TLExpr, after: TLExpr) -> Self {
        TLExpr::WeakUntil {
            before: Box::new(before),
            after: Box::new(after),
        }
    }

    /// Create a Strong Release operator (M).
    ///
    /// P M Q: Dual of weak until
    pub fn strong_release(released: TLExpr, releaser: TLExpr) -> Self {
        TLExpr::StrongRelease {
            released: Box::new(released),
            releaser: Box::new(releaser),
        }
    }

    // ====== ALPHA.3 ENHANCEMENT BUILDERS ======

    // Higher-order logic builders

    /// Create a lambda abstraction: λvar. body
    ///
    /// # Arguments
    /// * `var` - The parameter name
    /// * `var_type` - Optional type annotation for the parameter
    /// * `body` - The function body
    pub fn lambda(var: impl Into<String>, var_type: Option<String>, body: TLExpr) -> Self {
        TLExpr::Lambda {
            var: var.into(),
            var_type,
            body: Box::new(body),
        }
    }

    /// Create a lambda abstraction without type annotation.
    pub fn lambda_untyped(var: impl Into<String>, body: TLExpr) -> Self {
        Self::lambda(var, None, body)
    }

    /// Create a function application: f(arg)
    ///
    /// # Arguments
    /// * `function` - The function to apply
    /// * `argument` - The argument to apply to the function
    pub fn apply(function: TLExpr, argument: TLExpr) -> Self {
        TLExpr::Apply {
            function: Box::new(function),
            argument: Box::new(argument),
        }
    }

    // Set theory builders

    /// Create a set membership test: elem ∈ set
    pub fn set_membership(element: TLExpr, set: TLExpr) -> Self {
        TLExpr::SetMembership {
            element: Box::new(element),
            set: Box::new(set),
        }
    }

    /// Create a set union: A ∪ B
    pub fn set_union(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::SetUnion {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a set intersection: A ∩ B
    pub fn set_intersection(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::SetIntersection {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a set difference: A \ B
    pub fn set_difference(left: TLExpr, right: TLExpr) -> Self {
        TLExpr::SetDifference {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a set cardinality expression: |S|
    pub fn set_cardinality(set: TLExpr) -> Self {
        TLExpr::SetCardinality { set: Box::new(set) }
    }

    /// Create an empty set: ∅
    pub fn empty_set() -> Self {
        TLExpr::EmptySet
    }

    /// Create a set comprehension: { var : domain | condition }
    pub fn set_comprehension(
        var: impl Into<String>,
        domain: impl Into<String>,
        condition: TLExpr,
    ) -> Self {
        TLExpr::SetComprehension {
            var: var.into(),
            domain: domain.into(),
            condition: Box::new(condition),
        }
    }

    // Counting quantifier builders

    /// Create a counting existential quantifier: ∃≥k x. P(x)
    ///
    /// "There exist at least k elements satisfying P"
    pub fn counting_exists(
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        min_count: usize,
    ) -> Self {
        TLExpr::CountingExists {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            min_count,
        }
    }

    /// Create a counting universal quantifier: ∀≥k x. P(x)
    ///
    /// "At least k elements satisfy P"
    pub fn counting_forall(
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        min_count: usize,
    ) -> Self {
        TLExpr::CountingForAll {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            min_count,
        }
    }

    /// Create an exact count quantifier: ∃=k x. P(x)
    ///
    /// "Exactly k elements satisfy P"
    pub fn exact_count(
        var: impl Into<String>,
        domain: impl Into<String>,
        body: TLExpr,
        count: usize,
    ) -> Self {
        TLExpr::ExactCount {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
            count,
        }
    }

    /// Create a majority quantifier: Majority x. P(x)
    ///
    /// "More than half of the elements satisfy P"
    pub fn majority(var: impl Into<String>, domain: impl Into<String>, body: TLExpr) -> Self {
        TLExpr::Majority {
            var: var.into(),
            domain: domain.into(),
            body: Box::new(body),
        }
    }

    // Fixed-point operator builders

    /// Create a least fixed point: μX. F(X)
    ///
    /// Used for inductive definitions (smallest solution)
    pub fn least_fixpoint(var: impl Into<String>, body: TLExpr) -> Self {
        TLExpr::LeastFixpoint {
            var: var.into(),
            body: Box::new(body),
        }
    }

    /// Create a greatest fixed point: νX. F(X)
    ///
    /// Used for coinductive definitions (largest solution)
    pub fn greatest_fixpoint(var: impl Into<String>, body: TLExpr) -> Self {
        TLExpr::GreatestFixpoint {
            var: var.into(),
            body: Box::new(body),
        }
    }

    // Hybrid logic builders

    /// Create a nominal (named state): @i
    pub fn nominal(name: impl Into<String>) -> Self {
        TLExpr::Nominal { name: name.into() }
    }

    /// Create a satisfaction operator: @i φ
    ///
    /// "Formula φ is true at the nominal state i"
    pub fn at(nominal: impl Into<String>, formula: TLExpr) -> Self {
        TLExpr::At {
            nominal: nominal.into(),
            formula: Box::new(formula),
        }
    }

    /// Create a "somewhere" modality: E φ
    ///
    /// "φ is true in some reachable state"
    pub fn somewhere(formula: TLExpr) -> Self {
        TLExpr::Somewhere {
            formula: Box::new(formula),
        }
    }

    /// Create an "everywhere" modality: A φ
    ///
    /// "φ is true in all reachable states"
    pub fn everywhere(formula: TLExpr) -> Self {
        TLExpr::Everywhere {
            formula: Box::new(formula),
        }
    }

    // Constraint programming builders

    /// Create an all-different constraint.
    ///
    /// All variables must have different values.
    pub fn all_different(variables: Vec<String>) -> Self {
        TLExpr::AllDifferent { variables }
    }

    /// Create a global cardinality constraint.
    ///
    /// Each value must occur within specified bounds in the variables.
    pub fn global_cardinality(
        variables: Vec<String>,
        values: Vec<TLExpr>,
        min_occurrences: Vec<usize>,
        max_occurrences: Vec<usize>,
    ) -> Self {
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        }
    }

    // Abductive reasoning builders

    /// Create an abducible literal.
    ///
    /// Can be assumed true for explanation with the given cost.
    pub fn abducible(name: impl Into<String>, cost: f64) -> Self {
        TLExpr::Abducible {
            name: name.into(),
            cost,
        }
    }

    /// Mark a formula as needing explanation.
    pub fn explain(formula: TLExpr) -> Self {
        TLExpr::Explain {
            formula: Box::new(formula),
        }
    }

    /// Substitute a variable with an expression throughout this formula.
    ///
    /// This performs capture-avoiding substitution, respecting variable shadowing
    /// in quantifiers, lambda abstractions, and let bindings.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable name to replace
    /// * `value` - The expression to substitute in place of the variable
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term};
    ///
    /// // P(x) ∧ Q(x)
    /// let p = TLExpr::pred("P", vec![Term::var("x")]);
    /// let q = TLExpr::pred("Q", vec![Term::var("x")]);
    /// let expr = TLExpr::and(p.clone(), q);
    ///
    /// // Substitute x with y
    /// let y = TLExpr::pred("y", vec![]);
    /// let result = expr.substitute("x", &y);
    /// ```
    pub fn substitute(&self, var: &str, value: &TLExpr) -> Self {
        optimization::substitution::substitute(self, var, value)
    }
}
