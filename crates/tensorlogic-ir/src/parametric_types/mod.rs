//! Parametric type system with type constructors and unification.
//!
//! This module extends the basic TypeAnnotation system with support for parametric types
//! (generics), enabling types like `List<T>`, `Option<T>`, `Tuple<A,B>`, etc.
//!
//! # Examples
//!
//! ```
//! use tensorlogic_ir::parametric_types::{ParametricType, TypeConstructor, Kind};
//!
//! // Simple types
//! let int_type = ParametricType::concrete("Int");
//! let string_type = ParametricType::concrete("String");
//!
//! // Type variables
//! let t_var = ParametricType::variable("T");
//!
//! // Parametric types
//! let list_int = ParametricType::apply(TypeConstructor::List, vec![int_type.clone()]);
//! let option_string = ParametricType::apply(TypeConstructor::Option, vec![string_type.clone()]);
//! let pair = ParametricType::apply(
//!     TypeConstructor::Tuple,
//!     vec![int_type.clone(), string_type.clone()]
//! );
//! ```
//!
//! # Type Unification
//!
//! The module provides type unification for finding substitutions that make two types equal:
//!
//! ```
//! use tensorlogic_ir::parametric_types::{ParametricType, TypeConstructor, unify};
//!
//! let t_var = ParametricType::variable("T");
//! let int_type = ParametricType::concrete("Int");
//! let list_t = ParametricType::apply(TypeConstructor::List, vec![t_var.clone()]);
//! let list_int = ParametricType::apply(TypeConstructor::List, vec![int_type.clone()]);
//!
//! // Unify List<T> with List<Int> -> T = Int
//! let subst = unify(&list_t, &list_int).unwrap();
//! assert_eq!(subst.get("T").unwrap(), &int_type);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::IrError;

/// Kind of a type or type constructor.
///
/// Kinds classify types by their arity:
/// - `Star` (*): Proper types like `Int`, `String`, `List<Int>`
/// - `Arrow(k1, k2)`: Type constructors like `List` (* -> *), `Function` (* -> * -> *)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// Proper type (kind *)
    Star,
    /// Type constructor (kind k1 -> k2)
    Arrow(Box<Kind>, Box<Kind>),
}

impl Kind {
    /// Create a kind for a type constructor with n parameters
    pub fn constructor(arity: usize) -> Self {
        match arity {
            0 => Kind::Star,
            1 => Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)),
            n => Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::constructor(n - 1))),
        }
    }

    /// Get the arity of a type constructor kind
    pub fn arity(&self) -> usize {
        match self {
            Kind::Star => 0,
            Kind::Arrow(_, rest) => 1 + rest.arity(),
        }
    }

    /// Check if this is a proper type (kind *)
    pub fn is_star(&self) -> bool {
        matches!(self, Kind::Star)
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Star => write!(f, "*"),
            Kind::Arrow(k1, k2) => write!(f, "{} -> {}", k1, k2),
        }
    }
}

/// Built-in type constructors.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeConstructor {
    /// List type constructor (kind * -> *)
    List,
    /// Option type constructor (kind * -> *)
    Option,
    /// Pair/Tuple type constructor (kind * -> * -> *)
    Tuple,
    /// N-ary tuple type constructor (kind * -> * -> ... -> *)
    TupleN { arity: usize },
    /// Array type constructor with dimension (kind * -> *)
    Array { dimensions: usize },
    /// Function type constructor (kind * -> * -> *)
    Function,
    /// Set type constructor (kind * -> *)
    Set,
    /// Map type constructor (kind * -> * -> *)
    Map,
    /// Result type constructor (kind * -> * -> *)
    Result,
    /// Either type constructor (kind * -> * -> *)
    Either,
    /// Custom type constructor
    Custom { name: String, arity: usize },
}

impl TypeConstructor {
    /// Get the kind of this type constructor
    pub fn kind(&self) -> Kind {
        match self {
            TypeConstructor::List => Kind::constructor(1),
            TypeConstructor::Option => Kind::constructor(1),
            TypeConstructor::Tuple => Kind::constructor(2),
            TypeConstructor::TupleN { arity } => Kind::constructor(*arity),
            TypeConstructor::Array { .. } => Kind::constructor(1),
            TypeConstructor::Function => Kind::constructor(2),
            TypeConstructor::Set => Kind::constructor(1),
            TypeConstructor::Map => Kind::constructor(2),
            TypeConstructor::Result => Kind::constructor(2),
            TypeConstructor::Either => Kind::constructor(2),
            TypeConstructor::Custom { arity, .. } => Kind::constructor(*arity),
        }
    }

    /// Get the arity of this type constructor
    pub fn arity(&self) -> usize {
        self.kind().arity()
    }

    /// Create a custom type constructor
    pub fn custom(name: impl Into<String>, arity: usize) -> Self {
        TypeConstructor::Custom {
            name: name.into(),
            arity,
        }
    }
}

impl fmt::Display for TypeConstructor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeConstructor::List => write!(f, "List"),
            TypeConstructor::Option => write!(f, "Option"),
            TypeConstructor::Tuple => write!(f, "Tuple"),
            TypeConstructor::TupleN { arity } => write!(f, "Tuple{}", arity),
            TypeConstructor::Array { dimensions } => write!(f, "Array{}", dimensions),
            TypeConstructor::Function => write!(f, "->"),
            TypeConstructor::Set => write!(f, "Set"),
            TypeConstructor::Map => write!(f, "Map"),
            TypeConstructor::Result => write!(f, "Result"),
            TypeConstructor::Either => write!(f, "Either"),
            TypeConstructor::Custom { name, .. } => write!(f, "{}", name),
        }
    }
}

/// Parametric type with support for type variables and type constructors.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParametricType {
    /// Type variable (e.g., T, A, B)
    Variable(String),
    /// Concrete type (e.g., Int, String, Person)
    Concrete(String),
    /// Type application: constructor applied to arguments
    /// E.g., `List<Int>`, `Option<String>`, `Tuple<Int, String>`
    Application {
        constructor: TypeConstructor,
        arguments: Vec<ParametricType>,
    },
    /// Recursive type (mu-type)
    /// E.g., μα. 1 + α (natural numbers), μL. Nil | Cons(a, L) (lists)
    Recursive(Box<RecursiveType>),
    /// Row type for record polymorphism
    /// E.g., {name: String, age: Int | r}
    Row(RowType),
}

impl ParametricType {
    /// Create a type variable
    pub fn variable(name: impl Into<String>) -> Self {
        ParametricType::Variable(name.into())
    }

    /// Create a concrete type
    pub fn concrete(name: impl Into<String>) -> Self {
        ParametricType::Concrete(name.into())
    }

    /// Apply a type constructor to arguments
    pub fn apply(constructor: TypeConstructor, arguments: Vec<ParametricType>) -> Self {
        ParametricType::Application {
            constructor,
            arguments,
        }
    }

    /// Create a List type
    pub fn list(elem_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::List, vec![elem_type])
    }

    /// Create an Option type
    pub fn option(elem_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Option, vec![elem_type])
    }

    /// Create a Tuple type
    ///
    /// Supports tuples of any arity:
    /// - 0 elements: Unit type (Tuple0)
    /// - 2 elements: Uses the binary Tuple constructor
    /// - Other arities: Uses TupleN constructor
    pub fn tuple(types: Vec<ParametricType>) -> Self {
        match types.len() {
            0 => ParametricType::apply(TypeConstructor::TupleN { arity: 0 }, vec![]),
            2 => ParametricType::apply(TypeConstructor::Tuple, types),
            n => ParametricType::apply(TypeConstructor::TupleN { arity: n }, types),
        }
    }

    /// Create a unit type (empty tuple)
    pub fn unit() -> Self {
        ParametricType::tuple(vec![])
    }

    /// Create a triple (3-tuple)
    pub fn triple(t1: ParametricType, t2: ParametricType, t3: ParametricType) -> Self {
        ParametricType::tuple(vec![t1, t2, t3])
    }

    /// Create a Result type
    pub fn result(ok_type: ParametricType, err_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Result, vec![ok_type, err_type])
    }

    /// Create an Either type
    pub fn either(left_type: ParametricType, right_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Either, vec![left_type, right_type])
    }

    /// Create a Function type (A -> B)
    pub fn function(from: ParametricType, to: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Function, vec![from, to])
    }

    /// Create a Set type
    pub fn set(elem_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Set, vec![elem_type])
    }

    /// Create a Map type
    pub fn map(key_type: ParametricType, value_type: ParametricType) -> Self {
        ParametricType::apply(TypeConstructor::Map, vec![key_type, value_type])
    }

    /// Create an Array type with dimensions
    pub fn array(elem_type: ParametricType, dimensions: usize) -> Self {
        ParametricType::apply(TypeConstructor::Array { dimensions }, vec![elem_type])
    }

    /// Check if this is a type variable
    pub fn is_variable(&self) -> bool {
        matches!(self, ParametricType::Variable(_))
    }

    /// Check if this is a concrete type
    pub fn is_concrete(&self) -> bool {
        matches!(self, ParametricType::Concrete(_))
    }

    /// Get all free type variables in this type
    pub fn free_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_free_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_free_variables(&self, vars: &mut Vec<String>) {
        match self {
            ParametricType::Variable(name) => vars.push(name.clone()),
            ParametricType::Concrete(_) => {}
            ParametricType::Application { arguments, .. } => {
                for arg in arguments {
                    arg.collect_free_variables(vars);
                }
            }
            ParametricType::Recursive(rec) => {
                // Collect vars from body, excluding the bound variable
                let body_vars = rec.body.free_variables();
                for v in body_vars {
                    if v != rec.var {
                        vars.push(v);
                    }
                }
            }
            ParametricType::Row(row) => {
                vars.extend(row.free_variables());
            }
        }
    }

    /// Apply a substitution to this type
    pub fn substitute(&self, substitution: &TypeSubstitution) -> ParametricType {
        match self {
            ParametricType::Variable(name) => substitution
                .get(name)
                .cloned()
                .unwrap_or_else(|| self.clone()),
            ParametricType::Concrete(_) => self.clone(),
            ParametricType::Application {
                constructor,
                arguments,
            } => ParametricType::Application {
                constructor: constructor.clone(),
                arguments: arguments
                    .iter()
                    .map(|arg| arg.substitute(substitution))
                    .collect(),
            },
            ParametricType::Recursive(rec) => {
                // Don't substitute the bound variable
                let mut new_subst = substitution.clone();
                new_subst.remove(&rec.var);
                ParametricType::Recursive(Box::new(RecursiveType {
                    var: rec.var.clone(),
                    body: rec.body.substitute(&new_subst),
                }))
            }
            ParametricType::Row(row) => {
                let new_fields = row
                    .fields
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute(substitution)))
                    .collect();
                let new_rest = row.rest.as_ref().map(|r| {
                    if let Some(ParametricType::Variable(new_var)) = substitution.get(r) {
                        new_var.clone()
                    } else {
                        r.clone()
                    }
                });
                ParametricType::Row(RowType {
                    fields: new_fields,
                    rest: new_rest,
                })
            }
        }
    }

    /// Check if this type variable occurs in another type (occurs check for unification).
    /// This is used to detect infinite types like `T = List<T>`.
    /// Returns true if this type variable appears inside the structure of other type.
    pub fn occurs_in(&self, other: &ParametricType) -> bool {
        // Only makes sense for type variables
        if let ParametricType::Variable(v) = self {
            match other {
                // Variable occurs in itself (but this is OK, will be caught by unification)
                ParametricType::Variable(v2) => v == v2,
                ParametricType::Concrete(_) => false,
                // Check if variable occurs in application arguments
                ParametricType::Application { arguments, .. } => {
                    arguments.iter().any(|arg| self.occurs_in(arg))
                }
                // Check in recursive type body (excluding the bound variable)
                ParametricType::Recursive(rec) => {
                    if v == &rec.var {
                        false // Bound variable, doesn't count
                    } else {
                        self.occurs_in(&rec.body)
                    }
                }
                // Check in row type fields
                ParametricType::Row(row) => {
                    row.fields.iter().any(|(_, t)| self.occurs_in(t))
                        || row.rest.as_ref().is_some_and(|r| r == v)
                }
            }
        } else {
            // For non-variables, just check equality
            self == other
        }
    }

    /// Get the kind of this type
    pub fn kind(&self) -> Kind {
        match self {
            ParametricType::Variable(_) => Kind::Star,
            ParametricType::Concrete(_) => Kind::Star,
            ParametricType::Application {
                constructor,
                arguments,
            } => {
                let mut kind = constructor.kind();
                for _ in arguments {
                    match kind {
                        Kind::Arrow(_, rest) => kind = *rest,
                        Kind::Star => return Kind::Star, // Fully applied
                    }
                }
                kind
            }
            ParametricType::Recursive(_) => Kind::Star, // Recursive types are proper types
            ParametricType::Row(_) => Kind::Star,       // Row types are proper types
        }
    }

    /// Check if this is a well-kinded type (kind *)
    pub fn is_well_kinded(&self) -> bool {
        match self {
            ParametricType::Variable(_) | ParametricType::Concrete(_) => true,
            ParametricType::Application {
                constructor,
                arguments,
            } => {
                // Check arity matches
                if constructor.arity() != arguments.len() {
                    return false;
                }
                // Check all arguments are well-kinded
                arguments.iter().all(|arg| arg.is_well_kinded())
            }
            ParametricType::Recursive(rec) => {
                // Body must be well-kinded and must mention the bound variable
                rec.body.is_well_kinded() && rec.is_well_formed()
            }
            ParametricType::Row(row) => {
                // All field types must be well-kinded
                row.fields.iter().all(|(_, t)| t.is_well_kinded())
            }
        }
    }
}

impl fmt::Display for ParametricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParametricType::Variable(name) => write!(f, "{}", name),
            ParametricType::Concrete(name) => write!(f, "{}", name),
            ParametricType::Application {
                constructor,
                arguments,
            } => {
                write!(f, "{}", constructor)?;
                if !arguments.is_empty() {
                    write!(f, "<")?;
                    for (i, arg) in arguments.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            ParametricType::Recursive(rec) => write!(f, "{}", rec),
            ParametricType::Row(row) => write!(f, "{}", row),
        }
    }
}

/// Type substitution: mapping from type variables to types.
pub type TypeSubstitution = HashMap<String, ParametricType>;

/// Unify two types, returning a substitution that makes them equal.
///
/// This implements Robinson's unification algorithm.
///
/// # Examples
///
/// ```
/// use tensorlogic_ir::parametric_types::{ParametricType, TypeConstructor, unify};
///
/// let t = ParametricType::variable("T");
/// let int_type = ParametricType::concrete("Int");
/// let list_t = ParametricType::list(t.clone());
/// let list_int = ParametricType::list(int_type.clone());
///
/// let subst = unify(&list_t, &list_int).unwrap();
/// assert_eq!(subst.get("T").unwrap(), &int_type);
/// ```
pub fn unify(type1: &ParametricType, type2: &ParametricType) -> Result<TypeSubstitution, IrError> {
    let mut subst = HashMap::new();
    unify_with_subst(type1, type2, &mut subst)?;
    Ok(subst)
}

fn unify_with_subst(
    type1: &ParametricType,
    type2: &ParametricType,
    subst: &mut TypeSubstitution,
) -> Result<(), IrError> {
    // Apply current substitution
    let t1 = type1.substitute(subst);
    let t2 = type2.substitute(subst);

    match (&t1, &t2) {
        // Same type
        _ if t1 == t2 => Ok(()),

        // Unify variable with type
        (ParametricType::Variable(v), t) | (t, ParametricType::Variable(v)) => {
            // Occurs check: variable v should not occur in type t
            let var_type = ParametricType::Variable(v.clone());
            if var_type.occurs_in(t) && &var_type != t {
                return Err(IrError::OccursCheckFailure {
                    var: v.clone(),
                    ty: t.to_string(),
                });
            }
            subst.insert(v.clone(), t.clone());
            Ok(())
        }

        // Unify applications
        (
            ParametricType::Application {
                constructor: c1,
                arguments: args1,
            },
            ParametricType::Application {
                constructor: c2,
                arguments: args2,
            },
        ) => {
            // Constructors must match
            if c1 != c2 {
                return Err(IrError::UnificationFailure {
                    type1: t1.to_string(),
                    type2: t2.to_string(),
                });
            }

            // Argument counts must match
            if args1.len() != args2.len() {
                return Err(IrError::UnificationFailure {
                    type1: t1.to_string(),
                    type2: t2.to_string(),
                });
            }

            // Unify arguments
            for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                unify_with_subst(arg1, arg2, subst)?;
            }
            Ok(())
        }

        // Concrete types must match exactly
        _ => Err(IrError::UnificationFailure {
            type1: t1.to_string(),
            type2: t2.to_string(),
        }),
    }
}

/// Compose two substitutions: apply subst2 to all types in subst1, then merge.
pub fn compose_substitutions(
    subst1: &TypeSubstitution,
    subst2: &TypeSubstitution,
) -> TypeSubstitution {
    let mut result = HashMap::new();

    // Apply subst2 to all types in subst1
    for (var, ty) in subst1 {
        result.insert(var.clone(), ty.substitute(subst2));
    }

    // Add bindings from subst2 that aren't in subst1
    for (var, ty) in subst2 {
        result.entry(var.clone()).or_insert_with(|| ty.clone());
    }

    result
}

/// Generalize a type by replacing free type variables with fresh type variables.
pub fn generalize(ty: &ParametricType, env_vars: &[String]) -> ParametricType {
    let free_vars = ty.free_variables();
    let mut subst = HashMap::new();

    for (i, var) in free_vars.iter().enumerate() {
        if !env_vars.contains(var) {
            // Fresh variable: use Greek letters or numbers
            let fresh = format!("α{}", i);
            subst.insert(var.clone(), ParametricType::variable(fresh));
        }
    }

    ty.substitute(&subst)
}

/// Instantiate a polymorphic type with fresh type variables.
pub fn instantiate(ty: &ParametricType) -> ParametricType {
    let free_vars = ty.free_variables();
    let mut subst = HashMap::new();

    for (i, var) in free_vars.iter().enumerate() {
        // Fresh variable with timestamp to ensure uniqueness
        let fresh = format!("'t{}", i);
        subst.insert(var.clone(), ParametricType::variable(fresh));
    }

    ty.substitute(&subst)
}

// =============================================================================
// Type Constraints and Bounds
// =============================================================================

/// Type constraint representing bounds on type variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Type must implement a trait/typeclass
    Implements {
        type_var: String,
        trait_name: String,
    },
    /// Type must be a subtype of another
    Subtype {
        sub: ParametricType,
        super_type: ParametricType,
    },
    /// Type must equal another type
    Equal(ParametricType, ParametricType),
    /// Type must be numeric (Int, Float, etc.)
    Numeric(String),
    /// Type must be orderable
    Ord(String),
    /// Type must be equatable
    Eq(String),
    /// Type must be showable/displayable
    Show(String),
    /// Type must be serializable
    Serialize(String),
    /// Custom constraint
    Custom { name: String, type_var: String },
}

impl TypeConstraint {
    /// Create an "implements" constraint
    pub fn implements(type_var: impl Into<String>, trait_name: impl Into<String>) -> Self {
        TypeConstraint::Implements {
            type_var: type_var.into(),
            trait_name: trait_name.into(),
        }
    }

    /// Create a numeric constraint
    pub fn numeric(type_var: impl Into<String>) -> Self {
        TypeConstraint::Numeric(type_var.into())
    }

    /// Create an orderable constraint
    pub fn ord(type_var: impl Into<String>) -> Self {
        TypeConstraint::Ord(type_var.into())
    }

    /// Create an equality constraint
    pub fn eq_constraint(type_var: impl Into<String>) -> Self {
        TypeConstraint::Eq(type_var.into())
    }

    /// Get the type variable this constraint applies to
    pub fn type_var(&self) -> Option<&str> {
        match self {
            TypeConstraint::Implements { type_var, .. } => Some(type_var),
            TypeConstraint::Numeric(v) => Some(v),
            TypeConstraint::Ord(v) => Some(v),
            TypeConstraint::Eq(v) => Some(v),
            TypeConstraint::Show(v) => Some(v),
            TypeConstraint::Serialize(v) => Some(v),
            TypeConstraint::Custom { type_var, .. } => Some(type_var),
            TypeConstraint::Subtype { .. } | TypeConstraint::Equal(_, _) => None,
        }
    }

    /// Apply a substitution to this constraint
    pub fn substitute(&self, subst: &TypeSubstitution) -> TypeConstraint {
        match self {
            TypeConstraint::Implements {
                type_var,
                trait_name,
            } => {
                if let Some(new_type) = subst.get(type_var) {
                    if let ParametricType::Variable(new_var) = new_type {
                        TypeConstraint::Implements {
                            type_var: new_var.clone(),
                            trait_name: trait_name.clone(),
                        }
                    } else {
                        // Constraint is satisfied, keep original
                        self.clone()
                    }
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Subtype { sub, super_type } => TypeConstraint::Subtype {
                sub: sub.substitute(subst),
                super_type: super_type.substitute(subst),
            },
            TypeConstraint::Equal(t1, t2) => {
                TypeConstraint::Equal(t1.substitute(subst), t2.substitute(subst))
            }
            TypeConstraint::Numeric(v) => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(v) {
                    TypeConstraint::Numeric(new_var.clone())
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Ord(v) => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(v) {
                    TypeConstraint::Ord(new_var.clone())
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Eq(v) => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(v) {
                    TypeConstraint::Eq(new_var.clone())
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Show(v) => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(v) {
                    TypeConstraint::Show(new_var.clone())
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Serialize(v) => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(v) {
                    TypeConstraint::Serialize(new_var.clone())
                } else {
                    self.clone()
                }
            }
            TypeConstraint::Custom { name, type_var } => {
                if let Some(ParametricType::Variable(new_var)) = subst.get(type_var) {
                    TypeConstraint::Custom {
                        name: name.clone(),
                        type_var: new_var.clone(),
                    }
                } else {
                    self.clone()
                }
            }
        }
    }
}

impl fmt::Display for TypeConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeConstraint::Implements {
                type_var,
                trait_name,
            } => {
                write!(f, "{}: {}", type_var, trait_name)
            }
            TypeConstraint::Subtype { sub, super_type } => {
                write!(f, "{} <: {}", sub, super_type)
            }
            TypeConstraint::Equal(t1, t2) => write!(f, "{} = {}", t1, t2),
            TypeConstraint::Numeric(v) => write!(f, "{}: Num", v),
            TypeConstraint::Ord(v) => write!(f, "{}: Ord", v),
            TypeConstraint::Eq(v) => write!(f, "{}: Eq", v),
            TypeConstraint::Show(v) => write!(f, "{}: Show", v),
            TypeConstraint::Serialize(v) => write!(f, "{}: Serialize", v),
            TypeConstraint::Custom { name, type_var } => write!(f, "{}: {}", type_var, name),
        }
    }
}

/// A constrained type scheme: forall vars. constraints => type
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstrainedType {
    /// Universally quantified type variables
    pub type_vars: Vec<String>,
    /// Constraints on type variables
    pub constraints: Vec<TypeConstraint>,
    /// The actual type
    pub body: ParametricType,
}

impl ConstrainedType {
    /// Create a new constrained type
    pub fn new(
        type_vars: Vec<String>,
        constraints: Vec<TypeConstraint>,
        body: ParametricType,
    ) -> Self {
        Self {
            type_vars,
            constraints,
            body,
        }
    }

    /// Create a simple type without constraints
    pub fn simple(body: ParametricType) -> Self {
        Self {
            type_vars: vec![],
            constraints: vec![],
            body,
        }
    }

    /// Create a polymorphic type without constraints
    pub fn polymorphic(type_vars: Vec<String>, body: ParametricType) -> Self {
        Self {
            type_vars,
            constraints: vec![],
            body,
        }
    }

    /// Check if this type has any constraints
    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }

    /// Instantiate this type scheme with fresh variables
    pub fn instantiate_fresh(&self, counter: &mut usize) -> (ParametricType, Vec<TypeConstraint>) {
        let mut subst = HashMap::new();
        for var in &self.type_vars {
            let fresh = format!("'t{}", *counter);
            *counter += 1;
            subst.insert(var.clone(), ParametricType::variable(fresh));
        }

        let new_body = self.body.substitute(&subst);
        let new_constraints: Vec<_> = self
            .constraints
            .iter()
            .map(|c| c.substitute(&subst))
            .collect();

        (new_body, new_constraints)
    }
}

impl fmt::Display for ConstrainedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.type_vars.is_empty() {
            write!(f, "∀")?;
            for (i, var) in self.type_vars.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", var)?;
            }
            write!(f, ". ")?;
        }
        if !self.constraints.is_empty() {
            write!(f, "(")?;
            for (i, c) in self.constraints.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", c)?;
            }
            write!(f, ") => ")?;
        }
        write!(f, "{}", self.body)
    }
}

// =============================================================================
// Recursive Types
// =============================================================================

/// Recursive type definition using mu-types.
///
/// Represents types like: μα. 1 + α (natural numbers)
/// or μL. Nil | Cons(a, L) (lists)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecursiveType {
    /// The binding variable
    pub var: String,
    /// The body of the recursive type
    pub body: ParametricType,
}

impl RecursiveType {
    /// Create a new recursive type
    pub fn new(var: impl Into<String>, body: ParametricType) -> Self {
        Self {
            var: var.into(),
            body,
        }
    }

    /// Unfold the recursive type once: substitute var with the whole type
    pub fn unfold(&self) -> ParametricType {
        let rec_type = ParametricType::Recursive(Box::new(self.clone()));
        let mut subst = HashMap::new();
        subst.insert(self.var.clone(), rec_type);
        self.body.substitute(&subst)
    }

    /// Check if a type variable occurs in the body (for well-formedness)
    pub fn is_well_formed(&self) -> bool {
        self.body.free_variables().contains(&self.var)
    }
}

impl fmt::Display for RecursiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "μ{}. {}", self.var, self.body)
    }
}

// Extend ParametricType with recursive types
impl ParametricType {
    /// Create a recursive type
    pub fn recursive(var: impl Into<String>, body: ParametricType) -> Self {
        ParametricType::Recursive(Box::new(RecursiveType::new(var, body)))
    }

    /// Check if this is a recursive type
    pub fn is_recursive(&self) -> bool {
        matches!(self, ParametricType::Recursive(_))
    }

    /// Unfold a recursive type
    pub fn unfold(&self) -> Option<ParametricType> {
        match self {
            ParametricType::Recursive(rec) => Some(rec.unfold()),
            _ => None,
        }
    }
}

// =============================================================================
// Higher-Kinded Types
// =============================================================================

/// Higher-kinded type variable with its kind.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KindedVar {
    /// Variable name
    pub name: String,
    /// Kind of the variable
    pub kind: Kind,
}

impl KindedVar {
    /// Create a new kinded variable
    pub fn new(name: impl Into<String>, kind: Kind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    /// Create a type-level variable (kind *)
    pub fn type_var(name: impl Into<String>) -> Self {
        Self::new(name, Kind::Star)
    }

    /// Create a type constructor variable (kind * -> *)
    pub fn constructor_var(name: impl Into<String>) -> Self {
        Self::new(name, Kind::constructor(1))
    }

    /// Create a higher-kinded variable with custom arity
    pub fn higher_kinded(name: impl Into<String>, arity: usize) -> Self {
        Self::new(name, Kind::constructor(arity))
    }
}

impl fmt::Display for KindedVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} :: {})", self.name, self.kind)
    }
}

/// Higher-kinded type scheme supporting type constructor polymorphism.
///
/// Example: `forall F: * -> *. F Int -> F String`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HigherKindedType {
    /// Kinded type variables
    pub vars: Vec<KindedVar>,
    /// Constraints on type variables
    pub constraints: Vec<TypeConstraint>,
    /// The body type
    pub body: ParametricType,
}

impl HigherKindedType {
    /// Create a new higher-kinded type
    pub fn new(
        vars: Vec<KindedVar>,
        constraints: Vec<TypeConstraint>,
        body: ParametricType,
    ) -> Self {
        Self {
            vars,
            constraints,
            body,
        }
    }

    /// Create without constraints
    pub fn unconstrained(vars: Vec<KindedVar>, body: ParametricType) -> Self {
        Self {
            vars,
            constraints: vec![],
            body,
        }
    }

    /// Get all type-level variables (kind *)
    pub fn type_vars(&self) -> Vec<&str> {
        self.vars
            .iter()
            .filter(|v| v.kind.is_star())
            .map(|v| v.name.as_str())
            .collect()
    }

    /// Get all constructor-level variables (kind * -> * or higher)
    pub fn constructor_vars(&self) -> Vec<&str> {
        self.vars
            .iter()
            .filter(|v| !v.kind.is_star())
            .map(|v| v.name.as_str())
            .collect()
    }

    /// Check if this type has any higher-kinded variables
    pub fn has_higher_kinded_vars(&self) -> bool {
        self.vars.iter().any(|v| !v.kind.is_star())
    }
}

impl fmt::Display for HigherKindedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.vars.is_empty() {
            write!(f, "∀")?;
            for (i, var) in self.vars.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", var)?;
            }
            write!(f, ". ")?;
        }
        if !self.constraints.is_empty() {
            write!(f, "(")?;
            for (i, c) in self.constraints.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", c)?;
            }
            write!(f, ") => ")?;
        }
        write!(f, "{}", self.body)
    }
}

// =============================================================================
// Row Polymorphism
// =============================================================================

/// A row type for record/struct polymorphism.
///
/// Represents open records that can have additional fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RowType {
    /// Known fields with their types
    pub fields: Vec<(String, ParametricType)>,
    /// Optional row variable for extension
    pub rest: Option<String>,
}

impl RowType {
    /// Create a closed row (no extension)
    pub fn closed(fields: Vec<(String, ParametricType)>) -> Self {
        Self { fields, rest: None }
    }

    /// Create an open row with extension variable
    pub fn open(fields: Vec<(String, ParametricType)>, rest: impl Into<String>) -> Self {
        Self {
            fields,
            rest: Some(rest.into()),
        }
    }

    /// Create an empty row
    pub fn empty() -> Self {
        Self {
            fields: vec![],
            rest: None,
        }
    }

    /// Check if row is closed (no extension variable)
    pub fn is_closed(&self) -> bool {
        self.rest.is_none()
    }

    /// Get field type by name
    pub fn get_field(&self, name: &str) -> Option<&ParametricType> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Check if row has a field
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|(n, _)| n == name)
    }

    /// Get all field names
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Extend this row with additional fields
    pub fn extend(&self, additional: Vec<(String, ParametricType)>) -> Self {
        let mut fields = self.fields.clone();
        fields.extend(additional);
        Self {
            fields,
            rest: self.rest.clone(),
        }
    }

    /// Get free type variables including row variable
    pub fn free_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for (_, ty) in &self.fields {
            vars.extend(ty.free_variables());
        }
        if let Some(rest) = &self.rest {
            vars.push(rest.clone());
        }
        vars.sort();
        vars.dedup();
        vars
    }
}

impl fmt::Display for RowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (name, ty)) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, ty)?;
        }
        if let Some(rest) = &self.rest {
            if !self.fields.is_empty() {
                write!(f, " | ")?;
            }
            write!(f, "{}", rest)?;
        }
        write!(f, "}}")
    }
}

/// Record type using row polymorphism
impl ParametricType {
    /// Create a record type from a row
    pub fn record(fields: Vec<(String, ParametricType)>) -> Self {
        ParametricType::Row(RowType::closed(fields))
    }

    /// Create an extensible record type
    pub fn extensible_record(
        fields: Vec<(String, ParametricType)>,
        rest: impl Into<String>,
    ) -> Self {
        ParametricType::Row(RowType::open(fields, rest))
    }

    /// Check if this is a record/row type
    pub fn is_record(&self) -> bool {
        matches!(self, ParametricType::Row(_))
    }

    /// Get row type if this is a record
    pub fn as_row(&self) -> Option<&RowType> {
        match self {
            ParametricType::Row(row) => Some(row),
            _ => None,
        }
    }
}
