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
    /// Array type constructor with dimension (kind * -> *)
    Array { dimensions: usize },
    /// Function type constructor (kind * -> * -> *)
    Function,
    /// Set type constructor (kind * -> *)
    Set,
    /// Map type constructor (kind * -> * -> *)
    Map,
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
            TypeConstructor::Array { .. } => Kind::constructor(1),
            TypeConstructor::Function => Kind::constructor(2),
            TypeConstructor::Set => Kind::constructor(1),
            TypeConstructor::Map => Kind::constructor(2),
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
            TypeConstructor::Array { dimensions } => write!(f, "Array{}", dimensions),
            TypeConstructor::Function => write!(f, "->"),
            TypeConstructor::Set => write!(f, "Set"),
            TypeConstructor::Map => write!(f, "Map"),
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
    /// E.g., List<Int>, Option<String>, Tuple<Int, String>
    Application {
        constructor: TypeConstructor,
        arguments: Vec<ParametricType>,
    },
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
    pub fn tuple(types: Vec<ParametricType>) -> Self {
        if types.len() != 2 {
            // For n-ary tuples, use nested pairs
            // TODO: Could extend TypeConstructor::Tuple to support n-ary tuples
            unimplemented!("Only binary tuples currently supported")
        }
        ParametricType::apply(TypeConstructor::Tuple, types)
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
        }
    }

    /// Check if this type variable occurs in another type (occurs check for unification).
    /// This is used to detect infinite types like T = List<T>.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_creation() {
        let star = Kind::Star;
        assert_eq!(star.arity(), 0);
        assert!(star.is_star());

        let arrow1 = Kind::constructor(1);
        assert_eq!(arrow1.arity(), 1);

        let arrow2 = Kind::constructor(2);
        assert_eq!(arrow2.arity(), 2);
    }

    #[test]
    fn test_type_constructor_kind() {
        assert_eq!(TypeConstructor::List.kind().arity(), 1);
        assert_eq!(TypeConstructor::Option.kind().arity(), 1);
        assert_eq!(TypeConstructor::Tuple.kind().arity(), 2);
        assert_eq!(TypeConstructor::Function.kind().arity(), 2);
    }

    #[test]
    fn test_concrete_type() {
        let int_type = ParametricType::concrete("Int");
        assert!(int_type.is_concrete());
        assert!(!int_type.is_variable());
        assert!(int_type.free_variables().is_empty());
        assert!(int_type.is_well_kinded());
        assert_eq!(int_type.to_string(), "Int");
    }

    #[test]
    fn test_type_variable() {
        let t_var = ParametricType::variable("T");
        assert!(t_var.is_variable());
        assert!(!t_var.is_concrete());
        assert_eq!(t_var.free_variables(), vec!["T"]);
        assert!(t_var.is_well_kinded());
        assert_eq!(t_var.to_string(), "T");
    }

    #[test]
    fn test_list_type() {
        let int_type = ParametricType::concrete("Int");
        let list_int = ParametricType::list(int_type.clone());

        assert!(!list_int.is_variable());
        assert!(!list_int.is_concrete());
        assert!(list_int.free_variables().is_empty());
        assert!(list_int.is_well_kinded());
        assert_eq!(list_int.to_string(), "List<Int>");
    }

    #[test]
    fn test_option_type() {
        let string_type = ParametricType::concrete("String");
        let option_string = ParametricType::option(string_type);

        assert!(option_string.is_well_kinded());
        assert_eq!(option_string.to_string(), "Option<String>");
    }

    #[test]
    fn test_tuple_type() {
        let int_type = ParametricType::concrete("Int");
        let string_type = ParametricType::concrete("String");
        let pair = ParametricType::tuple(vec![int_type, string_type]);

        assert!(pair.is_well_kinded());
        assert_eq!(pair.to_string(), "Tuple<Int, String>");
    }

    #[test]
    fn test_nested_parametric_types() {
        let int_type = ParametricType::concrete("Int");
        let list_int = ParametricType::list(int_type.clone());
        let list_list_int = ParametricType::list(list_int);

        assert!(list_list_int.is_well_kinded());
        assert_eq!(list_list_int.to_string(), "List<List<Int>>");
    }

    #[test]
    fn test_free_variables() {
        let t = ParametricType::variable("T");
        let u = ParametricType::variable("U");
        let int_type = ParametricType::concrete("Int");

        // List<T>
        let list_t = ParametricType::list(t.clone());
        assert_eq!(list_t.free_variables(), vec!["T"]);

        // Tuple<T, U>
        let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);
        let mut free_vars = tuple_tu.free_variables();
        free_vars.sort();
        assert_eq!(free_vars, vec!["T", "U"]);

        // Tuple<T, Int>
        let tuple_t_int = ParametricType::tuple(vec![t.clone(), int_type]);
        assert_eq!(tuple_t_int.free_variables(), vec!["T"]);
    }

    #[test]
    fn test_substitution() {
        let t = ParametricType::variable("T");
        let int_type = ParametricType::concrete("Int");
        let list_t = ParametricType::list(t.clone());

        let mut subst = HashMap::new();
        subst.insert("T".to_string(), int_type.clone());

        let result = list_t.substitute(&subst);
        assert_eq!(result, ParametricType::list(int_type));
    }

    #[test]
    fn test_unify_concrete_types() {
        let int1 = ParametricType::concrete("Int");
        let int2 = ParametricType::concrete("Int");
        let string = ParametricType::concrete("String");

        // Int = Int
        let subst = unify(&int1, &int2).unwrap();
        assert!(subst.is_empty());

        // Int ≠ String
        assert!(unify(&int1, &string).is_err());
    }

    #[test]
    fn test_unify_variable_with_concrete() {
        let t = ParametricType::variable("T");
        let int_type = ParametricType::concrete("Int");

        let subst = unify(&t, &int_type).unwrap();
        assert_eq!(subst.get("T").unwrap(), &int_type);
    }

    #[test]
    fn test_unify_parametric_types() {
        let t = ParametricType::variable("T");
        let int_type = ParametricType::concrete("Int");
        let list_t = ParametricType::list(t.clone());
        let list_int = ParametricType::list(int_type.clone());

        let subst = unify(&list_t, &list_int).unwrap();
        assert_eq!(subst.get("T").unwrap(), &int_type);
    }

    #[test]
    fn test_unify_multiple_variables() {
        let t = ParametricType::variable("T");
        let u = ParametricType::variable("U");
        let int_type = ParametricType::concrete("Int");

        // Tuple<T, U> with Tuple<Int, Int>
        let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);
        let tuple_int_int = ParametricType::tuple(vec![int_type.clone(), int_type.clone()]);

        let subst = unify(&tuple_tu, &tuple_int_int).unwrap();
        assert_eq!(subst.get("T").unwrap(), &int_type);
        assert_eq!(subst.get("U").unwrap(), &int_type);
    }

    #[test]
    fn test_occurs_check() {
        let t = ParametricType::variable("T");
        let list_t = ParametricType::list(t.clone());

        // T = List<T> should fail (occurs check)
        assert!(unify(&t, &list_t).is_err());
    }

    #[test]
    fn test_unify_constructor_mismatch() {
        let int_type = ParametricType::concrete("Int");
        let list_int = ParametricType::list(int_type.clone());
        let option_int = ParametricType::option(int_type);

        // List<Int> ≠ Option<Int>
        assert!(unify(&list_int, &option_int).is_err());
    }

    #[test]
    fn test_compose_substitutions() {
        let u = ParametricType::variable("U");
        let int_type = ParametricType::concrete("Int");

        let mut subst1 = HashMap::new();
        subst1.insert("T".to_string(), u.clone());

        let mut subst2 = HashMap::new();
        subst2.insert("U".to_string(), int_type.clone());

        let composed = compose_substitutions(&subst1, &subst2);
        assert_eq!(composed.get("T").unwrap(), &int_type);
        assert_eq!(composed.get("U").unwrap(), &int_type);
    }

    #[test]
    fn test_generalize() {
        let t = ParametricType::variable("T");
        let u = ParametricType::variable("U");
        let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);

        // Generalize with empty environment
        let gen = generalize(&tuple_tu, &[]);
        let free_vars = gen.free_variables();
        assert_eq!(free_vars.len(), 2);
        assert!(free_vars.iter().all(|v| v.starts_with('α')));
    }

    #[test]
    fn test_instantiate() {
        let t = ParametricType::variable("T");
        let list_t = ParametricType::list(t);

        let inst = instantiate(&list_t);
        let free_vars = inst.free_variables();
        assert_eq!(free_vars.len(), 1);
        assert!(free_vars[0].starts_with("'t"));
    }

    #[test]
    fn test_function_type() {
        let int_type = ParametricType::concrete("Int");
        let string_type = ParametricType::concrete("String");
        let func = ParametricType::function(int_type, string_type);

        assert!(func.is_well_kinded());
        assert_eq!(func.to_string(), "-><Int, String>");
    }

    #[test]
    fn test_map_type() {
        let string_type = ParametricType::concrete("String");
        let int_type = ParametricType::concrete("Int");
        let map = ParametricType::map(string_type, int_type);

        assert!(map.is_well_kinded());
        assert_eq!(map.to_string(), "Map<String, Int>");
    }

    #[test]
    fn test_set_type() {
        let int_type = ParametricType::concrete("Int");
        let set = ParametricType::set(int_type);

        assert!(set.is_well_kinded());
        assert_eq!(set.to_string(), "Set<Int>");
    }

    #[test]
    fn test_array_type() {
        let float_type = ParametricType::concrete("Float");
        let array2d = ParametricType::array(float_type, 2);

        assert!(array2d.is_well_kinded());
        assert_eq!(array2d.to_string(), "Array2<Float>");
    }

    #[test]
    fn test_custom_type_constructor() {
        let int_type = ParametricType::concrete("Int");
        let custom = TypeConstructor::custom("MyType", 1);
        let my_int = ParametricType::apply(custom, vec![int_type]);

        assert!(my_int.is_well_kinded());
        assert_eq!(my_int.to_string(), "MyType<Int>");
    }

    #[test]
    fn test_ill_kinded_type() {
        let int_type = ParametricType::concrete("Int");
        // List expects 1 argument, giving it 2
        let ill_kinded =
            ParametricType::apply(TypeConstructor::List, vec![int_type.clone(), int_type]);

        assert!(!ill_kinded.is_well_kinded());
    }
}
