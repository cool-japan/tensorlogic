//! Effect system for tracking computational effects.
//!
//! The effect system allows tracking what side effects a computation may have,
//! enabling reasoning about purity, IO, state mutation, non-determinism, and more.
//! This is useful for optimization (pure functions can be cached/memoized) and
//! for ensuring safety properties.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{Effect, EffectSet, EffectContext, EffectHandler};
//!
//! // Create effect sets
//! let pure = EffectSet::pure();
//! let io = EffectSet::new().with(Effect::IO);
//! let stateful = EffectSet::new().with(Effect::State);
//!
//! // Combine effects
//! let combined = io.union(&stateful);
//! assert!(combined.has(Effect::IO));
//! assert!(combined.has(Effect::State));
//!
//! // Check purity
//! assert!(pure.is_pure());
//! assert!(!io.is_pure());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A computational effect that can be tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Input/output operations
    IO,
    /// State mutation
    State,
    /// Non-determinism (random number generation, etc.)
    NonDet,
    /// Exceptions/errors
    Exception,
    /// Divergence (non-termination)
    Diverge,
    /// Memory allocation
    Alloc,
    /// Network communication
    Network,
    /// File system access
    FileSystem,
    /// Logging/tracing
    Log,
    /// Time-dependent operations
    Time,
    /// GPU operations
    GPU,
    /// Concurrency/parallelism
    Concurrent,
    /// Environment variable access
    Env,
    /// System calls
    System,
}

impl Effect {
    /// Get the effect name.
    pub fn name(&self) -> &'static str {
        match self {
            Effect::IO => "IO",
            Effect::State => "State",
            Effect::NonDet => "NonDet",
            Effect::Exception => "Exception",
            Effect::Diverge => "Diverge",
            Effect::Alloc => "Alloc",
            Effect::Network => "Network",
            Effect::FileSystem => "FileSystem",
            Effect::Log => "Log",
            Effect::Time => "Time",
            Effect::GPU => "GPU",
            Effect::Concurrent => "Concurrent",
            Effect::Env => "Env",
            Effect::System => "System",
        }
    }

    /// Get a description of the effect.
    pub fn description(&self) -> &'static str {
        match self {
            Effect::IO => "Input/output operations",
            Effect::State => "State mutation",
            Effect::NonDet => "Non-deterministic computation",
            Effect::Exception => "May raise exceptions",
            Effect::Diverge => "May not terminate",
            Effect::Alloc => "Memory allocation",
            Effect::Network => "Network communication",
            Effect::FileSystem => "File system access",
            Effect::Log => "Logging/tracing",
            Effect::Time => "Time-dependent operations",
            Effect::GPU => "GPU computation",
            Effect::Concurrent => "Concurrent/parallel execution",
            Effect::Env => "Environment variable access",
            Effect::System => "System calls",
        }
    }

    /// Check if this effect implies another.
    ///
    /// For example, FileSystem implies IO.
    pub fn implies(&self, other: &Effect) -> bool {
        match (self, other) {
            (Effect::FileSystem, Effect::IO) => true,
            (Effect::Network, Effect::IO) => true,
            (Effect::GPU, Effect::Alloc) => true,
            _ => self == other,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A set of computational effects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectSet {
    /// The effects in this set
    effects: HashSet<Effect>,
}

impl EffectSet {
    /// Create an empty (pure) effect set.
    pub fn new() -> Self {
        EffectSet {
            effects: HashSet::new(),
        }
    }

    /// Create a pure effect set (alias for new).
    pub fn pure() -> Self {
        Self::new()
    }

    /// Create an effect set with all effects.
    pub fn all() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::IO);
        effects.insert(Effect::State);
        effects.insert(Effect::NonDet);
        effects.insert(Effect::Exception);
        effects.insert(Effect::Diverge);
        effects.insert(Effect::Alloc);
        effects.insert(Effect::Network);
        effects.insert(Effect::FileSystem);
        effects.insert(Effect::Log);
        effects.insert(Effect::Time);
        effects.insert(Effect::GPU);
        effects.insert(Effect::Concurrent);
        effects.insert(Effect::Env);
        effects.insert(Effect::System);
        EffectSet { effects }
    }

    /// Create an effect set from a single effect.
    pub fn singleton(effect: Effect) -> Self {
        let mut effects = HashSet::new();
        effects.insert(effect);
        EffectSet { effects }
    }

    /// Add an effect to the set.
    pub fn with(mut self, effect: Effect) -> Self {
        self.effects.insert(effect);
        self
    }

    /// Add an effect to the set (mutable).
    pub fn insert(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    /// Remove an effect from the set.
    pub fn remove(&mut self, effect: &Effect) {
        self.effects.remove(effect);
    }

    /// Check if the set contains an effect.
    pub fn has(&self, effect: Effect) -> bool {
        self.effects.contains(&effect)
    }

    /// Check if the set is pure (no effects).
    pub fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }

    /// Check if the computation is total (no divergence or exceptions).
    pub fn is_total(&self) -> bool {
        !self.has(Effect::Diverge) && !self.has(Effect::Exception)
    }

    /// Check if the computation is deterministic.
    pub fn is_deterministic(&self) -> bool {
        !self.has(Effect::NonDet)
    }

    /// Get the union of two effect sets.
    pub fn union(&self, other: &EffectSet) -> EffectSet {
        let effects: HashSet<_> = self.effects.union(&other.effects).cloned().collect();
        EffectSet { effects }
    }

    /// Get the intersection of two effect sets.
    pub fn intersection(&self, other: &EffectSet) -> EffectSet {
        let effects: HashSet<_> = self.effects.intersection(&other.effects).cloned().collect();
        EffectSet { effects }
    }

    /// Get the difference of two effect sets (effects in self but not in other).
    pub fn difference(&self, other: &EffectSet) -> EffectSet {
        let effects: HashSet<_> = self.effects.difference(&other.effects).cloned().collect();
        EffectSet { effects }
    }

    /// Check if this set is a subset of another.
    pub fn is_subset_of(&self, other: &EffectSet) -> bool {
        self.effects.is_subset(&other.effects)
    }

    /// Get the number of effects.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Iterate over the effects.
    pub fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }

    /// Get the effects as a vector.
    pub fn to_vec(&self) -> Vec<Effect> {
        self.effects.iter().cloned().collect()
    }

    /// Expand implied effects.
    ///
    /// For example, if FileSystem is present, IO is also added.
    pub fn expand_implications(&mut self) {
        let current: Vec<_> = self.effects.iter().cloned().collect();
        for effect in current {
            match effect {
                Effect::FileSystem | Effect::Network => {
                    self.effects.insert(Effect::IO);
                }
                Effect::GPU => {
                    self.effects.insert(Effect::Alloc);
                }
                _ => {}
            }
        }
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Pure")
        } else {
            let names: Vec<_> = self.effects.iter().map(|e| e.name()).collect();
            write!(f, "{{{}}}", names.join(", "))
        }
    }
}

/// An effect row for row polymorphism.
///
/// This allows functions to be polymorphic in their effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectRow {
    /// A closed row with specific effects
    Closed(EffectSet),
    /// An open row with effects and a tail variable
    Open { effects: EffectSet, tail: String },
}

impl EffectRow {
    /// Create a closed row from an effect set.
    pub fn closed(effects: EffectSet) -> Self {
        EffectRow::Closed(effects)
    }

    /// Create an open row with a tail variable.
    pub fn open(effects: EffectSet, tail: impl Into<String>) -> Self {
        EffectRow::Open {
            effects,
            tail: tail.into(),
        }
    }

    /// Create a pure closed row.
    pub fn pure() -> Self {
        EffectRow::Closed(EffectSet::pure())
    }

    /// Check if this row contains an effect.
    pub fn has(&self, effect: Effect) -> bool {
        match self {
            EffectRow::Closed(effects) => effects.has(effect),
            EffectRow::Open { effects, .. } => effects.has(effect),
        }
    }

    /// Get the free tail variables.
    pub fn free_variables(&self) -> Vec<String> {
        match self {
            EffectRow::Closed(_) => vec![],
            EffectRow::Open { tail, .. } => vec![tail.clone()],
        }
    }

    /// Substitute a tail variable with an effect row.
    pub fn substitute(&self, var: &str, row: &EffectRow) -> EffectRow {
        match self {
            EffectRow::Closed(effects) => EffectRow::Closed(effects.clone()),
            EffectRow::Open { effects, tail } => {
                if tail == var {
                    match row {
                        EffectRow::Closed(other_effects) => {
                            EffectRow::Closed(effects.union(other_effects))
                        }
                        EffectRow::Open {
                            effects: other_effects,
                            tail: other_tail,
                        } => EffectRow::Open {
                            effects: effects.union(other_effects),
                            tail: other_tail.clone(),
                        },
                    }
                } else {
                    EffectRow::Open {
                        effects: effects.clone(),
                        tail: tail.clone(),
                    }
                }
            }
        }
    }
}

impl fmt::Display for EffectRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectRow::Closed(effects) => write!(f, "{}", effects),
            EffectRow::Open { effects, tail } => {
                if effects.is_empty() {
                    write!(f, "{}", tail)
                } else {
                    let names: Vec<_> = effects.iter().map(|e| e.name()).collect();
                    write!(f, "{{{} | {}}}", names.join(", "), tail)
                }
            }
        }
    }
}

/// An effect handler that can handle specific effects.
#[derive(Debug, Clone)]
pub struct EffectHandler {
    /// Name of the handler
    pub name: String,
    /// Effects that this handler can handle
    pub handles: EffectSet,
    /// Description
    pub description: Option<String>,
}

impl EffectHandler {
    /// Create a new effect handler.
    pub fn new(name: impl Into<String>) -> Self {
        EffectHandler {
            name: name.into(),
            handles: EffectSet::new(),
            description: None,
        }
    }

    /// Add an effect that this handler handles.
    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.handles.insert(effect);
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Check if this handler handles a specific effect.
    pub fn handles(&self, effect: Effect) -> bool {
        self.handles.has(effect)
    }

    /// Get the residual effects after handling.
    pub fn residual(&self, effects: &EffectSet) -> EffectSet {
        effects.difference(&self.handles)
    }
}

/// Context for tracking effects.
#[derive(Debug, Clone, Default)]
pub struct EffectContext {
    /// Effect annotations for functions/expressions
    annotations: HashMap<String, EffectSet>,
    /// Installed effect handlers
    handlers: Vec<EffectHandler>,
    /// Effect variables for polymorphism
    variables: HashMap<String, EffectSet>,
}

impl EffectContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        EffectContext {
            annotations: HashMap::new(),
            handlers: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// Annotate an item with effects.
    pub fn annotate(&mut self, name: impl Into<String>, effects: EffectSet) {
        self.annotations.insert(name.into(), effects);
    }

    /// Get the effects for an item.
    pub fn get_effects(&self, name: &str) -> Option<&EffectSet> {
        self.annotations.get(name)
    }

    /// Install an effect handler.
    pub fn install_handler(&mut self, handler: EffectHandler) {
        self.handlers.push(handler);
    }

    /// Set an effect variable.
    pub fn set_variable(&mut self, name: impl Into<String>, effects: EffectSet) {
        self.variables.insert(name.into(), effects);
    }

    /// Get an effect variable.
    pub fn get_variable(&self, name: &str) -> Option<&EffectSet> {
        self.variables.get(name)
    }

    /// Compute the handled effects for a given effect set.
    pub fn compute_residual(&self, effects: &EffectSet) -> EffectSet {
        let mut residual = effects.clone();
        for handler in &self.handlers {
            residual = handler.residual(&residual);
        }
        residual
    }

    /// Check if all effects are handled.
    pub fn all_handled(&self, effects: &EffectSet) -> bool {
        self.compute_residual(effects).is_empty()
    }

    /// Get unhandled effects.
    pub fn unhandled(&self, effects: &EffectSet) -> EffectSet {
        self.compute_residual(effects)
    }
}

/// Registry for effect-annotated functions.
#[derive(Debug, Clone, Default)]
pub struct EffectRegistry {
    /// Function names to their effect signatures
    functions: HashMap<String, EffectSignature>,
}

/// An effect signature for a function.
#[derive(Debug, Clone)]
pub struct EffectSignature {
    /// Name of the function
    pub name: String,
    /// Effect row for the function
    pub effects: EffectRow,
    /// Description
    pub description: Option<String>,
}

impl EffectRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        EffectRegistry {
            functions: HashMap::new(),
        }
    }

    /// Create a registry with common function signatures.
    pub fn with_builtins() -> Self {
        let mut registry = EffectRegistry::new();

        // Pure math functions
        registry.register(EffectSignature {
            name: "sin".to_string(),
            effects: EffectRow::pure(),
            description: Some("Sine function".to_string()),
        });

        registry.register(EffectSignature {
            name: "cos".to_string(),
            effects: EffectRow::pure(),
            description: Some("Cosine function".to_string()),
        });

        registry.register(EffectSignature {
            name: "exp".to_string(),
            effects: EffectRow::pure(),
            description: Some("Exponential function".to_string()),
        });

        // IO functions
        registry.register(EffectSignature {
            name: "print".to_string(),
            effects: EffectRow::closed(EffectSet::singleton(Effect::IO)),
            description: Some("Print to stdout".to_string()),
        });

        registry.register(EffectSignature {
            name: "read_file".to_string(),
            effects: EffectRow::closed(
                EffectSet::new()
                    .with(Effect::IO)
                    .with(Effect::FileSystem)
                    .with(Effect::Exception),
            ),
            description: Some("Read file contents".to_string()),
        });

        // Random functions
        registry.register(EffectSignature {
            name: "random".to_string(),
            effects: EffectRow::closed(EffectSet::singleton(Effect::NonDet)),
            description: Some("Generate random number".to_string()),
        });

        // GPU functions
        registry.register(EffectSignature {
            name: "gpu_matmul".to_string(),
            effects: EffectRow::closed(EffectSet::new().with(Effect::GPU).with(Effect::Alloc)),
            description: Some("GPU matrix multiplication".to_string()),
        });

        registry
    }

    /// Register a function signature.
    pub fn register(&mut self, signature: EffectSignature) {
        self.functions.insert(signature.name.clone(), signature);
    }

    /// Get a function signature.
    pub fn get(&self, name: &str) -> Option<&EffectSignature> {
        self.functions.get(name)
    }

    /// Check if a function is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get all function names.
    pub fn function_names(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered functions.
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Check if a function is pure.
    pub fn is_pure(&self, name: &str) -> Option<bool> {
        self.functions.get(name).map(|sig| match &sig.effects {
            EffectRow::Closed(effects) => effects.is_pure(),
            EffectRow::Open { effects, .. } => effects.is_pure(),
        })
    }
}

/// Infer effects for a sequence of operations.
pub fn infer_effects(registry: &EffectRegistry, operations: &[&str]) -> EffectSet {
    let mut effects = EffectSet::new();
    for op in operations {
        if let Some(sig) = registry.get(op) {
            match &sig.effects {
                EffectRow::Closed(op_effects) => {
                    effects = effects.union(op_effects);
                }
                EffectRow::Open {
                    effects: op_effects,
                    ..
                } => {
                    effects = effects.union(op_effects);
                }
            }
        }
    }
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_set_operations() {
        let io = EffectSet::singleton(Effect::IO);
        let state = EffectSet::singleton(Effect::State);

        let combined = io.union(&state);
        assert!(combined.has(Effect::IO));
        assert!(combined.has(Effect::State));
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_effect_set_pure() {
        let pure = EffectSet::pure();
        assert!(pure.is_pure());
        assert!(pure.is_total());
        assert!(pure.is_deterministic());
    }

    #[test]
    fn test_effect_set_subset() {
        let io = EffectSet::singleton(Effect::IO);
        let combined = EffectSet::new().with(Effect::IO).with(Effect::State);

        assert!(io.is_subset_of(&combined));
        assert!(!combined.is_subset_of(&io));
    }

    #[test]
    fn test_effect_set_difference() {
        let combined = EffectSet::new().with(Effect::IO).with(Effect::State);
        let io = EffectSet::singleton(Effect::IO);

        let diff = combined.difference(&io);
        assert!(!diff.has(Effect::IO));
        assert!(diff.has(Effect::State));
    }

    #[test]
    fn test_effect_row_closed() {
        let row = EffectRow::closed(EffectSet::singleton(Effect::IO));
        assert!(row.has(Effect::IO));
        assert!(!row.has(Effect::State));
    }

    #[test]
    fn test_effect_row_open() {
        let row = EffectRow::open(EffectSet::singleton(Effect::IO), "e");
        assert!(row.has(Effect::IO));
        assert_eq!(row.free_variables(), vec!["e".to_string()]);
    }

    #[test]
    fn test_effect_row_substitute() {
        let row = EffectRow::open(EffectSet::singleton(Effect::IO), "e");
        let tail = EffectRow::closed(EffectSet::singleton(Effect::State));

        let result = row.substitute("e", &tail);
        match result {
            EffectRow::Closed(effects) => {
                assert!(effects.has(Effect::IO));
                assert!(effects.has(Effect::State));
            }
            _ => panic!("Expected closed row"),
        }
    }

    #[test]
    fn test_effect_handler() {
        let handler = EffectHandler::new("io_handler")
            .with_effect(Effect::IO)
            .with_effect(Effect::FileSystem);

        let effects = EffectSet::new()
            .with(Effect::IO)
            .with(Effect::State)
            .with(Effect::FileSystem);

        let residual = handler.residual(&effects);
        assert!(!residual.has(Effect::IO));
        assert!(!residual.has(Effect::FileSystem));
        assert!(residual.has(Effect::State));
    }

    #[test]
    fn test_effect_context() {
        let mut ctx = EffectContext::new();

        ctx.annotate("foo", EffectSet::singleton(Effect::IO));
        ctx.annotate("bar", EffectSet::pure());

        assert!(ctx.get_effects("foo").unwrap().has(Effect::IO));
        assert!(ctx.get_effects("bar").unwrap().is_pure());
    }

    #[test]
    fn test_effect_context_handlers() {
        let mut ctx = EffectContext::new();

        ctx.install_handler(EffectHandler::new("io").with_effect(Effect::IO));

        let effects = EffectSet::new().with(Effect::IO).with(Effect::State);
        let residual = ctx.compute_residual(&effects);

        assert!(!residual.has(Effect::IO));
        assert!(residual.has(Effect::State));
    }

    #[test]
    fn test_effect_registry_builtins() {
        let registry = EffectRegistry::with_builtins();

        assert!(registry.is_pure("sin").unwrap());
        assert!(!registry.is_pure("print").unwrap());
        assert!(!registry.is_pure("random").unwrap());
    }

    #[test]
    fn test_infer_effects() {
        let registry = EffectRegistry::with_builtins();

        let effects = infer_effects(&registry, &["sin", "cos"]);
        assert!(effects.is_pure());

        let effects = infer_effects(&registry, &["sin", "print"]);
        assert!(effects.has(Effect::IO));
    }

    #[test]
    fn test_effect_implies() {
        assert!(Effect::FileSystem.implies(&Effect::IO));
        assert!(Effect::Network.implies(&Effect::IO));
        assert!(!Effect::IO.implies(&Effect::FileSystem));
    }

    #[test]
    fn test_expand_implications() {
        let mut effects = EffectSet::new().with(Effect::FileSystem);
        effects.expand_implications();

        assert!(effects.has(Effect::IO));
        assert!(effects.has(Effect::FileSystem));
    }

    #[test]
    fn test_effect_display() {
        let pure = EffectSet::pure();
        assert_eq!(format!("{}", pure), "Pure");

        let row = EffectRow::open(EffectSet::singleton(Effect::IO), "e");
        assert!(format!("{}", row).contains("IO"));
        assert!(format!("{}", row).contains("e"));
    }

    #[test]
    fn test_is_total() {
        let effects = EffectSet::new().with(Effect::IO).with(Effect::State);
        assert!(effects.is_total());

        let effects = EffectSet::new().with(Effect::Exception);
        assert!(!effects.is_total());
    }

    #[test]
    fn test_all_effects() {
        let all = EffectSet::all();
        assert!(all.has(Effect::IO));
        assert!(all.has(Effect::State));
        assert!(all.has(Effect::GPU));
        assert!(all.has(Effect::System));
    }

    #[test]
    fn test_effect_signature() {
        let sig = EffectSignature {
            name: "my_func".to_string(),
            effects: EffectRow::closed(EffectSet::singleton(Effect::IO)),
            description: Some("My function".to_string()),
        };

        assert_eq!(sig.name, "my_func");
        assert!(sig.effects.has(Effect::IO));
    }
}
