//! Linear types for resource tracking and single-use guarantees.
//!
//! Linear types ensure that resources are used exactly once, preventing issues like
//! double-free, use-after-free, and resource leaks. This is particularly useful for
//! managing GPU memory, file handles, and other exclusive resources in tensor computations.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{LinearType, LinearKind, LinearContext, Ownership};
//!
//! // Create a linear tensor type (must be used exactly once)
//! let linear_tensor = LinearType::new("Tensor")
//!     .with_kind(LinearKind::Linear)
//!     .with_name("LinearTensor");
//!
//! // Create an affine type (can be used at most once)
//! let affine_tensor = LinearType::new("Tensor")
//!     .with_kind(LinearKind::Affine);
//!
//! // Create a relevant type (must be used at least once)
//! let relevant_tensor = LinearType::new("Tensor")
//!     .with_kind(LinearKind::Relevant);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

/// The kind of linearity constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearKind {
    /// Unrestricted: can be used any number of times (standard types)
    Unrestricted,
    /// Linear: must be used exactly once
    Linear,
    /// Affine: can be used at most once (can be dropped)
    Affine,
    /// Relevant: must be used at least once (can be copied)
    Relevant,
}

impl LinearKind {
    /// Check if this kind allows copying.
    pub fn allows_copy(&self) -> bool {
        matches!(self, LinearKind::Unrestricted | LinearKind::Relevant)
    }

    /// Check if this kind allows dropping without use.
    pub fn allows_drop(&self) -> bool {
        matches!(self, LinearKind::Unrestricted | LinearKind::Affine)
    }

    /// Check if this kind requires at least one use.
    pub fn requires_use(&self) -> bool {
        matches!(self, LinearKind::Linear | LinearKind::Relevant)
    }

    /// Check if this kind limits to at most one use.
    pub fn limits_use(&self) -> bool {
        matches!(self, LinearKind::Linear | LinearKind::Affine)
    }

    /// Get the join (least upper bound) of two kinds.
    pub fn join(&self, other: &LinearKind) -> LinearKind {
        match (*self, *other) {
            (LinearKind::Unrestricted, _) | (_, LinearKind::Unrestricted) => {
                LinearKind::Unrestricted
            }
            (LinearKind::Linear, LinearKind::Linear) => LinearKind::Linear,
            (LinearKind::Affine, LinearKind::Affine) => LinearKind::Affine,
            (LinearKind::Relevant, LinearKind::Relevant) => LinearKind::Relevant,
            (LinearKind::Linear, LinearKind::Affine) | (LinearKind::Affine, LinearKind::Linear) => {
                LinearKind::Affine
            }
            (LinearKind::Linear, LinearKind::Relevant)
            | (LinearKind::Relevant, LinearKind::Linear) => LinearKind::Relevant,
            (LinearKind::Affine, LinearKind::Relevant)
            | (LinearKind::Relevant, LinearKind::Affine) => LinearKind::Unrestricted,
        }
    }

    /// Get the meet (greatest lower bound) of two kinds.
    pub fn meet(&self, other: &LinearKind) -> LinearKind {
        match (*self, *other) {
            (LinearKind::Linear, _) | (_, LinearKind::Linear) => LinearKind::Linear,
            (LinearKind::Affine, LinearKind::Affine) => LinearKind::Affine,
            (LinearKind::Relevant, LinearKind::Relevant) => LinearKind::Relevant,
            (LinearKind::Affine, LinearKind::Relevant)
            | (LinearKind::Relevant, LinearKind::Affine) => LinearKind::Linear,
            (LinearKind::Unrestricted, other) | (other, LinearKind::Unrestricted) => other,
        }
    }
}

impl fmt::Display for LinearKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearKind::Unrestricted => write!(f, "unrestricted"),
            LinearKind::Linear => write!(f, "linear"),
            LinearKind::Affine => write!(f, "affine"),
            LinearKind::Relevant => write!(f, "relevant"),
        }
    }
}

/// Ownership state of a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// Resource is owned and can be used
    Owned,
    /// Resource has been moved/consumed
    Moved,
    /// Resource has been borrowed (still owned but in use)
    Borrowed,
    /// Resource has been dropped
    Dropped,
}

impl fmt::Display for Ownership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ownership::Owned => write!(f, "owned"),
            Ownership::Moved => write!(f, "moved"),
            Ownership::Borrowed => write!(f, "borrowed"),
            Ownership::Dropped => write!(f, "dropped"),
        }
    }
}

/// A linear type with usage constraints.
#[derive(Debug, Clone)]
pub struct LinearType {
    /// Base type name
    pub base_type: String,
    /// Linearity kind
    pub kind: LinearKind,
    /// Optional type name
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Resource tags (for grouping related resources)
    pub tags: Vec<String>,
}

impl LinearType {
    /// Create a new linear type with default unrestricted kind.
    pub fn new(base_type: impl Into<String>) -> Self {
        LinearType {
            base_type: base_type.into(),
            kind: LinearKind::Unrestricted,
            name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Create a linear type that must be used exactly once.
    pub fn linear(base_type: impl Into<String>) -> Self {
        LinearType {
            base_type: base_type.into(),
            kind: LinearKind::Linear,
            name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Create an affine type that can be used at most once.
    pub fn affine(base_type: impl Into<String>) -> Self {
        LinearType {
            base_type: base_type.into(),
            kind: LinearKind::Affine,
            name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Create a relevant type that must be used at least once.
    pub fn relevant(base_type: impl Into<String>) -> Self {
        LinearType {
            base_type: base_type.into(),
            kind: LinearKind::Relevant,
            name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Set the linearity kind.
    pub fn with_kind(mut self, kind: LinearKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the type name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a resource tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Get the effective type name.
    pub fn type_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.base_type)
    }

    /// Check if this type allows copying.
    pub fn allows_copy(&self) -> bool {
        self.kind.allows_copy()
    }

    /// Check if this type allows dropping.
    pub fn allows_drop(&self) -> bool {
        self.kind.allows_drop()
    }
}

/// A resource tracked by the linear type system.
#[derive(Debug, Clone)]
pub struct Resource {
    /// Resource name/identifier
    pub name: String,
    /// Resource type
    pub ty: LinearType,
    /// Current ownership state
    pub ownership: Ownership,
    /// Number of times the resource has been used
    pub use_count: usize,
    /// Location where the resource was created
    pub created_at: Option<String>,
    /// Location where the resource was last used
    pub last_used_at: Option<String>,
}

impl Resource {
    /// Create a new owned resource.
    pub fn new(name: impl Into<String>, ty: LinearType) -> Self {
        Resource {
            name: name.into(),
            ty,
            ownership: Ownership::Owned,
            use_count: 0,
            created_at: None,
            last_used_at: None,
        }
    }

    /// Set the creation location.
    pub fn with_created_at(mut self, location: impl Into<String>) -> Self {
        self.created_at = Some(location.into());
        self
    }

    /// Check if the resource can be used.
    pub fn can_use(&self) -> bool {
        matches!(self.ownership, Ownership::Owned)
            || (matches!(self.ownership, Ownership::Borrowed) && self.ty.kind.allows_copy())
    }

    /// Check if the resource can be moved.
    pub fn can_move(&self) -> bool {
        matches!(self.ownership, Ownership::Owned)
    }

    /// Check if the resource can be dropped.
    pub fn can_drop(&self) -> bool {
        self.ty.allows_drop() || self.use_count > 0
    }

    /// Use the resource, returning an error if not allowed.
    pub fn use_resource(&mut self, location: impl Into<String>) -> Result<(), LinearError> {
        if !self.can_use() {
            return Err(LinearError::UseAfterMove {
                resource: self.name.clone(),
                state: self.ownership,
            });
        }

        if self.ty.kind.limits_use() && self.use_count > 0 {
            return Err(LinearError::MultipleUse {
                resource: self.name.clone(),
                count: self.use_count + 1,
            });
        }

        self.use_count += 1;
        self.last_used_at = Some(location.into());
        Ok(())
    }

    /// Move the resource to a new owner.
    pub fn move_to(&mut self, location: impl Into<String>) -> Result<(), LinearError> {
        if !self.can_move() {
            return Err(LinearError::UseAfterMove {
                resource: self.name.clone(),
                state: self.ownership,
            });
        }

        self.ownership = Ownership::Moved;
        self.last_used_at = Some(location.into());
        Ok(())
    }

    /// Drop the resource.
    pub fn drop_resource(&mut self) -> Result<(), LinearError> {
        if self.ty.kind.requires_use() && self.use_count == 0 {
            return Err(LinearError::UnusedResource {
                resource: self.name.clone(),
                kind: self.ty.kind,
            });
        }

        self.ownership = Ownership::Dropped;
        Ok(())
    }

    /// Validate the resource state at the end of its scope.
    pub fn validate_end_of_scope(&self) -> Result<(), LinearError> {
        match self.ownership {
            Ownership::Owned => {
                if self.ty.kind.requires_use() && self.use_count == 0 {
                    Err(LinearError::UnusedResource {
                        resource: self.name.clone(),
                        kind: self.ty.kind,
                    })
                } else {
                    Ok(())
                }
            }
            Ownership::Borrowed => Err(LinearError::BorrowedAtEndOfScope {
                resource: self.name.clone(),
            }),
            _ => Ok(()),
        }
    }
}

/// Error types for linear type violations.
#[derive(Debug, Clone)]
pub enum LinearError {
    /// Attempted to use a resource after it was moved
    UseAfterMove { resource: String, state: Ownership },
    /// Attempted to use a linear resource multiple times
    MultipleUse { resource: String, count: usize },
    /// A linear/relevant resource was never used
    UnusedResource { resource: String, kind: LinearKind },
    /// Resource was still borrowed at end of scope
    BorrowedAtEndOfScope { resource: String },
    /// Unknown resource
    UnknownResource { resource: String },
    /// Resource already exists
    DuplicateResource { resource: String },
}

impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearError::UseAfterMove { resource, state } => {
                write!(f, "Cannot use resource '{}': it is {}", resource, state)
            }
            LinearError::MultipleUse { resource, count } => {
                write!(
                    f,
                    "Resource '{}' used {} times but must be used exactly once",
                    resource, count
                )
            }
            LinearError::UnusedResource { resource, kind } => {
                write!(f, "{} resource '{}' was never used", kind, resource)
            }
            LinearError::BorrowedAtEndOfScope { resource } => {
                write!(
                    f,
                    "Resource '{}' is still borrowed at end of scope",
                    resource
                )
            }
            LinearError::UnknownResource { resource } => {
                write!(f, "Unknown resource '{}'", resource)
            }
            LinearError::DuplicateResource { resource } => {
                write!(f, "Resource '{}' already exists", resource)
            }
        }
    }
}

impl std::error::Error for LinearError {}

/// Context for tracking linear resources.
#[derive(Debug, Clone, Default)]
pub struct LinearContext {
    /// Tracked resources
    resources: HashMap<String, Resource>,
    /// Resource aliases (for tracking moves)
    aliases: HashMap<String, String>,
    /// Scope stack for nested scopes
    scope_stack: Vec<HashSet<String>>,
}

impl LinearContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        LinearContext {
            resources: HashMap::new(),
            aliases: HashMap::new(),
            scope_stack: vec![HashSet::new()],
        }
    }

    /// Enter a new scope.
    pub fn enter_scope(&mut self) {
        self.scope_stack.push(HashSet::new());
    }

    /// Exit the current scope, validating all resources.
    pub fn exit_scope(&mut self) -> Result<(), Vec<LinearError>> {
        let scope = match self.scope_stack.pop() {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut errors = Vec::new();

        for resource_name in scope {
            if let Some(resource) = self.resources.get(&resource_name) {
                if let Err(e) = resource.validate_end_of_scope() {
                    errors.push(e);
                }
            }
        }

        // Remove resources from this scope
        for resource_name in self.scope_stack.last().cloned().unwrap_or_default() {
            self.resources.remove(&resource_name);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Add a new resource to the current scope.
    pub fn add_resource(&mut self, resource: Resource) -> Result<(), LinearError> {
        if self.resources.contains_key(&resource.name) {
            return Err(LinearError::DuplicateResource {
                resource: resource.name,
            });
        }

        let name = resource.name.clone();
        self.resources.insert(name.clone(), resource);

        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name);
        }

        Ok(())
    }

    /// Create and add a new resource.
    pub fn create_resource(
        &mut self,
        name: impl Into<String>,
        ty: LinearType,
        location: impl Into<String>,
    ) -> Result<(), LinearError> {
        let resource = Resource::new(name, ty).with_created_at(location);
        self.add_resource(resource)
    }

    /// Get a resource by name.
    pub fn get_resource(&self, name: &str) -> Option<&Resource> {
        self.resolve_alias(name).and_then(|n| self.resources.get(n))
    }

    /// Get a mutable reference to a resource.
    pub fn get_resource_mut(&mut self, name: &str) -> Option<&mut Resource> {
        let resolved = self.resolve_alias(name).map(|s| s.to_string());
        resolved.and_then(move |n| self.resources.get_mut(&n))
    }

    /// Resolve an alias to the actual resource name.
    fn resolve_alias<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        if self.resources.contains_key(name) {
            return Some(name);
        }
        self.aliases.get(name).map(|s| s.as_str())
    }

    /// Use a resource.
    pub fn use_resource(
        &mut self,
        name: &str,
        location: impl Into<String>,
    ) -> Result<(), LinearError> {
        let resource = self
            .get_resource_mut(name)
            .ok_or_else(|| LinearError::UnknownResource {
                resource: name.to_string(),
            })?;
        resource.use_resource(location)
    }

    /// Move a resource to a new name.
    pub fn move_resource(
        &mut self,
        from: &str,
        to: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<(), LinearError> {
        let to = to.into();
        let location = location.into();

        // Get the source resource
        let from_resolved = self
            .resolve_alias(from)
            .ok_or_else(|| LinearError::UnknownResource {
                resource: from.to_string(),
            })?
            .to_string();

        // Mark the source as moved
        let resource =
            self.resources
                .get_mut(&from_resolved)
                .ok_or_else(|| LinearError::UnknownResource {
                    resource: from.to_string(),
                })?;
        resource.move_to(&location)?;

        // Create an alias from the new name to the original
        self.aliases.insert(to.clone(), from_resolved.clone());

        // Add to current scope
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(to);
        }

        Ok(())
    }

    /// Drop a resource.
    pub fn drop_resource(&mut self, name: &str) -> Result<(), LinearError> {
        let resource = self
            .get_resource_mut(name)
            .ok_or_else(|| LinearError::UnknownResource {
                resource: name.to_string(),
            })?;
        resource.drop_resource()
    }

    /// Validate all resources at the end.
    pub fn validate_all(&self) -> Result<(), Vec<LinearError>> {
        let mut errors = Vec::new();

        for resource in self.resources.values() {
            if let Err(e) = resource.validate_end_of_scope() {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all resource names.
    pub fn resource_names(&self) -> Vec<&str> {
        self.resources.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of tracked resources.
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if the context is empty.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Get usage statistics.
    pub fn statistics(&self) -> LinearStatistics {
        let mut total = 0;
        let mut used = 0;
        let mut unused = 0;
        let mut moved = 0;

        for resource in self.resources.values() {
            total += 1;
            match resource.ownership {
                Ownership::Moved => moved += 1,
                _ => {
                    if resource.use_count > 0 {
                        used += 1;
                    } else {
                        unused += 1;
                    }
                }
            }
        }

        LinearStatistics {
            total,
            used,
            unused,
            moved,
        }
    }
}

/// Statistics about linear resource usage.
#[derive(Debug, Clone)]
pub struct LinearStatistics {
    /// Total number of resources
    pub total: usize,
    /// Number of used resources
    pub used: usize,
    /// Number of unused resources
    pub unused: usize,
    /// Number of moved resources
    pub moved: usize,
}

/// Registry for linear type definitions.
#[derive(Debug, Clone, Default)]
pub struct LinearTypeRegistry {
    /// Registered types
    types: HashMap<String, LinearType>,
}

impl LinearTypeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        LinearTypeRegistry {
            types: HashMap::new(),
        }
    }

    /// Create a registry with common types.
    pub fn with_builtins() -> Self {
        let mut registry = LinearTypeRegistry::new();

        // GPU tensor (linear - must be freed)
        registry.register(
            LinearType::linear("Tensor")
                .with_name("GpuTensor")
                .with_tag("gpu")
                .with_description("GPU tensor that must be explicitly freed"),
        );

        // File handle (affine - can be dropped)
        registry.register(
            LinearType::affine("FileHandle")
                .with_name("FileHandle")
                .with_tag("io")
                .with_description("File handle that can be closed or dropped"),
        );

        // Network connection (linear)
        registry.register(
            LinearType::linear("Connection")
                .with_name("NetworkConnection")
                .with_tag("network")
                .with_description("Network connection that must be closed"),
        );

        // Mutex guard (linear)
        registry.register(
            LinearType::linear("Guard")
                .with_name("MutexGuard")
                .with_tag("sync")
                .with_description("Mutex guard that must be released"),
        );

        registry
    }

    /// Register a linear type.
    pub fn register(&mut self, ty: LinearType) {
        let name = ty.type_name().to_string();
        self.types.insert(name, ty);
    }

    /// Get a type by name.
    pub fn get(&self, name: &str) -> Option<&LinearType> {
        self.types.get(name)
    }

    /// Check if a type exists.
    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Get all type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kind_properties() {
        assert!(!LinearKind::Linear.allows_copy());
        assert!(!LinearKind::Linear.allows_drop());

        assert!(!LinearKind::Affine.allows_copy());
        assert!(LinearKind::Affine.allows_drop());

        assert!(LinearKind::Relevant.allows_copy());
        assert!(!LinearKind::Relevant.allows_drop());

        assert!(LinearKind::Unrestricted.allows_copy());
        assert!(LinearKind::Unrestricted.allows_drop());
    }

    #[test]
    fn test_linear_kind_join() {
        assert_eq!(
            LinearKind::Linear.join(&LinearKind::Linear),
            LinearKind::Linear
        );
        assert_eq!(
            LinearKind::Affine.join(&LinearKind::Relevant),
            LinearKind::Unrestricted
        );
    }

    #[test]
    fn test_linear_kind_meet() {
        assert_eq!(
            LinearKind::Affine.meet(&LinearKind::Relevant),
            LinearKind::Linear
        );
        assert_eq!(
            LinearKind::Linear.meet(&LinearKind::Affine),
            LinearKind::Linear
        );
    }

    #[test]
    fn test_create_resource() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();

        assert!(ctx.get_resource("x").is_some());
    }

    #[test]
    fn test_use_linear_resource_once() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();
        ctx.use_resource("x", "line 5").unwrap();

        let resource = ctx.get_resource("x").unwrap();
        assert_eq!(resource.use_count, 1);
    }

    #[test]
    fn test_use_linear_resource_twice_fails() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();
        ctx.use_resource("x", "line 5").unwrap();

        let result = ctx.use_resource("x", "line 10");
        assert!(matches!(result, Err(LinearError::MultipleUse { .. })));
    }

    #[test]
    fn test_affine_can_be_dropped() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::affine("Handle");

        ctx.create_resource("h", ty, "line 1").unwrap();
        ctx.drop_resource("h").unwrap();
    }

    #[test]
    fn test_linear_unused_fails() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();

        let result = ctx.validate_all();
        assert!(result.is_err());
    }

    #[test]
    fn test_relevant_can_be_used_multiple_times() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::relevant("Value");

        ctx.create_resource("v", ty, "line 1").unwrap();
        ctx.use_resource("v", "line 5").unwrap();
        ctx.use_resource("v", "line 10").unwrap();
        ctx.use_resource("v", "line 15").unwrap();

        let resource = ctx.get_resource("v").unwrap();
        assert_eq!(resource.use_count, 3);
    }

    #[test]
    fn test_move_resource() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();
        ctx.move_resource("x", "y", "line 5").unwrap();

        // x should be moved
        let x = ctx.get_resource("x").unwrap();
        assert_eq!(x.ownership, Ownership::Moved);
    }

    #[test]
    fn test_use_after_move_fails() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty, "line 1").unwrap();
        ctx.move_resource("x", "y", "line 5").unwrap();

        let result = ctx.use_resource("x", "line 10");
        assert!(matches!(result, Err(LinearError::UseAfterMove { .. })));
    }

    #[test]
    fn test_scope_tracking() {
        let mut ctx = LinearContext::new();

        ctx.enter_scope();

        let ty = LinearType::linear("Tensor");
        ctx.create_resource("x", ty, "line 1").unwrap();
        ctx.use_resource("x", "line 5").unwrap();

        ctx.exit_scope().unwrap();
    }

    #[test]
    fn test_scope_with_unused_linear() {
        let mut ctx = LinearContext::new();

        ctx.enter_scope();

        let ty = LinearType::linear("Tensor");
        ctx.create_resource("x", ty, "line 1").unwrap();
        // Not using x

        let result = ctx.exit_scope();
        assert!(result.is_err());
    }

    #[test]
    fn test_statistics() {
        let mut ctx = LinearContext::new();

        ctx.create_resource("a", LinearType::linear("T"), "1")
            .unwrap();
        ctx.create_resource("b", LinearType::linear("T"), "2")
            .unwrap();
        ctx.create_resource("c", LinearType::linear("T"), "3")
            .unwrap();

        ctx.use_resource("a", "10").unwrap();
        ctx.move_resource("b", "d", "20").unwrap();

        let stats = ctx.statistics();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.used, 1);
        assert_eq!(stats.unused, 1);
        assert_eq!(stats.moved, 1);
    }

    #[test]
    fn test_registry_builtins() {
        let registry = LinearTypeRegistry::with_builtins();

        assert!(registry.contains("GpuTensor"));
        assert!(registry.contains("FileHandle"));
        assert!(registry.contains("NetworkConnection"));

        let gpu = registry.get("GpuTensor").unwrap();
        assert_eq!(gpu.kind, LinearKind::Linear);
    }

    #[test]
    fn test_duplicate_resource() {
        let mut ctx = LinearContext::new();
        let ty = LinearType::linear("Tensor");

        ctx.create_resource("x", ty.clone(), "line 1").unwrap();
        let result = ctx.create_resource("x", ty, "line 5");

        assert!(matches!(result, Err(LinearError::DuplicateResource { .. })));
    }

    #[test]
    fn test_unknown_resource() {
        let mut ctx = LinearContext::new();
        let result = ctx.use_resource("unknown", "line 1");

        assert!(matches!(result, Err(LinearError::UnknownResource { .. })));
    }

    #[test]
    fn test_linear_type_with_tags() {
        let ty = LinearType::linear("Resource")
            .with_tag("gpu")
            .with_tag("memory");

        assert_eq!(ty.tags.len(), 2);
        assert!(ty.tags.contains(&"gpu".to_string()));
    }
}
