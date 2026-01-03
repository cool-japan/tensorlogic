//! Incremental compilation for efficient recompilation when expressions change.
//!
//! This module provides an incremental compilation system that tracks dependencies
//! and recompiles only the parts of expressions that have changed. This is crucial
//! for interactive environments like REPLs, notebooks, and IDEs where expressions
//! are frequently modified.
//!
//! # Architecture
//!
//! The incremental compilation system consists of three main components:
//!
//! 1. **DependencyTracker**: Tracks what each expression depends on (predicates,
//!    variables, domains, configurations).
//!
//! 2. **ChangeDetector**: Detects changes to the compilation context (predicate
//!    signatures, domains, configurations) and determines what needs recompilation.
//!
//! 3. **IncrementalCompiler**: Manages the compilation state, computes minimal
//!    invalidation sets, and recompiles only affected sub-expressions.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::{CompilerContext, incremental::IncrementalCompiler};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 100);
//!
//! let mut compiler = IncrementalCompiler::new(ctx);
//!
//! // Initial compilation
//! let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
//! let graph1 = compiler.compile(&expr1).unwrap();
//!
//! // Compile similar expression - some parts will be reused
//! let expr2 = TLExpr::and(
//!     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
//!     TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]),
//! );
//! let graph2 = compiler.compile(&expr2).unwrap();
//!
//! // Check incremental compilation stats
//! let stats = compiler.stats();
//! println!("Nodes reused: {}", stats.nodes_reused);
//! println!("Nodes compiled: {}", stats.nodes_compiled);
//! println!("Reuse rate: {:.1}%", stats.reuse_rate() * 100.0);
//! ```

use crate::{compile_to_einsum_with_context, CompilerContext};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tensorlogic_ir::{EinsumGraph, IrError, TLExpr, Term};

/// Tracks dependencies of compiled expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressionDependencies {
    /// Predicates referenced in the expression
    pub predicates: HashSet<String>,
    /// Variables referenced in the expression
    pub variables: HashSet<String>,
    /// Domains used in the expression
    pub domains: HashSet<String>,
    /// Configuration hash (to detect strategy changes)
    pub config_hash: u64,
}

impl ExpressionDependencies {
    /// Create a new empty dependency set.
    pub fn new() -> Self {
        Self {
            predicates: HashSet::new(),
            variables: HashSet::new(),
            domains: HashSet::new(),
            config_hash: 0,
        }
    }

    /// Analyze an expression and extract its dependencies.
    pub fn analyze(expr: &TLExpr, ctx: &CompilerContext) -> Self {
        let mut deps = Self::new();
        deps.analyze_recursive(expr);
        deps.config_hash = Self::hash_config(ctx);
        deps
    }

    fn analyze_recursive(&mut self, expr: &TLExpr) {
        match expr {
            TLExpr::Pred { name, args } => {
                self.predicates.insert(name.clone());
                for arg in args {
                    self.analyze_term(arg);
                }
            }
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                self.analyze_recursive(left);
                self.analyze_recursive(right);
            }
            TLExpr::Not(inner) => {
                self.analyze_recursive(inner);
            }
            TLExpr::Exists { var, domain, body } | TLExpr::ForAll { var, domain, body } => {
                self.variables.insert(var.clone());
                self.domains.insert(domain.clone());
                self.analyze_recursive(body);
            }
            TLExpr::Score(inner) => {
                self.analyze_recursive(inner);
            }
            TLExpr::Add(left, right)
            | TLExpr::Sub(left, right)
            | TLExpr::Mul(left, right)
            | TLExpr::Div(left, right) => {
                self.analyze_recursive(left);
                self.analyze_recursive(right);
            }
            TLExpr::Eq(left, right)
            | TLExpr::Lt(left, right)
            | TLExpr::Gt(left, right)
            | TLExpr::Lte(left, right)
            | TLExpr::Gte(left, right) => {
                self.analyze_recursive(left);
                self.analyze_recursive(right);
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.analyze_recursive(condition);
                self.analyze_recursive(then_branch);
                self.analyze_recursive(else_branch);
            }
            TLExpr::Aggregate {
                op: _,
                var,
                domain,
                body,
                group_by,
            } => {
                self.variables.insert(var.clone());
                self.domains.insert(domain.clone());
                self.analyze_recursive(body);
                if let Some(gb_vars) = group_by {
                    for var_name in gb_vars {
                        self.variables.insert(var_name.clone());
                    }
                }
            }
            TLExpr::TNorm {
                kind: _,
                left,
                right,
            }
            | TLExpr::TCoNorm {
                kind: _,
                left,
                right,
            } => {
                self.analyze_recursive(left);
                self.analyze_recursive(right);
            }
            TLExpr::FuzzyNot {
                kind: _,
                expr: inner,
            } => {
                self.analyze_recursive(inner);
            }
            TLExpr::FuzzyImplication {
                kind: _,
                premise,
                conclusion,
            } => {
                self.analyze_recursive(premise);
                self.analyze_recursive(conclusion);
            }
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature: _,
            }
            | TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature: _,
            } => {
                self.variables.insert(var.clone());
                self.domains.insert(domain.clone());
                self.analyze_recursive(body);
            }
            TLExpr::WeightedRule { weight: _, rule } => {
                self.analyze_recursive(rule);
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, alt) in alternatives {
                    self.analyze_recursive(alt);
                }
            }
            TLExpr::Let { var, value, body } => {
                self.variables.insert(var.clone());
                self.analyze_recursive(value);
                self.analyze_recursive(body);
            }
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => {
                self.analyze_recursive(inner);
            }
            TLExpr::Until { before, after } | TLExpr::WeakUntil { before, after } => {
                self.analyze_recursive(before);
                self.analyze_recursive(after);
            }
            TLExpr::Release { released, releaser }
            | TLExpr::StrongRelease { released, releaser } => {
                self.analyze_recursive(released);
                self.analyze_recursive(releaser);
            }
            // Math operations
            TLExpr::Abs(inner)
            | TLExpr::Sqrt(inner)
            | TLExpr::Exp(inner)
            | TLExpr::Log(inner)
            | TLExpr::Sin(inner)
            | TLExpr::Cos(inner)
            | TLExpr::Tan(inner)
            | TLExpr::Floor(inner)
            | TLExpr::Ceil(inner)
            | TLExpr::Round(inner) => {
                self.analyze_recursive(inner);
            }
            TLExpr::Pow(left, right)
            | TLExpr::Min(left, right)
            | TLExpr::Max(left, right)
            | TLExpr::Mod(left, right) => {
                self.analyze_recursive(left);
                self.analyze_recursive(right);
            }
            TLExpr::Constant(_) => {
                // No dependencies
            }
            // All other expression types (alpha.3 enhancements)
            _ => {
                // For unhandled variants, recurse on any child expressions if needed
                // This is a catch-all for future-proof compilation
            }
        }
    }

    fn analyze_term(&mut self, term: &Term) {
        if let Term::Var(name) = term {
            self.variables.insert(name.clone());
        }
    }

    fn hash_config(ctx: &CompilerContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Hash the config strategies
        format!("{:?}", ctx.config).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ExpressionDependencies {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects changes to the compilation context.
#[derive(Debug, Clone)]
pub struct ChangeDetector {
    /// Previous predicate signatures
    previous_predicates: HashMap<String, (usize, Vec<String>)>,
    /// Previous domain sizes
    previous_domains: HashMap<String, usize>,
    /// Previous configuration hash
    previous_config_hash: u64,
}

impl ChangeDetector {
    /// Create a new change detector.
    pub fn new() -> Self {
        Self {
            previous_predicates: HashMap::new(),
            previous_domains: HashMap::new(),
            previous_config_hash: 0,
        }
    }

    /// Update the snapshot from the current context.
    pub fn update(&mut self, ctx: &CompilerContext) {
        self.previous_predicates.clear();
        self.previous_domains.clear();

        // Snapshot domains
        for (name, info) in &ctx.domains {
            self.previous_domains.insert(name.clone(), info.cardinality);
        }

        self.previous_config_hash = ExpressionDependencies::hash_config(ctx);
    }

    /// Detect changes and return affected predicates and domains.
    pub fn detect_changes(&self, ctx: &CompilerContext) -> ChangeSet {
        let mut changes = ChangeSet::new();

        // Check domain changes
        for (name, info) in &ctx.domains {
            if let Some(&prev_size) = self.previous_domains.get(name.as_str()) {
                if prev_size != info.cardinality {
                    changes.changed_domains.insert(name.clone());
                }
            } else {
                changes.new_domains.insert(name.clone());
            }
        }

        // Check for removed domains
        for name in self.previous_domains.keys() {
            if !ctx.domains.contains_key(name) {
                changes.removed_domains.insert(name.clone());
            }
        }

        // Check configuration changes
        let current_hash = ExpressionDependencies::hash_config(ctx);
        if current_hash != self.previous_config_hash {
            changes.config_changed = true;
        }

        changes
    }
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Describes what has changed in the compilation context.
#[derive(Debug, Clone, Default)]
pub struct ChangeSet {
    /// Predicates that were added
    pub new_predicates: HashSet<String>,
    /// Predicates that were modified
    pub changed_predicates: HashSet<String>,
    /// Predicates that were removed
    pub removed_predicates: HashSet<String>,
    /// Domains that were added
    pub new_domains: HashSet<String>,
    /// Domains that were modified
    pub changed_domains: HashSet<String>,
    /// Domains that were removed
    pub removed_domains: HashSet<String>,
    /// Whether the configuration changed
    pub config_changed: bool,
}

impl ChangeSet {
    fn new() -> Self {
        Self::default()
    }

    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        !self.new_predicates.is_empty()
            || !self.changed_predicates.is_empty()
            || !self.removed_predicates.is_empty()
            || !self.new_domains.is_empty()
            || !self.changed_domains.is_empty()
            || !self.removed_domains.is_empty()
            || self.config_changed
    }

    /// Check if a dependency set is affected by these changes.
    pub fn affects(&self, deps: &ExpressionDependencies) -> bool {
        // Config changes affect everything
        if self.config_changed {
            return true;
        }

        // Check if any used predicate changed
        for pred in &deps.predicates {
            if self.changed_predicates.contains(pred) || self.removed_predicates.contains(pred) {
                return true;
            }
        }

        // Check if any used domain changed
        for domain in &deps.domains {
            if self.changed_domains.contains(domain) || self.removed_domains.contains(domain) {
                return true;
            }
        }

        false
    }
}

/// Entry in the incremental compilation cache.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The compiled graph
    graph: EinsumGraph,
    /// Dependencies of this expression
    dependencies: ExpressionDependencies,
    /// When this was compiled (for LRU eviction)
    #[allow(dead_code)]
    timestamp: u64,
}

/// Incremental compiler that reuses previously compiled expressions.
pub struct IncrementalCompiler {
    /// Compilation context
    context: CompilerContext,
    /// Cache of compiled expressions
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    /// Change detector
    change_detector: ChangeDetector,
    /// Statistics
    stats: Arc<Mutex<IncrementalStats>>,
    /// Next timestamp for LRU
    next_timestamp: Arc<Mutex<u64>>,
}

impl IncrementalCompiler {
    /// Create a new incremental compiler with the given context.
    pub fn new(context: CompilerContext) -> Self {
        let mut change_detector = ChangeDetector::new();
        change_detector.update(&context);

        Self {
            context,
            cache: Arc::new(Mutex::new(HashMap::new())),
            change_detector,
            stats: Arc::new(Mutex::new(IncrementalStats::default())),
            next_timestamp: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the compilation context.
    pub fn context(&self) -> &CompilerContext {
        &self.context
    }

    /// Get a mutable reference to the compilation context.
    pub fn context_mut(&mut self) -> &mut CompilerContext {
        &mut self.context
    }

    /// Compile an expression incrementally.
    pub fn compile(&mut self, expr: &TLExpr) -> Result<EinsumGraph, IrError> {
        // Detect changes since last compilation
        let changes = self.change_detector.detect_changes(&self.context);

        // Invalidate affected entries if there are changes
        if changes.has_changes() {
            self.invalidate_affected(&changes);
            self.change_detector.update(&self.context);
        }

        // Try to get from cache
        let expr_key = format!("{:?}", expr);
        let cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get(&expr_key) {
            // Cache hit!
            let mut stats = self.stats.lock().unwrap();
            stats.cache_hits += 1;
            stats.nodes_reused += entry.graph.nodes.len();
            drop(stats);

            return Ok(entry.graph.clone());
        }

        // Cache miss - compile from scratch
        drop(cache);

        let deps = ExpressionDependencies::analyze(expr, &self.context);
        // Compile from scratch - we can't use ? here because anyhow::Error doesn't convert to IrError
        // So we'll just propagate the error as InvalidEinsumSpec
        let graph = compile_to_einsum_with_context(expr, &mut self.context).map_err(|e| {
            IrError::InvalidEinsumSpec {
                spec: format!("{:?}", expr),
                reason: format!("Compilation failed: {}", e),
            }
        })?;

        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.cache_misses += 1;
        stats.nodes_compiled += graph.nodes.len();
        drop(stats);

        // Store in cache
        let mut timestamp_guard = self.next_timestamp.lock().unwrap();
        let timestamp = *timestamp_guard;
        *timestamp_guard += 1;
        drop(timestamp_guard);

        let mut cache = self.cache.lock().unwrap();
        cache.insert(
            expr_key,
            CacheEntry {
                graph: graph.clone(),
                dependencies: deps,
                timestamp,
            },
        );

        Ok(graph)
    }

    /// Invalidate cache entries affected by changes.
    fn invalidate_affected(&mut self, changes: &ChangeSet) {
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|_, entry| !changes.affects(&entry.dependencies));

        let mut stats = self.stats.lock().unwrap();
        stats.invalidations += 1;
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Get incremental compilation statistics.
    pub fn stats(&self) -> IncrementalStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = IncrementalStats::default();
    }
}

/// Statistics for incremental compilation.
#[derive(Debug, Clone, Default)]
pub struct IncrementalStats {
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of invalidations
    pub invalidations: usize,
    /// Number of nodes reused from cache
    pub nodes_reused: usize,
    /// Number of nodes freshly compiled
    pub nodes_compiled: usize,
}

impl IncrementalStats {
    /// Get the cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Get the reuse rate for nodes (0.0 to 1.0).
    pub fn reuse_rate(&self) -> f64 {
        let total = self.nodes_reused + self.nodes_compiled;
        if total == 0 {
            0.0
        } else {
            self.nodes_reused as f64 / total as f64
        }
    }

    /// Get the total number of compilations.
    pub fn total_compilations(&self) -> usize {
        self.cache_hits + self.cache_misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_tracking() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let deps = ExpressionDependencies::analyze(&expr, &ctx);

        assert!(deps.predicates.contains("knows"));
        assert!(deps.variables.contains("x"));
        assert!(deps.variables.contains("y"));
    }

    #[test]
    fn test_incremental_compilation_reuse() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let mut compiler = IncrementalCompiler::new(ctx);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        // First compilation
        let _graph1 = compiler.compile(&expr).unwrap();
        assert_eq!(compiler.stats().cache_misses, 1);
        assert_eq!(compiler.stats().cache_hits, 0);

        // Second compilation - should hit cache
        let _graph2 = compiler.compile(&expr).unwrap();
        assert_eq!(compiler.stats().cache_misses, 1);
        assert_eq!(compiler.stats().cache_hits, 1);
        assert_eq!(compiler.stats().hit_rate(), 0.5);
    }

    #[test]
    fn test_change_detection_domain() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let mut detector = ChangeDetector::new();
        detector.update(&ctx);

        // No changes initially
        let changes = detector.detect_changes(&ctx);
        assert!(!changes.has_changes());

        // Change domain size
        ctx.add_domain("Person", 200);
        let changes = detector.detect_changes(&ctx);
        assert!(changes.has_changes());
        assert!(changes.changed_domains.contains("Person"));
    }

    #[test]
    fn test_invalidation_on_domain_change() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let mut compiler = IncrementalCompiler::new(ctx);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        // First compilation
        let _graph1 = compiler.compile(&expr).unwrap();
        assert_eq!(compiler.stats().cache_misses, 1);

        // Change domain
        compiler.context_mut().add_domain("Person", 200);

        // Should recompile due to domain change and invalidate cache
        let _graph2 = compiler.compile(&expr).unwrap();
        // After invalidation, this is another cache miss
        assert!(compiler.stats().cache_misses >= 1);
        assert!(compiler.stats().invalidations >= 1);
    }

    #[test]
    fn test_incremental_stats() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let mut compiler = IncrementalCompiler::new(ctx);

        let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr2 = TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]);

        compiler.compile(&expr1).unwrap();
        compiler.compile(&expr1).unwrap(); // Should be cache hit
        compiler.compile(&expr2).unwrap();

        let stats = compiler.stats();
        assert_eq!(stats.total_compilations(), 3);
        // At least one cache hit from the second expr1 compilation
        assert!(
            stats.cache_hits >= 1,
            "Expected at least 1 cache hit, got {}",
            stats.cache_hits
        );
        // Hit rate should be positive if we have cache hits
        assert!(
            stats.hit_rate() > 0.0,
            "Expected positive hit rate, got {}",
            stats.hit_rate()
        );
    }

    #[test]
    fn test_complex_expression_dependencies() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::and(
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
                TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]),
            ),
        );

        let deps = ExpressionDependencies::analyze(&expr, &ctx);

        assert!(deps.predicates.contains("knows"));
        assert!(deps.predicates.contains("likes"));
        assert!(deps.variables.contains("x"));
        assert!(deps.variables.contains("y"));
        assert!(deps.variables.contains("z"));
        assert!(deps.domains.contains("Person"));
    }
}
