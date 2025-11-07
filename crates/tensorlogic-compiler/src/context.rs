//! Compiler context for tracking domains, variables, and axes.
//!
//! The [`CompilerContext`] is the central state manager for the compilation process.
//! It tracks domain information, variable-to-domain bindings, axis assignments,
//! and manages temporary tensor names.

use anyhow::{bail, Result};
use std::collections::HashMap;

use crate::config::CompilationConfig;

// Re-export DomainInfo from adapters for backward compatibility
pub use tensorlogic_adapters::DomainInfo;

/// Compiler context for managing compilation state.
///
/// The `CompilerContext` tracks all stateful information needed during compilation:
/// - Domain definitions and their cardinalities
/// - Variable-to-domain bindings
/// - Variable-to-axis assignments (for einsum notation)
/// - Temporary tensor name generation
/// - Compilation configuration (logic-to-tensor mapping strategies)
/// - Optional SymbolTable integration for schema-driven compilation
///
/// # Lifecycle
///
/// 1. Create a new context with [`CompilerContext::new()`], [`CompilerContext::with_config()`],
///    or [`CompilerContext::from_symbol_table()`]
/// 2. Register domains with [`add_domain`](CompilerContext::add_domain)
/// 3. Optionally bind variables to domains with [`bind_var`](CompilerContext::bind_var)
/// 4. Pass the context to [`compile_to_einsum_with_context`](crate::compile_to_einsum_with_context)
/// 5. Axes are automatically assigned during compilation
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use tensorlogic_compiler::{CompilerContext, CompilationConfig};
///
/// // Use default soft_differentiable strategy
/// let mut ctx = CompilerContext::new();
///
/// // Or use a specific strategy
/// let mut ctx_fuzzy = CompilerContext::with_config(
///     CompilationConfig::fuzzy_lukasiewicz()
/// );
///
/// // Register domains
/// ctx.add_domain("Person", 100);
/// ctx.add_domain("City", 50);
///
/// // Optionally bind variables (or let the compiler infer)
/// ctx.bind_var("x", "Person").unwrap();
/// ```
///
/// ## Schema-Driven Compilation
///
/// ```
/// use tensorlogic_compiler::CompilerContext;
/// use tensorlogic_adapters::{SymbolTable, DomainInfo};
///
/// // Create a symbol table with schema
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
///
/// // Create context from symbol table
/// let ctx = CompilerContext::from_symbol_table(&table);
///
/// // Domains are automatically imported
/// assert!(ctx.domains.contains_key("Person"));
/// ```
#[derive(Debug, Clone)]
pub struct CompilerContext {
    /// Registered domains with their metadata
    pub domains: HashMap<String, DomainInfo>,
    /// Variable-to-domain bindings
    pub var_to_domain: HashMap<String, String>,
    /// Variable-to-axis assignments (e.g., 'x' → 'a', 'y' → 'b')
    pub var_to_axis: HashMap<String, char>,
    /// Next available axis character
    next_axis: char,
    /// Counter for generating unique temporary tensor names
    temp_counter: usize,
    /// Compilation configuration (strategies for logic operations)
    pub config: CompilationConfig,
    /// Optional reference to symbol table for schema-driven compilation
    symbol_table_ref: Option<String>, // Just a marker for now
    /// Let bindings: variable name to tensor index
    pub let_bindings: HashMap<String, usize>,
}

impl CompilerContext {
    /// Creates a new, empty compiler context with default configuration.
    ///
    /// The context starts with no domains, no variable bindings, axis
    /// assignment beginning at 'a', and uses the default `soft_differentiable`
    /// compilation strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    ///
    /// let ctx = CompilerContext::new();
    /// assert!(ctx.domains.is_empty());
    /// ```
    pub fn new() -> Self {
        CompilerContext {
            domains: HashMap::new(),
            var_to_domain: HashMap::new(),
            var_to_axis: HashMap::new(),
            next_axis: 'a',
            temp_counter: 0,
            config: CompilationConfig::default(),
            symbol_table_ref: None,
            let_bindings: HashMap::new(),
        }
    }

    /// Creates a new compiler context with a specific configuration.
    ///
    /// Use this to control how logical operations are compiled to tensor operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::{CompilerContext, CompilationConfig};
    ///
    /// // Use Łukasiewicz fuzzy logic (satisfies De Morgan's laws)
    /// let ctx = CompilerContext::with_config(
    ///     CompilationConfig::fuzzy_lukasiewicz()
    /// );
    ///
    /// // Use hard Boolean logic
    /// let ctx_bool = CompilerContext::with_config(
    ///     CompilationConfig::hard_boolean()
    /// );
    /// ```
    pub fn with_config(config: CompilationConfig) -> Self {
        CompilerContext {
            domains: HashMap::new(),
            var_to_domain: HashMap::new(),
            var_to_axis: HashMap::new(),
            next_axis: 'a',
            temp_counter: 0,
            config,
            symbol_table_ref: None,
            let_bindings: HashMap::new(),
        }
    }

    /// Creates a compiler context from a SymbolTable for schema-driven compilation.
    ///
    /// This constructor automatically imports all domains from the symbol table
    /// and validates the schema. It enables type-safe compilation with rich
    /// predicate signatures and domain hierarchies.
    ///
    /// # Arguments
    ///
    /// * `table` - The symbol table containing domain and predicate definitions
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new(
    ///     "knows",
    ///     vec!["Person".to_string(), "Person".to_string()]
    /// )).unwrap();
    ///
    /// let ctx = CompilerContext::from_symbol_table(&table);
    ///
    /// assert_eq!(ctx.domains.len(), 1);
    /// assert!(ctx.domains.contains_key("Person"));
    /// ```
    pub fn from_symbol_table(table: &tensorlogic_adapters::SymbolTable) -> Self {
        let mut ctx = Self::new();

        // Import all domains from the symbol table
        for domain in table.domains.values() {
            ctx.domains.insert(domain.name.clone(), domain.clone());
        }

        // Import variable bindings if any
        for (var, domain) in &table.variables {
            ctx.var_to_domain.insert(var.clone(), domain.clone());
        }

        ctx.symbol_table_ref = Some("imported".to_string());
        ctx
    }

    /// Creates a compiler context from a SymbolTable with a specific configuration.
    ///
    /// Combines schema-driven compilation with custom compilation strategies.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::{CompilerContext, CompilationConfig};
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let ctx = CompilerContext::from_symbol_table_with_config(
    ///     &table,
    ///     CompilationConfig::fuzzy_lukasiewicz()
    /// );
    /// ```
    pub fn from_symbol_table_with_config(
        table: &tensorlogic_adapters::SymbolTable,
        config: CompilationConfig,
    ) -> Self {
        let mut ctx = Self::from_symbol_table(table);
        ctx.config = config;
        ctx
    }

    /// Registers a new domain with its cardinality.
    ///
    /// Domains must be registered before they can be used for variable bindings
    /// or quantifiers. The cardinality determines the size of the tensor dimension
    /// for variables in this domain.
    ///
    /// # Arguments
    ///
    /// * `name` - The domain name (e.g., "Person", "City")
    /// * `cardinality` - The number of possible values in this domain
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    ///
    /// let mut ctx = CompilerContext::new();
    /// ctx.add_domain("Person", 100);
    /// ctx.add_domain("City", 50);
    ///
    /// assert_eq!(ctx.domains.len(), 2);
    /// assert_eq!(ctx.domains.get("Person").unwrap().cardinality, 100);
    /// ```
    pub fn add_domain(&mut self, name: impl Into<String>, cardinality: usize) {
        let name = name.into();
        self.domains
            .insert(name.clone(), DomainInfo::new(name, cardinality));
    }

    /// Registers a domain with full metadata.
    ///
    /// Use this method when you have a complete DomainInfo instance with
    /// metadata, descriptions, or parametric types.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    /// use tensorlogic_adapters::DomainInfo;
    ///
    /// let mut ctx = CompilerContext::new();
    /// let domain = DomainInfo::new("Person", 100)
    ///     .with_description("All persons in the system");
    ///
    /// ctx.add_domain_info(domain);
    ///
    /// assert!(ctx.domains.get("Person").unwrap().description.is_some());
    /// ```
    pub fn add_domain_info(&mut self, domain: DomainInfo) {
        self.domains.insert(domain.name.clone(), domain);
    }

    /// Binds a variable to a specific domain.
    ///
    /// This is optional - the compiler can often infer domains from quantifiers.
    /// However, explicit bindings can be useful for type checking and validation.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable name (e.g., "x", "y")
    /// * `domain` - The domain name (must be already registered)
    ///
    /// # Errors
    ///
    /// Returns an error if the specified domain has not been registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    ///
    /// let mut ctx = CompilerContext::new();
    /// ctx.add_domain("Person", 100);
    ///
    /// ctx.bind_var("x", "Person").unwrap();
    /// assert_eq!(ctx.var_to_domain.get("x"), Some(&"Person".to_string()));
    ///
    /// // Error: domain not registered
    /// assert!(ctx.bind_var("y", "Unknown").is_err());
    /// ```
    pub fn bind_var(&mut self, var: &str, domain: &str) -> Result<()> {
        if !self.domains.contains_key(domain) {
            bail!("Domain '{}' not found", domain);
        }
        self.var_to_domain
            .insert(var.to_string(), domain.to_string());
        Ok(())
    }

    /// Assigns an einsum axis to a variable.
    ///
    /// Axes are assigned in lexicographic order ('a', 'b', 'c', ...).
    /// If a variable already has an assigned axis, that axis is returned.
    /// Otherwise, a new axis is assigned and the counter is incremented.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable name
    ///
    /// # Returns
    ///
    /// The axis character assigned to this variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    ///
    /// let mut ctx = CompilerContext::new();
    ///
    /// let axis_x = ctx.assign_axis("x");
    /// assert_eq!(axis_x, 'a');
    ///
    /// let axis_y = ctx.assign_axis("y");
    /// assert_eq!(axis_y, 'b');
    ///
    /// // Re-assigning returns the same axis
    /// let axis_x_again = ctx.assign_axis("x");
    /// assert_eq!(axis_x_again, 'a');
    /// ```
    pub fn assign_axis(&mut self, var: &str) -> char {
        if let Some(&axis) = self.var_to_axis.get(var) {
            return axis;
        }
        let axis = self.next_axis;
        self.var_to_axis.insert(var.to_string(), axis);
        self.next_axis = ((axis as u8) + 1) as char;
        axis
    }

    /// Generates a fresh temporary tensor name.
    ///
    /// Temporary tensors are used for intermediate results during compilation.
    /// Names are generated as "temp_0", "temp_1", etc.
    ///
    /// # Returns
    ///
    /// A unique temporary tensor name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    ///
    /// let mut ctx = CompilerContext::new();
    ///
    /// let temp1 = ctx.fresh_temp();
    /// assert_eq!(temp1, "temp_0");
    ///
    /// let temp2 = ctx.fresh_temp();
    /// assert_eq!(temp2, "temp_1");
    /// ```
    pub fn fresh_temp(&mut self) -> String {
        let name = format!("temp_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    /// Gets the einsum axes string for a list of terms.
    ///
    /// This is used internally during predicate compilation to determine
    /// the axes string for a predicate's arguments.
    ///
    /// # Arguments
    ///
    /// * `terms` - The list of terms (usually predicate arguments)
    ///
    /// # Returns
    ///
    /// A string of axis characters (e.g., "ab" for two variables)
    ///
    /// # Errors
    ///
    /// Returns an error if a variable term has not been assigned an axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_compiler::CompilerContext;
    /// use tensorlogic_ir::Term;
    ///
    /// let mut ctx = CompilerContext::new();
    /// ctx.assign_axis("x");
    /// ctx.assign_axis("y");
    ///
    /// let terms = vec![Term::var("x"), Term::var("y")];
    /// let axes = ctx.get_axes(&terms).unwrap();
    /// assert_eq!(axes, "ab");
    /// ```
    pub fn get_axes(&self, terms: &[tensorlogic_ir::Term]) -> Result<String> {
        use anyhow::anyhow;
        use tensorlogic_ir::Term;

        let mut axes = String::new();
        for term in terms {
            if let Term::Var(v) = term {
                let axis = self
                    .var_to_axis
                    .get(v)
                    .ok_or_else(|| anyhow!("Variable '{}' not assigned an axis", v))?;
                axes.push(*axis);
            }
        }
        Ok(axes)
    }
}

impl Default for CompilerContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal state during compilation of a single expression
#[derive(Debug)]
pub(crate) struct CompileState {
    pub tensor_idx: usize,
    pub axes: String,
}
