//! Python bindings for SymbolTable, CompilerContext, and related adapters

use crate::types::PyTLExpr;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};
use tensorlogic_compiler::CompilerContext;

/// Python wrapper for DomainInfo
#[pyclass(name = "DomainInfo")]
#[derive(Clone)]
pub struct PyDomainInfo {
    pub inner: DomainInfo,
}

#[pymethods]
impl PyDomainInfo {
    #[new]
    fn new(name: String, cardinality: usize) -> PyResult<Self> {
        // Validate cardinality is positive
        if cardinality == 0 {
            return Err(PyRuntimeError::new_err(
                "Domain cardinality must be positive (> 0)",
            ));
        }
        Ok(PyDomainInfo {
            inner: DomainInfo::new(name, cardinality),
        })
    }

    /// Get the domain name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the domain cardinality (size)
    #[getter]
    fn cardinality(&self) -> usize {
        self.inner.cardinality
    }

    /// Set the domain description
    fn set_description(&mut self, description: String) {
        self.inner.description = Some(description);
    }

    /// Get the domain description
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// Set the domain elements
    fn set_elements(&mut self, elements: Vec<String>) {
        self.inner.elements = Some(elements);
    }

    /// Get the domain elements
    #[getter]
    fn elements(&self) -> Option<Vec<String>> {
        self.inner.elements.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DomainInfo(name='{}', cardinality={})",
            self.inner.name, self.inner.cardinality
        )
    }

    fn __str__(&self) -> String {
        let mut s = format!(
            "Domain '{}' (size: {})",
            self.inner.name, self.inner.cardinality
        );
        if let Some(desc) = &self.inner.description {
            s.push_str(&format!("\n  Description: {}", desc));
        }
        if let Some(elements) = &self.inner.elements {
            s.push_str(&format!("\n  Elements: {:?}", elements));
        }
        s
    }
}

/// Python wrapper for PredicateInfo
#[pyclass(name = "PredicateInfo")]
#[derive(Clone)]
pub struct PyPredicateInfo {
    pub inner: PredicateInfo,
}

#[pymethods]
impl PyPredicateInfo {
    #[new]
    fn new(name: String, arg_domains: Vec<String>) -> PyResult<Self> {
        // Validate signature is not empty (predicates must have at least one argument)
        if arg_domains.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Predicate signature must have at least one argument domain",
            ));
        }
        Ok(PyPredicateInfo {
            inner: PredicateInfo::new(name, arg_domains),
        })
    }

    /// Get the predicate name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the predicate arity (number of arguments)
    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity
    }

    /// Get the argument domains
    #[getter]
    fn arg_domains(&self) -> Vec<String> {
        self.inner.arg_domains.clone()
    }

    /// Set the predicate description
    fn set_description(&mut self, description: String) {
        self.inner.description = Some(description);
    }

    /// Get the predicate description
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PredicateInfo(name='{}', arity={})",
            self.inner.name, self.inner.arity
        )
    }

    fn __str__(&self) -> String {
        let mut s = format!(
            "Predicate '{}' (arity: {})",
            self.inner.name, self.inner.arity
        );
        s.push_str(&format!(
            "\n  Argument domains: {:?}",
            self.inner.arg_domains
        ));
        if let Some(desc) = &self.inner.description {
            s.push_str(&format!("\n  Description: {}", desc));
        }
        s
    }
}

/// Python wrapper for SymbolTable
#[pyclass(name = "SymbolTable")]
#[derive(Clone)]
pub struct PySymbolTable {
    pub inner: SymbolTable,
}

#[pymethods]
impl PySymbolTable {
    #[new]
    fn new() -> Self {
        PySymbolTable {
            inner: SymbolTable::new(),
        }
    }

    /// Add a domain to the symbol table
    ///
    /// Args:
    ///     domain: DomainInfo to add
    ///
    /// Example:
    ///     >>> symbol_table = SymbolTable()
    ///     >>> domain = DomainInfo("Person", 100)
    ///     >>> symbol_table.add_domain(domain)
    fn add_domain(&mut self, domain: &PyDomainInfo) -> PyResult<()> {
        self.inner
            .add_domain(domain.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add domain: {}", e)))
    }

    /// Add a predicate to the symbol table
    ///
    /// Args:
    ///     predicate: PredicateInfo to add
    ///
    /// Raises:
    ///     RuntimeError: If referenced domains don't exist
    ///
    /// Example:
    ///     >>> symbol_table = SymbolTable()
    ///     >>> symbol_table.add_domain(DomainInfo("Person", 100))
    ///     >>> predicate = PredicateInfo("knows", ["Person", "Person"])
    ///     >>> symbol_table.add_predicate(predicate)
    fn add_predicate(&mut self, predicate: &PyPredicateInfo) -> PyResult<()> {
        self.inner
            .add_predicate(predicate.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add predicate: {}", e)))
    }

    /// Bind a variable to a domain
    ///
    /// Args:
    ///     var: Variable name
    ///     domain: Domain name
    ///
    /// Raises:
    ///     RuntimeError: If domain doesn't exist
    ///
    /// Example:
    ///     >>> symbol_table = SymbolTable()
    ///     >>> symbol_table.add_domain(DomainInfo("Person", 100))
    ///     >>> symbol_table.bind_variable("x", "Person")
    fn bind_variable(&mut self, var: String, domain: String) -> PyResult<()> {
        self.inner
            .bind_variable(var, domain)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind variable: {}", e)))
    }

    /// Get a domain by name
    ///
    /// Args:
    ///     name: Domain name
    ///
    /// Returns:
    ///     DomainInfo if found, None otherwise
    ///
    /// Example:
    ///     >>> domain = symbol_table.get_domain("Person")
    fn get_domain(&self, name: String) -> Option<PyDomainInfo> {
        self.inner
            .get_domain(&name)
            .map(|d| PyDomainInfo { inner: d.clone() })
    }

    /// Get a predicate by name
    ///
    /// Args:
    ///     name: Predicate name
    ///
    /// Returns:
    ///     PredicateInfo if found, None otherwise
    ///
    /// Example:
    ///     >>> predicate = symbol_table.get_predicate("knows")
    fn get_predicate(&self, name: String) -> Option<PyPredicateInfo> {
        self.inner
            .get_predicate(&name)
            .map(|p| PyPredicateInfo { inner: p.clone() })
    }

    /// Get the domain bound to a variable
    ///
    /// Args:
    ///     var: Variable name
    ///
    /// Returns:
    ///     Domain name if found, None otherwise
    ///
    /// Example:
    ///     >>> domain_name = symbol_table.get_variable_domain("x")
    fn get_variable_domain(&self, var: String) -> Option<String> {
        self.inner.get_variable_domain(&var).map(|s| s.to_string())
    }

    /// Infer domains and predicates from a TLExpr
    ///
    /// Args:
    ///     expr: TLExpr to analyze
    ///
    /// Example:
    ///     >>> expr = exists("y", "Person", pred("knows", [var("x"), var("y")]))
    ///     >>> symbol_table.infer_from_expr(expr)
    fn infer_from_expr(&mut self, expr: &PyTLExpr) -> PyResult<()> {
        self.inner
            .infer_from_expr(&expr.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to infer from expr: {}", e)))
    }

    /// Get all domain names
    ///
    /// Returns:
    ///     List of domain names
    ///
    /// Example:
    ///     >>> domains = symbol_table.list_domains()
    fn list_domains(&self) -> Vec<String> {
        self.inner.domains.keys().cloned().collect()
    }

    /// Get all predicate names
    ///
    /// Returns:
    ///     List of predicate names
    ///
    /// Example:
    ///     >>> predicates = symbol_table.list_predicates()
    fn list_predicates(&self) -> Vec<String> {
        self.inner.predicates.keys().cloned().collect()
    }

    /// Get all variable bindings
    ///
    /// Returns:
    ///     Dictionary mapping variable names to domain names
    ///
    /// Example:
    ///     >>> bindings = symbol_table.get_variable_bindings()
    fn get_variable_bindings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (var, domain) in self.inner.variables.iter() {
            dict.set_item(var, domain)?;
        }
        Ok(dict)
    }

    /// Export to JSON string
    ///
    /// Returns:
    ///     JSON representation of the symbol table
    ///
    /// Example:
    ///     >>> json_str = symbol_table.to_json()
    pub fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export to JSON: {}", e)))
    }

    /// Import from JSON string
    ///
    /// Args:
    ///     json: JSON string
    ///
    /// Returns:
    ///     SymbolTable parsed from JSON
    ///
    /// Example:
    ///     >>> symbol_table = SymbolTable.from_json(json_str)
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        SymbolTable::from_json(&json)
            .map(|inner| PySymbolTable { inner })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolTable(domains={}, predicates={}, variables={})",
            self.inner.domains.len(),
            self.inner.predicates.len(),
            self.inner.variables.len()
        )
    }

    fn __str__(&self) -> String {
        let mut s = String::from("SymbolTable:\n");
        s.push_str(&format!(
            "  Domains ({}): {:?}\n",
            self.inner.domains.len(),
            self.list_domains()
        ));
        s.push_str(&format!(
            "  Predicates ({}): {:?}\n",
            self.inner.predicates.len(),
            self.list_predicates()
        ));
        s.push_str(&format!("  Variables ({})", self.inner.variables.len()));
        s
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::symbol_table_html;

        // Collect domains
        let domains: Vec<(String, usize, Option<String>)> = self
            .inner
            .domains
            .iter()
            .map(|(name, info)| (name.clone(), info.cardinality, info.description.clone()))
            .collect();

        // Collect predicates
        let predicates: Vec<(String, usize, Vec<String>)> = self
            .inner
            .predicates
            .iter()
            .map(|(name, info)| (name.clone(), info.arity, info.arg_domains.clone()))
            .collect();

        // Collect variable bindings
        let variables: Vec<(String, String)> = self
            .inner
            .variables
            .iter()
            .map(|(var, domain)| (var.clone(), domain.clone()))
            .collect();

        symbol_table_html(&domains, &predicates, &variables)
    }
}

/// Create a DomainInfo
///
/// Args:
///     name: Domain name
///     cardinality: Domain size
///
/// Returns:
///     A DomainInfo
///
/// Example:
///     >>> domain = domain_info("Person", 100)
#[pyfunction(name = "domain_info")]
pub fn py_domain_info(name: String, cardinality: usize) -> PyResult<PyDomainInfo> {
    PyDomainInfo::new(name, cardinality)
}

/// Create a PredicateInfo
///
/// Args:
///     name: Predicate name
///     arg_domains: List of domain names for arguments
///
/// Returns:
///     A PredicateInfo
///
/// Example:
///     >>> predicate = predicate_info("knows", ["Person", "Person"])
#[pyfunction(name = "predicate_info")]
pub fn py_predicate_info(name: String, arg_domains: Vec<String>) -> PyResult<PyPredicateInfo> {
    PyPredicateInfo::new(name, arg_domains)
}

/// Create a SymbolTable
///
/// Returns:
///     A new empty SymbolTable
///
/// Example:
///     >>> symbol_table = symbol_table()
#[pyfunction(name = "symbol_table")]
pub fn py_symbol_table() -> PySymbolTable {
    PySymbolTable::new()
}

// ============================================================================
// CompilerContext Bindings
// ============================================================================

/// Python wrapper for CompilerContext
#[pyclass(name = "CompilerContext")]
#[derive(Clone)]
pub struct PyCompilerContext {
    pub inner: CompilerContext,
}

#[pymethods]
impl PyCompilerContext {
    #[new]
    fn new() -> Self {
        PyCompilerContext {
            inner: CompilerContext::new(),
        }
    }

    /// Add a domain to the context
    ///
    /// Args:
    ///     name: Domain name
    ///     cardinality: Domain size
    ///
    /// Example:
    ///     >>> ctx = CompilerContext()
    ///     >>> ctx.add_domain("Person", 100)
    fn add_domain(&mut self, name: String, cardinality: usize) {
        self.inner.add_domain(name, cardinality);
    }

    /// Bind a variable to a domain
    ///
    /// Args:
    ///     var: Variable name
    ///     domain: Domain name (must exist)
    ///
    /// Raises:
    ///     RuntimeError: If domain doesn't exist
    ///
    /// Example:
    ///     >>> ctx.bind_var("x", "Person")
    fn bind_var(&mut self, var: String, domain: String) -> PyResult<()> {
        self.inner
            .bind_var(&var, &domain)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind variable: {}", e)))
    }

    /// Assign an einsum axis to a variable
    ///
    /// Args:
    ///     var: Variable name
    ///
    /// Returns:
    ///     Axis character (e.g., 'a', 'b', 'c')
    ///
    /// Example:
    ///     >>> axis = ctx.assign_axis("x")  # Returns 'a'
    fn assign_axis(&mut self, var: String) -> String {
        self.inner.assign_axis(&var).to_string()
    }

    /// Generate a fresh temporary tensor name
    ///
    /// Returns:
    ///     Unique temporary name (e.g., "temp_0", "temp_1")
    ///
    /// Example:
    ///     >>> temp = ctx.fresh_temp()  # Returns "temp_0"
    fn fresh_temp(&mut self) -> String {
        self.inner.fresh_temp()
    }

    /// Get all registered domains
    ///
    /// Returns:
    ///     Dictionary mapping domain names to cardinalities
    ///
    /// Example:
    ///     >>> domains = ctx.get_domains()
    fn get_domains<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, domain_info) in self.inner.domains.iter() {
            dict.set_item(name, domain_info.cardinality)?;
        }
        Ok(dict)
    }

    /// Get all variable bindings
    ///
    /// Returns:
    ///     Dictionary mapping variable names to domain names
    ///
    /// Example:
    ///     >>> bindings = ctx.get_variable_bindings()
    fn get_variable_bindings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (var, domain) in self.inner.var_to_domain.iter() {
            dict.set_item(var, domain)?;
        }
        Ok(dict)
    }

    /// Get all axis assignments
    ///
    /// Returns:
    ///     Dictionary mapping variable names to axis characters
    ///
    /// Example:
    ///     >>> axes = ctx.get_axis_assignments()
    fn get_axis_assignments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (var, axis) in self.inner.var_to_axis.iter() {
            dict.set_item(var, axis.to_string())?;
        }
        Ok(dict)
    }

    /// Get the domain for a variable
    ///
    /// Args:
    ///     var: Variable name
    ///
    /// Returns:
    ///     Domain name if bound, None otherwise
    ///
    /// Example:
    ///     >>> domain = ctx.get_variable_domain("x")
    fn get_variable_domain(&self, var: String) -> Option<String> {
        self.inner.var_to_domain.get(&var).cloned()
    }

    /// Get the axis for a variable
    ///
    /// Args:
    ///     var: Variable name
    ///
    /// Returns:
    ///     Axis character if assigned, None otherwise
    ///
    /// Example:
    ///     >>> axis = ctx.get_variable_axis("x")
    fn get_variable_axis(&self, var: String) -> Option<String> {
        self.inner.var_to_axis.get(&var).map(|c| c.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "CompilerContext(domains={}, vars={}, axes={})",
            self.inner.domains.len(),
            self.inner.var_to_domain.len(),
            self.inner.var_to_axis.len()
        )
    }

    fn __str__(&self) -> String {
        let mut s = String::from("CompilerContext:\n");
        s.push_str(&format!("  Domains: {}\n", self.inner.domains.len()));
        for (name, info) in &self.inner.domains {
            s.push_str(&format!("    - {} (size: {})\n", name, info.cardinality));
        }
        s.push_str(&format!(
            "  Variable bindings: {}\n",
            self.inner.var_to_domain.len()
        ));
        for (var, domain) in &self.inner.var_to_domain {
            s.push_str(&format!("    - {} -> {}\n", var, domain));
        }
        s.push_str(&format!(
            "  Axis assignments: {}",
            self.inner.var_to_axis.len()
        ));
        s
    }
}

/// Create a CompilerContext
///
/// Returns:
///     A new empty CompilerContext
///
/// Example:
///     >>> ctx = compiler_context()
#[pyfunction(name = "compiler_context")]
pub fn py_compiler_context() -> PyCompilerContext {
    PyCompilerContext::new()
}
