//! Rule Builder DSL - Python-native syntax for defining logic rules
//!
//! This module provides a high-level DSL for defining logic rules using Python decorators,
//! operator overloading, and type annotations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tensorlogic_ir::{TLExpr, Term};

use crate::adapters::{PyPredicateInfo, PySymbolTable};
use crate::compiler::PyCompilationConfig;
use crate::types::{PyEinsumGraph, PyTLExpr, PyTerm};

/// Variable wrapper with operator overloading
///
/// This class enables Python-native syntax for building logic expressions
/// using operators: & (AND), | (OR), ~ (NOT), >> (IMPLY)
///
/// Example:
///     >>> from pytensorlogic.dsl import Var, pred
///     >>> x = Var("x", domain="Person")
///     >>> y = Var("y", domain="Person")
///     >>> knows = pred("knows")
///     >>> expr = knows(x, y) & knows(y, x)  # Mutual knowledge
#[pyclass(name = "Var")]
#[derive(Clone)]
pub struct PyVar {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub domain: Option<String>,
}

#[pymethods]
impl PyVar {
    #[new]
    #[pyo3(signature = (name, domain=None))]
    fn new(name: String, domain: Option<String>) -> Self {
        PyVar { name, domain }
    }

    fn __repr__(&self) -> String {
        if let Some(ref dom) = self.domain {
            format!("Var('{}', domain='{}')", self.name, dom)
        } else {
            format!("Var('{}')", self.name)
        }
    }

    fn __str__(&self) -> String {
        self.name.clone()
    }

    /// Convert to PyTerm for internal use
    fn to_term(&self) -> PyTerm {
        PyTerm {
            inner: Term::Var(self.name.clone()),
        }
    }

    /// Get the underlying TLExpr representation as a predicate
    /// This wraps the variable in a unary predicate for compatibility
    fn to_expr(&self) -> PyTLExpr {
        PyTLExpr {
            inner: TLExpr::Pred {
                name: self.name.clone(),
                args: vec![],
            },
        }
    }
}

/// Predicate builder for function-call syntax
///
/// This class enables defining predicates that can be called with variables
/// to produce TLExpr instances.
///
/// Example:
///     >>> from pytensorlogic.dsl import pred, Var
///     >>> knows = pred("knows", arity=2)
///     >>> x = Var("x")
///     >>> y = Var("y")
///     >>> expr = knows(x, y)  # Creates a predicate expression
#[pyclass(name = "PredicateBuilder")]
#[derive(Clone)]
pub struct PyPredicateBuilder {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub arity: Option<usize>,
    #[pyo3(get)]
    pub domains: Option<Vec<String>>,
}

#[pymethods]
impl PyPredicateBuilder {
    #[new]
    #[pyo3(signature = (name, arity=None, domains=None))]
    fn new(name: String, arity: Option<usize>, domains: Option<Vec<String>>) -> Self {
        PyPredicateBuilder {
            name,
            arity,
            domains,
        }
    }

    fn __repr__(&self) -> String {
        format!("PredicateBuilder('{}')", self.name)
    }

    fn __str__(&self) -> String {
        self.name.clone()
    }

    /// Call the predicate with arguments to create a TLExpr
    #[pyo3(signature = (*args))]
    fn __call__(&self, _py: Python, args: &Bound<'_, PyTuple>) -> PyResult<PyTLExpr> {
        // Validate arity
        if let Some(expected_arity) = self.arity {
            if args.len() != expected_arity {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Predicate '{}' expects {} arguments, got {}",
                    self.name,
                    expected_arity,
                    args.len()
                )));
            }
        }

        // Convert arguments to Terms
        let mut terms = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let term = if let Ok(var) = arg.extract::<PyVar>() {
                // Validate domain if specified
                if let Some(ref expected_domains) = self.domains {
                    if i < expected_domains.len() {
                        if let Some(ref var_domain) = var.domain {
                            if var_domain != &expected_domains[i] {
                                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                    format!(
                                        "Argument {} expects domain '{}', got '{}'",
                                        i, expected_domains[i], var_domain
                                    ),
                                ));
                            }
                        }
                    }
                }
                Term::Var(var.name)
            } else if let Ok(s) = arg.extract::<String>() {
                Term::Const(s)
            } else if let Ok(term) = arg.extract::<PyTerm>() {
                term.inner
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Argument {} must be Var, str, or Term, got {}",
                    i,
                    arg.get_type().name()?
                )));
            };
            terms.push(term);
        }

        Ok(PyTLExpr {
            inner: TLExpr::Pred {
                name: self.name.clone(),
                args: terms,
            },
        })
    }

    /// Get predicate metadata as PredicateInfo
    fn to_predicate_info(&self) -> PyResult<PyPredicateInfo> {
        let arity = self.arity.unwrap_or(0);
        let arg_domains = self.domains.clone().unwrap_or_default();

        let pred_info = tensorlogic_adapters::PredicateInfo {
            name: self.name.clone(),
            arity,
            arg_domains,
            description: None,
            constraints: None,
            metadata: None,
        };

        // Add predicate to internal table if not already present
        Ok(PyPredicateInfo { inner: pred_info })
    }
}

/// Rule builder context manager for collecting and compiling rules
///
/// This class provides a context manager for defining multiple rules and
/// compiling them together into a single execution graph.
///
/// Example:
///     >>> from pytensorlogic.dsl import RuleBuilder
///     >>> with RuleBuilder() as rb:
///     ...     x, y, z = rb.vars("x", "y", "z", domain="Person")
///     ...     knows = rb.pred("knows", arity=2)
///     ...     rule1 = (knows(x, y) & knows(y, z)) >> knows(x, z)
///     ...     rb.add_rule(rule1, name="transitivity")
///     ...     graph = rb.compile()
#[pyclass(name = "RuleBuilder")]
pub struct PyRuleBuilder {
    rules: Vec<(String, PyTLExpr)>,
    symbol_table: PySymbolTable,
    config: Option<PyCompilationConfig>,
}

#[pymethods]
impl PyRuleBuilder {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCompilationConfig>) -> Self {
        PyRuleBuilder {
            rules: Vec::new(),
            symbol_table: PySymbolTable {
                inner: Default::default(),
            },
            config,
        }
    }

    fn __repr__(&self) -> String {
        format!("RuleBuilder({} rules)", self.rules.len())
    }

    /// Enter context manager
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Exit context manager
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(false) // Don't suppress exceptions
    }

    /// Create variables with optional domain
    ///
    /// Args:
    ///     names: Variable names (as separate arguments)
    ///     domain: Optional domain name for all variables
    ///
    /// Returns:
    ///     Single Var or tuple of Vars
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> x = rb.vars("x", domain="Person")
    ///     >>> x, y, z = rb.vars("x", "y", "z", domain="Person")
    #[pyo3(signature = (*names, domain=None))]
    fn vars(
        &mut self,
        py: Python,
        names: &Bound<'_, PyTuple>,
        domain: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        if names.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "At least one variable name required",
            ));
        }

        let mut vars = Vec::new();
        for name_obj in names.iter() {
            let name = name_obj.extract::<String>()?;

            // Try to bind variable to domain in symbol table if specified
            // If domain doesn't exist, just skip binding (but still set domain on Var)
            if let Some(ref dom) = domain {
                let _ = self.symbol_table.inner.bind_variable(&name, dom);
            }

            vars.push(PyVar {
                name,
                domain: domain.clone(),
            });
        }

        if vars.len() == 1 {
            let var = Py::new(py, vars.into_iter().next().unwrap())?;
            Ok(var.into())
        } else {
            let py_vars: Vec<Py<PyVar>> = vars
                .into_iter()
                .map(|v| Py::new(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, py_vars)?.into())
        }
    }

    /// Create a predicate builder
    ///
    /// Args:
    ///     name: Predicate name
    ///     arity: Number of arguments (optional)
    ///     domains: Domain names for each argument (optional)
    ///
    /// Returns:
    ///     PredicateBuilder instance
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> knows = rb.pred("knows", arity=2, domains=["Person", "Person"])
    #[pyo3(signature = (name, arity=None, domains=None))]
    fn pred(
        &mut self,
        name: String,
        arity: Option<usize>,
        domains: Option<Vec<String>>,
    ) -> PyResult<PyPredicateBuilder> {
        // Add predicate to symbol table if metadata provided
        if let (Some(a), Some(ref doms)) = (arity, &domains) {
            let pred_info = tensorlogic_adapters::PredicateInfo {
                name: name.clone(),
                arity: a,
                arg_domains: doms.clone(),
                description: None,
                constraints: None,
                metadata: None,
            };
            let _ = self.symbol_table.inner.add_predicate(pred_info);
        }

        Ok(PyPredicateBuilder {
            name,
            arity,
            domains,
        })
    }

    /// Add a domain to the symbol table
    ///
    /// Args:
    ///     name: Domain name
    ///     cardinality: Number of elements
    ///     description: Optional description
    ///     elements: Optional list of element names
    ///
    /// Returns:
    ///     Self (for chaining)
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> rb.add_domain("Person", cardinality=10, description="People")
    #[pyo3(signature = (name, cardinality, description=None, elements=None))]
    fn add_domain(
        &mut self,
        name: String,
        cardinality: usize,
        description: Option<String>,
        elements: Option<Vec<String>>,
    ) -> PyResult<()> {
        let domain_info = tensorlogic_adapters::DomainInfo {
            name,
            cardinality,
            description,
            elements,
            parametric_type: None,
            metadata: None,
        };

        let _ = self.symbol_table.inner.add_domain(domain_info);
        Ok(())
    }

    /// Add a rule to the builder
    ///
    /// Args:
    ///     expr: TLExpr representing the rule
    ///     name: Optional name for the rule (default: rule_N)
    ///
    /// Returns:
    ///     Self (for chaining)
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> x, y = rb.vars("x", "y")
    ///     >>> knows = rb.pred("knows")
    ///     >>> rule = knows(x, y) >> knows(y, x)
    ///     >>> rb.add_rule(rule, name="symmetry")
    #[pyo3(signature = (expr, name=None))]
    fn add_rule(&mut self, expr: PyTLExpr, name: Option<String>) -> PyResult<()> {
        let rule_name = name.unwrap_or_else(|| format!("rule_{}", self.rules.len()));
        self.rules.push((rule_name, expr));
        Ok(())
    }

    /// Get all defined rules
    ///
    /// Returns:
    ///     List of (name, expr) tuples
    fn get_rules(&self) -> Vec<(String, PyTLExpr)> {
        self.rules.clone()
    }

    /// Get the symbol table
    ///
    /// Returns:
    ///     SymbolTable instance
    fn get_symbol_table(&self) -> PySymbolTable {
        self.symbol_table.clone()
    }

    /// Compile all rules into a single execution graph
    ///
    /// Args:
    ///     config: Optional compilation config (overrides builder config)
    ///
    /// Returns:
    ///     EinsumGraph instance
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> # ... define rules ...
    ///     >>> graph = rb.compile()
    #[pyo3(signature = (config=None))]
    fn compile(&self, _py: Python, config: Option<PyCompilationConfig>) -> PyResult<PyEinsumGraph> {
        if self.rules.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No rules defined. Use add_rule() to add rules before compiling.",
            ));
        }

        // Use provided config, or builder config, or default soft_differentiable
        let compile_config = if let Some(cfg) = config {
            cfg
        } else if let Some(cfg) = self.config.clone() {
            cfg
        } else {
            // Create a default config using the public constructor
            PyCompilationConfig {
                inner: tensorlogic_compiler::CompilationConfig::default(),
            }
        };

        // If only one rule, compile it directly
        if self.rules.len() == 1 {
            let (_, expr) = &self.rules[0];
            return crate::compiler::py_compile(expr);
        }

        // Multiple rules: combine with AND
        let mut combined = self.rules[0].1.inner.clone();
        for (_, expr) in &self.rules[1..] {
            combined = TLExpr::And(Box::new(combined), Box::new(expr.inner.clone()));
        }

        let combined_expr = PyTLExpr { inner: combined };
        crate::compiler::py_compile_with_config(&combined_expr, &compile_config)
    }

    /// Compile each rule separately
    ///
    /// Args:
    ///     config: Optional compilation config
    ///
    /// Returns:
    ///     Dictionary mapping rule names to EinsumGraph instances
    ///
    /// Example:
    ///     >>> rb = RuleBuilder()
    ///     >>> # ... define rules ...
    ///     >>> graphs = rb.compile_separate()
    ///     >>> graphs['transitivity']  # Access specific rule's graph
    #[pyo3(signature = (config=None))]
    fn compile_separate(
        &self,
        py: Python,
        config: Option<PyCompilationConfig>,
    ) -> PyResult<Py<PyAny>> {
        if self.rules.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No rules defined",
            ));
        }

        let compile_config = if let Some(cfg) = config {
            cfg
        } else if let Some(cfg) = self.config.clone() {
            cfg
        } else {
            PyCompilationConfig {
                inner: tensorlogic_compiler::CompilationConfig::default(),
            }
        };

        let dict = PyDict::new(py);
        for (name, expr) in &self.rules {
            let graph = crate::compiler::py_compile_with_config(expr, &compile_config)?;
            dict.set_item(name, graph)?;
        }

        Ok(dict.into())
    }

    /// Clear all rules and symbol table
    fn clear(&mut self) {
        self.rules.clear();
        self.symbol_table = PySymbolTable {
            inner: Default::default(),
        };
    }

    /// Get number of rules
    fn __len__(&self) -> usize {
        self.rules.len()
    }
}

/// Create a variable with optional domain
///
/// Args:
///     name: Variable name
///     domain: Optional domain name
///
/// Returns:
///     Var instance
///
/// Example:
///     >>> from pytensorlogic.dsl import var
///     >>> x = var("x", domain="Person")
#[pyfunction]
#[pyo3(signature = (name, domain=None))]
pub fn py_var_dsl(name: String, domain: Option<String>) -> PyVar {
    PyVar { name, domain }
}

/// Create a predicate builder
///
/// Args:
///     name: Predicate name
///     arity: Number of arguments (optional)
///     domains: Domain names for each argument (optional)
///
/// Returns:
///     PredicateBuilder instance
///
/// Example:
///     >>> from pytensorlogic.dsl import pred
///     >>> knows = pred("knows", arity=2, domains=["Person", "Person"])
#[pyfunction]
#[pyo3(signature = (name, arity=None, domains=None))]
pub fn py_pred_dsl(
    name: String,
    arity: Option<usize>,
    domains: Option<Vec<String>>,
) -> PyPredicateBuilder {
    PyPredicateBuilder {
        name,
        arity,
        domains,
    }
}

/// Create a rule builder
///
/// Args:
///     config: Optional compilation config
///
/// Returns:
///     RuleBuilder instance
///
/// Example:
///     >>> from pytensorlogic.dsl import rule_builder
///     >>> rb = rule_builder()
///     >>> x, y = rb.vars("x", "y", domain="Person")
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn py_rule_builder(config: Option<PyCompilationConfig>) -> PyRuleBuilder {
    PyRuleBuilder {
        rules: Vec::new(),
        symbol_table: PySymbolTable {
            inner: Default::default(),
        },
        config,
    }
}

/// Register DSL types and functions with the Python module
pub fn register_dsl_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyVar>()?;
    m.add_class::<PyPredicateBuilder>()?;
    m.add_class::<PyRuleBuilder>()?;

    // Register functions
    m.add_function(wrap_pyfunction!(py_var_dsl, m)?)?;
    m.add_function(wrap_pyfunction!(py_pred_dsl, m)?)?;
    m.add_function(wrap_pyfunction!(py_rule_builder, m)?)?;

    Ok(())
}
