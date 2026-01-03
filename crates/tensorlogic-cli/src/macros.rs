//! Macro system for defining reusable logical patterns
//!
//! This module provides a powerful macro system that allows users to define
//! parameterized logical patterns that can be reused throughout their expressions.
//!
//! # Syntax
//!
//! Macros are defined using the following syntax:
//!
//! ```text
//! DEFINE MACRO name(param1, param2, ...) = expression
//! ```
//!
//! # Examples
//!
//! ```text
//! // Define a transitive relation macro
//! DEFINE MACRO transitive(R, x, z) = EXISTS y. (R(x, y) AND R(y, z))
//!
//! // Define a symmetric relation macro
//! DEFINE MACRO symmetric(R, x, y) = R(x, y) AND R(y, x)
//!
//! // Define a reflexive relation macro
//! DEFINE MACRO reflexive(R, x) = R(x, x)
//!
//! // Define an equivalence relation macro
//! DEFINE MACRO equivalence(R, x, y) = reflexive(R, x) AND reflexive(R, y) AND symmetric(R, x, y)
//!
//! // Use macros in expressions
//! transitive(friend, Alice, Bob)
//! ```

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A macro definition with parameters and body
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MacroDef {
    /// Macro name
    pub name: String,

    /// Parameter names
    pub params: Vec<String>,

    /// Macro body (as expression string)
    pub body: String,
}

impl MacroDef {
    /// Create a new macro definition
    pub fn new(name: String, params: Vec<String>, body: String) -> Self {
        Self { name, params, body }
    }

    /// Get the arity (number of parameters)
    pub fn arity(&self) -> usize {
        self.params.len()
    }

    /// Validate that the macro definition is well-formed
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow!("Macro name cannot be empty"));
        }

        if !self.name.chars().next().unwrap().is_alphabetic() {
            return Err(anyhow!(
                "Macro name must start with a letter: {}",
                self.name
            ));
        }

        if self.params.is_empty() {
            return Err(anyhow!("Macro must have at least one parameter"));
        }

        // Check for duplicate parameters
        let mut seen = HashMap::new();
        for (idx, param) in self.params.iter().enumerate() {
            if let Some(prev_idx) = seen.insert(param, idx) {
                return Err(anyhow!(
                    "Duplicate parameter '{}' at positions {} and {}",
                    param,
                    prev_idx,
                    idx
                ));
            }
        }

        if self.body.is_empty() {
            return Err(anyhow!("Macro body cannot be empty"));
        }

        Ok(())
    }

    /// Expand the macro with the given arguments
    pub fn expand(&self, args: &[String]) -> Result<String> {
        if args.len() != self.params.len() {
            return Err(anyhow!(
                "Macro {} expects {} arguments, got {}",
                self.name,
                self.params.len(),
                args.len()
            ));
        }

        // Create substitution map
        let mut substitutions: HashMap<&str, &str> = HashMap::new();
        for (param, arg) in self.params.iter().zip(args.iter()) {
            substitutions.insert(param.as_str(), arg.as_str());
        }

        // Perform substitution in the body
        let mut result = self.body.clone();

        // Sort parameters by length (descending) to handle overlapping names correctly
        let mut sorted_params: Vec<&String> = self.params.iter().collect();
        sorted_params.sort_by_key(|p| std::cmp::Reverse(p.len()));

        for param in sorted_params {
            if let Some(arg) = substitutions.get(param.as_str()) {
                // Use word boundaries to avoid partial replacements
                result = replace_word(&result, param, arg);
            }
        }

        Ok(result)
    }
}

/// Replace whole words only (not substrings)
fn replace_word(text: &str, from: &str, to: &str) -> String {
    let mut result = String::new();
    let mut current_word = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current_word.push(ch);
        } else {
            if current_word == from {
                result.push_str(to);
            } else {
                result.push_str(&current_word);
            }
            current_word.clear();
            result.push(ch);
        }
    }

    // Handle final word
    if current_word == from {
        result.push_str(to);
    } else {
        result.push_str(&current_word);
    }

    result
}

/// Registry for managing macro definitions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MacroRegistry {
    /// Map of macro name to definition
    macros: HashMap<String, MacroDef>,
}

impl MacroRegistry {
    /// Create a new empty macro registry
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
        }
    }

    /// Create a registry with built-in macros
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Add common built-in macros
        let builtins = vec![
            MacroDef::new(
                "transitive".to_string(),
                vec!["R".to_string(), "x".to_string(), "z".to_string()],
                "EXISTS y. (R(x, y) AND R(y, z))".to_string(),
            ),
            MacroDef::new(
                "symmetric".to_string(),
                vec!["R".to_string(), "x".to_string(), "y".to_string()],
                "R(x, y) AND R(y, x)".to_string(),
            ),
            MacroDef::new(
                "reflexive".to_string(),
                vec!["R".to_string(), "x".to_string()],
                "R(x, x)".to_string(),
            ),
            MacroDef::new(
                "antisymmetric".to_string(),
                vec!["R".to_string(), "x".to_string(), "y".to_string()],
                "(R(x, y) AND R(y, x)) IMPLIES (x = y)".to_string(),
            ),
            MacroDef::new(
                "total".to_string(),
                vec!["R".to_string(), "x".to_string(), "y".to_string()],
                "R(x, y) OR R(y, x)".to_string(),
            ),
        ];

        for macro_def in builtins {
            let _ = registry.define(macro_def);
        }

        registry
    }

    /// Define a new macro
    pub fn define(&mut self, macro_def: MacroDef) -> Result<()> {
        macro_def.validate()?;
        self.macros.insert(macro_def.name.clone(), macro_def);
        Ok(())
    }

    /// Get a macro definition by name
    pub fn get(&self, name: &str) -> Option<&MacroDef> {
        self.macros.get(name)
    }

    /// Check if a macro is defined
    pub fn contains(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Remove a macro definition
    pub fn undefine(&mut self, name: &str) -> Option<MacroDef> {
        self.macros.remove(name)
    }

    /// List all defined macros
    pub fn list(&self) -> Vec<&MacroDef> {
        self.macros.values().collect()
    }

    /// Clear all macros
    pub fn clear(&mut self) {
        self.macros.clear();
    }

    /// Get the number of defined macros
    pub fn len(&self) -> usize {
        self.macros.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.macros.is_empty()
    }

    /// Expand a macro call
    pub fn expand(&self, name: &str, args: &[String]) -> Result<String> {
        let macro_def = self
            .get(name)
            .ok_or_else(|| anyhow!("Undefined macro: {}", name))?;
        macro_def.expand(args)
    }

    /// Recursively expand all macros in an expression
    pub fn expand_all(&self, expr: &str) -> Result<String> {
        let mut result = expr.to_string();
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100; // Prevent infinite loops

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Try to find and expand macro calls
            for (name, macro_def) in &self.macros {
                if let Some(expanded) = self.try_expand_macro(&result, name, macro_def)? {
                    result = expanded;
                    changed = true;
                    break; // Start over to ensure proper nesting
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            return Err(anyhow!(
                "Macro expansion exceeded maximum iterations (possible circular definition)"
            ));
        }

        Ok(result)
    }

    /// Try to expand a specific macro in the expression
    fn try_expand_macro(
        &self,
        expr: &str,
        name: &str,
        macro_def: &MacroDef,
    ) -> Result<Option<String>> {
        // Simple pattern matching for macro calls: name(arg1, arg2, ...)
        if let Some(pos) = expr.find(name) {
            // Check if this is actually a macro call (followed by '(')
            let after_name = pos + name.len();
            if after_name < expr.len() && expr.chars().nth(after_name) == Some('(') {
                // Extract arguments
                if let Some(args) = self.extract_args(&expr[after_name..])? {
                    let expanded = macro_def.expand(&args)?;
                    let mut result = String::new();
                    result.push_str(&expr[..pos]);
                    result.push_str(&expanded);
                    result.push_str(
                        &expr[after_name + self.find_matching_paren(&expr[after_name..])? + 1..],
                    );
                    return Ok(Some(result));
                }
            }
        }
        Ok(None)
    }

    /// Extract arguments from a function call
    fn extract_args(&self, text: &str) -> Result<Option<Vec<String>>> {
        if !text.starts_with('(') {
            return Ok(None);
        }

        let closing = self.find_matching_paren(text)?;
        let args_str = &text[1..closing];

        if args_str.trim().is_empty() {
            return Ok(Some(Vec::new()));
        }

        // Split by commas (respecting nested parentheses)
        let mut args = Vec::new();
        let mut current_arg = String::new();
        let mut depth = 0;

        for ch in args_str.chars() {
            match ch {
                '(' => {
                    depth += 1;
                    current_arg.push(ch);
                }
                ')' => {
                    depth -= 1;
                    current_arg.push(ch);
                }
                ',' if depth == 0 => {
                    args.push(current_arg.trim().to_string());
                    current_arg.clear();
                }
                _ => {
                    current_arg.push(ch);
                }
            }
        }

        if !current_arg.is_empty() {
            args.push(current_arg.trim().to_string());
        }

        Ok(Some(args))
    }

    /// Find the position of the matching closing parenthesis
    fn find_matching_paren(&self, text: &str) -> Result<usize> {
        let mut depth = 0;
        for (i, ch) in text.chars().enumerate() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(i);
                    }
                }
                _ => {}
            }
        }
        Err(anyhow!("Unmatched parenthesis"))
    }
}

/// Parse a macro definition from a string
///
/// Expected format: `DEFINE MACRO name(param1, param2, ...) = body`
pub fn parse_macro_definition(input: &str) -> Result<MacroDef> {
    let input = input.trim();

    // Check for DEFINE MACRO prefix
    if !input.starts_with("DEFINE MACRO") && !input.starts_with("MACRO") {
        return Err(anyhow!(
            "Macro definition must start with 'DEFINE MACRO' or 'MACRO'"
        ));
    }

    let input = if let Some(stripped) = input.strip_prefix("DEFINE MACRO") {
        stripped
    } else if let Some(stripped) = input.strip_prefix("MACRO") {
        stripped
    } else {
        unreachable!("Already checked for prefixes above")
    }
    .trim();

    // Find the equals sign
    let eq_pos = input
        .find('=')
        .ok_or_else(|| anyhow!("Macro definition must contain '='"))?;

    let signature = input[..eq_pos].trim();
    let body = input[eq_pos + 1..].trim().to_string();

    // Parse signature: name(param1, param2, ...)
    let open_paren = signature
        .find('(')
        .ok_or_else(|| anyhow!("Macro definition must have parameter list"))?;

    let name = signature[..open_paren].trim().to_string();

    let close_paren = signature
        .rfind(')')
        .ok_or_else(|| anyhow!("Unmatched parenthesis in macro signature"))?;

    let params_str = &signature[open_paren + 1..close_paren];
    let params: Vec<String> = if params_str.trim().is_empty() {
        return Err(anyhow!("Macro must have at least one parameter"));
    } else {
        params_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    Ok(MacroDef::new(name, params, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_def_creation() {
        let macro_def = MacroDef::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            "pred(x, y)".to_string(),
        );
        assert_eq!(macro_def.name, "test");
        assert_eq!(macro_def.arity(), 2);
    }

    #[test]
    fn test_macro_validation() {
        let valid = MacroDef::new(
            "test".to_string(),
            vec!["x".to_string()],
            "pred(x)".to_string(),
        );
        assert!(valid.validate().is_ok());

        let invalid_name =
            MacroDef::new("".to_string(), vec!["x".to_string()], "pred(x)".to_string());
        assert!(invalid_name.validate().is_err());

        let duplicate_params = MacroDef::new(
            "test".to_string(),
            vec!["x".to_string(), "x".to_string()],
            "pred(x)".to_string(),
        );
        assert!(duplicate_params.validate().is_err());
    }

    #[test]
    fn test_macro_expansion() {
        let macro_def = MacroDef::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            "pred(x, y) AND pred(y, x)".to_string(),
        );

        let expanded = macro_def
            .expand(&["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(expanded, "pred(a, b) AND pred(b, a)");
    }

    #[test]
    fn test_macro_registry() {
        let mut registry = MacroRegistry::new();

        let macro_def = MacroDef::new(
            "test".to_string(),
            vec!["x".to_string()],
            "pred(x)".to_string(),
        );

        registry.define(macro_def).unwrap();
        assert!(registry.contains("test"));
        assert_eq!(registry.len(), 1);

        let expanded = registry.expand("test", &["a".to_string()]).unwrap();
        assert_eq!(expanded, "pred(a)");
    }

    #[test]
    fn test_builtin_macros() {
        let registry = MacroRegistry::with_builtins();
        assert!(registry.contains("transitive"));
        assert!(registry.contains("symmetric"));
        assert!(registry.contains("reflexive"));
    }

    #[test]
    fn test_parse_macro_definition() {
        let input = "DEFINE MACRO test(x, y) = pred(x, y)";
        let macro_def = parse_macro_definition(input).unwrap();
        assert_eq!(macro_def.name, "test");
        assert_eq!(macro_def.params, vec!["x", "y"]);
        assert_eq!(macro_def.body, "pred(x, y)");
    }

    #[test]
    fn test_replace_word() {
        assert_eq!(replace_word("x + y", "x", "a"), "a + y");
        assert_eq!(replace_word("xyz", "x", "a"), "xyz"); // Shouldn't replace
        assert_eq!(replace_word("x(x, x)", "x", "a"), "a(a, a)");
    }

    #[test]
    fn test_macro_expansion_recursive() {
        let mut registry = MacroRegistry::new();

        let transitive = MacroDef::new(
            "trans".to_string(),
            vec!["R".to_string(), "x".to_string(), "z".to_string()],
            "EXISTS y. (R(x, y) AND R(y, z))".to_string(),
        );
        registry.define(transitive).unwrap();

        let expr = "trans(friend, Alice, Bob)";
        let expanded = registry.expand_all(expr).unwrap();
        assert!(expanded.contains("EXISTS y"));
        assert!(expanded.contains("friend(Alice, y)"));
        assert!(expanded.contains("friend(y, Bob)"));
    }
}
