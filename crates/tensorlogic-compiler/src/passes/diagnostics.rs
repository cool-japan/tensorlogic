//! Enhanced error diagnostics and messaging.

use anyhow::{anyhow, Result};
use tensorlogic_ir::{IrError, SourceSpan, TLExpr, Term};

use super::scope_analysis::analyze_scopes;

/// Diagnostic level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Hint,
}

/// A diagnostic message with location and context
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<SourceSpan>,
    pub help: Option<String>,
    pub related: Vec<(String, Option<SourceSpan>)>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Diagnostic {
            level: DiagnosticLevel::Error,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Diagnostic {
            level: DiagnosticLevel::Warning,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_related(mut self, msg: impl Into<String>, span: Option<SourceSpan>) -> Self {
        self.related.push((msg.into(), span));
        self
    }

    /// Format the diagnostic for display
    pub fn format(&self) -> String {
        let mut output = String::new();

        let level_str = match self.level {
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Info => "info",
            DiagnosticLevel::Hint => "hint",
        };

        if let Some(ref span) = self.span {
            output.push_str(&format!("{}: {}: {}\n", level_str, span, self.message));
        } else {
            output.push_str(&format!("{}: {}\n", level_str, self.message));
        }

        if let Some(ref help) = self.help {
            output.push_str(&format!("  help: {}\n", help));
        }

        for (msg, span_opt) in &self.related {
            if let Some(span) = span_opt {
                output.push_str(&format!("  note: {}: {}\n", span, msg));
            } else {
                output.push_str(&format!("  note: {}\n", msg));
            }
        }

        output
    }
}

/// Enhanced error message builder
pub struct DiagnosticBuilder {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBuilder {
    pub fn new() -> Self {
        DiagnosticBuilder {
            diagnostics: Vec::new(),
        }
    }

    pub fn add(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .count()
    }

    pub fn into_result(self) -> Result<()> {
        if self.has_errors() {
            let mut msg = String::new();
            for diag in &self.diagnostics {
                msg.push_str(&diag.format());
            }
            Err(anyhow!("{}", msg))
        } else {
            Ok(())
        }
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl Default for DiagnosticBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhance an IrError with better diagnostics
pub fn enhance_error(error: IrError) -> Diagnostic {
    match error {
        IrError::ArityMismatch {
            name,
            expected,
            actual,
        } => Diagnostic::error(format!(
            "Predicate '{}' arity mismatch: expected {} arguments, got {}",
            name, expected, actual
        ))
        .with_help(format!(
            "Change the number of arguments to match the expected arity of {}",
            expected
        )),
        IrError::TypeMismatch {
            name,
            arg_index,
            expected,
            actual,
        } => Diagnostic::error(format!(
            "Type mismatch in predicate '{}' at argument {}: expected '{}', got '{}'",
            name, arg_index, expected, actual
        ))
        .with_help(format!(
            "Change argument {} to have type '{}'",
            arg_index, expected
        )),
        IrError::UnboundVariable { var } => {
            Diagnostic::error(format!("Variable '{}' is not bound by any quantifier", var))
                .with_help(format!(
                    "Add a quantifier: ∀{}. <expr> or ∃{}. <expr>",
                    var, var
                ))
        }
        IrError::InconsistentTypes { var, type1, type2 } => Diagnostic::error(format!(
            "Variable '{}' used with inconsistent types: '{}' and '{}'",
            var, type1, type2
        ))
        .with_help("Ensure the variable has the same type in all uses".to_string()),
        _ => Diagnostic::error(format!("{}", error)),
    }
}

/// Generate diagnostics for an expression
pub fn diagnose_expression(expr: &TLExpr) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Check for unbound variables
    if let Ok(scope_result) = analyze_scopes(expr) {
        for unbound_var in &scope_result.unbound_variables {
            let diag = Diagnostic::error(format!("Unbound variable '{}'", unbound_var)).with_help(
                format!(
                    "Consider adding a universal quantifier: ∀{}. <expr>",
                    unbound_var
                ),
            );
            diagnostics.push(diag);
        }

        // Check for type conflicts
        for conflict in &scope_result.type_conflicts {
            let diag = Diagnostic::error(format!(
                "Variable '{}' has conflicting types: '{}' and '{}'",
                conflict.variable, conflict.type1, conflict.type2
            ))
            .with_help("Ensure the variable has consistent types across all uses".to_string());
            diagnostics.push(diag);
        }
    }

    // Warn about variables bound but never used
    diagnose_unused_bindings(expr, &mut diagnostics);

    diagnostics
}

fn diagnose_unused_bindings(expr: &TLExpr, diagnostics: &mut Vec<Diagnostic>) {
    match expr {
        TLExpr::Exists {
            var,
            domain: _,
            body,
        }
        | TLExpr::ForAll {
            var,
            domain: _,
            body,
        }
        | TLExpr::SoftExists {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::SoftForAll {
            var,
            domain: _,
            body,
            ..
        }
        | TLExpr::Aggregate {
            var,
            domain: _,
            body,
            ..
        } => {
            // Check if variable is actually used in the body
            if !uses_variable(body, var) {
                diagnostics.push(
                    Diagnostic::warning(format!("Variable '{}' is bound but never used", var))
                        .with_help(format!("Consider removing the quantifier for '{}'", var)),
                );
            }
            diagnose_unused_bindings(body, diagnostics);
        }
        TLExpr::And(left, right)
        | TLExpr::Or(left, right)
        | TLExpr::Imply(left, right)
        | TLExpr::Add(left, right)
        | TLExpr::Sub(left, right)
        | TLExpr::Mul(left, right)
        | TLExpr::Div(left, right)
        | TLExpr::Pow(left, right)
        | TLExpr::Mod(left, right)
        | TLExpr::Min(left, right)
        | TLExpr::Max(left, right)
        | TLExpr::Eq(left, right)
        | TLExpr::Lt(left, right)
        | TLExpr::Gt(left, right)
        | TLExpr::Lte(left, right)
        | TLExpr::Gte(left, right)
        | TLExpr::TNorm { left, right, .. }
        | TLExpr::TCoNorm { left, right, .. }
        | TLExpr::FuzzyImplication {
            premise: left,
            conclusion: right,
            ..
        } => {
            diagnose_unused_bindings(left, diagnostics);
            diagnose_unused_bindings(right, diagnostics);
        }
        TLExpr::Not(inner)
        | TLExpr::Score(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::FuzzyNot { expr: inner, .. }
        | TLExpr::WeightedRule { rule: inner, .. } => {
            diagnose_unused_bindings(inner, diagnostics);
        }
        TLExpr::Let {
            var: _,
            value,
            body,
        } => {
            diagnose_unused_bindings(value, diagnostics);
            diagnose_unused_bindings(body, diagnostics);
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            diagnose_unused_bindings(condition, diagnostics);
            diagnose_unused_bindings(then_branch, diagnostics);
            diagnose_unused_bindings(else_branch, diagnostics);
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => {
            diagnose_unused_bindings(inner, diagnostics);
        }
        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            diagnose_unused_bindings(before, diagnostics);
            diagnose_unused_bindings(after, diagnostics);
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_weight, alt_expr) in alternatives {
                diagnose_unused_bindings(alt_expr, diagnostics);
            }
        }

        TLExpr::Pred { .. } => {}
        TLExpr::Constant(_) => {}
        // All other expression types (enhancements)
        _ => {}
    }
}

fn uses_variable(expr: &TLExpr, var_name: &str) -> bool {
    match expr {
        TLExpr::Pred { name: _, args } => args.iter().any(|term| match term {
            Term::Var(v) => v == var_name,
            Term::Typed { value, .. } => uses_variable_in_term(value, var_name),
            _ => false,
        }),
        TLExpr::And(left, right)
        | TLExpr::Or(left, right)
        | TLExpr::Imply(left, right)
        | TLExpr::Add(left, right)
        | TLExpr::Sub(left, right)
        | TLExpr::Mul(left, right)
        | TLExpr::Div(left, right)
        | TLExpr::Pow(left, right)
        | TLExpr::Mod(left, right)
        | TLExpr::Min(left, right)
        | TLExpr::Max(left, right)
        | TLExpr::Eq(left, right)
        | TLExpr::Lt(left, right)
        | TLExpr::Gt(left, right)
        | TLExpr::Lte(left, right)
        | TLExpr::Gte(left, right)
        | TLExpr::TNorm { left, right, .. }
        | TLExpr::TCoNorm { left, right, .. }
        | TLExpr::FuzzyImplication {
            premise: left,
            conclusion: right,
            ..
        } => uses_variable(left, var_name) || uses_variable(right, var_name),
        TLExpr::Not(inner)
        | TLExpr::Score(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::FuzzyNot { expr: inner, .. }
        | TLExpr::WeightedRule { rule: inner, .. } => uses_variable(inner, var_name),
        TLExpr::Let {
            var: _,
            value,
            body,
        } => uses_variable(value, var_name) || uses_variable(body, var_name),
        TLExpr::Exists {
            var: _,
            domain: _,
            body,
        }
        | TLExpr::ForAll {
            var: _,
            domain: _,
            body,
        }
        | TLExpr::SoftExists {
            var: _,
            domain: _,
            body,
            ..
        }
        | TLExpr::SoftForAll {
            var: _,
            domain: _,
            body,
            ..
        }
        | TLExpr::Aggregate {
            var: _,
            domain: _,
            body,
            ..
        } => uses_variable(body, var_name),
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            uses_variable(condition, var_name)
                || uses_variable(then_branch, var_name)
                || uses_variable(else_branch, var_name)
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => uses_variable(inner, var_name),
        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => uses_variable(before, var_name) || uses_variable(after, var_name),
        TLExpr::ProbabilisticChoice { alternatives } => alternatives
            .iter()
            .any(|(_weight, alt_expr)| uses_variable(alt_expr, var_name)),

        TLExpr::Constant(_) => false,
        // All other expression types (enhancements)
        _ => false,
    }
}

fn uses_variable_in_term(term: &Term, var_name: &str) -> bool {
    match term {
        Term::Var(v) => v == var_name,
        Term::Typed { value, .. } => uses_variable_in_term(value, var_name),
        _ => false,
    }
}

/// Pretty-print an expression for error messages
pub fn pretty_print_expr(expr: &TLExpr) -> String {
    match expr {
        TLExpr::Pred { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let args_str = args
                    .iter()
                    .map(pretty_print_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", name, args_str)
            }
        }
        TLExpr::And(left, right) => {
            format!(
                "({} ∧ {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Or(left, right) => {
            format!(
                "({} ∨ {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Not(inner) => format!("¬{}", pretty_print_expr(inner)),
        TLExpr::Imply(premise, conclusion) => {
            format!(
                "({} → {})",
                pretty_print_expr(premise),
                pretty_print_expr(conclusion)
            )
        }
        TLExpr::Exists { var, domain, body } => {
            format!("∃{}:{}. {}", var, domain, pretty_print_expr(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            format!("∀{}:{}. {}", var, domain, pretty_print_expr(body))
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let group_str = if let Some(gb) = group_by {
                format!(" GROUP BY {}", gb.join(", "))
            } else {
                String::new()
            };
            format!(
                "AGG[{}]({}:{}. {}){}",
                op,
                var,
                domain,
                pretty_print_expr(body),
                group_str
            )
        }
        TLExpr::Score(inner) => format!("score({})", pretty_print_expr(inner)),
        TLExpr::Add(left, right) => {
            format!(
                "({} + {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Sub(left, right) => {
            format!(
                "({} - {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Mul(left, right) => {
            format!(
                "({} * {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Div(left, right) => {
            format!(
                "({} / {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Eq(left, right) => {
            format!(
                "({} = {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Lt(left, right) => {
            format!(
                "({} < {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Gt(left, right) => {
            format!(
                "({} > {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Lte(left, right) => {
            format!(
                "({} ≤ {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Gte(left, right) => {
            format!(
                "({} ≥ {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Pow(left, right) => {
            format!(
                "({} ^ {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Mod(left, right) => {
            format!(
                "({} % {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Min(left, right) => {
            format!(
                "min({}, {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Max(left, right) => {
            format!(
                "max({}, {})",
                pretty_print_expr(left),
                pretty_print_expr(right)
            )
        }
        TLExpr::Abs(inner) => format!("abs({})", pretty_print_expr(inner)),
        TLExpr::Floor(inner) => format!("floor({})", pretty_print_expr(inner)),
        TLExpr::Ceil(inner) => format!("ceil({})", pretty_print_expr(inner)),
        TLExpr::Round(inner) => format!("round({})", pretty_print_expr(inner)),
        TLExpr::Sqrt(inner) => format!("sqrt({})", pretty_print_expr(inner)),
        TLExpr::Exp(inner) => format!("exp({})", pretty_print_expr(inner)),
        TLExpr::Log(inner) => format!("log({})", pretty_print_expr(inner)),
        TLExpr::Sin(inner) => format!("sin({})", pretty_print_expr(inner)),
        TLExpr::Cos(inner) => format!("cos({})", pretty_print_expr(inner)),
        TLExpr::Tan(inner) => format!("tan({})", pretty_print_expr(inner)),
        TLExpr::Let { var, value, body } => {
            format!(
                "let {} = {} in {}",
                var,
                pretty_print_expr(value),
                pretty_print_expr(body)
            )
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            format!(
                "if {} then {} else {}",
                pretty_print_expr(condition),
                pretty_print_expr(then_branch),
                pretty_print_expr(else_branch)
            )
        }

        // Modal/temporal logic operators
        TLExpr::Box(inner) => format!("□{}", pretty_print_expr(inner)),
        TLExpr::Diamond(inner) => format!("◇{}", pretty_print_expr(inner)),
        TLExpr::Next(inner) => format!("X{}", pretty_print_expr(inner)),
        TLExpr::Eventually(inner) => format!("F{}", pretty_print_expr(inner)),
        TLExpr::Always(inner) => format!("G{}", pretty_print_expr(inner)),
        TLExpr::Until { before, after } => {
            format!(
                "({} U {})",
                pretty_print_expr(before),
                pretty_print_expr(after)
            )
        }

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            format!(
                "({} ⊗_{:?} {})",
                pretty_print_expr(left),
                kind,
                pretty_print_expr(right)
            )
        }
        TLExpr::TCoNorm { kind, left, right } => {
            format!(
                "({} ⊕_{:?} {})",
                pretty_print_expr(left),
                kind,
                pretty_print_expr(right)
            )
        }
        TLExpr::FuzzyNot { kind, expr } => {
            format!("¬_{:?}({})", kind, pretty_print_expr(expr))
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            format!(
                "({} →_{:?} {})",
                pretty_print_expr(premise),
                kind,
                pretty_print_expr(conclusion)
            )
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            format!(
                "∃{}:{}[T={}]. {}",
                var,
                domain,
                temperature,
                pretty_print_expr(body)
            )
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            format!(
                "∀{}:{}[T={}]. {}",
                var,
                domain,
                temperature,
                pretty_print_expr(body)
            )
        }
        TLExpr::WeightedRule { weight, rule } => {
            format!("{}::{}", weight, pretty_print_expr(rule))
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let alt_strs: Vec<String> = alternatives
                .iter()
                .map(|(prob, expr)| format!("{}:{}", prob, pretty_print_expr(expr)))
                .collect();
            format!("CHOICE[{}]", alt_strs.join(", "))
        }
        TLExpr::Release { released, releaser } => {
            format!(
                "({} R {})",
                pretty_print_expr(released),
                pretty_print_expr(releaser)
            )
        }
        TLExpr::WeakUntil { before, after } => {
            format!(
                "({} W {})",
                pretty_print_expr(before),
                pretty_print_expr(after)
            )
        }
        TLExpr::StrongRelease { released, releaser } => {
            format!(
                "({} M {})",
                pretty_print_expr(released),
                pretty_print_expr(releaser)
            )
        }

        TLExpr::Constant(value) => format!("{}", value),
        // All other expression types (enhancements)
        _ => "<expr>".to_string(),
    }
}

fn pretty_print_term(term: &Term) -> String {
    match term {
        Term::Var(v) => v.clone(),
        Term::Const(c) => c.clone(),
        Term::Typed {
            value,
            type_annotation,
        } => {
            format!("{}:{}", pretty_print_term(value), type_annotation.type_name)
        }
    }
}

/// Create a detailed error message for common compilation errors
pub fn create_detailed_error(
    error_type: &str,
    expr: &TLExpr,
    context: &str,
    suggestion: Option<&str>,
) -> Diagnostic {
    let expr_str = pretty_print_expr(expr);
    let truncated = if expr_str.len() > 100 {
        // Safely truncate at a character boundary
        let mut end = 97;
        while end > 0 && !expr_str.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &expr_str[..end])
    } else {
        expr_str
    };

    let mut diag = Diagnostic::error(format!("{}: {}", error_type, context))
        .with_related(format!("In expression: {}", truncated), None);

    if let Some(sugg) = suggestion {
        diag = diag.with_help(sugg.to_string());
    }

    diag
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::SourceLocation;

    #[test]
    fn test_diagnostic_creation() {
        let diag = Diagnostic::error("Test error")
            .with_help("Fix it like this")
            .with_related("Related info", None);

        assert_eq!(diag.level, DiagnosticLevel::Error);
        assert_eq!(diag.message, "Test error");
        assert!(diag.help.is_some());
        assert_eq!(diag.related.len(), 1);
    }

    #[test]
    fn test_diagnostic_format() {
        let diag = Diagnostic::error("Test error").with_help("Fix it");
        let formatted = diag.format();

        assert!(formatted.contains("error"));
        assert!(formatted.contains("Test error"));
        assert!(formatted.contains("help"));
    }

    #[test]
    fn test_diagnostic_with_span() {
        let span = SourceSpan::single(SourceLocation::new("test.tl", 10, 5));
        let diag = Diagnostic::error("Test error").with_span(span);

        let formatted = diag.format();
        assert!(formatted.contains("test.tl"));
        assert!(formatted.contains("10"));
    }

    #[test]
    fn test_diagnostic_builder() {
        let mut builder = DiagnosticBuilder::new();

        builder.add(Diagnostic::error("Error 1"));
        builder.add(Diagnostic::warning("Warning 1"));
        builder.add(Diagnostic::error("Error 2"));

        assert!(builder.has_errors());
        assert_eq!(builder.error_count(), 2);
        assert_eq!(builder.diagnostics().len(), 3);
    }

    #[test]
    fn test_diagnose_unbound_variable() {
        let expr = TLExpr::pred("p", vec![Term::var("x")]);

        let diagnostics = diagnose_expression(&expr);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
        assert!(diagnostics[0].message.contains("Unbound"));
    }

    #[test]
    fn test_diagnose_unused_binding() {
        let expr = TLExpr::exists(
            "x",
            "Domain",
            TLExpr::pred("p", vec![Term::var("y")]), // x is bound but y is used
        );

        let diagnostics = diagnose_expression(&expr);
        // Should have warnings about unused binding
        let warnings: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .collect();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_enhance_arity_error() {
        let error = IrError::ArityMismatch {
            name: "knows".to_string(),
            expected: 2,
            actual: 1,
        };

        let diag = enhance_error(error);
        assert_eq!(diag.level, DiagnosticLevel::Error);
        assert!(diag.message.contains("arity mismatch"));
        assert!(diag.help.is_some());
    }

    #[test]
    fn test_enhance_type_error() {
        let error = IrError::TypeMismatch {
            name: "knows".to_string(),
            arg_index: 1,
            expected: "Person".to_string(),
            actual: "Thing".to_string(),
        };

        let diag = enhance_error(error);
        assert!(diag.message.contains("Type mismatch"));
        assert!(diag.help.is_some());
    }

    #[test]
    fn test_pretty_print_predicate() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let pretty = pretty_print_expr(&expr);
        assert_eq!(pretty, "knows(x, y)");
    }

    #[test]
    fn test_pretty_print_quantifier() {
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        );
        let pretty = pretty_print_expr(&expr);
        assert!(pretty.contains("∃x:Person"));
        assert!(pretty.contains("knows(x, y)"));
    }

    #[test]
    fn test_pretty_print_complex() {
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::negate(TLExpr::pred("q", vec![Term::var("y")])),
        );
        let pretty = pretty_print_expr(&expr);
        assert!(pretty.contains("∧"));
        assert!(pretty.contains("¬"));
    }

    #[test]
    fn test_pretty_print_arithmetic() {
        let expr = TLExpr::add(
            TLExpr::pred("x", vec![]),
            TLExpr::mul(TLExpr::pred("y", vec![]), TLExpr::constant(2.0)),
        );
        let pretty = pretty_print_expr(&expr);
        assert!(pretty.contains("+"));
        assert!(pretty.contains("*"));
        assert!(pretty.contains("2"));
    }

    #[test]
    fn test_create_detailed_error() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let diag = create_detailed_error(
            "Compilation error",
            &expr,
            "Variable x is unbound",
            Some("Add a quantifier: ∃x. <expr>"),
        );

        assert_eq!(diag.level, DiagnosticLevel::Error);
        assert!(diag.message.contains("Compilation error"));
        assert!(!diag.related.is_empty());
        assert!(diag.help.is_some());
    }

    #[test]
    fn test_pretty_print_truncation() {
        // Create a very long expression
        let mut expr = TLExpr::pred("p", vec![Term::var("x")]);
        for _ in 0..10 {
            expr = TLExpr::and(expr.clone(), TLExpr::pred("q", vec![Term::var("y")]));
        }

        let diag = create_detailed_error("Test", &expr, "context", None);
        // Should truncate if too long
        let related_msg = &diag.related[0].0;
        assert!(related_msg.len() < 200); // Reasonable length
    }
}
