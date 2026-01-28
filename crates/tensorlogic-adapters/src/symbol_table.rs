//! Symbol table for managing domains, predicates, and variables.

use anyhow::{bail, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tensorlogic_ir::TLExpr;

use crate::error::AdapterError;
use crate::{DomainInfo, PredicateInfo};

/// Symbol table containing all domain, predicate, and variable information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolTable {
    pub domains: IndexMap<String, DomainInfo>,
    pub predicates: IndexMap<String, PredicateInfo>,
    pub variables: IndexMap<String, String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            domains: IndexMap::new(),
            predicates: IndexMap::new(),
            variables: IndexMap::new(),
        }
    }

    pub fn add_domain(&mut self, domain: DomainInfo) -> Result<()> {
        self.domains.insert(domain.name.clone(), domain);
        Ok(())
    }

    pub fn add_predicate(&mut self, predicate: PredicateInfo) -> Result<()> {
        for domain in &predicate.arg_domains {
            if !self.domains.contains_key(domain) {
                bail!(
                    "Domain '{}' referenced by predicate '{}' does not exist",
                    domain,
                    predicate.name
                );
            }
        }
        self.predicates.insert(predicate.name.clone(), predicate);
        Ok(())
    }

    pub fn bind_variable(
        &mut self,
        var: impl Into<String>,
        domain: impl Into<String>,
    ) -> Result<()> {
        let var = var.into();
        let domain = domain.into();

        if !self.domains.contains_key(&domain) {
            return Err(AdapterError::DomainNotFound(domain).into());
        }

        self.variables.insert(var, domain);
        Ok(())
    }

    pub fn get_domain(&self, name: &str) -> Option<&DomainInfo> {
        self.domains.get(name)
    }

    pub fn get_predicate(&self, name: &str) -> Option<&PredicateInfo> {
        self.predicates.get(name)
    }

    pub fn get_variable_domain(&self, var: &str) -> Option<&str> {
        self.variables.get(var).map(|s| s.as_str())
    }

    pub fn infer_from_expr(&mut self, expr: &TLExpr) -> Result<()> {
        self.collect_domains_from_expr(expr)?;
        self.collect_predicates_from_expr(expr)?;
        Ok(())
    }

    fn collect_domains_from_expr(&mut self, expr: &TLExpr) -> Result<()> {
        match expr {
            TLExpr::Exists { domain, var, body } | TLExpr::ForAll { domain, var, body } => {
                if !self.domains.contains_key(domain) {
                    self.add_domain(DomainInfo::new(domain.clone(), 0))?;
                }
                self.bind_variable(var, domain)?;
                self.collect_domains_from_expr(body)?;
            }
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Imply(l, r)
            | TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                self.collect_domains_from_expr(l)?;
                self.collect_domains_from_expr(r)?;
            }
            TLExpr::Not(e)
            | TLExpr::Score(e)
            | TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e) => {
                self.collect_domains_from_expr(e)?;
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_domains_from_expr(condition)?;
                self.collect_domains_from_expr(then_branch)?;
                self.collect_domains_from_expr(else_branch)?;
            }
            TLExpr::Aggregate {
                domain, var, body, ..
            } => {
                if !self.domains.contains_key(domain) {
                    self.add_domain(DomainInfo::new(domain.clone(), 0))?;
                }
                self.bind_variable(var, domain)?;
                self.collect_domains_from_expr(body)?;
            }
            TLExpr::Let {
                var: _,
                value,
                body,
            } => {
                self.collect_domains_from_expr(value)?;
                // The variable binding is temporary, so we don't add it to the symbol table
                self.collect_domains_from_expr(body)?;
            }
            // Modal/temporal logic operators (future enhancement)
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => {
                self.collect_domains_from_expr(inner)?;
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
                self.collect_domains_from_expr(before)?;
                self.collect_domains_from_expr(after)?;
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                self.collect_domains_from_expr(left)?;
                self.collect_domains_from_expr(right)?;
            }
            TLExpr::FuzzyNot { expr, .. } => {
                self.collect_domains_from_expr(expr)?;
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                self.collect_domains_from_expr(premise)?;
                self.collect_domains_from_expr(conclusion)?;
            }
            TLExpr::SoftExists {
                domain, var, body, ..
            }
            | TLExpr::SoftForAll {
                domain, var, body, ..
            } => {
                if !self.domains.contains_key(domain) {
                    self.add_domain(DomainInfo::new(domain.clone(), 0))?;
                }
                self.bind_variable(var, domain)?;
                self.collect_domains_from_expr(body)?;
            }
            TLExpr::WeightedRule { rule, .. } => {
                self.collect_domains_from_expr(rule)?;
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_prob, expr) in alternatives {
                    self.collect_domains_from_expr(expr)?;
                }
            }
            // Counting quantifiers
            TLExpr::CountingExists {
                domain, var, body, ..
            }
            | TLExpr::CountingForAll {
                domain, var, body, ..
            }
            | TLExpr::ExactCount {
                domain, var, body, ..
            }
            | TLExpr::Majority { domain, var, body } => {
                if !self.domains.contains_key(domain) {
                    self.add_domain(DomainInfo::new(domain.clone(), 0))?;
                }
                self.bind_variable(var, domain)?;
                self.collect_domains_from_expr(body)?;
            }
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {}
            // All other expression types (enhancements) - don't introduce new domains
            _ => {
                // For now, skip domain collection for unimplemented expression types
                // This allows the code to compile while features are being implemented
            }
        }
        Ok(())
    }

    fn collect_predicates_from_expr(&mut self, expr: &TLExpr) -> Result<()> {
        match expr {
            TLExpr::Pred { name, args } => {
                if !self.predicates.contains_key(name) {
                    let arg_domains: Vec<String> =
                        args.iter().map(|_| "Unknown".to_string()).collect();
                    self.predicates
                        .insert(name.clone(), PredicateInfo::new(name.clone(), arg_domains));
                }
            }
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Imply(l, r)
            | TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                self.collect_predicates_from_expr(l)?;
                self.collect_predicates_from_expr(r)?;
            }
            TLExpr::Not(e)
            | TLExpr::Score(e)
            | TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e) => {
                self.collect_predicates_from_expr(e)?;
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                self.collect_predicates_from_expr(body)?;
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_predicates_from_expr(condition)?;
                self.collect_predicates_from_expr(then_branch)?;
                self.collect_predicates_from_expr(else_branch)?;
            }
            TLExpr::Aggregate { body, .. } => {
                self.collect_predicates_from_expr(body)?;
            }
            TLExpr::Let { value, body, .. } => {
                self.collect_predicates_from_expr(value)?;
                self.collect_predicates_from_expr(body)?;
            }
            // Modal/temporal logic operators (future enhancement)
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => {
                self.collect_predicates_from_expr(inner)?;
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
                self.collect_predicates_from_expr(before)?;
                self.collect_predicates_from_expr(after)?;
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                self.collect_predicates_from_expr(left)?;
                self.collect_predicates_from_expr(right)?;
            }
            TLExpr::FuzzyNot { expr, .. } => {
                self.collect_predicates_from_expr(expr)?;
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                self.collect_predicates_from_expr(premise)?;
                self.collect_predicates_from_expr(conclusion)?;
            }
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                self.collect_predicates_from_expr(body)?;
            }
            TLExpr::WeightedRule { rule, .. } => {
                self.collect_predicates_from_expr(rule)?;
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_prob, expr) in alternatives {
                    self.collect_predicates_from_expr(expr)?;
                }
            }
            // Counting quantifiers
            TLExpr::CountingExists { body, .. }
            | TLExpr::CountingForAll { body, .. }
            | TLExpr::ExactCount { body, .. }
            | TLExpr::Majority { body, .. } => {
                self.collect_predicates_from_expr(body)?;
            }
            TLExpr::Constant(_) => {}
            // All other expression types (enhancements) - don't introduce predicates
            _ => {
                // For now, skip predicate collection for unimplemented expression types
                // This allows the code to compile while features are being implemented
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn to_yaml(&self) -> Result<String> {
        Ok(serde_yaml::to_string(self)?)
    }

    pub fn from_yaml(yaml: &str) -> Result<Self> {
        Ok(serde_yaml::from_str(yaml)?)
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
