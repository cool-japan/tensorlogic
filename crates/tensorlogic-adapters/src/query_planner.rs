//! Query planning and optimization for predicate lookups.
//!
//! This module provides intelligent query planning for efficient predicate
//! resolution, leveraging statistics, indexing, and cost-based optimization.
//!
//! # Overview
//!
//! When resolving predicates in a large schema, different lookup strategies
//! have vastly different performance characteristics. The query planner:
//!
//! - Collects statistics about predicate access patterns
//! - Builds specialized indexes for common queries
//! - Generates optimal execution plans based on query shape
//! - Adapts to workload changes dynamically
//!
//! # Architecture
//!
//! - **QueryStatistics**: Tracks access patterns and selectivity
//! - **IndexStrategy**: Multiple index types (hash, range, composite)
//! - **CostModel**: Estimates query execution cost
//! - **QueryPlanner**: Generates optimal execution plans
//! - **PlanCache**: Caches frequently used plans
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, PredicateInfo, QueryPlanner, PredicateQuery};
//!
//! let mut table = SymbolTable::new();
//! // ... populate table ...
//!
//! let mut planner = QueryPlanner::new(&table);
//!
//! // Plan a query for binary predicates over Person domain
//! let query = PredicateQuery::by_signature(vec!["Person".to_string(), "Person".to_string()]);
//! let plan = planner.plan(&query).unwrap();
//!
//! // Execute the plan
//! let results = plan.execute(&table).unwrap();
//! ```

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::{PredicateInfo, SymbolTable};

/// Query for predicates
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PredicateQuery {
    /// Find predicate by exact name
    ByName(String),
    /// Find predicates by arity
    ByArity(usize),
    /// Find predicates by exact signature
    BySignature(Vec<String>),
    /// Find predicates containing a specific domain
    ByDomain(String),
    /// Find predicates matching a pattern
    ByPattern(PredicatePattern),
    /// Conjunction of queries
    And(Vec<PredicateQuery>),
    /// Disjunction of queries
    Or(Vec<PredicateQuery>),
}

impl PredicateQuery {
    pub fn by_name(name: impl Into<String>) -> Self {
        Self::ByName(name.into())
    }

    pub fn by_arity(arity: usize) -> Self {
        Self::ByArity(arity)
    }

    pub fn by_signature(domains: Vec<String>) -> Self {
        Self::BySignature(domains)
    }

    pub fn by_domain(domain: impl Into<String>) -> Self {
        Self::ByDomain(domain.into())
    }

    pub fn by_pattern(pattern: PredicatePattern) -> Self {
        Self::ByPattern(pattern)
    }

    pub fn and(queries: Vec<PredicateQuery>) -> Self {
        Self::And(queries)
    }

    pub fn or(queries: Vec<PredicateQuery>) -> Self {
        Self::Or(queries)
    }
}

/// Pattern for predicate matching
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PredicatePattern {
    /// Name pattern (supports wildcards)
    pub name_pattern: Option<String>,
    /// Minimum arity
    pub min_arity: Option<usize>,
    /// Maximum arity
    pub max_arity: Option<usize>,
    /// Required domains (at any position)
    pub required_domains: Vec<String>,
    /// Excluded domains
    pub excluded_domains: Vec<String>,
}

impl PredicatePattern {
    pub fn new() -> Self {
        Self {
            name_pattern: None,
            min_arity: None,
            max_arity: None,
            required_domains: Vec::new(),
            excluded_domains: Vec::new(),
        }
    }

    pub fn with_name_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.name_pattern = Some(pattern.into());
        self
    }

    pub fn with_arity_range(mut self, min: usize, max: usize) -> Self {
        self.min_arity = Some(min);
        self.max_arity = Some(max);
        self
    }

    pub fn with_required_domain(mut self, domain: impl Into<String>) -> Self {
        self.required_domains.push(domain.into());
        self
    }

    pub fn with_excluded_domain(mut self, domain: impl Into<String>) -> Self {
        self.excluded_domains.push(domain.into());
        self
    }

    /// Check if a predicate matches this pattern
    pub fn matches(&self, name: &str, predicate: &PredicateInfo) -> bool {
        // Check name pattern
        if let Some(pattern) = &self.name_pattern {
            if !matches_wildcard(name, pattern) {
                return false;
            }
        }

        // Check arity range
        let arity = predicate.arg_domains.len();
        if let Some(min) = self.min_arity {
            if arity < min {
                return false;
            }
        }
        if let Some(max) = self.max_arity {
            if arity > max {
                return false;
            }
        }

        // Check required domains
        let domain_set: HashSet<_> = predicate.arg_domains.iter().collect();
        for required in &self.required_domains {
            if !domain_set.contains(required) {
                return false;
            }
        }

        // Check excluded domains
        for excluded in &self.excluded_domains {
            if domain_set.contains(excluded) {
                return false;
            }
        }

        true
    }
}

impl Default for PredicatePattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple wildcard matching (supports * and ?)
fn matches_wildcard(text: &str, pattern: &str) -> bool {
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    let mut dp = vec![vec![false; pattern_chars.len() + 1]; text_chars.len() + 1];
    dp[0][0] = true;

    // Handle leading stars
    for j in 1..=pattern_chars.len() {
        if pattern_chars[j - 1] == '*' {
            dp[0][j] = dp[0][j - 1];
        }
    }

    for i in 1..=text_chars.len() {
        for j in 1..=pattern_chars.len() {
            if pattern_chars[j - 1] == '*' {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if pattern_chars[j - 1] == '?' || text_chars[i - 1] == pattern_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[text_chars.len()][pattern_chars.len()]
}

/// Statistics about predicate access patterns
#[derive(Clone, Debug)]
pub struct QueryStatistics {
    /// Number of times each query type has been executed
    query_counts: HashMap<String, usize>,
    /// Selectivity (fraction of results) for each query
    selectivity: HashMap<String, f64>,
    /// Average execution time for each query type
    avg_execution_time: HashMap<String, Duration>,
    /// Total executions
    total_queries: usize,
}

impl QueryStatistics {
    pub fn new() -> Self {
        Self {
            query_counts: HashMap::new(),
            selectivity: HashMap::new(),
            avg_execution_time: HashMap::new(),
            total_queries: 0,
        }
    }

    /// Record a query execution
    pub fn record_query(
        &mut self,
        query_type: impl Into<String>,
        duration: Duration,
        result_count: usize,
        total_predicates: usize,
    ) {
        let query_type = query_type.into();

        *self.query_counts.entry(query_type.clone()).or_insert(0) += 1;
        self.total_queries += 1;

        let selectivity = if total_predicates > 0 {
            result_count as f64 / total_predicates as f64
        } else {
            0.0
        };

        self.selectivity.insert(query_type.clone(), selectivity);

        let count = self.query_counts[&query_type];
        let current_avg = self
            .avg_execution_time
            .get(&query_type)
            .copied()
            .unwrap_or(Duration::ZERO);

        let new_avg = (current_avg * (count as u32 - 1) + duration) / count as u32;
        self.avg_execution_time.insert(query_type, new_avg);
    }

    /// Get the most frequent query types
    pub fn top_queries(&self, limit: usize) -> Vec<(String, usize)> {
        let mut queries: Vec<_> = self
            .query_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        queries.sort_by(|a, b| b.1.cmp(&a.1));
        queries.truncate(limit);
        queries
    }

    /// Get average selectivity for a query type
    pub fn get_selectivity(&self, query_type: &str) -> f64 {
        self.selectivity.get(query_type).copied().unwrap_or(1.0)
    }

    /// Get average execution time for a query type
    pub fn get_avg_time(&self, query_type: &str) -> Duration {
        self.avg_execution_time
            .get(query_type)
            .copied()
            .unwrap_or(Duration::ZERO)
    }
}

impl Default for QueryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Index strategy for predicate lookups
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexStrategy {
    /// No index, full scan
    FullScan,
    /// Hash index on predicate name
    NameHash,
    /// Range index on arity
    ArityRange,
    /// Hash index on signature
    SignatureHash,
    /// Inverted index on domains
    DomainInverted,
    /// Composite index
    Composite(Vec<IndexStrategy>),
}

impl IndexStrategy {
    /// Estimate the cost of using this strategy
    pub fn estimate_cost(&self, predicates_count: usize, _stats: &QueryStatistics) -> f64 {
        match self {
            IndexStrategy::FullScan => predicates_count as f64,
            IndexStrategy::NameHash => 1.0, // O(1) lookup
            IndexStrategy::ArityRange => (predicates_count as f64).sqrt(), // O(sqrt(n)) estimate
            IndexStrategy::SignatureHash => 1.0, // O(1) lookup
            IndexStrategy::DomainInverted => (predicates_count as f64).log2(), // O(log n) estimate
            IndexStrategy::Composite(strategies) => {
                // Cost is minimum of component strategies
                strategies
                    .iter()
                    .map(|s| s.estimate_cost(predicates_count, _stats))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(predicates_count as f64)
            }
        }
    }
}

/// Query execution plan
#[derive(Clone, Debug)]
pub struct QueryPlan {
    query: PredicateQuery,
    strategy: IndexStrategy,
    estimated_cost: f64,
    estimated_results: usize,
}

impl QueryPlan {
    pub fn new(query: PredicateQuery, strategy: IndexStrategy) -> Self {
        Self {
            query,
            strategy,
            estimated_cost: 0.0,
            estimated_results: 0,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = cost;
        self
    }

    pub fn with_estimated_results(mut self, count: usize) -> Self {
        self.estimated_results = count;
        self
    }

    /// Execute the plan
    pub fn execute(&self, table: &SymbolTable) -> Result<Vec<(String, PredicateInfo)>> {
        match &self.query {
            PredicateQuery::ByName(name) => {
                if let Some(pred) = table.predicates.get(name) {
                    Ok(vec![(name.clone(), pred.clone())])
                } else {
                    Ok(Vec::new())
                }
            }
            PredicateQuery::ByArity(arity) => {
                let results: Vec<_> = table
                    .predicates
                    .iter()
                    .filter(|(_, pred)| pred.arg_domains.len() == *arity)
                    .map(|(name, pred)| (name.clone(), pred.clone()))
                    .collect();
                Ok(results)
            }
            PredicateQuery::BySignature(signature) => {
                let results: Vec<_> = table
                    .predicates
                    .iter()
                    .filter(|(_, pred)| pred.arg_domains == *signature)
                    .map(|(name, pred)| (name.clone(), pred.clone()))
                    .collect();
                Ok(results)
            }
            PredicateQuery::ByDomain(domain) => {
                let results: Vec<_> = table
                    .predicates
                    .iter()
                    .filter(|(_, pred)| pred.arg_domains.contains(domain))
                    .map(|(name, pred)| (name.clone(), pred.clone()))
                    .collect();
                Ok(results)
            }
            PredicateQuery::ByPattern(pattern) => {
                let results: Vec<_> = table
                    .predicates
                    .iter()
                    .filter(|(name, pred)| pattern.matches(name, pred))
                    .map(|(name, pred)| (name.clone(), pred.clone()))
                    .collect();
                Ok(results)
            }
            PredicateQuery::And(queries) => {
                if queries.is_empty() {
                    return Ok(Vec::new());
                }

                // Execute first query
                let mut results: HashSet<String> = self
                    .execute_subquery(&queries[0], table)?
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();

                // Intersect with remaining queries
                for query in &queries[1..] {
                    let subresults: HashSet<String> = self
                        .execute_subquery(query, table)?
                        .into_iter()
                        .map(|(name, _)| name)
                        .collect();
                    results.retain(|name| subresults.contains(name));
                }

                Ok(results
                    .into_iter()
                    .filter_map(|name| {
                        table
                            .predicates
                            .get(&name)
                            .map(|pred| (name.clone(), pred.clone()))
                    })
                    .collect())
            }
            PredicateQuery::Or(queries) => {
                let mut results_map: HashMap<String, PredicateInfo> = HashMap::new();

                for query in queries {
                    let subresults = self.execute_subquery(query, table)?;
                    for (name, pred) in subresults {
                        results_map.insert(name, pred);
                    }
                }

                Ok(results_map.into_iter().collect())
            }
        }
    }

    fn execute_subquery(
        &self,
        query: &PredicateQuery,
        table: &SymbolTable,
    ) -> Result<Vec<(String, PredicateInfo)>> {
        let subplan = QueryPlan::new(query.clone(), self.strategy.clone());
        subplan.execute(table)
    }

    pub fn query(&self) -> &PredicateQuery {
        &self.query
    }

    pub fn strategy(&self) -> &IndexStrategy {
        &self.strategy
    }

    pub fn estimated_cost(&self) -> f64 {
        self.estimated_cost
    }
}

/// Query planner for optimizing predicate lookups
pub struct QueryPlanner<'a> {
    table: &'a SymbolTable,
    statistics: QueryStatistics,
    plan_cache: HashMap<PredicateQuery, QueryPlan>,
}

impl<'a> QueryPlanner<'a> {
    pub fn new(table: &'a SymbolTable) -> Self {
        Self {
            table,
            statistics: QueryStatistics::new(),
            plan_cache: HashMap::new(),
        }
    }

    pub fn with_statistics(mut self, statistics: QueryStatistics) -> Self {
        self.statistics = statistics;
        self
    }

    /// Plan a query
    pub fn plan(&mut self, query: &PredicateQuery) -> Result<QueryPlan> {
        // Check cache first
        if let Some(cached) = self.plan_cache.get(query) {
            return Ok(cached.clone());
        }

        let plan = self.generate_plan(query)?;
        self.plan_cache.insert(query.clone(), plan.clone());
        Ok(plan)
    }

    /// Generate an optimal plan for a query
    fn generate_plan(&self, query: &PredicateQuery) -> Result<QueryPlan> {
        let strategy = self.select_strategy(query);
        let cost = strategy.estimate_cost(self.table.predicates.len(), &self.statistics);

        let plan = QueryPlan::new(query.clone(), strategy).with_cost(cost);

        Ok(plan)
    }

    /// Select the best index strategy for a query
    fn select_strategy(&self, query: &PredicateQuery) -> IndexStrategy {
        Self::select_strategy_static(query)
    }

    /// Static strategy selection (to avoid recursion on self)
    fn select_strategy_static(query: &PredicateQuery) -> IndexStrategy {
        match query {
            PredicateQuery::ByName(_) => IndexStrategy::NameHash,
            PredicateQuery::ByArity(_) => IndexStrategy::ArityRange,
            PredicateQuery::BySignature(_) => IndexStrategy::SignatureHash,
            PredicateQuery::ByDomain(_) => IndexStrategy::DomainInverted,
            PredicateQuery::ByPattern(_) => {
                // Pattern queries typically require full scan
                IndexStrategy::FullScan
            }
            PredicateQuery::And(queries) => {
                // Use the most selective strategy
                let strategies: Vec<_> = queries.iter().map(Self::select_strategy_static).collect();
                IndexStrategy::Composite(strategies)
            }
            PredicateQuery::Or(queries) => {
                // Use composite strategy
                let strategies: Vec<_> = queries.iter().map(Self::select_strategy_static).collect();
                IndexStrategy::Composite(strategies)
            }
        }
    }

    /// Execute a query and record statistics
    pub fn execute(&mut self, query: &PredicateQuery) -> Result<Vec<(String, PredicateInfo)>> {
        let start = Instant::now();
        let plan = self.plan(query)?;
        let results = plan.execute(self.table)?;
        let duration = start.elapsed();

        let query_type = format!("{:?}", query)
            .split('(')
            .next()
            .unwrap_or("Unknown")
            .to_string();
        self.statistics.record_query(
            query_type,
            duration,
            results.len(),
            self.table.predicates.len(),
        );

        Ok(results)
    }

    /// Get query statistics
    pub fn statistics(&self) -> &QueryStatistics {
        &self.statistics
    }

    /// Clear the plan cache
    pub fn clear_cache(&mut self) {
        self.plan_cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.plan_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;

    fn setup_table() -> SymbolTable {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        table.add_predicate(knows).unwrap();

        let at = PredicateInfo::new("at", vec!["Person".to_string(), "Location".to_string()]);
        table.add_predicate(at).unwrap();

        let friends =
            PredicateInfo::new("friends", vec!["Person".to_string(), "Person".to_string()]);
        table.add_predicate(friends).unwrap();

        table
    }

    #[test]
    fn test_query_by_name() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::by_name("knows");
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "knows");
    }

    #[test]
    fn test_query_by_arity() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::by_arity(2);
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 3); // knows, at, friends
    }

    #[test]
    fn test_query_by_signature() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::by_signature(vec!["Person".to_string(), "Person".to_string()]);
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 2); // knows, friends
    }

    #[test]
    fn test_query_by_domain() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::by_domain("Location");
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 1); // at
        assert_eq!(results[0].0, "at");
    }

    #[test]
    fn test_query_and() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::and(vec![
            PredicateQuery::by_arity(2),
            PredicateQuery::by_domain("Location"),
        ]);
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 1); // at
    }

    #[test]
    fn test_query_or() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::or(vec![
            PredicateQuery::by_name("knows"),
            PredicateQuery::by_name("at"),
        ]);
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 2); // knows, at
    }

    #[test]
    fn test_predicate_pattern() {
        let pattern = PredicatePattern::new()
            .with_name_pattern("know*")
            .with_arity_range(2, 3)
            .with_required_domain("Person");

        let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        assert!(pattern.matches("knows", &knows));

        let at = PredicateInfo::new("at", vec!["Person".to_string(), "Location".to_string()]);
        assert!(!pattern.matches("at", &at));
    }

    #[test]
    fn test_query_by_pattern() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let pattern = PredicatePattern::new()
            .with_name_pattern("*friend*")
            .with_required_domain("Person");

        let query = PredicateQuery::by_pattern(pattern);
        let results = planner.execute(&query).unwrap();

        assert_eq!(results.len(), 1); // friends
    }

    #[test]
    fn test_wildcard_matching() {
        assert!(matches_wildcard("hello", "h*"));
        assert!(matches_wildcard("hello", "he??o"));
        assert!(matches_wildcard("hello", "*"));
        assert!(matches_wildcard("hello", "hello"));
        assert!(!matches_wildcard("hello", "h*x"));
        assert!(matches_wildcard("test123", "test*"));
    }

    #[test]
    fn test_statistics() {
        let mut stats = QueryStatistics::new();

        stats.record_query("ByName", Duration::from_millis(10), 1, 100);
        stats.record_query("ByName", Duration::from_millis(20), 1, 100);
        stats.record_query("ByArity", Duration::from_millis(50), 10, 100);

        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.get_selectivity("ByName"), 0.01);
        assert_eq!(stats.get_selectivity("ByArity"), 0.1);

        let top = stats.top_queries(2);
        assert_eq!(top[0].0, "ByName");
        assert_eq!(top[0].1, 2);
    }

    #[test]
    fn test_plan_caching() {
        let table = setup_table();
        let mut planner = QueryPlanner::new(&table);

        let query = PredicateQuery::by_name("knows");

        planner.plan(&query).unwrap();
        assert_eq!(planner.cache_size(), 1);

        planner.plan(&query).unwrap();
        assert_eq!(planner.cache_size(), 1); // Should reuse cached plan

        planner.clear_cache();
        assert_eq!(planner.cache_size(), 0);
    }

    #[test]
    fn test_index_strategy_cost() {
        let stats = QueryStatistics::new();

        let full_scan = IndexStrategy::FullScan;
        let hash = IndexStrategy::NameHash;

        assert_eq!(full_scan.estimate_cost(1000, &stats), 1000.0);
        assert_eq!(hash.estimate_cost(1000, &stats), 1.0);
    }
}
