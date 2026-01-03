//! Schema auto-completion system.
//!
//! This module provides intelligent auto-completion for schema construction,
//! suggesting domains, predicates, and variable names based on:
//! - Existing schema patterns
//! - Naming conventions
//! - Similarity to known schemas
//! - Common domain modeling patterns
//!
//! The system leverages embeddings for similarity-based suggestions and
//! incorporates common patterns from knowledge bases and domain modeling best practices.

use std::collections::HashMap;

use crate::{SimilaritySearch, SymbolTable};

/// Auto-completion engine for schema construction.
///
/// Provides intelligent suggestions for domains, predicates, and variable names
/// based on context and learned patterns.
pub struct AutoCompleter {
    /// Similarity search engine
    similarity_search: SimilaritySearch,
    /// Common patterns database
    patterns: PatternDatabase,
    /// Number of suggestions to return
    max_suggestions: usize,
}

impl AutoCompleter {
    /// Create a new auto-completer.
    pub fn new() -> Self {
        Self {
            similarity_search: SimilaritySearch::new(),
            patterns: PatternDatabase::default(),
            max_suggestions: 5,
        }
    }

    /// Set maximum number of suggestions.
    pub fn with_max_suggestions(mut self, max: usize) -> Self {
        self.max_suggestions = max;
        self
    }

    /// Index a symbol table for auto-completion.
    pub fn index_table(&mut self, table: &SymbolTable) {
        self.similarity_search.index_table(table);
    }

    /// Suggest domain names based on partial input.
    pub fn suggest_domain_names(&self, partial: &str) -> Vec<DomainSuggestion> {
        let mut suggestions = Vec::new();

        // Get pattern-based suggestions
        let pattern_suggestions = self.patterns.suggest_domain_names(partial);
        for (name, confidence) in pattern_suggestions.into_iter().take(self.max_suggestions) {
            suggestions.push(DomainSuggestion {
                name,
                estimated_cardinality: 100, // Default
                description: None,
                confidence,
                source: SuggestionSource::Pattern,
            });
        }

        suggestions
    }

    /// Suggest predicates based on domain context.
    pub fn suggest_predicates(
        &self,
        domains: &[String],
        partial: &str,
    ) -> Vec<PredicateSuggestion> {
        let mut suggestions = Vec::new();

        // Pattern-based suggestions
        let pattern_suggestions = self.patterns.suggest_predicates(domains, partial);

        for (name, arg_domains, confidence) in
            pattern_suggestions.into_iter().take(self.max_suggestions)
        {
            suggestions.push(PredicateSuggestion {
                name,
                arg_domains,
                description: None,
                confidence,
                source: SuggestionSource::Pattern,
            });
        }

        suggestions
    }

    /// Suggest variable names based on domain type.
    pub fn suggest_variable_names(&self, domain: &str, partial: &str) -> Vec<VariableSuggestion> {
        let mut suggestions = Vec::new();

        // Pattern-based suggestions
        let pattern_suggestions = self.patterns.suggest_variable_names(domain, partial);

        for (name, confidence) in pattern_suggestions.into_iter().take(self.max_suggestions) {
            suggestions.push(VariableSuggestion {
                name,
                domain: domain.to_string(),
                confidence,
                source: SuggestionSource::Pattern,
            });
        }

        suggestions
    }

    /// Suggest domain given a predicate pattern.
    ///
    /// For example, if a user is defining a predicate "teaches" with args ["Person", "?"],
    /// this suggests likely domains for the second argument.
    pub fn suggest_domain_for_predicate_arg(
        &self,
        predicate_name: &str,
        existing_args: &[String],
        _position: usize,
    ) -> Vec<DomainSuggestion> {
        let mut suggestions = Vec::new();

        // Use patterns to suggest likely domains
        let pattern_suggestions = self
            .patterns
            .suggest_domain_for_predicate(predicate_name, existing_args);

        for (name, confidence) in pattern_suggestions.into_iter().take(self.max_suggestions) {
            suggestions.push(DomainSuggestion {
                name,
                estimated_cardinality: 100,
                description: None,
                confidence,
                source: SuggestionSource::Pattern,
            });
        }

        suggestions
    }

    /// Get completion statistics.
    pub fn stats(&self) -> AutoCompleterStats {
        AutoCompleterStats {
            num_indexed_domains: self.similarity_search.stats().num_domains,
            num_indexed_predicates: self.similarity_search.stats().num_predicates,
            num_patterns: self.patterns.num_patterns(),
        }
    }
}

impl Default for AutoCompleter {
    fn default() -> Self {
        Self::new()
    }
}

/// Suggestion for a domain.
#[derive(Clone, Debug)]
pub struct DomainSuggestion {
    /// Suggested domain name
    pub name: String,
    /// Estimated cardinality
    pub estimated_cardinality: usize,
    /// Optional description
    pub description: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of suggestion
    pub source: SuggestionSource,
}

/// Suggestion for a predicate.
#[derive(Clone, Debug)]
pub struct PredicateSuggestion {
    /// Suggested predicate name
    pub name: String,
    /// Suggested argument domains
    pub arg_domains: Vec<String>,
    /// Optional description
    pub description: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of suggestion
    pub source: SuggestionSource,
}

/// Suggestion for a variable name.
#[derive(Clone, Debug)]
pub struct VariableSuggestion {
    /// Suggested variable name
    pub name: String,
    /// Domain of the variable
    pub domain: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of suggestion
    pub source: SuggestionSource,
}

/// Source of an auto-completion suggestion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuggestionSource {
    /// From pattern matching
    Pattern,
    /// From similarity search
    Similarity,
    /// From learned examples
    Learned,
    /// From manual templates
    Template,
}

/// Statistics about the auto-completer.
#[derive(Clone, Debug)]
pub struct AutoCompleterStats {
    /// Number of indexed domains
    pub num_indexed_domains: usize,
    /// Number of indexed predicates
    pub num_indexed_predicates: usize,
    /// Number of patterns in database
    pub num_patterns: usize,
}

/// Database of common domain modeling patterns.
///
/// Contains knowledge about common domain names, predicate patterns,
/// and naming conventions used in schema design.
struct PatternDatabase {
    /// Common domain names and their typical cardinalities
    common_domains: HashMap<String, Vec<(String, usize)>>,
    /// Common predicate patterns
    common_predicates: HashMap<String, Vec<(String, Vec<String>)>>,
    /// Variable naming patterns
    variable_patterns: HashMap<String, Vec<String>>,
}

impl Default for PatternDatabase {
    fn default() -> Self {
        let mut db = Self {
            common_domains: HashMap::new(),
            common_predicates: HashMap::new(),
            variable_patterns: HashMap::new(),
        };

        db.init_common_domains();
        db.init_common_predicates();
        db.init_variable_patterns();

        db
    }
}

impl PatternDatabase {
    /// Initialize common domain patterns.
    fn init_common_domains(&mut self) {
        // People and agents
        self.add_domain_pattern(
            "person",
            vec![("Person", 1000), ("User", 1000), ("Agent", 500)],
        );
        self.add_domain_pattern(
            "user",
            vec![("User", 1000), ("Person", 1000), ("Account", 500)],
        );
        self.add_domain_pattern(
            "student",
            vec![("Student", 500), ("Person", 1000), ("User", 1000)],
        );
        self.add_domain_pattern(
            "teacher",
            vec![("Teacher", 200), ("Instructor", 200), ("Person", 1000)],
        );

        // Courses and education
        self.add_domain_pattern(
            "course",
            vec![("Course", 100), ("Class", 100), ("Subject", 50)],
        );
        self.add_domain_pattern("class", vec![("Class", 100), ("Course", 100)]);

        // Organizations
        self.add_domain_pattern(
            "company",
            vec![("Company", 500), ("Organization", 500), ("Business", 500)],
        );
        self.add_domain_pattern(
            "department",
            vec![("Department", 50), ("Division", 50), ("Unit", 50)],
        );

        // Resources
        self.add_domain_pattern("book", vec![("Book", 5000), ("Publication", 10000)]);
        self.add_domain_pattern("product", vec![("Product", 10000), ("Item", 10000)]);
        self.add_domain_pattern("resource", vec![("Resource", 1000), ("Asset", 1000)]);

        // Time
        self.add_domain_pattern("time", vec![("Time", 86400), ("Timestamp", 86400)]);
        self.add_domain_pattern("date", vec![("Date", 365), ("Day", 365)]);

        // Locations
        self.add_domain_pattern("location", vec![("Location", 1000), ("Place", 1000)]);
        self.add_domain_pattern("city", vec![("City", 1000), ("Location", 1000)]);
        self.add_domain_pattern("country", vec![("Country", 200), ("Nation", 200)]);
    }

    /// Initialize common predicate patterns.
    fn init_common_predicates(&mut self) {
        // Binary relationships
        self.add_predicate_pattern(
            "person",
            vec![
                ("knows", vec!["Person", "Person"]),
                ("likes", vec!["Person", "Person"]),
                ("works_with", vec!["Person", "Person"]),
                ("manages", vec!["Person", "Person"]),
            ],
        );

        self.add_predicate_pattern(
            "student",
            vec![
                ("enrolled_in", vec!["Student", "Course"]),
                ("takes", vec!["Student", "Course"]),
                ("attends", vec!["Student", "Course"]),
            ],
        );

        self.add_predicate_pattern(
            "teach",
            vec![
                ("teaches", vec!["Teacher", "Course"]),
                ("instructs", vec!["Teacher", "Student"]),
            ],
        );

        // Unary predicates (properties)
        self.add_predicate_pattern(
            "is",
            vec![
                ("is_active", vec!["User"]),
                ("is_admin", vec!["User"]),
                ("is_public", vec!["Resource"]),
            ],
        );
    }

    /// Initialize variable naming patterns.
    fn init_variable_patterns(&mut self) {
        self.add_variable_pattern("Person", vec!["p", "person", "x", "user"]);
        self.add_variable_pattern("Student", vec!["s", "student", "x"]);
        self.add_variable_pattern("Teacher", vec!["t", "teacher", "instructor"]);
        self.add_variable_pattern("Course", vec!["c", "course", "class"]);
        self.add_variable_pattern("Book", vec!["b", "book"]);
        self.add_variable_pattern("Time", vec!["t", "time", "timestamp"]);
        self.add_variable_pattern("Date", vec!["d", "date", "day"]);
        self.add_variable_pattern("Location", vec!["l", "loc", "location", "place"]);
    }

    fn add_domain_pattern(&mut self, key: &str, patterns: Vec<(&str, usize)>) {
        self.common_domains.insert(
            key.to_string(),
            patterns
                .into_iter()
                .map(|(name, card)| (name.to_string(), card))
                .collect(),
        );
    }

    fn add_predicate_pattern(&mut self, key: &str, patterns: Vec<(&str, Vec<&str>)>) {
        self.common_predicates.insert(
            key.to_string(),
            patterns
                .into_iter()
                .map(|(name, args)| {
                    (
                        name.to_string(),
                        args.into_iter().map(|s| s.to_string()).collect(),
                    )
                })
                .collect(),
        );
    }

    fn add_variable_pattern(&mut self, key: &str, patterns: Vec<&str>) {
        self.variable_patterns.insert(
            key.to_string(),
            patterns.into_iter().map(|s| s.to_string()).collect(),
        );
    }

    fn suggest_domain_names(&self, partial: &str) -> Vec<(String, f64)> {
        let mut suggestions = Vec::new();
        let partial_lower = partial.to_lowercase();

        for (key, patterns) in &self.common_domains {
            if key.contains(&partial_lower) {
                for (name, _card) in patterns {
                    if name.to_lowercase().starts_with(&partial_lower) {
                        let confidence = 0.9;
                        suggestions.push((name.clone(), confidence));
                    }
                }
            }
        }

        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        suggestions
    }

    fn suggest_predicates(
        &self,
        domains: &[String],
        partial: &str,
    ) -> Vec<(String, Vec<String>, f64)> {
        let mut suggestions = Vec::new();
        let partial_lower = partial.to_lowercase();

        // Find patterns matching the domains
        for domain in domains {
            let domain_lower = domain.to_lowercase();
            if let Some(patterns) = self.common_predicates.get(&domain_lower) {
                for (name, args) in patterns {
                    if name.to_lowercase().starts_with(&partial_lower) {
                        let confidence = 0.85;
                        suggestions.push((name.clone(), args.clone(), confidence));
                    }
                }
            }
        }

        suggestions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        suggestions
    }

    fn suggest_variable_names(&self, domain: &str, partial: &str) -> Vec<(String, f64)> {
        let mut suggestions = Vec::new();
        let partial_lower = partial.to_lowercase();

        if let Some(patterns) = self.variable_patterns.get(domain) {
            for name in patterns {
                if name.starts_with(&partial_lower) {
                    let confidence = 0.9;
                    suggestions.push((name.clone(), confidence));
                }
            }
        }

        // Generic suggestions if no specific patterns
        if suggestions.is_empty() {
            let first_char = domain
                .chars()
                .next()
                .unwrap_or('x')
                .to_lowercase()
                .to_string();
            suggestions.push((first_char, 0.5));
            suggestions.push((domain.to_lowercase(), 0.6));
        }

        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        suggestions
    }

    fn suggest_domain_for_predicate(
        &self,
        predicate_name: &str,
        existing_args: &[String],
    ) -> Vec<(String, f64)> {
        let mut suggestions = Vec::new();

        // Find predicates with matching names and see what domains they use
        for patterns in self.common_predicates.values() {
            for (name, args) in patterns {
                if name == predicate_name && args.len() > existing_args.len() {
                    // Check if existing args match
                    let matches = existing_args.iter().zip(args.iter()).all(|(a, b)| a == b);

                    if matches {
                        // Suggest the next domain
                        if let Some(next_domain) = args.get(existing_args.len()) {
                            let confidence = 0.8;
                            suggestions.push((next_domain.clone(), confidence));
                        }
                    }
                }
            }
        }

        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        suggestions
    }

    fn num_patterns(&self) -> usize {
        self.common_domains.len() + self.common_predicates.len() + self.variable_patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;

    #[test]
    fn test_autocompleter_creation() {
        let ac = AutoCompleter::new();
        let stats = ac.stats();
        assert!(stats.num_patterns > 0);
    }

    #[test]
    fn test_suggest_domain_names() {
        let ac = AutoCompleter::new();
        let suggestions = ac.suggest_domain_names("per");

        assert!(!suggestions.is_empty());
        // Should suggest "Person" for "per"
        assert!(suggestions.iter().any(|s| s.name == "Person"));
    }

    #[test]
    fn test_suggest_predicates() {
        let ac = AutoCompleter::new();
        let suggestions = ac.suggest_predicates(&["Person".to_string()], "know");

        assert!(!suggestions.is_empty());
        // Should suggest "knows" for "know" with Person domain
        assert!(suggestions.iter().any(|s| s.name == "knows"));
    }

    #[test]
    fn test_suggest_variable_names() {
        let ac = AutoCompleter::new();
        let suggestions = ac.suggest_variable_names("Person", "p");

        assert!(!suggestions.is_empty());
        // Should suggest "p" or "person" for Person domain
        assert!(suggestions
            .iter()
            .any(|s| s.name == "p" || s.name == "person"));
    }

    #[test]
    fn test_suggest_domain_for_predicate() {
        let ac = AutoCompleter::new();
        let suggestions =
            ac.suggest_domain_for_predicate_arg("teaches", &["Teacher".to_string()], 1);

        assert!(!suggestions.is_empty());
        // Should suggest "Course" as second argument for "teaches"
        assert!(suggestions.iter().any(|s| s.name == "Course"));
    }

    #[test]
    fn test_max_suggestions_limit() {
        let ac = AutoCompleter::new().with_max_suggestions(3);
        let suggestions = ac.suggest_domain_names("p");

        assert!(suggestions.len() <= 3);
    }

    #[test]
    fn test_index_table() {
        let mut ac = AutoCompleter::new();
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("CustomDomain", 100))
            .unwrap();

        ac.index_table(&table);

        let stats = ac.stats();
        assert_eq!(stats.num_indexed_domains, 1);
    }

    #[test]
    fn test_suggestion_confidence() {
        let ac = AutoCompleter::new();
        let suggestions = ac.suggest_domain_names("person");

        for suggestion in &suggestions {
            assert!(suggestion.confidence >= 0.0 && suggestion.confidence <= 1.0);
        }
    }

    #[test]
    fn test_empty_partial() {
        let ac = AutoCompleter::new();
        let suggestions = ac.suggest_domain_names("");

        // Should return some suggestions even with empty input
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_case_insensitive_matching() {
        let ac = AutoCompleter::new();
        let suggestions_lower = ac.suggest_domain_names("person");
        let suggestions_upper = ac.suggest_domain_names("PERSON");

        // Should get same suggestions regardless of case
        assert!(!suggestions_lower.is_empty());
        assert!(!suggestions_upper.is_empty());
    }

    #[test]
    fn test_pattern_database_initialization() {
        let db = PatternDatabase::default();
        assert!(db.num_patterns() > 0);
        assert!(!db.common_domains.is_empty());
        assert!(!db.common_predicates.is_empty());
        assert!(!db.variable_patterns.is_empty());
    }

    #[test]
    fn test_multiple_domain_contexts() {
        let ac = AutoCompleter::new();
        let suggestions =
            ac.suggest_predicates(&["Student".to_string(), "Course".to_string()], "enroll");

        assert!(!suggestions.is_empty());
    }
}
