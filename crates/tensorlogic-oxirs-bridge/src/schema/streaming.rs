//! Streaming RDF processing for large graphs.
//!
//! This module provides memory-efficient streaming support for processing
//! large RDF datasets without loading everything into memory at once.
//!
//! # Features
//!
//! - **Chunked Processing**: Process RDF in configurable batch sizes
//! - **Callback-based**: Register handlers for different triple patterns
//! - **Statistics Tracking**: Monitor progress during streaming
//! - **Memory Efficient**: Avoid loading entire graph into memory
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_oxirs_bridge::schema::streaming::{StreamingRdfLoader, StreamStats};
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     let turtle = r#"
//!         @prefix ex: <http://example.org/> .
//!         ex:Alice ex:knows ex:Bob .
//!         ex:Bob ex:knows ex:Charlie .
//!     "#;
//!
//!     let mut loader = StreamingRdfLoader::new();
//!
//!     // Register a handler for all triples
//!     loader = loader.on_triple(|subject, predicate, object| {
//!         println!("{} {} {}", subject, predicate, object);
//!     });
//!
//!     // Process the data
//!     let (stats, _graph) = loader.process_turtle(turtle)?;
//!     println!("Processed {} triples", stats.triples_processed);
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use oxrdf::{Graph, Triple};
use oxttl::TurtleParser;
use std::io::BufRead;
use std::time::{Duration, Instant};

/// Statistics from streaming RDF processing.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total number of triples processed
    pub triples_processed: usize,
    /// Number of batches processed
    pub batches_processed: usize,
    /// Total processing time
    pub processing_time: Duration,
    /// Number of errors encountered
    pub errors_encountered: usize,
    /// Peak memory usage (estimated)
    pub peak_memory_bytes: usize,
}

impl StreamStats {
    /// Get processing rate in triples per second.
    pub fn triples_per_second(&self) -> f64 {
        if self.processing_time.as_secs_f64() > 0.0 {
            self.triples_processed as f64 / self.processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Handler function type for processing individual triples.
pub type TripleHandler = Box<dyn FnMut(&str, &str, &str) + Send>;

/// Handler function type for batch processing.
pub type BatchHandler = Box<dyn FnMut(&[Triple]) + Send>;

/// Handler function type for progress updates.
pub type ProgressHandler = Box<dyn FnMut(&StreamStats) + Send>;

/// Streaming RDF loader for memory-efficient processing.
///
/// This loader processes RDF data in chunks, allowing you to handle
/// large datasets without loading everything into memory.
pub struct StreamingRdfLoader {
    /// Batch size for chunked processing
    batch_size: usize,
    /// Handler for individual triples
    triple_handler: Option<TripleHandler>,
    /// Handler for batches
    batch_handler: Option<BatchHandler>,
    /// Handler for progress updates
    progress_handler: Option<ProgressHandler>,
    /// Progress update interval (in triples)
    progress_interval: usize,
    /// Whether to collect into a graph
    collect_graph: bool,
    /// Filter predicates (if set, only process these)
    predicate_filter: Option<Vec<String>>,
    /// Filter subjects by prefix
    subject_prefix_filter: Option<String>,
}

impl StreamingRdfLoader {
    /// Create a new streaming loader with default settings.
    pub fn new() -> Self {
        StreamingRdfLoader {
            batch_size: 1000,
            triple_handler: None,
            batch_handler: None,
            progress_handler: None,
            progress_interval: 10000,
            collect_graph: false,
            predicate_filter: None,
            subject_prefix_filter: None,
        }
    }

    /// Set the batch size for chunked processing.
    ///
    /// Larger batches are more efficient but use more memory.
    /// Default is 1000 triples per batch.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Set a handler for individual triples.
    ///
    /// The handler receives (subject, predicate, object) as strings.
    pub fn on_triple<F>(mut self, handler: F) -> Self
    where
        F: FnMut(&str, &str, &str) + Send + 'static,
    {
        self.triple_handler = Some(Box::new(handler));
        self
    }

    /// Set a handler for triple batches.
    ///
    /// The handler receives a slice of Triple objects.
    pub fn on_batch<F>(mut self, handler: F) -> Self
    where
        F: FnMut(&[Triple]) + Send + 'static,
    {
        self.batch_handler = Some(Box::new(handler));
        self
    }

    /// Set a handler for progress updates.
    ///
    /// The handler is called every `progress_interval` triples.
    pub fn on_progress<F>(mut self, handler: F) -> Self
    where
        F: FnMut(&StreamStats) + Send + 'static,
    {
        self.progress_handler = Some(Box::new(handler));
        self
    }

    /// Set the interval for progress updates.
    ///
    /// Default is every 10000 triples.
    pub fn with_progress_interval(mut self, interval: usize) -> Self {
        self.progress_interval = interval.max(1);
        self
    }

    /// Enable collecting triples into a graph.
    ///
    /// This is useful when you need the complete graph after streaming.
    /// Note: This increases memory usage.
    pub fn collect_into_graph(mut self) -> Self {
        self.collect_graph = true;
        self
    }

    /// Filter to only process triples with specific predicates.
    pub fn filter_predicates(mut self, predicates: Vec<String>) -> Self {
        self.predicate_filter = Some(predicates);
        self
    }

    /// Filter to only process triples whose subject starts with a prefix.
    pub fn filter_subject_prefix(mut self, prefix: String) -> Self {
        self.subject_prefix_filter = Some(prefix);
        self
    }

    /// Process Turtle data from a string.
    pub fn process_turtle(&mut self, data: &str) -> Result<(StreamStats, Option<Graph>)> {
        let reader = std::io::Cursor::new(data);
        self.process_turtle_reader(reader)
    }

    /// Process Turtle data from a reader.
    pub fn process_turtle_reader<R: BufRead>(
        &mut self,
        reader: R,
    ) -> Result<(StreamStats, Option<Graph>)> {
        let start = Instant::now();
        let mut stats = StreamStats::default();
        let mut graph = if self.collect_graph {
            Some(Graph::new())
        } else {
            None
        };
        let mut batch: Vec<Triple> = Vec::with_capacity(self.batch_size);

        let parser = TurtleParser::new().for_reader(reader);

        for result in parser {
            match result {
                Ok(triple) => {
                    // Apply filters
                    if !self.should_process_triple(&triple) {
                        continue;
                    }

                    stats.triples_processed += 1;

                    // Call triple handler
                    if self.triple_handler.is_some() {
                        let subject = self.subject_to_string(&triple.subject);
                        let predicate = triple.predicate.as_str().to_string();
                        let object = self.term_to_string(triple.object.as_ref());
                        if let Some(ref mut handler) = self.triple_handler {
                            handler(&subject, &predicate, &object);
                        }
                    }

                    // Add to batch
                    batch.push(triple);

                    // Process batch if full
                    if batch.len() >= self.batch_size {
                        self.process_batch(&batch, &mut graph, &mut stats);
                        batch.clear();
                        stats.batches_processed += 1;
                    }

                    // Progress update
                    if stats.triples_processed % self.progress_interval == 0 {
                        stats.processing_time = start.elapsed();
                        if let Some(ref mut handler) = self.progress_handler {
                            handler(&stats);
                        }
                    }
                }
                Err(e) => {
                    stats.errors_encountered += 1;
                    // Continue processing on error
                    eprintln!("Parse error: {}", e);
                }
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            self.process_batch(&batch, &mut graph, &mut stats);
            stats.batches_processed += 1;
        }

        stats.processing_time = start.elapsed();
        Ok((stats, graph))
    }

    /// Check if a triple should be processed based on filters.
    fn should_process_triple(&self, triple: &Triple) -> bool {
        // Check predicate filter
        if let Some(ref predicates) = self.predicate_filter {
            let pred_str = triple.predicate.as_str();
            if !predicates.iter().any(|p| pred_str.contains(p)) {
                return false;
            }
        }

        // Check subject prefix filter
        if let Some(ref prefix) = self.subject_prefix_filter {
            let subject_str = self.subject_to_string(&triple.subject);
            if !subject_str.starts_with(prefix) {
                return false;
            }
        }

        true
    }

    /// Convert subject to string.
    fn subject_to_string(&self, subject: &oxrdf::NamedOrBlankNode) -> String {
        match subject {
            oxrdf::NamedOrBlankNode::NamedNode(n) => n.as_str().to_string(),
            oxrdf::NamedOrBlankNode::BlankNode(b) => format!("_:{}", b.as_str()),
        }
    }

    /// Process a batch of triples.
    fn process_batch(
        &mut self,
        batch: &[Triple],
        graph: &mut Option<Graph>,
        _stats: &mut StreamStats,
    ) {
        // Call batch handler
        if let Some(ref mut handler) = self.batch_handler {
            handler(batch);
        }

        // Add to graph if collecting
        if let Some(ref mut g) = graph {
            for triple in batch {
                g.insert(triple);
            }
        }
    }

    /// Convert an RDF term to string.
    fn term_to_string(&self, term: oxrdf::TermRef) -> String {
        match term {
            oxrdf::TermRef::NamedNode(n) => n.as_str().to_string(),
            oxrdf::TermRef::BlankNode(b) => format!("_:{}", b.as_str()),
            oxrdf::TermRef::Literal(l) => {
                if let Some(lang) = l.language() {
                    format!("\"{}\"@{}", l.value(), lang)
                } else if l.datatype() != oxrdf::vocab::xsd::STRING {
                    format!("\"{}\"^^{}", l.value(), l.datatype().as_str())
                } else {
                    format!("\"{}\"", l.value())
                }
            }
            #[allow(unreachable_patterns)]
            _ => "[triple]".to_string(),
        }
    }
}

impl Default for StreamingRdfLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream processor for analyzing large RDF datasets.
///
/// This provides higher-level analysis operations on streaming data.
pub struct StreamAnalyzer {
    /// Count predicates
    predicate_counts: std::collections::HashMap<String, usize>,
    /// Count subjects
    subject_count: usize,
    /// Count unique subjects
    unique_subjects: std::collections::HashSet<String>,
    /// Track namespaces
    namespaces: std::collections::HashSet<String>,
}

impl StreamAnalyzer {
    /// Create a new stream analyzer.
    pub fn new() -> Self {
        StreamAnalyzer {
            predicate_counts: std::collections::HashMap::new(),
            subject_count: 0,
            unique_subjects: std::collections::HashSet::new(),
            namespaces: std::collections::HashSet::new(),
        }
    }

    /// Process a triple for analysis.
    pub fn process_triple(&mut self, subject: &str, predicate: &str, _object: &str) {
        self.subject_count += 1;
        self.unique_subjects.insert(subject.to_string());

        *self
            .predicate_counts
            .entry(predicate.to_string())
            .or_insert(0) += 1;

        // Extract namespace
        if let Some(ns) = Self::extract_namespace(predicate) {
            self.namespaces.insert(ns.to_string());
        }
    }

    /// Extract namespace from IRI.
    fn extract_namespace(iri: &str) -> Option<&str> {
        if let Some(hash_pos) = iri.rfind('#') {
            Some(&iri[..=hash_pos])
        } else if let Some(slash_pos) = iri.rfind('/') {
            Some(&iri[..=slash_pos])
        } else {
            None
        }
    }

    /// Get predicate statistics.
    pub fn predicate_stats(&self) -> &std::collections::HashMap<String, usize> {
        &self.predicate_counts
    }

    /// Get unique subject count.
    pub fn unique_subject_count(&self) -> usize {
        self.unique_subjects.len()
    }

    /// Get total triple count.
    pub fn total_triples(&self) -> usize {
        self.subject_count
    }

    /// Get discovered namespaces.
    pub fn namespaces(&self) -> &std::collections::HashSet<String> {
        &self.namespaces
    }

    /// Get top N predicates by frequency.
    pub fn top_predicates(&self, n: usize) -> Vec<(&str, usize)> {
        let mut predicates: Vec<_> = self.predicate_counts.iter().collect();
        predicates.sort_by(|a, b| b.1.cmp(a.1));
        predicates
            .into_iter()
            .take(n)
            .map(|(k, v)| (k.as_str(), *v))
            .collect()
    }
}

impl Default for StreamAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Process an N-Triples file line by line.
///
/// This is more efficient than the Turtle parser for simple N-Triples format.
pub fn process_ntriples_lines<F>(data: &str, mut handler: F) -> Result<usize>
where
    F: FnMut(&str, &str, &str),
{
    let mut count = 0;

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse N-Triples line: <subject> <predicate> <object> .
        if let Some((subject, rest)) = parse_ntriples_term(line) {
            let rest = rest.trim_start();
            if let Some((predicate, rest)) = parse_ntriples_term(rest) {
                let rest = rest.trim_start();
                if let Some((object, _)) = parse_ntriples_term(rest) {
                    handler(subject, predicate, object);
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

/// Parse a single N-Triples term.
fn parse_ntriples_term(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();

    if s.starts_with('<') {
        // IRI
        if let Some(end) = s.find('>') {
            return Some((&s[1..end], &s[end + 1..]));
        }
    } else if s.starts_with('"') {
        // Literal
        let mut i = 1;
        let chars: Vec<char> = s.chars().collect();
        while i < chars.len() {
            if chars[i] == '"' && (i == 0 || chars[i - 1] != '\\') {
                // Find end of literal (including optional language tag or datatype)
                let mut end = i + 1;
                if end < chars.len() && chars[end] == '@' {
                    // Language tag
                    while end < chars.len() && !chars[end].is_whitespace() {
                        end += 1;
                    }
                } else if end + 1 < chars.len() && chars[end] == '^' && chars[end + 1] == '^' {
                    // Datatype
                    end += 2;
                    if end < chars.len() && chars[end] == '<' {
                        while end < chars.len() && chars[end] != '>' {
                            end += 1;
                        }
                        if end < chars.len() {
                            end += 1;
                        }
                    }
                }
                return Some((&s[..end], &s[end..]));
            }
            i += 1;
        }
    } else if let Some(stripped) = s.strip_prefix("_:") {
        // Blank node
        let end = stripped
            .find(|c: char| c.is_whitespace() || c == '.')
            .map(|i| i + 2)
            .unwrap_or(s.len());
        return Some((&s[..end], &s[end..]));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_basic() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Charlie .
            ex:Charlie ex:knows ex:Alice .
        "#;

        let mut loader = StreamingRdfLoader::new();

        loader = loader.on_triple(|_s, _p, _o| {
            // Handler receives each triple
        });

        let (stats, _) = loader.process_turtle(turtle).unwrap();
        assert_eq!(stats.triples_processed, 3);
    }

    #[test]
    fn test_streaming_with_batch() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:A ex:p ex:B .
            ex:B ex:p ex:C .
            ex:C ex:p ex:D .
            ex:D ex:p ex:E .
            ex:E ex:p ex:F .
        "#;

        let mut loader = StreamingRdfLoader::new().with_batch_size(2);

        let (stats, _) = loader.process_turtle(turtle).unwrap();
        assert_eq!(stats.triples_processed, 5);
        assert_eq!(stats.batches_processed, 3); // 2 + 2 + 1
    }

    #[test]
    fn test_streaming_collect_graph() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Charlie .
        "#;

        let mut loader = StreamingRdfLoader::new().collect_into_graph();

        let (stats, graph) = loader.process_turtle(turtle).unwrap();
        assert_eq!(stats.triples_processed, 2);
        assert!(graph.is_some());
        assert_eq!(graph.unwrap().len(), 2);
    }

    #[test]
    fn test_streaming_filter_predicate() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:Alice ex:knows ex:Bob .
            ex:Alice rdfs:label "Alice" .
            ex:Bob ex:knows ex:Charlie .
        "#;

        let mut loader = StreamingRdfLoader::new().filter_predicates(vec!["knows".to_string()]);

        let (stats, _) = loader.process_turtle(turtle).unwrap();
        assert_eq!(stats.triples_processed, 2);
    }

    #[test]
    fn test_stream_analyzer() {
        let mut analyzer = StreamAnalyzer::new();

        analyzer.process_triple(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        );
        analyzer.process_triple(
            "http://example.org/Bob",
            "http://example.org/knows",
            "http://example.org/Charlie",
        );
        analyzer.process_triple("http://example.org/Alice", "http://example.org/age", "30");

        assert_eq!(analyzer.unique_subject_count(), 2);
        assert_eq!(analyzer.total_triples(), 3);
        assert_eq!(analyzer.predicate_stats().len(), 2);
        assert_eq!(analyzer.predicate_stats()["http://example.org/knows"], 2);
    }

    #[test]
    fn test_ntriples_processing() {
        let ntriples = r#"
            <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> .
            <http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
        "#;

        let mut count = 0;
        process_ntriples_lines(ntriples, |_s, _p, _o| {
            count += 1;
        })
        .unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn test_stats_rate() {
        let stats = StreamStats {
            triples_processed: 10000,
            batches_processed: 10,
            processing_time: Duration::from_secs(2),
            errors_encountered: 0,
            peak_memory_bytes: 0,
        };

        assert_eq!(stats.triples_per_second(), 5000.0);
    }
}
