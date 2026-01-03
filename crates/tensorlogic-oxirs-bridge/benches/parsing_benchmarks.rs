//! Benchmarks for tensorlogic-oxirs-bridge parsing and conversion operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_oxirs_bridge::{
    schema::{nquads::NQuadsProcessor, streaming::StreamingRdfLoader},
    SchemaAnalyzer, SparqlCompiler,
};

/// Generate Turtle data with specified number of triples.
fn generate_turtle(num_triples: usize) -> String {
    let mut output = String::new();
    output.push_str("@prefix ex: <http://example.org/> .\n");
    output.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n");

    for i in 0..num_triples {
        output.push_str(&format!("ex:s{} ex:p{} ex:o{} .\n", i % 100, i % 10, i));
    }

    output
}

/// Generate N-Quads data with specified number of quads.
fn generate_nquads(num_quads: usize) -> String {
    let mut output = String::new();

    for i in 0..num_quads {
        let graph = if i % 3 == 0 {
            format!(" <http://example.org/graph{}>", i % 5)
        } else {
            String::new()
        };
        output.push_str(&format!(
            "<http://example.org/s{}> <http://example.org/p{}> <http://example.org/o{}>{} .\n",
            i % 100,
            i % 10,
            i,
            graph
        ));
    }

    output
}

/// Benchmark Turtle parsing.
fn bench_turtle_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("turtle_parsing");

    for size in [100, 1000, 10000].iter() {
        let data = generate_turtle(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let mut analyzer = SchemaAnalyzer::new();
                analyzer.load_turtle(black_box(data)).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark schema analysis.
fn bench_schema_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_analysis");

    let schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Entity a rdfs:Class .
        ex:Person a rdfs:Class ; rdfs:subClassOf ex:Entity .
        ex:Employee a rdfs:Class ; rdfs:subClassOf ex:Person .
        ex:Manager a rdfs:Class ; rdfs:subClassOf ex:Employee .

        ex:name a rdf:Property ; rdfs:domain ex:Entity .
        ex:age a rdf:Property ; rdfs:domain ex:Person .
        ex:department a rdf:Property ; rdfs:domain ex:Employee .
        ex:manages a rdf:Property ; rdfs:domain ex:Manager ; rdfs:range ex:Employee .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(schema).unwrap();

    group.bench_function("analyze_schema", |b| {
        b.iter(|| {
            let mut analyzer_clone = SchemaAnalyzer::from_graph(analyzer.graph.clone());
            analyzer_clone.analyze().unwrap();
        });
    });

    group.finish();
}

/// Benchmark N-Quads parsing.
fn bench_nquads_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("nquads_parsing");

    for size in [100, 1000, 10000].iter() {
        let data = generate_nquads(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let mut processor = NQuadsProcessor::new();
                processor.load_nquads(black_box(data)).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark streaming RDF loading.
fn bench_streaming_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_loading");

    for size in [100, 1000, 10000].iter() {
        let data = generate_turtle(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let mut loader = StreamingRdfLoader::new();
                loader.process_turtle(black_box(data)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark SPARQL query parsing.
fn bench_sparql_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql_parsing");

    let queries = vec![
        ("simple_select", "SELECT ?x WHERE { ?x <http://example.org/p> ?y }"),
        ("select_with_filter", "SELECT ?x ?y WHERE { ?x <http://example.org/p> ?y . FILTER(?x != ?y) }"),
        ("select_with_optional", "SELECT ?x ?y ?z WHERE { ?x <http://example.org/p> ?y . OPTIONAL { ?y <http://example.org/q> ?z } }"),
        ("union_query", "SELECT ?x ?y WHERE { { ?x <http://example.org/p> ?y } UNION { ?x <http://example.org/q> ?y } }"),
        ("construct_query", "CONSTRUCT { ?x <http://example.org/rel> ?y } WHERE { ?x <http://example.org/p> ?z . ?z <http://example.org/q> ?y }"),
    ];

    let compiler = SparqlCompiler::new();

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("parse", name), &query, |b, query| {
            b.iter(|| compiler.parse_query(black_box(query)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark SPARQL compilation to TensorLogic.
fn bench_sparql_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql_compilation");

    let queries = vec![
        ("simple", "SELECT ?x WHERE { ?x <http://example.org/p> ?y }"),
        ("filter", "SELECT ?x ?y WHERE { ?x <http://example.org/p> ?y . FILTER(?x != ?y) }"),
        ("optional", "SELECT ?x ?y ?z WHERE { ?x <http://example.org/p> ?y . OPTIONAL { ?y <http://example.org/q> ?z } }"),
    ];

    let mut compiler = SparqlCompiler::new();
    compiler.add_predicate_mapping("http://example.org/p".to_string(), "p".to_string());
    compiler.add_predicate_mapping("http://example.org/q".to_string(), "q".to_string());

    for (name, query_str) in queries {
        let query = compiler.parse_query(query_str).unwrap();

        group.bench_with_input(BenchmarkId::new("compile", name), &query, |b, query| {
            b.iter(|| compiler.compile_to_tensorlogic(black_box(query)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark symbol table conversion.
fn bench_symbol_table_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_table_conversion");

    let schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Entity a rdfs:Class .
        ex:Person a rdfs:Class ; rdfs:subClassOf ex:Entity .
        ex:Organization a rdfs:Class ; rdfs:subClassOf ex:Entity .

        ex:name a rdf:Property ; rdfs:domain ex:Entity .
        ex:knows a rdf:Property ; rdfs:domain ex:Person ; rdfs:range ex:Person .
        ex:worksFor a rdf:Property ; rdfs:domain ex:Person ; rdfs:range ex:Organization .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(schema).unwrap();
    analyzer.analyze().unwrap();

    group.bench_function("to_symbol_table", |b| {
        b.iter(|| analyzer.to_symbol_table().unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_turtle_parsing,
    bench_schema_analysis,
    bench_nquads_parsing,
    bench_streaming_loading,
    bench_sparql_parsing,
    bench_sparql_compilation,
    bench_symbol_table_conversion,
);

criterion_main!(benches);
