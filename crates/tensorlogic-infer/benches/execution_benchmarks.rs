//! Benchmarks for tensorlogic-infer execution traits.
//!
//! Run with: cargo bench -p tensorlogic-infer

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::hint::black_box;
use tensorlogic_infer::{DummyExecutor, DummyTensor};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

fn create_simple_graph(num_nodes: usize) -> EinsumGraph {
    let mut graph = EinsumGraph::new();
    // Create initial tensor
    let _t0 = graph.add_tensor("x");

    // Add element-wise unary operations (Relu)
    for i in 1..num_nodes {
        let prev = i - 1;
        let current = graph.add_tensor(format!("t{}", i));
        let node = EinsumNode {
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            inputs: vec![prev],
            outputs: vec![current],
            metadata: None,
        };
        graph.nodes.push(node);
    }
    graph
}

fn create_inputs() -> HashMap<String, DummyTensor> {
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), DummyTensor::new("x", vec![64, 128]));
    inputs
}

fn bench_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution");
    for num_nodes in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*num_nodes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            num_nodes,
            |b, &num_nodes| {
                let graph = create_simple_graph(num_nodes);
                let _inputs = create_inputs();
                b.iter(|| {
                    // Benchmark graph validation and setup
                    let executor = DummyExecutor::new();
                    black_box(&graph);
                    black_box(executor);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_execution);
criterion_main!(benches);
