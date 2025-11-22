//! Benchmarks for tensorlogic-infer execution traits.
//!
//! Run with: cargo bench -p tensorlogic-infer

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use tensorlogic_infer::{DummyExecutor, DummyTensor, TlExecutor};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

fn create_simple_graph(num_nodes: usize) -> EinsumGraph {
    let mut graph = EinsumGraph::new();
    graph.nodes.push(EinsumNode {
        op: OpType::Input {
            name: "x".to_string(),
        },
        inputs: vec![],
        outputs: vec![0],
        metadata: None,
    });
    for i in 1..num_nodes {
        graph.nodes.push(EinsumNode {
            op: OpType::ElemUnary {
                op: tensorlogic_infer::ElemOp::Exp,
            },
            inputs: vec![i - 1],
            outputs: vec![i],
            metadata: None,
        });
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
                let inputs = create_inputs();
                let executor = DummyExecutor::new();
                b.iter(|| {
                    let _result = executor.execute(black_box(&graph), black_box(&inputs));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_execution);
criterion_main!(benches);
