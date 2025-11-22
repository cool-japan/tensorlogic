//! Comprehensive example demonstrating distributed execution with data/model/pipeline parallelism.
//!
//! This example shows:
//! - Multi-device distributed execution
//! - Data parallelism with gradient synchronization
//! - Model parallelism with tensor sharding
//! - Pipeline parallelism with stage coordination
//! - Performance monitoring and statistics

use std::collections::HashMap;
use tensorlogic_infer::{
    distributed::{
        CommunicationBackend, DataParallelCoordinator, DistributedConfig, DistributedExecutor,
        DistributedParallelismStrategy, DummyCommunicationBackend, ModelParallelCoordinator,
        PipelineParallelCoordinator, ShardingSpec,
    },
    placement::Device,
    DummyExecutor, DummyTensor,
};
use tensorlogic_ir::{EinsumGraph, EinsumNode, NodeId, OpType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TensorLogic Distributed Execution Demo ===\n");

    // Create a sample computation graph
    let graph = create_sample_graph();
    println!(
        "Created computation graph with {} nodes\n",
        graph.nodes.len()
    );

    // Demo 1: Data Parallelism
    println!("--- Demo 1: Data Parallelism ---");
    demo_data_parallel(&graph)?;

    // Demo 2: Model Parallelism
    println!("\n--- Demo 2: Model Parallelism ---");
    demo_model_parallel(&graph)?;

    // Demo 3: Pipeline Parallelism
    println!("\n--- Demo 3: Pipeline Parallelism ---");
    demo_pipeline_parallel(&graph)?;

    // Demo 4: Hybrid Parallelism
    println!("\n--- Demo 4: Hybrid Parallelism ---");
    demo_hybrid_parallel(&graph)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn create_sample_graph() -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    // Input node
    let input_node = EinsumNode {
        id: NodeId(0),
        op: OpType::Input {
            name: "x".to_string(),
        },
        inputs: vec![],
        output_shape: vec![64, 128],
        metadata: HashMap::new(),
    };
    graph.nodes.push(input_node);

    // Einsum operation (matrix multiplication)
    let einsum_node = EinsumNode {
        id: NodeId(1),
        op: OpType::Einsum {
            spec: "ij,jk->ik".to_string(),
        },
        inputs: vec![0, 2], // Input and weights
        output_shape: vec![64, 256],
        metadata: HashMap::new(),
    };
    graph.nodes.push(einsum_node);

    // Weights node
    let weights_node = EinsumNode {
        id: NodeId(2),
        op: OpType::Input {
            name: "weights".to_string(),
        },
        inputs: vec![],
        output_shape: vec![128, 256],
        metadata: HashMap::new(),
    };
    graph.nodes.push(weights_node);

    // Activation (ReLU)
    let relu_node = EinsumNode {
        id: NodeId(3),
        op: OpType::ElemUnary {
            op: tensorlogic_infer::ElemOp::Exp,
        },
        inputs: vec![1],
        output_shape: vec![64, 256],
        metadata: HashMap::new(),
    };
    graph.nodes.push(relu_node);

    graph
}

fn demo_data_parallel(graph: &EinsumGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up data parallelism across 4 devices...");

    // Setup devices
    let devices = vec![
        Device::CPU(0),
        Device::CPU(1),
        Device::CPU(2),
        Device::CPU(3),
    ];

    // Configure data parallelism
    let config = DistributedConfig::new(devices.clone())
        .with_strategy(DistributedParallelismStrategy::DataParallel { num_replicas: 4 });

    // Create distributed executor
    let backend = Box::new(DummyCommunicationBackend::new());
    let base_executor = DummyExecutor::new();
    let coordinator = DataParallelCoordinator::new(devices, backend);
    let mut executor = DistributedExecutor::new(base_executor, Box::new(coordinator));

    // Create sample inputs
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), DummyTensor::new(vec![64, 128]));
    inputs.insert("weights".to_string(), DummyTensor::new(vec![128, 256]));

    // Execute with data parallelism
    println!("Executing with data parallelism...");
    let _outputs = executor.execute_distributed(graph, &inputs, &config)?;

    // Get statistics
    let stats = executor.get_distributed_stats();
    println!("Communication time: {:.2}ms", stats.communication_time_ms);
    println!("Computation time: {:.2}ms", stats.computation_time_ms);
    println!("Efficiency: {:.2}%", stats.efficiency * 100.0);
    println!("Data transferred: {} bytes", stats.bytes_transferred);

    Ok(())
}

fn demo_model_parallel(graph: &EinsumGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up model parallelism with tensor sharding...");

    // Setup devices
    let devices = vec![Device::CPU(0), Device::CPU(1)];

    // Configure model parallelism with sharding
    let sharding_spec = ShardingSpec::new().shard_tensor("weights".to_string(), 0, 2); // Shard weights along dimension 0

    let config = DistributedConfig::new(devices.clone()).with_strategy(
        DistributedParallelismStrategy::ModelParallel {
            sharding_spec: sharding_spec.clone(),
        },
    );

    // Create distributed executor
    let backend = Box::new(DummyCommunicationBackend::new());
    let base_executor = DummyExecutor::new();
    let coordinator = ModelParallelCoordinator::new(devices, backend, sharding_spec);
    let mut executor = DistributedExecutor::new(base_executor, Box::new(coordinator));

    // Create sample inputs
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), DummyTensor::new(vec![64, 128]));
    inputs.insert("weights".to_string(), DummyTensor::new(vec![128, 256]));

    // Execute with model parallelism
    println!("Executing with model parallelism...");
    let _outputs = executor.execute_distributed(graph, &inputs, &config)?;

    // Get statistics
    let stats = executor.get_distributed_stats();
    println!("Communication time: {:.2}ms", stats.communication_time_ms);
    println!("Computation time: {:.2}ms", stats.computation_time_ms);
    println!("Shard overhead: {:.2}%", (1.0 - stats.efficiency) * 100.0);

    Ok(())
}

fn demo_pipeline_parallel(graph: &EinsumGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up pipeline parallelism across 4 stages...");

    // Setup devices (one per stage)
    let devices = vec![
        Device::CPU(0),
        Device::CPU(1),
        Device::CPU(2),
        Device::CPU(3),
    ];

    // Configure pipeline parallelism
    let config = DistributedConfig::new(devices.clone()).with_strategy(
        DistributedParallelismStrategy::PipelineParallel {
            num_stages: 4,
            micro_batch_size: 16,
        },
    );

    // Create distributed executor
    let backend = Box::new(DummyCommunicationBackend::new());
    let base_executor = DummyExecutor::new();
    let coordinator = PipelineParallelCoordinator::new(devices, backend, 4, 16);
    let mut executor = DistributedExecutor::new(base_executor, Box::new(coordinator));

    // Create sample inputs
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), DummyTensor::new(vec![64, 128]));
    inputs.insert("weights".to_string(), DummyTensor::new(vec![128, 256]));

    // Execute with pipeline parallelism
    println!("Executing with pipeline parallelism...");
    let _outputs = executor.execute_distributed(graph, &inputs, &config)?;

    // Get statistics
    let stats = executor.get_distributed_stats();
    println!("Pipeline stages: 4");
    println!("Micro-batch size: 16");
    println!("Pipeline efficiency: {:.2}%", stats.efficiency * 100.0);
    println!("Total time: {:.2}ms", stats.total_time_ms);

    Ok(())
}

fn demo_hybrid_parallel(graph: &EinsumGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up hybrid parallelism (data + model + pipeline)...");

    // Setup devices (8 devices for complex hybrid setup)
    let devices = vec![
        Device::CPU(0),
        Device::CPU(1),
        Device::CPU(2),
        Device::CPU(3),
        Device::CPU(4),
        Device::CPU(5),
        Device::CPU(6),
        Device::CPU(7),
    ];

    // Configure hybrid parallelism
    let config = DistributedConfig::new(devices.clone()).with_strategy(
        DistributedParallelismStrategy::Hybrid {
            data_parallel_groups: 2,
            model_parallel_size: 2,
            pipeline_stages: 2,
        },
    );

    // Create distributed executor
    let backend = Box::new(DummyCommunicationBackend::new());
    let base_executor = DummyExecutor::new();
    let coordinator = DataParallelCoordinator::new(devices, backend);
    let mut executor = DistributedExecutor::new(base_executor, Box::new(coordinator));

    // Create sample inputs
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), DummyTensor::new(vec![64, 128]));
    inputs.insert("weights".to_string(), DummyTensor::new(vec![128, 256]));

    // Execute with hybrid parallelism
    println!("Executing with hybrid parallelism...");
    let _outputs = executor.execute_distributed(graph, &inputs, &config)?;

    // Get statistics
    let stats = executor.get_distributed_stats();
    println!("Data parallel groups: 2");
    println!("Model parallel size: 2");
    println!("Pipeline stages: 2");
    println!("Total devices: 8");
    println!("Hybrid efficiency: {:.2}%", stats.efficiency * 100.0);
    println!(
        "Communication/computation ratio: {:.2}",
        stats.communication_time_ms / stats.computation_time_ms.max(1.0)
    );

    Ok(())
}
