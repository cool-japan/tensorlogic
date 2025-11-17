//! Distributed execution infrastructure for multi-device and multi-node computation.
//!
//! This module provides distributed training and inference capabilities:
//! - `DistributedExecutor`: Multi-device execution coordination
//! - `DataParallelism`: Data-parallel training across devices
//! - `ModelParallelism`: Model-parallel execution with tensor sharding
//! - `CommunicationBackend`: Abstract interface for device communication
//! - `TlDistributedExecutor`: Trait for executors that support distributed execution
//!
//! # Parallelism Strategies
//!
//! ## Data Parallelism
//! - Each device processes a different subset of the batch
//! - Gradients are averaged across devices
//! - Suitable for models that fit on a single device
//!
//! ## Model Parallelism
//! - Model is split across multiple devices
//! - Each device processes different parts of the model
//! - Suitable for large models that don't fit on a single device
//!
//! ## Hybrid Parallelism
//! - Combines data and model parallelism
//! - Model is split across devices, each replica processes different data
//!
//! # Example
//!
//! ```
//! use tensorlogic_infer::distributed::{DistributedConfig, ParallelismStrategy};
//!
//! let config = DistributedConfig {
//!     parallelism: ParallelismStrategy::DataParallel,
//!     num_devices: 4,
//!     ..Default::default()
//! };
//! ```

use crate::capabilities::DeviceType;
use crate::error::ExecutorError;
use crate::placement::Device;
use crate::shape::TensorShape;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tensorlogic_ir::EinsumGraph;

/// Parallelism strategy for distributed execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ParallelismStrategy {
    /// Data parallelism: each device processes different data
    #[default]
    DataParallel,
    /// Model parallelism: model is split across devices
    ModelParallel,
    /// Pipeline parallelism: model stages on different devices
    PipelineParallel,
    /// Hybrid: combination of data and model parallelism
    Hybrid { data_parallel_groups: usize },
}

/// Configuration for distributed execution.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Parallelism strategy to use
    pub parallelism: ParallelismStrategy,
    /// Number of devices to use
    pub num_devices: usize,
    /// Communication backend (e.g., "nccl", "gloo", "mpi")
    pub backend: String,
    /// Master address for multi-node setups
    pub master_addr: Option<String>,
    /// Master port for multi-node setups
    pub master_port: Option<u16>,
    /// Rank of this process (0 to world_size-1)
    pub rank: usize,
    /// Total number of processes (world size)
    pub world_size: usize,
    /// Enable gradient compression
    pub enable_gradient_compression: bool,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
    /// Bucket size for gradient bucketing (bytes)
    pub bucket_size: usize,
    /// Enable asynchronous communication
    pub enable_async_communication: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        DistributedConfig {
            parallelism: ParallelismStrategy::default(),
            num_devices: 1,
            backend: "gloo".to_string(),
            master_addr: None,
            master_port: None,
            rank: 0,
            world_size: 1,
            enable_gradient_compression: false,
            enable_mixed_precision: false,
            bucket_size: 25 * 1024 * 1024, // 25MB
            enable_async_communication: true,
        }
    }
}

/// Tensor sharding specification for model parallelism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardingSpec {
    /// Node ID that this sharding applies to
    pub node_id: usize,
    /// Dimension along which to shard
    pub shard_dim: usize,
    /// Number of shards
    pub num_shards: usize,
    /// Device assignment for each shard
    pub shard_to_device: Vec<Device>,
}

impl ShardingSpec {
    /// Create a new sharding specification.
    pub fn new(node_id: usize, shard_dim: usize, devices: Vec<Device>) -> Self {
        let num_shards = devices.len();
        ShardingSpec {
            node_id,
            shard_dim,
            num_shards,
            shard_to_device: devices,
        }
    }

    /// Get the device for a specific shard.
    pub fn device_for_shard(&self, shard_id: usize) -> Option<&Device> {
        self.shard_to_device.get(shard_id)
    }

    /// Check if a shard ID is valid.
    pub fn is_valid_shard(&self, shard_id: usize) -> bool {
        shard_id < self.num_shards
    }
}

/// Placement plan for distributed execution.
#[derive(Debug, Clone)]
pub struct DistributedPlacementPlan {
    /// Node to device mapping
    pub node_placement: HashMap<usize, Device>,
    /// Sharding specifications for model parallelism
    pub sharding_specs: Vec<ShardingSpec>,
    /// Communication dependencies (node -> nodes it depends on)
    pub communication_deps: HashMap<usize, Vec<usize>>,
}

impl DistributedPlacementPlan {
    /// Create a new empty placement plan.
    pub fn new() -> Self {
        DistributedPlacementPlan {
            node_placement: HashMap::new(),
            sharding_specs: Vec::new(),
            communication_deps: HashMap::new(),
        }
    }

    /// Add a node placement.
    pub fn place_node(&mut self, node_id: usize, device: Device) {
        self.node_placement.insert(node_id, device);
    }

    /// Add a sharding specification.
    pub fn add_sharding(&mut self, spec: ShardingSpec) {
        self.sharding_specs.push(spec);
    }

    /// Get the device for a node.
    pub fn get_device(&self, node_id: usize) -> Option<&Device> {
        self.node_placement.get(&node_id)
    }

    /// Get sharding spec for a node.
    pub fn get_sharding(&self, node_id: usize) -> Option<&ShardingSpec> {
        self.sharding_specs.iter().find(|s| s.node_id == node_id)
    }

    /// Check if a node is sharded.
    pub fn is_sharded(&self, node_id: usize) -> bool {
        self.get_sharding(node_id).is_some()
    }
}

impl Default for DistributedPlacementPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Communication operation for distributed execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommunicationOp {
    /// All-reduce: reduce values across all devices
    AllReduce {
        /// Reduction operation (sum, mean, max, min)
        reduction: ReductionOp,
    },
    /// Broadcast: send from one device to all others
    Broadcast {
        /// Source device rank
        src_rank: usize,
    },
    /// Scatter: distribute data from one device to all
    Scatter {
        /// Source device rank
        src_rank: usize,
    },
    /// Gather: collect data from all devices to one
    Gather {
        /// Destination device rank
        dst_rank: usize,
    },
    /// All-gather: gather data from all devices to all
    AllGather,
    /// Reduce-scatter: reduce and scatter results
    ReduceScatter {
        /// Reduction operation
        reduction: ReductionOp,
    },
    /// Peer-to-peer send
    Send {
        /// Destination device rank
        dst_rank: usize,
    },
    /// Peer-to-peer receive
    Recv {
        /// Source device rank
        src_rank: usize,
    },
}

/// Reduction operation for communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Sum reduction
    Sum,
    /// Mean reduction (sum / count)
    Mean,
    /// Maximum value
    Max,
    /// Minimum value
    Min,
    /// Product
    Product,
}

/// Abstract communication backend for device-to-device communication.
pub trait CommunicationBackend: Send + Sync {
    /// Initialize the communication backend.
    fn initialize(&mut self, config: &DistributedConfig) -> Result<(), ExecutorError>;

    /// Finalize and clean up the backend.
    fn finalize(&mut self) -> Result<(), ExecutorError>;

    /// Get the rank of this process.
    fn rank(&self) -> usize;

    /// Get the world size (total number of processes).
    fn world_size(&self) -> usize;

    /// Perform an all-reduce operation.
    fn all_reduce(&self, tensor_id: &str, reduction: ReductionOp) -> Result<(), ExecutorError>;

    /// Broadcast from source rank to all ranks.
    fn broadcast(&self, tensor_id: &str, src_rank: usize) -> Result<(), ExecutorError>;

    /// Scatter data from source rank to all ranks.
    fn scatter(&self, tensor_id: &str, src_rank: usize) -> Result<(), ExecutorError>;

    /// Gather data from all ranks to destination rank.
    fn gather(&self, tensor_id: &str, dst_rank: usize) -> Result<(), ExecutorError>;

    /// All-gather operation.
    fn all_gather(&self, tensor_id: &str) -> Result<(), ExecutorError>;

    /// Reduce-scatter operation.
    fn reduce_scatter(&self, tensor_id: &str, reduction: ReductionOp) -> Result<(), ExecutorError>;

    /// Point-to-point send.
    fn send(&self, tensor_id: &str, dst_rank: usize) -> Result<(), ExecutorError>;

    /// Point-to-point receive.
    fn recv(&self, tensor_id: &str, src_rank: usize) -> Result<(), ExecutorError>;

    /// Synchronize all processes.
    fn barrier(&self) -> Result<(), ExecutorError>;
}

/// Dummy communication backend for testing.
pub struct DummyCommunicationBackend {
    rank: usize,
    world_size: usize,
}

impl DummyCommunicationBackend {
    /// Create a new dummy backend.
    pub fn new() -> Self {
        DummyCommunicationBackend {
            rank: 0,
            world_size: 1,
        }
    }
}

impl Default for DummyCommunicationBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunicationBackend for DummyCommunicationBackend {
    fn initialize(&mut self, config: &DistributedConfig) -> Result<(), ExecutorError> {
        self.rank = config.rank;
        self.world_size = config.world_size;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, _tensor_id: &str, _reduction: ReductionOp) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn broadcast(&self, _tensor_id: &str, _src_rank: usize) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn scatter(&self, _tensor_id: &str, _src_rank: usize) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn gather(&self, _tensor_id: &str, _dst_rank: usize) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn all_gather(&self, _tensor_id: &str) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn reduce_scatter(
        &self,
        _tensor_id: &str,
        _reduction: ReductionOp,
    ) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn send(&self, _tensor_id: &str, _dst_rank: usize) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn recv(&self, _tensor_id: &str, _src_rank: usize) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Data parallelism coordinator.
pub struct DataParallelCoordinator {
    config: DistributedConfig,
    backend: Arc<RwLock<dyn CommunicationBackend>>,
    devices: Vec<Device>,
}

impl DataParallelCoordinator {
    /// Create a new data parallel coordinator.
    pub fn new(config: DistributedConfig, backend: Arc<RwLock<dyn CommunicationBackend>>) -> Self {
        let devices = (0..config.num_devices)
            .map(|i| Device::new(DeviceType::CPU, i))
            .collect();

        DataParallelCoordinator {
            config,
            backend,
            devices,
        }
    }

    /// Distribute batch across devices.
    pub fn distribute_batch(&self, batch_size: usize) -> Vec<(usize, usize)> {
        let per_device = batch_size / self.config.num_devices;
        let remainder = batch_size % self.config.num_devices;

        let mut distribution = Vec::new();
        let mut offset = 0;

        for i in 0..self.config.num_devices {
            let size = per_device + if i < remainder { 1 } else { 0 };
            distribution.push((offset, size));
            offset += size;
        }

        distribution
    }

    /// Synchronize gradients across devices.
    pub fn synchronize_gradients(&self) -> Result<(), ExecutorError> {
        let backend = self.backend.read().unwrap();

        // All-reduce gradients with mean reduction
        backend.all_reduce("gradients", ReductionOp::Mean)?;

        Ok(())
    }

    /// Get the list of devices.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }
}

/// Model parallelism coordinator.
pub struct ModelParallelCoordinator {
    config: DistributedConfig,
    backend: Arc<RwLock<dyn CommunicationBackend>>,
    placement_plan: DistributedPlacementPlan,
}

impl ModelParallelCoordinator {
    /// Create a new model parallel coordinator.
    pub fn new(config: DistributedConfig, backend: Arc<RwLock<dyn CommunicationBackend>>) -> Self {
        ModelParallelCoordinator {
            config,
            backend,
            placement_plan: DistributedPlacementPlan::new(),
        }
    }

    /// Create a sharding plan for the graph.
    pub fn create_sharding_plan(&mut self, graph: &EinsumGraph) -> Result<(), ExecutorError> {
        let num_devices = self.config.num_devices;
        let nodes_per_device = graph.nodes.len().div_ceil(num_devices);

        // Simple sharding: distribute nodes across devices
        for (node_id, _node) in graph.nodes.iter().enumerate() {
            let device_idx = node_id / nodes_per_device;
            let device = Device::new(DeviceType::CPU, device_idx);
            self.placement_plan.place_node(node_id, device);
        }

        Ok(())
    }

    /// Get the placement plan.
    pub fn placement_plan(&self) -> &DistributedPlacementPlan {
        &self.placement_plan
    }

    /// Shard a tensor along a dimension.
    pub fn shard_tensor(
        &self,
        _node_id: usize,
        shape: &TensorShape,
        shard_dim: usize,
    ) -> Result<Vec<TensorShape>, ExecutorError> {
        let num_shards = self.config.num_devices;

        if shard_dim >= shape.rank() {
            return Err(ExecutorError::InvalidInput(format!(
                "Shard dimension {} exceeds tensor rank {}",
                shard_dim,
                shape.rank()
            )));
        }

        let total_size = shape.dims[shard_dim].as_static().ok_or_else(|| {
            ExecutorError::InvalidInput("Cannot shard dynamic dimension".to_string())
        })?;

        let per_shard = total_size / num_shards;
        let remainder = total_size % num_shards;

        let mut shard_shapes = Vec::new();
        for i in 0..num_shards {
            let shard_size = per_shard + if i < remainder { 1 } else { 0 };
            let mut shard_shape = shape.clone();
            shard_shape.dims[shard_dim] = crate::shape::DimSize::Static(shard_size);
            shard_shapes.push(shard_shape);
        }

        Ok(shard_shapes)
    }

    /// Gather sharded tensors.
    pub fn gather_shards(&self, _shard_dim: usize) -> Result<(), ExecutorError> {
        let backend = self.backend.read().unwrap();
        backend.all_gather("sharded_tensor")?;
        Ok(())
    }
}

/// Pipeline parallelism coordinator.
pub struct PipelineParallelCoordinator {
    config: DistributedConfig,
    backend: Arc<RwLock<dyn CommunicationBackend>>,
    num_stages: usize,
    micro_batch_size: usize,
}

impl PipelineParallelCoordinator {
    /// Create a new pipeline parallel coordinator.
    pub fn new(
        config: DistributedConfig,
        backend: Arc<RwLock<dyn CommunicationBackend>>,
        num_stages: usize,
    ) -> Self {
        PipelineParallelCoordinator {
            config,
            backend,
            num_stages,
            micro_batch_size: 1,
        }
    }

    /// Set micro-batch size for pipeline parallelism.
    pub fn set_micro_batch_size(&mut self, size: usize) {
        self.micro_batch_size = size;
    }

    /// Get the stage assignment for this rank.
    pub fn stage_for_rank(&self, rank: usize) -> usize {
        rank % self.num_stages
    }

    /// Send activations to next stage.
    pub fn send_activations(&self, stage: usize) -> Result<(), ExecutorError> {
        if stage < self.num_stages - 1 {
            let next_rank = stage + 1;
            let backend = self.backend.read().unwrap();
            backend.send("activations", next_rank)?;
        }
        Ok(())
    }

    /// Receive activations from previous stage.
    pub fn recv_activations(&self, stage: usize) -> Result<(), ExecutorError> {
        if stage > 0 {
            let prev_rank = stage - 1;
            let backend = self.backend.read().unwrap();
            backend.recv("activations", prev_rank)?;
        }
        Ok(())
    }

    /// Send gradients to previous stage.
    pub fn send_gradients(&self, stage: usize) -> Result<(), ExecutorError> {
        if stage > 0 {
            let prev_rank = stage - 1;
            let backend = self.backend.read().unwrap();
            backend.send("gradients", prev_rank)?;
        }
        Ok(())
    }

    /// Receive gradients from next stage.
    pub fn recv_gradients(&self, stage: usize) -> Result<(), ExecutorError> {
        if stage < self.num_stages - 1 {
            let next_rank = stage + 1;
            let backend = self.backend.read().unwrap();
            backend.recv("gradients", next_rank)?;
        }
        Ok(())
    }

    /// Get the number of stages in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Get the micro-batch size.
    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}

/// Distributed executor that coordinates multi-device execution.
pub struct DistributedExecutor {
    config: DistributedConfig,
    backend: Arc<RwLock<dyn CommunicationBackend>>,
    data_parallel: Option<DataParallelCoordinator>,
    model_parallel: Option<ModelParallelCoordinator>,
    pipeline_parallel: Option<PipelineParallelCoordinator>,
}

impl DistributedExecutor {
    /// Create a new distributed executor.
    pub fn new(
        config: DistributedConfig,
        backend: Arc<RwLock<dyn CommunicationBackend>>,
    ) -> Result<Self, ExecutorError> {
        // Initialize backend
        backend.write().unwrap().initialize(&config)?;

        let mut executor = DistributedExecutor {
            config: config.clone(),
            backend: backend.clone(),
            data_parallel: None,
            model_parallel: None,
            pipeline_parallel: None,
        };

        // Setup coordinators based on strategy
        executor.setup_coordinators()?;

        Ok(executor)
    }

    /// Setup coordinators based on parallelism strategy.
    fn setup_coordinators(&mut self) -> Result<(), ExecutorError> {
        match self.config.parallelism {
            ParallelismStrategy::DataParallel => {
                self.data_parallel = Some(DataParallelCoordinator::new(
                    self.config.clone(),
                    self.backend.clone(),
                ));
            }
            ParallelismStrategy::ModelParallel => {
                self.model_parallel = Some(ModelParallelCoordinator::new(
                    self.config.clone(),
                    self.backend.clone(),
                ));
            }
            ParallelismStrategy::PipelineParallel => {
                let num_stages = self.config.num_devices;
                self.pipeline_parallel = Some(PipelineParallelCoordinator::new(
                    self.config.clone(),
                    self.backend.clone(),
                    num_stages,
                ));
            }
            ParallelismStrategy::Hybrid {
                data_parallel_groups: _,
            } => {
                self.data_parallel = Some(DataParallelCoordinator::new(
                    self.config.clone(),
                    self.backend.clone(),
                ));
                self.model_parallel = Some(ModelParallelCoordinator::new(
                    self.config.clone(),
                    self.backend.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Get the parallelism strategy.
    pub fn strategy(&self) -> ParallelismStrategy {
        self.config.parallelism
    }

    /// Get the rank of this process.
    pub fn rank(&self) -> usize {
        self.backend.read().unwrap().rank()
    }

    /// Get the world size.
    pub fn world_size(&self) -> usize {
        self.backend.read().unwrap().world_size()
    }

    /// Synchronize all processes.
    pub fn barrier(&self) -> Result<(), ExecutorError> {
        self.backend.read().unwrap().barrier()
    }

    /// Get data parallel coordinator.
    pub fn data_parallel(&self) -> Option<&DataParallelCoordinator> {
        self.data_parallel.as_ref()
    }

    /// Get model parallel coordinator.
    pub fn model_parallel(&self) -> Option<&ModelParallelCoordinator> {
        self.model_parallel.as_ref()
    }

    /// Get pipeline parallel coordinator.
    pub fn pipeline_parallel(&self) -> Option<&PipelineParallelCoordinator> {
        self.pipeline_parallel.as_ref()
    }
}

impl Drop for DistributedExecutor {
    fn drop(&mut self) {
        let _ = self.backend.write().unwrap().finalize();
    }
}

/// Trait for executors that support distributed execution.
pub trait TlDistributedExecutor {
    /// Get the distributed executor.
    fn distributed_executor(&self) -> Option<&DistributedExecutor>;

    /// Enable distributed execution.
    fn enable_distributed(&mut self, config: DistributedConfig) -> Result<(), ExecutorError>;

    /// Disable distributed execution.
    fn disable_distributed(&mut self);

    /// Check if distributed execution is enabled.
    fn is_distributed(&self) -> bool;

    /// Get the current rank.
    fn rank(&self) -> usize {
        self.distributed_executor().map(|d| d.rank()).unwrap_or(0)
    }

    /// Get the world size.
    fn world_size(&self) -> usize {
        self.distributed_executor()
            .map(|d| d.world_size())
            .unwrap_or(1)
    }
}

/// Statistics for distributed execution.
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total number of communication operations
    pub total_communications: usize,
    /// Total bytes communicated
    pub total_bytes_communicated: u64,
    /// Number of gradient synchronizations
    pub gradient_syncs: usize,
    /// Average communication time
    pub avg_communication_time_ms: f64,
    /// Load imbalance metric (0.0 = perfect, 1.0 = worst)
    pub load_imbalance: f64,
}

impl DistributedStats {
    /// Get a summary of distributed execution statistics.
    pub fn summary(&self) -> String {
        format!(
            "Distributed Stats: {} communications, {:.2} MB transferred, {} gradient syncs, {:.2}ms avg comm time, {:.2}% load imbalance",
            self.total_communications,
            self.total_bytes_communicated as f64 / 1_000_000.0,
            self.gradient_syncs,
            self.avg_communication_time_ms,
            self.load_imbalance * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.parallelism, ParallelismStrategy::DataParallel);
        assert_eq!(config.num_devices, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.world_size, 1);
    }

    #[test]
    fn test_sharding_spec() {
        let devices = vec![
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CPU, 1),
            Device::new(DeviceType::CPU, 2),
        ];
        let spec = ShardingSpec::new(0, 1, devices);

        assert_eq!(spec.num_shards, 3);
        assert_eq!(spec.shard_dim, 1);
        assert!(spec.is_valid_shard(0));
        assert!(spec.is_valid_shard(2));
        assert!(!spec.is_valid_shard(3));
    }

    #[test]
    fn test_distributed_placement_plan() {
        let mut plan = DistributedPlacementPlan::new();

        plan.place_node(0, Device::new(DeviceType::CPU, 0));
        plan.place_node(1, Device::new(DeviceType::CPU, 1));

        assert!(plan.get_device(0).is_some());
        assert!(plan.get_device(1).is_some());
        assert!(plan.get_device(2).is_none());
    }

    #[test]
    fn test_data_parallel_batch_distribution() {
        let config = DistributedConfig {
            num_devices: 4,
            ..Default::default()
        };
        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));
        let coordinator = DataParallelCoordinator::new(config, backend);

        let distribution = coordinator.distribute_batch(10);
        assert_eq!(distribution.len(), 4);

        // Check total size
        let total: usize = distribution.iter().map(|(_, size)| size).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_model_parallel_sharding() {
        let config = DistributedConfig {
            num_devices: 4,
            parallelism: ParallelismStrategy::ModelParallel,
            ..Default::default()
        };
        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));
        let coordinator = ModelParallelCoordinator::new(config, backend);

        let shape = TensorShape::static_shape(vec![8, 16]);
        let shards = coordinator.shard_tensor(0, &shape, 0).unwrap();

        assert_eq!(shards.len(), 4);
        // Each shard should have size 2 in dimension 0
        assert_eq!(shards[0].dims[0].as_static().unwrap(), 2);
    }

    #[test]
    fn test_pipeline_parallel_stage_assignment() {
        let config = DistributedConfig {
            num_devices: 4,
            parallelism: ParallelismStrategy::PipelineParallel,
            ..Default::default()
        };
        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));
        let coordinator = PipelineParallelCoordinator::new(config, backend, 4);

        assert_eq!(coordinator.stage_for_rank(0), 0);
        assert_eq!(coordinator.stage_for_rank(1), 1);
        assert_eq!(coordinator.stage_for_rank(2), 2);
        assert_eq!(coordinator.stage_for_rank(3), 3);
    }

    #[test]
    fn test_distributed_executor_creation() {
        let config = DistributedConfig::default();
        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));

        let executor = DistributedExecutor::new(config, backend);
        assert!(executor.is_ok());

        let executor = executor.unwrap();
        assert_eq!(executor.rank(), 0);
        assert_eq!(executor.world_size(), 1);
    }

    #[test]
    fn test_communication_ops() {
        let op1 = CommunicationOp::AllReduce {
            reduction: ReductionOp::Sum,
        };
        let op2 = CommunicationOp::Broadcast { src_rank: 0 };

        assert_ne!(op1, op2);
    }

    #[test]
    fn test_reduction_ops() {
        let ops = [
            ReductionOp::Sum,
            ReductionOp::Mean,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Product,
        ];

        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn test_dummy_backend() {
        let mut backend = DummyCommunicationBackend::new();
        let config = DistributedConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.world_size(), 1);
        assert!(backend.all_reduce("test", ReductionOp::Sum).is_ok());
        assert!(backend.barrier().is_ok());
        assert!(backend.finalize().is_ok());
    }

    #[test]
    fn test_distributed_stats() {
        let stats = DistributedStats {
            total_communications: 100,
            total_bytes_communicated: 1_000_000,
            gradient_syncs: 50,
            avg_communication_time_ms: 10.5,
            load_imbalance: 0.15,
        };

        let summary = stats.summary();
        assert!(summary.contains("100 communications"));
        assert!(summary.contains("50 gradient syncs"));
    }

    #[test]
    fn test_hybrid_parallelism() {
        let config = DistributedConfig {
            parallelism: ParallelismStrategy::Hybrid {
                data_parallel_groups: 2,
            },
            num_devices: 8,
            ..Default::default()
        };

        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));
        let executor = DistributedExecutor::new(config, backend).unwrap();

        assert!(executor.data_parallel().is_some());
        assert!(executor.model_parallel().is_some());
    }

    #[test]
    fn test_sharding_invalid_dimension() {
        let config = DistributedConfig {
            num_devices: 4,
            ..Default::default()
        };
        let backend = Arc::new(RwLock::new(DummyCommunicationBackend::new()));
        let coordinator = ModelParallelCoordinator::new(config, backend);

        let shape = TensorShape::static_shape(vec![8, 16]);
        let result = coordinator.shard_tensor(0, &shape, 5);

        assert!(result.is_err());
    }
}
