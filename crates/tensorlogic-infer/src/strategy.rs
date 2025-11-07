//! Execution strategy configuration and policies.

/// Execution mode for graph evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Execute graph immediately on each operation
    Eager,
    /// Build computation graph and execute lazily
    Lazy,
    /// Hybrid: lazy within subgraphs, eager between stages
    Hybrid,
}

/// Gradient computation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientStrategy {
    /// No gradient computation
    None,
    /// Full gradient computation (standard backprop)
    Full,
    /// Checkpointing (recompute forward pass to save memory)
    Checkpointed {
        /// Checkpoint every N nodes
        checkpoint_interval: usize,
    },
    /// Gradient accumulation across multiple steps
    Accumulated {
        /// Number of steps to accumulate before updating
        accumulation_steps: usize,
    },
}

/// Precision mode for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Full precision (F64)
    Full,
    /// Single precision (F32)
    Single,
    /// Mixed precision (automatic F16/F32 selection)
    Mixed,
    /// Half precision (F16)
    Half,
}

/// Memory management strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Standard allocation/deallocation
    Standard,
    /// Reuse tensors aggressively
    Pooled,
    /// Cache intermediate results
    Cached,
    /// Minimize peak memory usage
    MinimalPeak,
}

/// Parallelism strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismStrategy {
    /// No parallelism (sequential execution)
    None,
    /// Data parallelism across batches
    DataParallel { num_workers: usize },
    /// Model parallelism across devices
    ModelParallel { num_devices: usize },
    /// Pipeline parallelism
    PipelineParallel { num_stages: usize },
    /// Automatic selection based on graph structure
    Automatic,
}

/// Complete execution strategy configuration
#[derive(Debug, Clone)]
pub struct ExecutionStrategy {
    pub mode: ExecutionMode,
    pub gradient: GradientStrategy,
    pub precision: PrecisionMode,
    pub memory: MemoryStrategy,
    pub parallelism: ParallelismStrategy,
    pub enable_fusion: bool,
    pub enable_profiling: bool,
}

impl ExecutionStrategy {
    /// Create a new execution strategy with defaults
    pub fn new() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Eager,
            gradient: GradientStrategy::None,
            precision: PrecisionMode::Full,
            memory: MemoryStrategy::Standard,
            parallelism: ParallelismStrategy::None,
            enable_fusion: false,
            enable_profiling: false,
        }
    }

    /// Training strategy with full gradients and profiling
    pub fn training() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Lazy,
            gradient: GradientStrategy::Full,
            precision: PrecisionMode::Single,
            memory: MemoryStrategy::Pooled,
            parallelism: ParallelismStrategy::Automatic,
            enable_fusion: true,
            enable_profiling: true,
        }
    }

    /// Inference strategy optimized for speed
    pub fn inference() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Eager,
            gradient: GradientStrategy::None,
            precision: PrecisionMode::Single,
            memory: MemoryStrategy::Cached,
            parallelism: ParallelismStrategy::Automatic,
            enable_fusion: true,
            enable_profiling: false,
        }
    }

    /// Memory-efficient strategy for large models
    pub fn memory_efficient() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Hybrid,
            gradient: GradientStrategy::Checkpointed {
                checkpoint_interval: 10,
            },
            precision: PrecisionMode::Mixed,
            memory: MemoryStrategy::MinimalPeak,
            parallelism: ParallelismStrategy::None,
            enable_fusion: false,
            enable_profiling: false,
        }
    }

    /// High-throughput strategy for batch processing
    pub fn high_throughput() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Lazy,
            gradient: GradientStrategy::None,
            precision: PrecisionMode::Single,
            memory: MemoryStrategy::Pooled,
            parallelism: ParallelismStrategy::DataParallel { num_workers: 4 },
            enable_fusion: true,
            enable_profiling: false,
        }
    }

    /// Development/debugging strategy with profiling enabled
    pub fn debug() -> Self {
        ExecutionStrategy {
            mode: ExecutionMode::Eager,
            gradient: GradientStrategy::Full,
            precision: PrecisionMode::Full,
            memory: MemoryStrategy::Standard,
            parallelism: ParallelismStrategy::None,
            enable_fusion: false,
            enable_profiling: true,
        }
    }

    // Builder methods
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_gradient(mut self, gradient: GradientStrategy) -> Self {
        self.gradient = gradient;
        self
    }

    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_memory(mut self, memory: MemoryStrategy) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_parallelism(mut self, parallelism: ParallelismStrategy) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn enable_fusion(mut self) -> Self {
        self.enable_fusion = true;
        self
    }

    pub fn enable_profiling(mut self) -> Self {
        self.enable_profiling = true;
        self
    }

    /// Check if gradient computation is enabled
    pub fn computes_gradients(&self) -> bool {
        !matches!(self.gradient, GradientStrategy::None)
    }

    /// Check if strategy uses checkpointing
    pub fn uses_checkpointing(&self) -> bool {
        matches!(self.gradient, GradientStrategy::Checkpointed { .. })
    }

    /// Check if strategy is optimized for inference
    pub fn is_inference_mode(&self) -> bool {
        matches!(self.gradient, GradientStrategy::None)
    }

    /// Get checkpoint interval if using checkpointing
    pub fn checkpoint_interval(&self) -> Option<usize> {
        match self.gradient {
            GradientStrategy::Checkpointed {
                checkpoint_interval,
            } => Some(checkpoint_interval),
            _ => None,
        }
    }

    /// Get gradient accumulation steps if using accumulation
    pub fn accumulation_steps(&self) -> Option<usize> {
        match self.gradient {
            GradientStrategy::Accumulated { accumulation_steps } => Some(accumulation_steps),
            _ => None,
        }
    }

    /// Get number of parallel workers
    pub fn num_workers(&self) -> usize {
        match self.parallelism {
            ParallelismStrategy::None => 1,
            ParallelismStrategy::DataParallel { num_workers } => num_workers,
            ParallelismStrategy::ModelParallel { num_devices } => num_devices,
            ParallelismStrategy::PipelineParallel { num_stages } => num_stages,
            ParallelismStrategy::Automatic => num_cpus::get().min(8),
        }
    }

    /// Summary description of the strategy
    pub fn summary(&self) -> String {
        format!(
            "Execution Strategy:\n\
             - Mode: {:?}\n\
             - Gradient: {:?}\n\
             - Precision: {:?}\n\
             - Memory: {:?}\n\
             - Parallelism: {:?}\n\
             - Fusion: {}\n\
             - Profiling: {}",
            self.mode,
            self.gradient,
            self.precision,
            self.memory,
            self.parallelism,
            self.enable_fusion,
            self.enable_profiling
        )
    }
}

impl Default for ExecutionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy optimizer for automatic strategy selection
pub struct StrategyOptimizer;

impl StrategyOptimizer {
    /// Recommend strategy based on workload characteristics
    pub fn recommend(
        batch_size: usize,
        model_size_mb: usize,
        available_memory_mb: usize,
        is_training: bool,
    ) -> ExecutionStrategy {
        let memory_pressure = (model_size_mb * batch_size) as f64 / available_memory_mb as f64;

        if is_training {
            if memory_pressure > 0.8 {
                // High memory pressure: use checkpointing
                ExecutionStrategy::training().with_gradient(GradientStrategy::Checkpointed {
                    checkpoint_interval: 5,
                })
            } else if batch_size >= 64 {
                // Large batch: use accumulation
                ExecutionStrategy::training().with_gradient(GradientStrategy::Accumulated {
                    accumulation_steps: 4,
                })
            } else {
                ExecutionStrategy::training()
            }
        } else {
            // Inference
            if batch_size >= 32 {
                ExecutionStrategy::high_throughput()
            } else {
                ExecutionStrategy::inference()
            }
        }
    }

    /// Estimate memory overhead for a strategy
    pub fn estimate_memory_overhead(strategy: &ExecutionStrategy) -> f64 {
        let mut overhead = 1.0;

        // Execution mode overhead
        overhead *= match strategy.mode {
            ExecutionMode::Eager => 1.0,
            ExecutionMode::Lazy => 1.2, // Graph storage
            ExecutionMode::Hybrid => 1.1,
        };

        // Gradient overhead
        overhead *= match strategy.gradient {
            GradientStrategy::None => 1.0,
            GradientStrategy::Full => 3.0, // Forward + backward + gradients
            GradientStrategy::Checkpointed { .. } => 2.0, // Reduced memory
            GradientStrategy::Accumulated { .. } => 3.5, // Extra gradient buffers
        };

        // Memory strategy adjustment
        overhead *= match strategy.memory {
            MemoryStrategy::Standard => 1.0,
            MemoryStrategy::Pooled => 1.1,      // Pool overhead
            MemoryStrategy::Cached => 1.3,      // Cache overhead
            MemoryStrategy::MinimalPeak => 0.8, // Reduced peak
        };

        overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_strategy_presets() {
        let training = ExecutionStrategy::training();
        assert!(training.computes_gradients());
        assert!(training.enable_fusion);

        let inference = ExecutionStrategy::inference();
        assert!(!inference.computes_gradients());
        assert!(inference.is_inference_mode());

        let memory_eff = ExecutionStrategy::memory_efficient();
        assert!(memory_eff.uses_checkpointing());

        let throughput = ExecutionStrategy::high_throughput();
        assert!(throughput.num_workers() > 1);

        let debug = ExecutionStrategy::debug();
        assert!(debug.enable_profiling);
    }

    #[test]
    fn test_execution_strategy_builder() {
        let strategy = ExecutionStrategy::new()
            .with_mode(ExecutionMode::Lazy)
            .with_precision(PrecisionMode::Single)
            .enable_fusion()
            .enable_profiling();

        assert_eq!(strategy.mode, ExecutionMode::Lazy);
        assert_eq!(strategy.precision, PrecisionMode::Single);
        assert!(strategy.enable_fusion);
        assert!(strategy.enable_profiling);
    }

    #[test]
    fn test_gradient_strategies() {
        let no_grad = ExecutionStrategy::new().with_gradient(GradientStrategy::None);
        assert!(!no_grad.computes_gradients());

        let full_grad = ExecutionStrategy::new().with_gradient(GradientStrategy::Full);
        assert!(full_grad.computes_gradients());

        let checkpointed = ExecutionStrategy::new().with_gradient(GradientStrategy::Checkpointed {
            checkpoint_interval: 10,
        });
        assert!(checkpointed.uses_checkpointing());
        assert_eq!(checkpointed.checkpoint_interval(), Some(10));

        let accumulated = ExecutionStrategy::new().with_gradient(GradientStrategy::Accumulated {
            accumulation_steps: 4,
        });
        assert_eq!(accumulated.accumulation_steps(), Some(4));
    }

    #[test]
    fn test_parallelism_strategies() {
        let sequential = ExecutionStrategy::new().with_parallelism(ParallelismStrategy::None);
        assert_eq!(sequential.num_workers(), 1);

        let data_parallel = ExecutionStrategy::new()
            .with_parallelism(ParallelismStrategy::DataParallel { num_workers: 4 });
        assert_eq!(data_parallel.num_workers(), 4);

        let automatic = ExecutionStrategy::new().with_parallelism(ParallelismStrategy::Automatic);
        assert!(automatic.num_workers() >= 1);
    }

    #[test]
    fn test_strategy_optimizer_recommendations() {
        // Low memory, training
        let strategy1 = StrategyOptimizer::recommend(32, 1000, 2000, true);
        assert!(strategy1.computes_gradients());

        // High memory pressure, training
        let strategy2 = StrategyOptimizer::recommend(64, 2000, 2000, true);
        assert!(strategy2.uses_checkpointing() || strategy2.accumulation_steps().is_some());

        // Inference, large batch
        let strategy3 = StrategyOptimizer::recommend(64, 500, 4000, false);
        assert!(!strategy3.computes_gradients());

        // Inference, small batch
        let strategy4 = StrategyOptimizer::recommend(8, 500, 4000, false);
        assert!(!strategy4.computes_gradients());
    }

    #[test]
    fn test_memory_overhead_estimation() {
        let eager_no_grad = ExecutionStrategy::new();
        let overhead1 = StrategyOptimizer::estimate_memory_overhead(&eager_no_grad);
        assert_eq!(overhead1, 1.0); // Baseline

        let training = ExecutionStrategy::training();
        let overhead2 = StrategyOptimizer::estimate_memory_overhead(&training);
        assert!(overhead2 > 2.0); // Should have significant overhead

        let memory_eff = ExecutionStrategy::memory_efficient();
        let overhead3 = StrategyOptimizer::estimate_memory_overhead(&memory_eff);
        assert!(overhead3 < overhead2); // Should be more efficient than full training
    }

    #[test]
    fn test_execution_modes() {
        assert_eq!(ExecutionMode::Eager, ExecutionMode::Eager);
        assert_ne!(ExecutionMode::Eager, ExecutionMode::Lazy);
    }

    #[test]
    fn test_precision_modes() {
        let modes = vec![
            PrecisionMode::Full,
            PrecisionMode::Single,
            PrecisionMode::Mixed,
            PrecisionMode::Half,
        ];

        for mode in modes {
            let strategy = ExecutionStrategy::new().with_precision(mode);
            assert_eq!(strategy.precision, mode);
        }
    }

    #[test]
    fn test_strategy_summary() {
        let strategy = ExecutionStrategy::training();
        let summary = strategy.summary();

        assert!(summary.contains("Execution Strategy"));
        assert!(summary.contains("Mode"));
        assert!(summary.contains("Gradient"));
        assert!(summary.contains("Precision"));
    }
}
