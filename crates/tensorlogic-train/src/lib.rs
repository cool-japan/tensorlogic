//! Training scaffolds: loss wiring, schedules, callbacks.
//!
//! **Version**: 0.1.0-alpha.2 | **Status**: Production Ready
//!
//! This crate provides comprehensive training infrastructure for Tensorlogic models:
//! - Loss functions (standard and logical constraint-based)
//! - Optimizer wrappers around SciRS2
//! - Training loops with callbacks
//! - Batch management
//! - Validation and metrics
//! - Regularization techniques
//! - Data augmentation
//! - Logging and monitoring
//! - Curriculum learning strategies
//! - Transfer learning utilities
//! - Hyperparameter optimization (grid search, random search)
//! - Cross-validation utilities
//! - Model ensembling
//! - Model pruning and compression
//! - Model quantization (int8, int4, int2)
//! - Mixed precision training (FP16, BF16)
//! - Advanced sampling strategies

mod augmentation;
mod batch;
mod callbacks;
mod crossval;
mod curriculum;
mod data;
mod distillation;
mod dropblock;
mod ensemble;
mod error;
mod few_shot;
mod gradient_centralization;
mod hyperparameter;
mod label_smoothing;
mod logging;
mod loss;
mod memory;
mod meta_learning;
mod metrics;
mod mixed_precision;
mod model;
mod multitask;
mod optimizer;
mod optimizers;
mod pruning;
mod quantization;
mod regularization;
mod sampling;
mod scheduler;
mod stochastic_depth;
mod trainer;
mod transfer;
mod utils;

#[cfg(feature = "structured-logging")]
pub mod structured_logging;

pub use augmentation::{
    CompositeAugmenter, CutMixAugmenter, CutOutAugmenter, DataAugmenter, MixupAugmenter,
    NoAugmentation, NoiseAugmenter, RandomErasingAugmenter, RotationAugmenter, ScaleAugmenter,
};
pub use batch::{extract_batch, BatchConfig, BatchIterator, DataShuffler};
pub use callbacks::{
    BatchCallback, Callback, CallbackList, CheckpointCallback, CheckpointCompression,
    EarlyStoppingCallback, EpochCallback, GradientAccumulationCallback, GradientAccumulationStats,
    GradientMonitor, GradientScalingStrategy, GradientSummary, HistogramCallback, HistogramStats,
    LearningRateFinder, ModelEMACallback, ProfilingCallback, ProfilingStats,
    ReduceLrOnPlateauCallback, SWACallback, TrainingCheckpoint, ValidationCallback,
};
pub use error::{TrainError, TrainResult};
pub use logging::{
    ConsoleLogger, CsvLogger, FileLogger, JsonlLogger, LoggingBackend, MetricsLogger,
    TensorBoardLogger,
};
pub use loss::{
    BCEWithLogitsLoss, ConstraintViolationLoss, ContrastiveLoss, CrossEntropyLoss, DiceLoss,
    FocalLoss, HingeLoss, HuberLoss, KLDivergenceLoss, LogicalLoss, Loss, LossConfig, MseLoss,
    PolyLoss, RuleSatisfactionLoss, TripletLoss, TverskyLoss,
};
pub use metrics::{
    Accuracy, BalancedAccuracy, CohensKappa, ConfusionMatrix, DiceCoefficient,
    ExpectedCalibrationError, F1Score, IoU, MatthewsCorrelationCoefficient,
    MaximumCalibrationError, MeanAveragePrecision, MeanIoU, Metric, MetricTracker,
    NormalizedDiscountedCumulativeGain, PerClassMetrics, Precision, Recall, RocCurve, TopKAccuracy,
};
pub use model::{AutodiffModel, DynamicModel, LinearModel, Model};
pub use optimizer::{
    AdaBeliefOptimizer, AdaMaxOptimizer, AdagradOptimizer, AdamOptimizer, AdamPOptimizer,
    AdamWOptimizer, GradClipMode, LambOptimizer, LarsOptimizer, LionConfig, LionOptimizer,
    LookaheadOptimizer, NAdamOptimizer, Optimizer, OptimizerConfig, ProdigyConfig,
    ProdigyOptimizer, RAdamOptimizer, RMSpropOptimizer, SamOptimizer, ScheduleFreeAdamW,
    ScheduleFreeConfig, SgdOptimizer, SophiaConfig, SophiaOptimizer, SophiaVariant,
};
pub use regularization::{
    CompositeRegularization, ElasticNetRegularization, GroupLassoRegularization, L1Regularization,
    L2Regularization, MaxNormRegularization, OrthogonalRegularization, Regularizer,
    SpectralNormalization,
};
pub use scheduler::{
    CosineAnnealingLrScheduler, CyclicLrMode, CyclicLrScheduler, ExponentialLrScheduler,
    LrScheduler, MultiStepLrScheduler, NoamScheduler, OneCycleLrScheduler, PlateauMode,
    PolynomialDecayLrScheduler, ReduceLROnPlateauScheduler, SgdrScheduler, StepLrScheduler,
    WarmupCosineLrScheduler,
};
pub use trainer::{Trainer, TrainerConfig, TrainingHistory, TrainingState};

// Curriculum learning
pub use curriculum::{
    CompetenceCurriculum, CurriculumManager, CurriculumStrategy, ExponentialCurriculum,
    LinearCurriculum, SelfPacedCurriculum, TaskCurriculum,
};

// Transfer learning
pub use transfer::{
    DiscriminativeFineTuning, FeatureExtractorMode, LayerFreezingConfig, ProgressiveUnfreezing,
    TransferLearningManager,
};

// Hyperparameter optimization
pub use hyperparameter::{
    AcquisitionFunction, BayesianOptimization, GaussianProcess, GpKernel, GridSearch,
    HyperparamConfig, HyperparamResult, HyperparamSpace, HyperparamValue, RandomSearch,
};

// Cross-validation
pub use crossval::{
    CrossValidationResults, CrossValidationSplit, KFold, LeaveOneOut, StratifiedKFold,
    TimeSeriesSplit,
};

// Ensembling
pub use ensemble::{
    AveragingEnsemble, BaggingHelper, Ensemble, ModelSoup, SoupRecipe, StackingEnsemble,
    VotingEnsemble, VotingMode,
};

// Multi-task learning
pub use multitask::{MultiTaskLoss, PCGrad, TaskWeightingStrategy};

// Knowledge distillation
pub use distillation::{AttentionTransferLoss, DistillationLoss, FeatureDistillationLoss};

// Label smoothing
pub use label_smoothing::{LabelSmoothingLoss, MixupLoss};

// Memory management and profiling
pub use memory::{
    CheckpointStrategy, GradientCheckpointConfig, MemoryBudgetManager, MemoryEfficientTraining,
    MemoryProfilerCallback, MemorySettings, MemoryStats,
};

// Data loading and preprocessing
pub use data::{
    CsvLoader, DataPreprocessor, Dataset, LabelEncoder, OneHotEncoder, PreprocessingMethod,
};

// Utilities for model introspection and analysis
pub use utils::{
    compare_models, compute_gradient_stats, format_duration, print_gradient_report, GradientStats,
    LrRangeTestAnalyzer, ModelSummary, ParameterDifference, ParameterStats, TimeEstimator,
};

// Model pruning and compression
pub use pruning::{
    GlobalPruner, GradientPruner, LayerPruningStats, MagnitudePruner, Pruner, PruningConfig,
    PruningMask, PruningStats, StructuredPruner, StructuredPruningAxis,
};

// Advanced sampling strategies
pub use sampling::{
    BatchReweighter, ClassBalancedSampler, CurriculumSampler, FocalSampler, HardNegativeMiner,
    ImportanceSampler, MiningStrategy, OnlineHardExampleMiner, ReweightingStrategy,
};

// Model quantization and compression
pub use quantization::{
    BitWidth, DynamicRangeCalibrator, Granularity, QuantizationAwareTraining, QuantizationConfig,
    QuantizationMode, QuantizationParams, QuantizedTensor, Quantizer,
};

// Mixed precision training
pub use mixed_precision::{
    AutocastContext, GradientScaler, LossScaler, MixedPrecisionStats, MixedPrecisionTrainer,
    PrecisionMode,
};

// Few-shot learning
pub use few_shot::{
    DistanceMetric, EpisodeSampler, FewShotAccuracy, MatchingNetwork, PrototypicalDistance,
    ShotType, SupportSet,
};

// Meta-learning
pub use meta_learning::{
    MAMLConfig, MetaLearner, MetaStats, MetaTask, Reptile, ReptileConfig, MAML,
};

// Gradient centralization
pub use gradient_centralization::{GcConfig, GcStats, GcStrategy, GradientCentralization};

// Stochastic Depth (DropPath)
pub use stochastic_depth::{DropPath, ExponentialStochasticDepth, LinearStochasticDepth};

// DropBlock regularization
pub use dropblock::{DropBlock, LinearDropBlockScheduler};
