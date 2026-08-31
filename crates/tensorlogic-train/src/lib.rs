//! Training scaffolds: loss wiring, schedules, callbacks.
//!
//! **Version**: 0.1.2 | **Status**: Production Ready
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

pub mod adversarial;
mod augmentation;
mod batch;
mod callbacks;
pub mod checkpoint;
mod crossval;
mod curriculum;
mod data;
mod distillation;
mod dropblock;
pub mod early_stopping;
mod ensemble;
mod error;
mod few_shot;
mod gradient_accumulation;
mod gradient_centralization;
mod hyperparameter;
mod label_smoothing;
mod logging;
pub mod lora;
mod loss;
mod lr_scheduler;
mod memory;
mod meta_learning;
mod metrics;
mod mixed_precision;
mod model;
mod multitask;
pub mod nas;
pub mod neural_ode;
pub mod online_learning;
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
pub mod weight_init;

#[cfg(feature = "structured-logging")]
pub mod structured_logging;

pub use augmentation::{
    center_crop_2d,
    clip,
    cutmix,
    denormalize,
    dropout,
    dropout_mask,
    gaussian_noise,
    mixup,
    normalize,
    random_crop_2d,
    random_hflip,
    random_vflip,
    // Functional API (v2)
    AugRng,
    AugStats,
    AugmentationError,
    AugmentationPipeline,
    AugmentationStep,
    CompositeAugmenter,
    CutMixAugmenter,
    CutOutAugmenter,
    DataAugmenter,
    MixupAugmenter,
    NoAugmentation,
    NoiseAugmenter,
    RandomErasingAugmenter,
    RotationAugmenter,
    ScaleAugmenter,
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
pub use lr_scheduler::{
    CosineAnnealingScheduler, CyclicalScheduler, LrSchedulerV2,
    OneCycleLrScheduler as OneCyclePolicyScheduler, SchedulerConfig, SchedulerError, SchedulerType,
    StepDecayScheduler, WarmupScheduler,
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

// Gradient accumulation and micro-batching
pub use gradient_accumulation::{
    AccumulationConfig, AccumulationError, AccumulationStats, GradientAccumulator, GradientBuffer,
};

// Gradient centralization
pub use gradient_centralization::{GcConfig, GcStats, GcStrategy, GradientCentralization};

// Stochastic Depth (DropPath)
pub use stochastic_depth::{DropPath, ExponentialStochasticDepth, LinearStochasticDepth};

// DropBlock regularization
pub use dropblock::{DropBlock, LinearDropBlockScheduler};

// Early stopping
pub use early_stopping::{
    EarlyStoppingConfig, EarlyStoppingDecision, EarlyStoppingMonitor, MonitorMode,
    MultiMetricMonitor, MultiMetricPolicy, PlateauDetector, TrainingProgress,
};

// Optimizer checkpointing
pub use checkpoint::{
    deserialize_checkpoint, serialize_checkpoint, CheckpointError, CheckpointFormat,
    CheckpointManager, CheckpointMetadata, LossTracker, OptimizerCheckpoint, ParamState,
};

// Weight initialization strategies
pub use weight_init::{
    compute_fans, constant_init, gain_for_activation, kaiming_normal, kaiming_uniform,
    lecun_normal, lecun_uniform, normal_init, ones_init, orthogonal_init, uniform_init,
    xavier_normal, xavier_uniform, zeros_init, FanMode, InitError, InitRng, InitStats,
};

// Online learning algorithms
pub use online_learning::{
    online_evaluate, Ftrl, OGDLoss, OnlineError, OnlineGradientDescent, OnlineLearner, OnlineStats,
    OnlineUpdateResult, PAVariant, PassiveAggressive, Perceptron,
};

// Adversarial training utilities
pub use adversarial::{
    adversarial_training_loss, fgsm, pgd, project_l1, project_l2, project_linf, robustness_eval,
    AdversarialError, AdversarialExample, AdversarialTrainStats, AttackConfig, AttackLoss,
    AttackModel, CrossEntropyAttackLoss, LinearAttackModel, MseAttackLoss, PerturbNorm,
};

// Neural ODE — continuous-depth models with adjoint sensitivity
pub use neural_ode::{
    dopri5_solve, rk4_solve, AdaptiveSolution, AdjointResult, NeuralOde, OdeError, OdeFunc,
    OdeSolution, OdeSolverConfig,
};

// LoRA — low-rank adaptation for parameter-efficient fine-tuning
pub use lora::{LoraAdapter, LoraConfig, LoraError, LoraLayer};

// Neural Architecture Search
pub use nas::{
    ArchSampler, ArchSearchSpace, Architecture, LayerSpec, NasResult, RandomArchSearch,
    RegularizedEvolution,
};
