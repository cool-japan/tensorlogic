//! Training scaffolds: loss wiring, schedules, callbacks.
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

mod augmentation;
mod batch;
mod callbacks;
mod crossval;
mod curriculum;
mod ensemble;
mod error;
mod hyperparameter;
mod logging;
mod loss;
mod metrics;
mod model;
mod optimizer;
mod regularization;
mod scheduler;
mod trainer;
mod transfer;

pub use augmentation::{
    CompositeAugmenter, DataAugmenter, MixupAugmenter, NoAugmentation, NoiseAugmenter,
    RotationAugmenter, ScaleAugmenter,
};
pub use batch::{extract_batch, BatchConfig, BatchIterator, DataShuffler};
pub use callbacks::{
    BatchCallback, Callback, CallbackList, CheckpointCallback, EarlyStoppingCallback,
    EpochCallback, GradientAccumulationCallback, GradientMonitor, GradientSummary,
    HistogramCallback, HistogramStats, LearningRateFinder, ModelEMACallback, ProfilingCallback,
    ProfilingStats, ReduceLrOnPlateauCallback, SWACallback, TrainingCheckpoint, ValidationCallback,
};
pub use error::{TrainError, TrainResult};
pub use logging::{ConsoleLogger, FileLogger, LoggingBackend, MetricsLogger, TensorBoardLogger};
pub use loss::{
    BCEWithLogitsLoss, ConstraintViolationLoss, ContrastiveLoss, CrossEntropyLoss, DiceLoss,
    FocalLoss, HingeLoss, HuberLoss, KLDivergenceLoss, LogicalLoss, Loss, LossConfig, MseLoss,
    RuleSatisfactionLoss, TripletLoss, TverskyLoss,
};
pub use metrics::{
    Accuracy, BalancedAccuracy, CohensKappa, ConfusionMatrix, F1Score,
    MatthewsCorrelationCoefficient, Metric, MetricTracker, PerClassMetrics, Precision, Recall,
    RocCurve, TopKAccuracy,
};
pub use model::{AutodiffModel, DynamicModel, LinearModel, Model};
pub use optimizer::{
    AdaBeliefOptimizer, AdaMaxOptimizer, AdagradOptimizer, AdamOptimizer, AdamWOptimizer,
    GradClipMode, LambOptimizer, LarsOptimizer, LookaheadOptimizer, NAdamOptimizer, Optimizer,
    OptimizerConfig, RAdamOptimizer, RMSpropOptimizer, SamOptimizer, SgdOptimizer,
};
pub use regularization::{
    CompositeRegularization, ElasticNetRegularization, L1Regularization, L2Regularization,
    Regularizer,
};
pub use scheduler::{
    CosineAnnealingLrScheduler, CyclicLrMode, CyclicLrScheduler, ExponentialLrScheduler,
    LrScheduler, MultiStepLrScheduler, NoamScheduler, OneCycleLrScheduler, PlateauMode,
    PolynomialDecayLrScheduler, ReduceLROnPlateauScheduler, StepLrScheduler,
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
    GridSearch, HyperparamConfig, HyperparamResult, HyperparamSpace, HyperparamValue, RandomSearch,
};

// Cross-validation
pub use crossval::{
    CrossValidationResults, CrossValidationSplit, KFold, LeaveOneOut, StratifiedKFold,
    TimeSeriesSplit,
};

// Ensembling
pub use ensemble::{
    AveragingEnsemble, BaggingHelper, Ensemble, StackingEnsemble, VotingEnsemble, VotingMode,
};
