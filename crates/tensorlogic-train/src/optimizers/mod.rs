//! Optimizers for training neural networks.
//!
//! This module provides a collection of optimization algorithms used for training
//! neural networks and other gradient-based machine learning models.
//!
//! # Available Optimizers
//!
//! ## Basic Optimizers
//! - [`SgdOptimizer`]: Stochastic Gradient Descent with momentum
//!
//! ## Adaptive Learning Rate Optimizers
//! - [`AdagradOptimizer`]: Adaptive Gradient (per-parameter learning rates)
//! - [`RMSpropOptimizer`]: Root Mean Square Propagation
//! - [`AdamOptimizer`]: Adaptive Moment Estimation
//! - [`AdamWOptimizer`]: Adam with decoupled weight decay
//! - [`NAdamOptimizer`]: Nesterov-accelerated Adam
//! - [`AdaMaxOptimizer`]: Adam variant with infinity norm
//! - [`RAdamOptimizer`]: Rectified Adam with variance warmup
//! - [`AdaBeliefOptimizer`]: Adapts step size by belief in gradient direction
//!
//! ## Large Batch Training Optimizers
//! - [`LambOptimizer`]: Layer-wise Adaptive Moments optimizer for Batch training
//! - [`LarsOptimizer`]: Layer-wise Adaptive Rate Scaling
//!
//! ## Modern Optimizers
//! - [`LionOptimizer`]: EvoLved Sign Momentum (memory-efficient, sign-based updates)
//!
//! ## Meta-Optimizers (Wrappers)
//! - [`LookaheadOptimizer`]: Maintains slow and fast weights
//! - [`SamOptimizer`]: Sharpness Aware Minimization
//!
//! # Common Types
//! - [`Optimizer`]: Core trait that all optimizers implement
//! - [`OptimizerConfig`]: Configuration for optimizer parameters
//! - [`GradClipMode`]: Gradient clipping modes (by value or by norm)

pub mod adabelief;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod adamw;
pub mod common;
pub mod lamb;
pub mod lars;
pub mod lion;
pub mod lookahead;
pub mod nadam;
pub mod radam;
pub mod rmsprop;
pub mod sam;
pub mod sgd;

// Re-export common types
pub use common::{GradClipMode, Optimizer, OptimizerConfig};

// Re-export all optimizers
pub use adabelief::AdaBeliefOptimizer;
pub use adagrad::AdagradOptimizer;
pub use adam::AdamOptimizer;
pub use adamax::AdaMaxOptimizer;
pub use adamw::AdamWOptimizer;
pub use lamb::LambOptimizer;
pub use lars::LarsOptimizer;
pub use lion::{LionConfig, LionOptimizer};
pub use lookahead::LookaheadOptimizer;
pub use nadam::NAdamOptimizer;
pub use radam::RAdamOptimizer;
pub use rmsprop::RMSpropOptimizer;
pub use sam::SamOptimizer;
pub use sgd::SgdOptimizer;
