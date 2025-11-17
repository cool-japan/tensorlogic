//! Advanced training callbacks: EMA and SWA.

use crate::callbacks::core::Callback;
use crate::{TrainError, TrainResult, TrainingState};
use std::collections::HashMap;

/// Model EMA (Exponential Moving Average) callback.
///
/// Maintains an exponential moving average of model parameters during training.
/// This often leads to better generalization and more stable predictions.
///
/// The shadow parameters are updated as:
/// shadow_param = decay * shadow_param + (1 - decay) * param
///
/// Reference: Common practice in modern deep learning, popularized by Mean Teacher
/// and other semi-supervised learning methods.
pub struct ModelEMACallback {
    /// Decay rate for EMA (typically 0.999 or 0.9999).
    decay: f64,
    /// Shadow parameters (EMA of model parameters).
    shadow_params: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Whether to use warmup for the decay (start with smaller decay).
    use_warmup: bool,
    /// Current update step (for warmup).
    num_updates: usize,
    /// Whether callback is initialized.
    initialized: bool,
}

impl ModelEMACallback {
    /// Create a new Model EMA callback.
    ///
    /// # Arguments
    /// * `decay` - EMA decay rate (e.g., 0.999, 0.9999)
    /// * `use_warmup` - Whether to use decay warmup (recommended)
    pub fn new(decay: f64, use_warmup: bool) -> Self {
        Self {
            decay,
            shadow_params: HashMap::new(),
            use_warmup,
            num_updates: 0,
            initialized: false,
        }
    }

    /// Initialize shadow parameters from current model parameters.
    pub fn initialize(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) {
        self.shadow_params.clear();
        for (name, param) in parameters {
            self.shadow_params.insert(name.clone(), param.clone());
        }
        self.initialized = true;
    }

    /// Update EMA parameters.
    pub fn update(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        if !self.initialized {
            return Err(TrainError::CallbackError(
                "ModelEMA not initialized. Call initialize() first.".to_string(),
            ));
        }

        self.num_updates += 1;

        // Compute effective decay with warmup
        let decay = if self.use_warmup {
            // Gradual warmup: start with (1 + num_updates) / (10 + num_updates)
            // and approach self.decay
            let warmup_decay = (1.0 + self.num_updates as f64) / (10.0 + self.num_updates as f64);
            warmup_decay.min(self.decay)
        } else {
            self.decay
        };

        // Update shadow parameters
        for (name, param) in parameters {
            if let Some(shadow) = self.shadow_params.get_mut(name) {
                // shadow = decay * shadow + (1 - decay) * param
                *shadow = &*shadow * decay + &(param * (1.0 - decay));
            }
        }

        Ok(())
    }

    /// Get the EMA parameters.
    pub fn get_shadow_params(
        &self,
    ) -> &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        &self.shadow_params
    }

    /// Apply EMA parameters to the model (for evaluation).
    pub fn apply_shadow(
        &self,
        parameters: &mut HashMap<
            String,
            scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>,
        >,
    ) {
        for (name, shadow) in &self.shadow_params {
            if let Some(param) = parameters.get_mut(name) {
                *param = shadow.clone();
            }
        }
    }
}

impl Callback for ModelEMACallback {
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        // Note: Initialization must be done externally since we don't have access to parameters here
        Ok(())
    }

    fn on_batch_end(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        // Note: Update must be called externally since we don't have access to parameters here
        Ok(())
    }
}

/// SWA (Stochastic Weight Averaging) callback.
///
/// Averages model parameters over the course of training, typically starting
/// from a later epoch. This often leads to better generalization and wider optima.
///
/// Reference: Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018)
pub struct SWACallback {
    /// Epoch to start SWA (e.g., 75% through training).
    start_epoch: usize,
    /// Frequency of parameter averaging (every N epochs).
    update_frequency: usize,
    /// Running average of parameters.
    swa_params: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Number of models averaged so far.
    num_averaged: usize,
    /// Whether SWA is active.
    active: bool,
    /// Whether SWA parameters are initialized.
    initialized: bool,
    /// Verbose output.
    verbose: bool,
}

impl SWACallback {
    /// Create a new SWA callback.
    ///
    /// # Arguments
    /// * `start_epoch` - Epoch to start averaging (e.g., 0.75 * total_epochs)
    /// * `update_frequency` - Average parameters every N epochs (typically 1)
    /// * `verbose` - Whether to print progress
    pub fn new(start_epoch: usize, update_frequency: usize, verbose: bool) -> Self {
        Self {
            start_epoch,
            update_frequency,
            swa_params: HashMap::new(),
            num_averaged: 0,
            active: false,
            initialized: false,
            verbose,
        }
    }

    /// Update SWA parameters with current model parameters.
    pub fn update_average(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        if !self.active {
            return Ok(());
        }

        if !self.initialized {
            // Initialize with first model
            for (name, param) in parameters {
                self.swa_params.insert(name.clone(), param.clone());
            }
            self.initialized = true;
            self.num_averaged = 1;

            if self.verbose {
                println!("SWA: Initialized with model parameters");
            }
        } else {
            // Running average: swa = (swa * n + param) / (n + 1)
            let n = self.num_averaged as f64;
            for (name, param) in parameters {
                if let Some(swa_param) = self.swa_params.get_mut(name) {
                    *swa_param = &(&*swa_param * n + param) / (n + 1.0);
                }
            }
            self.num_averaged += 1;

            if self.verbose {
                println!("SWA: Updated average (n={})", self.num_averaged);
            }
        }

        Ok(())
    }

    /// Get the SWA parameters.
    pub fn get_swa_params(
        &self,
    ) -> &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        &self.swa_params
    }

    /// Apply SWA parameters to the model.
    pub fn apply_swa(
        &self,
        parameters: &mut HashMap<
            String,
            scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>,
        >,
    ) {
        if self.initialized {
            for (name, swa_param) in &self.swa_params {
                if let Some(param) = parameters.get_mut(name) {
                    *param = swa_param.clone();
                }
            }
        }
    }

    /// Check if SWA has collected any averages.
    pub fn is_ready(&self) -> bool {
        self.initialized && self.num_averaged > 0
    }
}

impl Callback for SWACallback {
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        // Activate SWA at start_epoch
        if epoch >= self.start_epoch && !self.active {
            self.active = true;
            if self.verbose {
                println!("\nSWA: Activated at epoch {}", epoch + 1);
            }
        }

        // Check if we should update average
        if self.active && epoch >= self.start_epoch {
            let relative_epoch = epoch - self.start_epoch;
            if relative_epoch.is_multiple_of(self.update_frequency) {
                // Note: Actual update must be called externally with parameters
                if self.verbose && self.initialized {
                    println!(
                        "SWA: Ready to update at epoch {} (call update_average with parameters)",
                        epoch + 1
                    );
                }
            }
        }

        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        if self.verbose && self.initialized {
            println!(
                "\nSWA: Training complete. Averaged {} models.",
                self.num_averaged
            );
            println!("SWA: Call apply_swa() to use averaged parameters.");
        }
        Ok(())
    }
}
