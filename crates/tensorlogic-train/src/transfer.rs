//! Transfer learning utilities for model fine-tuning.
//!
//! This module provides utilities for transfer learning:
//! - Layer freezing and unfreezing
//! - Progressive fine-tuning strategies
//! - Feature extraction mode
//! - Learning rate scheduling for transfer learning

use crate::{TrainError, TrainResult};
use std::collections::{HashMap, HashSet};

/// Layer freezing configuration for transfer learning.
#[derive(Debug, Clone)]
pub struct LayerFreezingConfig {
    /// Set of frozen layer names.
    frozen_layers: HashSet<String>,
    /// Whether to freeze all layers by default.
    freeze_all: bool,
}

impl LayerFreezingConfig {
    /// Create a new layer freezing configuration.
    pub fn new() -> Self {
        Self {
            frozen_layers: HashSet::new(),
            freeze_all: false,
        }
    }

    /// Freeze specific layers.
    ///
    /// # Arguments
    /// * `layer_names` - Names of layers to freeze
    pub fn freeze_layers(&mut self, layer_names: &[&str]) {
        for name in layer_names {
            self.frozen_layers.insert(name.to_string());
        }
    }

    /// Unfreeze specific layers.
    ///
    /// # Arguments
    /// * `layer_names` - Names of layers to unfreeze
    pub fn unfreeze_layers(&mut self, layer_names: &[&str]) {
        for name in layer_names {
            self.frozen_layers.remove(*name);
        }
    }

    /// Freeze all layers.
    pub fn freeze_all(&mut self) {
        self.freeze_all = true;
    }

    /// Unfreeze all layers.
    pub fn unfreeze_all(&mut self) {
        self.freeze_all = false;
        self.frozen_layers.clear();
    }

    /// Check if a layer is frozen.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer to check
    pub fn is_frozen(&self, layer_name: &str) -> bool {
        self.freeze_all || self.frozen_layers.contains(layer_name)
    }

    /// Get all frozen layer names.
    pub fn frozen_layers(&self) -> Vec<String> {
        self.frozen_layers.iter().cloned().collect()
    }

    /// Get the number of frozen layers.
    pub fn num_frozen(&self) -> usize {
        if self.freeze_all {
            usize::MAX
        } else {
            self.frozen_layers.len()
        }
    }
}

impl Default for LayerFreezingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Progressive unfreezing strategy for transfer learning.
///
/// Gradually unfreezes layers from top to bottom during training.
#[derive(Debug, Clone)]
pub struct ProgressiveUnfreezing {
    /// Layer names ordered from bottom (early) to top (late).
    layer_order: Vec<String>,
    /// Number of epochs to wait before unfreezing next layer.
    unfreeze_interval: usize,
    /// Current unfreezing stage.
    current_stage: usize,
}

impl ProgressiveUnfreezing {
    /// Create a new progressive unfreezing strategy.
    ///
    /// # Arguments
    /// * `layer_order` - Layer names ordered from bottom to top
    /// * `unfreeze_interval` - Epochs between unfreezing stages
    pub fn new(layer_order: Vec<String>, unfreeze_interval: usize) -> TrainResult<Self> {
        if layer_order.is_empty() {
            return Err(TrainError::InvalidParameter(
                "layer_order cannot be empty".to_string(),
            ));
        }
        if unfreeze_interval == 0 {
            return Err(TrainError::InvalidParameter(
                "unfreeze_interval must be positive".to_string(),
            ));
        }
        Ok(Self {
            layer_order,
            unfreeze_interval,
            current_stage: 0,
        })
    }

    /// Update the unfreezing stage based on current epoch.
    ///
    /// # Arguments
    /// * `epoch` - Current training epoch
    ///
    /// # Returns
    /// Whether the stage was updated
    pub fn update_stage(&mut self, epoch: usize) -> bool {
        let new_stage = epoch / self.unfreeze_interval;
        if new_stage > self.current_stage {
            self.current_stage = new_stage.min(self.layer_order.len());
            true
        } else {
            false
        }
    }

    /// Get layers that should be unfrozen at current stage.
    ///
    /// # Returns
    /// Layer names that should be trainable
    pub fn get_trainable_layers(&self) -> Vec<String> {
        // Unfreeze from top to bottom: start with last layers
        let num_trainable = self.current_stage.min(self.layer_order.len());
        let start_idx = self.layer_order.len().saturating_sub(num_trainable);

        self.layer_order[start_idx..].to_vec()
    }

    /// Get layers that should be frozen at current stage.
    ///
    /// # Returns
    /// Layer names that should be frozen
    pub fn get_frozen_layers(&self) -> Vec<String> {
        let num_trainable = self.current_stage.min(self.layer_order.len());
        let end_idx = self.layer_order.len().saturating_sub(num_trainable);

        self.layer_order[..end_idx].to_vec()
    }

    /// Check if unfreezing is complete.
    pub fn is_complete(&self) -> bool {
        self.current_stage >= self.layer_order.len()
    }

    /// Get current stage number.
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Get total number of stages.
    pub fn total_stages(&self) -> usize {
        self.layer_order.len()
    }
}

/// Discriminative fine-tuning: use different learning rates for different layers.
///
/// Typically, earlier layers use smaller learning rates than later layers.
#[derive(Debug, Clone)]
pub struct DiscriminativeFineTuning {
    /// Base learning rate for the last layer.
    pub base_lr: f64,
    /// Learning rate decay factor (each earlier layer uses lr * decay_factor).
    pub decay_factor: f64,
    /// Layer-specific learning rates.
    layer_lrs: HashMap<String, f64>,
}

impl DiscriminativeFineTuning {
    /// Create a new discriminative fine-tuning configuration.
    ///
    /// # Arguments
    /// * `base_lr` - Learning rate for the last layer
    /// * `decay_factor` - Decay factor for earlier layers (e.g., 0.5 means half LR)
    pub fn new(base_lr: f64, decay_factor: f64) -> TrainResult<Self> {
        if base_lr <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "base_lr must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&decay_factor) {
            return Err(TrainError::InvalidParameter(
                "decay_factor must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            base_lr,
            decay_factor,
            layer_lrs: HashMap::new(),
        })
    }

    /// Compute learning rates for all layers.
    ///
    /// # Arguments
    /// * `layer_order` - Layer names ordered from bottom to top
    pub fn compute_layer_lrs(&mut self, layer_order: &[String]) {
        self.layer_lrs.clear();

        let num_layers = layer_order.len();
        for (i, layer_name) in layer_order.iter().enumerate() {
            // Later layers get higher learning rates
            let depth = num_layers - 1 - i;
            let lr = self.base_lr * self.decay_factor.powi(depth as i32);
            self.layer_lrs.insert(layer_name.clone(), lr);
        }
    }

    /// Get the learning rate for a specific layer.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer
    ///
    /// # Returns
    /// Learning rate for the layer, or base_lr if not found
    pub fn get_layer_lr(&self, layer_name: &str) -> f64 {
        self.layer_lrs
            .get(layer_name)
            .copied()
            .unwrap_or(self.base_lr)
    }

    /// Get all layer learning rates.
    pub fn layer_lrs(&self) -> &HashMap<String, f64> {
        &self.layer_lrs
    }
}

/// Feature extraction mode: freeze entire feature extractor.
///
/// Only trains the final classification/regression head.
#[derive(Debug, Clone)]
pub struct FeatureExtractorMode {
    /// Name of the feature extractor (typically all layers except last).
    pub feature_extractor_name: String,
    /// Name of the head/classifier (typically the last layer).
    pub head_name: String,
}

impl FeatureExtractorMode {
    /// Create a new feature extractor mode.
    ///
    /// # Arguments
    /// * `feature_extractor_name` - Name/prefix of feature extractor layers
    /// * `head_name` - Name/prefix of head layers
    pub fn new(feature_extractor_name: String, head_name: String) -> Self {
        Self {
            feature_extractor_name,
            head_name,
        }
    }

    /// Check if a layer is part of the feature extractor.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer
    pub fn is_feature_extractor(&self, layer_name: &str) -> bool {
        layer_name.starts_with(&self.feature_extractor_name)
    }

    /// Check if a layer is part of the head.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer
    pub fn is_head(&self, layer_name: &str) -> bool {
        layer_name.starts_with(&self.head_name)
    }

    /// Get freezing configuration for feature extraction.
    ///
    /// # Arguments
    /// * `all_layers` - All layer names in the model
    ///
    /// # Returns
    /// Layer freezing configuration
    pub fn get_freezing_config(&self, all_layers: &[String]) -> LayerFreezingConfig {
        let mut config = LayerFreezingConfig::new();

        // Freeze all feature extractor layers
        let feature_layers: Vec<&str> = all_layers
            .iter()
            .filter(|name| self.is_feature_extractor(name))
            .map(|s| s.as_str())
            .collect();

        config.freeze_layers(&feature_layers);
        config
    }
}

/// Transfer learning strategy manager.
#[derive(Debug)]
pub struct TransferLearningManager {
    /// Layer freezing configuration.
    pub freezing_config: LayerFreezingConfig,
    /// Optional progressive unfreezing strategy.
    pub progressive_unfreezing: Option<ProgressiveUnfreezing>,
    /// Optional discriminative fine-tuning.
    pub discriminative_finetuning: Option<DiscriminativeFineTuning>,
    /// Current epoch counter.
    current_epoch: usize,
}

impl TransferLearningManager {
    /// Create a new transfer learning manager.
    pub fn new() -> Self {
        Self {
            freezing_config: LayerFreezingConfig::new(),
            progressive_unfreezing: None,
            discriminative_finetuning: None,
            current_epoch: 0,
        }
    }

    /// Set progressive unfreezing strategy.
    ///
    /// # Arguments
    /// * `strategy` - Progressive unfreezing configuration
    pub fn with_progressive_unfreezing(mut self, strategy: ProgressiveUnfreezing) -> Self {
        // Initialize freezing config with all layers frozen (stage 0)
        let frozen = strategy.get_frozen_layers();
        let frozen_refs: Vec<&str> = frozen.iter().map(|s| s.as_str()).collect();
        self.freezing_config.freeze_layers(&frozen_refs);

        self.progressive_unfreezing = Some(strategy);
        self
    }

    /// Set discriminative fine-tuning.
    ///
    /// # Arguments
    /// * `config` - Discriminative fine-tuning configuration
    pub fn with_discriminative_finetuning(mut self, config: DiscriminativeFineTuning) -> Self {
        self.discriminative_finetuning = Some(config);
        self
    }

    /// Set feature extraction mode.
    ///
    /// # Arguments
    /// * `mode` - Feature extraction configuration
    /// * `all_layers` - All layer names in the model
    pub fn with_feature_extraction(
        mut self,
        mode: FeatureExtractorMode,
        all_layers: &[String],
    ) -> Self {
        self.freezing_config = mode.get_freezing_config(all_layers);
        self
    }

    /// Update for new epoch.
    ///
    /// # Arguments
    /// * `epoch` - Current training epoch
    pub fn on_epoch_begin(&mut self, epoch: usize) {
        self.current_epoch = epoch;

        // Update progressive unfreezing if enabled
        if let Some(ref mut unfreezing) = self.progressive_unfreezing {
            if unfreezing.update_stage(epoch) {
                // Update freezing config based on new stage
                let frozen = unfreezing.get_frozen_layers();
                let trainable = unfreezing.get_trainable_layers();

                // Clear and rebuild freezing config
                self.freezing_config.unfreeze_all();
                let frozen_refs: Vec<&str> = frozen.iter().map(|s| s.as_str()).collect();
                self.freezing_config.freeze_layers(&frozen_refs);

                log::info!(
                    "Progressive unfreezing: Stage {}/{}, {} layers trainable",
                    unfreezing.current_stage(),
                    unfreezing.total_stages(),
                    trainable.len()
                );
            }
        }
    }

    /// Check if a layer should be updated during training.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer
    pub fn should_update_layer(&self, layer_name: &str) -> bool {
        !self.freezing_config.is_frozen(layer_name)
    }

    /// Get the learning rate for a specific layer.
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer
    /// * `base_lr` - Base learning rate
    ///
    /// # Returns
    /// Layer-specific learning rate
    pub fn get_layer_lr(&self, layer_name: &str, base_lr: f64) -> f64 {
        if let Some(ref finetuning) = self.discriminative_finetuning {
            finetuning.get_layer_lr(layer_name)
        } else {
            base_lr
        }
    }

    /// Get current epoch.
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }
}

impl Default for TransferLearningManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_freezing_config() {
        let mut config = LayerFreezingConfig::new();
        assert!(!config.is_frozen("layer1"));

        config.freeze_layers(&["layer1", "layer2"]);
        assert!(config.is_frozen("layer1"));
        assert!(config.is_frozen("layer2"));
        assert!(!config.is_frozen("layer3"));

        config.unfreeze_layers(&["layer1"]);
        assert!(!config.is_frozen("layer1"));
        assert!(config.is_frozen("layer2"));

        assert_eq!(config.num_frozen(), 1);
    }

    #[test]
    fn test_layer_freezing_all() {
        let mut config = LayerFreezingConfig::new();
        config.freeze_all();

        assert!(config.is_frozen("any_layer"));
        assert!(config.is_frozen("another_layer"));

        config.unfreeze_all();
        assert!(!config.is_frozen("any_layer"));
    }

    #[test]
    fn test_progressive_unfreezing() {
        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
        ];
        let mut unfreezing = ProgressiveUnfreezing::new(layers, 5).unwrap();

        // Stage 0: all frozen
        assert_eq!(unfreezing.get_trainable_layers().len(), 0);
        assert_eq!(unfreezing.get_frozen_layers().len(), 3);
        assert!(!unfreezing.is_complete());

        // Epoch 5: unfreeze last layer
        unfreezing.update_stage(5);
        assert_eq!(unfreezing.current_stage(), 1);
        assert_eq!(unfreezing.get_trainable_layers().len(), 1);
        assert_eq!(unfreezing.get_frozen_layers().len(), 2);

        // Epoch 10: unfreeze two last layers
        unfreezing.update_stage(10);
        assert_eq!(unfreezing.current_stage(), 2);
        assert_eq!(unfreezing.get_trainable_layers().len(), 2);

        // Epoch 15: all unfrozen
        unfreezing.update_stage(15);
        assert_eq!(unfreezing.current_stage(), 3);
        assert_eq!(unfreezing.get_trainable_layers().len(), 3);
        assert!(unfreezing.is_complete());
    }

    #[test]
    fn test_progressive_unfreezing_invalid() {
        let result = ProgressiveUnfreezing::new(vec![], 5);
        assert!(result.is_err());

        let result = ProgressiveUnfreezing::new(vec!["layer1".to_string()], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_discriminative_finetuning() {
        let mut finetuning = DiscriminativeFineTuning::new(1e-3, 0.5).unwrap();

        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
        ];
        finetuning.compute_layer_lrs(&layers);

        // Last layer should have base_lr
        assert!((finetuning.get_layer_lr("layer3") - 1e-3).abs() < 1e-10);

        // Second layer should have base_lr * decay_factor
        assert!((finetuning.get_layer_lr("layer2") - 5e-4).abs() < 1e-10);

        // First layer should have base_lr * decay_factor^2
        assert!((finetuning.get_layer_lr("layer1") - 2.5e-4).abs() < 1e-10);
    }

    #[test]
    fn test_discriminative_finetuning_invalid() {
        assert!(DiscriminativeFineTuning::new(0.0, 0.5).is_err());
        assert!(DiscriminativeFineTuning::new(-1e-3, 0.5).is_err());
        assert!(DiscriminativeFineTuning::new(1e-3, 1.5).is_err());
        assert!(DiscriminativeFineTuning::new(1e-3, -0.1).is_err());
    }

    #[test]
    fn test_feature_extractor_mode() {
        let mode = FeatureExtractorMode::new("encoder".to_string(), "classifier".to_string());

        assert!(mode.is_feature_extractor("encoder.layer1"));
        assert!(mode.is_feature_extractor("encoder.layer2"));
        assert!(!mode.is_feature_extractor("classifier.fc"));

        assert!(mode.is_head("classifier.fc"));
        assert!(mode.is_head("classifier.output"));
        assert!(!mode.is_head("encoder.layer1"));

        let all_layers = vec![
            "encoder.layer1".to_string(),
            "encoder.layer2".to_string(),
            "classifier.fc".to_string(),
        ];

        let config = mode.get_freezing_config(&all_layers);
        assert!(config.is_frozen("encoder.layer1"));
        assert!(config.is_frozen("encoder.layer2"));
        assert!(!config.is_frozen("classifier.fc"));
    }

    #[test]
    fn test_transfer_learning_manager() {
        let mut manager = TransferLearningManager::new();

        // Initially, all layers are trainable
        assert!(manager.should_update_layer("layer1"));

        // Freeze some layers
        manager.freezing_config.freeze_layers(&["layer1"]);
        assert!(!manager.should_update_layer("layer1"));
        assert!(manager.should_update_layer("layer2"));
    }

    #[test]
    fn test_transfer_learning_with_progressive_unfreezing() {
        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
        ];
        let unfreezing = ProgressiveUnfreezing::new(layers.clone(), 5).unwrap();

        let mut manager = TransferLearningManager::new().with_progressive_unfreezing(unfreezing);

        // Epoch 0: all should be frozen
        manager.on_epoch_begin(0);
        assert!(!manager.should_update_layer("layer1"));
        assert!(!manager.should_update_layer("layer2"));
        assert!(!manager.should_update_layer("layer3"));

        // Epoch 5: last layer unfrozen
        manager.on_epoch_begin(5);
        assert!(!manager.should_update_layer("layer1"));
        assert!(!manager.should_update_layer("layer2"));
        assert!(manager.should_update_layer("layer3"));
    }

    #[test]
    fn test_transfer_learning_with_discriminative_finetuning() {
        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
        ];
        let mut finetuning = DiscriminativeFineTuning::new(1e-3, 0.5).unwrap();
        finetuning.compute_layer_lrs(&layers);

        let manager = TransferLearningManager::new().with_discriminative_finetuning(finetuning);

        // Check layer-specific learning rates
        assert!((manager.get_layer_lr("layer3", 1e-3) - 1e-3).abs() < 1e-10);
        assert!((manager.get_layer_lr("layer2", 1e-3) - 5e-4).abs() < 1e-10);
        assert!((manager.get_layer_lr("layer1", 1e-3) - 2.5e-4).abs() < 1e-10);
    }

    #[test]
    fn test_transfer_learning_with_feature_extraction() {
        let mode = FeatureExtractorMode::new("encoder".to_string(), "classifier".to_string());
        let all_layers = vec![
            "encoder.layer1".to_string(),
            "encoder.layer2".to_string(),
            "classifier.fc".to_string(),
        ];

        let manager = TransferLearningManager::new().with_feature_extraction(mode, &all_layers);

        // Encoder should be frozen
        assert!(!manager.should_update_layer("encoder.layer1"));
        assert!(!manager.should_update_layer("encoder.layer2"));

        // Classifier should be trainable
        assert!(manager.should_update_layer("classifier.fc"));
    }

    #[test]
    fn test_frozen_layers_getter() {
        let mut config = LayerFreezingConfig::new();
        config.freeze_layers(&["layer1", "layer2"]);

        let frozen = config.frozen_layers();
        assert_eq!(frozen.len(), 2);
        assert!(frozen.contains(&"layer1".to_string()));
        assert!(frozen.contains(&"layer2".to_string()));
    }

    #[test]
    fn test_progressive_unfreezing_total_stages() {
        let layers = vec!["layer1".to_string(), "layer2".to_string()];
        let unfreezing = ProgressiveUnfreezing::new(layers, 5).unwrap();

        assert_eq!(unfreezing.total_stages(), 2);
    }
}
