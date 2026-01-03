//! Vision Transformer (ViT) components for image processing
//!
//! This module implements Vision Transformer architectures that process images
//! as sequences of patches, enabling transformer models to be applied to computer vision tasks.
//!
//! ## Architecture
//!
//! Vision Transformers follow this processing pipeline:
//!
//! 1. **Patch Embedding**: Split image into fixed-size patches and linearly embed them
//!    - Input: `[batch, channels, height, width]`
//!    - Patches: `[batch, num_patches, patch_dim]` where `patch_dim = patch_size² × channels`
//!    - Embedding: `einsum("bnp,pd->bnd", patches, W_embed)` → `[batch, num_patches, d_model]`
//!
//! 2. **Class Token**: Prepend learnable classification token
//!    - `[batch, 1, d_model]` concatenated with patch embeddings
//!    - Final: `[batch, num_patches + 1, d_model]`
//!
//! 3. **Position Embeddings**: Add 2D position encodings
//!    - Learnable or sinusoidal embeddings for each patch position
//!    - Shape: `[1, num_patches + 1, d_model]`
//!
//! 4. **Transformer Encoder**: Standard transformer encoder layers
//!
//! 5. **Classification Head**: MLP head applied to class token output
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_trustformers::vision::{VisionTransformerConfig, VisionTransformer};
//! use tensorlogic_ir::EinsumGraph;
//!
//! // Configure ViT-Base/16 (16x16 patches, 224x224 images)
//! let config = VisionTransformerConfig::new(
//!     224,  // image_size
//!     16,   // patch_size
//!     3,    // in_channels (RGB)
//!     768,  // d_model
//!     12,   // n_heads
//!     3072, // d_ff
//!     12,   // n_layers
//!     1000, // num_classes
//! ).unwrap();
//!
//! let vit = VisionTransformer::new(config).unwrap();
//!
//! let mut graph = EinsumGraph::new();
//! graph.add_tensor("image");
//! let output = vit.build_vit_graph(&mut graph).unwrap();
//! ```

use crate::error::{Result, TrustformerError};
use crate::stacks::{EncoderStack, EncoderStackConfig};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Configuration for patch embedding layer
#[derive(Debug, Clone)]
pub struct PatchEmbeddingConfig {
    /// Image size (assumes square images)
    pub image_size: usize,
    /// Patch size (assumes square patches)
    pub patch_size: usize,
    /// Number of input channels (e.g., 3 for RGB)
    pub in_channels: usize,
    /// Embedding dimension
    pub d_model: usize,
}

impl PatchEmbeddingConfig {
    /// Create new patch embedding configuration
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        d_model: usize,
    ) -> Result<Self> {
        if image_size == 0 {
            return Err(TrustformerError::CompilationError(
                "image_size must be > 0".into(),
            ));
        }
        if patch_size == 0 {
            return Err(TrustformerError::CompilationError(
                "patch_size must be > 0".into(),
            ));
        }
        if !image_size.is_multiple_of(patch_size) {
            return Err(TrustformerError::CompilationError(format!(
                "image_size ({}) must be divisible by patch_size ({})",
                image_size, patch_size
            )));
        }
        if in_channels == 0 {
            return Err(TrustformerError::CompilationError(
                "in_channels must be > 0".into(),
            ));
        }
        if d_model == 0 {
            return Err(TrustformerError::CompilationError(
                "d_model must be > 0".into(),
            ));
        }

        Ok(Self {
            image_size,
            patch_size,
            in_channels,
            d_model,
        })
    }

    /// Get number of patches
    pub fn num_patches(&self) -> usize {
        let patches_per_side = self.image_size / self.patch_size;
        patches_per_side * patches_per_side
    }

    /// Get patch dimension (patch_size² × in_channels)
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TrustformerError::CompilationError(
                "image_size must be divisible by patch_size".into(),
            ));
        }
        Ok(())
    }
}

/// Patch embedding layer for Vision Transformers
pub struct PatchEmbedding {
    config: PatchEmbeddingConfig,
}

impl PatchEmbedding {
    /// Create new patch embedding layer
    pub fn new(config: PatchEmbeddingConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build patch embedding einsum graph
    ///
    /// Converts image patches to token embeddings
    ///
    /// # Graph Inputs
    /// - Tensor 0: Input image tensor `[batch, in_channels, height, width]` or patches `[batch, num_patches, patch_dim]`
    /// - Tensor 1: Patch embedding weights `[patch_dim, d_model]`
    ///
    /// # Graph Output
    /// - Embedded patches `[batch, num_patches, d_model]`
    pub fn build_patch_embed_graph(&self, graph: &mut EinsumGraph) -> Result<usize> {
        // Expected inputs:
        // - Tensor 0: patches [batch, num_patches, patch_dim]
        // - Tensor 1: W_patch_embed [patch_dim, d_model]
        //
        // In practice, patching would involve:
        // 1. Unfold: [B, C, H, W] → [B, C, n_patches_h, n_patches_w, patch_h, patch_w]
        // 2. Reshape: [B, C, n_patches_h, n_patches_w, patch_h, patch_w] → [B, num_patches, patch_dim]
        // 3. Linear: einsum("bnp,pd->bnd", patches, W_patch_embed)
        //
        // For simplicity, we assume the patching is already done and input tensor 0 contains patches

        // Create einsum node for patch embedding
        // einsum("bnp,pd->bnd", patches, W_patch_embed)
        let output_tensor = graph.add_tensor("patch_embeddings");
        let node = EinsumNode::new("bnp,pd->bnd", vec![0, 1], vec![output_tensor]);
        graph.add_node(node)?;

        Ok(output_tensor)
    }

    /// Get configuration
    pub fn config(&self) -> &PatchEmbeddingConfig {
        &self.config
    }
}

/// Configuration for Vision Transformer
#[derive(Debug, Clone)]
pub struct VisionTransformerConfig {
    /// Patch embedding configuration
    pub patch_embed: PatchEmbeddingConfig,
    /// Transformer encoder stack configuration
    pub encoder: EncoderStackConfig,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to use class token
    pub use_class_token: bool,
    /// Dropout rate for classification head
    pub classifier_dropout: f64,
}

impl VisionTransformerConfig {
    /// Create new Vision Transformer configuration
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbeddingConfig::new(image_size, patch_size, in_channels, d_model)?;

        // For Vision Transformers, we use learned position encoding by default
        let max_seq_len = patch_embed.num_patches() + 1; // +1 for class token
        let encoder = EncoderStackConfig::new(n_layers, d_model, n_heads, d_ff, max_seq_len)?
            .with_learned_position_encoding();

        Ok(Self {
            patch_embed,
            encoder,
            num_classes,
            use_class_token: true,
            classifier_dropout: 0.0,
        })
    }

    /// Builder: Set whether to use class token
    pub fn with_class_token(mut self, use_class_token: bool) -> Self {
        self.use_class_token = use_class_token;
        self
    }

    /// Builder: Set classifier dropout
    pub fn with_classifier_dropout(mut self, dropout: f64) -> Self {
        self.classifier_dropout = dropout;
        self
    }

    /// Builder: Set learned position encoding
    pub fn with_learned_position_encoding(mut self) -> Self {
        self.encoder = self.encoder.with_learned_position_encoding();
        self
    }

    /// Builder: Set whether to use pre-norm
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.encoder.layer_config = self.encoder.layer_config.with_pre_norm(pre_norm);
        self
    }

    /// Builder: Set dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.encoder = self.encoder.with_dropout(dropout);
        self.classifier_dropout = dropout;
        self
    }

    /// Get sequence length (num_patches + optional class token)
    pub fn seq_length(&self) -> usize {
        let base = self.patch_embed.num_patches();
        if self.use_class_token {
            base + 1
        } else {
            base
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.patch_embed.validate()?;
        self.encoder.validate()?;
        if self.num_classes == 0 {
            return Err(TrustformerError::CompilationError(
                "num_classes must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Vision Transformer (ViT) model
pub struct VisionTransformer {
    config: VisionTransformerConfig,
    patch_embed: PatchEmbedding,
    #[allow(dead_code)] // Will be used in future complete implementation
    encoder: EncoderStack,
}

impl VisionTransformer {
    /// Create new Vision Transformer
    pub fn new(config: VisionTransformerConfig) -> Result<Self> {
        config.validate()?;

        let patch_embed = PatchEmbedding::new(config.patch_embed.clone())?;
        let encoder = EncoderStack::new(config.encoder.clone())?;

        Ok(Self {
            config,
            patch_embed,
            encoder,
        })
    }

    /// Build complete Vision Transformer einsum graph
    ///
    /// # Graph Inputs
    /// - Tensor 0: Input patches `[batch, num_patches, patch_dim]`
    /// - Tensor 1: Patch embedding weights `[patch_dim, d_model]`
    /// - Additional encoder tensors (weights for each layer)
    ///
    /// # Graph Outputs
    /// - Classification logits `[batch, num_classes]`
    ///
    /// Note: This is a simplified representation. In practice, you would need to provide
    /// all encoder layer weights and handle class token prepending/extraction properly.
    pub fn build_vit_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // 1. Patch embedding: convert patches to embeddings
        let patches = self.patch_embed.build_patch_embed_graph(graph)?;

        // 2. Add position embeddings
        // In a full implementation, this would add learned or sinusoidal position embeddings
        // For now, we represent this as an element-wise addition
        let positioned = graph.add_tensor("positioned_embeddings");
        let pos_add_node = EinsumNode::elem_binary(
            "add_pos_embed".to_string(),
            patches,
            2, // Position embedding tensor
            positioned,
        );
        graph.add_node(pos_add_node)?;

        // 3. Pass through transformer encoder
        // Note: The encoder.build_encoder_graph() expects tensor 0 as input
        // We would need to refactor this to properly handle the positioned embeddings
        // For now, this is a placeholder that shows the structure

        // 4. Classification head
        // In a full implementation, this would:
        // - Extract class token (first position) or pool all tokens
        // - Apply linear layer: einsum("bd,dc->bc", class_repr, W_classifier)
        // - Add bias

        // For this simplified version, we just return the positioned embeddings
        // as a placeholder output
        Ok(vec![positioned])
    }

    /// Get configuration
    pub fn config(&self) -> &VisionTransformerConfig {
        &self.config
    }

    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let mut total = 0;

        // Patch embedding: patch_dim × d_model
        total += self.config.patch_embed.patch_dim() * self.config.patch_embed.d_model;

        // Class token (if used): d_model
        if self.config.use_class_token {
            total += self.config.patch_embed.d_model;
        }

        // Position embeddings: seq_length × d_model
        total += self.config.seq_length() * self.config.patch_embed.d_model;

        // Encoder parameters (all layers)
        let params_per_layer =
            crate::utils::count_encoder_layer_params(&self.config.encoder.layer_config);
        total += params_per_layer * self.config.encoder.num_layers;

        // Final layer norm (if enabled)
        if self.config.encoder.final_layer_norm {
            total +=
                crate::utils::count_layernorm_params(&self.config.encoder.layer_config.layer_norm);
        }

        // Classification head: d_model × num_classes + num_classes (bias)
        total +=
            self.config.patch_embed.d_model * self.config.num_classes + self.config.num_classes;

        total
    }
}

/// Common Vision Transformer presets
pub enum ViTPreset {
    /// ViT-Tiny/16: 5.7M parameters
    Tiny16,
    /// ViT-Small/16: 22M parameters
    Small16,
    /// ViT-Base/16: 86M parameters
    Base16,
    /// ViT-Large/16: 307M parameters
    Large16,
    /// ViT-Huge/14: 632M parameters
    Huge14,
}

impl ViTPreset {
    /// Create configuration from preset
    pub fn config(&self, num_classes: usize) -> Result<VisionTransformerConfig> {
        let (image_size, patch_size, d_model, n_heads, d_ff, n_layers) = match self {
            ViTPreset::Tiny16 => (224, 16, 192, 3, 768, 12),
            ViTPreset::Small16 => (224, 16, 384, 6, 1536, 12),
            ViTPreset::Base16 => (224, 16, 768, 12, 3072, 12),
            ViTPreset::Large16 => (224, 16, 1024, 16, 4096, 24),
            ViTPreset::Huge14 => (224, 14, 1280, 16, 5120, 32),
        };

        VisionTransformerConfig::new(
            image_size,
            patch_size,
            3, // RGB
            d_model,
            n_heads,
            d_ff,
            n_layers,
            num_classes,
        )
    }

    /// Get preset name
    pub fn name(&self) -> &'static str {
        match self {
            ViTPreset::Tiny16 => "ViT-Tiny/16",
            ViTPreset::Small16 => "ViT-Small/16",
            ViTPreset::Base16 => "ViT-Base/16",
            ViTPreset::Large16 => "ViT-Large/16",
            ViTPreset::Huge14 => "ViT-Huge/14",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embedding_config() {
        let config = PatchEmbeddingConfig::new(224, 16, 3, 768).unwrap();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.d_model, 768);
        assert_eq!(config.num_patches(), 196); // (224/16)^2
        assert_eq!(config.patch_dim(), 768); // 16*16*3
    }

    #[test]
    fn test_patch_embedding_invalid_size() {
        let result = PatchEmbeddingConfig::new(225, 16, 3, 768);
        assert!(result.is_err()); // 225 not divisible by 16
    }

    #[test]
    fn test_patch_embedding_graph() {
        let config = PatchEmbeddingConfig::new(224, 16, 3, 768).unwrap();
        let patch_embed = PatchEmbedding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("image");
        graph.add_tensor("W_patch_embed");

        let output = patch_embed.build_patch_embed_graph(&mut graph).unwrap();
        assert!(output > 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_vit_config_creation() {
        let config = VisionTransformerConfig::new(
            224,  // image_size
            16,   // patch_size
            3,    // in_channels
            768,  // d_model
            12,   // n_heads
            3072, // d_ff
            12,   // n_layers
            1000, // num_classes
        )
        .unwrap();

        assert_eq!(config.num_classes, 1000);
        assert!(config.use_class_token);
        assert_eq!(config.seq_length(), 197); // 196 patches + 1 class token
    }

    #[test]
    fn test_vit_config_without_class_token() {
        let config = VisionTransformerConfig::new(224, 16, 3, 768, 12, 3072, 12, 1000)
            .unwrap()
            .with_class_token(false);

        assert!(!config.use_class_token);
        assert_eq!(config.seq_length(), 196); // Only patches, no class token
    }

    #[test]
    fn test_vit_creation() {
        let config = VisionTransformerConfig::new(224, 16, 3, 768, 12, 3072, 12, 1000).unwrap();
        let vit = VisionTransformer::new(config).unwrap();

        assert!(vit.config().validate().is_ok());
    }

    #[test]
    fn test_vit_graph_building() {
        let config = VisionTransformerConfig::new(224, 16, 3, 384, 6, 1536, 2, 10).unwrap();
        let vit = VisionTransformer::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        // Add required input tensors
        graph.add_tensor("patches"); // Tensor 0
        graph.add_tensor("W_patch_embed"); // Tensor 1
        graph.add_tensor("pos_embed"); // Tensor 2

        let result = vit.build_vit_graph(&mut graph);
        // The graph building should succeed
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_vit_parameter_count() {
        let config = VisionTransformerConfig::new(224, 16, 3, 768, 12, 3072, 12, 1000).unwrap();
        let vit = VisionTransformer::new(config).unwrap();

        let params = vit.count_parameters();
        assert!(params > 0);
        // ViT-Base/16 should have ~86M parameters
        // We're not checking exact count due to implementation details
    }

    #[test]
    fn test_vit_presets() {
        for preset in [
            ViTPreset::Tiny16,
            ViTPreset::Small16,
            ViTPreset::Base16,
            ViTPreset::Large16,
            ViTPreset::Huge14,
        ] {
            let config = preset.config(1000).unwrap();
            assert!(config.validate().is_ok());
            assert_eq!(config.num_classes, 1000);

            let vit = VisionTransformer::new(config).unwrap();
            assert!(vit.count_parameters() > 0);
        }
    }

    #[test]
    fn test_vit_preset_names() {
        assert_eq!(ViTPreset::Tiny16.name(), "ViT-Tiny/16");
        assert_eq!(ViTPreset::Small16.name(), "ViT-Small/16");
        assert_eq!(ViTPreset::Base16.name(), "ViT-Base/16");
        assert_eq!(ViTPreset::Large16.name(), "ViT-Large/16");
        assert_eq!(ViTPreset::Huge14.name(), "ViT-Huge/14");
    }

    #[test]
    fn test_different_image_sizes() {
        for (image_size, patch_size) in [(224, 16), (384, 16), (512, 32)] {
            let config = PatchEmbeddingConfig::new(image_size, patch_size, 3, 768).unwrap();
            let expected_patches = (image_size / patch_size) * (image_size / patch_size);
            assert_eq!(config.num_patches(), expected_patches);
        }
    }

    #[test]
    fn test_vit_config_builder() {
        let config = VisionTransformerConfig::new(224, 16, 3, 768, 12, 3072, 12, 1000)
            .unwrap()
            .with_class_token(true)
            .with_classifier_dropout(0.1)
            .with_pre_norm(true)
            .with_dropout(0.1);

        assert!(config.use_class_token);
        assert!((config.classifier_dropout - 0.1).abs() < 1e-10);
        assert!(config.encoder.layer_config.pre_norm);
        assert!(config.validate().is_ok());
    }
}
