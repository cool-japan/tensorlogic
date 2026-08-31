# tensorlogic-trustformers

**Transformer architectures as TensorLogic einsum graphs**

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--trustformers-orange)](https://crates.io/crates/tensorlogic-trustformers)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-trustformers)
[![Tests](https://img.shields.io/badge/tests-488%2F488-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-stable-success)](#)

This crate provides implementations of transformer components (self-attention, multi-head attention, feed-forward networks) as einsum operations that compile to TensorLogic IR and execute on any TensorLogic backend.

## Features

- **Self-Attention** - Scaled dot-product attention as einsum operations
- **Multi-Head Attention** - Parallel attention heads with automatic head splitting/merging
- **Feed-Forward Networks** - Position-wise FFN with configurable activations (GELU, ReLU, etc.)
- **Gated FFN** - GLU-style gated feed-forward networks
- **Position Encodings** - Sinusoidal, learned, relative, RoPE, and ALiBi position encodings
- **Layer Normalization** - Standard LayerNorm and RMSNorm implementations
- **Encoder Layers** - Complete transformer encoder layers with pre/post-norm variants
- **Decoder Layers** - Complete transformer decoder layers with masked self-attention
- **Encoder/Decoder Stacks** - Multi-layer transformer stacks with flexible configuration
- **Rule-Based Attention** - Logical rules guiding attention patterns (hard/soft/gated)
- **Sparse Attention** - Efficient attention for long sequences (strided, local, block-sparse)
- **Flash Attention** - Memory-efficient O(1) attention with tiled SRAM computation
- **Grouped-Query Attention (GQA)** - Reduce KV cache memory (MHA/GQA/MQA support)
- **Sliding Window Attention** - Efficient long-context with O(n*w) complexity
- **LoRA** - Low-Rank Adaptation for parameter-efficient fine-tuning
- **Mixture-of-Experts (MoE)** - Sparse expert routing (TopK, Softmax, Switch, ExpertChoice)
- **Vision Transformers (ViT)** - Patch embedding and ViT configurations (Tiny/Small/Base/Large/Huge)
- **Gradient Checkpointing** - Memory-efficient training with uniform/selective/dynamic strategies
- **KV-Cache** - Efficient autoregressive inference with 10-1000x speedup
- **TrustformeRS Integration** - Bidirectional conversion with TrustformeRS ecosystem
- **Utility Functions** - Parameter counting, FLOP calculations, model presets
- **Performance Benchmarks** - Criterion-based benchmark suite with HTML reports
- **Type-Safe Configuration** - Builder pattern with validation
- **Einsum-Native** - All operations expressed as einsum for maximum flexibility
- **Zero Warnings** - Strict code quality enforcement
- **488 Tests** - Comprehensive test coverage (100% passing)

## Quick Start

```rust
use tensorlogic_trustformers::{
    AttentionConfig, SelfAttention, MultiHeadAttention,
    FeedForwardConfig, FeedForward,
};
use tensorlogic_ir::EinsumGraph;

// Configure and build self-attention
let attn_config = AttentionConfig::new(512, 8).unwrap();
let self_attn = SelfAttention::new(attn_config).unwrap();

let mut graph = EinsumGraph::new();
graph.add_tensor("Q");
graph.add_tensor("K");
graph.add_tensor("V");

let outputs = self_attn.build_attention_graph(&mut graph).unwrap();

// Configure feed-forward network
let ffn_config = FeedForwardConfig::new(512, 2048)
    .with_activation("gelu")
    .with_dropout(0.1);
let ffn = FeedForward::new(ffn_config).unwrap();
```

## Architecture

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Einsum breakdown:**
1. Query-Key scores: `einsum("bqd,bkd->bqk", Q, K)`
2. Scale: `scores / sqrt(d_k)`
3. Softmax: `softmax(scores, axis=-1)`
4. Attention-Value: `einsum("bqk,bkv->bqv", attn, V)`

### Multi-Head Attention

```
1. Reshape: [B, S, D] -> [B, H, S, D_k] where D_k = D/H
2. Attention per head: einsum("bhqd,bhkd->bhqk", Q, K)
3. Scale and softmax
4. Apply to values: einsum("bhqk,bhkv->bhqv", attn, V)
5. Concatenate heads: [B, H, S, D_k] -> [B, S, D]
```

## Configuration

### Attention Configuration

```rust
use tensorlogic_trustformers::AttentionConfig;

let config = AttentionConfig::new(512, 8)?
    .with_causal(true)      // Enable causal masking
    .with_dropout(0.1);      // Set dropout probability

assert_eq!(config.d_model, 512);
assert_eq!(config.n_heads, 8);
assert_eq!(config.d_k, 64);  // Automatically computed
```

### Complete Transformer Layer

```rust
use tensorlogic_trustformers::TransformerLayerConfig;

let config = TransformerLayerConfig::new(512, 8, 2048)?
    .with_pre_norm(true);   // Use pre-layer normalization

assert!(config.validate().is_ok());
```

## Position Encodings

Five types of position encodings for sequence modeling:

```rust
use tensorlogic_trustformers::{
    SinusoidalPositionEncoding, PositionEncodingConfig,
    RotaryPositionEncoding, AlibiPositionEncoding,
};

// Sinusoidal (fixed) encoding
let config = PositionEncodingConfig::sinusoidal(512, 2048);
let pe = SinusoidalPositionEncoding::new(config).unwrap();

// Rotary Position Embedding (RoPE) - used in LLaMA
// Attention with Linear Biases (ALiBi) - used in BLOOM
```

## Flash Attention

Memory-efficient attention with tiled SRAM computation:

```rust
use tensorlogic_trustformers::{FlashAttention, FlashAttentionConfig, FlashAttentionPreset};

// A100 GPU preset
let config = FlashAttentionPreset::a100();
let flash = FlashAttention::new(config)?;

// Custom tiling
let config = FlashAttentionConfig::new(512, 8)
    .with_block_size_q(64)
    .with_block_size_kv(64)
    .with_causal(true);
```

## Grouped-Query Attention (GQA)

Reduce KV cache memory for efficient inference:

```rust
use tensorlogic_trustformers::{GroupedQueryAttention, GQAConfig, GQAPreset};

// LLaMA 2 70B style (8 KV heads, 64 query heads)
let config = GQAPreset::llama2_70b();
let gqa = GroupedQueryAttention::new(config)?;

// Memory savings compared to MHA
println!("KV cache memory: {:.1}x of MHA", config.memory_factor());
```

## Sliding Window Attention

Efficient long-context handling:

```rust
use tensorlogic_trustformers::{SlidingWindowAttention, SlidingWindowPreset};

// Mistral 7B style
let config = SlidingWindowPreset::mistral_7b();
let swa = SlidingWindowAttention::new(config)?;

// O(n*w) complexity instead of O(n^2)
println!("Complexity reduction: {:.1}x", config.complexity_reduction(4096));
```

## LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning:

```rust
use tensorlogic_trustformers::{LoRAConfig, LoRAAttention, LoRAPreset};

// Standard LoRA configuration
let config = LoRAPreset::standard(512, 8)?;
let lora_attn = LoRAAttention::new(config)?;

// Compression ratio
println!("Parameter reduction: {:.0}x", config.compression_ratio());
```

## Mixture-of-Experts (MoE)

Sparse conditional computation:

```rust
use tensorlogic_trustformers::{MoeConfig, MoeLayer, MoePreset, RouterType};

// Mixtral 8x7B style
let config = MoePreset::mixtral_8x7b();
let moe = MoeLayer::new(config)?;

// Custom MoE
let config = MoeConfig::new(512, 8, RouterType::TopK(2))?
    .with_load_balancing(0.01);
```

## Vision Transformers (ViT)

Image recognition with transformer architecture:

```rust
use tensorlogic_trustformers::{VisionTransformer, ViTPreset};

// ViT-Base/16 configuration
let config = ViTPreset::base();
let vit = VisionTransformer::new(config)?;

println!("Parameters: {:.1}M", config.num_parameters() as f64 / 1e6);
```

Available presets: Tiny (5.7M), Small (22M), Base (86M), Large (307M), Huge (632M)

## Gradient Checkpointing

Memory-efficient training for large models:

```rust
use tensorlogic_trustformers::{CheckpointConfig, EncoderStackConfig};

let config = EncoderStackConfig::new(12, 768, 12, 3072, 512)?;

// Uniform checkpointing: checkpoint every 2 layers
let checkpoint = CheckpointConfig::uniform(2);
println!("Memory savings: {:.1}%", checkpoint.memory_savings(12) * 100.0);
println!("Compute overhead: {:.2}x", checkpoint.compute_overhead(12));

// Selective checkpointing: checkpoint specific layers
let checkpoint = CheckpointConfig::selective(vec![0, 3, 6, 9]);

// Dynamic checkpointing: automatically balance memory vs. compute
let checkpoint = CheckpointConfig::dynamic(12, 0.3)?;
```

## KV-Cache for Fast Inference

Enable efficient autoregressive generation with dramatic speedups:

```rust
use tensorlogic_trustformers::{KVCache, KVCacheConfig};

// Create cache for 12-layer model (GPT-2 small)
let mut cache = KVCache::new(12, 12, 64);

// Monitor cache usage
let stats = cache.stats();
println!("{}", stats.summary());
```

Benefits:
- **10-1000x speedup** depending on sequence length
- Minimal memory cost: ~2-10 MB for typical models
- Essential for production text generation

## Rule-Based Attention

Integrate logical rules with attention mechanisms:

```rust
use tensorlogic_trustformers::{RuleAttentionConfig, RuleBasedAttention};

// Hard constraint: only attend where rule is satisfied
let base_attn = AttentionConfig::new(512, 8)?;
let config = RuleAttentionConfig::hard(base_attn);

// Soft constraint: bias attention towards rule-satisfying positions
let config = RuleAttentionConfig::soft(base_attn, 0.7);

// Gated: interpolate between content and rule attention
let config = RuleAttentionConfig::gated(base_attn, 0.5);
```

## TrustformeRS Integration

Bidirectional conversion with the TrustformeRS ecosystem:

```rust
use tensorlogic_trustformers::{TrustformersConverter, IntegrationConfig};

// Convert TrustformeRS architectures (BERT, GPT, T5) to TLExpr
let converter = TrustformersConverter::new(config)?;
let tlexpr = converter.convert_bert_encoder(bert_config)?;

// Load pretrained weights
let loader = TrustformersWeightLoader::new();
let weights = loader.load_checkpoint("model.bin")?;
```

## Model Presets

```rust
use tensorlogic_trustformers::{presets, utils::encoder_stack_stats};

// Standard presets
let gpt2 = presets::gpt2_small();
let bert = presets::bert_base();
let (encoder, decoder) = presets::transformer_base();

// Get model statistics
let stats = encoder_stack_stats(&gpt2);
println!("{}", stats.summary());
// ModelStats:
//   Total params: 117.00M
//   Trainable: 117.00M
//   Layers: 12
//   d_model: 768
//   Memory: 468 MB
```

## Integration with TensorLogic

The einsum graphs produced by this crate integrate seamlessly with the TensorLogic ecosystem:

```rust
use tensorlogic_compiler::CompilerContext;
use tensorlogic_scirs_backend::Scirs2Executor;

// Compile the transformer graph
let mut ctx = CompilerContext::new();
// ... compile transformer einsum graph

// Execute on SciRS2 backend
let executor = Scirs2Executor::new();
// ... execute the graph
```

## Design Philosophy

1. **Backend Independence**: Same graph works on CPU, GPU, TPU
2. **Einsum-Native**: Clear mathematical semantics
3. **Composability**: Mix transformer layers with logical rules
4. **Type Safety**: Compile-time dimension checking where possible
5. **Zero Cost Abstractions**: No runtime overhead

## Examples

See the [examples directory](examples/) for 10 complete examples:

- `01_basic_encoder.rs` - Basic transformer encoder usage
- `02_trustformers_integration.rs` - TrustformeRS integration
- `03_rule_based_attention.rs` - Rule-based attention patterns
- `04_sparse_attention.rs` - Sparse attention for long sequences
- `05_gradient_checkpointing.rs` - Memory-efficient training strategies
- `06_kv_cache_inference.rs` - Fast autoregressive generation with KV-cache
- `07_vision_transformers.rs` - Vision Transformer (ViT) for image classification
- `08_mixture_of_experts.rs` - Mixture-of-Experts for sparse models
- `09_modern_llm_optimizations.rs` - GQA, Sliding Window, LoRA
- `10_modern_llm_complete.rs` - Complete modern LLM configurations

## Testing

```bash
cargo nextest run -p tensorlogic-trustformers
# 488 tests, all passing, zero warnings
```

## Benchmarking

```bash
cargo bench --bench model_benchmarks
```

This generates HTML reports in `target/criterion/` with detailed performance metrics.

## Performance

The einsum-based approach enables:

- **Operation Fusion**: Compiler can fuse consecutive operations
- **Memory Efficiency**: Minimal intermediate tensors
- **Parallelization**: Natural SIMD/GPU mapping
- **Optimization**: Graph-level optimizations

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Tensor Logic Paper](https://arxiv.org/abs/2510.12269) - TensorLogic framework
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-rank adaptation

## License

Apache-2.0

---

**Status**: Stable (v0.1.3)
**Last Updated**: 2026-08-31
**Tests**: 488/488 passing (100%)
**Examples**: 10 comprehensive examples
**Benchmarks**: Criterion suite with HTML reports
**Features**: Complete transformer implementation with modern LLM optimizations
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
