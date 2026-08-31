# TensorLogic Trustformers — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Transformer building blocks (attention, KV-cache, rule-guided decoding).

## Completed

- [x] Basic crate structure
- [x] Error handling module with IrError conversion
- [x] Configuration system (AttentionConfig, FeedForwardConfig, TransformerLayerConfig)
- [x] **Self-attention as einsum**
  - [x] Q, K, V projections
  - [x] Attention scores: einsum("bqd,bkd->bqk")
  - [x] Scaled attention with sqrt(d_k)
  - [x] Softmax application
  - [x] Weighted values: einsum("bqk,bkv->bqv")
- [x] **Multi-head attention**
  - [x] Split heads (reshape to [batch, n_heads, seq, d_k])
  - [x] Parallel attention per head
  - [x] Concatenate outputs
  - [x] Transpose operations for head management
- [x] **Feed-forward networks**
  - [x] Linear transformations as einsum
  - [x] Non-linearities (GELU, ReLU, configurable)
  - [x] Bias addition
  - [x] Two-layer FFN architecture
- [x] **Gated FFN (GLU variant)**
  - [x] Gate and value projections
  - [x] Element-wise gating
  - [x] Output projection

## High Priority - COMPLETE

### Rule-Based Transformers
- [x] **Attention as logical rules**
  - [x] Define attention patterns with TLExpr
  - [x] Compile to tensor operations
  - [x] Interpretable attention
- [x] **Structured attention**
  - [x] Tree-based attention (via predicates)
  - [x] Graph-based attention (via predicates)
  - [x] Hierarchical attention (via patterns)

### TrustformeRS Integration
- [x] Implement TrustformeRS module trait adapter
- [x] Convert Transformer layers to TLExpr
- [x] Bidirectional integration (TensorLogic <-> TrustformeRS)
- [x] Pre-trained model loading (checkpoint format support)
- [x] Weight mapping utilities
- [x] 19 comprehensive integration tests

## Medium Priority - COMPLETE

### Advanced Features
- [x] Position encodings
  - [x] Sinusoidal
  - [x] Learned
  - [x] Relative (with bias)
  - [x] RoPE (Rotary Position Embedding)
  - [x] ALiBi (Attention with Linear Biases)
- [x] Layer normalization
  - [x] Standard LayerNorm
  - [x] RMSNorm (efficient variant)
- [x] Dropout (configuration support)
- [x] **Gradient checkpointing**
  - [x] Uniform checkpointing strategy
  - [x] Selective checkpointing strategy
  - [x] Dynamic checkpointing strategy
  - [x] Memory savings calculation
  - [x] Compute overhead estimation
  - [x] Configuration builder API
  - [x] 16 comprehensive tests
- [x] **Encoder layers and stacks**
  - [x] EncoderLayer with pre-norm and post-norm variants
  - [x] EncoderStack with configurable depth
  - [x] Position encoding integration
  - [x] Final layer norm option
- [x] **Decoder layers and stacks**
  - [x] DecoderLayer with masked self-attention
  - [x] Cross-attention to encoder
  - [x] DecoderStack configuration
- [x] **Sparse attention**
  - [x] Strided sparse attention
  - [x] Local windowed attention (LocalAttention)
  - [x] Block-sparse attention
  - [x] Global-local attention

### Model Variants
- [x] BERT-style encoders (via EncoderStack)
- [x] GPT-style decoders (via DecoderStack with causal masking)
- [x] Encoder-decoder models (via EncoderStack + DecoderStack)
- [x] **Vision Transformers (ViT)** (`vision` module)
  - [x] PatchEmbedding layer
  - [x] VisionTransformer configuration
  - [x] ViTPreset: Tiny (5.7M), Small (22M), Base (86M), Large (307M), Huge (632M)
  - [x] Parameter counting
  - [x] Graph building
  - [x] 12 comprehensive tests
  - [x] Complete example (07_vision_transformers.rs)
- [x] **Mixture-of-Experts (MoE)** (`moe` module)
  - [x] Expert networks (multiple FFN layers)
  - [x] Router/Gating mechanisms (TopK, Softmax, Switch, ExpertChoice)
  - [x] Load balancing support
  - [x] MoePreset: Switch, GShard, Mixtral8x7B, ExpertChoice
  - [x] Sparsity analysis and efficiency metrics
  - [x] FLOPs and memory usage calculations
  - [x] 15 comprehensive tests
  - [x] Complete example (08_mixture_of_experts.rs)

## Modern LLM Optimizations - COMPLETE

- [x] **Flash Attention** (`flash_attention` module)
  - [x] Memory-efficient O(1) attention
  - [x] Tiled computation with SRAM optimization
  - [x] Configurable block sizes for Q and KV
  - [x] FlashAttentionPreset: A100, H100 GPUs
  - [x] Causal masking support
  - [x] FlashAttentionV2Config
  - [x] FlashAttentionStats
  - [x] 14 comprehensive tests

- [x] **Grouped-Query Attention (GQA)** (`gqa` module)
  - [x] MHA/GQA/MQA support with configurable KV heads
  - [x] GQAPreset: LLaMA 2, Mistral, Falcon
  - [x] Memory savings calculations
  - [x] GQAStats
  - [x] 13 comprehensive tests

- [x] **Sliding Window Attention** (`sliding_window` module)
  - [x] O(n*w) complexity instead of O(n^2)
  - [x] SlidingWindowPreset: Mistral, Longformer, BigBird
  - [x] Complexity/memory reduction analysis
  - [x] SlidingWindowStats
  - [x] 9 comprehensive tests

- [x] **LoRA (Low-Rank Adaptation)** (`lora` module)
  - [x] Configurable rank and alpha
  - [x] Apply to Q/V projections
  - [x] LoRALinear, LoRAAttention
  - [x] LoRAPreset for standard configurations
  - [x] LoRAStats with compression ratio
  - [x] 14 comprehensive tests

- [x] Complete examples:
  - [x] 09_modern_llm_optimizations.rs - Individual optimization demos
  - [x] 10_modern_llm_complete.rs - Complete modern LLM configurations

## Low Priority - COMPLETE

### Documentation
- [x] Add README.md (comprehensive documentation)
- [x] Architecture guide (in README.md)
- [x] 10 complete examples in `examples/`
  - [x] 01_basic_encoder.rs
  - [x] 02_trustformers_integration.rs
  - [x] 03_rule_based_attention.rs
  - [x] 04_sparse_attention.rs
  - [x] 05_gradient_checkpointing.rs
  - [x] 06_kv_cache_inference.rs
  - [x] 07_vision_transformers.rs
  - [x] 08_mixture_of_experts.rs
  - [x] 09_modern_llm_optimizations.rs
  - [x] 10_modern_llm_complete.rs

### Performance Infrastructure
- [x] **Benchmark suite**
  - [x] Self-attention benchmarks
  - [x] Multi-head attention benchmarks
  - [x] Feed-forward network benchmarks
  - [x] Encoder stack benchmarks
  - [x] Configuration validation benchmarks
  - [x] Criterion integration with HTML reports
- [x] **KV-cache for efficient inference** (`kv_cache` module)
  - [x] Cache configuration with builder API
  - [x] Layer-wise cache management
  - [x] Memory usage tracking and statistics
  - [x] Automatic cache initialization
  - [x] Cache clearing and reset operations
  - [x] CacheStats with summary
  - [x] 21 comprehensive tests
- [ ] Performance comparison with baseline (future)

---

**Total Items:** 84 tasks
**Completion:** 100% (84/84)

**Tests:** 346/346 passing (100%)
**Warnings:** 0
**Build:** Success
**Documentation:** Complete
**Integration:** TrustformeRS fully integrated
**Benchmarks:** Criterion suite ready
**Examples:** 10 comprehensive examples
**Optimizations:** Flash Attention + GQA + SWA + LoRA + MoE + KV-cache + Checkpointing

**Status:** Stable (v0.1.0)
**Release Date:** 2026-04-06

## v0.1.10 Enhancements (2026-03-31)

- [x] **KV-Cache for Autoregressive Inference** (`kv_cache.rs`): `KvCache` per-layer key/value cache with configurable capacity and `append()`/`get_keys()`/`get_values()` API
- [x] **Rotary Position Embedding** (`kv_cache.rs`): `RotaryPositionEmbedding` (RoPE) with precomputed sin/cos caches, `apply()` on query/key tensors, `rotate_half()` primitive
- [x] **T5-style Relative Position Bias** (`kv_cache.rs`): `RelativePositionBias` with configurable bucket count, max distance, and bidirectional mode; `relative_position_bucket()` for log-spaced bucketing
- [x] **Cached Multi-Head Attention** (`kv_cache.rs`): `CachedAttention` combining KV-cache + causal mask generation (`causal_mask()`), `forward()` with optional RoPE application and incremental decode support
- [x] **Inference Statistics** (`kv_cache.rs`): `InferenceStats` tracking `tokens_generated`, `cache_hits`, `memory_bytes`; `record_step()` and `summary()` for monitoring autoregressive inference throughput

## v0.1.13

- [x] **RmsNorm** (`normalization_variants.rs`): Root-mean-square normalization with learnable gamma parameter, configurable epsilon, and `update_gamma()` for weight updates
- [x] **GroupNorm** (`normalization_variants.rs`): Group normalization dividing channels into configurable `num_groups`, with per-group mean/variance computation and optional affine transform
- [x] **InstanceNorm** (`normalization_variants.rs`): Per-instance per-channel normalization with optional affine parameters, suitable for style transfer and generative models
- [x] **BatchNorm** (`normalization_variants.rs`): Batch normalization with running mean/variance (EMA), configurable momentum, train/eval mode toggle, and affine transform
- [x] **WeightNorm** (`normalization_variants.rs`): Weight reparametrization separating direction `v` from magnitude `g`, with `apply()` and `remove()` operations for reparametrized forward passes
- [x] **NormStats** (`normalization_variants.rs`): Compute and summarize normalization statistics (mean, variance, min, max, range) across arbitrary tensors with `summary()` output

### 2026-04-14 — File split refactor

- Split `src/kv_cache.rs` (1959L) into `src/kv_cache/` directory with 6 files (mod.rs 54L, config.rs 688L, simple_cache.rs 287L, position.rs 472L, cached_attention.rs 390L, stats.rs 93L) to stay well under the 2,000-line hard cap and 1,500-line soft target.
- Public API surface preserved via `mod.rs` re-exports; all existing tests still pass (346/346).

## Future Enhancements

- [ ] Pre-trained model weight import (beyond checkpoint format)
- [ ] Advanced pattern composition
- [ ] GPU-specific optimizations
- [x] `quantization-int8-int4` (planned 2026-04-17)
  - **Goal:** Greenfield quantization module — `QuantizedLinear` with packed `Array2<i8>` weights + per-channel scale, CPU-first dequantize→matmul forward, PTQ via `tensorlogic-scirs-backend`'s existing `calibrate_quantization`. Also extend `tensorlogic-scirs-backend`'s per-channel helper (currently uses `scale[0]` even for PerChannel — fix to use `scale[c]` per output channel).
  - **Design (Paradigm B — numerical layers, modeled after `src/moe/`):** New module `src/quantization/` with `mod.rs`, `linear.rs`, `calibration.rs`, `tests.rs`. `pub struct QuantizedLinear { weight_q: Array2<i8>, scale: Vec<f64>, zero_point: Vec<i32>, granularity: QuantizationGranularity, bias: Option<Array1<f64>> }`. `QuantizedLinear::from_fp(linear: &LinearExpert, params: QuantizationParams) -> Self` — quantizes weights via scirs-backend's `quantize_int8`/`quantize_int4`. `QuantizedLinear::forward(&self, x: &Array2<f64>) -> Array2<f64>` — dequantize on the fly, then f64 matmul (CPU-first; integer-matmul kernel is a future follow-up). Calibration helper wraps `tensorlogic-scirs-backend::quantization::calibrate_quantization`. scirs-backend per-channel fix: extend `quantize_int8`/`quantize_int4` to use `scale[c]`/`zero_point[c]` per channel when `granularity == PerChannel` (param vectors are already Vec-based — only the loop index changes).
  - **Files:** `src/quantization/mod.rs` (NEW); `src/quantization/linear.rs` (NEW); `src/quantization/calibration.rs` (NEW); `src/quantization/tests.rs` (NEW); `src/lib.rs` (export quantization module); `Cargo.toml` (add `tensorlogic-scirs-backend = { workspace = true }`); `crates/tensorlogic-scirs-backend/src/quantization.rs` (extend per-channel paths in `quantize_int8`/`quantize_int4`).
  - **Prerequisites:** the per-channel scirs-backend extension is in-scope for this subagent — keeps the change atomic.
  - **Tests:** round-trip quantize/dequantize accuracy bounds (per-tensor + per-channel); `QuantizedLinear::forward` matches `LinearExpert::forward` within tolerance; calibration produces sensible (scale, zero_point); per-channel granularity uses different scales per output row. scirs-backend: extend existing quantization tests to assert per-channel uses `scale[c]`.
  - **Risk:** Dep coupling with scirs-backend is new but necessary; confined to quantization module — gate behind `[features] quantization = ["dep:tensorlogic-scirs-backend"]` if needed. Dequantize-then-matmul is slower than packed-int8 GEMM; documented in module; integer-matmul kernel is a future follow-up.
- [ ] Performance comparison with baseline implementations

## v0.2.0 Research Preview (2026-04-15)

- [x] **Speculative Decoding** — `speculative_decoding` module. Model-level speculative decoding per Leviathan et al. (2023) / Chen et al. (2023). A cheap draft model proposes `k` tokens; a target model verifies them in a single parallel forward pass. Features:
  - `DraftModel` + `TargetModel` traits returning full per-position log-prob distributions (not just per-token scores) so the adjusted resampling step over `max(0, p_target - p_draft)` is well-defined.
  - Pure-function `acceptance.rs` primitives: `accept(draft_lp, target_lp, rng)` (Bernoulli `min(1, p_target / p_draft)`) and `resample_from_adjusted_target(...)` (samples from the normalized `max(0, p_target - p_draft)` distribution); `sample_from_logprobs` for the bonus position on full acceptance.
  - `SpeculativeDecoder<D, T>` engine with configurable draft depth `k` (default 4), deterministic `StdRng` seeding, optional EOS early-stop, and a caller-supplied-RNG variant (`generate_with_rng`).
  - `SpeculativeMetrics` tracking `accept_rate`, `tokens_per_step_avg`, `speedup_estimate`, plus round/token counts; runtime-updated per round inside the engine.
  - Deterministic `FixedDistDraftModel` / `FixedDistTargetModel` (alias `MockDraftModel` / `MockTargetModel`) test fixtures over constant categorical distributions.
  - Self-contained `SpeculativeDecodingError` via `thiserror` with a `From` bridge into `TrustformerError`; dedicated `SpecRng` object-safe shim backed by any SciRS2 `Rng + RngExt`.
  - **Correctness theorem verified empirically**: `empirical_distribution_matches_target` and `empirical_distribution_matches_target_multi_step` tests run 10 000 samples through a deliberately miscalibrated draft and pass a Pearson chi-square fit to `p_target` at α = 0.01.
  - 44 unit tests + 2 integration tests (`tests/speculative_decoding_integration.rs`).
- [x] **Rule-Guided Sampling Decoder** — `rule_guided_decoder` module. Logic-constrained generation on top of `tensorlogic-infer::BeamSearchDecoder`. Features:
  - `RuleConstraint` compiled from `TLExpr::Pred` / `And` / `Or` via a user-supplied token-to-symbol mapper (`fn(TokenId) -> Option<SymbolName>`).
  - Two enforcement strategies: `HardMask` (forbidden logits → `-inf`) and `SoftPenaltyMask { lambda }` (log-penalty on soft violations; forbidden logits still banned).
  - `RuleGuidedBeamSearch` façade that wraps the existing beam search; the wrapping closure applies masks per-beam before softmax so length penalty / temperature / top-k / EOS bookkeeping are inherited unchanged.
  - Extension point (`extend_tlexpr_support` doc marker) for additional `TLExpr` variants — unsupported variants fall back to a no-op `SoftPenalty(0.0)` verdict.
  - 23 unit tests + 3 integration tests (`tests/rule_guided_decoder_integration.rs`).
  - Self-contained `RuleGuidedError` via `thiserror` with a `From` bridge into `TrustformerError`.

- [x] **Numerical Mixture-of-Experts** — `moe/` research-preview submodules (`error`, `expert`, `gate`, `layer`, `load_balance`, `tests`). Features:
  - `TopKGate` with `xavier_init` (deterministic seed) and `from_weights` constructors; top-k softmax gating with full softmax for auxiliary losses.
  - `LinearExpert` implementing the `Expert` trait (`forward` as `Wx + b`); identity / constant experts for testing.
  - `MoELayer`: gate + experts + optional Switch-Transformer capacity-factor dropping (Fedus et al., 2022). `forward` (single input) and `forward_batch` (batched, with capacity capping and `BatchGatingStats`).
  - Load-balancing losses: `importance_loss` (CV-squared of softmax means), `load_loss` (CV-squared of hard routing counts), `combined_aux_loss`.
  - 12 unit tests (top-k gating, linear expert, identity round-trip, top-1 routing, importance/load balance, capacity dropping, error paths) + 2 integration tests (4-expert batch-32 routing, capacity-factor overflow).

- [x] **Longformer-Style Sparse Attention** — `sparse_attention/` submodules (`error`, `config`, `mask`, `attention`, `tests`). Numerical sliding-window + global-token attention per Beltagy et al. (2020). Features:
  - `LongformerConfig` with `window_size`, `global_token_indices`, `num_heads`, `head_dim`, `causal`, `dropout`; builder API with validation.
  - `LongformerMask`: dense `seq_len x seq_len` boolean attendance matrix with `build_mask()`, `is_attended()`, `sparsity()` metrics.
  - `LongformerAttention` engine with `forward()` (auto-builds mask) and `forward_with_mask()` (pre-built mask reuse); multi-head scaled dot-product attention with sparse masking, numerically stable softmax via max-subtraction.
  - `attention_weights()` diagnostic method for per-head weight inspection.
  - Self-contained `SparseAttentionError` via `thiserror` with `From` bridge into `TrustformerError`.
  - Original graph-building types renamed to `SparseAttentionGraph` / `SparseAttentionGraphConfig` to avoid collision.
  - 12 unit tests across submodules + 4 integration tests (`tests/sparse_attention_integration.rs`).

## v0.2.0 / Future Work

(Previously enumerated items — MoE, sparse attention, Flash Attention 2 — are already shipped; see the respective "COMPLETE" sections above.  The rule-guided sampling decoder moved into the v0.2.0 research preview bullets above.)
