//! Complete Modern LLM Configuration Example
//!
//! This example demonstrates how to configure a complete modern LLM
//! using all available optimization techniques:
//!
//! - Flash Attention (memory-efficient O(1) attention)
//! - Grouped-Query Attention (GQA - reduced KV cache)
//! - Sliding Window Attention (efficient long sequences)
//! - LoRA (parameter-efficient fine-tuning)
//! - RoPE (Rotary Position Encoding)
//!
//! These configurations mirror real-world models like Mistral-7B and LLaMA-2-70B.

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    FlashAttentionConfig, FlashAttentionPreset, GQAConfig, GQAPreset, LoRAConfig, LoRAPreset,
    MoeConfig, MoePreset, PositionEncodingConfig, SlidingWindowConfig, SlidingWindowPreset,
};

/// Complete modern LLM configuration
#[derive(Debug, Clone)]
pub struct ModernLLMConfig {
    /// Model name
    pub name: String,
    /// Model dimension
    pub d_model: usize,
    /// Number of query heads
    pub n_heads: usize,
    /// Number of key-value heads (for GQA)
    pub n_kv_heads: usize,
    /// FFN hidden dimension
    pub d_ff: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Use Flash Attention
    pub use_flash_attention: bool,
    /// Use Sliding Window Attention
    pub use_sliding_window: bool,
    /// Window size for sliding window
    pub window_size: usize,
    /// Use RoPE
    pub use_rope: bool,
    /// RoPE base frequency
    pub rope_base: f64,
    /// Use LoRA for fine-tuning
    pub use_lora: bool,
    /// LoRA rank
    pub lora_rank: usize,
    /// Use Mixture of Experts
    pub use_moe: bool,
    /// Number of experts (for MoE)
    pub n_experts: usize,
    /// Top-k experts (for MoE)
    pub top_k: usize,
}

impl ModernLLMConfig {
    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self {
            name: "Mistral-7B".to_string(),
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8, // GQA with 8 KV heads
            d_ff: 14336,
            n_layers: 32,
            max_seq_len: 32768,
            vocab_size: 32000,
            use_flash_attention: true,
            use_sliding_window: true,
            window_size: 4096,
            use_rope: true,
            rope_base: 10000.0,
            use_lora: false,
            lora_rank: 0,
            use_moe: false,
            n_experts: 0,
            top_k: 0,
        }
    }

    /// LLaMA 2 70B configuration
    pub fn llama2_70b() -> Self {
        Self {
            name: "LLaMA-2-70B".to_string(),
            d_model: 8192,
            n_heads: 64,
            n_kv_heads: 8, // GQA with 8 KV heads
            d_ff: 28672,
            n_layers: 80,
            max_seq_len: 4096,
            vocab_size: 32000,
            use_flash_attention: true,
            use_sliding_window: false,
            window_size: 0,
            use_rope: true,
            rope_base: 10000.0,
            use_lora: false,
            lora_rank: 0,
            use_moe: false,
            n_experts: 0,
            top_k: 0,
        }
    }

    /// Mixtral 8x7B configuration
    pub fn mixtral_8x7b() -> Self {
        Self {
            name: "Mixtral-8x7B".to_string(),
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8, // GQA with 8 KV heads
            d_ff: 14336,
            n_layers: 32,
            max_seq_len: 32768,
            vocab_size: 32000,
            use_flash_attention: true,
            use_sliding_window: true,
            window_size: 4096,
            use_rope: true,
            rope_base: 10000.0,
            use_lora: false,
            lora_rank: 0,
            use_moe: true,
            n_experts: 8,
            top_k: 2,
        }
    }

    /// Phi-2 configuration (Microsoft)
    pub fn phi_2() -> Self {
        Self {
            name: "Phi-2".to_string(),
            d_model: 2560,
            n_heads: 32,
            n_kv_heads: 32, // Standard MHA
            d_ff: 10240,
            n_layers: 32,
            max_seq_len: 2048,
            vocab_size: 50304,
            use_flash_attention: true,
            use_sliding_window: false,
            window_size: 0,
            use_rope: true,
            rope_base: 10000.0,
            use_lora: false,
            lora_rank: 0,
            use_moe: false,
            n_experts: 0,
            top_k: 0,
        }
    }

    /// Add LoRA fine-tuning configuration
    pub fn with_lora(mut self, rank: usize) -> Self {
        self.use_lora = true;
        self.lora_rank = rank;
        self
    }

    /// Calculate total parameters (approximate)
    pub fn total_parameters(&self) -> u64 {
        let embed_params = self.vocab_size * self.d_model;
        let attn_params = self.n_layers * (3 * self.d_model * self.d_model + self.d_model);
        let ffn_params = self.n_layers * (3 * self.d_model * self.d_ff);
        let lm_head = self.vocab_size * self.d_model;

        let base_params = (embed_params + attn_params + ffn_params + lm_head) as u64;

        if self.use_moe {
            // MoE multiplies FFN params by num_experts
            let moe_multiplier = self.n_experts as u64;
            base_params
                + (self.n_layers * 3 * self.d_model * self.d_ff) as u64 * (moe_multiplier - 1)
        } else {
            base_params
        }
    }

    /// Calculate LoRA trainable parameters
    pub fn lora_parameters(&self) -> u64 {
        if !self.use_lora || self.lora_rank == 0 {
            return 0;
        }
        // Q, K, V, O projections with LoRA
        let lora_per_layer = 4 * 2 * self.d_model * self.lora_rank;
        (self.n_layers * lora_per_layer) as u64
    }

    /// Calculate KV cache memory reduction from GQA
    pub fn kv_cache_reduction(&self) -> f64 {
        if self.n_kv_heads == self.n_heads {
            1.0 // No reduction (standard MHA)
        } else {
            self.n_kv_heads as f64 / self.n_heads as f64
        }
    }

    /// Print configuration summary
    pub fn summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Model: {}", self.name);
        println!("{}", "=".repeat(60));

        println!("\nArchitecture:");
        println!("  d_model:      {}", self.d_model);
        println!("  n_heads:      {}", self.n_heads);
        println!("  n_kv_heads:   {}", self.n_kv_heads);
        println!("  d_ff:         {}", self.d_ff);
        println!("  n_layers:     {}", self.n_layers);
        println!("  max_seq_len:  {}", self.max_seq_len);
        println!("  vocab_size:   {}", self.vocab_size);

        println!("\nOptimizations:");
        println!(
            "  Flash Attention:    {}",
            if self.use_flash_attention {
                "Yes"
            } else {
                "No"
            }
        );
        if self.use_sliding_window {
            println!("  Sliding Window:     Yes (size: {})", self.window_size);
        } else {
            println!("  Sliding Window:     No");
        }
        println!(
            "  RoPE:               {}",
            if self.use_rope {
                format!("Yes (base: {})", self.rope_base)
            } else {
                "No".to_string()
            }
        );
        if self.use_lora {
            println!("  LoRA:               Yes (rank: {})", self.lora_rank);
        } else {
            println!("  LoRA:               No");
        }
        if self.use_moe {
            println!(
                "  MoE:                Yes ({} experts, top-{})",
                self.n_experts, self.top_k
            );
        } else {
            println!("  MoE:                No");
        }

        println!("\nStatistics:");
        let total_params = self.total_parameters();
        println!("  Total parameters:   {:.2}B", total_params as f64 / 1e9);

        let kv_reduction = self.kv_cache_reduction();
        if kv_reduction < 1.0 {
            println!(
                "  KV cache reduction: {:.1}% (via GQA)",
                (1.0 - kv_reduction) * 100.0
            );
        }

        if self.use_lora {
            let lora_params = self.lora_parameters();
            println!(
                "  LoRA parameters:    {:.2}M ({:.3}% of total)",
                lora_params as f64 / 1e6,
                lora_params as f64 / total_params as f64 * 100.0
            );
        }
    }
}

/// Build component graphs for the modern LLM
fn build_component_graphs(config: &ModernLLMConfig) {
    println!("\nBuilding Component Graphs:");
    println!("{}", "-".repeat(40));

    // 1. GQA Configuration
    if config.n_kv_heads != config.n_heads {
        let gqa_config = GQAConfig::new(config.d_model, config.n_heads, config.n_kv_heads)
            .unwrap()
            .with_causal(true);

        let mut gqa_graph = EinsumGraph::new();
        gqa_graph.add_tensor("Q");
        gqa_graph.add_tensor("K");
        gqa_graph.add_tensor("V");

        let gqa = tensorlogic_trustformers::GroupedQueryAttention::new(gqa_config).unwrap();
        let outputs = gqa.build_gqa_graph(&mut gqa_graph).unwrap();
        println!(
            "  GQA graph: {} nodes, {} outputs",
            gqa_graph.node_count(),
            outputs.len()
        );
    }

    // 2. Flash Attention
    if config.use_flash_attention {
        let flash_config = FlashAttentionConfig::new(config.d_model, config.n_heads)
            .unwrap()
            .with_causal(true);

        let mut flash_graph = EinsumGraph::new();
        flash_graph.add_tensor("Q");
        flash_graph.add_tensor("K");
        flash_graph.add_tensor("V");

        let flash = tensorlogic_trustformers::FlashAttention::new(flash_config).unwrap();
        let outputs = flash.build_flash_graph(&mut flash_graph).unwrap();
        println!(
            "  Flash Attention graph: {} nodes, {} outputs",
            flash_graph.node_count(),
            outputs.len()
        );
    }

    // 3. Sliding Window
    if config.use_sliding_window {
        let swa_config =
            SlidingWindowConfig::new(config.d_model, config.n_heads, config.window_size)
                .unwrap()
                .with_causal(true);

        let mut swa_graph = EinsumGraph::new();
        swa_graph.add_tensor("Q");
        swa_graph.add_tensor("K");
        swa_graph.add_tensor("V");

        let swa = tensorlogic_trustformers::SlidingWindowAttention::new(swa_config).unwrap();
        let outputs = swa.build_swa_graph(&mut swa_graph).unwrap();
        println!(
            "  Sliding Window graph: {} nodes, {} outputs",
            swa_graph.node_count(),
            outputs.len()
        );
    }

    // 4. LoRA
    if config.use_lora {
        let lora_config = LoRAConfig::new(config.lora_rank, 1.0);

        let mut lora_graph = EinsumGraph::new();
        lora_graph.add_tensor("Q");
        lora_graph.add_tensor("K");
        lora_graph.add_tensor("V");

        let lora = tensorlogic_trustformers::LoRAAttention::new(
            config.d_model,
            config.n_heads,
            lora_config,
        )
        .unwrap();
        let outputs = lora.build_lora_attention_graph(&mut lora_graph).unwrap();
        println!(
            "  LoRA Attention graph: {} nodes, {} outputs",
            lora_graph.node_count(),
            outputs.len()
        );
    }

    // 5. MoE
    if config.use_moe {
        let moe_config =
            MoeConfig::new(config.n_experts, config.d_model, config.d_ff, config.top_k).unwrap();

        let mut moe_graph = EinsumGraph::new();
        moe_graph.add_tensor("x");

        let moe = tensorlogic_trustformers::MoeLayer::new(moe_config).unwrap();
        let outputs = moe.build_moe_graph(&mut moe_graph).unwrap();
        println!(
            "  MoE graph: {} nodes, {} outputs",
            moe_graph.node_count(),
            outputs.len()
        );
    }

    // 6. Position Encoding (RoPE)
    if config.use_rope {
        let rope_config = PositionEncodingConfig::rotary(config.d_model, config.max_seq_len);

        let mut rope_graph = EinsumGraph::new();
        rope_graph.add_tensor("x");
        rope_graph.add_tensor("freqs");

        let rope = tensorlogic_trustformers::RotaryPositionEncoding::new(rope_config).unwrap();
        let outputs = rope.build_encoding_graph(&mut rope_graph).unwrap();
        println!(
            "  RoPE graph: {} nodes, {} outputs",
            rope_graph.node_count(),
            outputs.len()
        );
    }
}

fn main() {
    println!("Modern LLM Complete Configuration Examples");
    println!("{}", "=".repeat(60));

    // 1. Mistral 7B
    let mistral = ModernLLMConfig::mistral_7b();
    mistral.summary();
    build_component_graphs(&mistral);

    // 2. LLaMA 2 70B
    let llama = ModernLLMConfig::llama2_70b();
    llama.summary();
    build_component_graphs(&llama);

    // 3. Mixtral 8x7B (with MoE)
    let mixtral = ModernLLMConfig::mixtral_8x7b();
    mixtral.summary();
    build_component_graphs(&mixtral);

    // 4. Phi-2 with LoRA fine-tuning
    let phi2_lora = ModernLLMConfig::phi_2().with_lora(16);
    phi2_lora.summary();
    build_component_graphs(&phi2_lora);

    // 5. Compare using presets
    println!("\n{}", "=".repeat(60));
    println!("Preset Configurations Comparison");
    println!("{}", "=".repeat(60));

    println!("\nGQA Presets:");
    for preset in [
        GQAPreset::Llama2_7B,
        GQAPreset::Llama2_70B,
        GQAPreset::Mistral7B,
    ] {
        let config = preset.config().unwrap();
        println!(
            "  {:15} - {} heads, {} KV heads, ratio: {:.2}x",
            preset.name(),
            config.n_heads,
            config.n_kv_heads,
            config.n_heads as f64 / config.n_kv_heads as f64
        );
    }

    println!("\nFlash Attention Presets:");
    for preset in [
        FlashAttentionPreset::SmallBlocks,
        FlashAttentionPreset::Standard,
        FlashAttentionPreset::LargeBlocks,
    ] {
        let config = preset.config(512, 8).unwrap();
        println!(
            "  {:15} - {} blocks, d_model: {}",
            preset.name(),
            config.block_size_q,
            config.d_model
        );
    }

    println!("\nSliding Window Presets:");
    for preset in [
        SlidingWindowPreset::Mistral7B,
        SlidingWindowPreset::LongformerBase,
        SlidingWindowPreset::BigBirdBase,
    ] {
        let config = preset.config().unwrap();
        println!(
            "  {:15} - window: {}, d_model: {}",
            preset.name(),
            config.window_size,
            config.d_model
        );
    }

    println!("\nLoRA Presets:");
    for preset in [
        LoRAPreset::Minimal,
        LoRAPreset::Standard,
        LoRAPreset::Extended,
    ] {
        let config = preset.config();
        println!(
            "  {:15} - rank: {}, alpha: {:.1}",
            preset.name(),
            config.rank,
            config.alpha
        );
    }

    println!("\nMoE Presets:");
    for preset in [
        MoePreset::Mixtral8x7B,
        MoePreset::Switch,
        MoePreset::ExpertChoice,
    ] {
        let config = preset.config(4096, 14336).unwrap();
        println!(
            "  {:15} - {} experts, top-{}",
            preset.name(),
            config.num_experts,
            config.experts_per_tok
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("Complete Modern LLM Configuration Example Finished!");
    println!("{}", "=".repeat(60));
}
