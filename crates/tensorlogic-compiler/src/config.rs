//! Compilation configuration and strategy selection.
//!
//! This module provides configurable compilation strategies for mapping
//! logical operations to tensor operations. Different strategies optimize
//! for different objectives (accuracy, efficiency, differentiability).

use serde::{Deserialize, Serialize};

/// Strategy for compiling AND operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndStrategy {
    /// Hadamard product (element-wise multiplication): `a * b`
    /// - Differentiable, efficient
    /// - Soft semantics (values in \[0,1\])
    Product,

    /// Minimum: `min(a, b)`
    /// - Hard semantics (preserves Boolean values)
    /// - Not differentiable at equality points
    Min,

    /// Probabilistic product: `a + b - a*b`
    /// - Alternative soft semantics
    /// - Differentiable
    ProbabilisticSum,

    /// Gödel t-norm: `min(a, b)`
    /// - Same as Min but explicit fuzzy logic semantics
    Godel,

    /// Product t-norm: `a * b`
    /// - Same as Product but explicit fuzzy logic semantics
    ProductTNorm,

    /// Łukasiewicz t-norm: `max(0, a + b - 1)`
    /// - Strict t-norm
    /// - Differentiable
    Lukasiewicz,
}

/// Strategy for compiling OR operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrStrategy {
    /// Maximum: `max(a, b)`
    /// - Hard semantics
    /// - Not differentiable at equality points
    Max,

    /// Probabilistic sum: `a + b - a*b`
    /// - Soft semantics
    /// - Differentiable
    ProbabilisticSum,

    /// Gödel s-norm: `max(a, b)`
    /// - Same as Max but explicit fuzzy logic semantics
    Godel,

    /// Probabilistic s-norm: `a + b - a*b`
    /// - Same as ProbabilisticSum but explicit fuzzy logic semantics
    ProbabilisticSNorm,

    /// Łukasiewicz s-norm: `min(1, a + b)`
    /// - Dual of Łukasiewicz t-norm
    /// - Differentiable
    Lukasiewicz,
}

/// Strategy for compiling NOT operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotStrategy {
    /// Complement: `1 - a`
    /// - Standard negation
    /// - Differentiable
    Complement,

    /// Temperature-controlled sigmoid: `1 / (1 + exp(T * a))`
    /// - Smoother gradients
    /// - Configurable sharpness via temperature
    Sigmoid {
        /// Temperature parameter (higher = sharper)
        temperature: u8,
    },
}

/// Strategy for compiling existential quantifiers (∃).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExistsStrategy {
    /// Sum reduction: `sum(P, axis)`
    /// - Soft semantics (counts satisfying instances)
    /// - Differentiable
    Sum,

    /// Max reduction: `max(P, axis)`
    /// - Hard semantics (true if any instance satisfies)
    /// - Not differentiable at unique maximum
    Max,

    /// LogSumExp: `log(sum(exp(P), axis))`
    /// - Smooth approximation to max
    /// - Differentiable
    LogSumExp,

    /// Mean reduction: `mean(P, axis)`
    /// - Normalized soft semantics
    /// - Differentiable
    Mean,
}

/// Strategy for compiling universal quantifiers (∀).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForallStrategy {
    /// Dual of exists via double negation: `NOT(EXISTS(NOT(P)))`
    /// - Inherits properties from NOT and EXISTS strategies
    DualOfExists,

    /// Product reduction: `product(P, axis)`
    /// - Direct product semantics
    /// - Differentiable
    Product,

    /// Min reduction: `min(P, axis)`
    /// - Hard semantics (true if all instances satisfy)
    /// - Not differentiable at unique minimum
    Min,

    /// Mean reduction with threshold: `mean(P, axis) >= threshold`
    /// - Soft semantics with configurable strictness
    MeanThreshold {
        /// Threshold for satisfaction (typically 0.9-1.0)
        threshold_times_100: u8,
    },
}

/// Strategy for compiling implication (→).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplicationStrategy {
    /// ReLU-based: `ReLU(b - a)`
    /// - Differentiable
    /// - Soft semantics
    ReLU,

    /// Material implication: `NOT(a) OR b`
    /// - Classical logic semantics
    /// - Inherits properties from NOT and OR strategies
    Material,

    /// Gödel implication: `if a <= b then 1 else b`
    /// - Fuzzy logic semantics
    /// - Not differentiable
    Godel,

    /// Łukasiewicz implication: `min(1, 1 - a + b)`
    /// - T-norm based
    /// - Differentiable
    Lukasiewicz,

    /// Reichenbach implication: `1 - a + a*b`
    /// - Probabilistic interpretation
    /// - Differentiable
    Reichenbach,
}

/// Strategy for compiling modal logic operators (Box □, Diamond ◇).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModalStrategy {
    /// All worlds must satisfy (min reduction over worlds)
    /// - Box: min_w P(w)
    /// - Diamond: max_w P(w)
    AllWorldsMin,

    /// Product over all worlds
    /// - Box: ∏_w P(w)
    /// - Diamond: 1 - ∏_w (1 - P(w))
    AllWorldsProduct,

    /// Threshold-based satisfaction
    /// - Box: ∀w. P(w) > threshold
    /// - Diamond: ∃w. P(w) > threshold
    Threshold {
        /// Threshold value (0.0 to 1.0)
        threshold: f64,
    },
}

impl Eq for ModalStrategy {}

/// Strategy for compiling temporal logic operators (Next X, Eventually F, Always G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalStrategy {
    /// Max over time (for Eventually, min for Always)
    Max,

    /// Sum over time (probabilistic interpretation)
    Sum,

    /// LogSumExp (smooth max approximation)
    LogSumExp,
}

/// Complete compilation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilationConfig {
    /// Strategy for AND operations
    pub and_strategy: AndStrategy,
    /// Strategy for OR operations
    pub or_strategy: OrStrategy,
    /// Strategy for NOT operations
    pub not_strategy: NotStrategy,
    /// Strategy for existential quantifiers
    pub exists_strategy: ExistsStrategy,
    /// Strategy for universal quantifiers
    pub forall_strategy: ForallStrategy,
    /// Strategy for implication
    pub implication_strategy: ImplicationStrategy,
    /// Strategy for modal logic operators
    pub modal_strategy: ModalStrategy,
    /// Strategy for temporal logic operators
    pub temporal_strategy: TemporalStrategy,
    /// Number of possible worlds for modal logic (default: 10)
    pub modal_world_size: Option<usize>,
    /// Number of time steps for temporal logic (default: 100)
    pub temporal_time_steps: Option<usize>,
}

impl Eq for CompilationConfig {}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self::soft_differentiable()
    }
}

impl CompilationConfig {
    /// Soft, differentiable configuration (default).
    ///
    /// Optimized for neural network training and gradient-based optimization.
    /// All operations are differentiable with smooth gradients.
    pub fn soft_differentiable() -> Self {
        Self {
            and_strategy: AndStrategy::Product,
            or_strategy: OrStrategy::ProbabilisticSum,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::Sum,
            forall_strategy: ForallStrategy::DualOfExists,
            implication_strategy: ImplicationStrategy::ReLU,
            modal_strategy: ModalStrategy::AllWorldsProduct,
            temporal_strategy: TemporalStrategy::Sum,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Hard, Boolean-like configuration.
    ///
    /// Optimized for discrete reasoning with Boolean-like values.
    /// Uses min/max operations for crisp logic semantics.
    pub fn hard_boolean() -> Self {
        Self {
            and_strategy: AndStrategy::Min,
            or_strategy: OrStrategy::Max,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::Max,
            forall_strategy: ForallStrategy::Min,
            implication_strategy: ImplicationStrategy::Material,
            modal_strategy: ModalStrategy::AllWorldsMin,
            temporal_strategy: TemporalStrategy::Max,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Fuzzy logic configuration (Gödel semantics).
    ///
    /// Standard fuzzy logic with Gödel t-norms and s-norms.
    pub fn fuzzy_godel() -> Self {
        Self {
            and_strategy: AndStrategy::Godel,
            or_strategy: OrStrategy::Godel,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::Max,
            forall_strategy: ForallStrategy::Min,
            implication_strategy: ImplicationStrategy::Godel,
            modal_strategy: ModalStrategy::AllWorldsMin,
            temporal_strategy: TemporalStrategy::Max,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Fuzzy logic configuration (Product semantics).
    ///
    /// Fuzzy logic with product t-norms.
    pub fn fuzzy_product() -> Self {
        Self {
            and_strategy: AndStrategy::ProductTNorm,
            or_strategy: OrStrategy::ProbabilisticSNorm,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::Mean,
            forall_strategy: ForallStrategy::Product,
            implication_strategy: ImplicationStrategy::Reichenbach,
            modal_strategy: ModalStrategy::AllWorldsProduct,
            temporal_strategy: TemporalStrategy::Sum,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Fuzzy logic configuration (Łukasiewicz semantics).
    ///
    /// Fuzzy logic with Łukasiewicz t-norms, fully differentiable.
    pub fn fuzzy_lukasiewicz() -> Self {
        Self {
            and_strategy: AndStrategy::Lukasiewicz,
            or_strategy: OrStrategy::Lukasiewicz,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::LogSumExp,
            forall_strategy: ForallStrategy::DualOfExists,
            implication_strategy: ImplicationStrategy::Lukasiewicz,
            modal_strategy: ModalStrategy::Threshold { threshold: 0.5 },
            temporal_strategy: TemporalStrategy::LogSumExp,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Probabilistic logic configuration.
    ///
    /// Interprets logical operations as probabilistic events.
    pub fn probabilistic() -> Self {
        Self {
            and_strategy: AndStrategy::ProbabilisticSum,
            or_strategy: OrStrategy::ProbabilisticSum,
            not_strategy: NotStrategy::Complement,
            exists_strategy: ExistsStrategy::Mean,
            forall_strategy: ForallStrategy::Product,
            implication_strategy: ImplicationStrategy::Reichenbach,
            modal_strategy: ModalStrategy::AllWorldsProduct,
            temporal_strategy: TemporalStrategy::Sum,
            modal_world_size: Some(10),
            temporal_time_steps: Some(100),
        }
    }

    /// Create a custom configuration.
    pub fn custom() -> CompilationConfigBuilder {
        CompilationConfigBuilder::default()
    }
}

/// Builder for custom compilation configurations.
#[derive(Debug, Clone, Default)]
pub struct CompilationConfigBuilder {
    and_strategy: Option<AndStrategy>,
    or_strategy: Option<OrStrategy>,
    not_strategy: Option<NotStrategy>,
    exists_strategy: Option<ExistsStrategy>,
    forall_strategy: Option<ForallStrategy>,
    implication_strategy: Option<ImplicationStrategy>,
    modal_strategy: Option<ModalStrategy>,
    temporal_strategy: Option<TemporalStrategy>,
    modal_world_size: Option<usize>,
    temporal_time_steps: Option<usize>,
}

impl CompilationConfigBuilder {
    /// Set AND strategy.
    pub fn and_strategy(mut self, strategy: AndStrategy) -> Self {
        self.and_strategy = Some(strategy);
        self
    }

    /// Set OR strategy.
    pub fn or_strategy(mut self, strategy: OrStrategy) -> Self {
        self.or_strategy = Some(strategy);
        self
    }

    /// Set NOT strategy.
    pub fn not_strategy(mut self, strategy: NotStrategy) -> Self {
        self.not_strategy = Some(strategy);
        self
    }

    /// Set EXISTS strategy.
    pub fn exists_strategy(mut self, strategy: ExistsStrategy) -> Self {
        self.exists_strategy = Some(strategy);
        self
    }

    /// Set FORALL strategy.
    pub fn forall_strategy(mut self, strategy: ForallStrategy) -> Self {
        self.forall_strategy = Some(strategy);
        self
    }

    /// Set implication strategy.
    pub fn implication_strategy(mut self, strategy: ImplicationStrategy) -> Self {
        self.implication_strategy = Some(strategy);
        self
    }

    /// Set modal logic strategy.
    pub fn modal_strategy(mut self, strategy: ModalStrategy) -> Self {
        self.modal_strategy = Some(strategy);
        self
    }

    /// Set temporal logic strategy.
    pub fn temporal_strategy(mut self, strategy: TemporalStrategy) -> Self {
        self.temporal_strategy = Some(strategy);
        self
    }

    /// Set number of possible worlds for modal logic.
    pub fn modal_world_size(mut self, size: usize) -> Self {
        self.modal_world_size = Some(size);
        self
    }

    /// Set number of time steps for temporal logic.
    pub fn temporal_time_steps(mut self, steps: usize) -> Self {
        self.temporal_time_steps = Some(steps);
        self
    }

    /// Build the configuration.
    ///
    /// Uses default soft_differentiable() values for any unset strategies.
    pub fn build(self) -> CompilationConfig {
        let default = CompilationConfig::soft_differentiable();
        CompilationConfig {
            and_strategy: self.and_strategy.unwrap_or(default.and_strategy),
            or_strategy: self.or_strategy.unwrap_or(default.or_strategy),
            not_strategy: self.not_strategy.unwrap_or(default.not_strategy),
            exists_strategy: self.exists_strategy.unwrap_or(default.exists_strategy),
            forall_strategy: self.forall_strategy.unwrap_or(default.forall_strategy),
            implication_strategy: self
                .implication_strategy
                .unwrap_or(default.implication_strategy),
            modal_strategy: self.modal_strategy.unwrap_or(default.modal_strategy),
            temporal_strategy: self.temporal_strategy.unwrap_or(default.temporal_strategy),
            modal_world_size: self.modal_world_size.or(default.modal_world_size),
            temporal_time_steps: self.temporal_time_steps.or(default.temporal_time_steps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompilationConfig::default();
        assert_eq!(config.and_strategy, AndStrategy::Product);
        assert_eq!(config.or_strategy, OrStrategy::ProbabilisticSum);
        assert_eq!(config.exists_strategy, ExistsStrategy::Sum);
    }

    #[test]
    fn test_hard_boolean_config() {
        let config = CompilationConfig::hard_boolean();
        assert_eq!(config.and_strategy, AndStrategy::Min);
        assert_eq!(config.or_strategy, OrStrategy::Max);
        assert_eq!(config.exists_strategy, ExistsStrategy::Max);
    }

    #[test]
    fn test_fuzzy_godel_config() {
        let config = CompilationConfig::fuzzy_godel();
        assert_eq!(config.and_strategy, AndStrategy::Godel);
        assert_eq!(config.or_strategy, OrStrategy::Godel);
        assert_eq!(config.implication_strategy, ImplicationStrategy::Godel);
    }

    #[test]
    fn test_custom_config_builder() {
        let config = CompilationConfig::custom()
            .and_strategy(AndStrategy::Min)
            .or_strategy(OrStrategy::Max)
            .build();

        assert_eq!(config.and_strategy, AndStrategy::Min);
        assert_eq!(config.or_strategy, OrStrategy::Max);
        // Should use defaults for unset strategies
        assert_eq!(config.not_strategy, NotStrategy::Complement);
    }

    #[test]
    fn test_builder_with_all_strategies() {
        let config = CompilationConfig::custom()
            .and_strategy(AndStrategy::Lukasiewicz)
            .or_strategy(OrStrategy::Lukasiewicz)
            .not_strategy(NotStrategy::Sigmoid { temperature: 10 })
            .exists_strategy(ExistsStrategy::LogSumExp)
            .forall_strategy(ForallStrategy::Min)
            .implication_strategy(ImplicationStrategy::Lukasiewicz)
            .build();

        assert_eq!(config.and_strategy, AndStrategy::Lukasiewicz);
        assert_eq!(config.or_strategy, OrStrategy::Lukasiewicz);
        assert_eq!(
            config.not_strategy,
            NotStrategy::Sigmoid { temperature: 10 }
        );
    }

    #[test]
    fn test_probabilistic_config() {
        let config = CompilationConfig::probabilistic();
        assert_eq!(config.and_strategy, AndStrategy::ProbabilisticSum);
        assert_eq!(config.or_strategy, OrStrategy::ProbabilisticSum);
        assert_eq!(
            config.implication_strategy,
            ImplicationStrategy::Reichenbach
        );
    }

    #[test]
    fn test_fuzzy_lukasiewicz_config() {
        let config = CompilationConfig::fuzzy_lukasiewicz();
        assert_eq!(config.and_strategy, AndStrategy::Lukasiewicz);
        assert_eq!(config.or_strategy, OrStrategy::Lukasiewicz);
        assert_eq!(config.exists_strategy, ExistsStrategy::LogSumExp);
    }

    #[test]
    fn test_serialization_deserialization() {
        let original = CompilationConfig::custom()
            .and_strategy(AndStrategy::Lukasiewicz)
            .or_strategy(OrStrategy::ProbabilisticSum)
            .not_strategy(NotStrategy::Sigmoid { temperature: 5 })
            .exists_strategy(ExistsStrategy::LogSumExp)
            .forall_strategy(ForallStrategy::Product)
            .implication_strategy(ImplicationStrategy::Reichenbach)
            .modal_world_size(20)
            .temporal_time_steps(50)
            .build();

        // Serialize to JSON
        let json = serde_json::to_string(&original).expect("Failed to serialize");

        // Deserialize back
        let deserialized: CompilationConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");

        // Verify equality
        assert_eq!(original, deserialized);
        assert_eq!(deserialized.and_strategy, AndStrategy::Lukasiewicz);
        assert_eq!(deserialized.or_strategy, OrStrategy::ProbabilisticSum);
        assert_eq!(
            deserialized.not_strategy,
            NotStrategy::Sigmoid { temperature: 5 }
        );
        assert_eq!(deserialized.modal_world_size, Some(20));
        assert_eq!(deserialized.temporal_time_steps, Some(50));
    }

    #[test]
    fn test_serialization_all_presets() {
        let configs = vec![
            (
                "soft_differentiable",
                CompilationConfig::soft_differentiable(),
            ),
            ("hard_boolean", CompilationConfig::hard_boolean()),
            ("fuzzy_godel", CompilationConfig::fuzzy_godel()),
            ("fuzzy_lukasiewicz", CompilationConfig::fuzzy_lukasiewicz()),
            ("probabilistic", CompilationConfig::probabilistic()),
        ];

        for (name, config) in configs {
            let json = serde_json::to_string(&config)
                .unwrap_or_else(|_| panic!("Failed to serialize {}", name));
            let deserialized: CompilationConfig = serde_json::from_str(&json)
                .unwrap_or_else(|_| panic!("Failed to deserialize {}", name));
            assert_eq!(config, deserialized, "Mismatch for preset: {}", name);
        }
    }
}
