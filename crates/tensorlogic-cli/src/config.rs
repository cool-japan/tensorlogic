//! Configuration file support for TensorLogic CLI
//!
//! Supports loading configuration from .tensorlogicrc files in:
//! - Current directory
//! - User home directory
//! - Custom path via environment variable

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::macros::MacroDef;

/// Configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Default compilation strategy
    pub strategy: String,

    /// Default domains (name -> size)
    pub domains: HashMap<String, usize>,

    /// Default output format
    pub output_format: String,

    /// Enable validation by default
    pub validate: bool,

    /// Enable debug output by default
    pub debug: bool,

    /// Enable colored output
    pub colored: bool,

    /// REPL settings
    pub repl: ReplConfig,

    /// Watch settings
    pub watch: WatchConfig,

    /// Cache settings
    pub cache: CacheConfig,

    /// Macro definitions
    pub macros: Vec<MacroDef>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Enable compilation caching
    pub enabled: bool,

    /// Maximum number of cached entries (in-memory REPL cache)
    pub max_entries: usize,

    /// Enable persistent disk cache
    pub disk_cache_enabled: bool,

    /// Maximum disk cache size in MB
    pub disk_cache_max_size_mb: usize,

    /// Custom disk cache directory (None = use default)
    pub disk_cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 100,
            disk_cache_enabled: true,
            disk_cache_max_size_mb: 500,
            disk_cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplConfig {
    /// REPL prompt string
    pub prompt: String,

    /// History file path (relative to home)
    pub history_file: String,

    /// Maximum history entries
    pub max_history: usize,

    /// Auto-save history
    pub auto_save: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WatchConfig {
    /// Debounce delay in milliseconds
    pub debounce_ms: u64,

    /// Clear screen on reload
    pub clear_screen: bool,

    /// Show timestamps
    pub show_timestamps: bool,
}

impl Default for Config {
    fn default() -> Self {
        let mut domains = HashMap::new();
        domains.insert("D".to_string(), 100);

        Self {
            strategy: "soft_differentiable".to_string(),
            domains,
            output_format: "graph".to_string(),
            validate: false,
            debug: false,
            colored: true,
            repl: ReplConfig::default(),
            watch: WatchConfig::default(),
            cache: CacheConfig::default(),
            macros: Vec::new(),
        }
    }
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "tensorlogic> ".to_string(),
            history_file: ".tensorlogic_history".to_string(),
            max_history: 1000,
            auto_save: true,
        }
    }
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 500,
            clear_screen: true,
            show_timestamps: true,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("Failed to serialize configuration")?;

        fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))
    }

    /// Find and load configuration file
    ///
    /// Search order:
    /// 1. TENSORLOGIC_CONFIG environment variable
    /// 2. .tensorlogicrc in current directory
    /// 3. .tensorlogicrc in user home directory
    pub fn load_default() -> Self {
        // Try environment variable
        if let Ok(path) = std::env::var("TENSORLOGIC_CONFIG") {
            if let Ok(config) = Self::load(Path::new(&path)) {
                return config;
            }
        }

        // Try current directory
        let current_config = PathBuf::from(".tensorlogicrc");
        if current_config.exists() {
            if let Ok(config) = Self::load(&current_config) {
                return config;
            }
        }

        // Try home directory
        if let Some(home) = dirs::home_dir() {
            let home_config = home.join(".tensorlogicrc");
            if home_config.exists() {
                if let Ok(config) = Self::load(&home_config) {
                    return config;
                }
            }
        }

        // Return default if no config found
        Self::default()
    }

    /// Get configuration file path (current or home)
    pub fn config_path() -> PathBuf {
        // Check current directory first
        let current = PathBuf::from(".tensorlogicrc");
        if current.exists() {
            return current;
        }

        // Default to home directory
        if let Some(home) = dirs::home_dir() {
            home.join(".tensorlogicrc")
        } else {
            current
        }
    }

    /// Create a default configuration file
    pub fn create_default() -> Result<PathBuf> {
        let config = Self::default();
        let path = Self::config_path();
        config.save(&path)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.strategy, "soft_differentiable");
        assert!(config.domains.contains_key("D"));
        assert_eq!(config.output_format, "graph");
    }

    #[test]
    fn test_serialize_deserialize() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.strategy, deserialized.strategy);
    }
}
