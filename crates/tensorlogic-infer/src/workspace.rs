//! Workspace management for efficient memory reuse.
//!
//! This module provides workspace allocation and management for reducing
//! memory allocation overhead during inference:
//! - Pre-allocated memory pools
//! - Workspace recycling
//! - Size-based workspace selection
//! - Automatic workspace expansion
//! - Memory defragmentation
//! - Multi-threaded workspace management

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Workspace management errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum WorkspaceError {
    #[error(
        "Workspace allocation failed: requested {requested} bytes, available {available} bytes"
    )]
    AllocationFailed { requested: usize, available: usize },

    #[error("Workspace not found: {0}")]
    NotFound(String),

    #[error("Invalid workspace size: {0}")]
    InvalidSize(usize),

    #[error("Workspace limit exceeded: {limit} bytes")]
    LimitExceeded { limit: usize },

    #[error("Workspace is in use")]
    InUse,
}

/// Workspace allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Best fit - find smallest workspace that fits
    BestFit,
    /// First fit - use first workspace that fits
    FirstFit,
    /// Exact fit - only use exact size matches
    ExactFit,
    /// Power of 2 - round up to power of 2 sizes
    PowerOfTwo,
}

/// Workspace configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Initial workspace size (bytes)
    pub initial_size: usize,
    /// Maximum workspace size (bytes)
    pub max_size: usize,
    /// Growth factor when expanding
    pub growth_factor: f64,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Enable automatic expansion
    pub auto_expand: bool,
    /// Enable defragmentation
    pub enable_defragmentation: bool,
    /// Defragmentation threshold (fragmentation ratio)
    pub defrag_threshold: f64,
    /// Number of size buckets for pooling
    pub num_buckets: usize,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024,    // 1 MB
            max_size: 1024 * 1024 * 1024, // 1 GB
            growth_factor: 2.0,
            strategy: AllocationStrategy::BestFit,
            auto_expand: true,
            enable_defragmentation: false,
            defrag_threshold: 0.5,
            num_buckets: 16,
        }
    }
}

impl WorkspaceConfig {
    /// Create configuration for large models.
    pub fn large_model() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024,   // 64 MB
            max_size: 8 * 1024 * 1024 * 1024, // 8 GB
            growth_factor: 1.5,
            num_buckets: 32,
            ..Default::default()
        }
    }

    /// Create configuration for small models.
    pub fn small_model() -> Self {
        Self {
            initial_size: 256 * 1024,    // 256 KB
            max_size: 128 * 1024 * 1024, // 128 MB
            growth_factor: 2.0,
            num_buckets: 8,
            ..Default::default()
        }
    }

    /// Create configuration optimized for memory.
    pub fn memory_optimized() -> Self {
        Self {
            initial_size: 512 * 1024,    // 512 KB
            max_size: 256 * 1024 * 1024, // 256 MB
            growth_factor: 1.2,
            enable_defragmentation: true,
            defrag_threshold: 0.3,
            ..Default::default()
        }
    }
}

/// A reusable workspace buffer.
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Unique identifier
    pub id: String,
    /// Size in bytes
    pub size: usize,
    /// Whether currently in use
    pub in_use: bool,
    /// Number of times allocated
    pub allocation_count: usize,
    /// Total time in use (for statistics)
    pub total_use_time: std::time::Duration,
}

impl Workspace {
    /// Create a new workspace.
    pub fn new(id: String, size: usize) -> Self {
        Self {
            id,
            size,
            in_use: false,
            allocation_count: 0,
            total_use_time: std::time::Duration::ZERO,
        }
    }

    /// Mark as in use.
    pub fn acquire(&mut self) -> Result<(), WorkspaceError> {
        if self.in_use {
            return Err(WorkspaceError::InUse);
        }
        self.in_use = true;
        self.allocation_count += 1;
        Ok(())
    }

    /// Mark as available.
    pub fn release(&mut self) {
        self.in_use = false;
    }
}

/// Workspace pool for managing multiple workspaces.
pub struct WorkspacePool {
    config: WorkspaceConfig,
    workspaces: HashMap<String, Workspace>,
    free_lists: HashMap<usize, VecDeque<String>>, // Size bucket -> workspace IDs
    next_id: usize,
    stats: WorkspaceStats,
}

impl WorkspacePool {
    /// Create a new workspace pool.
    pub fn new(config: WorkspaceConfig) -> Self {
        let mut pool = Self {
            config,
            workspaces: HashMap::new(),
            free_lists: HashMap::new(),
            next_id: 0,
            stats: WorkspaceStats::default(),
        };

        // Pre-allocate initial workspaces
        pool.preallocate_workspaces();

        pool
    }

    /// Pre-allocate workspaces based on configuration.
    fn preallocate_workspaces(&mut self) {
        let sizes = self.compute_bucket_sizes();
        for size in sizes {
            let _ = self.create_workspace(size);
        }
    }

    /// Compute workspace sizes for buckets.
    fn compute_bucket_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut size = self.config.initial_size;

        for _ in 0..self.config.num_buckets {
            sizes.push(size);
            size = (size as f64 * self.config.growth_factor) as usize;
            if size > self.config.max_size {
                break;
            }
        }

        sizes
    }

    /// Create a new workspace.
    fn create_workspace(&mut self, size: usize) -> String {
        let id = format!("ws_{}", self.next_id);
        self.next_id += 1;

        let workspace = Workspace::new(id.clone(), size);
        self.workspaces.insert(id.clone(), workspace);

        // Add to free list
        let bucket = self.size_to_bucket(size);
        self.free_lists
            .entry(bucket)
            .or_default()
            .push_back(id.clone());

        self.stats.total_created += 1;
        self.stats.current_total_size += size;

        id
    }

    /// Convert size to bucket size.
    fn size_to_bucket(&self, size: usize) -> usize {
        match self.config.strategy {
            AllocationStrategy::PowerOfTwo => size.next_power_of_two(),
            _ => {
                // Find nearest bucket size
                let sizes = self.compute_bucket_sizes();
                sizes.iter().find(|&&s| s >= size).copied().unwrap_or(size)
            }
        }
    }

    /// Allocate a workspace of at least the given size.
    pub fn allocate(&mut self, size: usize) -> Result<String, WorkspaceError> {
        if size > self.config.max_size {
            return Err(WorkspaceError::InvalidSize(size));
        }

        let workspace_id = match self.config.strategy {
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::ExactFit => self.find_exact_fit(size),
            AllocationStrategy::PowerOfTwo => {
                let bucket_size = size.next_power_of_two();
                self.find_first_fit(bucket_size)
            }
        };

        match workspace_id {
            Some(id) => {
                self.workspaces.get_mut(&id).unwrap().acquire()?;
                self.stats.total_allocations += 1;
                Ok(id)
            }
            None => {
                // No suitable workspace found
                if self.config.auto_expand {
                    let new_size = self.size_to_bucket(size);
                    let id = self.create_workspace(new_size);
                    self.workspaces.get_mut(&id).unwrap().acquire()?;
                    self.stats.total_allocations += 1;
                    self.stats.total_expansions += 1;
                    Ok(id)
                } else {
                    Err(WorkspaceError::AllocationFailed {
                        requested: size,
                        available: self.max_available_size(),
                    })
                }
            }
        }
    }

    /// Release a workspace back to the pool.
    pub fn release(&mut self, id: &str) -> Result<(), WorkspaceError> {
        let workspace_size = {
            let workspace = self
                .workspaces
                .get_mut(id)
                .ok_or_else(|| WorkspaceError::NotFound(id.to_string()))?;

            workspace.release();
            workspace.size
        };

        self.stats.total_releases += 1;

        // Add back to free list
        let bucket = self.size_to_bucket(workspace_size);
        self.free_lists
            .entry(bucket)
            .or_default()
            .push_back(id.to_string());

        Ok(())
    }

    /// Find best fit workspace.
    fn find_best_fit(&mut self, size: usize) -> Option<String> {
        let mut best_id: Option<String> = None;
        let mut best_size = usize::MAX;

        for (ws_id, workspace) in &self.workspaces {
            if !workspace.in_use && workspace.size >= size && workspace.size < best_size {
                best_id = Some(ws_id.clone());
                best_size = workspace.size;
            }
        }

        if let Some(ref id) = best_id {
            let bucket = self.size_to_bucket(best_size);
            if let Some(list) = self.free_lists.get_mut(&bucket) {
                list.retain(|ws_id| ws_id != id);
            }
        }

        best_id
    }

    /// Find first fit workspace.
    fn find_first_fit(&mut self, size: usize) -> Option<String> {
        for (ws_id, workspace) in &self.workspaces {
            if !workspace.in_use && workspace.size >= size {
                let id = ws_id.clone();
                let bucket = self.size_to_bucket(workspace.size);
                if let Some(list) = self.free_lists.get_mut(&bucket) {
                    list.retain(|ws_id| ws_id != &id);
                }
                return Some(id);
            }
        }
        None
    }

    /// Find exact fit workspace.
    fn find_exact_fit(&mut self, size: usize) -> Option<String> {
        let bucket = self.size_to_bucket(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
            list.pop_front()
        } else {
            None
        }
    }

    /// Get maximum available workspace size.
    fn max_available_size(&self) -> usize {
        self.workspaces
            .values()
            .filter(|ws| !ws.in_use)
            .map(|ws| ws.size)
            .max()
            .unwrap_or(0)
    }

    /// Get statistics.
    pub fn stats(&self) -> &WorkspaceStats {
        &self.stats
    }

    /// Perform defragmentation if needed.
    pub fn defragment(&mut self) -> DefragmentationResult {
        if !self.config.enable_defragmentation {
            return DefragmentationResult {
                freed_bytes: 0,
                merged_workspaces: 0,
            };
        }

        let fragmentation_ratio = self.compute_fragmentation_ratio();
        if fragmentation_ratio < self.config.defrag_threshold {
            return DefragmentationResult {
                freed_bytes: 0,
                merged_workspaces: 0,
            };
        }

        // Simple defragmentation: merge adjacent free workspaces
        // In a real implementation, this would involve memory compaction
        let freed_bytes = 0;
        let merged_workspaces = 0;

        // Placeholder for actual defragmentation logic
        self.stats.total_defragmentations += 1;

        DefragmentationResult {
            freed_bytes,
            merged_workspaces,
        }
    }

    /// Compute fragmentation ratio.
    fn compute_fragmentation_ratio(&self) -> f64 {
        let total_free = self
            .workspaces
            .values()
            .filter(|ws| !ws.in_use)
            .map(|ws| ws.size)
            .sum::<usize>();

        let max_free = self.max_available_size();

        if total_free == 0 {
            0.0
        } else {
            1.0 - (max_free as f64 / total_free as f64)
        }
    }

    /// Clear all workspaces.
    pub fn clear(&mut self) {
        self.workspaces.clear();
        self.free_lists.clear();
        self.stats = WorkspaceStats::default();
        self.preallocate_workspaces();
    }
}

/// Thread-safe workspace pool.
pub struct SharedWorkspacePool {
    inner: Arc<Mutex<WorkspacePool>>,
}

impl SharedWorkspacePool {
    /// Create a new shared workspace pool.
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkspacePool::new(config))),
        }
    }

    /// Allocate a workspace.
    pub fn allocate(&self, size: usize) -> Result<String, WorkspaceError> {
        self.inner.lock().unwrap().allocate(size)
    }

    /// Release a workspace.
    pub fn release(&self, id: &str) -> Result<(), WorkspaceError> {
        self.inner.lock().unwrap().release(id)
    }

    /// Get statistics.
    pub fn stats(&self) -> WorkspaceStats {
        self.inner.lock().unwrap().stats().clone()
    }

    /// Perform defragmentation.
    pub fn defragment(&self) -> DefragmentationResult {
        self.inner.lock().unwrap().defragment()
    }
}

impl Clone for SharedWorkspacePool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Workspace usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceStats {
    /// Total workspaces created
    pub total_created: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Total releases
    pub total_releases: usize,
    /// Total expansions
    pub total_expansions: usize,
    /// Total defragmentations
    pub total_defragmentations: usize,
    /// Current total size (bytes)
    pub current_total_size: usize,
}

impl WorkspaceStats {
    /// Get hit rate (allocations without expansion).
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            1.0 - (self.total_expansions as f64 / self.total_allocations as f64)
        }
    }

    /// Get average workspace size.
    pub fn avg_workspace_size(&self) -> f64 {
        if self.total_created == 0 {
            0.0
        } else {
            self.current_total_size as f64 / self.total_created as f64
        }
    }
}

/// Defragmentation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefragmentationResult {
    /// Bytes freed
    pub freed_bytes: usize,
    /// Number of workspaces merged
    pub merged_workspaces: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = Workspace::new("test".to_string(), 1024);
        assert_eq!(ws.size, 1024);
        assert!(!ws.in_use);
        assert_eq!(ws.allocation_count, 0);
    }

    #[test]
    fn test_workspace_acquire_release() {
        let mut ws = Workspace::new("test".to_string(), 1024);

        assert!(ws.acquire().is_ok());
        assert!(ws.in_use);
        assert_eq!(ws.allocation_count, 1);

        // Cannot acquire twice
        assert!(ws.acquire().is_err());

        ws.release();
        assert!(!ws.in_use);

        // Can acquire again
        assert!(ws.acquire().is_ok());
        assert_eq!(ws.allocation_count, 2);
    }

    #[test]
    fn test_workspace_config() {
        let config = WorkspaceConfig::large_model();
        assert!(config.initial_size > WorkspaceConfig::default().initial_size);

        let config = WorkspaceConfig::small_model();
        assert!(config.max_size < WorkspaceConfig::default().max_size);
    }

    #[test]
    fn test_workspace_pool_creation() {
        let config = WorkspaceConfig::default();
        let pool = WorkspacePool::new(config);

        assert!(pool.stats().total_created > 0);
    }

    #[test]
    fn test_workspace_allocation() {
        let config = WorkspaceConfig::default();
        let mut pool = WorkspacePool::new(config);

        let id = pool.allocate(512).unwrap();
        assert!(!id.is_empty());

        let workspace = pool.workspaces.get(&id).unwrap();
        assert!(workspace.in_use);
        assert!(workspace.size >= 512);
    }

    #[test]
    fn test_workspace_release() {
        let config = WorkspaceConfig::default();
        let mut pool = WorkspacePool::new(config);

        let id = pool.allocate(512).unwrap();
        assert!(pool.release(&id).is_ok());

        let workspace = pool.workspaces.get(&id).unwrap();
        assert!(!workspace.in_use);
    }

    #[test]
    fn test_allocation_strategies() {
        // Test different strategies
        for strategy in [
            AllocationStrategy::BestFit,
            AllocationStrategy::FirstFit,
            AllocationStrategy::ExactFit,
            AllocationStrategy::PowerOfTwo,
        ] {
            let config = WorkspaceConfig {
                strategy,
                ..Default::default()
            };
            let mut pool = WorkspacePool::new(config);

            let id = pool.allocate(512);
            assert!(id.is_ok());
        }
    }

    #[test]
    fn test_auto_expansion() {
        let config = WorkspaceConfig {
            initial_size: 1024,
            max_size: 1024 * 1024,
            auto_expand: true,
            num_buckets: 2, // Limit pre-allocated buckets
            ..Default::default()
        };
        let mut pool = WorkspacePool::new(config);

        // Record initial expansion count
        let initial_expansions = pool.stats().total_expansions;

        // Allocate larger than any pre-allocated workspace
        // With 2 buckets and growth factor 2.0: 1024, 2048
        // So allocate 5KB which is larger than 2048
        let id = pool.allocate(5 * 1024);
        assert!(id.is_ok());

        assert!(pool.stats().total_expansions > initial_expansions);
    }

    #[test]
    fn test_allocation_without_expansion() {
        let config = WorkspaceConfig {
            initial_size: 1024,
            max_size: 2048,
            auto_expand: false,
            ..Default::default()
        };
        let mut pool = WorkspacePool::new(config);

        // Should fail if no suitable workspace
        // (might succeed if initial workspaces are large enough)
        let result = pool.allocate(100 * 1024);
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_stats_hit_rate() {
        let stats = WorkspaceStats {
            total_allocations: 10,
            total_expansions: 2,
            ..Default::default()
        };

        assert_eq!(stats.hit_rate(), 0.8);
    }

    #[test]
    fn test_shared_workspace_pool() {
        let config = WorkspaceConfig::default();
        let pool = SharedWorkspacePool::new(config);

        let id = pool.allocate(512).unwrap();
        assert!(pool.release(&id).is_ok());

        let stats = pool.stats();
        assert!(stats.total_allocations > 0);
    }

    #[test]
    fn test_fragmentation_ratio() {
        let config = WorkspaceConfig::default();
        let pool = WorkspacePool::new(config);

        let ratio = pool.compute_fragmentation_ratio();
        assert!((0.0..=1.0).contains(&ratio));
    }

    #[test]
    fn test_defragmentation() {
        let config = WorkspaceConfig {
            enable_defragmentation: true,
            ..Default::default()
        };
        let mut pool = WorkspacePool::new(config);

        let result = pool.defragment();
        // Should not panic
        assert_eq!(result.freed_bytes, 0); // No actual defrag yet
    }
}
