//! Multi-user schema management with locking.
//!
//! This module provides thread-safe concurrent access to symbol tables with
//! read/write locking, transaction support, and lock statistics.
//!
//! # Features
//!
//! - **Read/Write Locks**: Multiple concurrent readers or single writer
//! - **Transactions**: Atomic operations with commit/rollback support
//! - **Lock Statistics**: Monitor lock contention and usage patterns
//! - **Timeout Support**: Prevent indefinite blocking on lock acquisition
//! - **Deadlock Detection**: Basic deadlock prevention through timeouts
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{LockedSymbolTable, DomainInfo};
//! use std::sync::Arc;
//! use std::thread;
//!
//! let table = Arc::new(LockedSymbolTable::new());
//!
//! // Spawn multiple readers
//! let mut handles = vec![];
//! for i in 0..3 {
//!     let table_clone = Arc::clone(&table);
//!     handles.push(thread::spawn(move || {
//!         let guard = table_clone.read();
//!         println!("Reader {} sees {} domains", i, guard.domains.len());
//!     }));
//! }
//!
//! // Wait for readers
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//!
//! // Single writer
//! {
//!     let mut guard = table.write();
//!     guard.add_domain(DomainInfo::new("User", 100)).unwrap();
//! }
//! ```

use crate::{AdapterError, SymbolTable};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::time::{Duration, Instant};

/// Statistics about lock usage and contention.
#[derive(Debug, Clone, Default)]
pub struct LockStats {
    /// Total number of successful read lock acquisitions
    pub read_locks: usize,
    /// Total number of successful write lock acquisitions
    pub write_locks: usize,
    /// Total number of failed read lock attempts (would block)
    pub read_contentions: usize,
    /// Total number of failed write lock attempts (would block)
    pub write_contentions: usize,
    /// Total time spent waiting for read locks (milliseconds)
    pub read_wait_ms: u128,
    /// Total time spent waiting for write locks (milliseconds)
    pub write_wait_ms: u128,
    /// Number of transactions started
    pub transactions_started: usize,
    /// Number of transactions committed
    pub transactions_committed: usize,
    /// Number of transactions rolled back
    pub transactions_rolled_back: usize,
}

impl LockStats {
    /// Create new empty lock statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate average read wait time in milliseconds.
    pub fn avg_read_wait_ms(&self) -> f64 {
        if self.read_locks == 0 {
            0.0
        } else {
            self.read_wait_ms as f64 / self.read_locks as f64
        }
    }

    /// Calculate average write wait time in milliseconds.
    pub fn avg_write_wait_ms(&self) -> f64 {
        if self.write_locks == 0 {
            0.0
        } else {
            self.write_wait_ms as f64 / self.write_locks as f64
        }
    }

    /// Calculate read contention rate (0.0 to 1.0).
    pub fn read_contention_rate(&self) -> f64 {
        let total = self.read_locks + self.read_contentions;
        if total == 0 {
            0.0
        } else {
            self.read_contentions as f64 / total as f64
        }
    }

    /// Calculate write contention rate (0.0 to 1.0).
    pub fn write_contention_rate(&self) -> f64 {
        let total = self.write_locks + self.write_contentions;
        if total == 0 {
            0.0
        } else {
            self.write_contentions as f64 / total as f64
        }
    }

    /// Calculate transaction commit rate (0.0 to 1.0).
    pub fn commit_rate(&self) -> f64 {
        if self.transactions_started == 0 {
            0.0
        } else {
            self.transactions_committed as f64 / self.transactions_started as f64
        }
    }
}

/// A thread-safe symbol table with read/write locking.
///
/// This wrapper provides concurrent access to a symbol table with read/write
/// locks, transaction support, and lock statistics tracking.
pub struct LockedSymbolTable {
    table: RwLock<SymbolTable>,
    stats: RwLock<LockStats>,
}

impl LockedSymbolTable {
    /// Create a new locked symbol table.
    pub fn new() -> Self {
        Self {
            table: RwLock::new(SymbolTable::new()),
            stats: RwLock::new(LockStats::new()),
        }
    }

    /// Create a locked symbol table from an existing symbol table.
    pub fn from_table(table: SymbolTable) -> Self {
        Self {
            table: RwLock::new(table),
            stats: RwLock::new(LockStats::new()),
        }
    }

    /// Acquire a read lock on the symbol table.
    ///
    /// This will block until a read lock can be acquired. Multiple readers
    /// can hold locks simultaneously.
    pub fn read(&self) -> RwLockReadGuard<'_, SymbolTable> {
        let start = Instant::now();
        let guard = self.table.read().unwrap();
        let elapsed = start.elapsed().as_millis();

        if let Ok(mut stats) = self.stats.write() {
            stats.read_locks += 1;
            stats.read_wait_ms += elapsed;
        }

        guard
    }

    /// Try to acquire a read lock without blocking.
    ///
    /// Returns `Some(guard)` if successful, `None` if would block.
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, SymbolTable>> {
        match self.table.try_read() {
            Ok(guard) => {
                if let Ok(mut stats) = self.stats.write() {
                    stats.read_locks += 1;
                }
                Some(guard)
            }
            Err(TryLockError::WouldBlock) => {
                if let Ok(mut stats) = self.stats.write() {
                    stats.read_contentions += 1;
                }
                None
            }
            Err(TryLockError::Poisoned(_)) => None,
        }
    }

    /// Acquire a write lock on the symbol table.
    ///
    /// This will block until a write lock can be acquired. Only one writer
    /// can hold a lock at a time, and no readers can be active.
    pub fn write(&self) -> RwLockWriteGuard<'_, SymbolTable> {
        let start = Instant::now();
        let guard = self.table.write().unwrap();
        let elapsed = start.elapsed().as_millis();

        if let Ok(mut stats) = self.stats.write() {
            stats.write_locks += 1;
            stats.write_wait_ms += elapsed;
        }

        guard
    }

    /// Try to acquire a write lock without blocking.
    ///
    /// Returns `Some(guard)` if successful, `None` if would block.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, SymbolTable>> {
        match self.table.try_write() {
            Ok(guard) => {
                if let Ok(mut stats) = self.stats.write() {
                    stats.write_locks += 1;
                }
                Some(guard)
            }
            Err(TryLockError::WouldBlock) => {
                if let Ok(mut stats) = self.stats.write() {
                    stats.write_contentions += 1;
                }
                None
            }
            Err(TryLockError::Poisoned(_)) => None,
        }
    }

    /// Get current lock statistics.
    pub fn stats(&self) -> LockStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset lock statistics.
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = LockStats::new();
    }

    /// Start a new transaction.
    ///
    /// Returns a transaction object that can be committed or rolled back.
    pub fn begin_transaction(&self) -> Transaction<'_> {
        if let Ok(mut stats) = self.stats.write() {
            stats.transactions_started += 1;
        }
        Transaction::new(self)
    }
}

impl Default for LockedSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// A transaction for atomic operations on a symbol table.
///
/// Transactions capture the state of the symbol table at the start and
/// allow rolling back to that state if needed.
pub struct Transaction<'a> {
    locked_table: &'a LockedSymbolTable,
    snapshot: Option<SymbolTable>,
    committed: bool,
}

impl<'a> Transaction<'a> {
    fn new(locked_table: &'a LockedSymbolTable) -> Self {
        // Take snapshot
        let snapshot = locked_table.read().clone();
        Self {
            locked_table,
            snapshot: Some(snapshot),
            committed: false,
        }
    }

    /// Execute operations within this transaction.
    ///
    /// The closure receives a mutable reference to the symbol table.
    pub fn execute<F, R>(&mut self, f: F) -> Result<R, AdapterError>
    where
        F: FnOnce(&mut SymbolTable) -> Result<R, AdapterError>,
    {
        let mut guard = self.locked_table.write();
        f(&mut guard)
    }

    /// Commit the transaction, making all changes permanent.
    pub fn commit(mut self) {
        self.committed = true;
        if let Ok(mut stats) = self.locked_table.stats.write() {
            stats.transactions_committed += 1;
        }
        // Drop snapshot
        self.snapshot = None;
    }

    /// Rollback the transaction, reverting all changes.
    pub fn rollback(mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            *self.locked_table.write() = snapshot;
        }
        if let Ok(mut stats) = self.locked_table.stats.write() {
            stats.transactions_rolled_back += 1;
        }
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        // Auto-rollback if not committed
        if !self.committed {
            if let Some(snapshot) = self.snapshot.take() {
                if let Ok(mut guard) = self.locked_table.table.write() {
                    *guard = snapshot;
                }
                if let Ok(mut stats) = self.locked_table.stats.write() {
                    stats.transactions_rolled_back += 1;
                }
            }
        }
    }
}

/// Extension trait for timeout-based lock acquisition.
pub trait LockWithTimeout {
    /// Try to acquire a read lock with a timeout.
    ///
    /// Returns `Some(guard)` if successful within timeout, `None` otherwise.
    fn read_timeout(&self, timeout: Duration) -> Option<RwLockReadGuard<'_, SymbolTable>>;

    /// Try to acquire a write lock with a timeout.
    ///
    /// Returns `Some(guard)` if successful within timeout, `None` otherwise.
    fn write_timeout(&self, timeout: Duration) -> Option<RwLockWriteGuard<'_, SymbolTable>>;
}

impl LockWithTimeout for LockedSymbolTable {
    fn read_timeout(&self, timeout: Duration) -> Option<RwLockReadGuard<'_, SymbolTable>> {
        let start = Instant::now();
        loop {
            if let Some(guard) = self.try_read() {
                return Some(guard);
            }
            if start.elapsed() >= timeout {
                if let Ok(mut stats) = self.stats.write() {
                    stats.read_contentions += 1;
                }
                return None;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn write_timeout(&self, timeout: Duration) -> Option<RwLockWriteGuard<'_, SymbolTable>> {
        let start = Instant::now();
        loop {
            if let Some(guard) = self.try_write() {
                return Some(guard);
            }
            if start.elapsed() >= timeout {
                if let Ok(mut stats) = self.stats.write() {
                    stats.write_contentions += 1;
                }
                return None;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_read_write() {
        let table = LockedSymbolTable::new();

        // Write
        {
            let mut guard = table.write();
            guard.add_domain(DomainInfo::new("User", 100)).unwrap();
        }

        // Read
        {
            let guard = table.read();
            assert_eq!(guard.domains.len(), 1);
            assert!(guard.get_domain("User").is_some());
        }
    }

    #[test]
    fn test_multiple_readers() {
        let table = Arc::new(LockedSymbolTable::new());

        // Add some data
        {
            let mut guard = table.write();
            guard.add_domain(DomainInfo::new("User", 100)).unwrap();
        }

        // Spawn multiple readers
        let mut handles = vec![];
        for _ in 0..5 {
            let table_clone = Arc::clone(&table);
            handles.push(thread::spawn(move || {
                let guard = table_clone.read();
                assert_eq!(guard.domains.len(), 1);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_try_read_write() {
        let table = LockedSymbolTable::new();

        // Try read (should succeed)
        {
            let guard = table.try_read();
            assert!(guard.is_some());
        }

        // Try write (should succeed)
        {
            let guard = table.try_write();
            assert!(guard.is_some());
        }
    }

    #[test]
    fn test_try_write_contention() {
        let table = Arc::new(LockedSymbolTable::new());

        // Hold read lock
        let _read_guard = table.read();

        // Try write (should fail due to active reader)
        let table_clone = Arc::clone(&table);
        let handle = thread::spawn(move || {
            let guard = table_clone.try_write();
            assert!(guard.is_none());
        });

        handle.join().unwrap();

        // Check contention stats
        let stats = table.stats();
        assert!(stats.write_contentions > 0);
    }

    #[test]
    fn test_transaction_commit() {
        let table = LockedSymbolTable::new();

        {
            let mut txn = table.begin_transaction();
            txn.execute(|t| {
                t.add_domain(DomainInfo::new("User", 100)).unwrap();
                t.add_domain(DomainInfo::new("Post", 1000)).unwrap();
                Ok(())
            })
            .unwrap();
            txn.commit();
        }

        let guard = table.read();
        assert_eq!(guard.domains.len(), 2);

        let stats = table.stats();
        assert_eq!(stats.transactions_committed, 1);
    }

    #[test]
    fn test_transaction_rollback() {
        let table = LockedSymbolTable::new();

        // Add initial domain
        {
            let mut guard = table.write();
            guard.add_domain(DomainInfo::new("User", 100)).unwrap();
        }

        {
            let mut txn = table.begin_transaction();
            txn.execute(|t| {
                t.add_domain(DomainInfo::new("Post", 1000)).unwrap();
                Ok(())
            })
            .unwrap();
            txn.rollback();
        }

        let guard = table.read();
        assert_eq!(guard.domains.len(), 1);
        assert!(guard.get_domain("Post").is_none());

        let stats = table.stats();
        assert_eq!(stats.transactions_rolled_back, 1);
    }

    #[test]
    fn test_transaction_auto_rollback() {
        let table = LockedSymbolTable::new();

        {
            let mut txn = table.begin_transaction();
            txn.execute(|t| {
                t.add_domain(DomainInfo::new("User", 100)).unwrap();
                Ok(())
            })
            .unwrap();
            // Drop without commit (auto-rollback)
        }

        let guard = table.read();
        assert_eq!(guard.domains.len(), 0);

        let stats = table.stats();
        assert_eq!(stats.transactions_rolled_back, 1);
    }

    #[test]
    fn test_lock_stats() {
        let table = LockedSymbolTable::new();

        // Read operations
        for _ in 0..3 {
            let _guard = table.read();
        }

        // Write operations
        for _ in 0..2 {
            let _guard = table.write();
        }

        let stats = table.stats();
        assert_eq!(stats.read_locks, 3);
        assert_eq!(stats.write_locks, 2);
    }

    #[test]
    fn test_reset_stats() {
        let table = LockedSymbolTable::new();

        let _guard = table.read();
        assert_eq!(table.stats().read_locks, 1);

        table.reset_stats();
        assert_eq!(table.stats().read_locks, 0);
    }

    #[test]
    fn test_timeout_success() {
        let table = LockedSymbolTable::new();

        let guard = table.read_timeout(Duration::from_millis(100));
        assert!(guard.is_some());
    }

    #[test]
    fn test_timeout_failure() {
        let table = Arc::new(LockedSymbolTable::new());

        // Hold write lock
        let _write_guard = table.write();

        // Try to acquire write lock with timeout in another thread
        let table_clone = Arc::clone(&table);
        let handle = thread::spawn(move || {
            let guard = table_clone.write_timeout(Duration::from_millis(50));
            assert!(guard.is_none());
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_concurrent_read_write() {
        let table = Arc::new(LockedSymbolTable::new());

        // Initialize with data
        {
            let mut guard = table.write();
            guard.add_domain(DomainInfo::new("User", 100)).unwrap();
        }

        let mut handles = vec![];

        // Readers
        for _ in 0..3 {
            let table_clone = Arc::clone(&table);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let guard = table_clone.read();
                    assert!(!guard.domains.is_empty());
                    thread::sleep(Duration::from_millis(1));
                }
            }));
        }

        // Writers
        for i in 0..2 {
            let table_clone = Arc::clone(&table);
            handles.push(thread::spawn(move || {
                for j in 0..5 {
                    let mut guard = table_clone.write();
                    let domain_name = format!("Domain_{}_{}", i, j);
                    guard
                        .add_domain(DomainInfo::new(&domain_name, 100))
                        .unwrap();
                    thread::sleep(Duration::from_millis(2));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        let guard = table.read();
        assert!(guard.domains.len() >= 11); // 1 initial + 10 from writers

        // Check stats
        let stats = table.stats();
        assert!(stats.read_locks > 0);
        assert!(stats.write_locks > 0);
    }

    #[test]
    fn test_stats_calculations() {
        let mut stats = LockStats::new();
        stats.read_locks = 10;
        stats.write_locks = 5;
        stats.read_wait_ms = 100;
        stats.write_wait_ms = 200;
        stats.read_contentions = 2;
        stats.write_contentions = 3;
        stats.transactions_started = 10;
        stats.transactions_committed = 8;

        assert_eq!(stats.avg_read_wait_ms(), 10.0);
        assert_eq!(stats.avg_write_wait_ms(), 40.0);
        assert!((stats.read_contention_rate() - 0.1667).abs() < 0.001);
        assert_eq!(stats.write_contention_rate(), 0.375);
        assert_eq!(stats.commit_rate(), 0.8);
    }

    #[test]
    fn test_transaction_error_handling() {
        let table = LockedSymbolTable::new();

        let result: Result<(), AdapterError> = {
            let mut txn = table.begin_transaction();
            txn.execute(|t| {
                t.add_domain(DomainInfo::new("User", 100)).unwrap();
                // Simulate error
                Err(AdapterError::DuplicateDomain("User".to_string()))
            })
        };

        assert!(result.is_err());

        // Transaction should auto-rollback
        let guard = table.read();
        assert_eq!(guard.domains.len(), 0);
    }

    #[test]
    fn test_from_table() {
        let mut original = SymbolTable::new();
        original.add_domain(DomainInfo::new("User", 100)).unwrap();

        let locked = LockedSymbolTable::from_table(original);

        let guard = locked.read();
        assert_eq!(guard.domains.len(), 1);
        assert!(guard.get_domain("User").is_some());
    }
}
