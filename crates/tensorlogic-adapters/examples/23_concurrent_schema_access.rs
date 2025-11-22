//! Concurrent Schema Access with Locking Example
//!
//! This example demonstrates multi-user schema management with read/write locks,
//! transactions, and concurrent access patterns.

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tensorlogic_adapters::{DomainInfo, LockWithTimeout, LockedSymbolTable};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Adapters: Concurrent Schema Access ===\n");

    // Example 1: Basic read/write locking
    example_basic_locking()?;

    // Example 2: Multiple concurrent readers
    example_concurrent_readers()?;

    // Example 3: Reader-writer pattern
    example_reader_writer_pattern()?;

    // Example 4: Transactions with commit/rollback
    example_transactions()?;

    // Example 5: Lock timeout handling
    example_lock_timeouts()?;

    // Example 6: Lock statistics monitoring
    example_lock_statistics()?;

    Ok(())
}

fn example_basic_locking() -> anyhow::Result<()> {
    println!("--- Example 1: Basic Read/Write Locking ---");

    let table = LockedSymbolTable::new();

    // Write operation
    {
        println!("Acquiring write lock...");
        let mut guard = table.write();
        guard.add_domain(DomainInfo::new("User", 1000))?;
        guard.add_domain(DomainInfo::new("Role", 10))?;
        println!("Added 2 domains");
    } // Write lock released

    // Read operation
    {
        println!("Acquiring read lock...");
        let guard = table.read();
        println!("Schema has {} domains", guard.domains.len());
        for (name, domain) in &guard.domains {
            println!("  - {}: cardinality {}", name, domain.cardinality);
        }
    } // Read lock released

    println!();
    Ok(())
}

fn example_concurrent_readers() -> anyhow::Result<()> {
    println!("--- Example 2: Multiple Concurrent Readers ---");

    let table = Arc::new(LockedSymbolTable::new());

    // Initialize schema
    {
        let mut guard = table.write();
        for i in 0..5 {
            guard.add_domain(DomainInfo::new(format!("Domain{}", i), 100))?;
        }
    }

    println!("Spawning 5 concurrent readers...");

    let mut handles = vec![];
    for reader_id in 0..5 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let guard = table_clone.read();
            println!(
                "  Reader {}: Read {} domains",
                reader_id,
                guard.domains.len()
            );
            thread::sleep(Duration::from_millis(50));
            println!("  Reader {}: Done", reader_id);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("All readers completed successfully\n");
    Ok(())
}

fn example_reader_writer_pattern() -> anyhow::Result<()> {
    println!("--- Example 3: Reader-Writer Pattern ---");

    let table = Arc::new(LockedSymbolTable::new());

    // Initialize with some data
    {
        let mut guard = table.write();
        guard.add_domain(DomainInfo::new("User", 1000))?;
    }

    let mut handles = vec![];

    // Spawn readers
    for reader_id in 0..3 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            for iteration in 0..3 {
                let guard = table_clone.read();
                println!(
                    "  Reader {} (iteration {}): {} domains",
                    reader_id,
                    iteration,
                    guard.domains.len()
                );
                thread::sleep(Duration::from_millis(20));
            }
        }));
    }

    // Spawn writers
    for writer_id in 0..2 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            for iteration in 0..2 {
                let mut guard = table_clone.write();
                let domain_name = format!("WriterDomain_{}_{}", writer_id, iteration);
                guard
                    .add_domain(DomainInfo::new(&domain_name, 100))
                    .unwrap();
                println!(
                    "  Writer {} (iteration {}): Added {}",
                    writer_id, iteration, domain_name
                );
                thread::sleep(Duration::from_millis(30));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Final count
    {
        let guard = table.read();
        println!("Final schema has {} domains", guard.domains.len());
    }

    println!();
    Ok(())
}

fn example_transactions() -> anyhow::Result<()> {
    println!("--- Example 4: Transactions with Commit/Rollback ---");

    let table = LockedSymbolTable::new();

    // Successful transaction (commit)
    println!("Transaction 1: Adding domains with commit");
    {
        let mut txn = table.begin_transaction();
        txn.execute(|t| {
            t.add_domain(DomainInfo::new("User", 1000)).unwrap();
            t.add_domain(DomainInfo::new("Post", 5000)).unwrap();
            Ok(())
        })?;
        txn.commit();
        println!("  ✓ Transaction committed");
    }

    {
        let guard = table.read();
        println!("  Schema has {} domains", guard.domains.len());
    }

    // Failed transaction (rollback)
    println!("\nTransaction 2: Attempting to add domains then rollback");
    {
        let mut txn = table.begin_transaction();
        txn.execute(|t| {
            t.add_domain(DomainInfo::new("Comment", 10000)).unwrap();
            t.add_domain(DomainInfo::new("Tag", 500)).unwrap();
            Ok(())
        })?;
        println!("  Added 2 domains in transaction");
        txn.rollback();
        println!("  ✗ Transaction rolled back");
    }

    {
        let guard = table.read();
        println!("  Schema still has {} domains", guard.domains.len());
        assert!(!guard.domains.contains_key("Comment"));
    }

    // Auto-rollback (transaction dropped without commit)
    println!("\nTransaction 3: Auto-rollback on drop");
    {
        let mut txn = table.begin_transaction();
        txn.execute(|t| {
            t.add_domain(DomainInfo::new("AutoRollback", 100)).unwrap();
            Ok(())
        })?;
        println!("  Added domain but not committing");
        // txn dropped here, auto-rollback
    }

    {
        let guard = table.read();
        println!("  Schema still has {} domains", guard.domains.len());
        assert!(!guard.domains.contains_key("AutoRollback"));
    }

    println!();
    Ok(())
}

fn example_lock_timeouts() -> anyhow::Result<()> {
    println!("--- Example 5: Lock Timeout Handling ---");

    let table = Arc::new(LockedSymbolTable::new());

    println!("Thread 1: Acquiring write lock...");
    let table_clone = Arc::clone(&table);
    let handle = thread::spawn(move || {
        let _write_guard = table_clone.write();
        println!("  Thread 1: Got write lock, holding for 100ms");
        thread::sleep(Duration::from_millis(100));
        println!("  Thread 1: Releasing write lock");
    });

    // Give thread 1 time to acquire lock
    thread::sleep(Duration::from_millis(10));

    println!("Thread 2: Attempting to acquire write lock with 50ms timeout...");
    let result = table.write_timeout(Duration::from_millis(50));
    if result.is_none() {
        println!("  ✗ Thread 2: Timeout - could not acquire lock");
    } else {
        println!("  ✓ Thread 2: Got lock (unexpected!)");
    }

    handle.join().unwrap();

    println!("Thread 2: Attempting to acquire write lock with 200ms timeout...");
    let result = table.write_timeout(Duration::from_millis(200));
    if result.is_some() {
        println!("  ✓ Thread 2: Successfully acquired lock after Thread 1 released");
    }

    println!();
    Ok(())
}

fn example_lock_statistics() -> anyhow::Result<()> {
    println!("--- Example 6: Lock Statistics Monitoring ---");

    let table = Arc::new(LockedSymbolTable::new());

    // Initialize
    {
        let mut guard = table.write();
        guard.add_domain(DomainInfo::new("User", 1000))?;
    }

    // Perform various operations
    println!("Performing concurrent operations...");

    let mut handles = vec![];

    // Readers
    for i in 0..5 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            for _ in 0..3 {
                let _guard = table_clone.read();
                thread::sleep(Duration::from_millis(10 * i));
            }
        }));
    }

    // Writers
    for i in 0..3 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            for j in 0..2 {
                let mut guard = table_clone.write();
                guard
                    .add_domain(DomainInfo::new(format!("Domain_{}_{}", i, j), 100))
                    .unwrap();
                thread::sleep(Duration::from_millis(15));
            }
        }));
    }

    // Some try_read/try_write attempts (may cause contentions)
    for _ in 0..3 {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            for _ in 0..5 {
                let _ = table_clone.try_read();
                thread::sleep(Duration::from_millis(5));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Print statistics
    let stats = table.stats();
    println!("\n=== Lock Statistics ===");
    println!("Read locks acquired: {}", stats.read_locks);
    println!("Write locks acquired: {}", stats.write_locks);
    println!("Read contentions: {}", stats.read_contentions);
    println!("Write contentions: {}", stats.write_contentions);
    println!("Average read wait: {:.2}ms", stats.avg_read_wait_ms());
    println!("Average write wait: {:.2}ms", stats.avg_write_wait_ms());
    println!(
        "Read contention rate: {:.2}%",
        stats.read_contention_rate() * 100.0
    );
    println!(
        "Write contention rate: {:.2}%",
        stats.write_contention_rate() * 100.0
    );
    println!("Transactions started: {}", stats.transactions_started);
    println!("Transactions committed: {}", stats.transactions_committed);
    println!(
        "Transactions rolled back: {}",
        stats.transactions_rolled_back
    );
    println!(
        "Transaction commit rate: {:.2}%",
        stats.commit_rate() * 100.0
    );

    // Reset stats
    println!("\nResetting statistics...");
    table.reset_stats();
    let stats_after_reset = table.stats();
    println!("Read locks after reset: {}", stats_after_reset.read_locks);

    println!();
    Ok(())
}
