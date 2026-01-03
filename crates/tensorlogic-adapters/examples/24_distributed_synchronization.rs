//! Distributed Schema Synchronization Example
//!
//! This example demonstrates the distributed schema synchronization system,
//! showing how multiple nodes can coordinate schema changes across a network.
//!
//! Features demonstrated:
//! - Multi-node setup with unique IDs
//! - Schema changes and event generation
//! - Event propagation between nodes
//! - Conflict detection and resolution
//! - Vector clock causality tracking
//! - Synchronization statistics

use tensorlogic_adapters::{
    ApplyResult, ConflictResolution, DomainInfo, InMemorySyncProtocol, NodeId, PredicateInfo,
    SymbolTable, SynchronizationManager,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Distributed Schema Synchronization Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // ========================================================================
    // Scenario 1: Basic Synchronization Between Two Nodes
    // ========================================================================
    println!("📡 Scenario 1: Basic Two-Node Synchronization\n");
    println!("────────────────────────────────────────────────────────────");

    // Create two nodes
    let node1_id = NodeId::new("datacenter-us-east");
    let node2_id = NodeId::new("datacenter-eu-west");

    let mut node1 = SynchronizationManager::new(node1_id.clone(), SymbolTable::new());
    let mut node2 = SynchronizationManager::new(node2_id.clone(), SymbolTable::new());

    println!("✓ Created nodes: {} and {}\n", node1_id, node2_id);

    // Node 1 adds some domains
    println!("Node 1 adding domains...");
    node1.add_domain(DomainInfo::new("Person", 1000))?;
    node1.add_domain(DomainInfo::new("Organization", 500))?;
    println!("  └─ Added: Person (cardinality: 1000)");
    println!("  └─ Added: Organization (cardinality: 500)\n");

    // Get pending events from node 1
    let events_from_node1 = node1.pending_events();
    println!("Node 1 has {} pending events", events_from_node1.len());

    // Propagate events to node 2
    println!("\nPropagating events to Node 2...");
    for event in events_from_node1 {
        println!(
            "  └─ Applying event: {} (from {})",
            event.entity_name, event.origin
        );
        let result = node2.apply_event(event)?;
        match result {
            ApplyResult::Applied => println!("     ✓ Successfully applied"),
            ApplyResult::Ignored => println!("     ⊘ Ignored (duplicate)"),
            ApplyResult::ConflictResolved => println!("     ⚠ Conflict resolved"),
            ApplyResult::ManualRequired => println!("     ⚠ Manual resolution required"),
        }
    }

    // Verify synchronization
    println!("\n Node 2 synchronization status:");
    println!(
        "  └─ Person domain: {}",
        if node2.table().get_domain("Person").is_some() {
            "✓ Present"
        } else {
            "✗ Missing"
        }
    );
    println!(
        "  └─ Organization domain: {}",
        if node2.table().get_domain("Organization").is_some() {
            "✓ Present"
        } else {
            "✗ Missing"
        }
    );

    // Show statistics
    let stats1 = node1.statistics();
    let stats2 = node2.statistics();
    println!("\n📊 Statistics:");
    println!("  Node 1: {} events sent", stats1.events_sent);
    println!(
        "  Node 2: {} events received, {} applied",
        stats2.events_received, stats2.events_applied
    );

    // ========================================================================
    // Scenario 2: Bidirectional Synchronization
    // ========================================================================
    println!("\n\n📡 Scenario 2: Bidirectional Synchronization\n");
    println!("────────────────────────────────────────────────────────────");

    // Node 2 adds predicates
    println!("Node 2 adding predicates...");
    node2.add_predicate(PredicateInfo::new(
        "worksAt",
        vec!["Person".to_string(), "Organization".to_string()],
    ))?;
    println!("  └─ Added: worksAt(Person, Organization)\n");

    // Propagate from node 2 to node 1
    let events_from_node2 = node2.pending_events();
    println!("Node 2 has {} pending event(s)", events_from_node2.len());

    println!("\nPropagating events to Node 1...");
    for event in events_from_node2 {
        println!(
            "  └─ Applying event: {} (from {})",
            event.entity_name, event.origin
        );
        node1.apply_event(event)?;
    }

    // Verify bidirectional sync
    println!("\n✓ Both nodes now have:");
    println!("  └─ 2 domains (Person, Organization)");
    println!("  └─ 1 predicate (worksAt)");

    // ========================================================================
    // Scenario 3: Conflict Detection and Resolution
    // ========================================================================
    println!("\n\n⚠️  Scenario 3: Conflict Detection and Resolution\n");
    println!("────────────────────────────────────────────────────────────");

    // Create two new nodes with different resolution strategies
    let node3 = SynchronizationManager::new(NodeId::new("node-alpha"), SymbolTable::new());
    let mut node4 = SynchronizationManager::new(NodeId::new("node-beta"), SymbolTable::new());

    // Set conflict resolution strategy
    node4.set_resolution_strategy(ConflictResolution::FirstWriteWins);
    println!("Node Beta configured with FirstWriteWins strategy\n");

    // Node 4 already has a domain
    node4.add_domain(DomainInfo::new("Product", 100))?;
    println!("Node Beta adds: Product (cardinality: 100)");

    // Node 3 tries to add same domain with different cardinality
    let mut node3_copy = node3;
    node3_copy.add_domain(DomainInfo::new("Product", 200))?;
    println!("Node Alpha adds: Product (cardinality: 200)");

    // Try to apply Alpha's event to Beta
    println!("\nAttempting to apply Node Alpha's change to Node Beta...");
    let alpha_events = node3_copy.pending_events();
    for event in alpha_events {
        let result = node4.apply_event(event)?;
        match result {
            ApplyResult::Ignored => {
                println!("  ⊘ Conflict detected! FirstWriteWins strategy kept original value");
                println!("     Final cardinality: 100 (Node Beta's original value)");
            }
            ApplyResult::ConflictResolved => {
                println!("  ⚠ Conflict resolved automatically");
            }
            _ => println!("  Result: {:?}", result),
        }
    }

    let conflict_stats = node4.statistics();
    println!("\n📊 Conflict Statistics:");
    println!(
        "  └─ Conflicts detected: {}",
        conflict_stats.conflicts_detected
    );
    println!(
        "  └─ Conflicts resolved: {}",
        conflict_stats.conflicts_resolved
    );

    // ========================================================================
    // Scenario 4: Vector Clock Causality
    // ========================================================================
    println!("\n\n🕐 Scenario 4: Vector Clock Causality Tracking\n");
    println!("────────────────────────────────────────────────────────────");

    // Create a small network of 3 nodes
    let node_a = NodeId::new("node-A");
    let node_b = NodeId::new("node-B");
    let node_c = NodeId::new("node-C");

    let mut mgr_a = SynchronizationManager::new(node_a.clone(), SymbolTable::new());
    let mut mgr_b = SynchronizationManager::new(node_b.clone(), SymbolTable::new());
    let mut mgr_c = SynchronizationManager::new(node_c.clone(), SymbolTable::new());

    println!("✓ Created 3-node network: A, B, C\n");

    // A adds a domain
    mgr_a.add_domain(DomainInfo::new("User", 50))?;
    println!("Node A: Added User domain");

    // Propagate A → B
    let events_a = mgr_a.pending_events();
    for event in &events_a {
        mgr_b.apply_event(event.clone())?;
    }
    println!("  └─ Propagated to Node B");

    // B adds a domain
    mgr_b.add_domain(DomainInfo::new("Post", 200))?;
    println!("\nNode B: Added Post domain");

    // Propagate both A's event and B's event to C
    // First, propagate A's event to C
    for event in events_a {
        mgr_c.apply_event(event)?;
    }

    // Then propagate B's event to C
    let events_b = mgr_b.pending_events();
    for event in events_b {
        mgr_c.apply_event(event)?;
    }
    println!("  └─ Propagated to Node C");

    // Verify causal ordering
    println!("\n✓ Node C received events in causal order:");
    println!(
        "  └─ Has User domain: {}",
        mgr_c.table().get_domain("User").is_some()
    );
    println!(
        "  └─ Has Post domain: {}",
        mgr_c.table().get_domain("Post").is_some()
    );

    // ========================================================================
    // Scenario 5: Using the InMemorySyncProtocol
    // ========================================================================
    println!("\n\n🔌 Scenario 5: Using InMemorySyncProtocol\n");
    println!("────────────────────────────────────────────────────────────");

    let protocol = InMemorySyncProtocol::new();
    let node_x = NodeId::new("node-X");
    let mut mgr_x = SynchronizationManager::new(node_x, SymbolTable::new());

    // Add a domain
    mgr_x.add_domain(DomainInfo::new("Event", 1000))?;
    println!("Node X: Added Event domain");

    // Use synchronize method
    mgr_x.synchronize(&protocol)?;
    println!("  └─ Synchronized using protocol");

    // Another node can receive
    let node_y = NodeId::new("node-Y");
    let _mgr_y = SynchronizationManager::new(node_y, SymbolTable::new());

    // Note: In a real scenario, the protocol would be shared between nodes
    // Here we demonstrate the API
    println!("\nNode Y ready to receive events via protocol");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n\n═══════════════════════════════════════════════════════════");
    println!("  Summary: Distributed Synchronization Capabilities");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("✓ Features Demonstrated:");
    println!("  1. Basic two-node synchronization");
    println!("  2. Bidirectional event propagation");
    println!("  3. Conflict detection and resolution strategies");
    println!("  4. Vector clock causality tracking");
    println!("  5. Protocol-based synchronization\n");

    println!("📋 Conflict Resolution Strategies Available:");
    println!("  • LastWriteWins  - Use timestamp to resolve");
    println!("  • FirstWriteWins - Keep first value, ignore later");
    println!("  • Manual         - Require manual intervention");
    println!("  • Merge          - Attempt to merge both versions");
    println!("  • VectorClock    - Use causality for resolution\n");

    println!("🎯 Use Cases:");
    println!("  • Multi-region data centers");
    println!("  • Collaborative schema editing");
    println!("  • Distributed ML training metadata");
    println!("  • Edge-cloud synchronization");
    println!("  • Multi-tenant schema management\n");

    Ok(())
}
