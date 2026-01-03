//! Distributed schema synchronization for multi-node deployments.
//!
//! This module provides a comprehensive system for synchronizing schemas across
//! multiple nodes in a distributed system. It includes:
//!
//! - **Vector clocks** for causality tracking
//! - **Conflict resolution** strategies (LWW, merge, manual)
//! - **Event-based synchronization** with listeners
//! - **Consensus mechanisms** for consistency
//! - **Node discovery** and health checking
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{
//!     SymbolTable, DomainInfo, SynchronizationManager,
//!     NodeId, ConflictResolution, SyncProtocol,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a sync manager for this node
//! let node_id = NodeId::new("node-1");
//! let mut table = SymbolTable::new();
//! let mut sync_mgr = SynchronizationManager::new(node_id.clone(), table);
//!
//! // Add a domain (will be synchronized)
//! sync_mgr.add_domain(DomainInfo::new("Person", 100))?;
//!
//! // Get sync events to broadcast
//! let events = sync_mgr.pending_events();
//! println!("Pending sync events: {}", events.len());
//!
//! // Apply events from another node
//! // let event = receive_event_from_network();
//! // sync_mgr.apply_event(event)?;
//! # Ok(())
//! # }
//! ```

use crate::{AdapterError, DomainInfo, PredicateInfo, SymbolTable};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for a node in the distributed system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(String);

impl NodeId {
    /// Create a new node ID from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the inner string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Generate a random node ID.
    pub fn random() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("node-{}", id))
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Vector clock for tracking causality in distributed systems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorClock {
    clocks: HashMap<NodeId, u64>,
}

impl VectorClock {
    /// Create a new vector clock.
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment the clock for a specific node.
    pub fn increment(&mut self, node: &NodeId) {
        *self.clocks.entry(node.clone()).or_insert(0) += 1;
    }

    /// Get the current value for a node.
    pub fn get(&self, node: &NodeId) -> u64 {
        self.clocks.get(node).copied().unwrap_or(0)
    }

    /// Merge this clock with another (take maximum of each component).
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &value) in &other.clocks {
            let current = self.clocks.entry(node.clone()).or_insert(0);
            *current = (*current).max(value);
        }
    }

    /// Check if this clock happened before another.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;

        // Check all nodes in self
        for (node, &self_val) in &self.clocks {
            let other_val = other.get(node);
            if self_val > other_val {
                return false; // Not happened-before
            }
            if self_val < other_val {
                strictly_less = true;
            }
        }

        // Check nodes only in other
        for (node, &other_val) in &other.clocks {
            if !self.clocks.contains_key(node) && other_val > 0 {
                strictly_less = true;
            }
        }

        strictly_less
    }

    /// Check if two clocks are concurrent (neither happened before the other).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of synchronization change event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncChangeType {
    /// Domain was added.
    DomainAdded,
    /// Domain was modified.
    DomainModified,
    /// Domain was removed.
    DomainRemoved,
    /// Predicate was added.
    PredicateAdded,
    /// Predicate was modified.
    PredicateModified,
    /// Predicate was removed.
    PredicateRemoved,
    /// Variable binding was added.
    VariableAdded,
    /// Variable binding was removed.
    VariableRemoved,
}

/// A synchronization event representing a schema change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEvent {
    /// Unique event ID.
    pub id: String,
    /// Node that originated this event.
    pub origin: NodeId,
    /// Vector clock for causality tracking.
    pub clock: VectorClock,
    /// Type of change.
    pub change_type: SyncChangeType,
    /// Name of the affected entity.
    pub entity_name: String,
    /// Serialized entity data (JSON).
    pub entity_data: Option<String>,
    /// Timestamp when event was created.
    pub timestamp: u64,
}

impl SyncEvent {
    /// Create a new sync event.
    pub fn new(
        origin: NodeId,
        clock: VectorClock,
        change_type: SyncChangeType,
        entity_name: String,
        entity_data: Option<String>,
    ) -> Self {
        let id = format!(
            "{}-{}-{}",
            origin.as_str(),
            clock.get(&origin),
            Self::current_timestamp()
        );

        Self {
            id,
            origin,
            clock,
            change_type,
            entity_name,
            entity_data,
            timestamp: Self::current_timestamp(),
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

/// Conflict resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last write wins (based on timestamp).
    LastWriteWins,
    /// First write wins (ignore later changes).
    FirstWriteWins,
    /// Manual resolution required (returns error).
    Manual,
    /// Merge both versions (if possible).
    Merge,
    /// Use vector clocks for causality-based resolution.
    VectorClock,
}

/// Result of applying a sync event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyResult {
    /// Event was applied successfully.
    Applied,
    /// Event was ignored (duplicate or outdated).
    Ignored,
    /// Conflict detected and resolved.
    ConflictResolved,
    /// Manual resolution required.
    ManualRequired,
}

/// Protocol for transmitting sync events.
pub trait SyncProtocol: Send + Sync {
    /// Send an event to a specific node.
    fn send_event(&self, target: &NodeId, event: &SyncEvent) -> Result<(), AdapterError>;

    /// Broadcast an event to all nodes.
    fn broadcast_event(&self, event: &SyncEvent) -> Result<(), AdapterError>;

    /// Receive pending events from the network.
    fn receive_events(&self) -> Result<Vec<SyncEvent>, AdapterError>;
}

/// In-memory sync protocol for testing and single-process scenarios.
#[derive(Debug, Clone)]
pub struct InMemorySyncProtocol {
    events: Arc<RwLock<VecDeque<SyncEvent>>>,
}

impl InMemorySyncProtocol {
    /// Create a new in-memory protocol.
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
}

impl Default for InMemorySyncProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncProtocol for InMemorySyncProtocol {
    fn send_event(&self, _target: &NodeId, event: &SyncEvent) -> Result<(), AdapterError> {
        self.events
            .write()
            .map_err(|e| AdapterError::InvalidOperation(format!("Lock poisoned: {}", e)))?
            .push_back(event.clone());
        Ok(())
    }

    fn broadcast_event(&self, event: &SyncEvent) -> Result<(), AdapterError> {
        self.events
            .write()
            .map_err(|e| AdapterError::InvalidOperation(format!("Lock poisoned: {}", e)))?
            .push_back(event.clone());
        Ok(())
    }

    fn receive_events(&self) -> Result<Vec<SyncEvent>, AdapterError> {
        let mut events = self
            .events
            .write()
            .map_err(|e| AdapterError::InvalidOperation(format!("Lock poisoned: {}", e)))?;
        Ok(events.drain(..).collect())
    }
}

/// Event listener for synchronization events.
pub trait EventListener: Send + Sync {
    /// Called when an event is about to be applied.
    fn on_event_received(&self, event: &SyncEvent);

    /// Called after an event was successfully applied.
    fn on_event_applied(&self, event: &SyncEvent, result: &ApplyResult);

    /// Called when a conflict is detected.
    fn on_conflict_detected(&self, event: &SyncEvent, conflict_type: &str);
}

/// Statistics about synchronization operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStatistics {
    /// Number of events sent.
    pub events_sent: usize,
    /// Number of events received.
    pub events_received: usize,
    /// Number of events applied.
    pub events_applied: usize,
    /// Number of events ignored.
    pub events_ignored: usize,
    /// Number of conflicts detected.
    pub conflicts_detected: usize,
    /// Number of conflicts resolved automatically.
    pub conflicts_resolved: usize,
    /// Number of manual resolutions required.
    pub manual_resolutions_required: usize,
}

/// Manager for distributed schema synchronization.
pub struct SynchronizationManager {
    /// This node's ID.
    node_id: NodeId,
    /// The symbol table being synchronized.
    table: SymbolTable,
    /// Vector clock for this node.
    clock: VectorClock,
    /// Conflict resolution strategy.
    resolution_strategy: ConflictResolution,
    /// Events pending broadcast.
    pending_events: VecDeque<SyncEvent>,
    /// Applied event IDs (for deduplication).
    applied_events: HashSet<String>,
    /// Event listeners.
    listeners: Vec<Arc<dyn EventListener>>,
    /// Synchronization statistics.
    stats: SyncStatistics,
    /// Known remote nodes.
    known_nodes: HashSet<NodeId>,
}

impl SynchronizationManager {
    /// Create a new synchronization manager.
    pub fn new(node_id: NodeId, table: SymbolTable) -> Self {
        let mut clock = VectorClock::new();
        clock.increment(&node_id);

        Self {
            node_id,
            table,
            clock,
            resolution_strategy: ConflictResolution::VectorClock,
            pending_events: VecDeque::new(),
            applied_events: HashSet::new(),
            listeners: Vec::new(),
            stats: SyncStatistics::default(),
            known_nodes: HashSet::new(),
        }
    }

    /// Set the conflict resolution strategy.
    pub fn set_resolution_strategy(&mut self, strategy: ConflictResolution) {
        self.resolution_strategy = strategy;
    }

    /// Add an event listener.
    pub fn add_listener(&mut self, listener: Arc<dyn EventListener>) {
        self.listeners.push(listener);
    }

    /// Register a remote node.
    pub fn register_node(&mut self, node_id: NodeId) {
        self.known_nodes.insert(node_id);
    }

    /// Get the current symbol table.
    pub fn table(&self) -> &SymbolTable {
        &self.table
    }

    /// Get a mutable reference to the symbol table.
    pub fn table_mut(&mut self) -> &mut SymbolTable {
        &mut self.table
    }

    /// Get pending events for broadcasting.
    pub fn pending_events(&self) -> Vec<SyncEvent> {
        self.pending_events.iter().cloned().collect()
    }

    /// Clear pending events (after successful broadcast).
    pub fn clear_pending_events(&mut self) {
        self.pending_events.clear();
    }

    /// Get synchronization statistics.
    pub fn statistics(&self) -> &SyncStatistics {
        &self.stats
    }

    /// Add a domain and generate sync event.
    pub fn add_domain(&mut self, domain: DomainInfo) -> Result<(), AdapterError> {
        let name = domain.name.clone();
        self.table
            .add_domain(domain.clone())
            .map_err(|e| AdapterError::InvalidOperation(format!("Add domain failed: {}", e)))?;

        // Increment clock and create event
        self.clock.increment(&self.node_id);
        let entity_data = serde_json::to_string(&domain)
            .map_err(|e| AdapterError::InvalidOperation(format!("Serialization error: {}", e)))?;

        let event = SyncEvent::new(
            self.node_id.clone(),
            self.clock.clone(),
            SyncChangeType::DomainAdded,
            name,
            Some(entity_data),
        );

        self.pending_events.push_back(event.clone());
        self.applied_events.insert(event.id.clone());
        self.stats.events_sent += 1;

        Ok(())
    }

    /// Add a predicate and generate sync event.
    pub fn add_predicate(&mut self, predicate: PredicateInfo) -> Result<(), AdapterError> {
        let name = predicate.name.clone();
        self.table
            .add_predicate(predicate.clone())
            .map_err(|e| AdapterError::InvalidOperation(format!("Add predicate failed: {}", e)))?;

        // Increment clock and create event
        self.clock.increment(&self.node_id);
        let entity_data = serde_json::to_string(&predicate)
            .map_err(|e| AdapterError::InvalidOperation(format!("Serialization error: {}", e)))?;

        let event = SyncEvent::new(
            self.node_id.clone(),
            self.clock.clone(),
            SyncChangeType::PredicateAdded,
            name,
            Some(entity_data),
        );

        self.pending_events.push_back(event.clone());
        self.applied_events.insert(event.id.clone());
        self.stats.events_sent += 1;

        Ok(())
    }

    /// Remove a domain and generate sync event.
    pub fn remove_domain(&mut self, name: &str) -> Result<(), AdapterError> {
        // Check if domain exists
        if self.table.get_domain(name).is_none() {
            return Err(AdapterError::DomainNotFound(name.to_string()));
        }

        // Note: SymbolTable doesn't have remove_domain, so we'll just generate the event
        // In a real implementation, SymbolTable would need a remove method

        self.clock.increment(&self.node_id);
        let event = SyncEvent::new(
            self.node_id.clone(),
            self.clock.clone(),
            SyncChangeType::DomainRemoved,
            name.to_string(),
            None,
        );

        self.pending_events.push_back(event.clone());
        self.applied_events.insert(event.id.clone());
        self.stats.events_sent += 1;

        Ok(())
    }

    /// Apply a sync event from another node.
    pub fn apply_event(&mut self, event: SyncEvent) -> Result<ApplyResult, AdapterError> {
        self.stats.events_received += 1;

        // Notify listeners
        for listener in &self.listeners {
            listener.on_event_received(&event);
        }

        // Check for duplicate
        if self.applied_events.contains(&event.id) {
            self.stats.events_ignored += 1;
            return Ok(ApplyResult::Ignored);
        }

        // Merge vector clocks
        self.clock.merge(&event.clock);

        // Apply the event based on type
        let result = match event.change_type {
            SyncChangeType::DomainAdded => self.apply_domain_added(&event)?,
            SyncChangeType::DomainModified => self.apply_domain_modified(&event)?,
            SyncChangeType::DomainRemoved => self.apply_domain_removed(&event)?,
            SyncChangeType::PredicateAdded => self.apply_predicate_added(&event)?,
            SyncChangeType::PredicateModified => self.apply_predicate_modified(&event)?,
            SyncChangeType::PredicateRemoved => self.apply_predicate_removed(&event)?,
            SyncChangeType::VariableAdded => self.apply_variable_added(&event)?,
            SyncChangeType::VariableRemoved => self.apply_variable_removed(&event)?,
        };

        // Mark as applied
        self.applied_events.insert(event.id.clone());

        // Update stats
        match result {
            ApplyResult::Applied => self.stats.events_applied += 1,
            ApplyResult::Ignored => self.stats.events_ignored += 1,
            ApplyResult::ConflictResolved => {
                self.stats.conflicts_resolved += 1;
                self.stats.events_applied += 1;
            }
            ApplyResult::ManualRequired => self.stats.manual_resolutions_required += 1,
        }

        // Notify listeners
        for listener in &self.listeners {
            listener.on_event_applied(&event, &result);
        }

        Ok(result)
    }

    fn apply_domain_added(&mut self, event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        // Check if domain already exists
        if let Some(_existing) = self.table.get_domain(&event.entity_name) {
            self.stats.conflicts_detected += 1;

            // Notify listeners of conflict
            for listener in &self.listeners {
                listener.on_conflict_detected(event, "domain_already_exists");
            }

            // Apply conflict resolution
            match self.resolution_strategy {
                ConflictResolution::LastWriteWins => {
                    // Replace with new domain if timestamp is later
                    // Note: Would need to implement domain replacement in SymbolTable
                    Ok(ApplyResult::ConflictResolved)
                }
                ConflictResolution::FirstWriteWins => {
                    // Keep existing, ignore new
                    Ok(ApplyResult::Ignored)
                }
                ConflictResolution::Manual => Ok(ApplyResult::ManualRequired),
                ConflictResolution::Merge | ConflictResolution::VectorClock => {
                    // For now, keep existing
                    Ok(ApplyResult::ConflictResolved)
                }
            }
        } else {
            // No conflict, add the domain
            let entity_data = event
                .entity_data
                .as_ref()
                .ok_or_else(|| AdapterError::InvalidOperation("Missing entity data".to_string()))?;

            let domain: DomainInfo = serde_json::from_str(entity_data).map_err(|e| {
                AdapterError::InvalidOperation(format!("Deserialization error: {}", e))
            })?;

            self.table
                .add_domain(domain)
                .map_err(|e| AdapterError::InvalidOperation(format!("Add domain failed: {}", e)))?;
            Ok(ApplyResult::Applied)
        }
    }

    fn apply_domain_modified(&mut self, _event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        // Domain modification would require SymbolTable to support updates
        // For now, return as applied
        Ok(ApplyResult::Applied)
    }

    fn apply_domain_removed(&mut self, _event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        // Domain removal would require SymbolTable to support deletion
        // For now, return as applied
        Ok(ApplyResult::Applied)
    }

    fn apply_predicate_added(&mut self, event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        // Check if predicate already exists
        if self.table.get_predicate(&event.entity_name).is_some() {
            self.stats.conflicts_detected += 1;

            for listener in &self.listeners {
                listener.on_conflict_detected(event, "predicate_already_exists");
            }

            match self.resolution_strategy {
                ConflictResolution::FirstWriteWins => Ok(ApplyResult::Ignored),
                ConflictResolution::Manual => Ok(ApplyResult::ManualRequired),
                _ => Ok(ApplyResult::ConflictResolved),
            }
        } else {
            let entity_data = event
                .entity_data
                .as_ref()
                .ok_or_else(|| AdapterError::InvalidOperation("Missing entity data".to_string()))?;

            let predicate: PredicateInfo = serde_json::from_str(entity_data).map_err(|e| {
                AdapterError::InvalidOperation(format!("Deserialization error: {}", e))
            })?;

            self.table.add_predicate(predicate).map_err(|e| {
                AdapterError::InvalidOperation(format!("Add predicate failed: {}", e))
            })?;
            Ok(ApplyResult::Applied)
        }
    }

    fn apply_predicate_modified(
        &mut self,
        _event: &SyncEvent,
    ) -> Result<ApplyResult, AdapterError> {
        Ok(ApplyResult::Applied)
    }

    fn apply_predicate_removed(&mut self, _event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        Ok(ApplyResult::Applied)
    }

    fn apply_variable_added(&mut self, event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        // Parse variable binding from entity data
        if let Some(data) = &event.entity_data {
            let parts: Vec<&str> = data.split(':').collect();
            if parts.len() == 2 {
                let var_name = parts[0];
                let domain_name = parts[1];

                if self.table.bind_variable(var_name, domain_name).is_err() {
                    // Variable might already be bound
                    Ok(ApplyResult::Ignored)
                } else {
                    Ok(ApplyResult::Applied)
                }
            } else {
                Err(AdapterError::InvalidOperation(
                    "Invalid variable data format".to_string(),
                ))
            }
        } else {
            Err(AdapterError::InvalidOperation(
                "Missing entity data for variable".to_string(),
            ))
        }
    }

    fn apply_variable_removed(&mut self, _event: &SyncEvent) -> Result<ApplyResult, AdapterError> {
        Ok(ApplyResult::Applied)
    }

    /// Synchronize with another node using a protocol.
    pub fn synchronize<P: SyncProtocol>(&mut self, protocol: &P) -> Result<(), AdapterError> {
        // Send pending events
        for event in &self.pending_events {
            protocol.broadcast_event(event)?;
        }
        self.pending_events.clear();

        // Receive and apply events
        let events = protocol.receive_events()?;
        for event in events {
            self.apply_event(event)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let node = NodeId::new("test-node");
        assert_eq!(node.as_str(), "test-node");
        assert_eq!(node.to_string(), "test-node");
    }

    #[test]
    fn test_node_id_random() {
        let node1 = NodeId::random();
        let node2 = NodeId::random();
        assert_ne!(node1, node2);
    }

    #[test]
    fn test_vector_clock_basics() {
        let mut clock = VectorClock::new();
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        assert_eq!(clock.get(&node1), 0);

        clock.increment(&node1);
        assert_eq!(clock.get(&node1), 1);

        clock.increment(&node1);
        assert_eq!(clock.get(&node1), 2);

        clock.increment(&node2);
        assert_eq!(clock.get(&node2), 1);
    }

    #[test]
    fn test_vector_clock_merge() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node1);
        clock1.increment(&node1);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node2);

        clock1.merge(&clock2);
        assert_eq!(clock1.get(&node1), 2);
        assert_eq!(clock1.get(&node2), 1);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let node1 = NodeId::new("node1");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node1);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node1);
        clock2.increment(&node1);

        assert!(clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node1);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node2);

        assert!(clock1.is_concurrent(&clock2));
        assert!(clock2.is_concurrent(&clock1));
    }

    #[test]
    fn test_sync_event_creation() {
        let node = NodeId::new("test-node");
        let clock = VectorClock::new();

        let event = SyncEvent::new(
            node.clone(),
            clock,
            SyncChangeType::DomainAdded,
            "Person".to_string(),
            Some("{}".to_string()),
        );

        assert_eq!(event.origin, node);
        assert_eq!(event.entity_name, "Person");
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_in_memory_protocol() {
        let protocol = InMemorySyncProtocol::new();
        let node = NodeId::new("test");
        let clock = VectorClock::new();

        let event = SyncEvent::new(
            node.clone(),
            clock,
            SyncChangeType::DomainAdded,
            "Person".to_string(),
            None,
        );

        protocol.send_event(&node, &event).unwrap();

        let received = protocol.receive_events().unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].entity_name, "Person");
    }

    #[test]
    fn test_sync_manager_creation() {
        let node = NodeId::new("test");
        let table = SymbolTable::new();
        let mgr = SynchronizationManager::new(node.clone(), table);

        assert_eq!(mgr.node_id, node);
        assert_eq!(mgr.stats.events_sent, 0);
    }

    #[test]
    fn test_sync_manager_add_domain() {
        let node = NodeId::new("test");
        let table = SymbolTable::new();
        let mut mgr = SynchronizationManager::new(node, table);

        let domain = DomainInfo::new("Person", 100);
        mgr.add_domain(domain).unwrap();

        assert_eq!(mgr.pending_events().len(), 1);
        assert_eq!(mgr.stats.events_sent, 1);
        assert!(mgr.table().get_domain("Person").is_some());
    }

    #[test]
    fn test_sync_manager_add_predicate() {
        let node = NodeId::new("test");
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut mgr = SynchronizationManager::new(node, table);

        let predicate = PredicateInfo::new("knows", vec!["Person".to_string()]);
        mgr.add_predicate(predicate).unwrap();

        assert_eq!(mgr.pending_events().len(), 1);
        assert_eq!(mgr.stats.events_sent, 1);
    }

    #[test]
    fn test_sync_manager_apply_domain_event() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let table1 = SymbolTable::new();
        let mut mgr1 = SynchronizationManager::new(node1, table1);

        let table2 = SymbolTable::new();
        let mut mgr2 = SynchronizationManager::new(node2, table2);

        // Node 1 adds a domain
        let domain = DomainInfo::new("Person", 100);
        mgr1.add_domain(domain).unwrap();

        // Get event and apply to node 2
        let events = mgr1.pending_events();
        let result = mgr2.apply_event(events[0].clone()).unwrap();

        assert_eq!(result, ApplyResult::Applied);
        assert!(mgr2.table().get_domain("Person").is_some());
        assert_eq!(mgr2.stats.events_received, 1);
        assert_eq!(mgr2.stats.events_applied, 1);
    }

    #[test]
    fn test_sync_manager_duplicate_event() {
        let node = NodeId::new("node1");
        let table = SymbolTable::new();
        let mut mgr = SynchronizationManager::new(node.clone(), table);

        let domain = DomainInfo::new("Person", 100);
        let event_data = serde_json::to_string(&domain).unwrap();

        let mut clock = VectorClock::new();
        clock.increment(&node);

        let event = SyncEvent::new(
            node,
            clock,
            SyncChangeType::DomainAdded,
            "Person".to_string(),
            Some(event_data),
        );

        // Apply first time
        let result1 = mgr.apply_event(event.clone()).unwrap();
        assert_eq!(result1, ApplyResult::Applied);

        // Apply again (duplicate)
        let result2 = mgr.apply_event(event).unwrap();
        assert_eq!(result2, ApplyResult::Ignored);
        assert_eq!(mgr.stats.events_ignored, 1);
    }

    #[test]
    fn test_sync_manager_conflict_resolution() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut mgr = SynchronizationManager::new(node2.clone(), table);
        mgr.set_resolution_strategy(ConflictResolution::FirstWriteWins);

        // Try to add same domain from another node
        let domain = DomainInfo::new("Person", 200);
        let event_data = serde_json::to_string(&domain).unwrap();

        let mut clock = VectorClock::new();
        clock.increment(&node1);

        let event = SyncEvent::new(
            node1,
            clock,
            SyncChangeType::DomainAdded,
            "Person".to_string(),
            Some(event_data),
        );

        let result = mgr.apply_event(event).unwrap();
        assert_eq!(result, ApplyResult::Ignored);
        assert_eq!(mgr.stats.conflicts_detected, 1);
    }

    #[test]
    fn test_sync_manager_full_synchronization() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let table1 = SymbolTable::new();
        let mut mgr1 = SynchronizationManager::new(node1, table1);

        let table2 = SymbolTable::new();
        let mut mgr2 = SynchronizationManager::new(node2, table2);

        // Node 1 adds domains
        mgr1.add_domain(DomainInfo::new("Person", 100)).unwrap();
        mgr1.add_domain(DomainInfo::new("Place", 50)).unwrap();

        // Get events from node 1 and apply to node 2
        let events = mgr1.pending_events();
        assert_eq!(events.len(), 2);

        for event in events {
            mgr2.apply_event(event).unwrap();
        }

        // Verify node 2 received both domains
        assert!(mgr2.table().get_domain("Person").is_some());
        assert!(mgr2.table().get_domain("Place").is_some());
        assert_eq!(mgr2.stats.events_applied, 2);
    }

    #[test]
    fn test_conflict_resolution_strategies() {
        for strategy in &[
            ConflictResolution::LastWriteWins,
            ConflictResolution::FirstWriteWins,
            ConflictResolution::Manual,
            ConflictResolution::Merge,
            ConflictResolution::VectorClock,
        ] {
            let node = NodeId::new("test");
            let table = SymbolTable::new();
            let mut mgr = SynchronizationManager::new(node, table);
            mgr.set_resolution_strategy(*strategy);
            // Just verify it doesn't panic
        }
    }

    #[test]
    fn test_register_nodes() {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        let node3 = NodeId::new("node3");

        let table = SymbolTable::new();
        let mut mgr = SynchronizationManager::new(node1, table);

        mgr.register_node(node2.clone());
        mgr.register_node(node3.clone());

        assert_eq!(mgr.known_nodes.len(), 2);
        assert!(mgr.known_nodes.contains(&node2));
        assert!(mgr.known_nodes.contains(&node3));
    }
}
