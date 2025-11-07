//! Execution context and state management for coordinated execution.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::capabilities::DeviceType;
use crate::profiling::ProfileData;
use crate::strategy::ExecutionStrategy;

/// Execution phase for lifecycle tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    /// Preparing for execution (validation, optimization)
    Preparing,
    /// Currently executing
    Executing,
    /// Waiting for resources or synchronization
    Waiting,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl ExecutionPhase {
    pub fn as_str(&self) -> &str {
        match self {
            ExecutionPhase::Preparing => "Preparing",
            ExecutionPhase::Executing => "Executing",
            ExecutionPhase::Waiting => "Waiting",
            ExecutionPhase::Completed => "Completed",
            ExecutionPhase::Failed => "Failed",
            ExecutionPhase::Cancelled => "Cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ExecutionPhase::Completed | ExecutionPhase::Failed | ExecutionPhase::Cancelled
        )
    }
}

/// Execution state tracking
#[derive(Debug, Clone)]
pub struct ExecutionState {
    pub phase: ExecutionPhase,
    pub progress: f64, // 0.0 to 1.0
    pub current_node: Option<usize>,
    pub nodes_completed: usize,
    pub total_nodes: usize,
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub error_message: Option<String>,
}

impl ExecutionState {
    pub fn new(total_nodes: usize) -> Self {
        ExecutionState {
            phase: ExecutionPhase::Preparing,
            progress: 0.0,
            current_node: None,
            nodes_completed: 0,
            total_nodes,
            start_time: None,
            end_time: None,
            error_message: None,
        }
    }

    pub fn start(&mut self) {
        self.phase = ExecutionPhase::Executing;
        self.start_time = Some(Instant::now());
    }

    pub fn complete(&mut self) {
        self.phase = ExecutionPhase::Completed;
        self.end_time = Some(Instant::now());
        self.progress = 1.0;
    }

    pub fn fail(&mut self, error: impl Into<String>) {
        self.phase = ExecutionPhase::Failed;
        self.end_time = Some(Instant::now());
        self.error_message = Some(error.into());
    }

    pub fn cancel(&mut self) {
        self.phase = ExecutionPhase::Cancelled;
        self.end_time = Some(Instant::now());
    }

    pub fn update_progress(&mut self, node_idx: usize) {
        self.current_node = Some(node_idx);
        self.nodes_completed = node_idx + 1;
        self.progress = if self.total_nodes > 0 {
            self.nodes_completed as f64 / self.total_nodes as f64
        } else {
            0.0
        };
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| {
            self.end_time
                .unwrap_or_else(Instant::now)
                .duration_since(start)
        })
    }

    pub fn is_running(&self) -> bool {
        self.phase == ExecutionPhase::Executing
    }

    pub fn is_complete(&self) -> bool {
        self.phase.is_terminal()
    }
}

/// Hook for monitoring execution events
pub trait ExecutionHook: Send {
    /// Called when execution phase changes
    fn on_phase_change(&mut self, phase: ExecutionPhase, state: &ExecutionState);

    /// Called when a node starts executing
    fn on_node_start(&mut self, node_idx: usize, state: &ExecutionState);

    /// Called when a node completes
    fn on_node_complete(&mut self, node_idx: usize, duration: Duration, state: &ExecutionState);

    /// Called when an error occurs
    fn on_error(&mut self, error: &str, state: &ExecutionState);

    /// Called when execution completes
    fn on_complete(&mut self, state: &ExecutionState);
}

/// Simple logging hook for demonstration
pub struct LoggingHook {
    log_phase_changes: bool,
    log_node_execution: bool,
}

impl LoggingHook {
    pub fn new() -> Self {
        LoggingHook {
            log_phase_changes: true,
            log_node_execution: false,
        }
    }

    pub fn verbose() -> Self {
        LoggingHook {
            log_phase_changes: true,
            log_node_execution: true,
        }
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionHook for LoggingHook {
    fn on_phase_change(&mut self, phase: ExecutionPhase, _state: &ExecutionState) {
        if self.log_phase_changes {
            eprintln!("[ExecutionHook] Phase changed to: {}", phase.as_str());
        }
    }

    fn on_node_start(&mut self, node_idx: usize, _state: &ExecutionState) {
        if self.log_node_execution {
            eprintln!("[ExecutionHook] Starting node {}", node_idx);
        }
    }

    fn on_node_complete(&mut self, node_idx: usize, duration: Duration, _state: &ExecutionState) {
        if self.log_node_execution {
            eprintln!(
                "[ExecutionHook] Completed node {} in {:.3}ms",
                node_idx,
                duration.as_secs_f64() * 1000.0
            );
        }
    }

    fn on_error(&mut self, error: &str, _state: &ExecutionState) {
        eprintln!("[ExecutionHook] Error: {}", error);
    }

    fn on_complete(&mut self, state: &ExecutionState) {
        if self.log_phase_changes {
            if let Some(elapsed) = state.elapsed() {
                eprintln!(
                    "[ExecutionHook] Execution completed in {:.3}s",
                    elapsed.as_secs_f64()
                );
            }
        }
    }
}

/// Execution context for coordinated execution
pub struct ExecutionContext {
    pub state: ExecutionState,
    pub strategy: ExecutionStrategy,
    pub device: DeviceType,
    pub profile_data: Option<ProfileData>,
    pub metadata: HashMap<String, String>,
    hooks: Vec<Box<dyn ExecutionHook>>,
}

impl ExecutionContext {
    pub fn new(total_nodes: usize, strategy: ExecutionStrategy) -> Self {
        ExecutionContext {
            state: ExecutionState::new(total_nodes),
            strategy,
            device: DeviceType::CPU,
            profile_data: None,
            metadata: HashMap::new(),
            hooks: Vec::new(),
        }
    }

    pub fn with_device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        if enable {
            self.profile_data = Some(ProfileData::new());
        }
        self
    }

    pub fn add_hook(&mut self, hook: Box<dyn ExecutionHook>) {
        self.hooks.push(hook);
    }

    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    // Lifecycle methods
    pub fn start(&mut self) {
        self.state.start();
        self.notify_phase_change(ExecutionPhase::Executing);
    }

    pub fn complete(&mut self) {
        self.state.complete();
        self.notify_complete();
        self.notify_phase_change(ExecutionPhase::Completed);
    }

    pub fn fail(&mut self, error: impl Into<String>) {
        let error_msg = error.into();
        self.notify_error(&error_msg);
        self.state.fail(error_msg);
        self.notify_phase_change(ExecutionPhase::Failed);
    }

    pub fn cancel(&mut self) {
        self.state.cancel();
        self.notify_phase_change(ExecutionPhase::Cancelled);
    }

    pub fn begin_node(&mut self, node_idx: usize) {
        self.state.update_progress(node_idx);
        self.notify_node_start(node_idx);
    }

    pub fn end_node(&mut self, node_idx: usize, duration: Duration) {
        self.notify_node_complete(node_idx, duration);
    }

    // Hook notifications
    fn notify_phase_change(&mut self, phase: ExecutionPhase) {
        for hook in &mut self.hooks {
            hook.on_phase_change(phase, &self.state);
        }
    }

    fn notify_node_start(&mut self, node_idx: usize) {
        for hook in &mut self.hooks {
            hook.on_node_start(node_idx, &self.state);
        }
    }

    fn notify_node_complete(&mut self, node_idx: usize, duration: Duration) {
        for hook in &mut self.hooks {
            hook.on_node_complete(node_idx, duration, &self.state);
        }
    }

    fn notify_error(&mut self, error: &str) {
        for hook in &mut self.hooks {
            hook.on_error(error, &self.state);
        }
    }

    fn notify_complete(&mut self) {
        for hook in &mut self.hooks {
            hook.on_complete(&self.state);
        }
    }

    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Execution Context Summary\n");
        summary.push_str("=========================\n\n");
        summary.push_str(&format!("Phase: {}\n", self.state.phase.as_str()));
        summary.push_str(&format!("Progress: {:.1}%\n", self.state.progress * 100.0));
        summary.push_str(&format!(
            "Nodes: {}/{}\n",
            self.state.nodes_completed, self.state.total_nodes
        ));

        if let Some(elapsed) = self.state.elapsed() {
            summary.push_str(&format!("Elapsed: {:.3}s\n", elapsed.as_secs_f64()));
        }

        summary.push_str(&format!("Device: {}\n", self.device.as_str()));
        summary.push_str(&format!("Strategy: {:?}\n", self.strategy.mode));

        if let Some(error) = &self.state.error_message {
            summary.push_str(&format!("\nError: {}\n", error));
        }

        if !self.metadata.is_empty() {
            summary.push_str("\nMetadata:\n");
            for (key, value) in &self.metadata {
                summary.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_phase() {
        assert_eq!(ExecutionPhase::Preparing.as_str(), "Preparing");
        assert!(!ExecutionPhase::Executing.is_terminal());
        assert!(ExecutionPhase::Completed.is_terminal());
        assert!(ExecutionPhase::Failed.is_terminal());
    }

    #[test]
    fn test_execution_state_lifecycle() {
        let mut state = ExecutionState::new(10);

        assert_eq!(state.phase, ExecutionPhase::Preparing);
        assert_eq!(state.progress, 0.0);

        state.start();
        assert_eq!(state.phase, ExecutionPhase::Executing);
        assert!(state.is_running());

        state.update_progress(5);
        assert_eq!(state.current_node, Some(5));
        assert_eq!(state.progress, 0.6);

        state.complete();
        assert_eq!(state.phase, ExecutionPhase::Completed);
        assert!(state.is_complete());
        assert_eq!(state.progress, 1.0);
    }

    #[test]
    fn test_execution_state_failure() {
        let mut state = ExecutionState::new(10);
        state.start();
        state.fail("Test error");

        assert_eq!(state.phase, ExecutionPhase::Failed);
        assert_eq!(state.error_message, Some("Test error".to_string()));
        assert!(state.is_complete());
    }

    #[test]
    fn test_execution_state_elapsed() {
        let mut state = ExecutionState::new(5);
        state.start();
        std::thread::sleep(Duration::from_millis(10));
        state.complete();

        let elapsed = state.elapsed().unwrap();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_execution_context_creation() {
        let strategy = ExecutionStrategy::inference();
        let context = ExecutionContext::new(10, strategy);

        assert_eq!(context.state.total_nodes, 10);
        assert_eq!(context.device, DeviceType::CPU);
        assert!(context.profile_data.is_none());
    }

    #[test]
    fn test_execution_context_with_device() {
        let strategy = ExecutionStrategy::inference();
        let context = ExecutionContext::new(10, strategy).with_device(DeviceType::GPU);

        assert_eq!(context.device, DeviceType::GPU);
    }

    #[test]
    fn test_execution_context_with_profiling() {
        let strategy = ExecutionStrategy::inference();
        let context = ExecutionContext::new(10, strategy).with_profiling(true);

        assert!(context.profile_data.is_some());
    }

    #[test]
    fn test_execution_context_metadata() {
        let strategy = ExecutionStrategy::inference();
        let mut context = ExecutionContext::new(10, strategy);

        context.set_metadata("graph_id", "test-123");
        context.set_metadata("user", "test-user");

        assert_eq!(context.get_metadata("graph_id"), Some("test-123"));
        assert_eq!(context.get_metadata("user"), Some("test-user"));
        assert_eq!(context.get_metadata("missing"), None);
    }

    #[test]
    fn test_execution_context_lifecycle() {
        let strategy = ExecutionStrategy::inference();
        let mut context = ExecutionContext::new(5, strategy);

        context.start();
        assert!(context.state.is_running());

        context.begin_node(0);
        context.end_node(0, Duration::from_millis(10));

        context.begin_node(1);
        context.end_node(1, Duration::from_millis(15));

        assert_eq!(context.state.nodes_completed, 2);
        assert!(context.state.progress > 0.0);

        context.complete();
        assert!(context.state.is_complete());
        assert_eq!(context.state.phase, ExecutionPhase::Completed);
    }

    #[test]
    fn test_execution_context_failure() {
        let strategy = ExecutionStrategy::inference();
        let mut context = ExecutionContext::new(5, strategy);

        context.start();
        context.fail("Test error occurred");

        assert_eq!(context.state.phase, ExecutionPhase::Failed);
        assert_eq!(
            context.state.error_message,
            Some("Test error occurred".to_string())
        );
    }

    #[test]
    fn test_execution_context_summary() {
        let strategy = ExecutionStrategy::inference();
        let mut context = ExecutionContext::new(5, strategy);
        context.set_metadata("test_key", "test_value");

        context.start();
        context.begin_node(2);

        let summary = context.summary();
        assert!(summary.contains("Execution Context Summary"));
        assert!(summary.contains("Progress:"));
        assert!(summary.contains("test_key"));
    }

    #[test]
    fn test_logging_hook() {
        let hook = LoggingHook::new();
        assert!(hook.log_phase_changes);
        assert!(!hook.log_node_execution);

        let verbose_hook = LoggingHook::verbose();
        assert!(verbose_hook.log_phase_changes);
        assert!(verbose_hook.log_node_execution);
    }

    #[test]
    fn test_execution_with_hooks() {
        let strategy = ExecutionStrategy::inference();
        let mut context = ExecutionContext::new(3, strategy);

        // Add a logging hook
        context.add_hook(Box::new(LoggingHook::new()));

        context.start();
        context.begin_node(0);
        context.end_node(0, Duration::from_millis(10));
        context.complete();

        // Hooks should have been called (check via side effects in real implementation)
        assert_eq!(context.state.phase, ExecutionPhase::Completed);
    }
}
