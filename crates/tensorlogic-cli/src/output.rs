//! Colored output formatting for TensorLogic CLI

use colored::*;
use tensorlogic_ir::EinsumGraph;

pub fn print_success(message: &str) {
    println!("{} {}", "✓".green().bold(), message);
}

pub fn print_error(message: &str) {
    eprintln!("{} {}", "✗".red().bold(), message);
}

pub fn print_info(message: &str) {
    println!("{} {}", "ℹ".blue().bold(), message);
}

pub fn print_header(title: &str) {
    println!("\n{}", title.cyan().bold());
    println!("{}", "=".repeat(title.len()).cyan());
}

pub fn format_graph_stats(graph: &EinsumGraph) -> String {
    format!(
        "Graph: {} tensors, {} nodes, {} inputs, {} outputs",
        graph.tensors.len().to_string().green(),
        graph.nodes.len().to_string().cyan(),
        graph.inputs.len().to_string().yellow(),
        graph.outputs.len().to_string().magenta()
    )
}

pub fn print_compilation_success(graph: &EinsumGraph) {
    print_success(&format!("Compilation successful"));
    println!("  {}", format_graph_stats(graph));
}

pub fn enable_colors(enabled: bool) {
    colored::control::set_override(enabled);
}
