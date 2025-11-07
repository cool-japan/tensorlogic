//! Jupyter notebook integration for rich display and visualization
//!
//! This module provides `__repr_html__()` methods for TensorLogic types,
//! enabling rich interactive displays in Jupyter notebooks, JupyterLab,
//! and other IPython-based environments.

use std::collections::HashMap;

/// Generate HTML table from key-value pairs
pub fn html_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut html =
        String::from("<table style='border-collapse: collapse; border: 1px solid #ddd;'>");

    // Header
    html.push_str("<thead><tr style='background-color: #f2f2f2;'>");
    for header in headers {
        html.push_str(&format!(
            "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{}</th>",
            header
        ));
    }
    html.push_str("</tr></thead>");

    // Rows
    html.push_str("<tbody>");
    for row in rows {
        html.push_str("<tr>");
        for cell in row {
            html.push_str(&format!(
                "<td style='border: 1px solid #ddd; padding: 8px;'>{}</td>",
                cell
            ));
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");

    html
}

/// Generate HTML card with title and content
pub fn html_card(title: &str, content: &str, color: &str) -> String {
    format!(
        "<div style='border: 2px solid {}; border-radius: 8px; padding: 16px; margin: 8px 0; background-color: #f9f9f9;'>\
         <h3 style='margin-top: 0; color: {};'>{}</h3>{}</div>",
        color, color, title, content
    )
}

/// Generate HTML badge
pub fn html_badge(text: &str, color: &str) -> String {
    format!(
        "<span style='background-color: {}; color: white; padding: 4px 8px; border-radius: 4px; \
         font-size: 12px; font-weight: bold; margin-right: 4px;'>{}</span>",
        color, text
    )
}

/// Generate HTML key-value list
pub fn html_kv_list(items: &[(String, String)]) -> String {
    let mut html =
        String::from("<dl style='display: grid; grid-template-columns: auto 1fr; gap: 8px;'>");

    for (key, value) in items {
        html.push_str(&format!(
            "<dt style='font-weight: bold;'>{}</dt><dd style='margin: 0;'>{}</dd>",
            key, value
        ));
    }

    html.push_str("</dl>");
    html
}

/// Generate HTML for EinsumGraph visualization
pub fn einsum_graph_html(
    num_nodes: usize,
    num_inputs: usize,
    num_outputs: usize,
    node_types: &HashMap<String, usize>,
) -> String {
    let mut html = String::new();

    // Title
    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>📊 EinsumGraph</h2>");

    // Statistics badges
    html.push_str("<div style='margin-bottom: 16px;'>");
    html.push_str(&html_badge(&format!("{} nodes", num_nodes), "#3498db"));
    html.push_str(&html_badge(&format!("{} inputs", num_inputs), "#2ecc71"));
    html.push_str(&html_badge(&format!("{} outputs", num_outputs), "#e74c3c"));
    html.push_str("</div>");

    // Node type breakdown
    if !node_types.is_empty() {
        html.push_str("<h4 style='color: #34495e; margin-top: 16px;'>Node Types:</h4>");
        html.push_str("<ul style='list-style-type: none; padding-left: 0;'>");

        for (node_type, count) in node_types {
            let percentage = if num_nodes > 0 {
                ((*count as f64 / num_nodes as f64) * 100.0).round() as u32
            } else {
                0
            };

            html.push_str(&format!(
                "<li style='margin: 8px 0;'>\
                 <span style='display: inline-block; width: 120px; font-weight: bold;'>{}</span>\
                 <span style='color: #7f8c8d;'>{} ({}%)</span>\
                 <div style='background-color: #ecf0f1; height: 8px; width: 200px; display: inline-block; margin-left: 8px; border-radius: 4px;'>\
                 <div style='background-color: #3498db; height: 8px; width: {}%; border-radius: 4px;'></div>\
                 </div></li>",
                node_type, count, percentage, percentage
            ));
        }

        html.push_str("</ul>");
    }

    html.push_str("</div>");
    html
}

/// Generate HTML for SymbolTable visualization
pub fn symbol_table_html(
    domains: &[(String, usize, Option<String>)],
    predicates: &[(String, usize, Vec<String>)],
    variables: &[(String, String)],
) -> String {
    let mut html = String::new();

    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>🗂️ Symbol Table</h2>");

    // Summary badges
    html.push_str("<div style='margin-bottom: 16px;'>");
    html.push_str(&html_badge(
        &format!("{} domains", domains.len()),
        "#9b59b6",
    ));
    html.push_str(&html_badge(
        &format!("{} predicates", predicates.len()),
        "#e67e22",
    ));
    html.push_str(&html_badge(
        &format!("{} variables", variables.len()),
        "#1abc9c",
    ));
    html.push_str("</div>");

    // Domains table
    if !domains.is_empty() {
        html.push_str(&html_card(
            "Domains",
            &{
                let rows: Vec<Vec<String>> = domains
                    .iter()
                    .map(|(name, card, desc)| {
                        vec![
                            name.clone(),
                            card.to_string(),
                            desc.as_ref().unwrap_or(&"-".to_string()).clone(),
                        ]
                    })
                    .collect();
                html_table(&["Name", "Cardinality", "Description"], &rows)
            },
            "#9b59b6",
        ));
    }

    // Predicates table
    if !predicates.is_empty() {
        html.push_str(&html_card(
            "Predicates",
            &{
                let rows: Vec<Vec<String>> = predicates
                    .iter()
                    .map(|(name, arity, domains)| {
                        vec![name.clone(), arity.to_string(), domains.join(", ")]
                    })
                    .collect();
                html_table(&["Name", "Arity", "Domains"], &rows)
            },
            "#e67e22",
        ));
    }

    // Variable bindings table
    if !variables.is_empty() {
        html.push_str(&html_card(
            "Variable Bindings",
            &{
                let rows: Vec<Vec<String>> = variables
                    .iter()
                    .map(|(var, domain)| vec![var.clone(), domain.clone()])
                    .collect();
                html_table(&["Variable", "Domain"], &rows)
            },
            "#1abc9c",
        ));
    }

    html.push_str("</div>");
    html
}

/// Generate HTML for CompilationConfig visualization
pub fn compilation_config_html(config_name: &str, description: &str) -> String {
    let mut html = String::new();

    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>⚙️ Compilation Config</h2>");

    html.push_str(&html_card(
        config_name,
        &format!("<p style='margin: 0;'>{}</p>", description),
        "#16a085",
    ));

    html.push_str("</div>");
    html
}

/// Generate HTML for ModelPackage visualization
pub fn model_package_html(
    has_graph: bool,
    has_config: bool,
    has_symbol_table: bool,
    has_parameters: bool,
    metadata: &HashMap<String, String>,
) -> String {
    let mut html = String::new();

    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>📦 Model Package</h2>");

    // Components
    html.push_str("<div style='margin-bottom: 16px;'>");
    html.push_str("<h4 style='color: #34495e;'>Components:</h4>");
    html.push_str("<ul style='list-style-type: none; padding-left: 0;'>");

    let components = [
        ("Graph", has_graph, "🔵"),
        ("Config", has_config, "🟢"),
        ("Symbol Table", has_symbol_table, "🟡"),
        ("Parameters", has_parameters, "🟣"),
    ];

    for (name, present, emoji) in &components {
        let status = if *present { "✓" } else { "✗" };
        let color = if *present { "#2ecc71" } else { "#95a5a6" };
        html.push_str(&format!(
            "<li style='margin: 4px 0; color: {};'>{} {} {}</li>",
            color, emoji, status, name
        ));
    }

    html.push_str("</ul>");
    html.push_str("</div>");

    // Metadata
    if !metadata.is_empty() {
        let items: Vec<(String, String)> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        html.push_str(&html_card("Metadata", &html_kv_list(&items), "#3498db"));
    }

    html.push_str("</div>");
    html
}

/// Generate HTML for TrainingHistory visualization
pub fn training_history_html(
    num_epochs: usize,
    train_losses: &[f64],
    val_losses: &Option<Vec<f64>>,
) -> String {
    let mut html = String::new();

    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>📈 Training History</h2>");

    // Summary
    html.push_str("<div style='margin-bottom: 16px;'>");
    html.push_str(&html_badge(&format!("{} epochs", num_epochs), "#e74c3c"));

    if let Some(last_loss) = train_losses.last() {
        html.push_str(&html_badge(
            &format!("Final loss: {:.6}", last_loss),
            "#2ecc71",
        ));
    }

    if let Some(val) = val_losses {
        if let Some(last_val) = val.last() {
            html.push_str(&html_badge(
                &format!("Final val loss: {:.6}", last_val),
                "#3498db",
            ));
        }
    }
    html.push_str("</div>");

    // Loss table (last 5 epochs)
    let start_idx = num_epochs.saturating_sub(5);
    let mut rows = vec![];

    for i in start_idx..num_epochs {
        let mut row = vec![(i + 1).to_string(), format!("{:.6}", train_losses[i])];

        if let Some(val) = val_losses {
            row.push(format!("{:.6}", val[i]));
        }

        rows.push(row);
    }

    let headers = if val_losses.is_some() {
        vec!["Epoch", "Train Loss", "Val Loss"]
    } else {
        vec!["Epoch", "Train Loss"]
    };

    html.push_str(&html_card(
        "Recent Epochs",
        &html_table(&headers, &rows),
        "#e74c3c",
    ));

    html.push_str("</div>");
    html
}

/// Generate HTML for Provenance visualization
pub fn provenance_html(
    rule_id: &Option<String>,
    source_file: &Option<String>,
    attributes: &HashMap<String, String>,
) -> String {
    let mut html = String::new();

    html.push_str("<div style='font-family: Arial, sans-serif;'>");
    html.push_str("<h2 style='color: #2c3e50; margin-bottom: 16px;'>🔍 Provenance</h2>");

    let mut items = vec![];

    if let Some(rid) = rule_id {
        items.push(("Rule ID".to_string(), rid.clone()));
    }

    if let Some(sf) = source_file {
        items.push(("Source File".to_string(), sf.clone()));
    }

    for (k, v) in attributes {
        items.push((k.clone(), v.clone()));
    }

    if !items.is_empty() {
        html.push_str(&html_card(
            "Provenance Information",
            &html_kv_list(&items),
            "#8e44ad",
        ));
    } else {
        html.push_str("<p style='color: #95a5a6;'>No provenance information available.</p>");
    }

    html.push_str("</div>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_table() {
        let headers = vec!["Name", "Age"];
        let rows = vec![
            vec!["Alice".to_string(), "30".to_string()],
            vec!["Bob".to_string(), "25".to_string()],
        ];

        let html = html_table(&headers, &rows);
        assert!(html.contains("<table"));
        assert!(html.contains("Alice"));
        assert!(html.contains("30"));
    }

    #[test]
    fn test_html_badge() {
        let badge = html_badge("Test", "#ff0000");
        assert!(badge.contains("Test"));
        assert!(badge.contains("#ff0000"));
    }

    #[test]
    fn test_einsum_graph_html() {
        let mut node_types = HashMap::new();
        node_types.insert("Input".to_string(), 2);
        node_types.insert("Einsum".to_string(), 5);

        let html = einsum_graph_html(7, 2, 1, &node_types);
        assert!(html.contains("EinsumGraph"));
        assert!(html.contains("7 nodes"));
        assert!(html.contains("Input"));
    }
}
