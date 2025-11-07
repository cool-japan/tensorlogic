//! Integration with tensorlogic-adapters SymbolTable.

use anyhow::Result;
use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};
use tensorlogic_ir::{PredicateSignature, SignatureRegistry, TypeAnnotation};

use crate::CompilerContext;

/// Synchronize CompilerContext with SymbolTable
pub fn sync_context_with_symbol_table(
    ctx: &mut CompilerContext,
    symbol_table: &SymbolTable,
) -> Result<()> {
    // Import domains from symbol table
    for (name, domain_info) in &symbol_table.domains {
        ctx.add_domain(name, domain_info.cardinality);
    }

    // Import variable bindings
    for (var, domain) in &symbol_table.variables {
        ctx.bind_var(var, domain)?;
    }

    Ok(())
}

/// Build a SignatureRegistry from SymbolTable
pub fn build_signature_registry(symbol_table: &SymbolTable) -> SignatureRegistry {
    let mut registry = SignatureRegistry::new();

    for (_name, pred_info) in &symbol_table.predicates {
        let signature = predicate_info_to_signature(pred_info);
        registry.register(signature);
    }

    registry
}

/// Convert PredicateInfo to PredicateSignature
fn predicate_info_to_signature(pred_info: &PredicateInfo) -> PredicateSignature {
    let arg_types: Vec<TypeAnnotation> = pred_info
        .arg_domains
        .iter()
        .map(|domain| TypeAnnotation::new(domain.clone()))
        .collect();

    PredicateSignature::new(&pred_info.name, arg_types)
}

/// Convert SymbolTable domains to CompilerContext
pub fn import_domains(ctx: &mut CompilerContext, symbol_table: &SymbolTable) -> Result<()> {
    for (name, domain_info) in &symbol_table.domains {
        ctx.add_domain(name, domain_info.cardinality);
    }
    Ok(())
}

/// Export CompilerContext domains back to SymbolTable
pub fn export_domains(ctx: &CompilerContext, symbol_table: &mut SymbolTable) -> Result<()> {
    for (name, domain) in &ctx.domains {
        if !symbol_table.domains.contains_key(name) {
            symbol_table.add_domain(DomainInfo::new(name.clone(), domain.cardinality))?;
        }
    }
    Ok(())
}

/// Create a PredicateInfo from arguments
pub fn create_predicate_info(name: impl Into<String>, arg_domains: Vec<String>) -> PredicateInfo {
    PredicateInfo::new(name.into(), arg_domains)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_context_with_symbol_table() {
        let mut symbol_table = SymbolTable::new();
        symbol_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        symbol_table.bind_variable("x", "Person").unwrap();

        let mut ctx = CompilerContext::new();
        sync_context_with_symbol_table(&mut ctx, &symbol_table).unwrap();

        assert!(ctx.domains.contains_key("Person"));
        assert_eq!(ctx.domains["Person"].cardinality, 100);
        assert_eq!(ctx.var_to_domain.get("x"), Some(&"Person".to_string()));
    }

    #[test]
    fn test_build_signature_registry() {
        let mut symbol_table = SymbolTable::new();
        symbol_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let pred_info =
            PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        symbol_table.add_predicate(pred_info).unwrap();

        let registry = build_signature_registry(&symbol_table);

        let sig = registry.get("knows").unwrap();
        assert_eq!(sig.arity, 2);
        assert_eq!(sig.arg_types.len(), 2);
        assert_eq!(sig.arg_types[0].type_name, "Person");
    }

    #[test]
    fn test_predicate_info_to_signature() {
        let pred_info =
            PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);

        let sig = predicate_info_to_signature(&pred_info);
        assert_eq!(sig.name, "knows");
        assert_eq!(sig.arity, 2);
        assert_eq!(sig.arg_types[0].type_name, "Person");
    }

    #[test]
    fn test_import_domains() {
        let mut symbol_table = SymbolTable::new();
        symbol_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        symbol_table
            .add_domain(DomainInfo::new("Thing", 50))
            .unwrap();

        let mut ctx = CompilerContext::new();
        import_domains(&mut ctx, &symbol_table).unwrap();

        assert_eq!(ctx.domains.len(), 2);
        assert_eq!(ctx.domains["Person"].cardinality, 100);
        assert_eq!(ctx.domains["Thing"].cardinality, 50);
    }

    #[test]
    fn test_export_domains() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        ctx.add_domain("Thing", 50);

        let mut symbol_table = SymbolTable::new();
        export_domains(&ctx, &mut symbol_table).unwrap();

        assert_eq!(symbol_table.domains.len(), 2);
        assert_eq!(symbol_table.domains["Person"].cardinality, 100);
        assert_eq!(symbol_table.domains["Thing"].cardinality, 50);
    }

    #[test]
    fn test_create_predicate_info() {
        let pred_info =
            create_predicate_info("knows", vec!["Person".to_string(), "Person".to_string()]);

        assert_eq!(pred_info.name, "knows");
        assert_eq!(pred_info.arg_domains.len(), 2);
        assert_eq!(pred_info.arg_domains[0], "Person");
    }

    #[test]
    fn test_round_trip_domains() {
        let mut ctx1 = CompilerContext::new();
        ctx1.add_domain("Person", 100);

        let mut symbol_table = SymbolTable::new();
        export_domains(&ctx1, &mut symbol_table).unwrap();

        let mut ctx2 = CompilerContext::new();
        import_domains(&mut ctx2, &symbol_table).unwrap();

        assert_eq!(ctx2.domains["Person"].cardinality, 100);
    }
}
