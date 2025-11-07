#!/usr/bin/env python3
"""
Knowledge Graph Reasoning with TensorLogic
==========================================

Comprehensive example demonstrating real-world knowledge graph reasoning
combining multiple TensorLogic features:

1. Domain modeling with symbol tables
2. Rule-based inference
3. Multiple compilation strategies
4. Provenance tracking
5. Performance optimization with backends

Use Case: Academic Collaboration Network
- Entities: Researchers, Papers, Topics
- Relations: authored, cites, collaborates_with
- Inference: Find potential collaborators based on shared interests

Author: TensorLogic Team
License: Apache-2.0
"""

import numpy as np
import pytensorlogic as tl

print("=" * 70)
print("KNOWLEDGE GRAPH REASONING WITH TENSORLOGIC")
print("=" * 70)
print()

# ============================================================================
# Part 1: Domain Modeling
# ============================================================================

print("1. DOMAIN MODELING")
print("-" * 70)

# Define domains for our academic network
table = tl.symbol_table()

# Create domains with meaningful descriptions
researchers_domain = tl.domain_info("Researcher", 100)
researchers_domain.set_description("Academic researchers in the network")
researchers_domain.set_elements([
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"
])
table.add_domain(researchers_domain)

papers_domain = tl.domain_info("Paper", 200)
papers_domain.set_description("Published research papers")
table.add_domain(papers_domain)

topics_domain = tl.domain_info("Topic", 50)
topics_domain.set_description("Research topics and areas")
topics_domain.set_elements([
    "ML", "NLP", "CV", "RL", "Theory", "Systems", "Security", "HCI"
])
table.add_domain(topics_domain)

print(f"✓ Created {len(table.list_domains())} domains:")
for domain in table.list_domains():
    info = table.get_domain(domain)
    print(f"  - {domain} (size={info.cardinality})")

print()

# Define predicate signatures
authored_pred = tl.predicate_info("authored", ["Researcher", "Paper"])
authored_pred.set_description("Researcher x authored paper y")
table.add_predicate(authored_pred)

cites_pred = tl.predicate_info("cites", ["Paper", "Paper"])
cites_pred.set_description("Paper x cites paper y")
table.add_predicate(cites_pred)

works_on_pred = tl.predicate_info("works_on", ["Researcher", "Topic"])
works_on_pred.set_description("Researcher x works on topic y")
table.add_predicate(works_on_pred)

collaborates_pred = tl.predicate_info("collaborates_with", ["Researcher", "Researcher"])
collaborates_pred.set_description("Researcher x collaborates with researcher y")
table.add_predicate(collaborates_pred)

print(f"✓ Created {len(table.list_predicates())} predicate signatures")
print()

# ============================================================================
# Part 2: Building Inference Rules
# ============================================================================

print("2. BUILDING INFERENCE RULES")
print("-" * 70)

# Rule 1: Co-authorship implies collaboration
# ∀ x,y,p. authored(x,p) ∧ authored(y,p) ∧ (x ≠ y) → collaborates_with(x,y)
x = tl.var("x")
y = tl.var("y")
p = tl.var("p")

authored_x = tl.pred("authored", [x, p])
authored_y = tl.pred("authored", [y, p])
collaborates = tl.pred("collaborates_with", [x, y])

# Simplified: if both authored same paper, they collaborate
rule1_body = tl.and_(authored_x, authored_y)
rule1 = tl.imply(rule1_body, collaborates)

print("Rule 1: Co-authorship → Collaboration")
print(f"  {rule1}")
print()

# Rule 2: Citation patterns indicate shared interests
# ∃ p1,p2. authored(x,p1) ∧ authored(y,p2) ∧ cites(p1,p2) → potential_collab(x,y)
p1 = tl.var("p1")
p2 = tl.var("p2")

authored_x_p1 = tl.pred("authored", [x, p1])
authored_y_p2 = tl.pred("authored", [y, p2])
cites_papers = tl.pred("cites", [p1, p2])

rule2_body = tl.and_(authored_x_p1, tl.and_(authored_y_p2, cites_papers))
potential_collab = tl.pred("potential_collaboration", [x, y])
rule2 = tl.imply(rule2_body, potential_collab)

print("Rule 2: Citation patterns → Potential collaboration")
print(f"  {rule2}")
print()

# Rule 3: Shared research topics suggest compatibility
# ∃ t. works_on(x,t) ∧ works_on(y,t) → shared_interest(x,y)
t = tl.var("t")

works_on_x = tl.pred("works_on", [x, t])
works_on_y = tl.pred("works_on", [y, t])

rule3_body = tl.and_(works_on_x, works_on_y)
shared_interest = tl.pred("shared_interest", [x, y])
rule3 = tl.imply(rule3_body, shared_interest)

print("Rule 3: Shared topics → Shared interest")
print(f"  {rule3}")
print()

# Combine all rules with AND
all_rules = tl.and_(rule1, tl.and_(rule2, rule3))

print("✓ Created 3 inference rules")
print()

# ============================================================================
# Part 3: Compilation with Different Strategies
# ============================================================================

print("3. COMPILATION STRATEGIES COMPARISON")
print("-" * 70)

# Strategy 1: Soft Differentiable (for gradient-based learning)
config_soft = tl.CompilationConfig.soft_differentiable()
graph_soft = tl.compile_with_config(rule1, config_soft)

print("Soft Differentiable:")
print(f"  - AND: element-wise multiplication")
print(f"  - OR: probabilistic sum")
print(f"  - Nodes: {graph_soft.num_nodes}")
print()

# Strategy 2: Hard Boolean (for discrete logic)
config_hard = tl.CompilationConfig.hard_boolean()
graph_hard = tl.compile_with_config(rule1, config_hard)

print("Hard Boolean:")
print(f"  - AND: minimum")
print(f"  - OR: maximum")
print(f"  - Nodes: {graph_hard.num_nodes}")
print()

# Strategy 3: Fuzzy Gödel (for fuzzy logic reasoning)
config_fuzzy = tl.CompilationConfig.fuzzy_godel()
graph_fuzzy = tl.compile_with_config(rule1, config_fuzzy)

print("Fuzzy Gödel:")
print(f"  - AND: minimum (Gödel t-norm)")
print(f"  - OR: maximum (Gödel t-conorm)")
print(f"  - Nodes: {graph_fuzzy.num_nodes}")
print()

print("✓ Compared 3 compilation strategies")
print()

# ============================================================================
# Part 4: Execution with Real Data
# ============================================================================

print("4. EXECUTION WITH SAMPLE DATA")
print("-" * 70)

# Create sample data (8 researchers)
num_researchers = 8
num_papers = 12

# Authorship matrix: researchers × papers
# 1.0 = authored, 0.0 = did not author
authored_data = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Alice: papers 0,1
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Bob: papers 0,2,3
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Charlie: papers 1,2,4
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Diana: papers 3,4,5
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Eve: papers 5,6,7
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # Frank: papers 6,8,9
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Grace: papers 7,8,10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Henry: papers 9,10,11
], dtype=np.float64)

print(f"Sample data: {num_researchers} researchers, {num_papers} papers")
print(f"Authorship matrix shape: {authored_data.shape}")
print()

# Compile simple co-authorship rule
x = tl.var("x")
y = tl.var("y")
p = tl.var("p")

authored_x = tl.pred("authored", [x, p])
authored_y = tl.pred("authored", [y, p])

# This will find pairs (x,y) who co-authored papers
coauthor_expr = tl.and_(authored_x, authored_y)

# Compile with soft differentiable strategy
config = tl.CompilationConfig.soft_differentiable()
graph = tl.compile_with_config(coauthor_expr, config)

print(f"Compiled graph:")
print(f"  - Nodes: {graph.num_nodes}")
print(f"  - Outputs: {graph.num_outputs}")
print()

# Execute to find co-authorships
try:
    # Note: This is a simplified example - real execution would need proper
    # einsum setup for the specific graph structure
    print("Note: Full execution requires proper tensor setup")
    print("      (See basic_usage.py for execution examples)")
except Exception as e:
    print(f"Execution note: {type(e).__name__}")

print()

# ============================================================================
# Part 5: Provenance Tracking
# ============================================================================

print("5. PROVENANCE TRACKING")
print("-" * 70)

# Create provenance tracker with RDF* support
tracker = tl.provenance_tracker(enable_rdfstar=True)

# Track entity mappings (researchers to tensor indices)
tracker.track_entity("http://example.org/Alice", 0)
tracker.track_entity("http://example.org/Bob", 1)
tracker.track_entity("http://example.org/Charlie", 2)
tracker.track_entity("http://example.org/Diana", 3)

print(f"✓ Tracked {len(tracker.get_entity_mappings())} entity mappings")

# Track inference rules (SHACL shapes) with node indices
tracker.track_shape(
    "http://example.org/CoAuthorshipRule",
    "authored(x,p) AND authored(y,p) -> collaborates_with(x,y)",
    0  # Node index in the graph
)

tracker.track_shape(
    "http://example.org/CitationRule",
    "cites(p1,p2) AND authored(x,p1) AND authored(y,p2) -> potential_collab(x,y)",
    1  # Node index in the graph
)

print(f"✓ Tracked {len(tracker.get_shape_mappings())} inference rules")
print()

# Track inferred triples with confidence scores
tracker.track_inferred_triple(
    subject="http://example.org/Alice",
    predicate="http://example.org/collaborates_with",
    object="http://example.org/Bob",
    confidence=0.95,
    rule_id="rule_1"
)

tracker.track_inferred_triple(
    subject="http://example.org/Bob",
    predicate="http://example.org/collaborates_with",
    object="http://example.org/Charlie",
    confidence=0.88,
    rule_id="rule_1"
)

tracker.track_inferred_triple(
    subject="http://example.org/Alice",
    predicate="http://example.org/potential_collaboration",
    object="http://example.org/Diana",
    confidence=0.72,
    rule_id="rule_2"
)

print("High-confidence inferences (>= 0.85):")
high_conf = tracker.get_high_confidence_inferences(0.85)
for triple_str in high_conf:
    print(f"  {triple_str}")

print()

# Export provenance to RDF* Turtle format
print("RDF* Turtle export (first 300 chars):")
rdf_export = tracker.to_rdfstar_turtle()
print(rdf_export[:300] + "...")
print()

# ============================================================================
# Part 6: Backend Performance Comparison
# ============================================================================

print("6. BACKEND SELECTION FOR PERFORMANCE")
print("-" * 70)

# List available backends
available = tl.list_available_backends()
print(f"Available backends: {len(available)}")
for backend_name, is_available in available.items():
    if is_available:
        # Convert string name to Backend enum
        backend_enum = getattr(tl.Backend, backend_name)
        caps = tl.get_backend_capabilities(backend_enum)
        print(f"  - {backend_name}")
        print(f"    Devices: {', '.join(caps.devices)}")
        print(f"    Features: {', '.join(caps.features)}")

print()

# Get default backend
default = tl.get_default_backend()
print(f"Default backend: {default}")
print()

# Get system info
sys_info = tl.get_system_info()
print("System Information:")
print(f"  TensorLogic Version: {sys_info['tensorlogic_version']}")
print(f"  Rust Version: {sys_info['rust_version']}")
print(f"  Default Backend: {sys_info['default_backend']}")
print(f"  CPU Devices: {', '.join(sys_info['cpu_capabilities']['devices'])}")
print()

# ============================================================================
# Part 7: JSON Persistence
# ============================================================================

print("7. SCHEMA PERSISTENCE")
print("-" * 70)

# Export symbol table to JSON
json_export = table.to_json()
print(f"Exported symbol table to JSON ({len(json_export)} chars)")
print(f"First 200 chars: {json_export[:200]}...")
print()

# Export provenance to JSON
prov_json = tracker.to_json()
print(f"Exported provenance to JSON ({len(prov_json)} chars)")
print()

# Can restore from JSON later
restored_table = tl.SymbolTable.from_json(json_export)
print(f"✓ Restored symbol table: {len(restored_table.list_domains())} domains")

restored_tracker = tl.ProvenanceTracker.from_json(prov_json)
print(f"✓ Restored provenance tracker: {len(restored_tracker.get_entity_mappings())} entities")
print()

# ============================================================================
# Summary and Best Practices
# ============================================================================

print("=" * 70)
print("SUMMARY & BEST PRACTICES")
print("=" * 70)
print()

print("✓ Demonstrated features:")
print("  1. Domain modeling with rich metadata")
print("  2. Multi-rule inference systems")
print("  3. Compilation strategy selection")
print("  4. Provenance tracking with RDF*")
print("  5. Backend performance optimization")
print("  6. JSON persistence for reproducibility")
print()

print("Best Practices:")
print("  • Use SymbolTable for schema management")
print("  • Choose compilation strategy based on use case:")
print("    - Soft differentiable: neural training")
print("    - Hard boolean: discrete reasoning")
print("    - Fuzzy: uncertainty handling")
print("  • Track provenance for auditability")
print("  • Select appropriate backend for performance")
print("  • Persist schemas and provenance to JSON")
print()

print("Real-World Applications:")
print("  • Knowledge graph completion")
print("  • Recommendation systems")
print("  • Scientific collaboration networks")
print("  • Citation analysis")
print("  • Expert finding")
print("  • Research trend prediction")
print()

print("=" * 70)
print("EXAMPLE COMPLETE!")
print("=" * 70)
print()

print("Next Steps:")
print("  1. Modify domains/rules for your use case")
print("  2. Load real data from your knowledge graph")
print("  3. Experiment with different compilation strategies")
print("  4. Export results to RDF for integration")
print("  5. Scale up with SIMD/GPU backends")
print()

print("For more examples, see:")
print("  - basic_usage.py: Core concepts")
print("  - provenance_tracking.py: Full provenance workflow")
print("  - backend_selection.py: Performance tuning")
print()
