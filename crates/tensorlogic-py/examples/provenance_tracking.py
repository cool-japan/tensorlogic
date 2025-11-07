#!/usr/bin/env python3
"""
Provenance Tracking Example

This example demonstrates comprehensive provenance tracking capabilities in TensorLogic,
including:
1. Source location tracking
2. Provenance metadata for logical rules
3. RDF entity and tensor computation mappings
4. RDF* statement-level provenance
5. Metadata extraction from compiled graphs
"""

import json
import numpy as np

try:
    import pytensorlogic as tl
except ImportError:
    print("Error: pytensorlogic not installed. Run 'maturin develop' first.")
    exit(1)


def main():
    print("=" * 70)
    print("TENSORLOGIC PROVENANCE TRACKING DEMO")
    print("=" * 70)
    print()

    # =================================================================
    # 1. SOURCE LOCATION TRACKING
    # =================================================================
    print("1. SOURCE LOCATION TRACKING")
    print("-" * 70)

    # Create source locations
    rule_start = tl.SourceLocation("social_rules.tl", 10, 1)
    rule_end = tl.SourceLocation("social_rules.tl", 15, 40)
    rule_span = tl.SourceSpan(rule_start, rule_end)

    print(f"Rule location: {rule_start}")
    print(f"Rule span: {rule_span}")
    print()

    # =================================================================
    # 2. PROVENANCE METADATA
    # =================================================================
    print("2. PROVENANCE METADATA FOR LOGICAL RULES")
    print("-" * 70)

    # Create provenance for a rule
    prov = tl.Provenance()
    prov.set_rule_id("social_network_rule_1")
    prov.set_source_file("social_rules.tl")
    prov.set_span(rule_span)
    prov.add_attribute("author", "alice")
    prov.add_attribute("version", "1.0")
    prov.add_attribute("category", "social_reasoning")

    print(f"Rule ID: {prov.rule_id}")
    print(f"Source file: {prov.source_file}")
    print(f"Author: {prov.get_attribute('author')}")
    print(f"Version: {prov.get_attribute('version')}")
    print(f"Category: {prov.get_attribute('category')}")
    print()

    # Get all attributes
    attrs = prov.get_attributes()
    print("All attributes:")
    for key, value in attrs.items():
        print(f"  {key}: {value}")
    print()

    # =================================================================
    # 3. PROVENANCE TRACKER - ENTITY MAPPINGS
    # =================================================================
    print("3. RDF ENTITY TO TENSOR MAPPINGS")
    print("-" * 70)

    # Create provenance tracker
    tracker = tl.ProvenanceTracker()

    # Track RDF entities to tensor indices
    tracker.track_entity("http://example.org/alice", 0)
    tracker.track_entity("http://example.org/bob", 1)
    tracker.track_entity("http://example.org/charlie", 2)
    tracker.track_entity("http://example.org/dave", 3)

    print("Entity-to-tensor mappings:")
    for entity, tensor_idx in tracker.get_entity_mappings().items():
        print(f"  {entity} → tensor[{tensor_idx}]")
    print()

    # Reverse lookup: tensor to entity
    print("Tensor-to-entity lookups:")
    for i in range(4):
        entity = tracker.get_entity(i)
        print(f"  tensor[{i}] → {entity}")
    print()

    # =================================================================
    # 4. SHACL SHAPE MAPPINGS
    # =================================================================
    print("4. SHACL SHAPE TO RULE MAPPINGS")
    print("-" * 70)

    # Track SHACL shapes to logical rules
    tracker.track_shape(
        "http://example.org/PersonShape",
        "Person(x)",
        0
    )

    tracker.track_shape(
        "http://example.org/KnowsShape",
        "knows(x, y) AND Person(x) AND Person(y)",
        1
    )

    tracker.track_shape(
        "http://example.org/FriendshipShape",
        "FORALL x, y. knows(x, y) => knows(y, x)",
        2
    )

    print("Shape-to-rule mappings:")
    for shape, rule in tracker.get_shape_mappings().items():
        print(f"  {shape}")
        print(f"    → {rule}")
    print()

    # =================================================================
    # 5. RDF* STATEMENT-LEVEL PROVENANCE
    # =================================================================
    print("5. RDF* STATEMENT-LEVEL PROVENANCE")
    print("-" * 70)

    # Create tracker with RDF* support
    rdfstar_tracker = tl.ProvenanceTracker(enable_rdfstar=True)

    # Track inferred triples with metadata
    rdfstar_tracker.track_inferred_triple(
        subject="http://example.org/alice",
        predicate="http://example.org/knows",
        object="http://example.org/bob",
        rule_id="rule_1",
        confidence=0.95
    )

    rdfstar_tracker.track_inferred_triple(
        subject="http://example.org/bob",
        predicate="http://example.org/knows",
        object="http://example.org/charlie",
        rule_id="rule_1",
        confidence=0.88
    )

    rdfstar_tracker.track_inferred_triple(
        subject="http://example.org/charlie",
        predicate="http://example.org/knows",
        object="http://example.org/dave",
        rule_id="rule_2",
        confidence=0.72
    )

    # Get high-confidence inferences (>= 0.85)
    print("High-confidence inferences (confidence >= 0.85):")
    high_conf = rdfstar_tracker.get_high_confidence_inferences(min_confidence=0.85)
    for inf in high_conf:
        print(f"  {inf['subject']}")
        print(f"    {inf['predicate']}")
        print(f"      {inf['object']}")
        print(f"    Confidence: {inf.get('confidence', 'N/A')}")
        print(f"    Rule: {inf.get('rule_id', 'N/A')}")
        print()

    # =================================================================
    # 6. RDF* EXPORT TO TURTLE
    # =================================================================
    print("6. EXPORT TO RDF* TURTLE FORMAT")
    print("-" * 70)

    # Export to Turtle format
    turtle_output = rdfstar_tracker.to_rdfstar_turtle()
    print("RDF* Turtle output:")
    print(turtle_output[:500] + "..." if len(turtle_output) > 500 else turtle_output)
    print()

    # =================================================================
    # 7. JSON SERIALIZATION
    # =================================================================
    print("7. JSON SERIALIZATION AND DESERIALIZATION")
    print("-" * 70)

    # Export tracker to JSON
    json_output = tracker.to_json()
    print("JSON export (first 300 chars):")
    print(json_output[:300] + "...")
    print()

    # Parse and pretty-print
    data = json.loads(json_output)
    print(f"Entities tracked: {len(data['entity_to_tensor'])}")
    print(f"Shapes tracked: {len(data['shape_to_rule'])}")
    print()

    # Import from JSON
    restored_tracker = tl.ProvenanceTracker.from_json(json_output)
    print("Restored from JSON:")
    print(f"  Entity count: {len(restored_tracker.get_entity_mappings())}")
    print(f"  Shape count: {len(restored_tracker.get_shape_mappings())}")
    print()

    # =================================================================
    # 8. COMPILED GRAPH PROVENANCE
    # =================================================================
    print("8. PROVENANCE FROM COMPILED GRAPHS")
    print("-" * 70)

    # Create a logical expression
    person_x = tl.pred("Person", [tl.var("x")])
    person_y = tl.pred("Person", [tl.var("y")])
    knows = tl.pred("knows", [tl.var("x"), tl.var("y")])

    # Create rule: Person(x) AND Person(y) AND knows(x, y)
    rule = tl.and_(tl.and_(person_x, person_y), knows)

    # Compile to graph
    graph = tl.compile(rule)

    print("Compiled graph statistics:")
    stats = graph.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Get provenance from graph
    provenance_list = tl.get_provenance(graph)
    print(f"Provenance entries: {len(provenance_list)}")
    non_none_count = sum(1 for p in provenance_list if p is not None)
    print(f"Non-null provenance entries: {non_none_count}")
    print()

    # Get metadata from graph
    metadata_list = tl.get_metadata(graph)
    print(f"Metadata entries: {len(metadata_list)}")
    for idx, meta in enumerate(metadata_list):
        if meta is not None:
            print(f"  Node {idx}:")
            if "name" in meta and meta["name"]:
                print(f"    Name: {meta['name']}")
            if "provenance" in meta:
                prov_obj = meta["provenance"]
                if prov_obj and prov_obj.rule_id:
                    print(f"    Rule ID: {prov_obj.rule_id}")
    print()

    # =================================================================
    # 9. PRACTICAL USE CASE: SOCIAL NETWORK REASONING
    # =================================================================
    print("9. PRACTICAL USE CASE: SOCIAL NETWORK REASONING")
    print("-" * 70)

    # Create a provenance-aware social network analysis
    social_tracker = tl.ProvenanceTracker(enable_rdfstar=True)

    # Define our entities
    people = {
        "alice": 0,
        "bob": 1,
        "charlie": 2,
        "dave": 3,
        "eve": 4
    }

    # Track entities
    for name, idx in people.items():
        social_tracker.track_entity(f"http://social.net/{name}", idx)

    # Track inferred relationships with provenance
    relationships = [
        ("alice", "bob", "transitive_closure", 0.95),
        ("alice", "charlie", "transitive_closure", 0.90),
        ("bob", "dave", "friend_of_friend", 0.85),
        ("charlie", "eve", "shared_interests", 0.80),
        ("dave", "eve", "transitive_closure", 0.75),
    ]

    for subj, obj, rule, conf in relationships:
        social_tracker.track_inferred_triple(
            subject=f"http://social.net/{subj}",
            predicate="http://social.net/knows",
            object=f"http://social.net/{obj}",
            rule_id=rule,
            confidence=conf
        )

    print("Social network analysis:")
    print(f"  People: {len(people)}")
    print(f"  Relationships: {len(relationships)}")
    print()

    # Find strong relationships (confidence >= 0.88)
    strong_rels = social_tracker.get_high_confidence_inferences(min_confidence=0.88)
    print(f"Strong relationships (confidence >= 0.88): {len(strong_rels)}")
    for rel in strong_rels:
        subj = rel['subject'].split('/')[-1]
        obj = rel['object'].split('/')[-1]
        conf = rel.get('confidence', 0.0)
        rule = rel.get('rule_id', 'unknown')
        print(f"  {subj} → {obj} (confidence: {conf:.2f}, rule: {rule})")
    print()

    # Export for auditing
    audit_export = social_tracker.to_json()
    print("Exported for audit (size):", len(audit_export), "bytes")
    print()

    # =================================================================
    # 10. HELPER FUNCTION
    # =================================================================
    print("10. PROVENANCE TRACKER HELPER FUNCTION")
    print("-" * 70)

    # Create tracker using helper function
    helper_tracker = tl.provenance_tracker(enable_rdfstar=True)
    helper_tracker.track_entity("http://example.org/entity1", 0)

    print("Tracker created via helper function")
    print(f"  Entities: {len(helper_tracker.get_entity_mappings())}")
    print()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✅ Source location tracking with file, line, column")
    print("✅ Provenance metadata with rule IDs, source files, spans")
    print("✅ Custom attributes for provenance records")
    print("✅ RDF entity to tensor index mappings")
    print("✅ SHACL shape to rule expression mappings")
    print("✅ RDF* statement-level provenance with confidence scores")
    print("✅ High-confidence inference filtering")
    print("✅ RDF* Turtle export for interoperability")
    print("✅ JSON serialization and deserialization")
    print("✅ Provenance extraction from compiled graphs")
    print("✅ Metadata extraction with attributes")
    print("✅ Practical social network reasoning with audit trail")
    print()
    print("All provenance tracking features demonstrated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
