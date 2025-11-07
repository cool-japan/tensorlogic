"""
Model Persistence Example for TensorLogic

This example demonstrates how to save and load TensorLogic models
in various formats:
- JSON format (human-readable, cross-platform)
- Binary format (compact, efficient)
- Pickle support (Python-native)
- Full model packages with metadata, symbol tables, and configurations

Run with:
    python examples/model_persistence.py
"""

import os
import pickle
import tempfile
import numpy as np
import pytensorlogic as tl


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def scenario_1_basic_save_load():
    """Scenario 1: Basic model save and load in JSON format."""
    print_section("Scenario 1: Basic Save/Load (JSON)")

    # Create a simple logical rule
    print("Creating a simple rule: knows(x, y) AND likes(y, z)")
    expr = tl.and_(
        tl.pred("knows", [tl.var("x"), tl.var("y")]),
        tl.pred("likes", [tl.var("y"), tl.var("z")])
    )

    # Compile to tensor graph
    print("Compiling to tensor graph...")
    graph = tl.compile(expr)
    print(f"Graph has {len(graph.nodes)} nodes")

    # Save to JSON
    temp_path = tempfile.mktemp(suffix=".json")
    print(f"Saving to: {temp_path}")
    tl.save_model(graph, temp_path, format="json")
    print(f"Saved! File size: {os.path.getsize(temp_path)} bytes")

    # Load from JSON
    print("Loading from JSON...")
    loaded_graph = tl.load_model(temp_path, format="json")
    print(f"Loaded graph has {len(loaded_graph.nodes)} nodes")
    print("✓ Save/load successful!")

    # Cleanup
    os.remove(temp_path)


def scenario_2_binary_format():
    """Scenario 2: Saving in binary format for efficiency."""
    print_section("Scenario 2: Binary Format (Compact)")

    # Create a more complex rule
    print("Creating rule: ∃x. (employee(x) OR contractor(x)) AND active(x)")
    expr = tl.exists(
        "x",
        tl.and_(
            tl.or_(
                tl.pred("employee", [tl.var("x")]),
                tl.pred("contractor", [tl.var("x")])
            ),
            tl.pred("active", [tl.var("x")])
        )
    )

    graph = tl.compile(expr)

    # Save in JSON
    json_path = tempfile.mktemp(suffix=".json")
    tl.save_model(graph, json_path, format="json")
    json_size = os.path.getsize(json_path)

    # Save in binary
    bin_path = tempfile.mktemp(suffix=".bin")
    tl.save_model(graph, bin_path, format="binary")
    bin_size = os.path.getsize(bin_path)

    print(f"JSON format size: {json_size} bytes")
    print(f"Binary format size: {bin_size} bytes")
    print(f"Compression ratio: {json_size / bin_size:.2f}x")

    # Verify both can be loaded
    loaded_json = tl.load_model(json_path, format="json")
    loaded_bin = tl.load_model(bin_path, format="binary")

    print(f"JSON graph nodes: {len(loaded_json.nodes)}")
    print(f"Binary graph nodes: {len(loaded_bin.nodes)}")
    print("✓ Both formats work correctly!")

    # Cleanup
    os.remove(json_path)
    os.remove(bin_path)


def scenario_3_full_model_with_metadata():
    """Scenario 3: Saving a full model with symbol table and metadata."""
    print_section("Scenario 3: Full Model Package")

    # Create symbol table
    print("Creating symbol table...")
    sym_table = tl.symbol_table()
    sym_table.add_domain("Person", 100)
    sym_table.add_domain("Item", 50)
    sym_table.add_predicate("owns", ["Person", "Item"])
    sym_table.add_predicate("likes", ["Person", "Item"])

    # Bind variables
    sym_table.bind_variable("x", "Person")
    sym_table.bind_variable("y", "Item")

    print(f"  Domains: {sym_table.list_domains()}")
    print(f"  Predicates: {sym_table.list_predicates()}")

    # Create rule
    print("\nCreating rule: owns(x, y) => likes(x, y)")
    expr = tl.imply(
        tl.pred("owns", [tl.var("x"), tl.var("y")]),
        tl.pred("likes", [tl.var("x"), tl.var("y")])
    )
    graph = tl.compile(expr)

    # Save full model
    temp_path = tempfile.mktemp(suffix=".json")
    print(f"\nSaving full model to: {temp_path}")

    tl.save_full_model(
        graph,
        temp_path,
        symbol_table=sym_table,
        metadata={
            "model_name": "ownership_implies_preference",
            "author": "TensorLogic Demo",
            "version": "1.0.0",
            "description": "If someone owns an item, they likely like it",
            "domain": "preference_modeling"
        },
        format="json"
    )

    print(f"Saved! File size: {os.path.getsize(temp_path)} bytes")

    # Load full model
    print("\nLoading full model...")
    model = tl.load_full_model(temp_path)

    print(f"Model components: {list(model.keys())}")

    # Access components
    loaded_graph = model["graph"]
    loaded_st = model["symbol_table"]
    metadata = model["metadata"]

    print(f"\nGraph: {len(loaded_graph.nodes)} nodes")
    print(f"Symbol table: {len(loaded_st.list_domains())} domains, {len(loaded_st.list_predicates())} predicates")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\n✓ Full model saved and loaded successfully!")

    # Cleanup
    os.remove(temp_path)


def scenario_4_model_package_manipulation():
    """Scenario 4: Creating and manipulating ModelPackage directly."""
    print_section("Scenario 4: ModelPackage Direct Manipulation")

    # Create package
    print("Creating model package...")
    package = tl.model_package()

    # Add metadata
    print("Adding metadata...")
    package.add_metadata("project", "social_network_analysis")
    package.add_metadata("researcher", "Dr. Alice Smith")
    package.add_metadata("institution", "TensorLogic Labs")
    package.add_metadata("date", "2025-11-07")

    print(f"\nPackage: {package}")
    print("\nMetadata:")
    print(f"  Project: {package.get_metadata('project')}")
    print(f"  Researcher: {package.get_metadata('researcher')}")
    print(f"  Institution: {package.get_metadata('institution')}")
    print(f"  Date: {package.get_metadata('date')}")

    # Save package
    temp_path = tempfile.mktemp(suffix=".json")
    print(f"\nSaving package to: {temp_path}")
    package.save_json(temp_path)

    # Load package
    print("Loading package...")
    loaded_package = tl.ModelPackage.load_json(temp_path)

    print(f"Loaded package: {loaded_package}")
    print(f"Researcher: {loaded_package.get_metadata('researcher')}")

    print("\n✓ ModelPackage manipulation successful!")

    # Cleanup
    os.remove(temp_path)


def scenario_5_json_serialization():
    """Scenario 5: JSON string serialization."""
    print_section("Scenario 5: JSON String Serialization")

    # Create package
    package = tl.model_package()
    package.add_metadata("format", "json_string")
    package.add_metadata("test", "serialization")

    # Serialize to JSON string
    print("Serializing to JSON string...")
    json_str = package.to_json()
    print(f"JSON string length: {len(json_str)} characters")
    print(f"\nFirst 200 characters:\n{json_str[:200]}...")

    # Deserialize from JSON string
    print("\nDeserializing from JSON string...")
    restored_package = tl.ModelPackage.from_json(json_str)

    print(f"Format: {restored_package.get_metadata('format')}")
    print(f"Test: {restored_package.get_metadata('test')}")

    print("\n✓ JSON string serialization successful!")


def scenario_6_pickle_support():
    """Scenario 6: Python pickle support."""
    print_section("Scenario 6: Pickle Support")

    # Create package
    print("Creating model package...")
    package = tl.model_package()
    package.add_metadata("serialization", "pickle")
    package.add_metadata("python_version", "3.9+")

    # Pickle
    print("Pickling package...")
    pickled = pickle.dumps(package)
    print(f"Pickled size: {len(pickled)} bytes")

    # Unpickle
    print("Unpickling package...")
    unpickled = pickle.loads(pickled)

    print(f"Serialization: {unpickled.get_metadata('serialization')}")
    print(f"Python version: {unpickled.get_metadata('python_version')}")

    print("\n✓ Pickle support verified!")


def scenario_7_auto_format_detection():
    """Scenario 7: Automatic format detection from file extension."""
    print_section("Scenario 7: Auto Format Detection")

    # Create a simple rule
    expr = tl.not_(tl.pred("inactive", [tl.var("user")]))
    graph = tl.compile(expr)

    # Save with .json extension (auto-detect format)
    json_path = tempfile.mktemp(suffix=".json")
    print(f"Saving to {json_path} (format auto-detected as JSON)")
    tl.save_model(graph, json_path, format="json")
    loaded_json = tl.load_model(json_path)  # Format auto-detected
    print("✓ JSON auto-detection works!")

    # Save with .bin extension (auto-detect format)
    bin_path = tempfile.mktemp(suffix=".bin")
    print(f"Saving to {bin_path} (format auto-detected as binary)")
    tl.save_model(graph, bin_path, format="binary")
    loaded_bin = tl.load_model(bin_path)  # Format auto-detected
    print("✓ Binary auto-detection works!")

    # Cleanup
    os.remove(json_path)
    os.remove(bin_path)


def scenario_8_transitive_closure_model():
    """Scenario 8: Real-world example - transitive closure model."""
    print_section("Scenario 8: Transitive Closure Model")

    print("Building transitive closure model:")
    print("  Rule: knows(x,y) ∧ knows(y,z) → knows(x,z)")

    # Build expression
    x, y, z = tl.var("x"), tl.var("y"), tl.var("z")
    expr = tl.imply(
        tl.and_(
            tl.pred("knows", [x, y]),
            tl.pred("knows", [y, z])
        ),
        tl.pred("knows", [x, z])
    )

    # Compile
    graph = tl.compile(expr)

    # Create symbol table
    sym_table = tl.symbol_table()
    sym_table.add_domain("Person", 50)
    sym_table.add_predicate("knows", ["Person", "Person"])
    sym_table.bind_variable("x", "Person")
    sym_table.bind_variable("y", "Person")
    sym_table.bind_variable("z", "Person")

    # Save complete model
    temp_path = tempfile.mktemp(suffix=".json")
    print(f"\nSaving to: {temp_path}")

    tl.save_full_model(
        graph,
        temp_path,
        symbol_table=sym_table,
        metadata={
            "model": "transitive_closure",
            "relation": "knows",
            "domain": "social_network",
            "algorithm": "einsum_compilation",
            "use_case": "friend_of_friend_inference"
        }
    )

    file_size = os.path.getsize(temp_path)
    print(f"Model saved! Size: {file_size} bytes")

    # Load and verify
    print("\nLoading model...")
    model = tl.load_full_model(temp_path)

    print(f"✓ Graph loaded with {len(model['graph'].nodes)} nodes")
    print(f"✓ Symbol table with {len(model['symbol_table'].list_domains())} domains")
    print(f"✓ Metadata:")
    for key in ["model", "relation", "use_case"]:
        print(f"    {key}: {model['metadata'][key]}")

    # Cleanup
    os.remove(temp_path)


def scenario_9_binary_vs_json_comparison():
    """Scenario 9: Performance comparison between formats."""
    print_section("Scenario 9: Format Performance Comparison")

    # Create a complex multi-rule model
    print("Creating complex multi-rule model...")

    rules = []
    for i in range(10):
        rule = tl.and_(
            tl.pred(f"pred_{i}", [tl.var("x")]),
            tl.pred(f"pred_{i+1}", [tl.var("y")])
        )
        rules.append(rule)

    # Combine all rules
    combined = rules[0]
    for rule in rules[1:]:
        combined = tl.or_(combined, rule)

    graph = tl.compile(combined)
    print(f"Graph complexity: {len(graph.nodes)} nodes")

    # Measure JSON
    json_path = tempfile.mktemp(suffix=".json")
    tl.save_model(graph, json_path, format="json")
    json_size = os.path.getsize(json_path)

    # Measure binary
    bin_path = tempfile.mktemp(suffix=".bin")
    tl.save_model(graph, bin_path, format="binary")
    bin_size = os.path.getsize(bin_path)

    print(f"\nFormat comparison:")
    print(f"  JSON:   {json_size:,} bytes")
    print(f"  Binary: {bin_size:,} bytes")
    print(f"  Ratio:  {json_size / bin_size:.2f}x")

    savings = json_size - bin_size
    print(f"  Space saved: {savings:,} bytes ({100 * savings / json_size:.1f}%)")

    # Cleanup
    os.remove(json_path)
    os.remove(bin_path)


def scenario_10_model_versioning():
    """Scenario 10: Model versioning with metadata."""
    print_section("Scenario 10: Model Versioning")

    print("Creating versioned models...")

    # Version 1.0
    expr_v1 = tl.pred("active", [tl.var("user")])
    graph_v1 = tl.compile(expr_v1)

    path_v1 = tempfile.mktemp(suffix="_v1.0.json")
    tl.save_full_model(
        graph_v1,
        path_v1,
        metadata={
            "version": "1.0",
            "release_date": "2025-01-01",
            "features": "basic_active_check",
            "status": "deprecated"
        }
    )
    print(f"✓ Version 1.0 saved to {path_v1}")

    # Version 2.0 (improved)
    expr_v2 = tl.and_(
        tl.pred("active", [tl.var("user")]),
        tl.not_(tl.pred("suspended", [tl.var("user")]))
    )
    graph_v2 = tl.compile(expr_v2)

    path_v2 = tempfile.mktemp(suffix="_v2.0.json")
    tl.save_full_model(
        graph_v2,
        path_v2,
        metadata={
            "version": "2.0",
            "release_date": "2025-06-01",
            "features": "active_check_with_suspension",
            "status": "stable"
        }
    )
    print(f"✓ Version 2.0 saved to {path_v2}")

    # Load and compare
    model_v1 = tl.load_full_model(path_v1)
    model_v2 = tl.load_full_model(path_v2)

    print(f"\nVersion comparison:")
    print(f"  V1.0: {len(model_v1['graph'].nodes)} nodes, {model_v1['metadata']['features']}")
    print(f"  V2.0: {len(model_v2['graph'].nodes)} nodes, {model_v2['metadata']['features']}")
    print(f"  Improvement: {len(model_v2['graph'].nodes) - len(model_v1['graph'].nodes)} additional nodes")

    # Cleanup
    os.remove(path_v1)
    os.remove(path_v2)


def main():
    """Run all persistence scenarios."""
    print("\n" + "=" * 70)
    print("  TensorLogic Model Persistence Examples")
    print("  Demonstrating save/load functionality in multiple formats")
    print("=" * 70)

    try:
        scenario_1_basic_save_load()
        scenario_2_binary_format()
        scenario_3_full_model_with_metadata()
        scenario_4_model_package_manipulation()
        scenario_5_json_serialization()
        scenario_6_pickle_support()
        scenario_7_auto_format_detection()
        scenario_8_transitive_closure_model()
        scenario_9_binary_vs_json_comparison()
        scenario_10_model_versioning()

        print_section("Summary")
        print("✓ All 10 persistence scenarios completed successfully!")
        print("\nKey features demonstrated:")
        print("  • JSON format (human-readable)")
        print("  • Binary format (compact)")
        print("  • Pickle support (Python-native)")
        print("  • Full model packages with metadata")
        print("  • Symbol table persistence")
        print("  • Auto format detection")
        print("  • Model versioning")
        print("  • Performance comparisons")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
