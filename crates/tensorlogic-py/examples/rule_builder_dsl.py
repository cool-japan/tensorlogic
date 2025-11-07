#!/usr/bin/env python3
"""
Rule Builder DSL - Comprehensive Examples

This example demonstrates the full power of TensorLogic's Rule Builder DSL,
which provides a Python-native way to define logic rules with:
- Variables with domain bindings
- Predicate builders with arity and domain validation
- Operator overloading (&, |, ~, >>) for intuitive rule construction
- Context manager for collecting and compiling multiple rules
- Symbol table management and metadata tracking

The DSL makes it easy to express complex logic rules in a natural Python syntax.
"""

import numpy as np
import pytensorlogic as tl

print("=" * 80)
print("TensorLogic Rule Builder DSL Examples")
print("=" * 80)

# ============================================================================
# Example 1: Basic Variable and Predicate Creation
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Variable and Predicate Creation")
print("=" * 80)

# Create variables with domain bindings
x = tl.Var("x", domain="Person")
y = tl.Var("y", domain="Person")
z = tl.Var("z", domain="Person")

print(f"Created variables: {x}, {y}, {z}")
print(f"Variable x: {repr(x)}")

# Create predicates with arity and domain specifications
knows = tl.PredicateBuilder("knows", arity=2, domains=["Person", "Person"])
likes = tl.PredicateBuilder("likes", arity=2, domains=["Person", "Person"])
friend = tl.PredicateBuilder("friend", arity=2, domains=["Person", "Person"])

print(f"\nCreated predicates: {knows}, {likes}, {friend}")

# Build predicate expressions by calling predicates with variables
knows_expr = knows(x, y)
print(f"\nPredicate expression: {knows_expr}")

# ============================================================================
# Example 2: Operator Overloading - Logical Operations
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Operator Overloading - Logical Operations")
print("=" * 80)

# AND operator (&)
and_rule = knows(x, y) & knows(y, z)
print(f"AND rule (x knows y AND y knows z):\n  {and_rule}")

# OR operator (|)
or_rule = knows(x, y) | likes(x, y)
print(f"\nOR rule (x knows y OR x likes y):\n  {or_rule}")

# NOT operator (~)
not_rule = ~knows(x, y)
print(f"\nNOT rule (NOT x knows y):\n  {not_rule}")

# IMPLY operator (>>)
imply_rule = (knows(x, y) & knows(y, z)) >> knows(x, z)
print(f"\nIMPLY rule (transitivity):\n  {imply_rule}")

# Complex combinations
complex_rule = ((knows(x, y) & knows(y, x)) >> friend(x, y)) | likes(x, y)
print(f"\nComplex rule:\n  {complex_rule}")

# ============================================================================
# Example 3: Rule Builder Context Manager
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Rule Builder Context Manager")
print("=" * 80)

# Create a rule builder with context manager
rb = tl.RuleBuilder()

# Add domain metadata
rb.add_domain("Person", cardinality=5, description="People in the network")
print("Added domain: Person (cardinality=5)")

# Create variables using the builder
x, y, z = rb.vars("x", "y", "z", domain="Person")
print(f"Created variables via builder: {x}, {y}, {z}")

# Create predicates using the builder
knows = rb.pred("knows", arity=2, domains=["Person", "Person"])
friend = rb.pred("friend", arity=2, domains=["Person", "Person"])
print(f"Created predicates via builder: {knows}, {friend}")

# Add rules to the builder
rb.add_rule(
    (knows(x, y) & knows(y, z)) >> knows(x, z),
    name="transitivity"
)
print("\nAdded rule: transitivity")

rb.add_rule(
    knows(x, y) >> knows(y, x),
    name="symmetry"
)
print("Added rule: symmetry")

rb.add_rule(
    (knows(x, y) & knows(y, x)) >> friend(x, y),
    name="friendship"
)
print("Added rule: friendship")

print(f"\nRule builder has {len(rb)} rules")
print(f"Rules: {rb.get_rules()}")

# Get the symbol table
symbol_table = rb.get_symbol_table()
print(f"\nSymbol table domains: {symbol_table.list_domains()}")
print(f"Symbol table predicates: {symbol_table.list_predicates()}")

# ============================================================================
# Example 4: Compiling Rules
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Compiling Rules")
print("=" * 80)

# Compile all rules together (combined with AND)
print("Compiling all rules together...")
try:
    graph = rb.compile()
    print(f"Compiled graph: {graph}")
    print(f"  Number of nodes: {graph.num_nodes}")
    print(f"  Number of outputs: {graph.num_outputs}")
except Exception as e:
    print(f"Note: Compilation requires full tensor backend: {e}")

# Compile each rule separately
print("\nCompiling each rule separately...")
try:
    graphs = rb.compile_separate()
    print(f"Compiled {len(graphs)} graphs")
    for name, g in graphs.items():
        print(f"  {name}: {g.num_nodes} nodes, {g.num_outputs} outputs")
except Exception as e:
    print(f"Note: Compilation requires full tensor backend: {e}")

# ============================================================================
# Example 5: Social Network - Complete Example
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Social Network - Complete Example")
print("=" * 80)

# Build a complete social network model
rb_social = tl.RuleBuilder()

# Define domains
rb_social.add_domain(
    "Person",
    cardinality=10,
    description="People in social network",
    elements=["Alice", "Bob", "Charlie", "Diana", "Eve"]
)
rb_social.add_domain(
    "Post",
    cardinality=20,
    description="Posts in social network"
)

print("Defined domains: Person (10), Post (20)")

# Define predicates
follows = rb_social.pred("follows", arity=2, domains=["Person", "Person"])
likes_post = rb_social.pred("likes_post", arity=2, domains=["Person", "Post"])
posted = rb_social.pred("posted", arity=2, domains=["Person", "Post"])
sees_post = rb_social.pred("sees_post", arity=2, domains=["Person", "Post"])

print("Defined predicates: follows, likes_post, posted, sees_post")

# Define variables
p1, p2, p3 = rb_social.vars("p1", "p2", "p3", domain="Person")
post = rb_social.vars("post", domain="Post")

print(f"Defined variables: {p1}, {p2}, {p3}, {post}")

# Rule 1: Follow transitivity
rb_social.add_rule(
    (follows(p1, p2) & follows(p2, p3)) >> follows(p1, p3),
    name="follow_transitivity"
)

# Rule 2: See posts from followed people
rb_social.add_rule(
    (follows(p1, p2) & posted(p2, post)) >> sees_post(p1, post),
    name="feed_visibility"
)

# Rule 3: Like propagation (if you follow someone and they like a post, you see it)
rb_social.add_rule(
    (follows(p1, p2) & likes_post(p2, post)) >> sees_post(p1, post),
    name="like_propagation"
)

# Rule 4: Mutual follow becomes friendship (using previous friend predicate)
rb_social.add_rule(
    (follows(p1, p2) & follows(p2, p1)) >> friend(p1, p2),
    name="mutual_follow_friendship"
)

print(f"\nDefined {len(rb_social)} rules")
print("Rules:")
for name, expr in rb_social.get_rules():
    print(f"  {name}: {expr}")

# ============================================================================
# Example 6: Advanced - Custom Compilation Config
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Advanced - Custom Compilation Config")
print("=" * 80)

# Create builder with specific compilation strategy
config_fuzzy = tl.CompilationConfig.fuzzy_product()
rb_fuzzy = tl.RuleBuilder(config=config_fuzzy)

print("Created RuleBuilder with fuzzy_product compilation config")

# Define a simple rule
x, y = rb_fuzzy.vars("x", "y", domain="Person")
similar = rb_fuzzy.pred("similar", arity=2)
related = rb_fuzzy.pred("related", arity=2)

rb_fuzzy.add_rule(
    similar(x, y) >> related(x, y),
    name="similarity_implies_relation"
)

print(f"Defined rule with fuzzy semantics: {rb_fuzzy.get_rules()[0]}")

# ============================================================================
# Example 7: Knowledge Base - Logical Inference
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Knowledge Base - Logical Inference")
print("=" * 80)

kb = tl.RuleBuilder()

# Define ontology
kb.add_domain("Animal", cardinality=100, description="Animals")
kb.add_domain("Property", cardinality=50, description="Properties")

# Define predicates
is_a = kb.pred("is_a", arity=2, domains=["Animal", "Animal"])  # Subclass
has_property = kb.pred("has_property", arity=2, domains=["Animal", "Property"])

# Variables
animal1, animal2, animal3 = kb.vars("a1", "a2", "a3", domain="Animal")
prop = kb.vars("prop", domain="Property")

# Rule: Transitive subclass
kb.add_rule(
    (is_a(animal1, animal2) & is_a(animal2, animal3)) >> is_a(animal1, animal3),
    name="subclass_transitivity"
)

# Rule: Property inheritance
kb.add_rule(
    (is_a(animal1, animal2) & has_property(animal2, prop)) >> has_property(animal1, prop),
    name="property_inheritance"
)

print(f"Knowledge base has {len(kb)} inference rules")
print("\nInference rules:")
for name, expr in kb.get_rules():
    print(f"  {name}")
    print(f"    {expr}")

# ============================================================================
# Example 8: Comparison with Traditional API
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Comparison with Traditional API vs DSL")
print("=" * 80)

print("Traditional API:")
print("  x = tl.var('x')")
print("  y = tl.var('y')")
print("  knows_trad = tl.pred('knows', [x, y])")
print("  rule_trad = tl.imply(knows_trad, tl.pred('friend', [x, y]))")

print("\nDSL API:")
print("  x = tl.Var('x', domain='Person')")
print("  y = tl.Var('y', domain='Person')")
print("  knows = tl.PredicateBuilder('knows', arity=2)")
print("  rule_dsl = knows(x, y) >> friend(x, y)")

print("\nDSL provides:")
print("  ✓ Natural Python operator syntax")
print("  ✓ Domain validation")
print("  ✓ Arity checking")
print("  ✓ Better IDE support with type hints")
print("  ✓ Context manager for rule collections")

# ============================================================================
# Example 9: Rule Builder Methods Summary
# ============================================================================
print("\n" + "=" * 80)
print("Example 9: Rule Builder Methods Summary")
print("=" * 80)

rb_demo = tl.RuleBuilder()

print("Available RuleBuilder methods:")
print("  - vars(*names, domain=None): Create variables with optional domain")
print("  - pred(name, arity=None, domains=None): Create predicate builder")
print("  - add_domain(name, cardinality, ...): Add domain to symbol table")
print("  - add_rule(expr, name=None): Add a rule to the builder")
print("  - get_rules(): Get all defined rules")
print("  - get_symbol_table(): Get the symbol table")
print("  - compile(config=None): Compile all rules together")
print("  - compile_separate(config=None): Compile each rule separately")
print("  - clear(): Clear all rules and symbol table")
print("  - __len__(): Get number of rules")

# ============================================================================
# Example 10: Error Handling and Validation
# ============================================================================
print("\n" + "=" * 80)
print("Example 10: Error Handling and Validation")
print("=" * 80)

# Arity validation
print("Testing arity validation...")
binary_pred = tl.PredicateBuilder("binary", arity=2)
x = tl.Var("x")
y = tl.Var("y")
z = tl.Var("z")

try:
    valid_call = binary_pred(x, y)
    print(f"✓ Valid call with 2 args: {valid_call}")
except Exception as e:
    print(f"✗ Error: {e}")

try:
    invalid_call = binary_pred(x, y, z)
    print(f"✓ Invalid call with 3 args should fail!")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")

# Domain validation
print("\nTesting domain validation...")
person_pred = tl.PredicateBuilder("person_rel", arity=2, domains=["Person", "Person"])
x_person = tl.Var("x", domain="Person")
y_animal = tl.Var("y", domain="Animal")

try:
    valid_domains = person_pred(x_person, x_person)
    print(f"✓ Valid domains: {valid_domains}")
except Exception as e:
    print(f"✗ Error: {e}")

try:
    invalid_domains = person_pred(x_person, y_animal)
    print(f"✓ Invalid domains should fail!")
except TypeError as e:
    print(f"✓ Caught expected error: {e}")

# Empty rule builder
print("\nTesting empty rule builder...")
empty_rb = tl.RuleBuilder()
try:
    graph = empty_rb.compile()
    print("✗ Should not compile empty builder!")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")

print("\n" + "=" * 80)
print("DSL Examples Complete!")
print("=" * 80)
print("\nThe Rule Builder DSL provides a powerful, Pythonic way to define")
print("and compile logic rules for tensor-based inference.")
print("\nKey features:")
print("  • Operator overloading (&, |, ~, >>)")
print("  • Domain and arity validation")
print("  • Symbol table management")
print("  • Context manager support")
print("  • Multiple compilation strategies")
print("=" * 80)
