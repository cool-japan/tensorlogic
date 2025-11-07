#!/usr/bin/env python3
"""
TensorLogic Python Bindings - Basic Usage Examples

This file demonstrates the core features of pytensorlogic Python bindings.

Requirements:
    - pytensorlogic (built with maturin)
    - numpy

To build and run:
    $ cd crates/pytensorlogic
    $ maturin develop
    $ python examples/basic_usage.py
"""

import numpy as np

# Note: This example assumes pytensorlogic has been built with maturin
# Uncomment the following line after building:
# import pytensorlogic as tl

print("=" * 70)
print("TensorLogic Python Bindings - Examples")
print("=" * 70)

# Example 1: Basic Predicate
print("\n[Example 1] Basic Predicate")
print("-" * 70)

# Create variables
# x = tl.var("x")
# y = tl.var("y")

# Create a predicate: knows(x, y)
# knows = tl.pred("knows", [x, y])

# Compile to tensor graph
# graph = tl.compile(knows)
# print(f"Graph nodes: {graph.num_nodes}")
# print(f"Graph outputs: {graph.num_outputs}")

# Create sample data: 100x100 adjacency matrix
# knows_matrix = np.random.rand(100, 100)

# Execute the graph
# result = tl.execute(graph, {"knows": knows_matrix})
# print(f"Result shape: {result['output'].shape}")
# print(f"Result sample: {result['output'][:5, :5]}")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 2: Existential Quantifier
print("\n[Example 2] Existential Quantifier")
print("-" * 70)

# ∃y. knows(x, y) - "x knows someone"
# x = tl.var("x")
# y = tl.var("y")
# knows = tl.pred("knows", [x, y])
# knows_someone = tl.exists("y", "Person", knows)

# graph = tl.compile(knows_someone)
# knows_matrix = np.random.rand(100, 100)
# result = tl.execute(graph, {"knows": knows_matrix})

# print(f"Result shape: {result['output'].shape}")  # Should be (100,)
# print(f"People who know someone: {np.sum(result['output'] > 0.5)}")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 3: Logical Operators
print("\n[Example 3] Logical Operators (AND, OR, NOT)")
print("-" * 70)

# AND: knows(x,y) AND likes(x,y)
# x = tl.var("x")
# y = tl.var("y")
# knows = tl.pred("knows", [x, y])
# likes = tl.pred("likes", [x, y])
# friends = tl.and_(knows, likes)

# graph = tl.compile(friends)

# OR: knows(x,y) OR likes(x,y)
# acquaintance = tl.or_(knows, likes)
# graph_or = tl.compile(acquaintance)

# NOT: NOT knows(x,y)
# strangers = tl.not_(knows)
# graph_not = tl.compile(strangers)

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 4: Implication Rule
print("\n[Example 4] Implication Rule (Transitivity)")
print("-" * 70)

# Rule: knows(x,y) ∧ knows(y,z) → knows(x,z)
# x, y, z = tl.var("x"), tl.var("y"), tl.var("z")
# knows_xy = tl.pred("knows", [x, y])
# knows_yz = tl.pred("knows", [y, z])
# knows_xz = tl.pred("knows", [x, z])

# premise = tl.and_(knows_xy, knows_yz)
# conclusion = knows_xz
# rule = tl.imply(premise, conclusion)

# Wrap in quantifier: ∀y. (premise → conclusion)
# transitivity = tl.forall("y", "Person", rule)

# graph = tl.compile(transitivity)
# print(f"Transitivity rule compiled: {graph.num_nodes} nodes")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 5: Arithmetic Operations
print("\n[Example 5] Arithmetic Operations")
print("-" * 70)

# age(x) + 5
# x = tl.var("x")
# age = tl.pred("age", [x])
# age_plus_5 = tl.add(age, tl.constant(5.0))

# graph = tl.compile(age_plus_5)

# Sample data: ages of 100 people
# ages = np.random.randint(18, 80, size=(100,)).astype(float)
# result = tl.execute(graph, {"age": ages})
# print(f"Original ages: {ages[:5]}")
# print(f"Ages + 5: {result['output'][:5]}")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 6: Comparison Operations
print("\n[Example 6] Comparison Operations")
print("-" * 70)

# age(x) > 18
# x = tl.var("x")
# age = tl.pred("age", [x])
# adult = tl.gt(age, tl.constant(18.0))

# graph = tl.compile(adult)

# ages = np.random.randint(10, 30, size=(100,)).astype(float)
# result = tl.execute(graph, {"age": ages})

# print(f"Total adults: {np.sum(result['output'] > 0.5)}")
# print(f"Total minors: {np.sum(result['output'] < 0.5)}")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 7: Conditional Expression
print("\n[Example 7] Conditional Expression (if-then-else)")
print("-" * 70)

# if age(x) > 18 then 1.0 else 0.0
# x = tl.var("x")
# age = tl.pred("age", [x])
# adult = tl.gt(age, tl.constant(18.0))
# classification = tl.if_then_else(
#     adult,
#     tl.constant(1.0),  # adult
#     tl.constant(0.0)   # minor
# )

# graph = tl.compile(classification)
# ages = np.array([15, 20, 17, 25, 30])
# result = tl.execute(graph, {"age": ages})

# print(f"Ages: {ages}")
# print(f"Classifications: {result['output']}")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 8: Compilation Configurations
print("\n[Example 8] Compilation Configurations")
print("-" * 70)

# Different logic semantics
# x = tl.var("x")
# y = tl.var("y")
# expr = tl.and_(tl.pred("P", [x]), tl.pred("Q", [y]))

# Soft differentiable (default)
# config_soft = tl.CompilationConfig.soft_differentiable()
# graph_soft = tl.compile_with_config(expr, config_soft)

# Hard Boolean
# config_hard = tl.CompilationConfig.hard_boolean()
# graph_hard = tl.compile_with_config(expr, config_hard)

# Fuzzy logic
# config_godel = tl.CompilationConfig.fuzzy_godel()
# graph_godel = tl.compile_with_config(expr, config_godel)

# config_product = tl.CompilationConfig.fuzzy_product()
# graph_product = tl.compile_with_config(expr, config_product)

# config_lukasiewicz = tl.CompilationConfig.fuzzy_lukasiewicz()
# graph_luka = tl.compile_with_config(expr, config_lukasiewicz)

# Probabilistic
# config_prob = tl.CompilationConfig.probabilistic()
# graph_prob = tl.compile_with_config(expr, config_prob)

print("Compilation strategies:")
print("  - soft_differentiable: For neural network training (default)")
print("  - hard_boolean: Discrete Boolean logic")
print("  - fuzzy_godel: Gödel fuzzy logic")
print("  - fuzzy_product: Product fuzzy logic")
print("  - fuzzy_lukasiewicz: Łukasiewicz fuzzy logic")
print("  - probabilistic: Probabilistic interpretation")

print("\nStatus: Example ready (uncomment after building pytensorlogic)")


# Example 9: Complex Rule Composition
print("\n[Example 9] Complex Rule Composition")
print("-" * 70)

# Rule: ∀x,y,z. (knows(x,y) ∧ knows(y,z) ∧ likes(x,y)) → likes(x,z)
# "If x knows y, y knows z, and x likes y, then x likes z"

# x, y, z = tl.var("x"), tl.var("y"), tl.var("z")
# knows_xy = tl.pred("knows", [x, y])
# knows_yz = tl.pred("knows", [y, z])
# likes_xy = tl.pred("likes", [x, y])
# likes_xz = tl.pred("likes", [x, z])

# premise = tl.and_(tl.and_(knows_xy, knows_yz), likes_xy)
# conclusion = likes_xz
# rule = tl.imply(premise, conclusion)

# Quantify over all intermediate variables
# rule_quantified = tl.forall("y", "Person", rule)

# graph = tl.compile(rule_quantified)
# print(f"Complex rule compiled with {graph.num_nodes} nodes")

print("Status: Example ready (uncomment after building pytensorlogic)")


# Example 10: Practical Application - Friend Recommendation
print("\n[Example 10] Friend Recommendation System")
print("-" * 70)

print("Goal: Recommend friends based on mutual connections and interests")
print("")
print("Rules:")
print("  1. If x and y have mutual friends, recommend y to x")
print("  2. If x and y share interests, recommend y to x")
print("  3. Don't recommend if already friends")
print("")
print("Implementation:")
print("  recommend(x,y) := (∃z. knows(x,z) ∧ knows(y,z)) ∧ ¬knows(x,y)")
print("")

# x, y, z = tl.var("x"), tl.var("y"), tl.var("z")
# knows_xz = tl.pred("knows", [x, z])
# knows_yz = tl.pred("knows", [y, z])
# knows_xy = tl.pred("knows", [x, y])

# mutual_friends = tl.exists("z", "Person", tl.and_(knows_xz, knows_yz))
# not_already_friends = tl.not_(knows_xy)
# recommendation = tl.and_(mutual_friends, not_already_friends)

# graph = tl.compile(recommendation)

# Sample social network: 50 people
# num_people = 50
# knows_matrix = (np.random.rand(num_people, num_people) > 0.7).astype(float)
# Make symmetric (undirected friendship)
# knows_matrix = np.maximum(knows_matrix, knows_matrix.T)
# No self-loops
# np.fill_diagonal(knows_matrix, 0.0)

# result = tl.execute(graph, {"knows": knows_matrix})

# recommendations = result['output']
# print(f"Recommendation matrix shape: {recommendations.shape}")
# print(f"Total recommendations: {np.sum(recommendations > 0.5)}")

# Find top recommendations for person 0
# person_0_recommendations = recommendations[0, :]
# top_recommendations = np.argsort(person_0_recommendations)[-5:][::-1]
# print(f"Top 5 recommendations for person 0: {top_recommendations}")

print("Status: Example ready (uncomment after building pytensorlogic)")


print("\n" + "=" * 70)
print("Examples Complete!")
print("=" * 70)
print("\nTo run these examples:")
print("1. Build pytensorlogic: maturin develop")
print("2. Uncomment the import and example code")
print("3. Run: python examples/basic_usage.py")
print("")
print("For more information, see:")
print("  - README.md: Comprehensive documentation")
print("  - TODO.md: Implementation status and roadmap")
print("=" * 70)
