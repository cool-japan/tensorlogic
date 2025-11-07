#!/usr/bin/env python3
"""
TensorLogic - Advanced SymbolTable and CompilerContext Example

Demonstrates how to use SymbolTable for domain management and CompilerContext
for advanced compilation control in pytensorlogic.

Build and install:
    $ cd crates/pytensorlogic
    $ maturin develop
    $ python examples/advanced_symbol_table.py
"""

import numpy as np

try:
    import pytensorlogic as tl
except ImportError:
    print("Error: pytensorlogic not found. Please build with 'maturin develop' first.")
    exit(1)

print("=" * 70)
print("TensorLogic - Advanced SymbolTable and CompilerContext")
print("=" * 70)


# Example 1: Building a Symbol Table
print("\n[Example 1] Building a Symbol Table")
print("-" * 70)

# Create a symbol table
symbol_table = tl.SymbolTable()

# Add domains
person_domain = tl.DomainInfo("Person", 100)
city_domain = tl.DomainInfo("City", 50)
company_domain = tl.DomainInfo("Company", 30)

symbol_table.add_domain(person_domain)
symbol_table.add_domain(city_domain)
symbol_table.add_domain(company_domain)

print(f"Symbol table created: {symbol_table}")
print(f"Domains: {symbol_table.list_domains()}")

# Add predicates
lives_in = tl.PredicateInfo("lives_in", ["Person", "City"])
works_at = tl.PredicateInfo("works_at", ["Person", "Company"])
located_in = tl.PredicateInfo("located_in", ["Company", "City"])

symbol_table.add_predicate(lives_in)
symbol_table.add_predicate(works_at)
symbol_table.add_predicate(located_in)

print(f"Predicates: {symbol_table.list_predicates()}")

# Bind variables
symbol_table.bind_variable("x", "Person")
symbol_table.bind_variable("y", "City")
symbol_table.bind_variable("z", "Company")

print(f"Variable bindings: {symbol_table.get_variable_bindings()}")

# Example 2: Inferring Domains from Expressions
print("\n[Example 2] Inferring Domains from Expressions")
print("-" * 70)

# Create a new symbol table
auto_symbol_table = tl.SymbolTable()

# Create an expression
x = tl.var("x")
y = tl.var("y")
z = tl.var("z")

# ∃y ∈ City. (lives_in(x, y) ∧ ∃z ∈ Company. (works_at(x, z) ∧ located_in(z, y)))
lives_in_expr = tl.pred("lives_in", [x, y])
works_at_expr = tl.pred("works_at", [x, z])
located_in_expr = tl.pred("located_in", [z, y])

inner_exists = tl.exists("z", "Company", tl.and_(works_at_expr, located_in_expr))
full_expr = tl.exists("y", "City", tl.and_(lives_in_expr, inner_exists))

# Infer domains and predicates from expression
auto_symbol_table.infer_from_expr(full_expr)

print(f"Inferred domains: {auto_symbol_table.list_domains()}")
print(f"Inferred predicates: {auto_symbol_table.list_predicates()}")
print(f"Inferred variables: {auto_symbol_table.get_variable_bindings()}")

# Example 3: Using CompilerContext for Manual Domain Control
print("\n[Example 3] Using CompilerContext for Advanced Compilation")
print("-" * 70)

# Create a compiler context
ctx = tl.CompilerContext()

# Register domains with specific cardinalities
ctx.add_domain("Person", 100)
ctx.add_domain("City", 50)

# Bind variables to domains
ctx.bind_var("x", "Person")
ctx.bind_var("y", "City")

print(f"Compiler context: {ctx}")
print(f"Registered domains: {ctx.get_domains()}")
print(f"Variable bindings: {ctx.get_variable_bindings()}")

# Assign axes (for einsum notation)
axis_x = ctx.assign_axis("x")
axis_y = ctx.assign_axis("y")

print(f"Axis for 'x': {axis_x}")
print(f"Axis for 'y': {axis_y}")
print(f"All axis assignments: {ctx.get_axis_assignments()}")

# Generate temporary names
temp1 = ctx.fresh_temp()
temp2 = ctx.fresh_temp()

print(f"Generated temp names: {temp1}, {temp2}")

# Example 4: Domain Metadata and Descriptions
print("\n[Example 4] Domain Metadata and Descriptions")
print("-" * 70)

# Create domain with metadata
person_domain = tl.DomainInfo("Person", 100)
person_domain.set_description("Represents individual people in the system")
person_domain.set_elements(["Alice", "Bob", "Charlie", "Diana", "Eve"])

print(person_domain)

# Create predicate with description
knows_pred = tl.PredicateInfo("knows", ["Person", "Person"])
knows_pred.set_description("Indicates that one person knows another person")

print(knows_pred)

# Example 5: Export/Import Symbol Table as JSON
print("\n[Example 5] Export/Import Symbol Table as JSON")
print("-" * 70)

# Create a symbol table
export_table = tl.SymbolTable()
export_table.add_domain(tl.DomainInfo("Person", 100))
export_table.add_domain(tl.DomainInfo("City", 50))
export_table.add_predicate(tl.PredicateInfo("lives_in", ["Person", "City"]))
export_table.bind_variable("x", "Person")

# Export to JSON
json_str = export_table.to_json()
print("Exported JSON (first 200 chars):")
print(json_str[:200] + "...")

# Import from JSON
imported_table = tl.SymbolTable.from_json(json_str)
print(f"\nImported symbol table: {imported_table}")
print(f"Domains: {imported_table.list_domains()}")
print(f"Predicates: {imported_table.list_predicates()}")

# Example 6: Querying Symbol Table Information
print("\n[Example 6] Querying Symbol Table Information")
print("-" * 70)

query_table = tl.SymbolTable()
query_table.add_domain(tl.DomainInfo("Person", 100))
query_table.add_domain(tl.DomainInfo("City", 50))
query_table.add_predicate(tl.PredicateInfo("lives_in", ["Person", "City"]))
query_table.bind_variable("x", "Person")
query_table.bind_variable("y", "City")

# Query specific information
person_domain = query_table.get_domain("Person")
if person_domain:
    print(f"Person domain: {person_domain.name}, size={person_domain.cardinality}")

lives_in_pred = query_table.get_predicate("lives_in")
if lives_in_pred:
    print(f"lives_in predicate: arity={lives_in_pred.arity}, args={lives_in_pred.arg_domains}")

x_domain = query_table.get_variable_domain("x")
print(f"Variable 'x' is bound to domain: {x_domain}")

# Example 7: Real-World Application - Social Network Analysis
print("\n[Example 7] Real-World Application - Social Network Analysis")
print("-" * 70)

# Set up the schema
social_network = tl.SymbolTable()

# Define domains
social_network.add_domain(tl.DomainInfo("Person", 1000))
social_network.add_domain(tl.DomainInfo("Post", 5000))
social_network.add_domain(tl.DomainInfo("Topic", 50))

# Define predicates
follows = tl.PredicateInfo("follows", ["Person", "Person"])
follows.set_description("Person A follows Person B")
social_network.add_predicate(follows)

likes = tl.PredicateInfo("likes", ["Person", "Post"])
likes.set_description("Person likes a post")
social_network.add_predicate(likes)

authored = tl.PredicateInfo("authored", ["Person", "Post"])
authored.set_description("Person authored a post")
social_network.add_predicate(authored)

about = tl.PredicateInfo("about", ["Post", "Topic"])
about.set_description("Post is about a topic")
social_network.add_predicate(about)

interested_in = tl.PredicateInfo("interested_in", ["Person", "Topic"])
interested_in.set_description("Person is interested in a topic")
social_network.add_predicate(interested_in)

print("Social Network Schema:")
print(f"  Domains: {social_network.list_domains()}")
print(f"  Predicates: {social_network.list_predicates()}")

# Define a complex query: "Find posts that a person might like"
# Rule: A person might like a post if:
#   1. They follow the author, OR
#   2. The post is about a topic they're interested in

p = tl.var("p")  # person
post = tl.var("post")  # post
author = tl.var("author")  # author
topic = tl.var("topic")  # topic

# Condition 1: Person follows the author of the post
follows_author = tl.exists(
    "author",
    "Person",
    tl.and_(
        tl.pred("follows", [p, author]),
        tl.pred("authored", [author, post])
    )
)

# Condition 2: Post is about a topic the person is interested in
interested_topic = tl.exists(
    "topic",
    "Topic",
    tl.and_(
        tl.pred("interested_in", [p, topic]),
        tl.pred("about", [post, topic])
    )
)

# Combined rule
might_like = tl.or_(follows_author, interested_topic)

print("\nComplex query created:")
print(f"Expression: {might_like}")

# Verify the expression uses the correct domains
social_network.infer_from_expr(might_like)
print(f"Expression uses domains: {social_network.get_variable_bindings()}")

# Example 8: CompilerContext for Multi-Stage Compilation
print("\n[Example 8] Multi-Stage Compilation with CompilerContext")
print("-" * 70)

# Stage 1: Set up base context
stage1_ctx = tl.CompilerContext()
stage1_ctx.add_domain("Person", 100)
stage1_ctx.add_domain("City", 50)

print(f"Stage 1 context: {stage1_ctx}")

# Stage 2: Add more domains
stage1_ctx.add_domain("Country", 20)
stage1_ctx.bind_var("x", "Person")
stage1_ctx.bind_var("y", "City")
stage1_ctx.bind_var("z", "Country")

print(f"Stage 2 context: {stage1_ctx}")

# Allocate axes for all variables
for var in ["x", "y", "z"]:
    axis = stage1_ctx.assign_axis(var)
    print(f"  Variable '{var}' -> axis '{axis}'")

print("\n" + "=" * 70)
print("Advanced SymbolTable and CompilerContext Examples Complete!")
print("=" * 70)

print("\nKey Takeaways:")
print("1. SymbolTable manages domains, predicates, and variable bindings")
print("2. Can infer schema from logical expressions")
print("3. CompilerContext provides low-level control over compilation")
print("4. Can export/import symbol tables as JSON")
print("5. Useful for building structured knowledge bases")
print("6. Enables complex multi-domain reasoning")
