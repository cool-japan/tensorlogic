# Graph Analysis Rules
# Path finding and connectivity rules

# Direct edge implies reachability
FORALL x IN Node. FORALL y IN Node.
  edge(x, y) -> reachable(x, y)

# Transitive reachability
FORALL x IN Node. FORALL y IN Node. FORALL z IN Node.
  (reachable(x, y) AND reachable(y, z)) -> reachable(x, z)

# Cycle detection: if X reaches Y and Y reaches X
FORALL x IN Node. FORALL y IN Node.
  (reachable(x, y) AND reachable(y, x) AND NOT (x = y)) -> in_cycle(x, y)

# Common neighbors
FORALL x IN Node. FORALL y IN Node.
  (EXISTS z IN Node. edge(x, z) AND edge(y, z)) -> share_neighbor(x, y)

# Triangle detection
FORALL x IN Node. FORALL y IN Node. FORALL z IN Node.
  (edge(x, y) AND edge(y, z) AND edge(z, x)) -> triangle(x, y, z)

# Compile with:
# tensorlogic graph_analysis.tl \
#   --domains Node:1000 \
#   --strategy fuzzy_product \
#   --output-format dot | dot -Tpng -o graph.png
