#!/bin/bash
# Example: Graph optimization pipeline
#
# This example demonstrates how to optimize logical expressions
# at different optimization levels and analyze the results.

set -e

echo "===================================="
echo "Graph Optimization Pipeline Example"
echo "===================================="
echo

# Expression with redundancy (for CSE demonstration)
EXPR="(p(x) AND q(y)) OR (p(x) AND r(z)) OR (p(x) AND s(w))"

echo "Original Expression:"
echo "  $EXPR"
echo
echo "This expression has common subexpressions (p(x) appears 3 times)"
echo

echo "1. Compile without optimization"
echo "-----------------------------------"
tensorlogic "$EXPR" \
  --output-format stats
echo

echo "2. Optimize: None (0 passes)"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level none \
  --stats \
  --output-format stats
echo

echo "3. Optimize: Basic (1 pass)"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level basic \
  --stats \
  --verbose \
  --output-format stats
echo

echo "4. Optimize: Standard (2 passes)"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level standard \
  --stats \
  --verbose \
  --output-format stats
echo

echo "5. Optimize: Aggressive (until convergence)"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level aggressive \
  --stats \
  --verbose \
  --output-format stats
echo

echo "6. Save optimized graph"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level aggressive \
  --stats \
  --output optimized_graph.json \
  --output-format json

echo "Optimized graph saved to: optimized_graph.json"
echo

echo "7. Visualize with Graphviz"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" \
  --level aggressive \
  --output-format dot > optimized_graph.dot

if command -v dot &> /dev/null; then
    dot -Tpng optimized_graph.dot -o optimized_graph.png
    echo "Graph visualization saved to: optimized_graph.png"
else
    echo "Graphviz 'dot' not found. Install it to generate visualization."
    echo "DOT file saved to: optimized_graph.dot"
fi
echo

echo "===================================="
echo "Optimization pipeline complete!"
echo "===================================="
