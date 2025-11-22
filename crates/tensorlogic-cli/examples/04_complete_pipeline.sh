#!/bin/bash
# Example: Complete compile-optimize-execute pipeline
#
# This example demonstrates a complete workflow from expression
# to optimized execution with analysis and visualization.

set -e

echo "===================================="
echo "Complete Pipeline Example"
echo "===================================="
echo

# Social network reasoning example
EXPR='FORALL x IN Person.
    (friend(x, y) AND friend(y, z)) -> potential_friend(x, z)'

echo "Use Case: Social Network Friend Recommendation"
echo
echo "Expression:"
echo "  For all persons x, if x is friends with y, and y is friends with z,"
echo "  then x might be interested in connecting with z (potential friend)."
echo

DOMAINS="--domain Person:100"

echo "Step 1: Compile and Analyze"
echo "-----------------------------------"
tensorlogic "$EXPR" $DOMAINS \
  --analyze \
  --output-format stats
echo

echo "Step 2: Optimize the Graph"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" $DOMAINS \
  --level aggressive \
  --stats \
  --verbose \
  --output optimized.json \
  --output-format json

echo "✓ Optimized graph saved to: optimized.json"
echo

echo "Step 3: Visualize (DOT format)"
echo "-----------------------------------"
tensorlogic optimize "$EXPR" $DOMAINS \
  --level aggressive \
  --output-format dot > pipeline.dot

if command -v dot &> /dev/null; then
    dot -Tpng pipeline.dot -o pipeline.png
    echo "✓ Visualization saved to: pipeline.png"
else
    echo "ℹ DOT file saved to: pipeline.dot"
    echo "  Install Graphviz to generate PNG"
fi
echo

echo "Step 4: Execute with Different Backends"
echo "-----------------------------------"

for backend in cpu parallel profiled; do
    echo "Backend: $backend"
    tensorlogic execute "$EXPR" $DOMAINS \
        --backend "$backend" \
        --metrics \
        --output-format json > "results_${backend}.json"
    echo "  ✓ Results saved to: results_${backend}.json"
done
echo

echo "Step 5: Compare Performance"
echo "-----------------------------------"
echo "Analyzing execution times from JSON outputs..."

for backend in cpu parallel profiled; do
    if command -v jq &> /dev/null; then
        time_ms=$(jq -r '.execution_time_ms' "results_${backend}.json")
        memory=$(jq -r '.memory_bytes' "results_${backend}.json")
        echo "  $backend:"
        echo "    Time: ${time_ms} ms"
        echo "    Memory: ${memory} bytes"
    else
        echo "  $backend: (install 'jq' for detailed comparison)"
    fi
done
echo

echo "Step 6: Full Pipeline Summary"
echo "-----------------------------------"
echo "Generated files:"
echo "  - optimized.json      (Optimized graph)"
echo "  - pipeline.dot        (Graph visualization)"
if [ -f "pipeline.png" ]; then
    echo "  - pipeline.png        (Rendered graph image)"
fi
echo "  - results_cpu.json    (CPU execution results)"
echo "  - results_parallel.json (Parallel execution results)"
echo "  - results_profiled.json (Profiled execution results)"
echo

echo "===================================="
echo "Complete pipeline example finished!"
echo "===================================="
echo
echo "Next steps:"
echo "  1. Review the generated files"
echo "  2. Analyze performance differences between backends"
echo "  3. Visualize the optimized graph (pipeline.png)"
echo "  4. Try with your own expressions!"
