#!/bin/bash
# Example: Interactive REPL workflow
#
# This example demonstrates an interactive REPL session using
# a series of commands to compile, optimize, and execute expressions.

echo "===================================="
echo "REPL Workflow Example"
echo "===================================="
echo
echo "Starting REPL with sample commands..."
echo
echo "Commands that will be executed:"
echo "  1. Show help"
echo "  2. Add domains"
echo "  3. Set compilation strategy"
echo "  4. Compile an expression"
echo "  5. Execute the compiled graph"
echo "  6. Optimize the graph"
echo "  7. Execute the optimized graph"
echo "  8. Change backend"
echo "  9. Execute with new backend"
echo " 10. Exit"
echo

# Create a command file for automated REPL interaction
cat > /tmp/repl_commands.txt << 'EOF'
.help
.domain Person 50
.domain City 20
.strategy soft_differentiable
knows(x, y) AND lives_in(x, c) AND lives_in(y, c)
.execute --metrics
.optimize aggressive --stats
.execute --metrics
.backend parallel
.execute --metrics
.context
.history
.exit
EOF

echo "===================================="
echo "Sample REPL Session"
echo "===================================="
echo
echo "To run interactively:"
echo "  tensorlogic repl"
echo
echo "Example commands:"
echo

cat << 'EOF'
tensorlogic> .help
tensorlogic> .domain Person 100
✓ Added domain 'Person' with size 100

tensorlogic> .backend
Current backend: SciRS2 CPU

Available backends:
*   SciRS2 CPU
    SciRS2 Parallel
    SciRS2 Profiled

tensorlogic> knows(x, y) AND likes(y, z)
✓ Compilation successful
  3 tensors, 3 nodes, depth 2

tensorlogic> .execute --metrics
ℹ Executing with SciRS2 CPU backend...
✓ Execution completed with SciRS2 CPU in 1.234 ms

Output shape: [100, 100]
Output dtype: f64

Performance Metrics:
  Execution time: 1.234 ms
  Memory used: 80000 bytes
  Throughput: 162.01 GFLOPS

tensorlogic> .optimize aggressive --stats
ℹ Optimizing with aggressive level...
✓ Optimization complete: eliminated 0 nodes, 1 CSE, 0 identities
  2 tensors, 2 nodes, depth 2

Optimization Impact:
  DCE eliminated: 0
  CSE eliminated: 1
  Identity simplified: 0

tensorlogic> .backend parallel
✓ Backend set to: SciRS2 Parallel

tensorlogic> .execute --metrics
ℹ Executing with SciRS2 Parallel backend...
✓ Execution completed with SciRS2 Parallel in 0.856 ms
[faster execution with parallel backend]

tensorlogic> .exit
Goodbye!
EOF

echo
echo "===================================="
echo "REPL workflow example complete!"
echo "===================================="
echo
echo "Try it yourself:"
echo "  tensorlogic repl"
