#!/bin/bash
# Example: Comparing different execution backends
#
# This example demonstrates how to execute the same logical expression
# with different backends and compare their performance.

set -e

echo "===================================="
echo "Backend Comparison Example"
echo "===================================="
echo

# Complex expression for meaningful comparison
EXPR="FORALL x IN Person. (knows(x, alice) -> likes(alice, x)) AND EXISTS y IN Person. knows(x, y)"

echo "Expression:"
echo "  $EXPR"
echo

# Create domains
DOMAINS="--domain Person:100"

echo "1. List Available Backends"
echo "-----------------------------------"
tensorlogic backends
echo

echo "2. Execute with CPU Backend"
echo "-----------------------------------"
tensorlogic execute "$EXPR" $DOMAINS \
  --backend cpu \
  --metrics \
  --output-format table
echo

echo "3. Execute with Parallel Backend"
echo "-----------------------------------"
tensorlogic execute "$EXPR" $DOMAINS \
  --backend parallel \
  --metrics \
  --output-format table
echo

echo "4. Execute with Profiled Backend"
echo "-----------------------------------"
tensorlogic execute "$EXPR" $DOMAINS \
  --backend profiled \
  --metrics \
  --output-format table
echo

echo "5. Export Results as JSON"
echo "-----------------------------------"
tensorlogic execute "$EXPR" $DOMAINS \
  --backend cpu \
  --output-format json > execution_results.json

echo "Results saved to: execution_results.json"
echo

echo "6. Export as CSV for Analysis"
echo "-----------------------------------"
tensorlogic execute "$EXPR" $DOMAINS \
  --backend cpu \
  --output-format csv > execution_results.csv

echo "Results saved to: execution_results.csv"
echo

echo "===================================="
echo "Backend comparison complete!"
echo "===================================="
