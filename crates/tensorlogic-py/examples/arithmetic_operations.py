#!/usr/bin/env python3
"""
TensorLogic - Arithmetic Operations Example

Demonstrates all arithmetic operations (add, sub, mul, div) in pytensorlogic.

Build and install:
    $ cd crates/pytensorlogic
    $ maturin develop
    $ python examples/arithmetic_operations.py
"""

import numpy as np

try:
    import pytensorlogic as tl
except ImportError:
    print("Error: pytensorlogic not found. Please build with 'maturin develop' first.")
    exit(1)

print("=" * 70)
print("TensorLogic - Arithmetic Operations")
print("=" * 70)


# Example 1: Addition
print("\n[Example 1] Addition: age(x) + 5")
print("-" * 70)

x = tl.var("x")
age = tl.pred("age", [x])
age_plus_5 = tl.add(age, tl.constant(5.0))

graph = tl.compile(age_plus_5)
print(f"Graph: {graph}")
print(f"Nodes: {graph.num_nodes}, Outputs: {graph.num_outputs}")

# Sample data: ages of 10 people
ages = np.array([18, 25, 30, 45, 50, 22, 35, 40, 28, 33], dtype=np.float64)
result = tl.execute(graph, {"age": ages})

print(f"Original ages: {ages}")
print(f"Ages + 5: {result['output']}")


# Example 2: Subtraction
print("\n[Example 2] Subtraction: age(x) - 10")
print("-" * 70)

age_minus_10 = tl.sub(age, tl.constant(10.0))
graph2 = tl.compile(age_minus_10)
result2 = tl.execute(graph2, {"age": ages})

print(f"Original ages: {ages}")
print(f"Ages - 10: {result2['output']}")


# Example 3: Multiplication
print("\n[Example 3] Multiplication: salary(x) * 1.1 (10% raise)")
print("-" * 70)

salary = tl.pred("salary", [x])
salary_with_raise = tl.mul(salary, tl.constant(1.1))

graph3 = tl.compile(salary_with_raise)

salaries = np.array([50000, 60000, 75000, 90000, 100000], dtype=np.float64)
result3 = tl.execute(graph3, {"salary": salaries})

print(f"Original salaries: {salaries}")
print(f"With 10% raise: {result3['output']}")


# Example 4: Division
print("\n[Example 4] Division: total_score(x) / num_tests(x)")
print("-" * 70)

total_score = tl.pred("total_score", [x])
num_tests = tl.pred("num_tests", [x])
average_score = tl.div(total_score, num_tests)

graph4 = tl.compile(average_score)

total_scores = np.array([450, 380, 490, 420, 460], dtype=np.float64)
num_tests_data = np.array([5, 5, 5, 5, 5], dtype=np.float64)

result4 = tl.execute(graph4, {"total_score": total_scores, "num_tests": num_tests_data})

print(f"Total scores: {total_scores}")
print(f"Number of tests: {num_tests_data}")
print(f"Average scores: {result4['output']}")


# Example 5: Combined Operations
print("\n[Example 5] Combined: (age(x) + 5) * 2 - 10")
print("-" * 70)

step1 = tl.add(age, tl.constant(5.0))
step2 = tl.mul(step1, tl.constant(2.0))
combined = tl.sub(step2, tl.constant(10.0))

graph5 = tl.compile(combined)
result5 = tl.execute(graph5, {"age": ages})

print(f"Original ages: {ages}")
print(f"After (age + 5) * 2 - 10: {result5['output']}")


# Example 6: Complex Expression
print("\n[Example 6] BMI Calculation: weight(x) / (height(x) * height(x))")
print("-" * 70)

weight = tl.pred("weight", [x])
height = tl.pred("height", [x])
height_squared = tl.mul(height, height)
bmi = tl.div(weight, height_squared)

graph6 = tl.compile(bmi)

# Sample data: 5 people
weights = np.array([70, 85, 60, 95, 75], dtype=np.float64)  # kg
heights = np.array([1.75, 1.80, 1.65, 1.90, 1.70], dtype=np.float64)  # meters

result6 = tl.execute(graph6, {"weight": weights, "height": heights})

print(f"Weights (kg): {weights}")
print(f"Heights (m): {heights}")
print(f"BMI: {result6['output']}")
print(f"BMI categories:")
for i, bmi_val in enumerate(result6['output']):
    if bmi_val < 18.5:
        category = "Underweight"
    elif bmi_val < 25:
        category = "Normal"
    elif bmi_val < 30:
        category = "Overweight"
    else:
        category = "Obese"
    print(f"  Person {i}: BMI={bmi_val:.2f} -> {category}")


print("\n" + "=" * 70)
print("Arithmetic Operations Examples Complete!")
print("=" * 70)
