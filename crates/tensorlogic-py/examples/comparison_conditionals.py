#!/usr/bin/env python3
"""
TensorLogic - Comparison and Conditional Operations Example

Demonstrates comparison operations (eq, lt, gt, lte, gte) and conditional
expressions (if_then_else) in pytensorlogic.

Build and install:
    $ cd crates/pytensorlogic
    $ maturin develop
    $ python examples/comparison_conditionals.py
"""

import numpy as np

try:
    import pytensorlogic as tl
except ImportError:
    print("Error: pytensorlogic not found. Please build with 'maturin develop' first.")
    exit(1)

print("=" * 70)
print("TensorLogic - Comparison and Conditional Operations")
print("=" * 70)


# Example 1: Greater Than
print("\n[Example 1] Greater Than: age(x) > 18 (adult check)")
print("-" * 70)

x = tl.var("x")
age = tl.pred("age", [x])
is_adult = tl.gt(age, tl.constant(18.0))

graph = tl.compile(is_adult)

ages = np.array([15, 20, 17, 25, 30, 16, 22, 19, 14, 21], dtype=np.float64)
result = tl.execute(graph, {"age": ages})

print(f"Ages: {ages}")
print(f"Is adult (>18): {result['output']}")
print(f"Number of adults: {np.sum(result['output'] > 0.5)}")


# Example 2: Less Than
print("\n[Example 2] Less Than: score(x) < 60 (failing grade)")
print("-" * 70)

score = tl.pred("score", [x])
is_failing = tl.lt(score, tl.constant(60.0))

graph2 = tl.compile(is_failing)

scores = np.array([45, 78, 55, 92, 38, 82, 67, 51, 88, 73], dtype=np.float64)
result2 = tl.execute(graph2, {"score": scores})

print(f"Scores: {scores}")
print(f"Is failing (<60): {result2['output']}")
print(f"Number failing: {np.sum(result2['output'] > 0.5)}")


# Example 3: Greater Than or Equal
print("\n[Example 3] Greater Than or Equal: temperature(x) >= 100 (boiling)")
print("-" * 70)

temp = tl.pred("temperature", [x])
is_boiling = tl.gte(temp, tl.constant(100.0))

graph3 = tl.compile(is_boiling)

temperatures = np.array([85, 95, 100, 105, 98, 110, 92, 100, 88, 115], dtype=np.float64)
result3 = tl.execute(graph3, {"temperature": temperatures})

print(f"Temperatures (°C): {temperatures}")
print(f"Is boiling (>=100): {result3['output']}")


# Example 4: Less Than or Equal
print("\n[Example 4] Less Than or Equal: speed(x) <= 60 (speed limit)")
print("-" * 70)

speed = tl.pred("speed", [x])
within_limit = tl.lte(speed, tl.constant(60.0))

graph4 = tl.compile(within_limit)

speeds = np.array([55, 65, 58, 72, 45, 61, 50, 68, 60, 75], dtype=np.float64)
result4 = tl.execute(graph4, {"speed": speeds})

print(f"Speeds (mph): {speeds}")
print(f"Within limit (<=60): {result4['output']}")
print(f"Number speeding: {np.sum(result4['output'] < 0.5)}")


# Example 5: Equality
print("\n[Example 5] Equality: answer(x) == 42 (correct answer)")
print("-" * 70)

answer = tl.pred("answer", [x])
is_correct = tl.eq(answer, tl.constant(42.0))

graph5 = tl.compile(is_correct)

answers = np.array([40, 42, 38, 42, 45, 42, 50, 42, 35, 42], dtype=np.float64)
result5 = tl.execute(graph5, {"answer": answers})

print(f"Answers: {answers}")
print(f"Is correct (==42): {result5['output']}")
print(f"Number correct: {np.sum(result5['output'] > 0.5)}")


# Example 6: If-Then-Else (Adult Classification)
print("\n[Example 6] If-Then-Else: Classify as Adult or Minor")
print("-" * 70)

age = tl.pred("age", [x])
is_adult_cond = tl.gt(age, tl.constant(18.0))
classification = tl.if_then_else(
    is_adult_cond,
    tl.constant(1.0),  # Adult = 1
    tl.constant(0.0)   # Minor = 0
)

graph6 = tl.compile(classification)

ages = np.array([15, 20, 17, 25, 30, 16, 22, 19, 14, 21], dtype=np.float64)
result6 = tl.execute(graph6, {"age": ages})

print(f"Ages: {ages}")
print(f"Classification (1=Adult, 0=Minor): {result6['output']}")


# Example 7: If-Then-Else (Grade Calculation)
print("\n[Example 7] If-Then-Else: Pass/Fail Grade System")
print("-" * 70)

score = tl.pred("score", [x])
is_passing = tl.gte(score, tl.constant(60.0))
grade = tl.if_then_else(
    is_passing,
    tl.constant(100.0),  # Pass = 100
    tl.constant(0.0)     # Fail = 0
)

graph7 = tl.compile(grade)

scores = np.array([45, 78, 55, 92, 38, 82, 67, 51, 88, 73], dtype=np.float64)
result7 = tl.execute(graph7, {"score": scores})

print(f"Scores: {scores}")
print(f"Pass/Fail (100=Pass, 0=Fail): {result7['output']}")


# Example 8: Nested If-Then-Else (Temperature Warning System)
print("\n[Example 8] Nested If-Then-Else: Temperature Warning Levels")
print("-" * 70)

temp = tl.pred("temperature", [x])

# Level 3: temp > 100 (critical)
is_critical = tl.gt(temp, tl.constant(100.0))

# Level 2: temp > 85 (warning)
is_warning = tl.gt(temp, tl.constant(85.0))

# Nested: if critical then 3, else if warning then 2, else 1
warning_level = tl.if_then_else(
    is_critical,
    tl.constant(3.0),  # Critical
    tl.if_then_else(
        is_warning,
        tl.constant(2.0),  # Warning
        tl.constant(1.0)   # Normal
    )
)

graph8 = tl.compile(warning_level)

temperatures = np.array([75, 88, 95, 105, 80, 110, 92, 78, 85, 115], dtype=np.float64)
result8 = tl.execute(graph8, {"temperature": temperatures})

print(f"Temperatures (°C): {temperatures}")
print(f"Warning levels (1=Normal, 2=Warning, 3=Critical): {result8['output']}")

# Interpret results
for i, (temp_val, level) in enumerate(zip(temperatures, result8['output'])):
    if level > 2.5:
        status = "CRITICAL"
    elif level > 1.5:
        status = "WARNING"
    else:
        status = "Normal"
    print(f"  Sensor {i}: {temp_val}°C -> {status}")


# Example 9: Combined Comparison and Arithmetic
print("\n[Example 9] Combined: Bonus Calculation")
print("-" * 70)
print("Rule: if performance >= 90 then salary * 1.2 else salary * 1.05")

salary = tl.pred("salary", [x])
performance = tl.pred("performance", [x])

is_high_performer = tl.gte(performance, tl.constant(90.0))
high_bonus = tl.mul(salary, tl.constant(1.2))   # 20% bonus
standard_bonus = tl.mul(salary, tl.constant(1.05))  # 5% bonus

bonus_salary = tl.if_then_else(is_high_performer, high_bonus, standard_bonus)

graph9 = tl.compile(bonus_salary)

salaries = np.array([50000, 60000, 75000, 90000, 100000], dtype=np.float64)
performances = np.array([92, 85, 95, 88, 91], dtype=np.float64)

result9 = tl.execute(graph9, {"salary": salaries, "performance": performances})

print(f"Salaries: {salaries}")
print(f"Performance scores: {performances}")
print(f"New salaries with bonus: {result9['output']}")

for i, (sal, perf, new_sal) in enumerate(zip(salaries, performances, result9['output'])):
    bonus_pct = ((new_sal - sal) / sal) * 100
    print(f"  Employee {i}: ${sal:.0f} (score={perf}) -> ${new_sal:.0f} (+{bonus_pct:.0f}%)")


print("\n" + "=" * 70)
print("Comparison and Conditional Operations Examples Complete!")
print("=" * 70)
