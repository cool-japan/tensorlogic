//! # Example 14: Constraint Logic Programming (CLP)
//!
//! This example demonstrates the constraint satisfaction problem (CSP) solving capabilities
//! of TensorLogic's CLP module.
//!
//! ## What You'll Learn
//!
//! - Creating constraint domains (finite, interval, boolean, enumeration)
//! - Defining variables with domain constraints
//! - Adding binary, unary, and n-ary constraints
//! - Solving CSPs with different propagation algorithms
//! - Using search heuristics for efficient solving
//!
//! ## Key Concepts
//!
//! **Constraint Satisfaction Problem (CSP)**: Find values for variables that satisfy all constraints
//! - **Variables**: Named entities with domains of possible values
//! - **Domains**: Sets of values variables can take (finite, continuous, etc.)
//! - **Constraints**: Relations that must hold between variables
//! - **Propagation**: Automatically reducing domains based on constraints

use tensorlogic_ir::clp::{
    BinaryRelation, Constraint, CspSolver, Domain, NAryRelation, PropagationAlgorithm, Variable,
};

fn main() {
    println!("=== Constraint Logic Programming Examples ===\n");

    // Example 1: Basic CSP with Finite Domains
    example_1_basic_csp();

    // Example 2: Domain Types
    example_2_domain_types();

    // Example 3: Binary Constraints
    example_3_binary_constraints();

    // Example 4: N-ary Constraints
    example_4_nary_constraints();

    // Example 5: Map Coloring Problem
    example_5_map_coloring();

    // Example 6: Scheduling Problem
    example_6_scheduling();

    // Example 7: Sudoku-style All-Different
    example_8_all_different();

    // Example 8: Search Heuristics
    example_9_search_heuristics();

    // Example 9: Propagation Algorithms
    example_10_propagation();
}

fn example_1_basic_csp() {
    println!("Example 1: Basic CSP - Two Variables, Not Equal");
    println!("Find values for X and Y where X ≠ Y\n");

    let mut solver = CspSolver::new();

    // Variables with small domains
    let x = Variable::new("X", Domain::finite_domain(vec![1, 2, 3]));
    let y = Variable::new("Y", Domain::finite_domain(vec![2, 3, 4]));

    solver.add_variable(x);
    solver.add_variable(y);

    // Constraint: X ≠ Y
    solver.add_constraint(Constraint::Binary {
        var1: "X".to_string(),
        var2: "Y".to_string(),
        relation: BinaryRelation::NotEqual,
    });

    match solver.solve() {
        Some(solution) => {
            println!("  ✓ Solution found:");
            println!("    X = {}", solution["X"]);
            println!("    Y = {}", solution["Y"]);
            println!("  Statistics:");
            println!("    Assignments tried: {}", solver.stats.assignments_tried);
            println!("    Constraint checks: {}", solver.stats.constraint_checks);
        }
        None => println!("  ✗ No solution exists"),
    }
    println!();
}

fn example_2_domain_types() {
    println!("Example 2: Different Domain Types\n");

    // Finite domain
    let finite = Domain::finite_domain(vec![1, 2, 3, 4, 5]);
    println!("  Finite domain: {:?}", finite);
    println!("    Size: {:?}", finite.size());
    println!("    Contains 3: {}", finite.contains_int(3));

    // Range domain
    let range = Domain::range(1..=10);
    println!("  Range domain (1..=10):");
    println!("    Size: {:?}", range.size());

    // Interval domain
    let interval = Domain::interval(0.0, 100.0);
    println!("  Interval domain [0.0, 100.0]:");
    println!("    Size: {:?} (infinite)", interval.size());
    println!("    Contains 50: {}", interval.contains_int(50));

    // Boolean domain
    let boolean = Domain::boolean();
    println!("  Boolean domain:");
    println!("    Size: {:?}", boolean.size());

    // Domain intersection
    let d1 = Domain::finite_domain(vec![1, 2, 3, 4, 5]);
    let d2 = Domain::finite_domain(vec![3, 4, 5, 6, 7]);
    if let Ok(intersection) = d1.intersect(&d2) {
        println!("  Intersection of [1,2,3,4,5] and [3,4,5,6,7]:");
        println!("    Result size: {:?}", intersection.size());
    }
    println!();
}

fn example_3_binary_constraints() {
    println!("Example 3: Binary Constraints");
    println!("Various relations between two variables\n");

    // Example: X < Y
    let mut solver = CspSolver::new();

    let x = Variable::new("X", Domain::range(1..=5));
    let y = Variable::new("Y", Domain::range(1..=5));

    solver.add_variable(x);
    solver.add_variable(y);

    solver.add_constraint(Constraint::Binary {
        var1: "X".to_string(),
        var2: "Y".to_string(),
        relation: BinaryRelation::LessThan,
    });

    match solver.solve() {
        Some(solution) => {
            println!("  Constraint: X < Y");
            println!("  Solution: X={}, Y={}", solution["X"], solution["Y"]);
            println!("  Valid: {}", solution["X"] < solution["Y"]);
        }
        None => println!("  No solution found"),
    }
    println!();
}

fn example_4_nary_constraints() {
    println!("Example 4: N-ary Constraints");
    println!("Constraints involving multiple variables\n");

    let mut solver = CspSolver::new();

    // Three variables
    let x = Variable::new("X", Domain::range(1..=5));
    let y = Variable::new("Y", Domain::range(1..=5));
    let z = Variable::new("Z", Domain::range(1..=5));

    solver.add_variable(x);
    solver.add_variable(y);
    solver.add_variable(z);

    // Constraint: X + Y + Z = 10
    solver.add_constraint(Constraint::NAry {
        vars: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        relation: NAryRelation::SumEquals(10),
    });

    match solver.solve() {
        Some(solution) => {
            let x_val = solution["X"];
            let y_val = solution["Y"];
            let z_val = solution["Z"];
            println!("  Constraint: X + Y + Z = 10");
            println!("  Solution: X={}, Y={}, Z={}", x_val, y_val, z_val);
            println!("  Sum: {}", x_val + y_val + z_val);
        }
        None => println!("  No solution found"),
    }
    println!();
}

fn example_5_map_coloring() {
    println!("Example 5: Map Coloring Problem");
    println!("Color adjacent regions with different colors\n");

    // Simplified map with 4 regions
    // A -- B
    // |    |
    // C -- D

    let mut solver = CspSolver::new();

    // 3 colors: Red=1, Green=2, Blue=3
    let colors = vec![1, 2, 3];

    let a = Variable::new("RegionA", Domain::finite_domain(colors.clone()));
    let b = Variable::new("RegionB", Domain::finite_domain(colors.clone()));
    let c = Variable::new("RegionC", Domain::finite_domain(colors.clone()));
    let d = Variable::new("RegionD", Domain::finite_domain(colors.clone()));

    solver.add_variable(a);
    solver.add_variable(b);
    solver.add_variable(c);
    solver.add_variable(d);

    // Adjacent regions must have different colors
    let adjacencies = vec![
        ("RegionA", "RegionB"),
        ("RegionA", "RegionC"),
        ("RegionB", "RegionD"),
        ("RegionC", "RegionD"),
    ];

    for (r1, r2) in adjacencies {
        solver.add_constraint(Constraint::Binary {
            var1: r1.to_string(),
            var2: r2.to_string(),
            relation: BinaryRelation::NotEqual,
        });
    }

    match solver.solve() {
        Some(solution) => {
            println!("  ✓ Coloring found:");
            let color_name = |c: i64| match c {
                1 => "Red",
                2 => "Green",
                3 => "Blue",
                _ => "Unknown",
            };
            println!("    Region A: {}", color_name(solution["RegionA"]));
            println!("    Region B: {}", color_name(solution["RegionB"]));
            println!("    Region C: {}", color_name(solution["RegionC"]));
            println!("    Region D: {}", color_name(solution["RegionD"]));
        }
        None => println!("  ✗ No valid coloring exists"),
    }
    println!();
}

fn example_6_scheduling() {
    println!("Example 6: Scheduling Problem");
    println!("Schedule tasks with temporal constraints\n");

    let mut solver = CspSolver::new();

    // 3 tasks, each can start at time 0, 1, 2, or 3
    let task1 = Variable::new("Task1", Domain::range(0..=3));
    let task2 = Variable::new("Task2", Domain::range(0..=3));
    let task3 = Variable::new("Task3", Domain::range(0..=3));

    solver.add_variable(task1);
    solver.add_variable(task2);
    solver.add_variable(task3);

    // Task1 must finish before Task2 starts (assuming duration=1)
    // So Task2 > Task1
    solver.add_constraint(Constraint::Binary {
        var1: "Task2".to_string(),
        var2: "Task1".to_string(),
        relation: BinaryRelation::GreaterThan,
    });

    // Task2 must finish before Task3 starts
    solver.add_constraint(Constraint::Binary {
        var1: "Task3".to_string(),
        var2: "Task2".to_string(),
        relation: BinaryRelation::GreaterThan,
    });

    match solver.solve() {
        Some(solution) => {
            println!("  ✓ Schedule found:");
            println!("    Task 1 starts at time {}", solution["Task1"]);
            println!("    Task 2 starts at time {}", solution["Task2"]);
            println!("    Task 3 starts at time {}", solution["Task3"]);
            println!(
                "  Precedence satisfied: {} < {} < {}",
                solution["Task1"], solution["Task2"], solution["Task3"]
            );
        }
        None => println!("  ✗ No valid schedule exists"),
    }
    println!();
}

fn example_8_all_different() {
    println!("Example 7: All-Different Constraint");
    println!("Sudoku-style constraint: all variables must have different values\n");

    let mut solver = CspSolver::new();

    // 4 variables, domain 1-4
    for i in 1..=4 {
        let var = Variable::new(format!("V{}", i), Domain::range(1..=4));
        solver.add_variable(var);
    }

    // All different constraint
    solver.add_constraint(Constraint::all_different(vec!["V1", "V2", "V3", "V4"]));

    // Note: Our simplified solver doesn't fully implement AllDifferent,
    // but we can approximate with pairwise NotEqual constraints
    let vars = ["V1", "V2", "V3", "V4"];
    for i in 0..vars.len() {
        for j in (i + 1)..vars.len() {
            solver.add_constraint(Constraint::Binary {
                var1: vars[i].to_string(),
                var2: vars[j].to_string(),
                relation: BinaryRelation::NotEqual,
            });
        }
    }

    match solver.solve() {
        Some(solution) => {
            println!("  ✓ All-different solution:");
            for i in 1..=4 {
                println!("    V{} = {}", i, solution[&format!("V{}", i)]);
            }
        }
        None => println!("  ✗ No solution found"),
    }
    println!();
}

fn example_9_search_heuristics() {
    println!("Example 8: Search Heuristics");
    println!("Choose variables intelligently during search\n");

    // Create a problem where heuristics matter
    let mut solver = CspSolver::new();

    let x1 = Variable::new("X1", Domain::range(1..=10)); // Large domain
    let x2 = Variable::new("X2", Domain::range(1..=2)); // Small domain
    let x3 = Variable::new("X3", Domain::range(1..=10));

    solver.add_variable(x1);
    solver.add_variable(x2);
    solver.add_variable(x3);

    // MinDomain heuristic would choose X2 first
    println!("  Using MinDomainMaxDegree heuristic");
    println!("  (Prefers variables with smallest domains)");

    match solver.solve() {
        Some(_solution) => {
            println!(
                "  Solution found with {} assignments tried",
                solver.stats.assignments_tried
            );
        }
        None => println!("  No solution found"),
    }
    println!();
}

fn example_10_propagation() {
    println!("Example 9: Constraint Propagation");
    println!("Reduce search space through constraint propagation\n");

    let mut solver = CspSolver::new();
    solver.set_propagation(PropagationAlgorithm::ArcConsistency);

    let x = Variable::new("X", Domain::range(1..=5));
    let y = Variable::new("Y", Domain::range(1..=5));

    solver.add_variable(x);
    solver.add_variable(y);

    solver.add_constraint(Constraint::Binary {
        var1: "X".to_string(),
        var2: "Y".to_string(),
        relation: BinaryRelation::NotEqual,
    });

    println!("  Using Arc Consistency (AC-3) propagation");

    match solver.solve() {
        Some(solution) => {
            println!("  Solution: X={}, Y={}", solution["X"], solution["Y"]);
            println!("  Propagations performed: {}", solver.stats.propagations);
        }
        None => println!("  No solution found"),
    }
    println!();
}
