//! Constraint Programming Example
//!
//! This example demonstrates compilation of constraint programming operators
//! used in combinatorial optimization, scheduling, and planning problems.
//!
//! Run with:
//! ```bash
//! cargo run --example 20_constraint_programming
//! ```

use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::TLExpr;

fn main() -> anyhow::Result<()> {
    println!("=== Constraint Programming in TensorLogic ===\n");

    // ========== Example 1: AllDifferent - Basic Usage ==========
    println!("Example 1: AllDifferent - Basic Usage");
    println!("--------------------------------------");
    println!("Constraint: AllDifferent([x, y, z])");
    println!("Ensures x, y, and z all have different values\n");

    let all_different_basic = TLExpr::AllDifferent {
        variables: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    };

    let mut ctx1 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&all_different_basic, &mut ctx1)?;
    println!("Compiled graph:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Compiles to: (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)\n");

    // ========== Example 2: AllDifferent - N-Queens Pattern ==========
    println!("Example 2: AllDifferent - N-Queens Pattern");
    println!("-------------------------------------------");
    println!("In N-Queens, queens must be in different rows, columns, and diagonals");
    println!("Here we ensure 4 queens are in different columns:\n");

    let queens_columns = TLExpr::AllDifferent {
        variables: vec![
            "queen1_col".to_string(),
            "queen2_col".to_string(),
            "queen3_col".to_string(),
            "queen4_col".to_string(),
        ],
    };

    let mut ctx2 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&queens_columns, &mut ctx2)?;
    println!("4-Queens column constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!(
        "  Creates {} pairwise inequality constraints\n",
        (4 * 3) / 2
    );

    // ========== Example 3: Global Cardinality - Resource Allocation ==========
    println!("Example 3: GlobalCardinality - Resource Allocation");
    println!("---------------------------------------------------");
    println!("Assigning 5 tasks to 3 workers");
    println!("Each worker gets between 1 and 2 tasks\n");

    let tasks = vec![
        "task1_assignment".to_string(),
        "task2_assignment".to_string(),
        "task3_assignment".to_string(),
        "task4_assignment".to_string(),
        "task5_assignment".to_string(),
    ];

    let workers = vec![
        TLExpr::Constant(1.0), // Worker 1
        TLExpr::Constant(2.0), // Worker 2
        TLExpr::Constant(3.0), // Worker 3
    ];

    let min_tasks_per_worker = vec![1, 1, 1]; // Each worker gets at least 1 task
    let max_tasks_per_worker = vec![2, 2, 2]; // Each worker gets at most 2 tasks

    let resource_allocation = TLExpr::GlobalCardinality {
        variables: tasks.clone(),
        values: workers.clone(),
        min_occurrences: min_tasks_per_worker.clone(),
        max_occurrences: max_tasks_per_worker.clone(),
    };

    let mut ctx3 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&resource_allocation, &mut ctx3)?;
    println!("Resource allocation constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Ensures fair distribution: 1-2 tasks per worker\n");

    // ========== Example 4: GlobalCardinality - Course Scheduling ==========
    println!("Example 4: GlobalCardinality - Course Scheduling");
    println!("-------------------------------------------------");
    println!("Scheduling 6 courses across 3 time slots");
    println!("Each time slot can have 1-3 courses\n");

    let courses = vec![
        "course1_time".to_string(),
        "course2_time".to_string(),
        "course3_time".to_string(),
        "course4_time".to_string(),
        "course5_time".to_string(),
        "course6_time".to_string(),
    ];

    let time_slots = vec![
        TLExpr::Constant(9.0),  // 9:00 AM
        TLExpr::Constant(11.0), // 11:00 AM
        TLExpr::Constant(13.0), // 1:00 PM
    ];

    let min_courses_per_slot = vec![1, 1, 1]; // At least 1 course per slot
    let max_courses_per_slot = vec![3, 3, 3]; // At most 3 courses per slot

    let course_scheduling = TLExpr::GlobalCardinality {
        variables: courses.clone(),
        values: time_slots.clone(),
        min_occurrences: min_courses_per_slot,
        max_occurrences: max_courses_per_slot,
    };

    let mut ctx4 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&course_scheduling, &mut ctx4)?;
    println!("Course scheduling constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Balances courses: 1-3 per time slot\n");

    // ========== Example 5: Combined Constraints - Sudoku Pattern ==========
    println!("Example 5: Combined Constraints - Sudoku Pattern");
    println!("-------------------------------------------------");
    println!("In a 4x4 Sudoku-like puzzle:");
    println!("- Each row must have all different values");
    println!("- Values 1,2,3,4 must each appear exactly once per row\n");

    // Row constraint: all cells in row must be different
    let row1_cells = vec![
        "r1c1".to_string(),
        "r1c2".to_string(),
        "r1c3".to_string(),
        "r1c4".to_string(),
    ];

    let row_all_different = TLExpr::AllDifferent {
        variables: row1_cells.clone(),
    };

    // Cardinality constraint: each value 1-4 appears exactly once
    let values = vec![
        TLExpr::Constant(1.0),
        TLExpr::Constant(2.0),
        TLExpr::Constant(3.0),
        TLExpr::Constant(4.0),
    ];

    let exactly_once = vec![1, 1, 1, 1]; // Each value exactly once

    let row_cardinality = TLExpr::GlobalCardinality {
        variables: row1_cells.clone(),
        values: values.clone(),
        min_occurrences: exactly_once.clone(),
        max_occurrences: exactly_once.clone(),
    };

    // Combine both constraints
    let sudoku_row = TLExpr::and(row_all_different, row_cardinality);

    let mut ctx5 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&sudoku_row, &mut ctx5)?;
    println!("Sudoku row constraints compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Combines AllDifferent + GlobalCardinality\n");

    // ========== Example 6: Graph Coloring ==========
    println!("Example 6: Graph Coloring");
    println!("-------------------------");
    println!("Coloring a graph with 4 vertices and 5 edges");
    println!("Adjacent vertices must have different colors\n");

    // Graph edges (vertices that must be different)
    let edge1 = TLExpr::AllDifferent {
        variables: vec!["v1_color".to_string(), "v2_color".to_string()],
    };
    let edge2 = TLExpr::AllDifferent {
        variables: vec!["v2_color".to_string(), "v3_color".to_string()],
    };
    let edge3 = TLExpr::AllDifferent {
        variables: vec!["v3_color".to_string(), "v4_color".to_string()],
    };
    let edge4 = TLExpr::AllDifferent {
        variables: vec!["v4_color".to_string(), "v1_color".to_string()],
    };
    let edge5 = TLExpr::AllDifferent {
        variables: vec!["v1_color".to_string(), "v3_color".to_string()],
    };

    // Combine all edge constraints
    let graph_coloring = vec![edge1, edge2, edge3, edge4, edge5]
        .into_iter()
        .reduce(TLExpr::and)
        .unwrap();

    let mut ctx6 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&graph_coloring, &mut ctx6)?;
    println!("Graph coloring constraints compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  5 edge constraints combined with AND\n");

    // ========== Example 7: Load Balancing ==========
    println!("Example 7: Load Balancing");
    println!("-------------------------");
    println!("Distributing 10 jobs across 3 servers");
    println!("Each server handles 3-4 jobs for balanced load\n");

    let jobs: Vec<String> = (1..=10).map(|i| format!("job{}_server", i)).collect();

    let servers = vec![
        TLExpr::Constant(1.0), // Server A
        TLExpr::Constant(2.0), // Server B
        TLExpr::Constant(3.0), // Server C
    ];

    let min_jobs = vec![3, 3, 3]; // Each server gets at least 3 jobs
    let max_jobs = vec![4, 4, 4]; // Each server gets at most 4 jobs

    let load_balancing = TLExpr::GlobalCardinality {
        variables: jobs.clone(),
        values: servers.clone(),
        min_occurrences: min_jobs,
        max_occurrences: max_jobs,
    };

    let mut ctx7 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&load_balancing, &mut ctx7)?;
    println!("Load balancing constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Ensures even distribution: 3-4 jobs per server\n");

    // ========== Example 8: Team Assignment with Skills ==========
    println!("Example 8: Team Assignment with Skills");
    println!("---------------------------------------");
    println!("Assigning 6 people to 2 teams");
    println!("Each team gets exactly 3 people\n");

    let people = vec![
        "alice_team".to_string(),
        "bob_team".to_string(),
        "carol_team".to_string(),
        "dave_team".to_string(),
        "eve_team".to_string(),
        "frank_team".to_string(),
    ];

    let teams = vec![
        TLExpr::Constant(1.0), // Team A
        TLExpr::Constant(2.0), // Team B
    ];

    let exactly_3 = vec![3, 3]; // Exactly 3 people per team

    let team_assignment = TLExpr::GlobalCardinality {
        variables: people.clone(),
        values: teams.clone(),
        min_occurrences: exactly_3.clone(),
        max_occurrences: exactly_3.clone(),
    };

    let mut ctx8 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&team_assignment, &mut ctx8)?;
    println!("Team assignment constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Ensures balanced teams: exactly 3 per team\n");

    // ========== Example 9: Slot Filling with Capacity ==========
    println!("Example 9: Slot Filling with Capacity");
    println!("--------------------------------------");
    println!("Scheduling 8 meetings in 4 rooms");
    println!("Each room can host 1-3 meetings simultaneously\n");

    let meetings: Vec<String> = (1..=8).map(|i| format!("meeting{}_room", i)).collect();

    let rooms = vec![
        TLExpr::Constant(101.0), // Room 101
        TLExpr::Constant(102.0), // Room 102
        TLExpr::Constant(103.0), // Room 103
        TLExpr::Constant(104.0), // Room 104
    ];

    let min_meetings = vec![1, 1, 1, 1]; // Each room gets at least 1 meeting
    let max_meetings = vec![3, 3, 3, 3]; // Each room gets at most 3 meetings

    let room_scheduling = TLExpr::GlobalCardinality {
        variables: meetings.clone(),
        values: rooms.clone(),
        min_occurrences: min_meetings,
        max_occurrences: max_meetings,
    };

    let mut ctx9 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&room_scheduling, &mut ctx9)?;
    println!("Room scheduling constraint compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Respects room capacity: 1-3 meetings per room\n");

    // ========== Example 10: Complex Constraint Composition ==========
    println!("Example 10: Complex Constraint Composition");
    println!("-------------------------------------------");
    println!("Tournament scheduling:");
    println!("- 4 teams play in 3 rounds");
    println!("- Each team plays exactly once per round (AllDifferent per round)");
    println!("- Each team plays 1-2 times total (GlobalCardinality)\n");

    // Round 1: teams must be different
    let round1 = TLExpr::AllDifferent {
        variables: vec!["match1_team".to_string(), "match2_team".to_string()],
    };

    // Round 2: teams must be different
    let round2 = TLExpr::AllDifferent {
        variables: vec!["match3_team".to_string(), "match4_team".to_string()],
    };

    // Round 3: teams must be different
    let round3 = TLExpr::AllDifferent {
        variables: vec!["match5_team".to_string(), "match6_team".to_string()],
    };

    // All matches across all rounds
    let all_matches = vec![
        "match1_team".to_string(),
        "match2_team".to_string(),
        "match3_team".to_string(),
        "match4_team".to_string(),
        "match5_team".to_string(),
        "match6_team".to_string(),
    ];

    let team_ids = vec![
        TLExpr::Constant(1.0), // Team 1
        TLExpr::Constant(2.0), // Team 2
        TLExpr::Constant(3.0), // Team 3
        TLExpr::Constant(4.0), // Team 4
    ];

    // Each team appears 1-2 times
    let appearances = TLExpr::GlobalCardinality {
        variables: all_matches.clone(),
        values: team_ids.clone(),
        min_occurrences: vec![1, 1, 1, 1],
        max_occurrences: vec![2, 2, 2, 2],
    };

    // Combine all constraints
    let tournament = vec![round1, round2, round3, appearances]
        .into_iter()
        .reduce(TLExpr::and)
        .unwrap();

    let mut ctx10 = CompilerContext::new();
    let graph = compile_to_einsum_with_context(&tournament, &mut ctx10)?;
    println!("Tournament scheduling constraints compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Combines 3 AllDifferent + 1 GlobalCardinality constraints\n");

    println!("=== Summary ===");
    println!("\nConstraint programming operators compiled successfully!");
    println!("Applications demonstrated:");
    println!("  • AllDifferent: N-Queens, Graph Coloring, Sudoku");
    println!("  • GlobalCardinality: Resource allocation, Load balancing, Scheduling");
    println!("  • Combined: Tournament scheduling, Course scheduling");
    println!("\nThese constraints compile to tensor operations:");
    println!("  • AllDifferent → Pairwise inequality checks");
    println!("  • GlobalCardinality → Count aggregations with bounds");
    println!("\nUseful for combinatorial optimization and planning problems!");

    Ok(())
}
