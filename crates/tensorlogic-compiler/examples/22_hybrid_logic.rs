//! Hybrid Logic Example
//!
//! This example demonstrates the compilation of hybrid logic operators into tensor
//! computations. Hybrid logic extends modal logic with nominals (named states) and
//! satisfaction operators for reasoning about specific states in a state space.
//!
//! # Hybrid Logic Operators
//!
//! - **Nominal (@i)**: A named state in the state space (one-hot vector)
//! - **At (@i φ)**: Formula φ is satisfied at the specific state i
//! - **Somewhere (E φ)**: Formula φ is satisfied at some reachable state (∃)
//! - **Everywhere (A φ)**: Formula φ is satisfied at all reachable states (∀)
//!
//! # Applications
//!
//! - Knowledge representation with named individuals
//! - Multi-agent systems with agent-specific states
//! - Temporal reasoning with named time points
//! - Graph reachability with named nodes
//!
//! Run with:
//! ```bash
//! cargo run --example 22_hybrid_logic
//! ```

use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() -> anyhow::Result<()> {
    println!("=== Hybrid Logic in TensorLogic ===\n");
    println!("Hybrid logic combines modal logic with named states (nominals)");
    println!("for precise reasoning about specific points in a state space.\n");

    // ========== Example 1: Simple Nominal ==========
    println!("Example 1: Nominal (Named State)");
    println!("---------------------------------");
    println!("@home - \"The state named 'home'\"");
    println!("Represents a specific state in the state space\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Agent", 5); // Domain for agents (though not used in this example)

    let home_nominal = TLExpr::Nominal {
        name: "home".to_string(),
    };

    let graph = compile_to_einsum_with_context(&home_nominal, &mut ctx)?;
    println!("Compiled nominal:");
    println!(
        "  Tensors: {} (one-hot vector over state space)",
        graph.tensors.len()
    );
    println!("  Nodes: {} (nominal creation)", graph.nodes.len());
    println!("  Representation: One-hot encoding where state 'home' = 1, others = 0\n");

    // ========== Example 2: At Operator ==========
    println!("Example 2: At Operator (@i φ)");
    println!("------------------------------");
    println!("@home Safe(agent) - \"Agent is safe at the home state\"");
    println!("Evaluates a formula at a specific named state\n");

    let mut ctx2 = CompilerContext::new();
    ctx2.add_domain("Agent", 5);

    let at_home_safe = TLExpr::At {
        nominal: "home".to_string(),
        formula: Box::new(TLExpr::pred("Safe", vec![Term::var("agent")])),
    };

    let graph = compile_to_einsum_with_context(&at_home_safe, &mut ctx2)?;
    println!("At operator compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Projects formula onto the 'home' state only\n");

    // ========== Example 3: Somewhere (Existential Modal) ==========
    println!("Example 3: Somewhere (E φ)");
    println!("---------------------------");
    println!("E Safe(agent) - \"Agent is safe at some reachable state\"");
    println!("Existential quantification over reachable states\n");

    let mut ctx3 = CompilerContext::new();
    ctx3.add_domain("Agent", 5);

    let somewhere_safe = TLExpr::Somewhere {
        formula: Box::new(TLExpr::pred("Safe", vec![Term::var("agent")])),
    };

    let graph = compile_to_einsum_with_context(&somewhere_safe, &mut ctx3)?;
    println!("Somewhere operator compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: max(Safe(agent, state)) over all reachable states");
    println!("  Use case: \"Is there any state where the agent is safe?\"\n");

    // ========== Example 4: Everywhere (Universal Modal) ==========
    println!("Example 4: Everywhere (A φ)");
    println!("----------------------------");
    println!("A Safe(agent) - \"Agent is safe at all reachable states\"");
    println!("Universal quantification over reachable states\n");

    let mut ctx4 = CompilerContext::new();
    ctx4.add_domain("Agent", 5);

    let everywhere_safe = TLExpr::Everywhere {
        formula: Box::new(TLExpr::pred("Safe", vec![Term::var("agent")])),
    };

    let graph = compile_to_einsum_with_context(&everywhere_safe, &mut ctx4)?;
    println!("Everywhere operator compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: min(Safe(agent, state)) over all reachable states");
    println!("  Use case: \"Is the agent safe in all possible states?\"\n");

    // ========== Example 5: Nested Operators ==========
    println!("Example 5: Nested Hybrid Logic");
    println!("-------------------------------");
    println!("@start E @goal Reachable");
    println!("\"From the start state, there exists a path to the goal state\"\n");

    let mut ctx5 = CompilerContext::new();
    ctx5.add_domain("Node", 10);

    let at_goal_reachable = TLExpr::At {
        nominal: "goal".to_string(),
        formula: Box::new(TLExpr::pred("Reachable", vec![])),
    };

    let somewhere_goal = TLExpr::Somewhere {
        formula: Box::new(at_goal_reachable),
    };

    let at_start_somewhere_goal = TLExpr::At {
        nominal: "start".to_string(),
        formula: Box::new(somewhere_goal),
    };

    let graph = compile_to_einsum_with_context(&at_start_somewhere_goal, &mut ctx5)?;
    println!("Nested hybrid logic compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Evaluates reachability from 'start' to 'goal' state");
    println!("  Use case: Path planning, reachability analysis\n");

    // ========== Example 6: Combining with Standard Logic ==========
    println!("Example 6: Hybrid Logic with Conjunction");
    println!("-----------------------------------------");
    println!("@home (Safe ∧ Comfortable)");
    println!("\"At home, it is both safe and comfortable\"\n");

    let mut ctx6 = CompilerContext::new();
    ctx6.add_domain("Location", 8);

    let safe_and_comfortable = TLExpr::and(
        TLExpr::pred("Safe", vec![]),
        TLExpr::pred("Comfortable", vec![]),
    );

    let at_home_safe_comfortable = TLExpr::At {
        nominal: "home".to_string(),
        formula: Box::new(safe_and_comfortable),
    };

    let graph = compile_to_einsum_with_context(&at_home_safe_comfortable, &mut ctx6)?;
    println!("Hybrid logic with conjunction compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Both predicates must hold at the 'home' state\n");

    // ========== Example 7: Multi-Agent Reasoning ==========
    println!("Example 7: Multi-Agent Scenario");
    println!("--------------------------------");
    println!("E (∃agent. @agent Happy(agent))");
    println!("\"There exists a state where some agent is happy at their location\"\n");

    let mut ctx7 = CompilerContext::new();
    ctx7.add_domain("Agent", 3);

    // Inner formula: Happy(agent) at a specific location
    let happy_at_location = TLExpr::At {
        nominal: "agent_location".to_string(),
        formula: Box::new(TLExpr::pred("Happy", vec![Term::var("agent")])),
    };

    // Existential quantification over agents
    let exists_happy_agent = TLExpr::exists("agent", "Agent", happy_at_location);

    // Somewhere over states
    let somewhere_happy = TLExpr::Somewhere {
        formula: Box::new(exists_happy_agent),
    };

    let graph = compile_to_einsum_with_context(&somewhere_happy, &mut ctx7)?;
    println!("Multi-agent scenario compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Combines spatial reasoning (states) with agent reasoning\n");

    // ========== Example 8: Reachability with Constraints ==========
    println!("Example 8: Constrained Reachability");
    println!("------------------------------------");
    println!("A (Safe → E @goal Arrived)");
    println!("\"In all states, if safe, then goal is reachable\"\n");

    let mut ctx8 = CompilerContext::new();
    ctx8.add_domain("State", 6);

    let goal_reached = TLExpr::At {
        nominal: "goal".to_string(),
        formula: Box::new(TLExpr::pred("Arrived", vec![])),
    };

    let goal_reachable = TLExpr::Somewhere {
        formula: Box::new(goal_reached),
    };

    let safe_implies_reachable = TLExpr::imply(TLExpr::pred("Safe", vec![]), goal_reachable);

    let everywhere_safe_reachable = TLExpr::Everywhere {
        formula: Box::new(safe_implies_reachable),
    };

    let graph = compile_to_einsum_with_context(&everywhere_safe_reachable, &mut ctx8)?;
    println!("Constrained reachability compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Safety condition ensures goal reachability");
    println!("  Use case: Safe path planning, robot navigation\n");

    // ========== Example 9: Bidirectional Reachability ==========
    println!("Example 9: Bidirectional Reachability");
    println!("--------------------------------------");
    println!("(@start E @goal Reachable) ∧ (@goal E @start Reachable)");
    println!("\"Start and goal are mutually reachable\"\n");

    let mut ctx9 = CompilerContext::new();
    ctx9.add_domain("Vertex", 15);

    // Forward reachability: start to goal
    let forward = TLExpr::At {
        nominal: "start".to_string(),
        formula: Box::new(TLExpr::Somewhere {
            formula: Box::new(TLExpr::At {
                nominal: "goal".to_string(),
                formula: Box::new(TLExpr::pred("Reachable", vec![])),
            }),
        }),
    };

    // Backward reachability: goal to start
    let backward = TLExpr::At {
        nominal: "goal".to_string(),
        formula: Box::new(TLExpr::Somewhere {
            formula: Box::new(TLExpr::At {
                nominal: "start".to_string(),
                formula: Box::new(TLExpr::pred("Reachable", vec![])),
            }),
        }),
    };

    let bidirectional = TLExpr::and(forward, backward);

    let graph = compile_to_einsum_with_context(&bidirectional, &mut ctx9)?;
    println!("Bidirectional reachability compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Checks for cycles or bidirectional paths");
    println!("  Use case: Strongly connected components, reversible paths\n");

    // ========== Example 10: Named Checkpoints ==========
    println!("Example 10: Sequential Checkpoints");
    println!("-----------------------------------");
    println!("@checkpoint1 E (@checkpoint2 E @checkpoint3 Completed)");
    println!("\"From checkpoint1, reach checkpoint2, then checkpoint3\"\n");

    let mut ctx10 = CompilerContext::new();
    ctx10.add_domain("Waypoint", 20);

    let at_checkpoint3 = TLExpr::At {
        nominal: "checkpoint3".to_string(),
        formula: Box::new(TLExpr::pred("Completed", vec![])),
    };

    let reach_checkpoint3 = TLExpr::Somewhere {
        formula: Box::new(at_checkpoint3),
    };

    let at_checkpoint2 = TLExpr::At {
        nominal: "checkpoint2".to_string(),
        formula: Box::new(reach_checkpoint3),
    };

    let reach_checkpoint2 = TLExpr::Somewhere {
        formula: Box::new(at_checkpoint2),
    };

    let at_checkpoint1 = TLExpr::At {
        nominal: "checkpoint1".to_string(),
        formula: Box::new(reach_checkpoint2),
    };

    let graph = compile_to_einsum_with_context(&at_checkpoint1, &mut ctx10)?;
    println!("Sequential checkpoints compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Semantics: Sequential reachability through named waypoints");
    println!("  Use case: Mission planning, multi-stage objectives\n");

    println!("=== Summary ===");
    println!("Hybrid logic extends modal logic with:");
    println!("  • Nominals: Named states for precise reference");
    println!("  • @-operator: Evaluate formulas at specific states");
    println!("  • E/A operators: Existential/universal over reachable states");
    println!("\nApplications:");
    println!("  • Knowledge graphs with named entities");
    println!("  • Multi-agent coordination");
    println!("  • Path planning with waypoints");
    println!("  • Temporal reasoning with named events");
    println!("  • Reachability analysis in state spaces\n");

    Ok(())
}
