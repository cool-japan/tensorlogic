//! Abductive Reasoning Example
//!
//! This example demonstrates the compilation of abductive reasoning into tensor
//! computations. Abduction is the process of finding the best explanation for
//! observations given a set of possible hypotheses with associated costs.
//!
//! # Abductive Reasoning
//!
//! Given:
//! - Observations: facts we know to be true
//! - Hypotheses: potential explanations (abducibles) with costs
//! - Goal: Find the most plausible explanation
//!
//! The compiler generates an optimization objective:
//!   **maximize**: satisfaction(formula) - λ * total_cost(abducibles)
//!
//! # Operators
//!
//! - **Abducible(name, cost)**: Declares a hypothesis with an associated cost
//! - **Explain(formula)**: Marks a formula for explanation and cost minimization
//!
//! # Applications
//!
//! - Medical diagnosis (symptoms → diseases)
//! - Fault detection (failures → root causes)
//! - Robot planning (goals → actions)
//! - Scientific hypothesis generation
//! - Debugging (bugs → code errors)
//!
//! Run with:
//! ```bash
//! cargo run --example 23_abductive_reasoning
//! ```

use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() -> anyhow::Result<()> {
    println!("=== Abductive Reasoning in TensorLogic ===\n");
    println!("Abduction: Inference to the best explanation");
    println!("Find the most plausible hypotheses that explain observations.\n");

    // ========== Example 1: Simple Abducible ==========
    println!("Example 1: Single Abducible");
    println!("----------------------------");
    println!("Abducible(rain, 0.3) - \"Hypothesis: It rained (cost = 0.3)\"");
    println!("Low cost = more plausible hypothesis\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Event", 10);

    let rain_abducible = TLExpr::Abducible {
        name: "rain".to_string(),
        cost: 0.3,
    };

    let graph = compile_to_einsum_with_context(&rain_abducible, &mut ctx)?;
    println!("Abducible compiled:");
    println!(
        "  Tensors: {} (hypothesis tensor + cost tracking)",
        graph.tensors.len()
    );
    println!("  Nodes: {} (cost assignment)", graph.nodes.len());
    println!("  Representation: Tensor with metadata cost=0.3");
    println!("  Storage: Registered in context for later retrieval\n");

    // ========== Example 2: Simple Explanation ==========
    println!("Example 2: Explain with Single Hypothesis");
    println!("------------------------------------------");
    println!("Observation: WetGrass");
    println!("Hypothesis: Abducible(rain, 0.3)");
    println!("Explain: rain → WetGrass\n");

    let mut ctx2 = CompilerContext::new();
    ctx2.add_domain("Condition", 5);

    // Register abducible
    let rain = TLExpr::Abducible {
        name: "rain".to_string(),
        cost: 0.3,
    };

    // Causal rule: rain → WetGrass
    let rain_causes_wet_grass = TLExpr::imply(rain.clone(), TLExpr::pred("WetGrass", vec![]));

    // Explain the observation
    let explanation = TLExpr::Explain {
        formula: Box::new(rain_causes_wet_grass),
    };

    let graph = compile_to_einsum_with_context(&explanation, &mut ctx2)?;
    println!("Explanation compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(rain → WetGrass) - λ * cost(rain)");
    println!("  Optimization: Find rain value that maximizes objective");
    println!("  λ = 1.0 (default trade-off parameter)\n");

    // ========== Example 3: Multiple Competing Hypotheses ==========
    println!("Example 3: Competing Hypotheses");
    println!("--------------------------------");
    println!("Observation: WetGrass");
    println!("Hypothesis 1: Abducible(rain, 0.3) - Natural, common");
    println!("Hypothesis 2: Abducible(sprinkler, 0.5) - Artificial, less common");
    println!("Explain: (rain ∨ sprinkler) → WetGrass\n");

    let mut ctx3 = CompilerContext::new();
    ctx3.add_domain("Cause", 8);

    let rain3 = TLExpr::Abducible {
        name: "rain".to_string(),
        cost: 0.3,
    };

    let sprinkler = TLExpr::Abducible {
        name: "sprinkler".to_string(),
        cost: 0.5,
    };

    // Either cause leads to wet grass
    let causes = TLExpr::or(rain3, sprinkler);
    let causes_wet_grass = TLExpr::imply(causes, TLExpr::pred("WetGrass", vec![]));

    let explanation = TLExpr::Explain {
        formula: Box::new(causes_wet_grass),
    };

    let graph = compile_to_einsum_with_context(&explanation, &mut ctx3)?;
    println!("Competing hypotheses compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction - λ * (cost_rain + cost_sprinkler)");
    println!("  Expected result: Prefer 'rain' (lower cost)");
    println!("  Cost difference: 0.5 - 0.3 = 0.2 favors rain\n");

    // ========== Example 4: Medical Diagnosis ==========
    println!("Example 4: Medical Diagnosis");
    println!("-----------------------------");
    println!("Symptoms: Fever, Cough, Fatigue");
    println!("Possible diseases with costs (prevalence-based):");
    println!("  - Common Cold: cost=0.1 (very common)");
    println!("  - Flu: cost=0.3 (common)");
    println!("  - Pneumonia: cost=0.7 (rare)\n");

    let mut ctx4 = CompilerContext::new();
    ctx4.add_domain("Patient", 50);

    let cold = TLExpr::Abducible {
        name: "cold".to_string(),
        cost: 0.1,
    };

    let flu = TLExpr::Abducible {
        name: "flu".to_string(),
        cost: 0.3,
    };

    let pneumonia = TLExpr::Abducible {
        name: "pneumonia".to_string(),
        cost: 0.7,
    };

    // Symptom rules
    let cold_symptoms = TLExpr::imply(
        cold.clone(),
        TLExpr::and(
            TLExpr::pred("Cough", vec![Term::var("patient")]),
            TLExpr::pred("Fatigue", vec![Term::var("patient")]),
        ),
    );

    let flu_symptoms = TLExpr::imply(
        flu.clone(),
        TLExpr::and(
            TLExpr::and(
                TLExpr::pred("Fever", vec![Term::var("patient")]),
                TLExpr::pred("Cough", vec![Term::var("patient")]),
            ),
            TLExpr::pred("Fatigue", vec![Term::var("patient")]),
        ),
    );

    let pneumonia_symptoms = TLExpr::imply(
        pneumonia.clone(),
        TLExpr::and(
            TLExpr::and(
                TLExpr::pred("Fever", vec![Term::var("patient")]),
                TLExpr::pred("Cough", vec![Term::var("patient")]),
            ),
            TLExpr::pred("Fatigue", vec![Term::var("patient")]),
        ),
    );

    // Combine all rules
    let all_rules = TLExpr::and(cold_symptoms, TLExpr::and(flu_symptoms, pneumonia_symptoms));

    let diagnosis = TLExpr::Explain {
        formula: Box::new(all_rules),
    };

    let graph = compile_to_einsum_with_context(&diagnosis, &mut ctx4)?;
    println!("Medical diagnosis compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(symptoms|disease) - λ * Σ cost(diseases)");
    println!("  Expected: Flu is best explanation (matches all symptoms, moderate cost)");
    println!("  Use case: Differential diagnosis in healthcare AI\n");

    // ========== Example 5: Fault Detection ==========
    println!("Example 5: System Fault Detection");
    println!("----------------------------------");
    println!("Observation: ServerDown, NetworkSlow");
    println!("Possible faults:");
    println!("  - DiskFull: cost=0.2 (common)");
    println!("  - MemoryLeak: cost=0.4 (less common)");
    println!("  - HardwareFailure: cost=0.9 (rare)\n");

    let mut ctx5 = CompilerContext::new();
    ctx5.add_domain("Server", 20);

    let disk_full = TLExpr::Abducible {
        name: "disk_full".to_string(),
        cost: 0.2,
    };

    let memory_leak = TLExpr::Abducible {
        name: "memory_leak".to_string(),
        cost: 0.4,
    };

    let hardware_failure = TLExpr::Abducible {
        name: "hardware_failure".to_string(),
        cost: 0.9,
    };

    // Fault manifestation rules
    let disk_effects = TLExpr::imply(
        disk_full.clone(),
        TLExpr::pred("ServerDown", vec![Term::var("server")]),
    );

    let memory_effects = TLExpr::imply(
        memory_leak.clone(),
        TLExpr::and(
            TLExpr::pred("ServerDown", vec![Term::var("server")]),
            TLExpr::pred("NetworkSlow", vec![Term::var("server")]),
        ),
    );

    let hardware_effects = TLExpr::imply(
        hardware_failure.clone(),
        TLExpr::pred("ServerDown", vec![Term::var("server")]),
    );

    let fault_model = TLExpr::and(disk_effects, TLExpr::and(memory_effects, hardware_effects));

    let fault_diagnosis = TLExpr::Explain {
        formula: Box::new(fault_model),
    };

    let graph = compile_to_einsum_with_context(&fault_diagnosis, &mut ctx5)?;
    println!("Fault detection compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(symptoms|fault) - λ * Σ cost(faults)");
    println!("  Expected: MemoryLeak (explains both symptoms)");
    println!("  Use case: Automated root cause analysis\n");

    // ========== Example 6: Robot Planning ==========
    println!("Example 6: Robot Action Planning");
    println!("----------------------------------");
    println!("Goal: ObjectAtLocation(book, shelf)");
    println!("Possible actions:");
    println!("  - PickUp(book): cost=0.1 (easy)");
    println!("  - Move(shelf): cost=0.2 (moderate)");
    println!("  - Place(book, shelf): cost=0.1 (easy)\n");

    let mut ctx6 = CompilerContext::new();
    ctx6.add_domain("Object", 15);
    ctx6.add_domain("Location", 10);

    let pick_up = TLExpr::Abducible {
        name: "pick_up".to_string(),
        cost: 0.1,
    };

    let move_robot = TLExpr::Abducible {
        name: "move_robot".to_string(),
        cost: 0.2,
    };

    let place_object = TLExpr::Abducible {
        name: "place_object".to_string(),
        cost: 0.1,
    };

    // Action sequence
    let action_sequence = TLExpr::and(
        TLExpr::and(pick_up.clone(), move_robot.clone()),
        place_object.clone(),
    );

    // Goal achievement
    let goal_achieved = TLExpr::imply(
        action_sequence,
        TLExpr::pred(
            "ObjectAtLocation",
            vec![Term::constant("book"), Term::constant("shelf")],
        ),
    );

    let plan = TLExpr::Explain {
        formula: Box::new(goal_achieved),
    };

    let graph = compile_to_einsum_with_context(&plan, &mut ctx6)?;
    println!("Robot planning compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(goal|actions) - λ * Σ cost(actions)");
    println!("  Total cost: 0.1 + 0.2 + 0.1 = 0.4");
    println!("  Use case: Automated task planning, STRIPS-style reasoning\n");

    // ========== Example 7: Minimal Explanation ==========
    println!("Example 7: Minimal Explanation Preference");
    println!("------------------------------------------");
    println!("Observation: Alarm");
    println!("Possible causes:");
    println!("  - SmokeFire: cost=0.6 (explains alarm alone)");
    println!("  - Burglar: cost=0.8 (explains alarm alone)");
    println!("  - Both: cost=1.4 (unnecessary, Occam's razor violated)\n");

    let mut ctx7 = CompilerContext::new();
    ctx7.add_domain("Event", 12);

    let smoke_fire = TLExpr::Abducible {
        name: "smoke_fire".to_string(),
        cost: 0.6,
    };

    let burglar = TLExpr::Abducible {
        name: "burglar".to_string(),
        cost: 0.8,
    };

    // Either cause triggers alarm
    let alarm_rule = TLExpr::imply(
        TLExpr::or(smoke_fire.clone(), burglar.clone()),
        TLExpr::pred("Alarm", vec![]),
    );

    let minimal_explanation = TLExpr::Explain {
        formula: Box::new(alarm_rule),
    };

    let graph = compile_to_einsum_with_context(&minimal_explanation, &mut ctx7)?;
    println!("Minimal explanation compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(Alarm|cause) - λ * Σ cost(causes)");
    println!("  Expected: SmokeFire (single explanation, lower cost)");
    println!("  Principle: Occam's razor - prefer simpler explanations\n");

    // ========== Example 8: Nested Explanations ==========
    println!("Example 8: Hierarchical Explanation");
    println!("------------------------------------");
    println!("Observation: CarWontStart");
    println!("Primary hypotheses:");
    println!("  - BatteryDead: cost=0.2");
    println!("    - Secondary: LightsLeftOn (cost=0.1) or BatteryOld (cost=0.3)");
    println!("  - NoFuel: cost=0.4\n");

    let mut ctx8 = CompilerContext::new();
    ctx8.add_domain("Component", 8);

    let battery_dead = TLExpr::Abducible {
        name: "battery_dead".to_string(),
        cost: 0.2,
    };

    let lights_left_on = TLExpr::Abducible {
        name: "lights_left_on".to_string(),
        cost: 0.1,
    };

    let battery_old = TLExpr::Abducible {
        name: "battery_old".to_string(),
        cost: 0.3,
    };

    let no_fuel = TLExpr::Abducible {
        name: "no_fuel".to_string(),
        cost: 0.4,
    };

    // Battery failure causes
    let battery_failure_causes = TLExpr::imply(
        TLExpr::or(lights_left_on.clone(), battery_old.clone()),
        battery_dead.clone(),
    );

    // Primary causes
    let car_wont_start = TLExpr::imply(
        TLExpr::or(battery_dead.clone(), no_fuel.clone()),
        TLExpr::pred("CarWontStart", vec![]),
    );

    // Hierarchical explanation
    let hierarchical = TLExpr::and(battery_failure_causes, car_wont_start);

    let explanation = TLExpr::Explain {
        formula: Box::new(hierarchical),
    };

    let graph = compile_to_einsum_with_context(&explanation, &mut ctx8)?;
    println!("Hierarchical explanation compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: Multi-level cost minimization");
    println!("  Expected: BatteryDead (0.2) + LightsLeftOn (0.1) = 0.3 total");
    println!("  Use case: Cascading failure analysis, debugging\n");

    // ========== Example 9: Quantified Abduction ==========
    println!("Example 9: Quantified Abduction");
    println!("--------------------------------");
    println!("Observation: ∃sensor. HighReading(sensor)");
    println!("Hypothesis: Abducible(contamination, 0.5)");
    println!("Explain: contamination → ∀sensor. HighReading(sensor)\n");

    let mut ctx9 = CompilerContext::new();
    ctx9.add_domain("Sensor", 25);

    let contamination = TLExpr::Abducible {
        name: "contamination".to_string(),
        cost: 0.5,
    };

    // Contamination affects all sensors
    let all_sensors_affected = TLExpr::forall(
        "sensor",
        "Sensor",
        TLExpr::pred("HighReading", vec![Term::var("sensor")]),
    );

    let contamination_rule = TLExpr::imply(contamination.clone(), all_sensors_affected);

    let quantified_explanation = TLExpr::Explain {
        formula: Box::new(contamination_rule),
    };

    let graph = compile_to_einsum_with_context(&quantified_explanation, &mut ctx9)?;
    println!("Quantified abduction compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction(∀sensor. HighReading) - λ * 0.5");
    println!("  Semantics: Single cause explains all sensor readings");
    println!("  Use case: Environmental monitoring, anomaly detection\n");

    // ========== Example 10: Cost-Benefit Trade-off ==========
    println!("Example 10: Cost-Benefit Analysis");
    println!("-----------------------------------");
    println!("Observation: ProjectDelay");
    println!("Possible explanations:");
    println!("  - LackOfResources: cost=0.3 (easy to fix)");
    println!("  - TechnicalDebt: cost=0.7 (hard to fix)");
    println!("  - PoorPlanning: cost=0.5 (moderate)");
    println!("Explain: Find most cost-effective root cause\n");

    let mut ctx10 = CompilerContext::new();
    ctx10.add_domain("Project", 30);

    let lack_resources = TLExpr::Abducible {
        name: "lack_resources".to_string(),
        cost: 0.3,
    };

    let technical_debt = TLExpr::Abducible {
        name: "technical_debt".to_string(),
        cost: 0.7,
    };

    let poor_planning = TLExpr::Abducible {
        name: "poor_planning".to_string(),
        cost: 0.5,
    };

    // All causes can lead to delays
    let causes_delay = TLExpr::or(
        lack_resources.clone(),
        TLExpr::or(technical_debt.clone(), poor_planning.clone()),
    );

    let delay_rule = TLExpr::imply(
        causes_delay,
        TLExpr::pred("ProjectDelay", vec![Term::var("project")]),
    );

    let cost_benefit_analysis = TLExpr::Explain {
        formula: Box::new(delay_rule),
    };

    let graph = compile_to_einsum_with_context(&cost_benefit_analysis, &mut ctx10)?;
    println!("Cost-benefit analysis compiled:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Objective: satisfaction - λ * cost");
    println!("  Expected: LackOfResources (easiest to address)");
    println!("  Use case: Project management, risk mitigation\n");

    println!("=== Summary ===");
    println!("Abductive reasoning finds the best explanation by:");
    println!("  1. Declaring hypotheses as Abducible(name, cost)");
    println!("  2. Defining causal rules: hypothesis → observation");
    println!("  3. Using Explain to minimize total cost");
    println!("  4. Optimizing: maximize satisfaction - λ * Σ costs\n");
    println!("Applications:");
    println!("  • Medical diagnosis (symptoms → diseases)");
    println!("  • Fault detection (failures → root causes)");
    println!("  • Robot planning (goals → action sequences)");
    println!("  • Debugging (bugs → code errors)");
    println!("  • Scientific hypothesis generation\n");
    println!("Cost interpretation:");
    println!("  • Low cost = more plausible/common/easy");
    println!("  • High cost = less plausible/rare/difficult");
    println!("  • Occam's razor: prefer minimal explanations\n");

    Ok(())
}
