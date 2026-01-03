//! Example demonstrating Model Soups - Weight-space averaging
//!
//! Model Soups is a technique from 2022 that averages the *weights* (not predictions)
//! of multiple fine-tuned models to improve accuracy without increasing inference cost.
//!
//! Key benefits:
//! - Improves accuracy compared to individual models
//! - No inference overhead (single model at test time)
//! - Works across different hyperparameters and random seeds
//! - Particularly effective for models fine-tuned from same initialization
//!
//! Reference:
//! "Model soups: averaging weights of multiple fine-tuned models improves
//! accuracy without increasing inference time"
//! Mitchell Wortsman et al., ICML 2022
//! https://arxiv.org/abs/2203.05482

use scirs2_core::ndarray::array;
use std::collections::HashMap;
use tensorlogic_train::ModelSoup;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Soups Example ===\n");
    println!("Weight-space averaging for improved generalization\n");

    // Example 1: Uniform Soup (simple averaging)
    println!("Example 1: Uniform Soup");
    println!("-----------------------");
    println!("Scenario: Three models fine-tuned with different learning rates\n");

    // Model 1: Fine-tuned with lr=0.001
    let mut model1_weights = HashMap::new();
    model1_weights.insert("weight".to_string(), array![[2.1, 1.9], [0.9, 1.1]]);
    model1_weights.insert("bias".to_string(), array![[0.95, 1.05]]);
    println!("Model 1 (lr=0.001): validation accuracy = 0.82");

    // Model 2: Fine-tuned with lr=0.01
    let mut model2_weights = HashMap::new();
    model2_weights.insert("weight".to_string(), array![[1.8, 2.2], [1.1, 0.9]]);
    model2_weights.insert("bias".to_string(), array![[1.1, 0.9]]);
    println!("Model 2 (lr=0.01):  validation accuracy = 0.85");

    // Model 3: Fine-tuned with lr=0.001 (different seed)
    let mut model3_weights = HashMap::new();
    model3_weights.insert("weight".to_string(), array![[2.0, 2.0], [1.0, 1.0]]);
    model3_weights.insert("bias".to_string(), array![[1.0, 1.0]]);
    println!("Model 3 (lr=0.001, seed=42): validation accuracy = 0.83\n");

    // Create uniform soup
    let uniform_soup = ModelSoup::uniform_soup(vec![
        model1_weights.clone(),
        model2_weights.clone(),
        model3_weights.clone(),
    ])?;

    println!("Uniform Soup:");
    println!("  Recipe: {:?}", uniform_soup.recipe());
    println!("  Number of models: {}", uniform_soup.num_models());
    println!("  Averaged weights:");
    let soup_weight = uniform_soup.get_parameter("weight").unwrap();
    println!("    weight = {:?}", soup_weight);
    let soup_bias = uniform_soup.get_parameter("bias").unwrap();
    println!("    bias = {:?}", soup_bias);
    println!("  Expected validation accuracy: 0.87 (improved!)");
    println!();

    // Example 2: Greedy Soup
    println!("Example 2: Greedy Soup");
    println!("---------------------");
    println!("Iteratively add models that improve validation performance\n");

    // Collect validation accuracies
    let val_accuracies = vec![0.82, 0.85, 0.83, 0.80, 0.84];
    println!("5 models with validation accuracies:");
    for (i, acc) in val_accuracies.iter().enumerate() {
        println!("  Model {}: {:.2}", i + 1, acc);
    }
    println!();

    // Create model weights (simplified for demo)
    let mut model_weights_collection = vec![];
    for i in 0..5 {
        let mut weights = HashMap::new();
        let offset = i as f64 * 0.1;
        weights.insert(
            "weight".to_string(),
            array![[2.0 + offset, 2.0 - offset], [1.0 + offset, 1.0 - offset]],
        );
        weights.insert("bias".to_string(), array![[1.0 + offset, 1.0 - offset]]);
        model_weights_collection.push(weights);
    }

    let greedy_soup = ModelSoup::greedy_soup(model_weights_collection.clone(), val_accuracies)?;

    println!("Greedy Soup:");
    println!("  Recipe: {:?}", greedy_soup.recipe());
    println!("  Number of models selected: {}", greedy_soup.num_models());
    println!("  (Started with best model, added others that improved performance)");
    println!("  Expected validation accuracy: 0.88 (best greedy selection)");
    println!();

    // Example 3: Weighted Soup
    println!("Example 3: Weighted Soup");
    println!("-----------------------");
    println!("Weight models based on validation performance\n");

    // Use accuracy as weights
    let model_weights_vec = vec![
        model1_weights.clone(),
        model2_weights.clone(),
        model3_weights.clone(),
    ];
    let accuracy_weights = vec![0.82, 0.85, 0.83]; // Use accuracies as weights

    let weighted_soup = ModelSoup::weighted_soup(model_weights_vec, accuracy_weights)?;

    println!("Weighted Soup:");
    println!("  Recipe: {:?}", weighted_soup.recipe());
    println!("  Weights: [0.82, 0.85, 0.83] (normalized)");
    let weighted_weight = weighted_soup.get_parameter("weight").unwrap();
    println!("  Averaged weights:");
    println!("    weight = {:?}", weighted_weight);
    println!("  (Better models get more influence)");
    println!();

    // Example 4: Real-world scenario - Hyperparameter grid search
    println!("Example 4: Hyperparameter Grid Search Soup");
    println!("------------------------------------------");
    println!("Combine models from grid search without picking a single winner\n");

    // Simulate grid search over learning rates and batch sizes
    let grid_configs = vec![
        ("lr=0.001, bs=32", 0.81),
        ("lr=0.001, bs=64", 0.83),
        ("lr=0.01, bs=32", 0.85),
        ("lr=0.01, bs=64", 0.82),
        ("lr=0.1, bs=32", 0.79),
        ("lr=0.1, bs=64", 0.80),
    ];

    println!("Grid search results:");
    for (config, acc) in &grid_configs {
        println!("  {}: acc = {:.2}", config, acc);
    }
    println!();

    // Create model weights for each config
    let mut grid_weights = vec![];
    let mut grid_accs = vec![];
    for (idx, (_, acc)) in grid_configs.iter().enumerate() {
        let mut weights = HashMap::new();
        let offset = idx as f64 * 0.05;
        weights.insert(
            "weight".to_string(),
            array![[2.0 + offset, 2.0 - offset], [1.0, 1.0]],
        );
        weights.insert("bias".to_string(), array![[1.0, 1.0]]);
        grid_weights.push(weights);
        grid_accs.push(*acc);
    }

    // Strategy 1: Uniform soup of all configs
    let _grid_uniform = ModelSoup::uniform_soup(grid_weights.clone())?;
    println!("Uniform Soup (all configs): expected acc = 0.84");

    // Strategy 2: Greedy soup
    let grid_greedy = ModelSoup::greedy_soup(grid_weights.clone(), grid_accs.clone())?;
    println!(
        "Greedy Soup: {} models selected, expected acc = 0.86",
        grid_greedy.num_models()
    );
    println!();

    // Example 5: Comparison with ensemble methods
    println!("Example 5: Soup vs Ensemble Comparison");
    println!("--------------------------------------");
    println!();
    println!("Traditional Ensemble (prediction averaging):");
    println!("  - Averages *predictions* at inference time");
    println!("  - Inference cost: N × single model");
    println!("  - Memory: Need to store all N models");
    println!("  - Example: 3 models → 3× slower inference");
    println!();
    println!("Model Soup (weight averaging):");
    println!("  - Averages *weights* before inference");
    println!("  - Inference cost: Same as single model!");
    println!("  - Memory: Only need final averaged model");
    println!("  - Example: 3 models → No slowdown!");
    println!();
    println!("When to use Model Soups:");
    println!("  ✓ Models fine-tuned from same initialization");
    println!("  ✓ Different hyperparameters or random seeds");
    println!("  ✓ Need fast inference (production deployment)");
    println!("  ✓ Limited memory budget");
    println!();
    println!("When to use Traditional Ensembles:");
    println!("  ✓ Models with different architectures");
    println!("  ✓ Models trained on different data");
    println!("  ✓ Inference time not critical");
    println!("  ✓ Want maximum accuracy (at cost of speed)");
    println!();

    // Example 6: Loading soup weights into a model
    println!("Example 6: Using Soup Weights");
    println!("-----------------------------");
    println!();

    let soup = ModelSoup::uniform_soup(vec![model1_weights, model2_weights, model3_weights])?;

    println!("Step 1: Create soup from fine-tuned models");
    println!("Step 2: Extract averaged weights");
    let averaged_weights = soup.into_weights();

    println!("Step 3: Load into fresh model:");
    println!("  model.load_state_dict(averaged_weights);");
    println!();
    println!("Step 4: Deploy!");
    println!("  - Single model inference");
    println!("  - Improved accuracy from soup");
    println!("  - No computational overhead");
    println!();

    println!("Saved {} parameters from soup", averaged_weights.len());

    // Key takeaways
    println!("Key Takeaways:");
    println!("=============");
    println!();
    println!("1. **Simple yet powerful**: Just average weights, big improvement");
    println!("2. **No inference cost**: Unlike ensembles, no speed penalty");
    println!("3. **Three recipes**:");
    println!("   - Uniform: Average all models equally");
    println!("   - Greedy: Select models that improve validation");
    println!("   - Weighted: Weight by validation performance");
    println!("4. **Best for**: Fine-tuned models from same initialization");
    println!("5. **Typical gains**: 1-2% accuracy improvement for free");
    println!("6. **Production-ready**: Deploy as single model");
    println!();
    println!("7. **Empirical findings** (from paper):");
    println!("   - Works across vision, NLP, and multimodal tasks");
    println!("   - ImageNet: +1.5% accuracy over best single model");
    println!("   - CLIP: +3.0% zero-shot accuracy");
    println!("   - Uniform soup often sufficient");
    println!("   - Greedy soup provides marginal gains");
    println!();
    println!("8. **When it fails**:");
    println!("   - Models trained with completely different methods");
    println!("   - Vastly different architectures");
    println!("   - Models in different parts of weight space");

    Ok(())
}
