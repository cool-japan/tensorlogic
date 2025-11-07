#!/usr/bin/env python3
"""
Training Workflow Example for TensorLogic-py

This example demonstrates the complete training API including:
1. Loss functions (MSE, BCE, Cross-Entropy)
2. Optimizers (SGD, Adam, RMSprop)
3. Callbacks (EarlyStopping, ModelCheckpoint, Logger)
4. Trainer class for custom training loops
5. Convenience fit() function for quick training
6. Training history and visualization

Note: This demonstrates the training API structure. Actual parameter
optimization requires gradient computation which is not yet implemented.
"""

import numpy as np
import pytensorlogic as tl


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_1_basic_training():
    """Example 1: Basic training with Trainer class."""
    print_section("Example 1: Basic Training with Trainer Class")

    # Create a simple logical expression
    expr = tl.pred("person", [tl.var("x")])
    graph = tl.compile(expr)

    # Create synthetic training data
    np.random.seed(42)
    train_inputs = {"person": np.random.rand(10, 10)}
    train_targets = np.random.rand(10, 10)

    # Create loss function and optimizer
    loss_fn = tl.mse_loss()
    optimizer = tl.adam(learning_rate=0.01)

    # Create trainer
    trainer = tl.Trainer(graph, loss_fn, optimizer)

    # Train the model
    print("Training for 10 epochs...")
    history = trainer.fit(train_inputs, train_targets, epochs=10, verbose=1)

    # Print final results
    print(f"\nTraining complete!")
    print(f"Final training loss: {history.train_losses[-1]:.6f}")
    print(f"Best training loss: {history.best_train_loss()[1]:.6f} at epoch {history.best_train_loss()[0] + 1}")


def example_2_different_loss_functions():
    """Example 2: Training with different loss functions."""
    print_section("Example 2: Different Loss Functions")

    expr = tl.pred("x", [tl.var("i")])
    graph = tl.compile(expr)

    np.random.seed(42)
    inputs = {"x": np.random.rand(5, 5)}
    targets = np.random.rand(5, 5)

    # Test different loss functions
    loss_functions = [
        ("MSE", tl.mse_loss()),
        ("BCE", tl.bce_loss()),
        ("Cross-Entropy", tl.cross_entropy_loss()),
    ]

    for name, loss_fn in loss_functions:
        print(f"\nTraining with {name} loss:")
        trainer = tl.Trainer(graph, loss_fn, tl.adam())
        history = trainer.fit(inputs, targets, epochs=5, verbose=0)
        print(f"  Final loss: {history.train_losses[-1]:.6f}")


def example_3_different_optimizers():
    """Example 3: Training with different optimizers."""
    print_section("Example 3: Different Optimizers")

    expr = tl.pred("x", [tl.var("i")])
    graph = tl.compile(expr)

    np.random.seed(42)
    inputs = {"x": np.random.rand(5, 5)}
    targets = np.random.rand(5, 5)

    # Test different optimizers
    optimizers = [
        ("SGD (lr=0.01)", tl.sgd(learning_rate=0.01)),
        ("SGD with momentum", tl.sgd(learning_rate=0.01, momentum=0.9)),
        ("Adam", tl.adam(learning_rate=0.001)),
        ("RMSprop", tl.rmsprop(learning_rate=0.01)),
    ]

    for name, optimizer in optimizers:
        print(f"\nTraining with {name}:")
        trainer = tl.Trainer(graph, tl.mse_loss(), optimizer)
        history = trainer.fit(inputs, targets, epochs=5, verbose=0)
        print(f"  Final loss: {history.train_losses[-1]:.6f}")


def example_4_validation_data():
    """Example 4: Training with validation data."""
    print_section("Example 4: Training with Validation Data")

    expr = tl.pred("x", [tl.var("i"), tl.var("j")])
    graph = tl.compile(expr)

    np.random.seed(42)

    # Training data
    train_inputs = {"x": np.random.rand(10, 10)}
    train_targets = np.random.rand(10, 10)

    # Validation data
    val_inputs = {"x": np.random.rand(10, 10)}
    val_targets = np.random.rand(10, 10)

    # Create trainer
    trainer = tl.Trainer(graph, tl.mse_loss(), tl.adam())

    # Train with validation
    print("Training with validation data...")
    history = trainer.fit(
        train_inputs,
        train_targets,
        epochs=10,
        validation_data=(val_inputs, val_targets),
        verbose=1
    )

    print(f"\nTraining complete!")
    print(f"Final training loss: {history.train_losses[-1]:.6f}")
    print(f"Final validation loss: {history.val_losses[-1]:.6f}")


def example_5_callbacks():
    """Example 5: Training with callbacks."""
    print_section("Example 5: Training with Callbacks")

    expr = tl.pred("x", [tl.var("i")])
    graph = tl.compile(expr)

    np.random.seed(42)
    train_inputs = {"x": np.random.rand(8, 8)}
    train_targets = np.random.rand(8, 8)
    val_inputs = {"x": np.random.rand(8, 8)}
    val_targets = np.random.rand(8, 8)

    # Create callbacks
    callbacks = [
        tl.early_stopping(patience=5, min_delta=0.0001),
        tl.model_checkpoint(save_best_only=1),
        tl.logger(verbose=1),
    ]

    # Create trainer with callbacks
    trainer = tl.Trainer(
        graph,
        tl.mse_loss(),
        tl.adam(),
        callbacks=callbacks
    )

    print("Training with callbacks (EarlyStopping, ModelCheckpoint, Logger)...")
    history = trainer.fit(
        train_inputs,
        train_targets,
        epochs=20,
        validation_data=(val_inputs, val_targets),
        verbose=1
    )

    print(f"\nTraining stopped after {history.num_epochs()} epochs")


def example_6_convenience_fit():
    """Example 6: Using convenience fit() function."""
    print_section("Example 6: Convenience fit() Function")

    # Create expression
    expr = tl.and_(
        tl.pred("Person", [tl.var("x")]),
        tl.pred("knows", [tl.var("x"), tl.var("y")])
    )

    # Create training data
    np.random.seed(42)
    train_inputs = {
        "Person": np.random.rand(10),
        "knows": np.random.rand(10, 10),
    }
    train_targets = np.random.rand(10, 10)

    # Use convenience function
    print("Training with convenience fit() function...")
    graph, history = tl.fit(
        expr,
        train_inputs,
        train_targets,
        loss_fn=tl.mse_loss(),
        optimizer=tl.adam(learning_rate=0.01),
        epochs=10
    )

    print(f"\nTraining complete!")
    print(f"Final loss: {history.train_losses[-1]:.6f}")


def example_7_training_history():
    """Example 7: Working with training history."""
    print_section("Example 7: Working with Training History")

    expr = tl.pred("x", [tl.var("i")])
    graph = tl.compile(expr)

    np.random.seed(42)
    inputs = {"x": np.random.rand(5, 5)}
    targets = np.random.rand(5, 5)

    # Train model
    trainer = tl.Trainer(graph, tl.mse_loss(), tl.adam())
    history = trainer.fit(inputs, targets, epochs=10, verbose=0)

    # Analyze training history
    print("Training History Analysis:")
    print(f"  Total epochs: {history.num_epochs()}")
    print(f"  Training losses: {[f'{loss:.6f}' for loss in history.train_losses]}")

    best_epoch, best_loss = history.best_train_loss()
    print(f"  Best epoch: {best_epoch + 1} with loss {best_loss:.6f}")

    # Add custom metrics
    print("\nAdding custom metrics to history...")
    history.add_metric("accuracy", 0.85)
    history.add_metric("accuracy", 0.90)
    history.add_metric("accuracy", 0.92)

    accuracy_history = history.get_metric("accuracy")
    print(f"  Accuracy history: {accuracy_history}")


def example_8_evaluate_and_predict():
    """Example 8: Evaluate and predict with trained model."""
    print_section("Example 8: Evaluate and Predict")

    expr = tl.pred("x", [tl.var("i"), tl.var("j")])
    graph = tl.compile(expr)

    np.random.seed(42)
    train_inputs = {"x": np.random.rand(5, 5)}
    train_targets = np.random.rand(5, 5)

    # Train model
    trainer = tl.Trainer(graph, tl.mse_loss(), tl.adam())
    print("Training model...")
    history = trainer.fit(train_inputs, train_targets, epochs=5, verbose=0)
    print(f"Training loss: {history.train_losses[-1]:.6f}")

    # Evaluate on test data
    test_inputs = {"x": np.random.rand(5, 5)}
    test_targets = np.random.rand(5, 5)
    test_loss = trainer.evaluate(test_inputs, test_targets)
    print(f"Test loss: {test_loss:.6f}")

    # Make predictions
    new_inputs = {"x": np.random.rand(5, 5)}
    predictions = trainer.predict(new_inputs)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")


def example_9_multi_output_training():
    """Example 9: Training with multi-output expressions."""
    print_section("Example 9: Multi-Output Training")

    # Create expression with multiple operations
    x_var = tl.var("x")
    y_var = tl.var("y")
    expr = tl.and_(
        tl.pred("knows", [x_var, y_var]),
        tl.pred("likes", [x_var, y_var])
    )

    graph = tl.compile(expr)

    np.random.seed(42)
    train_inputs = {
        "knows": np.random.rand(8, 8),
        "likes": np.random.rand(8, 8),
    }
    train_targets = np.random.rand(8, 8)

    # Train
    trainer = tl.Trainer(graph, tl.mse_loss(), tl.adam())
    print("Training multi-output model...")
    history = trainer.fit(train_inputs, train_targets, epochs=5, verbose=1)

    print(f"\nFinal loss: {history.train_losses[-1]:.6f}")


def example_10_learning_rate_schedule():
    """Example 10: Manually adjusting learning rate during training."""
    print_section("Example 10: Learning Rate Schedule")

    expr = tl.pred("x", [tl.var("i")])
    graph = tl.compile(expr)

    np.random.seed(42)
    inputs = {"x": np.random.rand(5, 5)}
    targets = np.random.rand(5, 5)

    # Create optimizer
    optimizer = tl.adam(learning_rate=0.01)
    trainer = tl.Trainer(graph, tl.mse_loss(), optimizer)

    print("Training with decreasing learning rate:")
    learning_rates = [0.01, 0.005, 0.001]

    all_losses = []
    for lr in learning_rates:
        # Update learning rate
        trainer.optimizer.learning_rate = lr
        print(f"\nLearning rate: {lr}")

        # Train for a few epochs
        history = trainer.fit(inputs, targets, epochs=3, verbose=0)
        all_losses.extend(history.train_losses)
        print(f"  Final loss: {history.train_losses[-1]:.6f}")

    print(f"\nAll losses: {[f'{loss:.6f}' for loss in all_losses]}")


def main():
    """Run all training examples."""
    print("\n" + "=" * 70)
    print("  TensorLogic-py Training API Examples")
    print("=" * 70)

    examples = [
        example_1_basic_training,
        example_2_different_loss_functions,
        example_3_different_optimizers,
        example_4_validation_data,
        example_5_callbacks,
        example_6_convenience_fit,
        example_7_training_history,
        example_8_evaluate_and_predict,
        example_9_multi_output_training,
        example_10_learning_rate_schedule,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nExample {i} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print_section("All Examples Complete!")
    print("The training API provides a high-level interface for training")
    print("neural-symbolic models with various loss functions, optimizers,")
    print("and callbacks. See the documentation for more details.")


if __name__ == "__main__":
    main()
