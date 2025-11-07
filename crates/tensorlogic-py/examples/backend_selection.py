#!/usr/bin/env python3
"""
Backend Selection Example

Demonstrates how to:
1. Query available backends
2. Get backend capabilities
3. Select a specific backend for execution
4. Get system information

This example shows the backend selection API that enables users to choose
between different execution engines (CPU, GPU, etc.) based on their needs.
"""

import numpy as np

try:
    import pytensorlogic as tl
except ImportError:
    print("Error: pytensorlogic not installed")
    print("Run 'maturin develop' in crates/pytensorlogic first")
    exit(1)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_list_backends():
    """Demonstrate listing available backends"""
    print_section("1. List Available Backends")

    backends = tl.list_available_backends()
    print("Available backends on this system:")
    for name, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {name:20} {status}")


def demo_backend_capabilities():
    """Demonstrate querying backend capabilities"""
    print_section("2. Backend Capabilities")

    # Get CPU backend capabilities
    print("Querying CPU backend capabilities...")
    caps = tl.get_backend_capabilities(tl.Backend.SCIRS2_CPU)

    print(f"\nBackend: {caps.name} v{caps.version}")
    print(f"\nSupported devices: {', '.join(caps.devices)}")
    print(f"Supported data types: {', '.join(caps.dtypes)}")
    print(f"Supported features: {', '.join(caps.features)}")
    print(f"Max tensor dimensions: {caps.max_dims}")

    # Query specific capabilities
    print("\nCapability queries:")
    print(f"  Supports CPU? {caps.supports_device('CPU')}")
    print(f"  Supports GPU? {caps.supports_device('GPU')}")
    print(f"  Supports f64? {caps.supports_dtype('f64')}")
    print(f"  Supports Autodiff? {caps.supports_feature('Autodiff')}")
    print(f"  Supports BatchExecution? {caps.supports_feature('BatchExecution')}")

    # Get full summary
    print("\nFull capability summary:")
    print(caps.summary())


def demo_default_backend():
    """Demonstrate getting the default backend"""
    print_section("3. Default Backend")

    default = tl.get_default_backend()
    print(f"Default backend: {default}")


def demo_system_info():
    """Demonstrate getting system information"""
    print_section("4. System Information")

    info = tl.get_system_info()
    print(f"TensorLogic version: {info['tensorlogic_version']}")
    print(f"Rust version: {info['rust_version']}")
    print(f"Default backend: {info['default_backend']}")
    print(f"Backend version: {info['backend_version']}")

    print("\nAvailable backends:")
    for name, available in info['available_backends'].items():
        status = "Yes" if available else "No"
        print(f"  {name:15} {status}")

    print("\nCPU backend capabilities:")
    cpu_caps = info['cpu_capabilities']
    print(f"  Devices: {', '.join(cpu_caps['devices'])}")
    print(f"  Data types: {', '.join(cpu_caps['dtypes'])}")
    print(f"  Features: {', '.join(cpu_caps['features'])}")
    print(f"  Max dims: {cpu_caps['max_dims']}")


def demo_execution_with_backend():
    """Demonstrate execution with backend selection"""
    print_section("5. Execution with Backend Selection")

    # Create a simple logical expression: AND(P(x), Q(x))
    print("Creating expression: AND(P(x), Q(x))")
    x = tl.var("x")
    p = tl.pred("P", [x])
    q = tl.pred("Q", [x])
    expr = tl.and_expr(p, q)

    # Compile
    print("Compiling...")
    graph = tl.compile(expr)

    # Prepare input data
    p_data = np.array([0.8, 0.9, 0.7])
    q_data = np.array([0.6, 0.8, 0.9])
    inputs = {"P": p_data, "Q": q_data}

    print(f"\nInput P: {p_data}")
    print(f"Input Q: {q_data}")

    # Execute with default backend (Auto)
    print("\n--- Execution with default backend (Auto) ---")
    result_auto = tl.execute(graph, inputs)
    print(f"Result: {result_auto['output']}")

    # Execute with explicit CPU backend
    print("\n--- Execution with explicit CPU backend ---")
    result_cpu = tl.execute(graph, inputs, backend=tl.Backend.SCIRS2_CPU)
    print(f"Result: {result_cpu['output']}")

    # Verify results are identical
    np.testing.assert_array_almost_equal(
        result_auto['output'],
        result_cpu['output']
    )
    print("\n✓ Results match across backends")


def demo_backend_error_handling():
    """Demonstrate error handling for unavailable backends"""
    print_section("6. Error Handling")

    print("Attempting to use GPU backend (not yet implemented)...")

    x = tl.var("x")
    p = tl.pred("P", [x])
    graph = tl.compile(p)
    p_data = np.array([0.8, 0.9, 0.7])

    try:
        result = tl.execute(graph, {"P": p_data}, backend=tl.Backend.SCIRS2_GPU)
        print("GPU execution succeeded (unexpected)")
    except RuntimeError as e:
        print(f"✓ Expected error caught: {e}")


def demo_backend_selection_workflow():
    """Demonstrate a complete workflow with backend selection"""
    print_section("7. Complete Workflow")

    print("Step 1: Check available backends")
    backends = tl.list_available_backends()
    available_cpu = backends['SciRS2CPU']
    print(f"  CPU backend available: {available_cpu}")

    if not available_cpu:
        print("  Error: CPU backend not available!")
        return

    print("\nStep 2: Get and validate backend capabilities")
    caps = tl.get_backend_capabilities(tl.Backend.SCIRS2_CPU)
    print(f"  Backend: {caps.name}")
    print(f"  Supports Autodiff: {caps.supports_feature('Autodiff')}")

    print("\nStep 3: Create and compile expression")
    x = tl.var("x")
    p = tl.pred("P", [x])
    q = tl.pred("Q", [x])
    expr = tl.or_expr(p, q)  # OR(P(x), Q(x))
    graph = tl.compile(expr)
    print("  Expression compiled: OR(P(x), Q(x))")

    print("\nStep 4: Execute with selected backend")
    p_data = np.array([0.3, 0.8, 0.2])
    q_data = np.array([0.7, 0.4, 0.9])
    result = tl.execute(
        graph,
        {"P": p_data, "Q": q_data},
        backend=tl.Backend.SCIRS2_CPU
    )
    print(f"  Input P: {p_data}")
    print(f"  Input Q: {q_data}")
    print(f"  Result:  {result['output']}")

    print("\n✓ Workflow completed successfully")


def main():
    """Run all demonstrations"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║          TensorLogic Backend Selection Example            ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)

    demo_list_backends()
    demo_backend_capabilities()
    demo_default_backend()
    demo_system_info()
    demo_execution_with_backend()
    demo_backend_error_handling()
    demo_backend_selection_workflow()

    print("\n" + "="*60)
    print("  All demonstrations completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
