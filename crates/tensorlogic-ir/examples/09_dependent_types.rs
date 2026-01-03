//! Dependent Types Example
//!
//! Demonstrates the dependent type system where types can depend on runtime values.
//! This is crucial for tensor operations where dimensions are first-class values.

use std::collections::HashMap;
use tensorlogic_ir::dependent::{DependentType, DependentTypeContext, DimConstraint, IndexExpr};

fn main() {
    println!("=== Dependent Types in TensorLogic ===\n");

    // Example 1: Vector with dependent length
    example_dependent_vector();

    // Example 2: Matrix with dependent dimensions
    example_dependent_matrix();

    // Example 3: Dependent function types
    example_dependent_function();

    // Example 4: Dimension constraints
    example_dimension_constraints();

    // Example 5: Index expression arithmetic
    example_index_arithmetic();

    // Example 6: Type context and validation
    example_type_context();

    // Example 7: Refinement with dependent types
    example_refinement_with_dependent();

    // Example 8: Well-formedness checking
    example_well_formedness();
}

fn example_dependent_vector() {
    println!("--- Example 1: Dependent Vector Types ---");

    // Vector of length n: Vec<n, Int>
    let n = IndexExpr::var("n");
    let vec_n_int = DependentType::vector(n.clone(), "Int");

    println!("Type: {}", vec_n_int);
    println!("Free index variables: {:?}", vec_n_int.free_index_vars());

    // Vector with constant length: Vec<10, Float>
    let vec_10_float = DependentType::vector(IndexExpr::constant(10), "Float");
    println!("Fixed-size type: {}", vec_10_float);
    println!(
        "Is well-formed (no free vars): {}\n",
        vec_10_float.is_well_formed()
    );
}

fn example_dependent_matrix() {
    println!("--- Example 2: Dependent Matrix Types ---");

    let m = IndexExpr::var("m");
    let n = IndexExpr::var("n");

    // Matrix<m, n, Float>
    let matrix_type = DependentType::matrix(m.clone(), n.clone(), "Float");
    println!("Matrix type: {}", matrix_type);

    // Square matrix: Matrix<n, n, Float>
    let square_matrix = DependentType::matrix(n.clone(), n.clone(), "Float");
    println!("Square matrix: {}", square_matrix);

    // Fixed-size matrix: Matrix<3, 4, Int>
    let fixed_matrix = DependentType::matrix(IndexExpr::constant(3), IndexExpr::constant(4), "Int");
    println!("Fixed matrix: {}\n", fixed_matrix);
}

fn example_dependent_function() {
    println!("--- Example 3: Dependent Function Types ---");

    use tensorlogic_ir::ParametricType;

    // (n: Int) -> Vec<n, Bool>
    // Function that takes an integer n and returns a vector of length n
    let n_param = DependentType::base(ParametricType::concrete("Int"));
    let n_var = IndexExpr::var("n");
    let return_type = DependentType::vector(n_var, "Bool");

    let func_type = DependentType::dependent_function("n", n_param, return_type);
    println!("Function type: {}", func_type);
    println!("Is well-formed: {}", func_type.is_well_formed());

    // (m: Int, n: Int) -> Matrix<m, n, Float>
    // This would require nested function types
    let m_param = DependentType::base(ParametricType::concrete("Int"));
    let m_var = IndexExpr::var("m");
    let n_var2 = IndexExpr::var("n");

    let inner_return = DependentType::matrix(m_var, n_var2, "Float");
    let inner_func = DependentType::dependent_function("n", m_param.clone(), inner_return);
    let outer_func = DependentType::dependent_function("m", m_param, inner_func);

    println!("Nested function type: {}\n", outer_func);
}

fn example_dimension_constraints() {
    println!("--- Example 4: Dimension Constraints ---");

    let n = IndexExpr::var("n");

    // Bounded vector: Vec<n, Int> where n <= 100
    let constraint = DimConstraint::lte(n.clone(), IndexExpr::constant(100));
    let vec_type = DependentType::vector(n.clone(), "Int");
    let constrained_vec = vec_type.with_constraints(vec![constraint]);

    println!("Constrained type: {}", constrained_vec);

    // Multiple constraints: Vec<n, Float> where n >= 10 && n <= 100
    let lower_bound = DimConstraint::gte(n.clone(), IndexExpr::constant(10));
    let upper_bound = DimConstraint::lte(n.clone(), IndexExpr::constant(100));
    let combined = DimConstraint::and(lower_bound, upper_bound);

    let vec_type2 = DependentType::vector(n.clone(), "Float");
    let doubly_constrained = vec_type2.with_constraints(vec![combined]);

    println!("Doubly constrained: {}", doubly_constrained);

    // Relationship between dimensions: Matrix<m, n, T> where m == n
    let m = IndexExpr::var("m");
    let n_expr = IndexExpr::var("n");
    let eq_constraint = DimConstraint::eq(m.clone(), n_expr.clone());

    let matrix = DependentType::matrix(m, n_expr, "Float");
    let square_only = matrix.with_constraints(vec![eq_constraint]);

    println!("Square matrix constraint: {}\n", square_only);
}

fn example_index_arithmetic() {
    println!("--- Example 5: Index Expression Arithmetic ---");

    let n = IndexExpr::var("n");
    let m = IndexExpr::var("m");

    // n + m
    let sum = IndexExpr::add(n.clone(), m.clone());
    println!("Sum: {}", sum);

    // n * 2
    let doubled = IndexExpr::mul(n.clone(), IndexExpr::constant(2));
    println!("Doubled: {}", doubled);

    // (n + m) / 2 - average
    let avg = IndexExpr::div(IndexExpr::add(n.clone(), m.clone()), IndexExpr::constant(2));
    println!("Average: {}", avg);

    // Simplification
    let complex_expr = IndexExpr::add(
        IndexExpr::mul(n.clone(), IndexExpr::constant(1)),
        IndexExpr::constant(0),
    );
    println!("Before simplification: {}", complex_expr);
    println!("After simplification: {}", complex_expr.simplify());

    // Evaluation
    let concrete = IndexExpr::add(IndexExpr::constant(5), IndexExpr::constant(3));
    println!(
        "Concrete expression: {} = {:?}",
        concrete,
        concrete.try_eval()
    );

    // Min/Max operations
    let min_expr = IndexExpr::min(n.clone(), IndexExpr::constant(100));
    let max_expr = IndexExpr::max(IndexExpr::constant(1), n);

    println!("Min: {}", min_expr);
    println!("Max: {}\n", max_expr);
}

fn example_type_context() {
    println!("--- Example 6: Type Context and Validation ---");

    let mut ctx = DependentTypeContext::new();

    // Bind index variables
    ctx.bind_index("n", 50);
    ctx.bind_index("m", 30);

    // Add constraints
    let n = IndexExpr::var("n");
    let constraint = DimConstraint::lte(n, IndexExpr::constant(100));
    ctx.add_constraint(constraint);

    println!("Context is satisfiable: {}", ctx.is_satisfiable());

    // Add contradictory constraint
    let n2 = IndexExpr::var("n");
    let bad_constraint = DimConstraint::gt(n2, IndexExpr::constant(100));
    ctx.add_constraint(bad_constraint);

    println!(
        "After adding contradictory constraint: {}\n",
        ctx.is_satisfiable()
    );
}

fn example_refinement_with_dependent() {
    println!("--- Example 7: Refinement with Dependent Types ---");

    use tensorlogic_ir::Term;

    // {v: Vec<n, Int> | n > 0}  - non-empty vector
    let n = IndexExpr::var("n");
    let vec_type = DependentType::vector(n, "Int");
    let predicate = Term::var("n"); // Simplified - would be actual predicate

    let refined = DependentType::refinement("v", vec_type, predicate);

    println!("Refined dependent type: {}\n", refined);
}

fn example_well_formedness() {
    println!("--- Example 8: Well-Formedness Checking ---");

    // Well-formed: (n: Int) -> Vec<n, Float>
    use tensorlogic_ir::ParametricType;

    let n_param = DependentType::base(ParametricType::concrete("Int"));
    let n_var = IndexExpr::var("n");
    let return_type = DependentType::vector(n_var, "Float");
    let func_type = DependentType::dependent_function("n", n_param, return_type);

    println!("Function type: {}", func_type);
    println!("Is well-formed: {}", func_type.is_well_formed());

    // NOT well-formed: Vec<n, Int> with free variable n
    let free_n = IndexExpr::var("n");
    let bad_vec = DependentType::vector(free_n, "Int");

    println!("\nVector with free variable: {}", bad_vec);
    println!("Is well-formed: {}", bad_vec.is_well_formed());
    println!("Free variables: {:?}", bad_vec.free_index_vars());

    // Complex example: Tensor<[m, n, k], Float>
    let m = IndexExpr::var("m");
    let n = IndexExpr::var("n");
    let k = IndexExpr::var("k");

    let tensor_type = DependentType::tensor(vec![m, n, k], "Float");
    println!("\nTensor type: {}", tensor_type);
    println!("Free variables: {:?}", tensor_type.free_index_vars());
    println!("Is well-formed: {}", tensor_type.is_well_formed());

    // Substitution example
    let m_var = IndexExpr::var("m");
    let n_var = IndexExpr::var("n");
    let expr = IndexExpr::add(m_var.clone(), n_var);

    let mut subst = HashMap::new();
    subst.insert("m".to_string(), IndexExpr::constant(10));

    let result = expr.substitute(&subst);
    println!("\nOriginal: {}", expr);
    println!("After substituting m=10: {}", result);
    println!("Simplified: {}", result.simplify());
}
