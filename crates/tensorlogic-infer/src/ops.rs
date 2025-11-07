//! Operation type enumerations.

#[derive(Clone, Copy, Debug)]
pub enum ElemOp {
    // Activation functions
    Relu,
    Sigmoid,

    // Unary operations
    OneMinus,

    // Arithmetic binary operations
    Add,
    Subtract,
    Multiply,
    Divide,
    Min, // Element-wise minimum
    Max, // Element-wise maximum

    // Comparison operations (return 0.0 or 1.0)
    Eq,  // Equal
    Lt,  // Less than
    Gt,  // Greater than
    Lte, // Less than or equal
    Gte, // Greater than or equal

    // Extended logical operations
    OrMax,     // OR using max(a, b)
    OrProbSum, // OR using probabilistic sum: 1 - (1-a)(1-b) = a + b - ab
    Nand,      // NAND: 1 - (a * b)
    Nor,       // NOR: 1 - max(a, b)
    Xor,       // XOR: |a - b| or (a + b) - 2*a*b for soft version
}

#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
    Min,
    Product, // Product reduction for FORALL quantifier
}
