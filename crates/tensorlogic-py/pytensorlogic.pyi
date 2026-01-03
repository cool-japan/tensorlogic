"""Type stubs for tensorlogic_py module.

This file provides type hints for IDE support and static type checking.
"""

from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt

# Version
__version__: str

# Core Types
class Term:
    """A term in a logical expression (variable or constant)."""
    def name(self) -> str: ...
    def is_var(self) -> bool: ...
    def is_const(self) -> bool: ...

class TLExpr:
    """A logical expression."""
    ...

class EinsumGraph:
    """A compiled tensor computation graph."""
    def stats(self) -> Dict[str, int]: ...

# Adapter Types
class DomainInfo: ...
class PredicateInfo: ...
class SymbolTable: ...
class CompilerContext: ...
class CompilationConfig: ...

# Provenance Types
class SourceLocation:
    """Source code location information."""
    def __init__(self, file: str, line: int, column: int) -> None: ...
    @property
    def file(self) -> str: ...
    @property
    def line(self) -> int: ...
    @property
    def column(self) -> int: ...

class SourceSpan:
    """Source code span from start to end location."""
    def __init__(self, start: SourceLocation, end: SourceLocation) -> None: ...
    @property
    def start(self) -> SourceLocation: ...
    @property
    def end(self) -> SourceLocation: ...

class Provenance:
    """Provenance information tracking origin of IR nodes."""
    def __init__(self) -> None: ...
    @property
    def rule_id(self) -> Optional[str]: ...
    @property
    def source_file(self) -> Optional[str]: ...
    @property
    def span(self) -> Optional[SourceSpan]: ...
    def set_rule_id(self, rule_id: str) -> None: ...
    def set_source_file(self, source_file: str) -> None: ...
    def set_span(self, span: SourceSpan) -> None: ...
    def add_attribute(self, key: str, value: str) -> None: ...
    def get_attribute(self, key: str) -> Optional[str]: ...
    def get_attributes(self) -> Dict[str, str]: ...

class ProvenanceTracker:
    """Provenance tracker for RDF and tensor computation mappings."""
    def __init__(self, enable_rdfstar: bool = False) -> None: ...
    def track_entity(self, entity_iri: str, tensor_idx: int) -> None: ...
    def track_shape(self, shape_iri: str, rule_expr: str, node_idx: int) -> None: ...
    def get_entity(self, tensor_idx: int) -> Optional[str]: ...
    def get_tensor(self, entity_iri: str) -> Optional[int]: ...
    def track_inferred_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        rule_id: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> None: ...
    def get_entity_mappings(self) -> Dict[str, int]: ...
    def get_shape_mappings(self) -> Dict[str, str]: ...
    def to_rdf_star(self) -> List[str]: ...
    def to_rdfstar_turtle(self) -> str: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> "ProvenanceTracker": ...
    def get_high_confidence_inferences(self, min_confidence: float = 0.8) -> List[Dict[str, any]]: ...

# Backend Types
class Backend:
    """Backend selection for execution.

    Available backends:
    - AUTO: Automatically select the best available backend
    - SCIRS2_CPU: SciRS2 backend with CPU execution
    - SCIRS2_GPU: SciRS2 backend with GPU execution (requires 'gpu' feature)
    """
    AUTO: "Backend"
    SCIRS2_CPU: "Backend"
    SCIRS2_GPU: "Backend"

class BackendCapabilities:
    """Backend capability information.

    Provides details about a backend's supported features, devices, and data types.
    """
    @property
    def name(self) -> str:
        """Get the backend name."""
        ...

    @property
    def version(self) -> str:
        """Get the backend version."""
        ...

    @property
    def devices(self) -> List[str]:
        """Get list of supported device types."""
        ...

    @property
    def dtypes(self) -> List[str]:
        """Get list of supported data types."""
        ...

    @property
    def features(self) -> List[str]:
        """Get list of supported features."""
        ...

    @property
    def max_dims(self) -> int:
        """Get maximum number of tensor dimensions supported."""
        ...

    def supports_device(self, device: str) -> bool:
        """Check if a specific device type is supported."""
        ...

    def supports_dtype(self, dtype: str) -> bool:
        """Check if a specific data type is supported."""
        ...

    def supports_feature(self, feature: str) -> bool:
        """Check if a specific feature is supported."""
        ...

    def summary(self) -> str:
        """Get a human-readable summary of capabilities."""
        ...

    def to_dict(self) -> Dict[str, any]:
        """Get capabilities as a dictionary."""
        ...

# Term Constructors
def var(name: str) -> Term: ...
def const(name: str) -> Term: ...

# Expression Constructors
def pred(name: str, args: List[Term]) -> TLExpr: ...
def and_expr(left: TLExpr, right: TLExpr) -> TLExpr: ...
def or_expr(left: TLExpr, right: TLExpr) -> TLExpr: ...
def not_expr(expr: TLExpr) -> TLExpr: ...
def exists(var: str, domain: str, body: TLExpr) -> TLExpr: ...
def forall(var: str, domain: str, body: TLExpr) -> TLExpr: ...
def imply(left: TLExpr, right: TLExpr) -> TLExpr: ...
def constant(value: float) -> TLExpr: ...

# Arithmetic
def add(left: TLExpr, right: TLExpr) -> TLExpr: ...
def sub(left: TLExpr, right: TLExpr) -> TLExpr: ...
def mul(left: TLExpr, right: TLExpr) -> TLExpr: ...
def div(left: TLExpr, right: TLExpr) -> TLExpr: ...

# Comparison
def eq(left: TLExpr, right: TLExpr) -> TLExpr: ...
def lt(left: TLExpr, right: TLExpr) -> TLExpr: ...
def gt(left: TLExpr, right: TLExpr) -> TLExpr: ...
def lte(left: TLExpr, right: TLExpr) -> TLExpr: ...
def gte(left: TLExpr, right: TLExpr) -> TLExpr: ...

# Conditional
def if_then_else(cond: TLExpr, then_expr: TLExpr, else_expr: TLExpr) -> TLExpr: ...

# Compilation
def compile(expr: TLExpr) -> EinsumGraph: ...
def compile_with_config(expr: TLExpr, config: CompilationConfig) -> EinsumGraph: ...

# Execution
def execute(
    graph: EinsumGraph,
    inputs: Dict[str, npt.NDArray[np.float64]],
    backend: Backend | None = None
) -> Dict[str, npt.NDArray[np.float64]]:
    """Execute a compiled graph with given inputs.

    Args:
        graph: The compiled EinsumGraph to execute
        inputs: Dictionary mapping input names to NumPy arrays
        backend: Optional backend selection (defaults to Auto, which selects SciRS2CPU)

    Returns:
        Dictionary mapping output names to NumPy arrays

    Raises:
        RuntimeError: If execution fails or backend is not available
    """
    ...

# Adapter Constructors
def domain_info(name: str, cardinality: int) -> DomainInfo: ...
def predicate_info(name: str, domains: List[str]) -> PredicateInfo: ...
def symbol_table() -> SymbolTable: ...
def compiler_context() -> CompilerContext: ...

# Backend Functions
def get_backend_capabilities(backend: Backend | None = None) -> BackendCapabilities:
    """Get capabilities for a specific backend.

    Args:
        backend: The backend to query (defaults to Auto)

    Returns:
        BackendCapabilities: Detailed capability information

    Raises:
        RuntimeError: If the backend is not available
    """
    ...

def list_available_backends() -> Dict[str, bool]:
    """List all available backends.

    Returns:
        Dictionary mapping backend names to their availability status
    """
    ...

def get_default_backend() -> Backend:
    """Get the default backend for this system.

    Returns:
        Backend: The default backend (currently SciRS2CPU)
    """
    ...

def get_system_info() -> Dict[str, any]:
    """Get detailed system and backend information.

    Returns:
        Dictionary with system and backend information including:
        - tensorlogic_version: TensorLogic version
        - rust_version: Rust compiler version
        - default_backend: Default backend name
        - backend_version: Backend version
        - available_backends: Dict of available backends
        - cpu_capabilities: CPU backend capabilities
    """
    ...

# Provenance Functions
def get_provenance(graph: EinsumGraph) -> List[Optional[Provenance]]:
    """Get provenance information from an einsum graph.

    Extracts provenance metadata from all nodes in the graph.

    Args:
        graph: EinsumGraph to extract provenance from

    Returns:
        List of provenance records for each node (None if node has no provenance)
    """
    ...

def get_metadata(graph: EinsumGraph) -> List[Optional[Dict[str, any]]]:
    """Get metadata from an einsum graph.

    Extracts all metadata (names, spans, provenance, attributes) from graph nodes.

    Args:
        graph: EinsumGraph to extract metadata from

    Returns:
        List of metadata dictionaries for each node (None if node has no metadata)
    """
    ...

def provenance_tracker(enable_rdfstar: bool = False) -> ProvenanceTracker:
    """Create a provenance tracker.

    Helper function to create a new provenance tracker.

    Args:
        enable_rdfstar: Enable RDF* support (default: False)

    Returns:
        ProvenanceTracker: New provenance tracker
    """
    ...

# Training API Types

class LossFunction:
    """Loss function for training neural-symbolic models."""
    def __init__(self, loss_type: str) -> None: ...
    @property
    def loss_type(self) -> str: ...
    def __call__(
        self,
        predictions: npt.NDArray[np.float64],
        targets: npt.NDArray[np.float64]
    ) -> float: ...

class Optimizer:
    """Optimizer for updating model parameters during training."""
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float = 0.01,
        config: Optional[Dict[str, float]] = None
    ) -> None: ...
    @property
    def optimizer_type(self) -> str: ...
    @property
    def learning_rate(self) -> float: ...
    @learning_rate.setter
    def learning_rate(self, lr: float) -> None: ...

class Callback:
    """Callback for monitoring and controlling training."""
    def __init__(
        self,
        callback_type: str,
        config: Optional[Dict[str, float]] = None
    ) -> None: ...
    @property
    def callback_type(self) -> str: ...

class TrainingHistory:
    """Training history containing loss and metrics over epochs."""
    def __init__(self) -> None: ...
    @property
    def train_losses(self) -> List[float]: ...
    @property
    def val_losses(self) -> List[float]: ...
    def add_train_loss(self, loss: float) -> None: ...
    def add_val_loss(self, loss: float) -> None: ...
    def add_metric(self, name: str, value: float) -> None: ...
    def get_metric(self, name: str) -> Optional[List[float]]: ...
    def num_epochs(self) -> int: ...
    def best_train_loss(self) -> Optional[tuple[int, float]]: ...
    def best_val_loss(self) -> Optional[tuple[int, float]]: ...

class Trainer:
    """High-level trainer for neural-symbolic models."""
    def __init__(
        self,
        graph: EinsumGraph,
        loss_fn: LossFunction,
        optimizer: Optimizer,
        output_name: str = "result",
        callbacks: Optional[List[Callback]] = None
    ) -> None: ...

    def fit(
        self,
        train_inputs: Dict[str, npt.NDArray[np.float64]],
        train_targets: npt.NDArray[np.float64],
        epochs: int = 10,
        validation_data: Optional[tuple[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None,
        verbose: int = 1
    ) -> TrainingHistory:
        """Train the model on data."""
        ...

    def evaluate(
        self,
        inputs: Dict[str, npt.NDArray[np.float64]],
        targets: npt.NDArray[np.float64]
    ) -> float:
        """Evaluate model on data without training."""
        ...

    def predict(
        self,
        inputs: Dict[str, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Make predictions on new data."""
        ...

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        ...

# Loss Functions
def mse_loss() -> LossFunction:
    """Create a Mean Squared Error (MSE) loss function."""
    ...

def bce_loss() -> LossFunction:
    """Create a Binary Cross-Entropy (BCE) loss function."""
    ...

def cross_entropy_loss() -> LossFunction:
    """Create a Cross-Entropy loss function."""
    ...

# Optimizers
def sgd(learning_rate: float = 0.01, momentum: float = 0.0) -> Optimizer:
    """Create a Stochastic Gradient Descent (SGD) optimizer."""
    ...

def adam(
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> Optimizer:
    """Create an Adam optimizer."""
    ...

def rmsprop(
    learning_rate: float = 0.01,
    alpha: float = 0.99,
    epsilon: float = 1e-8
) -> Optimizer:
    """Create an RMSprop optimizer."""
    ...

# Callbacks
def early_stopping(patience: float = 5.0, min_delta: float = 0.0001) -> Callback:
    """Create an EarlyStopping callback."""
    ...

def model_checkpoint(save_best_only: float = 1.0) -> Callback:
    """Create a ModelCheckpoint callback."""
    ...

def logger(verbose: float = 1.0) -> Callback:
    """Create a Logger callback."""
    ...

# Training Functions
def fit(
    expr: TLExpr,
    train_inputs: Dict[str, npt.NDArray[np.float64]],
    train_targets: npt.NDArray[np.float64],
    loss_fn: Optional[LossFunction] = None,
    optimizer: Optional[Optimizer] = None,
    epochs: int = 10,
    config: Optional[CompilationConfig] = None
) -> tuple[EinsumGraph, TrainingHistory]:
    """Train a model with a simple API.

    Convenience function for training without explicitly creating a Trainer.

    Args:
        expr: TensorLogic expression to train
        train_inputs: Training input data
        train_targets: Training target values
        loss_fn: Loss function (default: MSE)
        optimizer: Optimizer (default: Adam with lr=0.001)
        epochs: Number of training epochs (default: 10)
        config: Optional compilation configuration

    Returns:
        Tuple of (trained_graph, training_history)
    """
    ...

# ============================================================================
# Model Persistence
# ============================================================================

class ModelPackage:
    """Container for saving and loading complete TensorLogic models.

    Supports multiple serialization formats:
    - JSON: Human-readable, cross-platform
    - Binary: Compact, efficient (bincode)
    - Pickle: Python-native serialization

    Example:
        >>> package = tl.model_package()
        >>> package.add_metadata("author", "Alice")
        >>> package.save_json("model.json")
        >>> loaded = tl.ModelPackage.load_json("model.json")
    """

    def __init__(self) -> None:
        """Create a new empty model package."""
        ...

    @property
    def graph(self) -> Optional[str]:
        """Serialized einsum graph."""
        ...

    @graph.setter
    def graph(self, value: Optional[str]) -> None: ...

    @property
    def config(self) -> Optional[str]:
        """Serialized compilation configuration."""
        ...

    @config.setter
    def config(self, value: Optional[str]) -> None: ...

    @property
    def symbol_table(self) -> Optional[str]:
        """Serialized symbol table."""
        ...

    @symbol_table.setter
    def symbol_table(self, value: Optional[str]) -> None: ...

    @property
    def compiler_context(self) -> Optional[str]:
        """Serialized compiler context."""
        ...

    @compiler_context.setter
    def compiler_context(self, value: Optional[str]) -> None: ...

    @property
    def parameters(self) -> Optional[Dict[str, bytes]]:
        """Training parameters (tensor data)."""
        ...

    @parameters.setter
    def parameters(self, value: Optional[Dict[str, bytes]]) -> None: ...

    @property
    def metadata(self) -> Dict[str, str]:
        """Metadata key-value pairs."""
        ...

    @metadata.setter
    def metadata(self, value: Dict[str, str]) -> None: ...

    def add_metadata(self, key: str, value: str) -> None:
        """Add a metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        ...

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key.

        Args:
            key: Metadata key

        Returns:
            Metadata value or None if not found
        """
        ...

    def save_json(self, path: str) -> None:
        """Save package to JSON file.

        Args:
            path: File path to save to

        Example:
            >>> package.save_json("model.json")
        """
        ...

    @staticmethod
    def load_json(path: str) -> "ModelPackage":
        """Load package from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded ModelPackage

        Example:
            >>> package = ModelPackage.load_json("model.json")
        """
        ...

    def save_binary(self, path: str) -> None:
        """Save package to binary file (bincode format).

        Args:
            path: File path to save to

        Example:
            >>> package.save_binary("model.bin")
        """
        ...

    @staticmethod
    def load_binary(path: str) -> "ModelPackage":
        """Load package from binary file.

        Args:
            path: File path to load from

        Returns:
            Loaded ModelPackage

        Example:
            >>> package = ModelPackage.load_binary("model.bin")
        """
        ...

    def to_json(self) -> str:
        """Convert package to JSON string.

        Returns:
            JSON representation
        """
        ...

    @staticmethod
    def from_json(json: str) -> "ModelPackage":
        """Create package from JSON string.

        Args:
            json: JSON string

        Returns:
            ModelPackage instance
        """
        ...

    def to_bytes(self) -> bytes:
        """Convert package to binary bytes (for pickle support).

        Returns:
            Binary representation
        """
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "ModelPackage":
        """Create package from binary bytes.

        Args:
            data: Binary bytes

        Returns:
            ModelPackage instance
        """
        ...

# Persistence Functions

def model_package() -> ModelPackage:
    """Create a new model package.

    Returns:
        Empty ModelPackage instance

    Example:
        >>> package = tl.model_package()
        >>> package.add_metadata("author", "John Doe")
    """
    ...

def save_model(
    graph: EinsumGraph,
    path: str,
    format: str = "json"
) -> None:
    """Save a compiled graph to file.

    Args:
        graph: EinsumGraph to save
        path: File path to save to
        format: Format to use ("json" or "binary", default: "json")

    Example:
        >>> graph = tl.compile(expr)
        >>> tl.save_model(graph, "model.json")
        >>> tl.save_model(graph, "model.bin", format="binary")
    """
    ...

def load_model(
    path: str,
    format: Optional[str] = None
) -> EinsumGraph:
    """Load a compiled graph from file.

    Args:
        path: File path to load from
        format: Format to use (default: auto-detect from extension)

    Returns:
        Loaded EinsumGraph

    Example:
        >>> graph = tl.load_model("model.json")
        >>> graph = tl.load_model("model.bin", format="binary")
    """
    ...

def save_full_model(
    graph: EinsumGraph,
    path: str,
    config: Optional[CompilationConfig] = None,
    symbol_table: Optional[SymbolTable] = None,
    compiler_context: Optional[CompilerContext] = None,
    metadata: Optional[Dict[str, str]] = None,
    format: str = "json"
) -> None:
    """Save a complete model with all components.

    Args:
        graph: EinsumGraph to save
        path: File path to save to
        config: Optional compilation configuration
        symbol_table: Optional symbol table
        compiler_context: Optional compiler context
        metadata: Optional metadata dictionary
        format: Format to use ("json" or "binary", default: "json")

    Example:
        >>> tl.save_full_model(
        ...     graph,
        ...     "model.json",
        ...     config=config,
        ...     symbol_table=sym_table,
        ...     metadata={"description": "My model", "version": "1.0"}
        ... )
    """
    ...

def load_full_model(
    path: str,
    format: Optional[str] = None
) -> Dict[str, any]:
    """Load a complete model with all components.

    Args:
        path: File path to load from
        format: Format to use (default: auto-detect from extension)

    Returns:
        Dictionary with keys:
        - 'graph': EinsumGraph
        - 'config': CompilationConfig (if saved)
        - 'symbol_table': SymbolTable (if saved)
        - 'metadata': Dict[str, str]

    Example:
        >>> model = tl.load_full_model("model.json")
        >>> graph = model['graph']
        >>> config = model.get('config')
        >>> metadata = model['metadata']
    """
    ...

# ============================================================================
# Rule Builder DSL - Python-native syntax for defining logic rules
# ============================================================================

class Var:
    """Variable wrapper with domain binding for DSL.

    Enables Python-native syntax for building logic expressions with
    operator overloading: & (AND), | (OR), ~ (NOT), >> (IMPLY)

    Example:
        >>> x = tl.Var("x", domain="Person")
        >>> y = tl.Var("y", domain="Person")
        >>> knows = tl.PredicateBuilder("knows", arity=2)
        >>> expr = knows(x, y) & knows(y, x)  # Mutual knowledge
    """

    def __init__(self, name: str, domain: Optional[str] = None) -> None:
        """Create a variable.

        Args:
            name: Variable name
            domain: Optional domain name for type checking
        """
        ...

    @property
    def name(self) -> str:
        """Get variable name."""
        ...

    @property
    def domain(self) -> Optional[str]:
        """Get variable domain."""
        ...

    def to_term(self) -> Term:
        """Convert to PyTerm for internal use."""
        ...

    def to_expr(self) -> TLExpr:
        """Get the underlying TLExpr representation."""
        ...

class PredicateBuilder:
    """Predicate builder for function-call syntax.

    Enables defining predicates that can be called with variables
    to produce TLExpr instances, with automatic arity and domain validation.

    Example:
        >>> knows = tl.PredicateBuilder("knows", arity=2, domains=["Person", "Person"])
        >>> x = tl.Var("x", domain="Person")
        >>> y = tl.Var("y", domain="Person")
        >>> expr = knows(x, y)  # Creates a predicate expression
    """

    def __init__(
        self,
        name: str,
        arity: Optional[int] = None,
        domains: Optional[List[str]] = None
    ) -> None:
        """Create a predicate builder.

        Args:
            name: Predicate name
            arity: Number of arguments (for validation)
            domains: Domain names for each argument (for validation)
        """
        ...

    @property
    def name(self) -> str:
        """Get predicate name."""
        ...

    @property
    def arity(self) -> Optional[int]:
        """Get predicate arity."""
        ...

    @property
    def domains(self) -> Optional[List[str]]:
        """Get argument domains."""
        ...

    def __call__(self, *args) -> TLExpr:
        """Call predicate with arguments to create a TLExpr.

        Args:
            *args: Variables, constants, or Terms

        Returns:
            TLExpr representing the predicate application

        Raises:
            ValueError: If arity doesn't match
            TypeError: If domains don't match
        """
        ...

    def to_predicate_info(self) -> PredicateInfo:
        """Get predicate metadata as PredicateInfo."""
        ...

class RuleBuilder:
    """Rule builder context manager for collecting and compiling rules.

    Provides a high-level DSL for defining multiple rules and compiling
    them together into execution graphs. Manages symbol tables and
    domain/predicate metadata.

    Example:
        >>> with tl.RuleBuilder() as rb:
        ...     x, y, z = rb.vars("x", "y", "z", domain="Person")
        ...     knows = rb.pred("knows", arity=2)
        ...     rule1 = (knows(x, y) & knows(y, z)) >> knows(x, z)
        ...     rb.add_rule(rule1, name="transitivity")
        ...     graph = rb.compile()
    """

    def __init__(self, config: Optional[CompilationConfig] = None) -> None:
        """Create a rule builder.

        Args:
            config: Optional compilation configuration
        """
        ...

    def __enter__(self) -> "RuleBuilder":
        """Enter context manager."""
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit context manager."""
        ...

    def vars(self, *names: str, domain: Optional[str] = None):
        """Create variables with optional domain.

        Args:
            *names: Variable names
            domain: Optional domain name for all variables

        Returns:
            Single Var or tuple of Vars

        Example:
            >>> rb = tl.RuleBuilder()
            >>> x = rb.vars("x", domain="Person")
            >>> x, y, z = rb.vars("x", "y", "z", domain="Person")
        """
        ...

    def pred(
        self,
        name: str,
        arity: Optional[int] = None,
        domains: Optional[List[str]] = None
    ) -> PredicateBuilder:
        """Create a predicate builder.

        Args:
            name: Predicate name
            arity: Number of arguments (optional)
            domains: Domain names for each argument (optional)

        Returns:
            PredicateBuilder instance

        Example:
            >>> rb = tl.RuleBuilder()
            >>> knows = rb.pred("knows", arity=2, domains=["Person", "Person"])
        """
        ...

    def add_domain(
        self,
        name: str,
        cardinality: int,
        description: Optional[str] = None,
        elements: Optional[List[str]] = None
    ) -> None:
        """Add a domain to the symbol table.

        Args:
            name: Domain name
            cardinality: Number of elements
            description: Optional description
            elements: Optional list of element names

        Example:
            >>> rb = tl.RuleBuilder()
            >>> rb.add_domain("Person", cardinality=10, description="People")
        """
        ...

    def add_rule(self, expr: TLExpr, name: Optional[str] = None) -> None:
        """Add a rule to the builder.

        Args:
            expr: TLExpr representing the rule
            name: Optional name for the rule (default: rule_N)

        Example:
            >>> rb = tl.RuleBuilder()
            >>> x, y = rb.vars("x", "y")
            >>> knows = rb.pred("knows")
            >>> rule = knows(x, y) >> knows(y, x)
            >>> rb.add_rule(rule, name="symmetry")
        """
        ...

    def get_rules(self) -> List[tuple]:
        """Get all defined rules.

        Returns:
            List of (name, expr) tuples
        """
        ...

    def get_symbol_table(self) -> SymbolTable:
        """Get the symbol table.

        Returns:
            SymbolTable instance
        """
        ...

    def compile(self, config: Optional[CompilationConfig] = None) -> EinsumGraph:
        """Compile all rules into a single execution graph.

        Args:
            config: Optional compilation config (overrides builder config)

        Returns:
            EinsumGraph instance

        Raises:
            ValueError: If no rules defined

        Example:
            >>> rb = tl.RuleBuilder()
            >>> # ... define rules ...
            >>> graph = rb.compile()
        """
        ...

    def compile_separate(
        self,
        config: Optional[CompilationConfig] = None
    ) -> Dict[str, EinsumGraph]:
        """Compile each rule separately.

        Args:
            config: Optional compilation config

        Returns:
            Dictionary mapping rule names to EinsumGraph instances

        Example:
            >>> rb = tl.RuleBuilder()
            >>> # ... define rules ...
            >>> graphs = rb.compile_separate()
            >>> graphs['transitivity']  # Access specific rule's graph
        """
        ...

    def clear(self) -> None:
        """Clear all rules and symbol table."""
        ...

    def __len__(self) -> int:
        """Get number of rules."""
        ...

# DSL convenience functions
def var_dsl(name: str, domain: Optional[str] = None) -> Var:
    """Create a variable with optional domain.

    Alias for Var() constructor provided as a function.

    Args:
        name: Variable name
        domain: Optional domain name

    Returns:
        Var instance

    Example:
        >>> x = tl.var_dsl("x", domain="Person")
    """
    ...

def pred_dsl(
    name: str,
    arity: Optional[int] = None,
    domains: Optional[List[str]] = None
) -> PredicateBuilder:
    """Create a predicate builder.

    Alias for PredicateBuilder() constructor provided as a function.

    Args:
        name: Predicate name
        arity: Number of arguments (optional)
        domains: Domain names for each argument (optional)

    Returns:
        PredicateBuilder instance

    Example:
        >>> knows = tl.pred_dsl("knows", arity=2, domains=["Person", "Person"])
    """
    ...

def rule_builder(config: Optional[CompilationConfig] = None) -> RuleBuilder:
    """Create a rule builder.

    Alias for RuleBuilder() constructor provided as a function.

    Args:
        config: Optional compilation config

    Returns:
        RuleBuilder instance

    Example:
        >>> rb = tl.rule_builder()
        >>> x, y = rb.vars("x", "y", domain="Person")
    """
    ...

# ============================================================================
# Async Execution Support
# ============================================================================

class AsyncResult:
    """Future-like result for async execution.

    Represents a pending or completed async execution result.
    Use is_ready() to check if computation is complete, and result() to retrieve it.
    Supports cancellation through a CancellationToken.
    """

    def is_ready(self) -> bool:
        """Check if the async computation is ready."""
        ...

    def result(self) -> Dict[str, npt.NDArray[np.float64]]:
        """Wait for computation to complete and get the result."""
        ...

    def wait(self, timeout_secs: float) -> bool:
        """Wait for computation with timeout."""
        ...

    def cancel(self) -> None:
        """Cancel the async operation."""
        ...

    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        ...

    def get_cancellation_token(self) -> Optional["CancellationToken"]:
        """Get the cancellation token if available."""
        ...

class BatchExecutor:
    """Batch executor for processing multiple inputs through the same graph."""

    def __init__(self, graph: EinsumGraph, backend: Optional[Backend] = None) -> None: ...

    def execute_batch(
        self,
        inputs_list: List[Dict[str, npt.NDArray[np.float64]]],
        parallel: bool = True
    ) -> List[Dict[str, npt.NDArray[np.float64]]]: ...

def execute_async(
    graph: EinsumGraph,
    inputs: Dict[str, npt.NDArray[np.float64]],
    backend: Optional[Backend] = None
) -> AsyncResult:
    """Execute a graph asynchronously without blocking."""
    ...

def execute_parallel(
    graphs: List[EinsumGraph],
    inputs_list: List[Dict[str, npt.NDArray[np.float64]]],
    backend: Optional[Backend] = None
) -> List[AsyncResult]:
    """Execute multiple graphs in parallel."""
    ...

# ============================================================================
# Cancellation Support
# ============================================================================

class CancellationToken:
    """Cancellation token for async operations."""
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def reset(self) -> None: ...

def cancellation_token() -> CancellationToken:
    """Create a cancellation token."""
    ...

# ============================================================================
# Performance Monitoring
# ============================================================================

class MemorySnapshot:
    """Memory usage snapshot."""
    @property
    def current_bytes(self) -> int: ...
    @property
    def peak_bytes(self) -> int: ...
    @property
    def allocation_count(self) -> int: ...
    @property
    def current_mb(self) -> float: ...
    @property
    def peak_mb(self) -> float: ...

class Profiler:
    """Performance profiler for tracking execution metrics."""
    def __init__(self) -> None: ...
    @property
    def is_active(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def record_time(self, name: str, time_ms: float) -> None: ...
    def snapshot(self, label: str) -> None: ...
    def get_timing_stats(self, name: str) -> Dict[str, any]: ...
    def get_operation_names(self) -> List[str]: ...
    def get_memory_snapshots(self) -> List[tuple[str, MemorySnapshot]]: ...
    def elapsed_ms(self) -> float: ...
    def summary(self) -> str: ...

class Timer:
    """Timer context for measuring execution time."""
    def __init__(self, name: Optional[str] = None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> float: ...
    def elapsed(self) -> float: ...
    def __enter__(self) -> "Timer": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...

def memory_snapshot() -> MemorySnapshot:
    """Get current memory snapshot."""
    ...

def profiler() -> Profiler:
    """Create a new profiler."""
    ...

def timer(name: Optional[str] = None) -> Timer:
    """Create a new timer."""
    ...

def get_memory_info() -> Dict[str, any]:
    """Get system memory information."""
    ...

def reset_memory_tracking() -> None:
    """Reset all memory tracking statistics."""
    ...

# ============================================================================
# Streaming Execution
# ============================================================================

class StreamingExecutor:
    """Streaming executor for processing large datasets in chunks."""
    def __init__(
        self,
        graph: EinsumGraph,
        chunk_size: int = 1000,
        overlap: int = 0,
        backend: Optional[Backend] = None
    ) -> None: ...

    def execute_streaming(
        self,
        inputs: Dict[str, npt.NDArray[np.float64]],
        output_key: Optional[str] = None
    ) -> List[npt.NDArray[np.float64]]: ...

    @property
    def chunk_size(self) -> int: ...

    @chunk_size.setter
    def chunk_size(self, size: int) -> None: ...

class DataGenerator:
    """Data generator for memory-efficient data loading."""
    def __init__(
        self,
        data_sources: List[str],
        batch_size: int = 32,
        shuffle: bool = False
    ) -> None: ...

    def __len__(self) -> int: ...
    def reset(self) -> None: ...

class ResultAccumulator:
    """Accumulator for streaming results."""
    def __init__(self) -> None: ...
    def add(self, result: npt.NDArray[np.float64]) -> None: ...
    def count(self) -> int: ...
    def total_elements(self) -> int: ...
    def combine(self) -> npt.NDArray[np.float64]: ...
    def clear(self) -> None: ...
    def stats(self) -> Dict[str, any]: ...

def streaming_executor(
    graph: EinsumGraph,
    chunk_size: int = 1000,
    overlap: int = 0,
    backend: Optional[Backend] = None
) -> StreamingExecutor:
    """Create a streaming executor."""
    ...

def result_accumulator() -> ResultAccumulator:
    """Create a result accumulator."""
    ...

def process_stream(
    graph: EinsumGraph,
    input_iterator: any,
    input_name: str,
    backend: Optional[Backend] = None
) -> List[npt.NDArray[np.float64]]:
    """Process an iterator of inputs through a graph."""
    ...

# ============================================================================
# Utility Functions and Context Managers
# ============================================================================

# Custom Exceptions
class CompilationError(Exception):
    """Raised when compilation fails."""
    ...

class ExecutionError(Exception):
    """Raised when execution fails."""
    ...

class ValidationError(Exception):
    """Raised when input validation fails."""
    ...

class BackendError(Exception):
    """Raised when backend operations fail."""
    ...

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    ...

class ExecutionContext:
    """Execution context manager for managed graph execution."""
    def __init__(self, graph: EinsumGraph) -> None: ...
    def execute(self, inputs: Dict[str, npt.NDArray[np.float64]]) -> Dict[str, npt.NDArray[np.float64]]: ...
    def get_results(self) -> List[Dict[str, npt.NDArray[np.float64]]]: ...
    def execution_count(self) -> int: ...
    def clear_results(self) -> None: ...
    def __enter__(self) -> "ExecutionContext": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...

class CompilationContext:
    """Compilation context manager for managed compilation."""
    def __init__(self, config: Optional[CompilationConfig] = None) -> None: ...
    def compile(self, expr: TLExpr, name: Optional[str] = None) -> EinsumGraph: ...
    def get_graphs(self) -> List[tuple[str, EinsumGraph]]: ...
    def get_graph(self, name: str) -> Optional[EinsumGraph]: ...
    def graph_count(self) -> int: ...
    def __enter__(self) -> "CompilationContext": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...

def quick_execute(
    expr: TLExpr,
    inputs: Dict[str, npt.NDArray[np.float64]],
    config: Optional[CompilationConfig] = None
) -> Dict[str, npt.NDArray[np.float64]]:
    """Quick compile and execute in one step."""
    ...

def validate_inputs(
    inputs: Dict[str, npt.NDArray[np.float64]],
    expected_names: List[str],
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None
) -> None:
    """Validate input tensors against expected schema."""
    ...

def batch_compile(
    expressions: List[TLExpr],
    config: Optional[CompilationConfig] = None
) -> List[EinsumGraph]:
    """Create multiple graphs from expressions in batch."""
    ...

def batch_predict(
    graph: EinsumGraph,
    inputs_list: List[Dict[str, npt.NDArray[np.float64]]]
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """Predict on multiple inputs using the same graph."""
    ...

def execution_context(graph: EinsumGraph) -> ExecutionContext:
    """Create an execution context."""
    ...

def compilation_context(config: Optional[CompilationConfig] = None) -> CompilationContext:
    """Create a compilation context."""
    ...
