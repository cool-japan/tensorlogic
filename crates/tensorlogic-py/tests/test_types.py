"""Tests for TensorLogic Python type bindings."""

import pytest
import numpy as np

try:
    import pytensorlogic as tl
    HAS_TENSORLOGIC = True
except ImportError:
    HAS_TENSORLOGIC = False

pytestmark = pytest.mark.skipif(
    not HAS_TENSORLOGIC,
    reason="pytensorlogic not available - build with 'maturin develop'"
)


class TestTerm:
    """Tests for Term type."""

    def test_var_creation(self):
        """Test creating a variable term."""
        x = tl.var("x")
        assert x.name() == "x"
        assert x.is_var()
        assert not x.is_const()

    def test_const_creation(self):
        """Test creating a constant term."""
        alice = tl.const("alice")
        assert alice.name() == "alice"
        assert alice.is_const()
        assert not alice.is_var()

    def test_term_repr(self):
        """Test term string representation."""
        x = tl.var("x")
        assert "x" in repr(x)
        assert "x" in str(x)


class TestTLExpr:
    """Tests for TLExpr type."""

    def test_pred_creation(self):
        """Test creating a predicate."""
        x = tl.var("x")
        y = tl.var("y")
        knows = tl.pred("knows", [x, y])
        assert knows is not None

    def test_and_operation(self):
        """Test AND logical operation."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        and_expr = tl.and_(p, q)
        assert and_expr is not None

    def test_or_operation(self):
        """Test OR logical operation."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        or_expr = tl.or_(p, q)
        assert or_expr is not None

    def test_not_operation(self):
        """Test NOT logical operation."""
        x = tl.var("x")
        p = tl.pred("P", [x])
        not_expr = tl.not_(p)
        assert not_expr is not None

    def test_exists_quantifier(self):
        """Test existential quantifier."""
        x = tl.var("x")
        y = tl.var("y")
        knows = tl.pred("knows", [x, y])
        exists_expr = tl.exists("y", "Person", knows)
        assert exists_expr is not None

    def test_forall_quantifier(self):
        """Test universal quantifier."""
        x = tl.var("x")
        y = tl.var("y")
        knows = tl.pred("knows", [x, y])
        forall_expr = tl.forall("y", "Person", knows)
        assert forall_expr is not None

    def test_imply_operation(self):
        """Test implication operation."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        imply_expr = tl.imply(p, q)
        assert imply_expr is not None

    def test_constant_creation(self):
        """Test numeric constant creation."""
        const = tl.constant(3.14)
        assert const is not None

    def test_free_vars(self):
        """Test free variables extraction."""
        x = tl.var("x")
        y = tl.var("y")
        knows = tl.pred("knows", [x, y])
        free_vars = knows.free_vars()
        assert "x" in free_vars
        assert "y" in free_vars


class TestArithmeticOperations:
    """Tests for arithmetic operations."""

    def test_add_operation(self):
        """Test addition operation."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        add_expr = tl.add(age, tl.constant(5.0))
        assert add_expr is not None

    def test_sub_operation(self):
        """Test subtraction operation."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        sub_expr = tl.sub(age, tl.constant(5.0))
        assert sub_expr is not None

    def test_mul_operation(self):
        """Test multiplication operation."""
        x = tl.var("x")
        salary = tl.pred("salary", [x])
        mul_expr = tl.mul(salary, tl.constant(1.1))
        assert mul_expr is not None

    def test_div_operation(self):
        """Test division operation."""
        x = tl.var("x")
        total = tl.pred("total", [x])
        count = tl.pred("count", [x])
        div_expr = tl.div(total, count)
        assert div_expr is not None


class TestComparisonOperations:
    """Tests for comparison operations."""

    def test_eq_operation(self):
        """Test equality comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        eq_expr = tl.eq(age, tl.constant(18.0))
        assert eq_expr is not None

    def test_lt_operation(self):
        """Test less-than comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        lt_expr = tl.lt(age, tl.constant(18.0))
        assert lt_expr is not None

    def test_gt_operation(self):
        """Test greater-than comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        gt_expr = tl.gt(age, tl.constant(18.0))
        assert gt_expr is not None

    def test_lte_operation(self):
        """Test less-than-or-equal comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        lte_expr = tl.lte(age, tl.constant(18.0))
        assert lte_expr is not None

    def test_gte_operation(self):
        """Test greater-than-or-equal comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        gte_expr = tl.gte(age, tl.constant(18.0))
        assert gte_expr is not None


class TestConditionalOperations:
    """Tests for conditional operations."""

    def test_if_then_else(self):
        """Test if-then-else conditional."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        is_adult = tl.gt(age, tl.constant(18.0))
        conditional = tl.if_then_else(
            is_adult,
            tl.constant(1.0),
            tl.constant(0.0)
        )
        assert conditional is not None


class TestEinsumGraph:
    """Tests for EinsumGraph type."""

    def test_graph_compilation(self):
        """Test compiling expression to graph."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        # Use AND to create actual computation nodes
        expr = tl.and_(p, q)
        graph = tl.compile(expr)

        assert graph is not None
        assert graph.num_nodes > 0  # AND creates an einsum node
        assert graph.num_outputs > 0

    def test_graph_stats(self):
        """Test graph statistics."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        expr = tl.and_(p, q)
        graph = tl.compile(expr)

        stats = graph.stats()
        assert isinstance(stats, dict)
        assert stats["num_tensors"] >= 2
        assert stats["num_nodes"] >= 1
        assert "num_outputs" in stats
        assert "avg_inputs_per_node" in stats

    def test_graph_repr(self):
        """Test graph string representation."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        expr = tl.and_(p, q)
        graph = tl.compile(expr)

        repr_str = repr(graph)
        assert "EinsumGraph" in repr_str
        assert "nodes" in repr_str


class TestCompilationConfig:
    """Tests for compilation configuration."""

    def test_soft_differentiable_config(self):
        """Test soft differentiable configuration."""
        config = tl.CompilationConfig.soft_differentiable()
        assert config is not None

    def test_hard_boolean_config(self):
        """Test hard Boolean configuration."""
        config = tl.CompilationConfig.hard_boolean()
        assert config is not None

    def test_fuzzy_godel_config(self):
        """Test Gödel fuzzy logic configuration."""
        config = tl.CompilationConfig.fuzzy_godel()
        assert config is not None

    def test_fuzzy_product_config(self):
        """Test product fuzzy logic configuration."""
        config = tl.CompilationConfig.fuzzy_product()
        assert config is not None

    def test_fuzzy_lukasiewicz_config(self):
        """Test Łukasiewicz fuzzy logic configuration."""
        config = tl.CompilationConfig.fuzzy_lukasiewicz()
        assert config is not None

    def test_probabilistic_config(self):
        """Test probabilistic configuration."""
        config = tl.CompilationConfig.probabilistic()
        assert config is not None

    def test_compile_with_config(self):
        """Test compilation with custom config."""
        x = tl.var("x")
        y = tl.var("y")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [y])
        expr = tl.and_(p, q)

        config = tl.CompilationConfig.soft_differentiable()
        graph = tl.compile_with_config(expr, config)
        assert graph is not None
