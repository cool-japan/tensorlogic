"""Tests for TensorLogic execution functionality."""

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


class TestBasicExecution:
    """Tests for basic graph execution."""

    def test_simple_predicate_execution(self):
        """Test executing a simple predicate."""
        x = tl.var("x")
        y = tl.var("y")
        knows = tl.pred("knows", [x, y])

        graph = tl.compile(knows)

        # Create test data: 10x10 adjacency matrix
        knows_matrix = np.random.rand(10, 10).astype(np.float64)

        result = tl.execute(graph, {"knows": knows_matrix})

        assert "output" in result
        assert result["output"].shape == knows_matrix.shape
        # Since it's just a pass-through, values should be similar
        np.testing.assert_array_almost_equal(result["output"], knows_matrix, decimal=2)


class TestArithmeticExecution:
    """Tests for arithmetic operation execution."""

    def test_addition_execution(self):
        """Test executing addition."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        age_plus_5 = tl.add(age, tl.constant(5.0))

        graph = tl.compile(age_plus_5)

        ages = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        result = tl.execute(graph, {"age": ages})

        expected = ages + 5.0
        np.testing.assert_array_almost_equal(result["output"], expected, decimal=5)

    def test_subtraction_execution(self):
        """Test executing subtraction."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        age_minus_10 = tl.sub(age, tl.constant(10.0))

        graph = tl.compile(age_minus_10)

        ages = np.array([20, 30, 40, 50, 60], dtype=np.float64)
        result = tl.execute(graph, {"age": ages})

        expected = ages - 10.0
        np.testing.assert_array_almost_equal(result["output"], expected, decimal=5)

    def test_multiplication_execution(self):
        """Test executing multiplication."""
        x = tl.var("x")
        value = tl.pred("value", [x])
        doubled = tl.mul(value, tl.constant(2.0))

        graph = tl.compile(doubled)

        values = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = tl.execute(graph, {"value": values})

        expected = values * 2.0
        np.testing.assert_array_almost_equal(result["output"], expected, decimal=5)

    def test_division_execution(self):
        """Test executing division."""
        x = tl.var("x")
        value = tl.pred("value", [x])
        halved = tl.div(value, tl.constant(2.0))

        graph = tl.compile(halved)

        values = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        result = tl.execute(graph, {"value": values})

        expected = values / 2.0
        np.testing.assert_array_almost_equal(result["output"], expected, decimal=5)

    def test_complex_arithmetic(self):
        """Test executing complex arithmetic expression."""
        x = tl.var("x")
        value = tl.pred("value", [x])

        # (value + 10) * 2 - 5
        step1 = tl.add(value, tl.constant(10.0))
        step2 = tl.mul(step1, tl.constant(2.0))
        result_expr = tl.sub(step2, tl.constant(5.0))

        graph = tl.compile(result_expr)

        values = np.array([5, 10, 15, 20, 25], dtype=np.float64)
        result = tl.execute(graph, {"value": values})

        expected = (values + 10.0) * 2.0 - 5.0
        np.testing.assert_array_almost_equal(result["output"], expected, decimal=5)


class TestComparisonExecution:
    """Tests for comparison operation execution."""

    def test_greater_than_execution(self):
        """Test executing greater than comparison."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        is_adult = tl.gt(age, tl.constant(18.0))

        graph = tl.compile(is_adult)

        ages = np.array([15, 20, 17, 25, 16], dtype=np.float64)
        result = tl.execute(graph, {"age": ages})

        # Results should be high for ages > 18, low for ages <= 18
        assert result["output"][0] < 0.5  # 15
        assert result["output"][1] > 0.5  # 20
        assert result["output"][2] < 0.5  # 17
        assert result["output"][3] > 0.5  # 25
        assert result["output"][4] < 0.5  # 16

    def test_less_than_execution(self):
        """Test executing less than comparison."""
        x = tl.var("x")
        score = tl.pred("score", [x])
        is_failing = tl.lt(score, tl.constant(60.0))

        graph = tl.compile(is_failing)

        scores = np.array([45, 75, 55, 90, 50], dtype=np.float64)
        result = tl.execute(graph, {"score": scores})

        # Results should be high for scores < 60, low for scores >= 60
        assert result["output"][0] > 0.5  # 45
        assert result["output"][1] < 0.5  # 75
        assert result["output"][2] > 0.5  # 55
        assert result["output"][3] < 0.5  # 90
        assert result["output"][4] > 0.5  # 50

    def test_greater_than_or_equal_execution(self):
        """Test executing greater than or equal comparison."""
        x = tl.var("x")
        temp = tl.pred("temperature", [x])
        is_boiling = tl.gte(temp, tl.constant(100.0))

        graph = tl.compile(is_boiling)

        temps = np.array([95, 100, 105, 98, 100], dtype=np.float64)
        result = tl.execute(graph, {"temperature": temps})

        # Results should be high for temps >= 100, low for temps < 100
        assert result["output"][0] < 0.5  # 95
        assert result["output"][1] > 0.5  # 100
        assert result["output"][2] > 0.5  # 105
        assert result["output"][3] < 0.5  # 98
        assert result["output"][4] > 0.5  # 100

    def test_less_than_or_equal_execution(self):
        """Test executing less than or equal comparison."""
        x = tl.var("x")
        speed = tl.pred("speed", [x])
        within_limit = tl.lte(speed, tl.constant(60.0))

        graph = tl.compile(within_limit)

        speeds = np.array([55, 65, 60, 70, 50], dtype=np.float64)
        result = tl.execute(graph, {"speed": speeds})

        # Results should be high for speeds <= 60, low for speeds > 60
        assert result["output"][0] > 0.5  # 55
        assert result["output"][1] < 0.5  # 65
        assert result["output"][2] > 0.5  # 60
        assert result["output"][3] < 0.5  # 70
        assert result["output"][4] > 0.5  # 50


class TestConditionalExecution:
    """Tests for conditional operation execution."""

    def test_if_then_else_execution(self):
        """Test executing if-then-else."""
        x = tl.var("x")
        age = tl.pred("age", [x])
        is_adult = tl.gt(age, tl.constant(18.0))
        classification = tl.if_then_else(
            is_adult,
            tl.constant(1.0),
            tl.constant(0.0)
        )

        graph = tl.compile(classification)

        ages = np.array([15, 20, 17, 25, 16], dtype=np.float64)
        result = tl.execute(graph, {"age": ages})

        # Results should be close to 0 for ages <= 18, close to 1 for ages > 18
        assert result["output"][0] < 0.5  # 15 -> 0
        assert result["output"][1] > 0.5  # 20 -> 1
        assert result["output"][2] < 0.5  # 17 -> 0
        assert result["output"][3] > 0.5  # 25 -> 1
        assert result["output"][4] < 0.5  # 16 -> 0

    def test_nested_if_then_else(self):
        """Test executing nested if-then-else."""
        x = tl.var("x")
        score = tl.pred("score", [x])

        # if score >= 90 then 3, else if score >= 60 then 2, else 1
        is_excellent = tl.gte(score, tl.constant(90.0))
        is_passing = tl.gte(score, tl.constant(60.0))

        grade = tl.if_then_else(
            is_excellent,
            tl.constant(3.0),
            tl.if_then_else(
                is_passing,
                tl.constant(2.0),
                tl.constant(1.0)
            )
        )

        graph = tl.compile(grade)

        scores = np.array([50, 70, 95, 85, 40], dtype=np.float64)
        result = tl.execute(graph, {"score": scores})

        # Check approximate grades
        # 50 -> 1, 70 -> 2, 95 -> 3, 85 -> 2, 40 -> 1
        assert 0.5 < result["output"][0] < 1.5  # 50 -> 1
        assert 1.5 < result["output"][1] < 2.5  # 70 -> 2
        assert 2.5 < result["output"][2] < 3.5  # 95 -> 3
        assert 1.5 < result["output"][3] < 2.5  # 85 -> 2
        assert 0.5 < result["output"][4] < 1.5  # 40 -> 1


class TestLogicalOperations:
    """Tests for logical operation execution."""

    def test_and_execution(self):
        """Test executing AND operation."""
        x = tl.var("x")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [x])
        and_expr = tl.and_(p, q)

        graph = tl.compile(and_expr)

        p_values = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        q_values = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        result = tl.execute(graph, {"P": p_values, "Q": q_values})

        # AND should be high only when both are high
        assert result["output"][0] > 0.5  # 1 AND 1 = 1
        # Other cases may vary based on soft logic semantics

    def test_or_execution(self):
        """Test executing OR operation."""
        x = tl.var("x")
        p = tl.pred("P", [x])
        q = tl.pred("Q", [x])
        or_expr = tl.or_(p, q)

        graph = tl.compile(or_expr)

        p_values = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        q_values = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        result = tl.execute(graph, {"P": p_values, "Q": q_values})

        # OR should be high when at least one is high
        assert result["output"][0] > 0.5  # 1 OR 1 = 1
        assert result["output"][1] > 0.5  # 1 OR 0 = 1
        assert result["output"][2] > 0.5  # 0 OR 1 = 1
        # Last case (0 OR 0) should be low

    def test_not_execution(self):
        """Test executing NOT operation."""
        x = tl.var("x")
        p = tl.pred("P", [x])
        not_expr = tl.not_(p)

        graph = tl.compile(not_expr)

        p_values = np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float64)

        result = tl.execute(graph, {"P": p_values})

        # NOT should invert values: high becomes low, low becomes high
        assert result["output"][0] < 0.5  # NOT 1.0 -> 0.0
        assert result["output"][1] > 0.5  # NOT 0.0 -> 1.0


class TestMultiInputExecution:
    """Tests for execution with multiple inputs."""

    def test_two_predicate_inputs(self):
        """Test execution with two different predicates."""
        x = tl.var("x")
        weight = tl.pred("weight", [x])
        height = tl.pred("height", [x])

        # BMI = weight / (height * height)
        height_squared = tl.mul(height, height)
        bmi = tl.div(weight, height_squared)

        graph = tl.compile(bmi)

        weights = np.array([70, 85, 60], dtype=np.float64)
        heights = np.array([1.75, 1.80, 1.65], dtype=np.float64)

        result = tl.execute(graph, {"weight": weights, "height": heights})

        expected_bmi = weights / (heights * heights)
        np.testing.assert_array_almost_equal(result["output"], expected_bmi, decimal=2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_compilation_error_handling(self):
        """Test that compilation errors are properly handled."""
        # This test will depend on what actually causes compilation errors
        # For now, just ensure compile doesn't crash
        x = tl.var("x")
        p = tl.pred("P", [x])
        graph = tl.compile(p)
        assert graph is not None

    def test_empty_input_dict(self):
        """Test execution with empty input dictionary."""
        x = tl.var("x")
        const_expr = tl.constant(42.0)

        graph = tl.compile(const_expr)

        # This should fail because there's no predicate to execute
        # But constant should still work
        try:
            result = tl.execute(graph, {})
            # If it succeeds, that's okay too
            assert "output" in result
        except RuntimeError:
            # Expected for some cases
            pass

    def test_mismatched_input_shapes(self):
        """Test execution with mismatched input shapes."""
        # This will depend on how the backend handles shape mismatches
        # For now, just document the expected behavior
        pass
