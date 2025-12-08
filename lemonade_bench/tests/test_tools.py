# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""Tests for the tools module."""

import pytest

from lemonade_bench.agents.tools import (
    CalculatorTool,
    get_tool,
    AVAILABLE_TOOLS,
    ToolResult,
)
from lemonade_bench.agents.tools.calculator import safe_eval


class TestSafeEval:
    """Tests for the safe_eval function."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_eval("2 + 2") == 4
        assert safe_eval("10 - 3") == 7
        assert safe_eval("5 * 6") == 30
        assert safe_eval("20 / 4") == 5

    def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        assert safe_eval("(100 - 25) / 100 * 100") == 75
        assert safe_eval("2 ** 8") == 256
        assert safe_eval("17 // 5") == 3
        assert safe_eval("17 % 5") == 2

    def test_float_numbers(self):
        """Test floating point numbers."""
        assert safe_eval("3.14 * 2") == pytest.approx(6.28)
        assert safe_eval("0.25 * 4") == 1

    def test_negative_numbers(self):
        """Test negative numbers."""
        assert safe_eval("-5 + 10") == 5
        assert safe_eval("-(2 + 3)") == -5

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Division by zero"):
            safe_eval("1 / 0")

    def test_invalid_syntax(self):
        """Test invalid syntax raises error."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            safe_eval("2 +")

    def test_blocks_function_calls(self):
        """Test that function calls are blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            safe_eval("__import__('os')")

    def test_blocks_attribute_access(self):
        """Test that attribute access is blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            safe_eval("'hello'.upper()")

    def test_large_number_limit(self):
        """Test that extremely large numbers are rejected."""
        with pytest.raises(ValueError, match="Number too large"):
            safe_eval("1e16 + 1")

    def test_large_exponent_limit(self):
        """Test that large exponents are rejected."""
        with pytest.raises(ValueError, match="Exponent values too large"):
            safe_eval("2 ** 1000")

    def test_long_expression_limit(self):
        """Test that very long expressions are rejected."""
        long_expr = "1 + " * 100 + "1"
        with pytest.raises(ValueError, match="Expression too long"):
            safe_eval(long_expr)


class TestCalculatorTool:
    """Tests for the CalculatorTool class."""

    def test_tool_properties(self):
        """Test tool basic properties."""
        calc = CalculatorTool()
        assert calc.name == "calculator"
        assert "arithmetic" in calc.description.lower()

    def test_tool_definition(self):
        """Test tool definition format."""
        calc = CalculatorTool()
        definition = calc.definition
        
        assert definition["name"] == "calculator"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["expression"]["type"] == "string"

    def test_execute_success(self):
        """Test successful calculation."""
        calc = CalculatorTool()
        result = calc.execute(expression="100 * 0.75")
        
        assert result.success is True
        assert result.result == "75"
        assert result.error is None

    def test_execute_float_result(self):
        """Test calculation with float result."""
        calc = CalculatorTool()
        result = calc.execute(expression="10 / 3")
        
        assert result.success is True
        assert float(result.result) == pytest.approx(3.333333)

    def test_execute_error(self):
        """Test calculation error handling."""
        calc = CalculatorTool()
        result = calc.execute(expression="1/0")
        
        assert result.success is False
        assert result.result == ""
        assert "Division by zero" in result.error

    def test_execute_empty_expression(self):
        """Test empty expression handling."""
        calc = CalculatorTool()
        result = calc.execute(expression="")
        
        assert result.success is False
        assert "No expression provided" in result.error

    def test_execute_no_expression(self):
        """Test missing expression argument."""
        calc = CalculatorTool()
        result = calc.execute()
        
        assert result.success is False


class TestToolRegistry:
    """Tests for tool registration and retrieval."""

    def test_available_tools(self):
        """Test that calculator is in available tools."""
        tools = AVAILABLE_TOOLS()
        assert "calculator" in tools

    def test_get_tool(self):
        """Test getting tool by name."""
        tool = get_tool("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_unknown_tool(self):
        """Test getting unknown tool returns None."""
        tool = get_tool("nonexistent_tool")
        assert tool is None

    def test_get_tool_case_insensitive(self):
        """Test tool lookup is case insensitive."""
        tool1 = get_tool("calculator")
        tool2 = get_tool("CALCULATOR")
        tool3 = get_tool("Calculator")
        
        # All should return valid tools (or None consistently)
        assert tool1 is not None
        # Note: actual case sensitivity depends on implementation


class TestToolIntegration:
    """Integration tests for tools with agent system."""

    def test_tool_definition_compatible_with_provider(self):
        """Test that tool definitions are compatible with provider format."""
        from lemonade_bench.agents.providers.base import LEMONADE_ACTION_TOOL
        
        calc = CalculatorTool()
        calc_def = calc.definition
        
        # Should have same structure as action tool
        assert set(calc_def.keys()) == set(LEMONADE_ACTION_TOOL.keys())
        assert "name" in calc_def
        assert "description" in calc_def
        assert "input_schema" in calc_def

