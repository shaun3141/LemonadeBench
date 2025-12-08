# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Calculator tool for LemonadeBench agents.

Provides basic arithmetic operations to help agents with business calculations
like profit margins, cost analysis, and pricing decisions.
"""

import ast
import operator
from typing import Any

from .base import Tool, ToolDefinition, ToolResult, register_tool


# Safe operators for evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Maximum allowed values to prevent abuse
MAX_VALUE = 1e15
MAX_EXPRESSION_LENGTH = 200


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    
    Only allows basic arithmetic operations on numbers.
    Prevents code injection and resource exhaustion.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Evaluated result as float
        
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} chars)")
    
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                if abs(node.value) > MAX_VALUE:
                    raise ValueError(f"Number too large (max {MAX_VALUE})")
                return float(node.value)
            raise ValueError(f"Invalid constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            # Prevent division by zero
            if op_type == ast.Div and right == 0:
                raise ValueError("Division by zero")
            # Prevent excessive exponentiation
            if op_type == ast.Pow and (abs(left) > 1000 or abs(right) > 100):
                raise ValueError("Exponent values too large")
            result = SAFE_OPERATORS[op_type](left, right)
            if abs(result) > MAX_VALUE:
                raise ValueError(f"Result too large (max {MAX_VALUE})")
            return result
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            return SAFE_OPERATORS[op_type](operand)
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            if abs(node.n) > MAX_VALUE:
                raise ValueError(f"Number too large (max {MAX_VALUE})")
            return float(node.n)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
    return _eval(tree)


@register_tool
class CalculatorTool(Tool):
    """
    Calculator tool for basic arithmetic operations.
    
    Allows agents to perform calculations for business decisions like:
    - Profit margins: (revenue - costs) / revenue * 100
    - Break-even analysis: fixed_costs / (price - variable_cost)
    - Inventory planning: expected_demand * days_to_plan
    
    Example:
        tool = CalculatorTool()
        result = tool.execute(expression="(100 - 25) / 100 * 100")
        print(result.result)  # "75.0"
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return (
            "Perform arithmetic calculations. Useful for computing profit margins, "
            "break-even points, inventory costs, and other business math. "
            "Supports +, -, *, /, //, %, and ** operators."
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Mathematical expression to evaluate. "
                            "Examples: '100 * 0.75', '(500 - 200) / 500 * 100', '12 * 4 * 0.25'"
                        )
                    }
                },
                "required": ["expression"]
            }
        }
    
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the calculator with the given expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            ToolResult with the calculated value or error message
        """
        expression = kwargs.get("expression", "")
        
        if not expression:
            return ToolResult(
                success=False,
                result="",
                error="No expression provided"
            )
        
        try:
            result = safe_eval(expression)
            # Format nicely - remove trailing zeros for clean integers
            if result == int(result):
                formatted = str(int(result))
            else:
                formatted = f"{result:.6f}".rstrip('0').rstrip('.')
            
            return ToolResult(
                success=True,
                result=formatted
            )
        except ValueError as e:
            return ToolResult(
                success=False,
                result="",
                error=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Calculation error: {e}"
            )

