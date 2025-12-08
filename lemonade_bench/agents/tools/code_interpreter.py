# Copyright (c) 2024 LemonadeBench Contributors
# BSD-3-Clause License

"""
Code interpreter tool for LemonadeBench agents.

Provides sandboxed Python code execution for complex calculations,
optimization, and data analysis. Only safe mathematical and data
manipulation modules are available.

Security measures:
- AST validation to block dangerous operations
- Timeout enforcement via signal/threading
- Restricted builtins (no type introspection)
- No file/network/system access
"""

import ast
import io
import math
import signal
import contextlib
import threading
from typing import Any

from .base import Tool, ToolDefinition, ToolResult, register_tool


# Safe built-in functions that can be used in code execution
# Note: 'type' is excluded to prevent class introspection attacks
SAFE_BUILTINS = {
    # Math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    # Type conversions
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Iteration helpers
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    # Boolean
    "all": all,
    "any": any,
    # Printing (captured to stdout)
    "print": print,
    # Type checking (isinstance only, not type)
    "isinstance": isinstance,
}


# Safe modules that can be imported
SAFE_MODULES = {
    "math": math,
}

# Try to import numpy and statistics if available
try:
    import numpy as np
    SAFE_MODULES["numpy"] = np
    SAFE_MODULES["np"] = np
except ImportError:
    pass

try:
    import statistics
    SAFE_MODULES["statistics"] = statistics
except ImportError:
    pass


class RestrictedImporter:
    """Custom importer that only allows safe modules."""
    
    def find_module(self, name: str, path: Any = None):
        if name in SAFE_MODULES:
            return self
        return None
    
    def load_module(self, name: str):
        return SAFE_MODULES[name]


class CodeValidator(ast.NodeVisitor):
    """
    AST visitor that validates code for safety.
    
    Blocks:
    - Import statements (except allowed modules)
    - Attribute access to dunder methods (__class__, __bases__, etc.)
    - Dangerous function calls (eval, exec, compile, open, etc.)
    """
    
    # Allowed module imports
    ALLOWED_IMPORTS = {"math", "numpy", "np", "statistics"}
    
    # Blocked attribute names (dunder attributes for introspection)
    BLOCKED_ATTRIBUTES = {
        "__class__", "__bases__", "__mro__", "__subclasses__",
        "__init__", "__new__", "__del__", "__repr__", "__str__",
        "__dict__", "__globals__", "__locals__", "__builtins__",
        "__code__", "__func__", "__self__", "__module__",
        "__call__", "__getattr__", "__setattr__", "__delattr__",
        "__getattribute__", "__import__", "__loader__", "__spec__",
    }
    
    # Blocked function names
    BLOCKED_FUNCTIONS = {
        "eval", "exec", "compile", "open", "input",
        "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr",
        "type", "__import__", "breakpoint",
    }
    
    def __init__(self):
        self.errors: list[str] = []
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module not in self.ALLOWED_IMPORTS:
            self.errors.append(f"Import not allowed: {node.module}")
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.BLOCKED_ATTRIBUTES:
            self.errors.append(f"Attribute access not allowed: {node.attr}")
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        # Check for blocked function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BLOCKED_FUNCTIONS:
                self.errors.append(f"Function not allowed: {node.func.id}")
        self.generic_visit(node)
    
    def validate(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate code for safety.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        
        self.errors = []
        self.visit(tree)
        
        return len(self.errors) == 0, self.errors


class TimeoutError(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out")


@register_tool
class CodeInterpreterTool(Tool):
    """
    Sandboxed Python code execution tool.
    
    Allows agents to execute Python code for complex calculations like:
    - Profit optimization across price points
    - Monte Carlo simulation of weather scenarios
    - Break-even analysis
    - Demand forecasting calculations
    
    Only safe mathematical modules (math, numpy, statistics) are available.
    No file system access, network access, or dangerous operations allowed.
    """
    
    # Maximum execution time in seconds
    MAX_EXECUTION_TIME = 5.0
    
    # Maximum output length
    MAX_OUTPUT_LENGTH = 4000
    
    @property
    def name(self) -> str:
        return "run_python"
    
    @property
    def description(self) -> str:
        return (
            "Execute Python code for complex calculations, optimization, or data analysis. "
            "Has access to math, numpy (as np), and statistics modules. "
            "Use for profit calculations, demand forecasting, or break-even analysis."
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. Has access to: math, numpy (np), statistics. "
                            "Print results to see them. Example: "
                            "print(sum([p * d for p, d in zip(prices, demands)]))"
                        ),
                    },
                },
                "required": ["code"],
            },
        }
    
    def _create_safe_globals(self) -> dict[str, Any]:
        """Create a restricted globals dict for code execution."""
        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__main__",
            "__doc__": None,
        }
        
        # Add safe modules directly to globals
        safe_globals.update(SAFE_MODULES)
        
        return safe_globals
    
    def _execute_with_timeout(
        self,
        code: str,
        safe_globals: dict[str, Any],
        safe_locals: dict[str, Any],
        stdout_capture: io.StringIO,
    ) -> None:
        """Execute code with timeout using signal (Unix) or threading fallback."""
        # Try signal-based timeout (Unix only, more reliable)
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.MAX_EXECUTION_TIME))
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(code, safe_globals, safe_locals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, AttributeError):
            # signal.alarm not available (Windows), use threading fallback
            result_container: dict[str, Any] = {"exception": None}
            
            def target():
                try:
                    with contextlib.redirect_stdout(stdout_capture):
                        exec(code, safe_globals, safe_locals)
                except Exception as e:
                    result_container["exception"] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.MAX_EXECUTION_TIME)
            
            if thread.is_alive():
                raise TimeoutError("Code execution timed out")
            
            if result_container["exception"]:
                raise result_container["exception"]
    
    def execute(self, code: str, **kwargs: Any) -> ToolResult:
        """
        Execute Python code in a sandboxed environment.
        
        Security measures:
        1. AST validation to block dangerous operations
        2. Timeout enforcement (5 seconds max)
        3. Restricted builtins and globals
        
        Args:
            code: Python code to execute
            
        Returns:
            ToolResult with output or error
        """
        if not code or not code.strip():
            return ToolResult(
                success=False,
                result="",
                error="No code provided",
            )
        
        # Validate code using AST analysis (more robust than string matching)
        validator = CodeValidator()
        is_valid, errors = validator.validate(code)
        
        if not is_valid:
            return ToolResult(
                success=False,
                result="",
                error=f"Code validation failed: {'; '.join(errors)}",
            )
        
        # Capture stdout
        stdout_capture = io.StringIO()
        
        try:
            # Create restricted execution environment
            safe_globals = self._create_safe_globals()
            safe_locals: dict[str, Any] = {}
            
            # Execute with timeout
            self._execute_with_timeout(code, safe_globals, safe_locals, stdout_capture)
            
            # Get output
            output = stdout_capture.getvalue()
            
            # If no output but there's a result expression, try to get it
            if not output and safe_locals:
                # Check if there's a 'result' variable
                if "result" in safe_locals:
                    output = str(safe_locals["result"])
            
            # Truncate if too long
            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[:self.MAX_OUTPUT_LENGTH] + "\n...(truncated)"
            
            if not output:
                output = "(No output. Use print() to see results.)"
            
            return ToolResult(
                success=True,
                result=output,
            )
            
        except SyntaxError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Syntax error: {e}",
            )
        except NameError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Name error (undefined variable or function): {e}",
            )
        except TypeError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Type error: {e}",
            )
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                result="",
                error="Division by zero",
            )
        except TimeoutError as e:
            return ToolResult(
                success=False,
                result="",
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Execution error: {type(e).__name__}: {e}",
            )
        finally:
            stdout_capture.close()
