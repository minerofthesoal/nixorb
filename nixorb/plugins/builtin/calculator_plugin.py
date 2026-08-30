"""
nixorb/plugins/builtin/calculator_plugin.py

Built-in plugin: evaluate arithmetic and math expressions.

Uses Python's `ast` module to walk a restricted grammar instead of calling
`eval()` on LLM-supplied text — no name lookups, no attribute access, no
calls other than the whitelisted `math` functions below.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Callable, cast

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic or math expression, e.g. '2 + 2', "
            "'(15 * 3.5) / 7', 'sqrt(144)', 'sin(pi/2)', '2**10'. "
            "Supports +, -, *, /, //, %, **, and the functions/constants "
            "in Python's math module (sqrt, sin, cos, tan, log, log2, "
            "log10, exp, floor, ceil, factorial, pi, e, tau, inf)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
}

_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Whitelisted names — only math-module callables/constants, nothing else.
_NAMES: dict[str, object] = {
    name: getattr(math, name)
    for name in (
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "log", "log2", "log10", "exp", "floor", "ceil", "factorial",
        "degrees", "radians", "hypot", "pow", "pi", "e", "tau", "inf",
    )
}
_NAMES["abs"] = abs
_NAMES["round"] = round


class _UnsafeExpression(ValueError):
    pass


def _eval_node(node: ast.AST):  # noqa: ANN201 - recursive, mixed return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise _UnsafeExpression(f"constant {node.value!r} is not a number")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _NAMES and not callable(_NAMES[node.id]):
            return _NAMES[node.id]
        raise _UnsafeExpression(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _NAMES:
            raise _UnsafeExpression("only whitelisted math functions may be called")
        func = _NAMES[node.func.id]
        if not callable(func):
            raise _UnsafeExpression(f"'{node.func.id}' is not callable")
        if node.keywords:
            raise _UnsafeExpression("keyword arguments are not supported")
        args = [_eval_node(a) for a in node.args]
        return cast(Callable[..., float], func)(*args)
    raise _UnsafeExpression(f"expression contains unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except _UnsafeExpression as exc:
        return f"Can't evaluate that: {exc}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, TypeError, OverflowError, ValueError) as exc:
        return f"Error: {exc}"

    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return str(result)
