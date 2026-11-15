import ast
import math
import operator

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}

def calculator(expression: str) -> str:
    """
    Evaluate a basic mathematical expression.

    Supported operations:
    +, -, *, /, //, %, parentheses, unary + and -
    """

    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception as err:
        raise ValueError("Invalid mathematical expression.") from err

    if(len(list(ast.walk(tree))) > 50):
        raise ValueError("Expression is too complex.")

    def evaluate(node):
        
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            
            if type(node.value) not in (float, int):
                raise ValueError("Only numeric values are allowed.")

            if abs(node.value) > 10**12:
                raise ValueError("Numeric input is too large.")

            return node.value

        if isinstance(node, ast.BinOp):

            operation = ALLOWED_OPERATORS.get(type(node.op))
            
            if operation is None:
                raise ValueError("Unsupported arithmatic operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            try:
                result = operation(left, right)
            except ZeroDivisionError as err:
                raise ValueError("Division by zero is not allowed.") from err

            if isinstance(result, (int, float)):
                
                if abs(result) > 10**12:
                    raise ValueError('Calculation result is too large.')

                if isinstance(result, float) and not math.isfinite(result):
                    raise ValueError('Calculation produced and invalid number.')

            return result

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_UNARY_OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported unary operator")

            return operation(evaluate(node.operand))

        raise ValueError("Unsupported expression.")

    result = evaluate(tree)

    if isinstance(result, float):
        return f"{result:.12g}"

    return str(result)
