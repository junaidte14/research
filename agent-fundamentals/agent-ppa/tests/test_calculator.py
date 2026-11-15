import pytest
from tools.calculator import calculator

def test_addition():
    assert calculator("3+5") == "8"

def test_complex_expression():
    assert calculator("(125 * 8) + 450") == "1450"

def test_division():
    assert calculator("10 / 2") == "5"

def test_parentheses():
    assert calculator("(10 + 5) * 2") == "30"

def test_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero is not allowed."):
        calculator("10 / 0")

def test_reject_python_code():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('whoami')")

