"""
Simple calculator functions.
"""

def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return a - b."""
    return a - b


def multiply(a, b):
    """Return a * b."""
    return a * b


def divide(a, b):
    """Return a / b.

    Raises ZeroDivisionError if b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
