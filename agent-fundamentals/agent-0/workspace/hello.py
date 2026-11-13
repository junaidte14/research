#!/usr/bin/env python3

"""
Print the first 10 Fibonacci numbers.
"""

def fib(n):
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

if __name__ == "__main__":
    numbers = list(fib(10))
    print(numbers)
