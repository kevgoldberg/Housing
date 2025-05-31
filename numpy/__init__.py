"""Lightweight fallback implementation of the small subset of NumPy needed for tests."""

__all__ = ["add"]


def add(a, b):
    """Return the sum of *a* and *b*."""
    return a + b
