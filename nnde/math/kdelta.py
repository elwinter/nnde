"""Python module to implement the Kronecker delta function

This module provides the Kronecker delta function:

kdelta(i, j) = 1 if i == j, else 0

Example:
    Calculate the Kronecker delta for 2 integers.
        kd = kdelta.kdelta(i, j)
"""


def kdelta(i: int, j: int) -> int:
    """The Konecker delta function

    Given 2 integers i and j, return 1 if i == j, else 0.
    """

    return 1 if i == j else 0
