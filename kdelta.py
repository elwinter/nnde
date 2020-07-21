"""
kdelta - Python module to implement the Kronecker delta function

This module provides the Kronecker delta function:

kdelta(i, j) = 1 if i == j, else 0

Example:
    Calculate the Kronecker delta for 2 integers.
        kd = kdelta.kdelta(i, j)

To do:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = ['kdelta']
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


def kdelta(i, j):
    """
    The Konecker delta function.

    Given 2 integers i and j, return 1 if i == j, else 0.
    """

    return 1 if i == j else 0


if __name__ == '__main__':
    for i in range(-2, 3):
        for j in range(-2, 3):
            print("Testing kdelta[%d, %d]." % (i, j))
            kd = kdelta(i, j)
            if i == j:
                assert kd == 1
            else:
                assert kd == 0
