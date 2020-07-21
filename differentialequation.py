"""
DifferentialEquation - Base class for differential equations

This module provides the base functionality for all differential equation
objects used in the nnde software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty DifferentialEquation object.
        diffeq = DifferentialEquation()

Attributes:
    None

Methods:
    __init__() - Constructor
    G - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), and the Jacobian, Hessian, and higher derivatives of Y(x).

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


class DifferentialEquation:
    """Abstract base class for all differential equation objects"""

    def __init__(self):
        """Constructor for DifferentialEquation objects - must be
        overridden in subclass."""
        pass

    def G(self):
        """Differential equation  to be solved, in the form G() = 0 -
        must be overridden in subclass."""
        return None


if __name__ == '__main__':
    diffeq = DifferentialEquation()
    print(diffeq)
