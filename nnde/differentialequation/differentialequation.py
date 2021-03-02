"""Abstract base class for differential equations

This module defines the methods required for all differential equation
objects used in the nnde software.

This class is an abstract base class. It must be subclassed to be useful. All
methods in this class will raise NNDEException if called.

Attributes:
    None

Methods:
    __init__() - Constructor
    G() - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), and its derivatives.
"""


from nnde.exceptions.nndeexception import NNDEException


class DifferentialEquation:
    """Abstract base class for differential equation objects"""

    def __init__(self, *args, **kwargs):
        """Constructor for DifferentialEquation objects"""
        raise NNDEException

    def G(self, *args, **kwargs) -> float:
        """Differential equation implementation"""
        raise NNDEException
