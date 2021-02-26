"""DifferentialEquation - Base class for differential equations

This module provides the base functionality for all differential equation
objects used in the nnde software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Attributes:
    None

Methods:
    __init__() - Constructor
    G - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), and its derivatives.
"""


from nnde.exceptions.nndeexception import NNDEException


class DifferentialEquation:
    """Abstract base class for all differential equation objects
    
    Since this is an abstract class, it must not be instantiated. All methods
    will raise an Exception.
    """

    def __init__(self, *args, **kwargs):
        """Constructor for DifferentialEquation objects"""
        raise NNDEException

    def G(self, *args, **kwargs):
        """Differential equation to be solved, in the form G() = 0
        Must be overridden in subclass."""
        raise NNDEException
