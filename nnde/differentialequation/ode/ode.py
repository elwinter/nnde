"""
ODE - Base class for ordinary differential equations

This module provides the base functionality for all ordinary differential
equation objects used in the nnde software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty ODE object.
        ode = ODE()

Attributes:
    None

Methods:
    __init__() - Constructor
"""


from nnde.differentialequation.differentialequation import DifferentialEquation
from nnde.exceptions.nndeexception import NNDEException


class ODE(DifferentialEquation):
    """Abstract base class for all ordinary differential equation objects"""

    def __init__(self):
        """Constructor for ODE objects - must be  overridden in subclass."""
        raise NNDEException
