"""Abstract base class for ordinary differential equations

This module defines the methods required for all ordinary differential equation
objects used in the nnde software.

This class is an abstract base class. It must be subclassed to be useful. All
methods in this class will raise NNDEException if called.

Attributes:
    None

Methods:
    None
"""


from nnde.differentialequation.differentialequation import DifferentialEquation


class ODE(DifferentialEquation):
    """Abstract base class for all ordinary differential equation objects"""
    pass
