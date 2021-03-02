"""PDE - Abstract base class for partial differential equations

This module provides the base functionality for all partial differential
equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Attributes:
    None

Methods:
    None
"""


from nnde.differentialequation.differentialequation import DifferentialEquation


class PDE(DifferentialEquation):
    """Abstract base class for all partial differential equation objects"""
    pass
