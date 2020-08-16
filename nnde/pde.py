"""
PDE - Base class for partial differential equations

This module provides the base functionality for all partial differential
equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty PDE object.
        pde = PDE()

Attributes:
    None

Methods:
    __init__() - Constructor

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from nnde.differentialequation import DifferentialEquation


class PDE(DifferentialEquation):
    """Abstract base class for all partial differential equation objects"""

    def __init__(self):
        """Constructor for PDE objects - must be  overridden in subclass."""
        super().__init__()


if __name__ == '__main__':
    pde = PDE()
    print(pde)
