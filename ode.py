"""
ODE - Base class for ordinary differential equations

This module provides the base functionality for all ordinary differential
equation objects used in the nnde software.

This class is currently an abstract ancestor class. It must be subclassed    
to be useful.

Example:
    Create an empty ODE object.
        ode = ODE()

Methods:
    __init__() - Constructor

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from differentialequation import DifferentialEquation


class ODE(DifferentialEquation):
    """Abstract base class for all ordinary differential equation objects"""

    def __init__(self):
        """Constructor for ODE objects - must be  overridden in subclass."""
        super().__init__()


if __name__ == '__main__':
    ode = ODE()
    print(ode)