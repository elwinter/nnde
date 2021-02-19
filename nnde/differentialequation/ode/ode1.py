"""
ODE1 - Base class for 1st-order ordinary differential equations

This module provides the base functionality for all 1st-order ordinary
differential equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty ODE1 object.
        ode1 = ODE1()

Attributes:
    None

Methods:
    __init__() - Constructor
    G - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variable x, the solution
    Y(x), and the first derivative dY/x.

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from nnde.ode import ODE


class ODE1(ODE):
    """Abstract base class for all 1st-order ordinary differential
    equation objects"""

    def __init__(self):
        """Constructor for ODE1 objects - must be  overridden in subclass."""
        super().__init__()

    def G(self, x, Y, dY_dx):
        """Differential equation  to be solved, in the form G() = 0 -
        must be overridden in subclass.

        x is the value of the independent variable.
        Y is the value of the solution.
        dY_dx is the value of the 1st derivative of the solution.
        """
        return None


if __name__ == '__main__':
    ode1 = ODE1()
    print(ode1)
