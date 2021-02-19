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
"""


from nnde.differentialequation.ode.ode import ODE


class ODE1(ODE):
    """Abstract base class for all 1st-order ordinary differential
    equation objects"""

    def __init__(self):
        """Constructor for ODE1 objects - must be  overridden in subclass."""
        ODE.__init__(self)

    def G(self, x, Y, dY_dx):
        """Differential equation  to be solved, in the form G() = 0 -
        must be overridden in subclass.

        x is the value of the independent variable.
        Y is the value of the solution.
        dY_dx is the value of the 1st derivative of the solution.
        """
        raise Exception
