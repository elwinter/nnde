"""Abstract base class for 1st-order ordinary differential equations

This module defines the methods required for all 1st-order ordinary
differential equation objects used in the nnde software.

This class is an abstract base class. It must be subclassed to be useful. All
methods in this class will raise NNDEException if called.

Attributes:
    None

Methods:
    G() - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variable x, the solution
    Y(x), and the first derivative dY/x.
"""


from nnde.differentialequation.ode.ode import ODE
from nnde.exceptions.nndeexception import NNDEException


class ODE1(ODE):
    """Abstract base class for all 1st-order ordinary differential
    equation objects"""

    def G(self, x: float, Y: float, dY_dx: float) -> float:
        """Differential equation  to be solved, in the form G(x, Y, dY_dx) = 0

        x is the value of the independent variable.
        Y is the value of the solution.
        dY_dx is the value of the 1st derivative of the solution.
        """
        raise NNDEException
