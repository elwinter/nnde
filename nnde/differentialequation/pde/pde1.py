"""PDE1 - Base class for 1st-order partial differential equations

This module provides the base functionality for all 1st-order partial
differential equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty PDE1 object.
        pde1 = PDE1()

The solution is assumed to be a function of m independent variables. In the
methods below, x is a vector of independent variables, and delY is the
Jacobian of the solution wrt the independent variables.

Attributes:
    None

Methods:
    G - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), and the Jacobian of Y(x).
"""


from nnde.differentialequation.pde.pde import PDE
from nnde.exceptions.nndeexception import NNDEException


class PDE1(PDE):
    """Base class for all 1st-order partial differential equation objects"""

    def G(self, x: list, Y: float, delY: list) -> float:
        """Differential equation  to be solved, in the form G() = 0 -
        must be overridden in subclass.

        x is the vector of independent variables.
        Y is the value of the solution.
        delY is the gradient of the solution.
        """
        raise NNDEException
