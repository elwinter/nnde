"""
PDE2 - Base class for 2nd-order partial differential equations

This module provides the base functionality for all 2nd-order partial
differential equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty PDE2 object.
        pde1 = PDE2()

The solution is assumed to be a function of m independent variables. In the
methods below, x is a vector of independent variables, and delY is the
Jacobian of the solution wrt the independent variables.

Attributes:
    None

Methods:
    __init__() - Constructor
    G - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), the Jacobian of Y(x), and the Hessian of Y(x).

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from nnde.differentialequation.pde.pde import PDE


class PDE2(PDE):
    """Base class for all 2nd-order partial differential equation objects"""

    def __init__(self):
        PDE.__init__(self)

    def G(self, x, Y, delY, deldelY):
        raise Exception
