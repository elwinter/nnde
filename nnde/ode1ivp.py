"""
ODE1IVP - Base class for 1st-order ordinary differential equation initial-
value problems

This module provides the base functionality for all 1st-order ordinary
differential equation initial-value problem objects used in the nnde
software.

This class is typically used to create an object from the contents of an
appropriately-designed Python module. The module must define all of the
variables and functions listed under Attributes.

Since this is a 1st-order ODE, the general form of the equation is:

G(x, Y, dY_dx) = 0

where x is the (scalar) independent variable, Y is the solution,
and dY_dx is the 1st derivative of the solution.

The initial condition (ic) is the a priori specified value of Y(0).

dG_dY() is the derivative of the differential equation wrt the
solution Y.

dG_ddYdx() is the derivative of the differential equation wrt the first
derivative dY/dx.

Ya() is the (optional) analytical solution, used for validating
results.

dYa_dx() is the (optional) analytical 1st derivative, used for validating
results.

Example:
    Create an empty ODE1IVP object.
        ode1ivp = ODE1IVP()
    Create an ODE1IVP object from a Python module.
        ode1ivp = ODE1IVP(modname)

Attributes:
    name - String containing name of equation definition module
    G(x, Y, dY_dx) - Function for differential equation
    ic - Scalar for initial condition Y(x=0)
    dG_dY(x, Y, dY_dx) - Function for derivative of G wrt Y
    dG_ddYdx(x, Y, dY_dx) - Function for derivative of G wrt dY/dx
    Ya(x) - (Optional) function for analytical solution
    dYa_dx(x) - (Optional) function for analytical 1st derivative

Methods:
    __init__() - Constructor

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from importlib import import_module

from nnde.ode1 import ODE1


class ODE1IVP(ODE1):
    """Base class for all 1st-order ordinary differential equation initial-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        """Constructor for ODE1IVP objects

        Parameters:
        diffeqmod - The name of the Python module containing the problem
        definition.
        """
        super().__init__()

        # Initialize all attributes.
        self.name = None
        self.G = None
        self.ic = None
        self.dG_dY = None
        self.dG_ddYdx = None
        self.Ya = None
        self.dYa_dx = None

        # If a module is specified, use the contents to populate the
        # attributes of this object.
        if diffeqmod:

            # Save the name of the Python module which defines the equation.
            self.name = diffeqmod

            # Read and validate the equation definition module.
            odemod = import_module(diffeqmod)
            assert odemod.G
            assert odemod.ic is not None
            assert odemod.dG_dY
            assert odemod.dG_ddYdx

            # Save the evaluation function for the differential equation.
            self.G = odemod.G

            # Save the initial condition.
            self.ic = odemod.ic

            # Save the derivatives of the differential equation wrt Y and
            # dY/dx.
            self.dG_dY = odemod.dG_dY
            self.dG_ddYdx = odemod.dG_ddYdx

            # If provided, save the analytical solution and 1st derivative.
            if odemod.Ya:
                self.Ya = odemod.Ya
            if odemod.dYa_dx:
                self.dYa_dx = odemod.dYa_dx


if __name__ == '__main__':
    ode1ivp = ODE1IVP()
    print(ode1ivp)
    ode1ivp = ODE1IVP('eq.lagaris_01')
    print(ode1ivp)
