"""
This module implements a simple ODE.

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [0,1].

The analytical form of the equation is:

    dY/dx - x = 0

with initial condition:

Y(0) = 0

This equation has the analytical solution for the supplied initial
conditions:

Ya(x) = 0.5*x**2
"""


import numpy as np


# Specify the initial condition for this ODE: Y(0) = 1
ic = 0


def G(x, Y, dY_dx):
    """Compute the differential equation.
    x is the scalar value of the independent variable.
    Y is the scalar value of the solution.
    dY_dx is the scalar value of the 1st derivative of the solution.
    """
    return dY_dx - x


def dG_dY(x, Y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt Y"""
    return 0


def dG_ddYdx(x, y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt dY/dx"""
    return 1


def Ya(x):
    """Analytical solution"""
    return 0.5*x**2


def dYa_dx(x):
    """1st derivative of analytical solution"""
    return x
