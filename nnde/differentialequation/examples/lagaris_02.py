"""
This module implements problem 2 in Lagaris et al (1998) (1st order ODE
IVP).

Note that an upper-case 'Y' is used to represent the Greek psi from the
original equation.

The equation is defined on the domain [0,1]:

The analytical form of the equation is:
    G(x, Y, dY/dx) = dY/dx + Y/5 - exp(-x/5)*cos(x) = 0

with initial condition:

Y(0) = 0

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis,
"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999,
1998

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import cos, exp, sin
import numpy as np


# Specify the initial condition for this ODE: Y(0) = 1
ic = 0


def G(x, Y, dY_dx):
    """Compute the differential equation.
    x is the scalar value of the independent variable.
    Y is the scalar value of the solution.
    dY_dx is the scalar value of the 1st derivative of the solution.
    """
    return dY_dx + Y/5 - exp(-x/5)*cos(x)


def dG_dY(x, Y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt Y"""
    return 1/5


def dG_ddYdx(x, y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt dY/dx"""
    return 1


def Ya(x):
    """Analytical solution"""
    return exp(-x/5)*sin(x)


def dYa_dx(x):
    """1st derivative of analytical solution"""
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))
