"""
This module implements problem 3 in Lagaris et al (1998) (2nd order ODE
BVP).

Note that an upper-case 'Y' is used to represent the Greek psi from the
original equation.

The equation is defined on the domain [0,1]:

The analytical form of the equation is:
    G(x, Y, dY/dx) = d2Y/dx2 + 1/5 dY/dx + Y + exp(-x/5)*cos(x) = 0

with boundary condition:

Y(0) = 0
Y(1) = sin(1)*exp(-1/5)

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis,
"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999,
1998
"""


from math import cos, exp, sin
import numpy as np


# Specify the initial condition for this ODE: Y(0) = 0
bc0 = 0
bc1 = sin(1)*exp(-1/5)


def G(x, Y, dY_dx, d2Y_dx2):
    """Compute the differential equation.
    x is the scalar value of the independent variable.
    Y is the scalar value of the solution.
    dY_dx is the scalar value of the 1st derivative of the solution.
    d2Y_dx2 is the scalar value of the 2nd derivative of the solution.
    """
    return d2Y_dx2 + 1/5*dY_dx + Y + exp(-x/5)*cos(x)


def dG_dY(x, Y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt Y"""
    return 1/5


def dG_ddYdx(x, y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt dY/dx"""
    return 1/5


def dG_dd2Ydx2(x, y, dY_dx):
    """Derivative of G(x, Y, dY_dx) wrt d2Y/dx2"""
    return 1


def Ya(x):
    """Analytical solution"""
    return exp(-x/5)*sin(x)


def dYa_dx(x):
    """1st derivative of analytical solution"""
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))


def d2Ya_dx2(x):
    """2nd derivative of analytical solution"""
    return 1/5*exp(-x/5)*(-5*sin(x) - cos(x)) - 1/25*exp(-x/5)*(5*cos(x) - sin(x))
