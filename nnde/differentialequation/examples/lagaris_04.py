"""
This module implements problem 4 in Lagaris et al (1998) (coupled
1st-order ODE IVP).

Note that an upper-case 'Y' is used to represent the Greek psi from the
original equation.

The equation is defined on the domain [0,1]:

The analytical forms of the equations are:

    G1(x, Y1, Y2, dY1/dx, dY2/dx) = XXX = 0
    G2(x, Y1, Y2, dY1/dx, dY2/dx) = XXX = 0

with initial condition:

Y1(0) = 0
Y2(0) = 1

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis,
"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999,
1998
"""


from math import cos, sin
import numpy as np


# # Specify the initial conditions.
# ic = [0, 1]


# def G1(x, Y, dY_dx):
#     """Compute the differential equation.
#     x is the scalar value of the independent variable.
#     Y is the scalar value of the solution.
#     dY_dx is the scalar value of the 1st derivative of the solution.
#     d2Y_dx2 is the scalar value of the 2nd derivative of the solution.
#     """
#     return d2Y_dx2 + 1/5*dY_dx + Y + exp(-x/5)*cos(x)


# def dG_dY(x, Y, dY_dx):
#     """Derivative of G(x, Y, dY_dx) wrt Y"""
#     return 1/5


# def dG_ddYdx(x, y, dY_dx):
#     """Derivative of G(x, Y, dY_dx) wrt dY/dx"""
#     return 1/5


# def dG_dd2Ydx2(x, y, dY_dx):
#     """Derivative of G(x, Y, dY_dx) wrt d2Y/dx2"""
#     return 1


def Y1a(x):
    """Analytical solution"""
    return sin(x)


def dY1a_dx(x):
    """1st derivative of analytical solution"""
    return cos(x)


def Y2a(x):
    """Analytical solution"""
    return 1 + x**2


def dY2a_dx(x):
    """1st derivative of analytical solution"""
    return 2*x
