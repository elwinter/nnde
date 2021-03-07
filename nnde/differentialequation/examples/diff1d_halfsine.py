"""
This module implements a 1-D diffusion PDE

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x,t).

The equation is defined on the domain (x,t) in [[0,1],[0,]].

The analytical form of the equation is:

  G(x, Y, delY, del2Y) = dY_dt - D*d2Y_dx2 = 0

where:

xv is the vector (x,t)
delY is the vector (dY/dx, dY/dt)
del2Y is the vector (d2Y/dx2, d2Y/dt2)

With boundary conditions:

Y(0, t) = 0
Y(1, t) = 0
Y(x, 0) = sin(pi*x)

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, t) = exp(-pi**2*D*t)*sin(pi*x)

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import exp, cos, pi, sin
import numpy as np


# Diffusion coefficient
D = 0.1


def G(xt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return dY_dt - D*d2Y_dx2


def f0(xt):
    """Boundary condition at (x,t) = (0,t)"""
    return 0


def f1(xt):
    """Boundary condition at (x,t) = (1,t)"""
    return 0


def Y0(xt):
    """Boundary condition at (x,t) = (x,0)"""
    (x, t) = xt
    return sin(pi*x)


bc = [[f0, f1], [Y0, None]]


def df0_dx(xt):
    """1st derivative of BC wrt x at (x,t) = (0,t)"""
    return 0


def df0_dt(xt):
    """1st derivative of BC wrt t at (x,t) = (0,t)"""
    return 0


def df1_dx(xt):
    """1st derivative of BC wrt x at (x,t) = (1,t)"""
    return 0


def df1_dt(xt):
    """1st derivative of BC wrt t at (x,t) = (1,t)"""
    return 0


def dY0_dx(xt):
    """1st derivative of BC wrt x at (x,t) = (x,0)"""
    (x, t) = xt
    return pi*cos(pi*x)


def dY0_dt(xt):
    """1st derivative of BC wrt t at (x,t) = (x,0)"""
    return 0


delbc = [[[df0_dx, df0_dt], [df1_dx, df1_dt]],
         [[dY0_dx, dY0_dt], [None, None]]]


def d2f0_dx2(xt):
    """2nd derivative of BC wrt x at (x,t) = (0,t)"""
    return 0


def d2f0_dt2(xt):
    """2nd derivative of BC wrt t at (x,t) = (0,t)"""
    return 0


def d2f1_dx2(xt):
    """2nd derivative of BC wrt x at (x,t) = (1,t)"""
    return 0


def d2f1_dt2(xt):
    """2nd derivative of BC wrt t at (x,t) = (1,t)"""
    return 0


def d2Y0_dx2(xt):
    """2nd derivative of BC wrt x at (x,t) = (x,0)"""
    (x, t) = xt
    return -pi**2*sin(pi*x)


def d2Y0_dt2(xt):
    """2nd derivative of BC wrt t at (x,t) = (x,0)"""
    return 0


del2bc = [[[d2f0_dx2, d2f0_dt2], [d2f1_dx2, d2f1_dt2]],
          [[d2Y0_dx2, d2Y0_dt2], [None, None]]]


def dG_dY(xt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    return 0


def dG_dY_dx(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    return 0


def dG_dY_dt(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    return 1


dG_ddelY = [dG_dY_dx, dG_dY_dt]


def dG_d2Y_dx2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    return -D


def dG_d2Y_dt2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    return 0


dG_ddel2Y = [dG_d2Y_dx2, dG_d2Y_dt2]


def A(xt):
    """Optimized version of boundary condition function"""
    (x, t) = xt
    A = (1 - t)*sin(pi*x)
    return A


def delA(xt):
    """Optimized version of boundary condition function gradient"""
    (x, t) = xt
    dA_dx = pi*(1 - t)*cos(pi*x)
    dA_dt = -sin(pi*x)
    delA = [dA_dx, dA_dt]
    return delA


def del2A(xt):
    """Optimized version of boundary condition function Laplacian"""
    (x, t) = xt
    d2A_dx2 = -pi**2*(1 - t)*sin(pi*x)
    d2A_dt2 = 0
    del2A = [d2A_dx2, d2A_dt2]
    return del2A


def Ya(xt):
    """Analytical solution"""
    (x, t) = xt
    return exp(-pi**2*D*t)*sin(pi*x)


def dYa_dx(xt):
    """Analytical x-gradient"""
    (x, t) = xt
    return pi*exp(-pi**2*D*t)*cos(pi*x)


def dYa_dt(xt):
    """Analytical t-gradient"""
    (x, t) = xt
    return -pi**2*D*exp(-pi**2*D*t)*sin(pi*x)


delYa = [dYa_dx, dYa_dt]


def d2Ya_dx2(xt):
    """Analytical x-Laplacian"""
    (x, t) = xt
    return -pi**2*exp(-pi**2*D*t)*sin(pi*x)


def d2Ya_dt2(xt):
    """Analytical t-Laplacian"""
    (x, t) = xt
    return (-pi**2*D)**2*exp(-pi**2*D*t)*sin(pi*x)


del2Ya = [d2Ya_dx2, d2Ya_dt2]
