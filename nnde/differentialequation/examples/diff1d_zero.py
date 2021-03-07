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

Y(0, t) = C = 0
Y(1, t) = C = 0
Y(x, 0) = C = 0

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, t) = 0

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


import numpy as np


# Diffusion coefficient
D = 0.1

# Constant value of profile
C = 0


def G(xt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return dY_dt - D*d2Y_dx2


def f0(xt):
    """Boundary condition at (x,t) = (0,t)"""
    return C


def f1(xt):
    """Boundary condition at (x,t) = (1,t)"""
    return C


def Y0(xt):
    """Boundary condition at (x,t) = (x,0)"""
    return C


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
    return 0


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
    return 0


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
    return C


def delA(xt):
    """Optimized version of boundary condition function gradient"""
    return [0, 0]


def del2A(xt):
    """Optimized version of boundary condition function Laplacian"""
    return [0, 0]


def Ya(xt):
    """Analytical solution"""
    return C


def dYa_dx(xt):
    """Analytical x-gradient"""
    return 0


def dYa_dt(xt):
    """Analytical t-gradient"""
    return 0


delYa = [dYa_dx, dYa_dt]


def d2Ya_dx2(xt):
    """Analytical x-Laplacian"""
    return 0


def d2Ya_dt2(xt):
    """Analytical t-Laplacian"""
    return 0


del2Ya = [d2Ya_dx2, d2Ya_dt2]
