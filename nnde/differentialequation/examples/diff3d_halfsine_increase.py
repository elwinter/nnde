"""
This module implements a 3-D diffusion PDE

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x, y, z, t).

The equation is defined on the domain (x, y, z, t) in
[[0, 1], [0, 1], [0, 1], [0, inf]].

The analytical form of the equation is:

  G([x, y, z, t], Y, delY, del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2) = 0

where:

[x, y, z, t] are the independent variables
delY is the vector (dY/dx, dY/dy, dY/dz, dY/dt)
del2Y is the vector (d2Y/dx2, d2Y/dy2, d2Y/dz2, d2Y/dt2)

With boundary conditions (note the BC are continuous at domain corners):

Y(0, y, z, t) = 0
Y(1, y, z, t) = a*t*sin(pi*y)*sin(pi*z)
Y(x, 0, z, t) = 0
Y(x, 1, z, t) = 0
Y(x, y, 0, t) = 0
Y(x, y, 1, t) = 0
Y(x, y, z, 0) = sin(pi*x)*sin(pi*y)*sin(pi*z)

This equation has no analytical solution for the supplied initial
conditions.

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import cos, pi, sin
import numpy as np


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 1.0


def G(xyzt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2)


def f0(xyzt):
    """Boundary condition at (x, y, z, t) = (0, y, z, t)"""
    return 0


def f1(xyzt):
    """Boundary condition at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return a*t*sin(pi*y)*sin(pi*z)


def g0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def g1(xyzt):
    """Boundary condition at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def h0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def h1(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def Y0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return sin(pi*x)*sin(pi*y)*sin(pi*z)


bc = [[f0, f1], [g0, g1], [h0, h1], [Y0, None]]


def df0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (0, y, z, t)"""
    return 0


def df0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (0, y, z, t)"""
    return 0


def df0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (0, y, z, t)"""
    return 0


def df0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (0, y, z, t)"""
    return 0


def df1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (1, y, z, t)"""
    return 0


def df1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return a*pi*t*cos(pi*y)*sin(pi*z)


def df1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return a*pi*t*sin(pi*y)*cos(pi*z)


def df1_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return a*sin(pi*y)*sin(pi*z)


def dg0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def dg0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def dg0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def dg0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def dg1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def dg1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def dg1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def dg1_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def dh0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def dh0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def dh0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def dh0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def dh1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def dh1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def dh1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def dh1_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def dY0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return pi*cos(pi*x)*sin(pi*y)*sin(pi*z)


def dY0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*cos(pi*y)*sin(pi*z)


def dY0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*sin(pi*y)*cos(pi*z)


def dY0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0)"""
    return 0


delbc = [[[df0_dx, df0_dy, df0_dz, df0_dt], [df1_dx, df1_dy, df1_dz, df1_dt]],
         [[dg0_dx, dg0_dy, dg0_dz, dg0_dt], [dg1_dx, dg1_dy, dg1_dz, dg1_dt]],
         [[dh0_dx, dh0_dy, dh0_dz, dh0_dt], [dh1_dx, dh1_dy, dh1_dz, dh1_dt]],
         [[dY0_dx, dY0_dy, dY0_dz, dY0_dt], [None, None, None, None]]]


def d2f0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (0, y, z, t)"""
    return 0


def d2f0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (0, y, z, t)"""
    return 0


def d2f0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (0, y, z, t)"""
    return 0


def d2f0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (0, y, z, t)"""
    return 0


def d2f1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (1, y, z, t)"""
    return 0


def d2f1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)


def d2f1_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (1, y, z, t)"""
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)


def d2f1_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (1, y, z, t)"""
    return 0


def d2g0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def d2g0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def d2g0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def d2g0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t)"""
    return 0


def d2g1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def d2g1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def d2g1_dz2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def d2g1_dt2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t)"""
    return 0


def d2h0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def d2h0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def d2h0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def d2h0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t)"""
    return 0


def d2h1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def d2h1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def d2h1_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def d2h1_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t)"""
    return 0


def d2Y0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dt2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0)"""
    return 0


del2bc = [[[d2f0_dx2, d2f0_dy2, d2f0_dz2, d2f0_dt2],
           [d2f1_dx2, d2f1_dy2, d2f1_dz2, d2f1_dt2]],
          [[d2g0_dx2, d2g0_dy2, d2g0_dz2, d2g0_dt2],
           [d2g1_dx2, d2g1_dy2, d2g1_dz2, d2g1_dt2]],
          [[d2h0_dx2, d2h0_dy2, d2h0_dz2, d2h0_dt2],
           [d2h1_dx2, d2h1_dy2, d2h1_dz2, d2h1_dt2]],
          [[d2Y0_dx2, d2Y0_dy2, d2Y0_dz2, d2Y0_dt2],
           [None, None, None, None]]]


def dG_dY(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    return 0


def dG_ddY_dx(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    return 0


def dG_ddY_dy(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dy"""
    return 0


def dG_ddY_dz(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dz"""
    return 0


def dG_ddY_dt(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    return 1


dG_ddelY = [dG_ddY_dx, dG_ddY_dy, dG_ddY_dz, dG_ddY_dt]


def dG_dd2Y_dx2(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    return -D


def dG_dd2Y_dy2(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dy2"""
    return -D


def dG_dd2Y_dz2(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dz2"""
    return -D


def dG_dd2Y_dt2(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    return 0


dG_ddel2Y = [dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dz2, dG_dd2Y_dt2]


def A(xyzt):
    """Optimized version of boundary condition function"""
    (x, y, z, t) = xyzt
    A = (a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    return A


def delA(xyzt):
    """Optimized version of boundary condition function gradient"""
    (x, y, z, t) = xyzt
    dA_dx = (a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)*sin(pi*z)
    dA_dy = pi*cos(pi*y)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*z)
    dA_dz = pi*cos(pi*z)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)
    dA_dt = (a*x - sin(pi*x))*sin(pi*y)*sin(pi*z)
    return [dA_dx, dA_dy, dA_dz, dA_dt]


def del2A(xyzt):
    """Optimized version of boundary condition function Laplacian"""
    (x, y, z, t) = xyzt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)*sin(pi*y)*sin(pi*z)
    d2A_dy2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dz2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]
