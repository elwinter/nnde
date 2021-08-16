"""The 1-D diffusion equation.

This module implements a 12-D diffusion PDE.

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x, y, t). Y(x, y, t) is normalized to the
range [0, 1], and is treated as unitless.

The equation is defined on the domain:

  0 <= x <= 1
  0 <= y <= 1
  0 <= t

The analytical form of the equation is:

  G(xyt, Y, delY, del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2) = 0

where:

xyt is the vector (x, y, t).
Y is the solution to be found.
delY is the gradient vector (dY/dx, dY/dy, dY/dt).
del2Y is the Laplacian component vector (d2Y/dx2, d2Y/dy2, d2Y/dt2).

The boundary conditions are:

Y(0, y, t) = 0
Y(1, y, t) = 0
Y(x, 0, t) = 0
Y(x, 1, t) = 0
Y(x, y, 0) = sin(pi*x)*sin(pi*y)

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, y, t) = exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


from math import cos, exp, pi, sin


# Diffusion coefficient (L**2/T, where L is a length, and T is a time).
D = 0.1


def G(xyt, Y, delY, del2Y):
    """Compute the differential equation in standard form.

    Compute the value of the differential equation in standard form. For a
    perfect solution, the value should be 0.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of the differential equation.
    """
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2)


def f0(xyt):
    """Boundary condition at (x, y, t) = (0, y, t).

    Compute the value of the solution at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, t) = (0, y, t).
    """
    return 0


def f1(xyt):
    """Boundary condition at (x, y, t) = (1, y, t).

    Compute the value of the solution at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, t) = (1, y, t).
    """
    return 0


def g0(xyt):
    """Boundary condition at (x, y, t) = (x, 0, t).

    Compute the value of the solution at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, t) = (x, 0, t).
    """
    return 0


def g1(xyt):
    """Boundary condition at (x, y, t) = (x, 1, t).

    Compute the value of the solution at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, t) = (x, 1, t).
    """
    return 0


def Y0(xyt):
    """Boundary condition at (x, y, t) = (x, y, 0).

    Compute the value of the solution at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, t) = (x, y, 0).
    """
    (x, y, t) = xyt
    return sin(pi*x)*sin(pi*y)


# Gather the boundary condition functions in a single array.
bc = [[f0, f1], [g0, g1], [Y0, None]]


def df0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (0, y, t).

    Compute the 1st derivative of BC wrt x at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (0, y, t).
    """
    return 0


def df0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (0, y, t).

    Compute the 1st derivative of BC wrt y at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, t) = (0, y, t).
    """
    return 0


def df0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (0, y, t).

    Compute the 1st derivative of BC wrt t at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, t) = (0, y, t).
    """
    return 0


def df1_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (1, y, t).

    Compute the 1st derivative of BC wrt x at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (1, y, t).
    """
    return 0


def df1_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (1, y, t).

    Compute the 1st derivative of BC wrt y at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, t) = (1, y, t).
    """
    return 0


def df1_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (1, y, t).

    Compute the 1st derivative of BC wrt t at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, t) = (1, y, t).
    """
    return 0


def dg0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, 0, t).

    Compute the 1st derivative of BC wrt x at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (x, 0, t).
    """
    return 0


def dg0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, 0, t).

    Compute the 1st derivative of BC wrt y at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, t) = (x, 0, t).
    """
    return 0


def dg0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, 0, t).

    Compute the 1st derivative of BC wrt t at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (t, 0, t).
    """
    return 0


def dg1_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, 1, t).

    Compute the 1st derivative of BC wrt x at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (x, 1, t).
    """
    return 0


def dg1_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, 1, t).

    Compute the 1st derivative of BC wrt y at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, t) = (x, 1, t).
    """
    return 0


def dg1_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, 1, t).

    Compute the 1st derivative of BC wrt t at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, t) = (x, 1, t).
    """
    return 0


def dY0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, y, 0).

    Compute the 1st derivative of BC wrt x at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, t) = (x, y, 0).
    """
    (x, y, t) = xyt
    return pi*cos(pi*x)*sin(pi*y)


def dY0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, y, 0).

    Compute the 1st derivative of BC wrt y at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, t) = (x, y, 0).
    """
    (x, y, t) = xyt
    return pi*sin(pi*x)*cos(pi*y)


def dY0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, y, 0).

    Compute the 1st derivative of BC wrt t at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, t) = (x, y, 0).
    """
    return 0


# Gather the gradient functions into a single array.
delbc = [[[df0_dx, df0_dy, df0_dt], [df1_dx, df1_dy, df1_dt]],
         [[dg0_dx, dg0_dy, dg0_dt], [dg1_dx, dg1_dy, dg1_dt]],
         [[dY0_dx, dY0_dy, dY0_dt], [None, None, None]]]


def d2f0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (0, y, t).

    Compute the 2nd derivative of BC wrt x at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, t) = (0, y, t).
    """
    return 0


def d2f0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (0, y, t).

    Compute the 2nd derivative of BC wrt y at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, t) = (0, y, t).
    """
    return 0


def d2f0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (0, y, t).

    Compute the 2nd derivative of BC wrt t at (x, y, t) = (0, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, t) = (0, y, t).
    """
    return 0


def d2f1_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (1, y, t).

    Compute the 2nd derivative of BC wrt x at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, t) = (1, y, t).
    """
    return 0


def d2f1_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (1, y, t).

    Compute the 2nd derivative of BC wrt y at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, t) = (1, y, t).
    """
    return 0


def d2f1_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (1, y, t).

    Compute the 2nd derivative of BC wrt t at (x, y, t) = (1, y, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, t) = (1, y, t).
    """
    return 0


def d2g0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, 0, t).

    Compute the 2nd derivative of BC wrt x at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, t) = (x, 0, t).
    """
    return 0


def d2g0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, 0, t).

    Compute the 2nd derivative of BC wrt y at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, t) = (x, 0, t).
    """
    return 0


def d2g0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, 0, t).

    Compute the 2nd derivative of BC wrt t at (x, y, t) = (x, 0, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, t) = (x, 0, t).
    """
    return 0


def d2g1_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, 1, t).

    Compute the 2nd derivative of BC wrt x at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, t) = (x, 1, t).
    """
    return 0


def d2g1_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, 1, t).

    Compute the 2nd derivative of BC wrt y at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, t) = (x, 1, t).
    """
    return 0


def d2g1_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, 1, t).

    Compute the 2nd derivative of BC wrt t at (x, y, t) = (x, 1, t).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, t) = (x, 1, t).
    """
    return 0


def d2Y0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, y, 0).

    Compute the 2nd derivative of BC wrt x at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, t) = (x, y, 0).
    """
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)


def d2Y0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, y, 0).

    Compute the 2nd derivative of BC wrt y at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, t) = (x, y, 0).
    """
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)


def d2Y0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, y, 0).

    Compute the 2nd derivative of BC wrt t at (x, y, t) = (x, y, 0).

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, t) = (x, y, 0).
    """
    return 0


# Gather the functions for the Laplacian components into a single array.
del2bc = [[[d2f0_dx2, d2f0_dy2, d2f0_dt2], [d2f1_dx2, d2f1_dy2, d2f1_dt2]],
          [[d2g0_dx2, d2g0_dy2, d2g0_dt2], [d2g1_dx2, d2g1_dy2, d2g1_dt2]],
          [[d2Y0_dx2, d2Y0_dy2, d2Y0_dt2], [None, None, None]]]


def dG_dY(xyt, Y, delY, del2Y):
    """1st derivative of G wrt Y.

    Compute the 1st derivative of G wrt Y.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/dY.
    """
    return 0


def dG_ddY_dx(xyt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dx.

    Compute the 1st derivative of G wrt dY/dx.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dx).
    """
    return 0


def dG_ddY_dy(xyt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dy.

    Compute the 1st derivative of G wrt dY/dy.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dy).
    """
    return 0


def dG_ddY_dt(xyt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dt.

    Compute the 1st derivative of G wrt dY/dt.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dt).
    """
    return 1


# Gather the derivatives into a single array.
dG_ddelY = [dG_ddY_dx, dG_ddY_dy, dG_ddY_dt]


def dG_dd2Y_dx2(xyt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dx2.

    Compute the 1st derivative of G wrt d2Y/dx2.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dx2).
    """
    return -D


def dG_dd2Y_dy2(xyt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dy2.

    Compute the 1st derivative of G wrt d2Y/dy2.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dy2).
    """
    return -D


def dG_dd2Y_dt2(xyt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dt2.

    Compute the 1st derivative of G wrt d2Y/dt2.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 3 float
        Values for dY/dx, dY/dy, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dt2).
    """
    return 0


# Gather the derivatives into a single array.
dG_ddel2Y = [dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dt2]


def A(xyt):
    """Optimized version of the boundary condition function.

    Compute the optimized version of the boundary condition function.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function.
    """
    (x, y, t) = xyt
    A = (1 - t)*sin(pi*x)*sin(pi*y)
    return A


def delA(xyt):
    """Gradient of optimized version of the boundary condition function.

    Compute the gradient of the optimized version of the boundary condition
    function.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function gradient.
    """
    (x, y, t) = xyt
    dA_dx = pi*(1 - t)*cos(pi*x)*sin(pi*y)
    dA_dy = pi*(1 - t)*sin(pi*x)*cos(pi*y)
    dA_dt = -sin(pi*x)*sin(pi*y)
    delA = [dA_dx, dA_dy, dA_dt]
    return delA


def del2A(xyt):
    """Laplacian components of optimized boundary condition function.

    Compute the Laplacian components of the optimized the boundary condition
    function.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function Laplacian components.
    """
    (x, y, t) = xyt
    d2A_dx2 = -pi**2*(1 - t)*sin(pi*x)*sin(pi*y)
    d2A_dy2 = -pi**2*(1 - t)*sin(pi*x)*sin(pi*y)
    d2A_dt2 = 0
    del2A = [d2A_dx2, d2A_dy2,  d2A_dt2]
    return del2A


def Ya(xyt):
    """Analytical solution of G.

    Compute the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the analytical solution.
    """
    (x, y, t) = xyt
    return exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)


def dYa_dx(xyt):
    """1st x-derivative of the analytical solution of G.

    Compute the 1st x-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st x-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return pi*exp(-2*pi**2*D*t)*cos(pi*x)*sin(pi*y)


def dYa_dy(xyt):
    """1st y-derivative of the analytical solution of G.

    Compute the 1st y-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st y-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return pi*exp(-2*pi**2*D*t)*sin(pi*x)*cos(pi*y)


def dYa_dt(xyt):
    """1st t-derivative of the analytical solution of G.

    Compute the 1st t-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st t-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return -2*pi**2*D*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)


# Gather the analytical gradient functions into a single array.
delYa = [dYa_dx, dYa_dy, dYa_dt]


def d2Ya_dx2(xyt):
    """2nd x-derivative of the analytical solution of G.

    Compute the 2nd x-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd x-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return -pi**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)


def d2Ya_dy2(xyt):
    """2nd y-derivative of the analytical solution of G.

    Compute the 2nd y-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd x-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return -pi**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)


def d2Ya_dt2(xyt):
    """2nd t-derivative of the analytical solution of G.

    Compute the 2nd t-derivative of the analytical solution of G.

    Parameters
    ----------
    xyt : array-like of 3 float
        Values for x, y, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd t-derivative of the analytical solution of G.
    """
    (x, y, t) = xyt
    return 4*pi**4*D**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)


# Gather the Laplacian component functions into a single array.
del2Ya = [d2Ya_dx2, d2Ya_dy2, d2Ya_dt2]
