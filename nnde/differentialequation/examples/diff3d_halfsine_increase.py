"""The 3-D diffusion equation.

This module implements a 3-D diffusion PDE.

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x, y, t). Y(x, y, t) is normalized to the
range [0, 1], and is treated as unitless.

The equation is defined on the domain:

  0 <= x <= 1
  0 <= y <= 1
  0 <= z <= 1
  0 <= t

The analytical form of the equation is:

  G(xyzt, Y, delY, del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2) = 0

where:

xyzt is the vector (x, y, z, t).
Y is the solution to be found.
delY is the gradient vector (dY/dx, dY/dy, dY/dz, dY/dt).
del2Y is the Laplacian component vector (d2Y/dx2, d2Y/dy2, d2Y/dz2, d2Y/dt2).

The boundary conditions are:

Y(0, y, z, t) = 0
Y(1, y, z, t) = a*t*sin(pi*y)*sin(pi*z)
Y(x, 0, z, t) = 0
Y(x, 1, z, t) = 0
Y(x, y, 0, t) = 0
Y(x, y, 1, t) = 0
Y(x, y, z, 0) = sin(pi*x)*sin(pi*y)*sin(pi*z)

This equation has no analytical solution for the supplied initial
conditions.

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


from math import cos, pi, sin


# Diffusion coefficient (L**2/T, where L is a length, and T is a time).
D = 0.1

# Boundary increase rate at x=1 (1/T).
a = 1.0


def G(xyzt, Y, delY, del2Y):
    """Compute the differential equation in standard form.

    Compute the value of the differential equation in standard form. For a
    perfect solution, the value should be 0.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of the differential equation.
    """
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2)


def f0(xyzt):
    """Boundary condition at (x, y, z, t) = (0, y, z, t).

    Compute the value of the solution at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def f1(xyzt):
    """Boundary condition at (x, y, z, t) = (1, y, z, t).

    Compute the value of the solution at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return a*t*sin(pi*y)*sin(pi*z)


def g0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, 0, z, t).

    Compute the value of the solution at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def g1(xyzt):
    """Boundary condition at (x, y, z, t) = (x, 1, z, t).

    Compute the value of the solution at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def h0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, 0, t).

    Compute the value of the solution at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def h1(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, 1, t).

    Compute the value of the solution at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def Y0(xyzt):
    """Boundary condition at (x, y, z, t) = (x, y, z, 0).

    Compute the value of the solution at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return sin(pi*x)*sin(pi*y)*sin(pi*z)


# Gather the boundary condition functions in a single array.
bc = [[f0, f1], [g0, g1], [h0, h1], [Y0, None]]


def df0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def df0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def df0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def df0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def df1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).
    """
    return 0


def df1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return a*pi*t*cos(pi*y)*sin(pi*z)


def df1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return a*pi*t*sin(pi*y)*cos(pi*z)


def df1_dt(xyzt):
    """1st derivative of BC wrt xt at (x, y, z, t) = (1, y, z, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return a*sin(pi*y)*sin(pi*z)


def dg0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def dg0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def dg0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def dg0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def dg1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def dg1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def dg1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def dg1_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def dh0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def dh0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def dh0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def dh0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def dh1_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def dh1_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def dh1_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def dh1_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def dY0_dx(xyzt):
    """1st derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).

    Compute the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return pi*cos(pi*x)*sin(pi*y)*sin(pi*z)


def dY0_dy(xyzt):
    """1st derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).

    Compute the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*cos(pi*y)*sin(pi*z)


def dY0_dz(xyzt):
    """1st derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).

    Compute the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*sin(pi*y)*cos(pi*z)


def dY0_dt(xyzt):
    """1st derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).

    Compute the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).
    """
    return 0


# Gather the gradient functions into a single array.
delbc = [[[df0_dx, df0_dy, df0_dz, df0_dt], [df1_dx, df1_dy, df1_dz, df1_dt]],
         [[dg0_dx, dg0_dy, dg0_dz, dg0_dt], [dg1_dx, dg1_dy, dg1_dz, dg1_dt]],
         [[dh0_dx, dh0_dy, dh0_dz, dh0_dt], [dh1_dx, dh1_dy, dh1_dz, dh1_dt]],
         [[dY0_dx, dY0_dy, dY0_dz, dY0_dt], [None, None, None, None]]]


def d2f0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def d2f0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def d2f0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def d2f0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (0, y, z, t).
    """
    return 0


def d2f1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (1, y, z, t).
    """
    return 0


def d2f1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)


def d2f1_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (1, y, z, t).
    """
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)


def d2f1_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (1, y, z, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (1, y, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (1, y, z, t).
    """
    return 0


def d2g0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def d2g0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def d2g0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def d2g0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (x, 0, z, t).
    """
    return 0


def d2g1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def d2g1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def d2g1_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def d2g1_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (x, 1, z, t).
    """
    return 0


def d2h0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def d2h0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def d2h0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def d2h0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 0, t).
    """
    return 0


def d2h1_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def d2h1_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def d2h1_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def d2h1_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, 1, t).
    """
    return 0


def d2Y0_dx2(xyzt):
    """2nd derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).

    Compute the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt x at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dy2(xyzt):
    """2nd derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).

    Compute the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt y at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dz2(xyzt):
    """2nd derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).

    Compute the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt z at (x, y, z, t) = (x, y, z, 0).
    """
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)


def d2Y0_dt2(xyzt):
    """2nd derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).

    Compute the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).

    Parameters
    ----------
    xyt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd derivative of BC wrt t at (x, y, z, t) = (x, y, z, 0).
    """
    return 0


# Gather the functions for the Laplacian components into a single array.
del2bc = [[[d2f0_dx2, d2f0_dy2, d2f0_dz2, d2f0_dt2],
           [d2f1_dx2, d2f1_dy2, d2f1_dz2, d2f1_dt2]],
          [[d2g0_dx2, d2g0_dy2, d2g0_dz2, d2g0_dt2],
           [d2g1_dx2, d2g1_dy2, d2g1_dz2, d2g1_dt2]],
          [[d2h0_dx2, d2h0_dy2, d2h0_dz2, d2h0_dt2],
           [d2h1_dx2, d2h1_dy2, d2h1_dz2, d2h1_dt2]],
          [[d2Y0_dx2, d2Y0_dy2, d2Y0_dz2, d2Y0_dt2],
           [None, None, None, None]]]


def dG_dY(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt Y.

    Compute the 1st derivative of G wrt Y.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 4 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/dY.
    """
    return 0


def dG_ddY_dx(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dx.

    Compute the 1st derivative of G wrt dY/dx.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 4 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dx).
    """
    return 0


def dG_ddY_dy(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dy.

    Compute the 1st derivative of G wrt dY/dy.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 4 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dy).
    """
    return 0


def dG_ddY_dz(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dz.

    Compute the 1st derivative of G wrt dY/dz.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 4 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dz).
    """
    return 0


def dG_ddY_dt(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dt.

    Compute the 1st derivative of G wrt dY/dt.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 4 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dt).
    """
    return 1


# Gather the derivatives into a single array.
dG_ddelY = [dG_ddY_dx, dG_ddY_dy, dG_ddY_dz, dG_ddY_dt]


def dG_dd2Y_dx2(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dx2.

    Compute the 1st derivative of G wrt d2Y/dx2.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dx2).
    """
    return -D


def dG_dd2Y_dy2(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dy2.

    Compute the 1st derivative of G wrt d2Y/dy2.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dy2).
    """
    return -D


def dG_dd2Y_dz2(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dz2.

    Compute the 1st derivative of G wrt d2Y/dz2.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dz2).
    """
    return -D


def dG_dd2Y_dt2(xyzt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dt2.

    Compute the 1st derivative of G wrt d2Y/dt2.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 4 float
        Values for dY/dx, dY/dy, dY/dz, and dY/dt, in that order.
    del2Y : array-like of 3 float
        Values for d2Y/dx2, d2Y/dy2, d2Y/dz2, and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dt2).
    """
    return 0


# Gather the derivatives into a single array.
dG_ddel2Y = [dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dz2, dG_dd2Y_dt2]


def A(xyzt):
    """Optimized version of the boundary condition function.

    Compute the optimized version of the boundary condition function.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function.
    """
    (x, y, z, t) = xyzt
    A = (a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    return A


def delA(xyzt):
    """Gradient of optimized version of the boundary condition function.

    Compute the gradient of the optimized version of the boundary condition
    function.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function gradient.
    """
    (x, y, z, t) = xyzt
    dA_dx = (a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)*sin(pi*z)
    dA_dy = pi*cos(pi*y)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*z)
    dA_dz = pi*cos(pi*z)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)
    dA_dt = (a*x - sin(pi*x))*sin(pi*y)*sin(pi*z)
    return [dA_dx, dA_dy, dA_dz, dA_dt]


def del2A(xyzt):
    """Laplacian components of optimized boundary condition function.

    Compute the Laplacian components of the optimized the boundary condition
    function.

    Parameters
    ----------
    xyzt : array-like of 4 float
        Values for x, y, z, and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function Laplacian components.
    """
    (x, y, z, t) = xyzt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)*sin(pi*y)*sin(pi*z)
    d2A_dy2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dz2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]
