"""The 1-D diffusion equation.

This module implements a 1-D diffusion PDE.

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x,t). Y(x, t) is normalized to the
range [0, 1], and is treated as unitless.

The equation is defined on the domain:

  0 <= x <= 1
  0 <= t

The analytical form of the equation is:

  G(xt, Y, delY, del2Y) = dY_dt - D*d2Y_dx2 = 0

where:

xt is the vector (x, t).
Y is the solution to be found.
delY is the gradient vector (dY/dx, dY/dt)->(dY_dx, dY_dt).
del2Y is the Laplacian component vector (d2Y/dx2, d2Y/dt2)->
(d2Y_dx2, d2Y_dt2).

The boundary conditions are:

Y(0, t) = 0
Y(1, t) = 0
Y(x, 0) = sin(pi*x)

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, t) = exp(-pi**2*D*t)*sin(pi*x)

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


from math import exp, cos, pi, sin


# Diffusion coefficient (L**2/T, where L is a length, and T is a time).
D = 0.1


def G(xt, Y, delY, del2Y):
    """Compute the differential equation in standard form.

    Compute the value of the differential equation in standard form. For a
    perfect solution, the value should be 0.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of the differential equation.
    """
    dY_dt = delY[1]
    d2Y_dx2 = del2Y[0]
    return dY_dt - D*d2Y_dx2


def f0(xt):
    """Boundary condition at (x, t) = (0, t).

    Compute the value of the solution at (x, t) = (0, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, t) = (0, t).
    """
    return 0


def f1(xt):
    """Boundary condition at (x, t) = (1, t).

    Compute the value of the solution at (x, t) = (1, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, t) = (1, t).
    """
    return 0


def Y0(xt):
    """Boundary condition at (x, t) = (x, 0).

    Compute the value of the solution at (x, t) = (x, 0).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, t) = (x, 0).
    """
    (x, t) = xt
    return sin(pi*x)


# Gather the boundary condition functions in a single array.
bc = [[f0, f1], [Y0, None]]


def df0_dx(xt):
    """1st derivative of BC wrt x at (x, t) = (0, t).

    Compute the 1st derivative of BC wrt x at (x, t) = (0, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, t) = (0, t).
    """
    return 0


def df0_dt(xt):
    """1st derivative of BC wrt t at (x, t) = (0, t).

    Compute the 1st derivative of BC wrt t at (x, t) = (0, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, t) = (0, t).
    """
    return 0


def df1_dx(xt):
    """1st derivative of BC wrt x at (x, t) = (1, t).

    Compute the 1st derivative of BC wrt x at (x, t) = (1, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, t) = (1, t).
    """
    return 0


def df1_dt(xt):
    """1st derivative of BC wrt t at (x, t) = (1, t).

    Compute the 1st derivative of BC wrt t at (x, t) = (1, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, t) = (1, t).
    """
    return 0


def dY0_dx(xt):
    """1st derivative of BC wrt x at (x, t) = (x, 0).

    Compute the 1st derivative of BC wrt x at (x, t) = (x, 0).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt x at (x, t) = (x, 0).
    """
    (x, t) = xt
    return pi*cos(pi*x)


def dY0_dt(xt):
    """1st derivative of BC wrt t at (x, t) = (x, 0).

    Compute the 1st derivative of BC wrt t at (x, t) = (x, 0).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st derivative of BC wrt t at (x, t) = (x, 0).
    """
    return 0


# Gather the gradient functions into a single array.
delbc = [[[df0_dx, df0_dt], [df1_dx, df1_dt]],
         [[dY0_dx, dY0_dt], [None, None]]]


def d2f0_dx2(xt):
    """2nd derivative of BC wrt x at (x, t) = (0, t).

    Compute the 2nd derivative of BC wrt x at (x, t) = (0, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt x at (x, t) = (0, t).
    """
    return 0


def d2f0_dt2(xt):
    """2nd derivative of BC wrt t at (x, t) = (0, t).

    Compute the 2nd derivative of BC wrt t at (x, t) = (0, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt t at (x, t) = (0, t).
    """
    return 0


def d2f1_dx2(xt):
    """2nd derivative of BC wrt x at (x, t) = (1, t).

    Compute the 2nd derivative of BC wrt x at (x, t) = (1, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt x at (x, t) = (1, t).
    """
    return 0


def d2f1_dt2(xt):
    """2nd derivative of BC wrt t at (x, t) = (1, t).

    Compute the 2nd derivative of BC wrt t at (x, t) = (1, t).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt t at (x, t) = (1, t).
    """
    return 0


def d2Y0_dx2(xt):
    """2nd derivative of BC wrt x at (x, t) = (x, 0).

    Compute the 2nd derivative of BC wrt x at (x, t) = (x, 0).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt x at (x, t) = (x, 0).
    """
    (x, t) = xt
    return -pi**2*sin(pi*x)


def d2Y0_dt2(xt):
    """2nd derivative of BC wrt t at (x, t) = (x, 0).

    Compute the 2nd derivative of BC wrt t at (x, t) = (x, 0).

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2ns derivative of BC wrt x at (x, t) = (t, 0).
    """
    return 0


# Gather the functions for the Laplacian components into a single array.
del2bc = [[[d2f0_dx2, d2f0_dt2], [d2f1_dx2, d2f1_dt2]],
          [[d2Y0_dx2, d2Y0_dt2], [None, None]]]


def dG_dY(xt, Y, delY, del2Y):
    """1st derivative of G wrt Y.

    Compute the 1st derivative of G wrt Y.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/dY.
    """
    return 0


# N.B. Should be named dG_ddY_dx.
def dG_dY_dx(xt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dx.

    Compute the 1st derivative of G wrt dY/dx.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dx).
    """
    return 0


# N.B. Should be named dG_ddY_dt.
def dG_dY_dt(xt, Y, delY, del2Y):
    """1st derivative of G wrt dY/dt.

    Compute the 1st derivative of G wrt dY/dt.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(dY/dt).
    """
    return 1


# Gather the derivatives into a single array.
dG_ddelY = [dG_dY_dx, dG_dY_dt]


# N.B. Should be named dG_dd2Y_dx2.
def dG_d2Y_dx2(xt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dx2.

    Compute the 1st derivative of G wrt d2Y/dx2.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dx2).
    """
    return -D


# N.B. Should be named dG_dd2Y_dt2.
def dG_d2Y_dt2(xt, Y, delY, del2Y):
    """1st derivative of G wrt d2Y/dt2.

    Compute the 1st derivative of G wrt d2Y/dt2.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dt, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dt2, in that order.

    Returns
    -------
    result : float
        Value of dG/d(d2Y/dt2).
    """
    return 0


# Gather the derivatives into a single array.
dG_ddel2Y = [dG_d2Y_dx2, dG_d2Y_dt2]


def A(xt):
    """Optimized version of the boundary condition function.

    Compute the optimized version of the boundary condition function.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function.
    """
    (x, t) = xt
    A = (1 - t)*sin(pi*x)
    return A


def delA(xt):
    """Gradient of optimized version of the boundary condition function.

    Compute the gradient of the optimized version of the boundary condition
    function.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function gradient.
    """
    (x, t) = xt
    dA_dx = pi*(1 - t)*cos(pi*x)
    dA_dt = -sin(pi*x)
    delA = [dA_dx, dA_dt]
    return delA


def del2A(xt):
    """Laplacian components of optimized boundary condition function.

    Compute the Laplacian components of the optimized the boundary condition
    function.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the boundary condition function Laplacian components.
    """
    (x, t) = xt
    d2A_dx2 = -pi**2*(1 - t)*sin(pi*x)
    d2A_dt2 = 0
    del2A = [d2A_dx2, d2A_dt2]
    return del2A


def Ya(xt):
    """Analytical solution of G.

    Compute the analytical solution of G.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the analytical solution.
    """
    (x, t) = xt
    return exp(-pi**2*D*t)*sin(pi*x)


def dYa_dx(xt):
    """1st x-derivative of the analytical solution of G.

    Compute the 1st x-derivative of the analytical solution of G.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st x-derivative of the analytical solution of G.
    """
    (x, t) = xt
    return pi*exp(-pi**2*D*t)*cos(pi*x)


def dYa_dt(xt):
    """1st t-derivative of the analytical solution of G.

    Compute the 1st t-derivative of the analytical solution of G.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 1st t-derivative of the analytical solution of G.
    """
    (x, t) = xt
    return -pi**2*D*exp(-pi**2*D*t)*sin(pi*x)


# Gather the analytical gradient functions into a single array.
delYa = [dYa_dx, dYa_dt]


def d2Ya_dx2(xt):
    """2nd x-derivative of the analytical solution of G.

    Compute the 2nd x-derivative of the analytical solution of G.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd x-derivative of the analytical solution of G.
    """
    (x, t) = xt
    return -pi**2*exp(-pi**2*D*t)*sin(pi*x)


def d2Ya_dt2(xt):
    """2nd t-derivative of the analytical solution of G.

    Compute the 2nd t-derivative of the analytical solution of G.

    Parameters
    ----------
    xt : array-like of 2 float
        Values for x and t, in that order.

    Returns
    -------
    result : float
        Value of the 2nd t-derivative of the analytical solution of G.
    """
    (x, t) = xt
    return (-pi**2*D)**2*exp(-pi**2*D*t)*sin(pi*x)


# Gather the Laplacian component functions into a single array.
del2Ya = [d2Ya_dx2, d2Ya_dt2]
