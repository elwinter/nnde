"""The 1-D diffusion equation.

This module implements a 1-D diffusion PDE.

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x, t). Y(x, t) is normalized to the
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

Y(0, t) = C = 1
Y(1, t) = C = 1
Y(x, 0) = C = 1

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, t) = 1

Note
----
This simple equation has a constant as a solution, and was used to ensure
solution stability.

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


# Diffusion coefficient (L**2/T, where L is a length, and T is a time).
D = 0.1

# Initial value of profile (unitless).
C = 1


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
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
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
    return C


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
    return C


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
    return C


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
    return 0


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
    return 0


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


# NOTE: Should be named dG_ddY_dx() but that name will not import!
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


# NOTE: Should be named dG_ddY_dt() but that name will not import!
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


# NOTE: Should be named dG_dd2Y_dx2() but that name will not import!
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


# NOTE: Should be named dG_dd2Y_dt2() but that name will not import!
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
    return C


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
    return [0, 0]


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
    return [0, 0]


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
    return C


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
    return 0


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
    return 0


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
    return 0


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
    return 0


# Gather the Laplacian component functions into a single array.
del2Ya = [d2Ya_dx2, d2Ya_dt2]
