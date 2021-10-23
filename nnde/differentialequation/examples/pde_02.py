"""
This module implements a simple 2-D 2nd-order PDE BVP.

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [[0, 1], [0, 1]]:

    d2Y_dx2 + d2Y_dy2 = 2*x

The analytical form of the equation is:

    G(x, y, Y, dY/dx, dY/dy) = 0 =
    d2Y_dx2 + d2Y_dy2 - 2*x

with boundary conditions:

    Y(0, y) = f0(x, y) = 0
    Y(1, y) = f1(x, y) = y**2
    Y(x, 0) = g0(x, y) = 0
    Y(x, 0) = g1(x, y) = x

This equation has the analytical solution for the supplied boundary
conditions:

Ya(x, y) = x*y**2
dYa_dx(x, y) = y**2
dYa_dy(x, y) = 2*x*y
d2Ya_dx2(x, y) = 0
d2Ya_dy2(x, y) = 2*x

The boundary condition function is:

A(x, y) = (1 - x)*f0(x, y) + x * f1(x, y)
          + (1 - y)*(g0(x, y) - ((1 - x)*g0(0, 0) + x*g0(1, 0)))
          + y*(g1(x, y) - ((1 - x)*g1(0, y) + x*g1(1, y)))
        = 0 + x * y**2
          + 0
          + y*(x - ((1 - x)*0 + x*1))
        = x + y**2

The trial solution is:

Yt(x, y) = A(x, y) + x*(1 - x)*y*(1 - y)*N(x, y, p)
         = x + y**2 + x*(1 - x)*y*(1 - y)*N(x, y, p)
"""


def G(xy, Y, delY, del2Y):
    """Compute the differential equation in standard form.

    Compute the value of the differential equation in standard form. For a
    perfect solution, the value should be 0.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.
    Y : float
        Current solution value.
    delY : array-like of 2 float
        Values for dY/dx and dY/dy, in that order.
    del2Y : array-like of 2 float
        Values for d2Y/dx2 and d2Y/dy2, in that order.

    Returns
    -------
    result : float
        Value of the differential equation.
    """
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    _G = d2Y_dx2 + d2Y_dy2 - 2*x
    return _G


def f0(xy):
    """Boundary condition at (x, y) = (0, y).

    Compute the value of the solution at (x, y) = (0, y).

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y) = (0, y).
    """
    (x, y) = xy
    return 0


def f1(xy):
    """Boundary condition at (x, y) = (1, y).

    Compute the value of the solution at (x, y) = (1, y).

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y) = (1, y).
    """
    (x, y) = xy
    return y**2


def g0(xy):
    """Boundary condition at (x, y) = (x, 0).

    Compute the value of the solution at (x, y) = (x, 0).

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y) = (x, 0).
    """
    (x, y) = xy
    return 0


def g1(xy):
    """Boundary condition at (x, y) = (x, 1).

    Compute the value of the solution at (x, y) = (x, 1).

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the solution at (x, y) = (x, 1).
    """
    (x, y) = xy
    return x


# Gather the boundary condition functions in a single array.
bc = [[f0, f1], [g0, g1]]


# def df0_dx(xt):
#     """1st derivative of BC wrt x at (x, t) = (0, t).

#     Compute the 1st derivative of BC wrt x at (x, t) = (0, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt x at (x, t) = (0, t).
#     """
#     return 0


# def df0_dt(xt):
#     """1st derivative of BC wrt t at (x, t) = (0, t).

#     Compute the 1st derivative of BC wrt t at (x, t) = (0, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt t at (x, t) = (0, t).
#     """
#     return 0


# def df1_dx(xt):
#     """1st derivative of BC wrt x at (x, t) = (1, t).

#     Compute the 1st derivative of BC wrt x at (x, t) = (1, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt x at (x, t) = (1, t).
#     """
#     return 0


# def df1_dt(xt):
#     """1st derivative of BC wrt t at (x, t) = (1, t).

#     Compute the 1st derivative of BC wrt t at (x, t) = (1, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt t at (x, t) = (1, t).
#     """
#     return 0


# def dY0_dx(xt):
#     """1st derivative of BC wrt x at (x, t) = (x, 0).

#     Compute the 1st derivative of BC wrt x at (x, t) = (x, 0).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt x at (x, t) = (x, 0).
#     """
#     return 0


# def dY0_dt(xt):
#     """1st derivative of BC wrt t at (x, t) = (x, 0).

#     Compute the 1st derivative of BC wrt t at (x, t) = (x, 0).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 1st derivative of BC wrt t at (x, t) = (x, 0).
#     """
#     return 0


# # Gather the gradient functions into a single array.
# delbc = [[[df0_dx, df0_dt], [df1_dx, df1_dt]],
#          [[dY0_dx, dY0_dt], [None, None]]]


# def d2f0_dx2(xt):
#     """2nd derivative of BC wrt x at (x, t) = (0, t).

#     Compute the 2nd derivative of BC wrt x at (x, t) = (0, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt x at (x, t) = (0, t).
#     """
#     return 0


# def d2f0_dt2(xt):
#     """2nd derivative of BC wrt t at (x, t) = (0, t).

#     Compute the 2nd derivative of BC wrt t at (x, t) = (0, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt t at (x, t) = (0, t).
#     """
#     return 0


# def d2f1_dx2(xt):
#     """2nd derivative of BC wrt x at (x, t) = (1, t).

#     Compute the 2nd derivative of BC wrt x at (x, t) = (1, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt x at (x, t) = (1, t).
#     """
#     return 0


# def d2f1_dt2(xt):
#     """2nd derivative of BC wrt t at (x, t) = (1, t).

#     Compute the 2nd derivative of BC wrt t at (x, t) = (1, t).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt t at (x, t) = (1, t).
#     """
#     return 0


# def d2Y0_dx2(xt):
#     """2nd derivative of BC wrt x at (x, t) = (x, 0).

#     Compute the 2nd derivative of BC wrt x at (x, t) = (x, 0).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt x at (x, t) = (x, 0).
#     """
#     return 0


# def d2Y0_dt2(xt):
#     """2nd derivative of BC wrt t at (x, t) = (x, 0).

#     Compute the 2nd derivative of BC wrt t at (x, t) = (x, 0).

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the 2ns derivative of BC wrt x at (x, t) = (t, 0).
#     """
#     return 0


# # Gather the functions for the Laplacian components into a single array.
# del2bc = [[[d2f0_dx2, d2f0_dt2], [d2f1_dx2, d2f1_dt2]],
#           [[d2Y0_dx2, d2Y0_dt2], [None, None]]]


# def dG_dY(xt, Y, delY, del2Y):
#     """1st derivative of G wrt Y.

#     Compute the 1st derivative of G wrt Y.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.
#     Y : float
#         Current solution value.
#     delY : array-like of 2 float
#         Values for dY/dx and dY/dt, in that order.
#     del2Y : array-like of 2 float
#         Values for d2Y/dx2 and d2Y/dt2, in that order.

#     Returns
#     -------
#     result : float
#         Value of dG/dY.
#     """
#     return 0


# # NOTE: Should be named dG_ddY_dx() but that name will not import!
# def dG_dY_dx(xt, Y, delY, del2Y):
#     """1st derivative of G wrt dY/dx.

#     Compute the 1st derivative of G wrt dY/dx.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.
#     Y : float
#         Current solution value.
#     delY : array-like of 2 float
#         Values for dY/dx and dY/dt, in that order.
#     del2Y : array-like of 2 float
#         Values for d2Y/dx2 and d2Y/dt2, in that order.

#     Returns
#     -------
#     result : float
#         Value of dG/d(dY/dx).
#     """
#     return 0


# # NOTE: Should be named dG_ddY_dt() but that name will not import!
# def dG_dY_dt(xt, Y, delY, del2Y):
#     """1st derivative of G wrt dY/dt.

#     Compute the 1st derivative of G wrt dY/dt.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.
#     Y : float
#         Current solution value.
#     delY : array-like of 2 float
#         Values for dY/dx and dY/dt, in that order.
#     del2Y : array-like of 2 float
#         Values for d2Y/dx2 and d2Y/dt2, in that order.

#     Returns
#     -------
#     result : float
#         Value of dG/d(dY/dt).
#     """
#     return 1


# # Gather the derivatives into a single array.
# dG_ddelY = [dG_dY_dx, dG_dY_dt]


# # NOTE: Should be named dG_dd2Y_dx2() but that name will not import!
# def dG_d2Y_dx2(xt, Y, delY, del2Y):
#     """1st derivative of G wrt d2Y/dx2.

#     Compute the 1st derivative of G wrt d2Y/dx2.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.
#     Y : float
#         Current solution value.
#     delY : array-like of 2 float
#         Values for dY/dx and dY/dt, in that order.
#     del2Y : array-like of 2 float
#         Values for d2Y/dx2 and d2Y/dt2, in that order.

#     Returns
#     -------
#     result : float
#         Value of dG/d(d2Y/dx2).
#     """
#     return -D


# # NOTE: Should be named dG_dd2Y_dt2() but that name will not import!
# def dG_d2Y_dt2(xt, Y, delY, del2Y):
#     """1st derivative of G wrt d2Y/dt2.

#     Compute the 1st derivative of G wrt d2Y/dt2.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.
#     Y : float
#         Current solution value.
#     delY : array-like of 2 float
#         Values for dY/dx and dY/dt, in that order.
#     del2Y : array-like of 2 float
#         Values for d2Y/dx2 and d2Y/dt2, in that order.

#     Returns
#     -------
#     result : float
#         Value of dG/d(d2Y/dt2).
#     """
#     return 0


# # Gather the derivatives into a single array.
# dG_ddel2Y = [dG_d2Y_dx2, dG_d2Y_dt2]


# def A(xt):
#     """Optimized version of the boundary condition function.

#     Compute the optimized version of the boundary condition function.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the boundary condition function.
#     """
#     return C


# def delA(xt):
#     """Gradient of optimized version of the boundary condition function.

#     Compute the gradient of the optimized version of the boundary condition
#     function.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the boundary condition function gradient.
#     """
#     return [0, 0]


# def del2A(xt):
#     """Laplacian components of optimized boundary condition function.

#     Compute the Laplacian components of the optimized the boundary condition
#     function.

#     Parameters
#     ----------
#     xt : array-like of 2 float
#         Values for x and t, in that order.

#     Returns
#     -------
#     result : float
#         Value of the boundary condition function Laplacian components.
#     """
#     return [0, 0]


def Ya(xy):
    """Analytical solution of G.

    Compute the analytical solution of G.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the analytical solution.
    """
    (x, y) = xy
    _Ya = x*y**2
    return _Ya


def dYa_dx(xy):
    """1st x-derivative of the analytical solution of G.

    Compute the 1st x-derivative of the analytical solution of G.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the 1st x-derivative of the analytical solution of G.
    """
    (x, y) = xy
    _dYa_dx = y**2
    return _dYa_dx


def dYa_dy(xy):
    """1st y-derivative of the analytical solution of G.

    Compute the 1st y-derivative of the analytical solution of G.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the 1st y-derivative of the analytical solution of G.
    """
    (x, y) = xy
    _dYa_dy = 2*x*y
    return _dYa_dy


# Gather the analytical gradient functions into a single array.
delYa = [dYa_dx, dYa_dy]


def d2Ya_dx2(xy):
    """2nd x-derivative of the analytical solution of G.

    Compute the 2nd x-derivative of the analytical solution of G.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the 2nd x-derivative of the analytical solution of G.
    """
    (x, y) = xy
    _d2Ya_dx2 = 0
    return _d2Ya_dx2


def d2Ya_dy2(xy):
    """2nd y-derivative of the analytical solution of G.

    Compute the 2nd y-derivative of the analytical solution of G.

    Parameters
    ----------
    xy : array-like of 2 float
        Values for x and y, in that order.

    Returns
    -------
    result : float
        Value of the 2nd y-derivative of the analytical solution of G.
    """
    (x, y) = xy
    _d2Ya_dy2 = 2*x
    return _d2Ya_dy2


# Gather the Laplacian component functions into a single array.
del2Ya = [d2Ya_dx2, d2Ya_dy2]
