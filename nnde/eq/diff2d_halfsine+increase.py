"""
This module implements a 2-D diffusion PDE

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x, y, t).

The equation is defined on the domain (x, y, t) in [[0, 1], [0, 1], [0, inf]].

The analytical form of the equation is:

  G([x, y, t], Y, delY, del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2) = 0

where:

[x, y, t] are the independent variables
delY is the vector (dY/dx, dY/dy, dY/dt)
del2Y is the vector (d2Y/dx2, d2Y/dy2, d2Y/dt2)

With boundary conditions (note the BC are continuous at domain corners):

Y(0, y, t) = 0
Y(1, y, t) = a*t*sin(pi*y)
Y(x, 0, t) = 0
Y(x, 1, t) = 0
Y(x, y, 0) = sin(pi*x)*sin(pi*y)

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


def G(xyt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2)


def f0(xyt):
    """Boundary condition at (x, y, t) = (0, y, t)"""
    return 0


def f1(xyt):
    """Boundary condition at (x, y, t) = (1, y, t)"""
    (x, y, t) = xyt
    return a*t*sin(pi*y)


def g0(xyt):
    """Boundary condition at (x, y, t) = (x, 0, t)"""
    return 0


def g1(xyt):
    """Boundary condition at (x, y, t) = (x, 1, t)"""
    return 0


def Y0(xyt):
    """Boundary condition at (x, y, t) = (x, y, 0)"""
    (x, y, t) = xyt
    return sin(pi*x)*sin(pi*y)


bc = [[f0, f1], [g0, g1], [Y0, None]]


def df0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (0, y, t)"""
    return 0


def df0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (0, y, t)"""
    return 0


def df0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (0, y, t)"""
    return 0


def df1_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (1, y, t)"""
    return 0


def df1_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (1, y, t)"""
    (x, y, t) = xyt
    return a*pi*t*cos(pi*y)


def df1_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (1, y, t)"""
    (x, y, t) = xyt
    return a*sin(pi*y)


def dg0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, 0, t)"""
    return 0


def dg0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, 0, t)"""
    return 0


def dg0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, 0, t)"""
    return 0


def dg1_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, 1, t)"""
    return 0


def dg1_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, 1, t)"""
    return 0


def dg1_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, 1, t)"""
    return 0


def dY0_dx(xyt):
    """1st derivative of BC wrt x at (x, y, t) = (x, y, 0)"""
    (x, y, t) = xyt
    return pi*cos(pi*x)*sin(pi*y)


def dY0_dy(xyt):
    """1st derivative of BC wrt y at (x, y, t) = (x, y, 0)"""
    (x, y, t) = xyt
    return pi*sin(pi*x)*cos(pi*y)


def dY0_dt(xyt):
    """1st derivative of BC wrt t at (x, y, t) = (x, y, 0)"""
    return 0


delbc = [[[df0_dx, df0_dy, df0_dt], [df1_dx, df1_dy, df1_dt]],
         [[dg0_dx, dg0_dy, dg0_dt], [dg1_dx, dg1_dy, dg1_dt]],
         [[dY0_dx, dY0_dy, dY0_dt], [None, None, None]]]


def d2f0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (0, y, t)"""
    return 0


def d2f0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (0, y, t)"""
    return 0


def d2f0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (0, y, t)"""
    return 0


def d2f1_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (1, y, t)"""
    return 0


def d2f1_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (1, y, t)"""
    (x, y, t) = xyt
    return -a*pi**2*t*sin(pi*y)


def d2f1_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (1, y, t)"""
    return 0


def d2g0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, 0, t)"""
    return 0


def d2g0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, 0, t)"""
    return 0


def d2g0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, 0, t)"""
    return 0


def d2g1_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, 1, t)"""
    return 0


def d2g1_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, 1, t)"""
    return 0


def d2g1_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, 1, t)"""
    return 0


def d2Y0_dx2(xyt):
    """2nd derivative of BC wrt x at (x, y, t) = (x, y, 0)"""
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)


def d2Y0_dy2(xyt):
    """2nd derivative of BC wrt y at (x, y, t) = (x, y, 0)"""
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)


def d2Y0_dt2(xyt):
    """2nd derivative of BC wrt t at (x, y, t) = (x, y, 0)"""
    return 0


del2bc = [[[d2f0_dx2, d2f0_dy2, d2f0_dt2], [d2f1_dx2, d2f1_dy2, d2f1_dt2]],
          [[d2g0_dx2, d2g0_dy2, d2g0_dt2], [d2g1_dx2, d2g1_dy2, d2g1_dt2]],
          [[d2Y0_dx2, d2Y0_dy2, d2Y0_dt2], [None, None, None]]]


def dG_dY(xyt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    return 0


def dG_ddY_dx(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    return 0


def dG_ddY_dy(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dy"""
    return 0


def dG_ddY_dt(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    return 1


dG_ddelY = [dG_ddY_dx, dG_ddY_dy, dG_ddY_dt]


def dG_dd2Y_dx2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    return -D


def dG_dd2Y_dy2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dy2"""
    return -D


def dG_dd2Y_dt2(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    return 0


dG_ddel2Y = [dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dt2]


def A(xyt):
    """Optimized version of boundary condition function"""
    (x, y, t) = xyt
    A = (a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)
    return A


def delA(xyt):
    """Optimized version of boundary condition function gradient"""
    (x, y, t) = xyt
    dA_dx = (a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)
    dA_dy = pi*cos(pi*y)*(a*t*x + (1 - t)*sin(pi*x))
    dA_dt = (a*x - sin(pi*x))*sin(pi*y)
    return [dA_dx, dA_dy, dA_dt]


def del2A(xyt):
    """Optimized version of boundary condition function Laplacian"""
    (x, y, t) = xyt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)*sin(pi*y)
    d2A_dy2 = pi**2*(-a*t*x + (t - 1)*sin(pi*x))*sin(pi*y)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dt2]


if __name__ == '__main__':

    # Test values
    xyt_test = (0.4, 0.5, 0.6)
    m = len(xyt_test)
    Y_test = 0.11
    delY_test = (0.22, 0.33, 0.44)
    del2Y_test = (-0.55, -0.66, -0.77)

    # # Reference values for tests.
    G_ref = 0.561
    bc_ref = ((0, 0.6),
              (0, 0),
              (0.951056516295154, None))
    delbc_ref = (((0, 0, 0), (0, 0, a)),
                 ((0, 0, 0), (0, 0, 0)),
                 ((0.970805519362733, 0, 0), (None, None, None)))
    del2bc_ref = (((0, 0, 0), (0, -5.921762640653615, 0)),
                  ((0, 0, 0), (0, 0, 0)),
                  ((-9.38655157891136, -9.38655157891136, 0),
                   (None, None, None)))
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 0, 1)
    dG_ddel2Y_ref = (-D, -D, 0)
    A_ref = 0.6204226065180613
    delA_ref = [0.9883222077450933, 0, -0.5510565162951535]
    del2A_ref = [-3.7546206315645443, -6.12332568782599, 0]

    print("Testing differential equation.")
    assert np.isclose(G(xyt_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](xyt_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0, 0]), g0([0, 0, 0]))
    assert np.isclose(f0([0, 0, 0]), Y0([0, 0, 0]))
    assert np.isclose(f1([1, 0, 0]), g0([1, 0, 0]))
    assert np.isclose(f1([1, 0, 0]), Y0([1, 0, 0]))
    assert np.isclose(f1([1, 1, 0]), g1([1, 1, 0]))
    assert np.isclose(f1([1, 1, 0]), Y0([1, 1, 0]))
    assert np.isclose(f0([0, 1, 0]), g1([0, 1, 0]))
    assert np.isclose(f0([0, 1, 0]), Y0([0, 1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](xyt_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](xyt_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(xyt_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](xyt_test, Y_test, delY_test, del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](xyt_test, Y_test, delY_test,
                                       del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing optimized BC function.")
    A_ = A(xyt_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(xyt_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(xyt_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])
