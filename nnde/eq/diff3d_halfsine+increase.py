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


if __name__ == '__main__':

    # Test values
    xyzt_test = (0.4, 0.5, 0.6, 0.7)
    m = len(xyzt_test)
    Y_test = 0.1138378166982095
    delY_test = (0.11620169660719469, 0,
                 -0.11620169660719464, -0.3370602650085157)
    del2Y_test = (-1.1235342166950524, -1.1235342166950524,
                  -1.1235342166950524, 0.9979954424881178)

    # Reference values for tests.
    G_ref = 0
    bc_ref = ((0, 0.6657395614066075),
              (0, 0),
              (0, 0),
              (0.904508497187474, None))
    delbc_ref = (((0, 0, 0, 0),
                  (0, 0, -0.6795638635539131, 0.9510565162951536)),
                 ((0, 0, 0, 0),
                  (0, 0, 0, 0)),
                 ((0, 0, 0, 0),
                  (0, 0, 0, 0)),
                 ((0.9232909152452285, 0, -0.923290915245228, 0),
                  (None, None, None, None)))
    del2bc_ref = [[[0, 0, 0, 0],
                   [0, -6.570586105237952, -6.570586105237952, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0]],
                  [[-8.927141044664213, -8.927141044664213,
                    -8.927141044664213, 0],
                   [None, None, None, None]]]
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 0, 0, 1)
    dG_ddel2Y_ref = (-D, -D, -D, 0)
    A_ref = 0.5376483737188851
    delA_ref = [0.9427268359801761, 0, -0.5488128199951336,
                -0.5240858906694122]
    del2A_ref = [-2.6781423133992637, -5.306376755494444,
                 -5.306376755494444, 0]

    print("Testing differential equation.")
    assert np.isclose(G(xyzt_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](xyzt_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0, 0, 0]), g0([0, 0, 0, 0]))
    assert np.isclose(f0([0, 0, 0, 0]), h0([0, 0, 0, 0]))
    assert np.isclose(f0([0, 0, 0, 0]), Y0([0, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), g0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), h0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), Y0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), g1([1, 1, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), h0([1, 1, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), Y0([1, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), g1([0, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), h0([0, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), Y0([0, 1, 0, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), g0([0, 0, 1, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), h1([0, 0, 1, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), Y0([0, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), g0([1, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), h1([1, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), Y0([1, 0, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), g1([1, 1, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), h1([1, 1, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), Y0([1, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), g1([0, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), h1([0, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), Y0([0, 1, 1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](xyzt_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](xyzt_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(xyzt_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](xyzt_test, Y_test, delY_test,
                                      del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](xyzt_test, Y_test, delY_test,
                                       del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing optimized BC function.")
    A_ = A(xyzt_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(xyzt_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(xyzt_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])
