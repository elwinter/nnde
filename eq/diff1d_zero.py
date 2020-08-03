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


# def Af(xt):
#     """Optimized version of boundary condition function"""
#     return C

# def delAf(xt):
#     """Optimized version of boundary condition function gradient"""
#     return [0, 0]

# def del2Af(xt):
#     """Optimized version of boundary condition function Laplacian"""
#     return [0, 0]


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


if __name__ == '__main__':

    # Test values
    x_test = (0, 0)
    m = len(x_test)
    Y_test = 0
    delY_test = (0, 0)
    del2Y_test = (0, 0)

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    bc_ref = ((C, C),
              (C, None))
    delbc_ref = (((0, 0), (0, 0)),
                 ((0, 0), (None, None)))
    del2bc_ref = (((0, 0), (0, 0)),
                  ((0, 0), (None, None)))
    dG_ddelY_ref = (0, 1)
    dG_ddel2Y_ref = (-D, 0)
    # A_ref = C
    # delA_ref = [0, 0]
    # del2A_ref = [0, 0]
    Ya_ref = C
    delYa_ref = (0, 0)
    del2Ya_ref = (0, 0)

    print("Testing differential equation.")
    assert np.isclose(G(x_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](x_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0]), Y0([0, 0]))
    assert np.isclose(f1([1, 0]), Y0([1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(m):
            for jjj in range(2):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](x_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(m):
            for jjj in range(2):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](x_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(x_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](x_test, Y_test, delY_test, del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](x_test, Y_test, delY_test, del2Y_test),
                          dG_ddel2Y_ref[j])

    # print("Testing optimized BC function.")
    # assert np.isclose(Af(xt), A_ref)

    # print("Testing optimized BC function gradient.")
    # delA = delAf(xt)
    # for i in range(len(delA_ref)):
    #     assert np.isclose(delA[i], delA_ref[i])

    # print("Testing optimized BC function Laplacian.")
    # del2A = del2Af(xt)
    # for i in range(len(del2A_ref)):
    #     assert np.isclose(del2A[i], del2A_ref[i])

    print("Testing analytical solution.")
    assert np.isclose(Ya(x_test), Ya_ref)

    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](x_test, Y_test, delY_test, del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing analytical solution gradient.")
    for j in range(m):
        assert np.isclose(delYa[j](x_test), delYa_ref[j])

    print("Testing analytical solution Laplacian.")
    for j in range(m):
        assert np.isclose(del2Ya[j](x_test), del2Ya_ref[j])
