"""
1-D diffusion PDE

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x,t).

The equation is defined on the domain (x,t) in [[0,1],[0,]].

The analytical form of the equation is:

  G(xv, Y, delY, deldelY) = dY_dt - D*d2Y_dx2 = 0

where:

xv is the vector (x,t)
delY is the vector (dY/dx, dY/dt)
deldelY is the matrix:

d2Y/dx2  d2Y/dxdt
d2Y/dtdx d2Y/dt2

With boundary conditions:

Y(0, t) = C = 0
Y(1, t) = C = 0
Y(x, 0) = C = 0

This equation has the analytical solution for the supplied initial conditions:

Ya(x, t) = 0
"""


import numpy as np


# Diffusion coefficient
D = 0.1

# Constant value of profile
C = 0


def Gf(xv, Y, delY, deldelY):
    """The differential equation in standard form"""
    (x, t) = xv
    (dY_dx, dY_dt) = delY
    ((d2Y_dx2, d2Y_dxdt), (d2Y_dtdx, d2Y_dt2)) = deldelY
    return dY_dt - D*d2Y_dx2

def dG_dYf(xv, Y, delY, deldelY):
    """Partial of PDE wrt Y"""
    (x, t) = xv
    (dY_dx, dY_dt) = delY
    ((d2Y_dx2, d2Y_dxdt), (d2Y_dtdx, d2Y_dt2)) = deldelY
    return 0

def dG_dY_dxf(xv, Y, delY, deldelY):
    """Partial of PDE wrt dY/dx"""
    (x, t) = xv
    (dY_dx, dY_dt) = delY
    ((d2Y_dx2, d2Y_dxdt), (d2Y_dtdx, d2Y_dt2)) = deldelY
    return 0

# def dG_dY_dtf(xt, Y, delY, del2Y):
#     """Partial of PDE wrt dY/dt"""
#     (x, t) = xt
#     (dY_dx, dY_dt) = delY
#     (d2Y_dx2, d2Y_dt2) = del2Y
#     return 1

# dG_ddelYf = [dG_dY_dxf, dG_dY_dtf]


# def dG_d2Y_dx2f(xt, Y, delY, del2Y):
#     """Partial of PDE wrt d2Y/dx2"""
#     (x, t) = xt
#     (dY_dx, dY_dt) = delY
#     (d2Y_dx2, d2Y_dt2) = del2Y
#     return -D

# def dG_d2Y_dt2f(xt, Y, delY, del2Y):
#     """Partial of PDE wrt d2Y/dt2"""
#     (x, t) = xt
#     (dY_dx, dY_dt) = delY
#     (d2Y_dx2, d2Y_dt2) = del2Y
#     return 0

# dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dt2f]


# def f0f(xt):
#     """Boundary condition at (x,t) = (0,t)"""
#     (x, t) = xt
#     return C

# def f1f(xt):
#     """Boundary condition at (x,t) = (1,t)"""
#     (x, t) = xt
#     return C

# def Y0f(xt):
#     """Boundary condition at (x,t) = (x,0)"""
#     (x, t) = xt
#     return C

# def Y1f(xt):
#     """Boundary condition at (x,t) = (x,1) NOT USED"""
#     (x, t) = xt
#     return None

# bcf = [[f0f, f1f], [Y0f, Y1f]]


# def df0_dxf(xt):
#     """1st derivative of BC wrt x at (x,t) = (0,t)"""
#     (x, t) = xt
#     return 0

# def df0_dtf(xt):
#     """1st derivative of BC wrt t at (x,t) = (0,t)"""
#     (x, t) = xt
#     return 0

# def df1_dxf(xt):
#     """1st derivative of BC wrt x at (x,t) = (1,t)"""
#     (x, t) = xt
#     return 0

# def df1_dtf(xt):
#     """1st derivative of BC wrt t at (x,t) = (1,t)"""
#     (x, t) = xt
#     return 0

# def dY0_dxf(xt):
#     """1st derivative of BC wrt x at (x,t) = (x,0)"""
#     (x, t) = xt
#     return 0

# def dY0_dtf(xt):
#     """1st derivative of BC wrt t at (x,t) = (x,0)"""
#     (x, t) = xt
#     return 0

# def dY1_dxf(xt):
#     """1st derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
#     (x, t) = xt
#     return None

# def dY1_dtf(xt):
#     """1st derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
#     (x, t) = xt
#     return None

# delbcf = [[[df0_dxf, df0_dtf], [df1_dxf, df1_dtf]],
#           [[dY0_dxf, dY0_dtf], [dY1_dxf, dY1_dtf]]]


# def d2f0_dx2f(xt):
#     """2nd derivative of BC wrt x at (x,t) = (0,t)"""
#     (x, t) = xt
#     return 0

# def d2f0_dt2f(xt):
#     """2nd derivative of BC wrt t at (x,t) = (0,t)"""
#     (x, t) = xt
#     return 0

# def d2f1_dx2f(xt):
#     """2nd derivative of BC wrt x at (x,t) = (1,t)"""
#     (x, t) = xt
#     return 0

# def d2f1_dt2f(xt):
#     """2nd derivative of BC wrt t at (x,t) = (1,t)"""
#     (x, t) = xt
#     return 0

# def d2Y0_dx2f(xt):
#     """2nd derivative of BC wrt x at (x,t) = (x,0)"""
#     (x, t) = xt
#     return 0

# def d2Y0_dt2f(xt):
#     """2nd derivative of BC wrt t at (x,t) = (x,0)"""
#     (x, t) = xt
#     return 0

# def d2Y1_dx2f(xt):
#     """2nd derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
#     (x, t) = xt
#     return None

# def d2Y1_dt2f(xt):
#     """2nd derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
#     (x, t) = xt
#     return None

# del2bcf = [[[d2f0_dx2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dt2f]],
#            [[d2Y0_dx2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dt2f]]]


# def Af(xt):
#     """Optimized version of boundary condition function"""
#     return C

# def delAf(xt):
#     """Optimized version of boundary condition function gradient"""
#     return [0, 0]

# def del2Af(xt):
#     """Optimized version of boundary condition function Laplacian"""
#     return [0, 0]


# def Yaf(xt):
#     """Analytical solution"""
#     (x, t) = xt
#     Ya = C
#     return Ya

# def dYa_dxf(xt):
#     """Analytical x-gradient"""
#     (x, t) = xt
#     dYa_dx = 0
#     return dYa_dx

# def dYa_dtf(xt):
#     """Analytical t-gradient"""
#     (x, t) = xt
#     dYa_dt = 0
#     return dYa_dt

# delYaf = [dYa_dxf, dYa_dtf]


# def d2Ya_dx2f(xt):
#     """Analytical x-Laplacian"""
#     (x, t) = xt
#     d2Ya_dx2 = 0
#     return d2Ya_dx2

# def d2Ya_dt2f(xt):
#     """Analytical t-Laplacian"""
#     (x, t) = xt
#     d2Ya_dt2 = 0
#     return d2Ya_dt2

# del2Yaf = [d2Ya_dx2f, d2Ya_dt2f]




if __name__ == '__main__':
    pass

    # Test values
    xv_test = [0, 0]
    Y_test = 0
    delY_test = [0, 0]
    deldelY_test = [[0, 0], [0, 0]]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    # dG_ddelY_ref = (0, 1)
    # dG_ddel2Y_ref = (-D, 0)
    # bc_ref = [[C, C],
    #           [C, None]]
    # delbc_ref = [[[0, 0], [0, 0]],
    #              [[0, 0], [None, None]]]
    # del2bc_ref = [[[0, 0], [0, 0]],
    #               [[0, 0], [None, None]]]
    # A_ref = C
    # delA_ref = [0, 0]
    # del2A_ref = [0, 0]
    # Ya_ref = C
    # delYa_ref = [0, 0]
    # del2Ya_ref = [0, 0]

    print("Testing differential equation.")
    assert np.isclose(Gf(xv_test, Y_test, delY_test, deldelY_test), G_ref)

    print("Testing differential equation Y-derivative.")
    assert np.isclose(dG_dYf(xv_test, Y_test, delY_test, deldelY_test), dG_dY_ref)

    # print("Testing differential equation gradient-derivatives.")
    # for (i, f) in enumerate(dG_ddelYf):
    #     assert np.isclose(f(xt, Ya_ref, delYa_ref, del2Ya_ref),
    #                       dG_ddelY_ref[i])

    # print("Testing differential equation Laplacian-derivatives.")
    # for (i, f) in enumerate(dG_ddel2Yf):
    #     assert np.isclose(f(xt, Ya_ref, delYa_ref, del2Ya_ref),
    #                       dG_ddel2Y_ref[i])

    # print("Testing boundary conditions.")
    # for i in range(len(bcf)):
    #     for (j, f) in enumerate(bcf[i]):
    #         if bc_ref[i][j] is None:
    #             assert f(xt) is None
    #         else:
    #             assert np.isclose(f(xt), bc_ref[i][j])

    # print("Testing boundary condition gradients.")
    # for i in range(len(delbcf)):
    #     for j in range(len(delbcf[i])):
    #         for (k, f) in enumerate(delbcf[i][j]):
    #             if delbc_ref[i][j][k] is None:
    #                 assert f(xt) is None
    #             else:
    #                 assert np.isclose(f(xt), delbc_ref[i][j][k])

    # print("Testing boundary condition Laplacians.")
    # for i in range(len(del2bcf)):
    #     for j in range(len(del2bcf[i])):
    #         for (k, f) in enumerate(del2bcf[i][j]):
    #             if del2bc_ref[i][j][k] is None:
    #                 assert f(xt) is None
    #             else:
    #                 assert np.isclose(f(xt), del2bc_ref[i][j][k])

    # print("Verifying BC continuity constraints.")
    # assert np.isclose(f0f([0, 0]), Y0f([0, 0]))
    # assert np.isclose(f1f([1, 0]), Y0f([1, 0]))
    # # t=1 not used

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

    # print("Testing analytical solution.")
    # assert np.isclose(Yaf(xt), Ya_ref)

    # print("Testing analytical solution gradient.")
    # for (i, f) in enumerate(delYaf):
    #     if delYa_ref[i] is None:
    #         assert f(xt) is None
    #     else:
    #         assert np.isclose(f(xt), delYa_ref[i])

    # print("Testing analytical solution Laplacian.")
    # for (i, f) in enumerate(del2Yaf):
    #     if del2Ya_ref[i] is None:
    #         assert f(xt) is None
    #     else:
    #         assert np.isclose(f(xt), del2Ya_ref[i])
