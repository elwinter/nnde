"""
This module implements a 1-D diffusion PDE

Note that an upper-case 'Y' is used to represent the Greek psi, which
represents the problem solution Y(x,t).

The equation is defined on the domain (x, t) in [[0, 1],[0, ]].

The analytical form of the equation is:

  G([x, t], Y, delY, del2Y) = dY_dt - D*d2Y_dx2 = 0

where:

[x, t] are the independent variables.
delY is the vector (dY/dx, dY/dt)
del2Y is the vector (d2Y/dx2, d2Y/dt2)

With boundary conditions:

Y(0, t) = 0
Y(1, t) = a*t
Y(x, 0) = sin(pi*x)

This equation has the analytical solution for the supplied initial
conditions:

Ya(x, t) = a*t*x + sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
           + 0.5*SUM(k=1, inf, 2*(-1)**k*a*(1 - exp(-pi**2*t*D*k**2))
                               *sin(pi*x*k)/(pi**3*k**3)

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import exp, cos, cosh, pi, sin, sinh
import numpy as np


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 0.1

# Number of terms in analytical summation
kmax = 800


def G(xt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return dY_dt - D*d2Y_dx2


def f0(xt):
    """Boundary condition at (x,t) = (0,t)"""
    return 0


def f1(xt):
    """Boundary condition at (x,t) = (1,t)"""
    (x, t) = xt
    return a*t


def Y0(xt):
    """Boundary condition at (x,t) = (x,0)"""
    (x, t) = xt
    return sin(pi*x)


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
    return a


def dY0_dx(xt):
    """1st derivative of BC wrt x at (x,t) = (x,0)"""
    (x, t) = xt
    return pi*cos(pi*x)


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
    (x, t) = xt
    return -pi**2*sin(pi*x)


def d2Y0_dt2(xt):
    """2nd derivative of BC wrt t at (x,t) = (x,0)"""
    return 0


del2bc = [[[d2f0_dx2, d2f0_dt2], [d2f1_dx2, d2f1_dt2]],
          [[d2Y0_dx2, d2Y0_dt2], [None, None]]]


def dG_dY(xt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    return 0


def dG_ddY_dx(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    return 0


def dG_ddY_dt(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    return 1


dG_ddelY = [dG_ddY_dx, dG_ddY_dt]


def dG_dd2Y_dx2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    return -D


def dG_dd2Y_dt2(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    return 0


dG_ddel2Y = [dG_dd2Y_dx2, dG_dd2Y_dt2]


def A(xt):
    """Optimized version of boundary condition function"""
    (x, t) = xt
    A = a*t*x + (1 - t)*sin(pi*x)
    return A


def delA(xt):
    """Optimized version of boundary condition function gradient"""
    (x, t) = xt
    dA_dx = pi*(1 - t)*cos(pi*x)
    dA_dx = a*t + pi*(1 - t)*cos(pi*x)
    dA_dt = a*x - sin(pi*x)
    delA = [dA_dx, dA_dt]
    return delA


def del2A(xt):
    """Optimized version of boundary condition function Laplacian"""
    (x, t) = xt
    d2A_dx2 = -pi**2*(1 - t)*sin(pi*x)
    d2A_dt2 = 0
    del2A = [d2A_dx2, d2A_dt2]
    return del2A


def Ya(xt):
    """Analytical solution"""
    (x, t) = xt
    Ya = a*t*x + sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        Ya += (
            2*(-1)**k*a*(1 - exp(-pi**2*t*D*k**2))*sin(pi*x*k)
            / (pi**3*k**3)
        )
    return Ya


def dYa_dx(xt):
    """Analytical x-gradient"""
    (x, t) = xt
    dYa_dx = a*t + pi*cos(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dx += (
            2*(-1)**k*a*(1 - exp(-k**2*pi**2*t*D))*cos(k*pi*x)
            / (k**2*pi**2)
        )
    return dYa_dx


def dYa_dt(xt):
    """Analytical t-gradient"""
    (x, t) = xt
    dYa_dt = a*x + sin(pi*x)*pi**2*D*(sinh(pi**2*t*D) - cosh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dt += (
            2*(-1)**k*a*exp(-k**2*pi**2*t*D)*D*sin(k*pi*x)
            / (k*pi)
        )
    return dYa_dt


delYa = [dYa_dx, dYa_dt]


def d2Ya_dx2(xt):
    """Analytical x-Laplacian"""
    (x, t) = xt
    d2Ya_dx2 = -pi**2*sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dx2 += (
            -2*(-1)**k*a*(1 - exp(-pi**2*t*D*k**2))*sin(pi*x*k)
            / (pi*k)
        )
    return d2Ya_dx2


def d2Ya_dt2(xt):
    """Analytical t-Laplacian"""
    (x, t) = xt
    d2Ya_dt2 = sin(pi*x)*pi**4*D**2*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dt2 += -2*(-1)**k*a*exp(-pi**2*t*D*k**2)*k*pi*D**2*sin(pi*x*k)
    return d2Ya_dt2


del2Ya = [d2Ya_dx2, d2Ya_dt2]


if __name__ == '__main__':

    # Test values
    x_test = (0.4, 0.5)
    m = len(x_test)
    Y_test = 0.5986958383019955
    delY_test = (0.6383788665776866, -0.536469420056056)
    del2Y_test = (-5.724723108734729, 0.5680753052926439)

    # Reference values for tests.
    G_ref = 0.03600289081741688   # Non-zero due to roundoff error?
    bc_ref = ((0, 0.05),
              (0.951057, None))
    delbc_ref = [[[0, 0], [0, a]],
                 [[0.970805519362733, 0], [None, None]]]
    del2bc_ref = [[[0, 0], [0, 0]],
                  [[-9.38655157891136, 0], [None, None]]]
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 1)
    dG_ddel2Y_ref = (-D, 0)
    A_ref = 0.4955282581475768
    delA_ref = [0.5354027596813666, -0.9110565162951535]
    del2A_ref = [-4.69327578945568, 0]
    Ya_ref = 0.5986958383019955
    delYa_ref = [0.6383788665776866, -0.536469420056056]
    del2Ya_ref = [-5.724723108734729, 0.5680753052926439]

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
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](x_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
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

    print("Testing optimized BC function.")
    A_ = A(x_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(x_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(x_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])

    print("Testing analytical solution.")
    assert np.isclose(Ya(x_test), Ya_ref)

    print("Testing analytical solution gradient.")
    for j in range(m):
        assert np.isclose(delYa[j](x_test), delYa_ref[j])

    print("Testing analytical solution Laplacian.")
    for j in range(m):
        assert np.isclose(del2Ya[j](x_test), del2Ya_ref[j])
